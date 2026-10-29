import argparse
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from flint import ctx
from quotaguard.train.dp_certify import ball, case_a, certify_order
PRECISIONS = (192, 256)
NEIGHBOURS = 2
SHARED = ('sigma', 'gamma', 'delta', 'kdeg', 'steps', 'epsilon_budget')

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def collect(run_dir):
    groups = {}
    for path in sorted(Path(run_dir).glob('*/task.json')):
        task = json.loads(path.read_text())
        key = f'{Path(run_dir).parent.name}/{task["source"]}-{task["target"]}'
        signature = dict({k: task[k] for k in SHARED}, calibration=task['calibration'], zero_insertion=task.get('zero_insertion', False))
        group = groups.setdefault(key, dict(id=key, signature=signature, sources=[]))
        if group['signature'] != signature:
            raise ValueError(f'Runs of {key} do not share one calibration: {path}')
        group['sources'].append(dict(path=str(path), sha256=sha256(path), arm=task['arm'], seed=task['seed']))
    if not groups:
        raise ValueError(f'No task.json under {run_dir}')
    return list(groups.values())

def plan(group):
    s = group['signature']
    c = s['calibration']
    for key in ('gamma', 'delta', 'kdeg'):
        if c[key] != s[key]:
            raise ValueError(f'{group["id"]}: calibration {key} differs from the training task')
    if c['nominal_epsilon'] != s['epsilon_budget']:
        raise ValueError(f'{group["id"]}: nominal epsilon differs from the budget')
    if c['accountant'] == 'mog_case_A_v1':
        if not s['zero_insertion'] or c['sigma'] != s['sigma'] or c['steps'] != s['steps']:
            raise ValueError(f'{group["id"]}: plain accountant used without zero insertion or with other noise')
        sigma, steps, rule = (s['sigma'], s['steps'], 'zero insertion: Case A at the training noise over T')
    elif c['accountant'] == 'stable_mog_case_A_v1':
        if s['zero_insertion'] or c['accounting_steps'] != 2 * s['steps'] or c['sigma'] != s['sigma']:
            raise ValueError(f'{group["id"]}: inconsistent pool-truncation calibration')
        sigma, steps = (c['accounting_sigma'], c['accounting_steps'])
        rule = 'pool truncation: Case A at sigma(2T) over 2T; training noise >= sqrt(2) sigma(2T)'
    else:
        raise ValueError(f'{group["id"]}: unsupported accountant {c["accountant"]}')
    grid = list(c['orders'])
    i = grid.index(c['best_order'])
    orders = [str(Fraction(str(a))) for a in grid[max(0, i - NEIGHBOURS):i + NEIGHBOURS + 1]]
    return dict(sigma=sigma, steps=steps, gamma=s['gamma'], delta=s['delta'], kdeg=s['kdeg'], nominal=s['epsilon_budget'], orders=orders, rule=rule, accountant=c['accountant'], training_sigma=s['sigma'])

def certify(group):
    p = plan(group)
    out = dict(group, parameters=p, precisions={})
    for bits in PRECISIONS:
        ctx.prec, ctx.threads = (bits, 1)
        branches, tail = case_a(p['gamma'], p['kdeg'])
        certificates = [certify_order(branches, ball(p['sigma']) * p['kdeg'], a, p['steps'], p['delta'], beta=tail.get('beta')) for a in p['orders']]
        best = min(certificates, key=lambda c: Fraction(c['epsilon']['upper']))
        out['precisions'][str(bits)] = dict(selected_order=best['alpha'], epsilon=best['epsilon'], certificates=certificates, passed=Fraction(best['epsilon']['upper']) <= Fraction(p['nominal']))
    noise_ok = True
    if p['accountant'] == 'stable_mog_case_A_v1':
        noise_ok = Fraction.from_float(p['training_sigma']) ** 2 >= 2 * Fraction.from_float(p['sigma']) ** 2
    lo = [Fraction(v['epsilon']['lower']) for v in out['precisions'].values()]
    hi = [Fraction(v['epsilon']['upper']) for v in out['precisions'].values()]
    out.update(noise_check=noise_ok, precisions_intersect=max(lo) <= min(hi), status='supported' if noise_ok and max(lo) <= min(hi) and all((v['passed'] for v in out['precisions'].values())) else 'FAILED')
    return out

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit('Output directory must be empty')
    groups = [g for run_dir in args.runs for g in collect(run_dir)]
    for g in groups:
        plan(g)
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(certify, groups))
    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    for r in results:
        name = r['id'].replace('/', '__')
        (args.output / f'{name}.json').write_text(json.dumps(r, indent=2, sort_keys=True) + '\n')
        best = r['precisions'][str(PRECISIONS[-1])]
        summary.append(dict(id=r['id'], status=r['status'], nominal=r['parameters']['nominal'], epsilon_upper=best['epsilon']['upper'], order=best['selected_order'], sigma_certified=r['parameters']['sigma'], steps=r['parameters']['steps'], training_sigma=r['parameters']['training_sigma'], rule=r['parameters']['rule'], runs=len(r['sources'])))
    (args.output / 'summary.json').write_text(json.dumps(dict(rows=summary, script_sha256=sha256(__file__), precisions=PRECISIONS), indent=2) + '\n')
    lines = ['| group | runs | nominal ε | certified ε upper | order | status |', '|---|---|---|---|---|---|']
    lines += [f'| {s["id"]} | {s["runs"]} | {s["nominal"]} | {s["epsilon_upper"][:20]} | {s["order"]} | {s["status"]} |' for s in summary]
    (args.output / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    if any((s['status'] != 'supported' for s in summary)):
        raise SystemExit(1)
if __name__ == '__main__':
    main()
