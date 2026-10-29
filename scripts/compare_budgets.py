import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import yaml
REPO = Path(__file__).resolve().parents[1]
QUESTIONS = {'materials': ('mat', 'confirm'), 'budget4': ('eps4', 'confirm'), 'budget1': ('eps1', 'confirm')}
CONFIGS = {'mat_cg': 'configs/train/mat_candidate_graph_eps10.yaml', 'mat_lg': 'configs/train/mat_public_groups_eps10.yaml', 'eps4_cg': 'configs/train/cs_candidate_graph_eps4.yaml', 'eps4_lg': 'configs/train/cs_public_groups_eps4.yaml', 'eps1_cg': 'configs/train/cs_candidate_graph_eps1.yaml', 'eps1_lg': 'configs/train/cs_public_groups_eps1.yaml'}
CONFIGS = {k: REPO / v for k, v in CONFIGS.items()}
REPORT_FILE = {'confirm': 'final_report.json', 'diagnose': 'diagnose_report.json'}
T975 = {4: 2.7764451051977934}
MRR_THRESHOLD = 0.01
METRICS = {'batch_mrr': lambda t: t['mrr'], 'full_mrr': lambda t: t['full_corpus']['mrr'], 'recall_at_10': lambda t: t['full_corpus']['recall_at_10'], 'recall_at_100': lambda t: t['full_corpus']['recall_at_100']}

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()

def pairing_signature(run):
    test = run.get('target_test', {})
    return (test.get('n_relations'), test.get('batch_size'), test.get('tail_batch_size'), test.get('full_corpus', {}).get('n'), run.get('initial_parameter_sha256'))

def interval(diffs):
    n = len(diffs)
    if n - 1 not in T975:
        raise ValueError('Configured analysis expects exactly five paired seeds')
    mean = statistics.mean(diffs)
    half = T975[n - 1] * statistics.stdev(diffs) / math.sqrt(n)
    return dict(n=n, df=n - 1, mean=mean, lower=mean - half, upper=mean + half, differences=diffs)

def load(name, mode, problems, configs):
    cfg = yaml.safe_load(Path(configs[name]).read_text(encoding='utf-8'))
    root = Path(cfg['output_dir']) / mode
    path = root / REPORT_FILE[mode]
    if not path.is_file():
        problems.append(f'{name}: {REPORT_FILE[mode]} missing')
        return (cfg, {}, {}, str(path))
    if (mode == 'diagnose') != (cfg.get('exploratory') is True):
        problems.append(f'{name}: exploratory flag does not match {mode} mode')
    snapshot = root / 'configuration.json'
    if snapshot.is_file():
        stored = json.loads(snapshot.read_text(encoding='utf-8'))
        if stored.get('mode') != mode or {k: v for k, v in stored.get('config', {}).items() if k != 'data_dir'} != {k: v for k, v in cfg.items() if k != 'data_dir'}:
            problems.append(f'{name}: configuration snapshot differs from the configured configuration')
    else:
        stored = None
        problems.append(f'{name}: configuration snapshot missing')
    report = json.loads(path.read_text(encoding='utf-8'))
    if stored is not None and report.get('config_hash') != config_digest(stored.get('config')):
        problems.append(f'{name}: report config_hash does not match the configuration snapshot')
    runs = {}
    for run in report['runs']:
        key = (run['source'], run['target'], run['arm'], run['seed'])
        if key in runs:
            problems.append(f'{name}: duplicate run {key}')
        if run.get('status') != 'complete' or not run.get('target_test_accessed'):
            problems.append(f'{name}: incomplete run {key}')
        if not math.isfinite(run['epsilon']) or run['epsilon'] > cfg['epsilon_budget'] * (1 + 1e-09):
            problems.append(f'{name}: epsilon not finite or above budget {key}')
        test = run['target_test']
        if run.get('eval_seed') != cfg['eval_seed'] or test.get('domain') != run['target'] or test.get('seed') != cfg['eval_seed'] or (test.get('split') != 'test') or (test.get('role') != 'final_report'):
            problems.append(f'{name}: evaluation protocol mismatch {key}')
        if not all((math.isfinite(f(test)) for f in METRICS.values())):
            problems.append(f'{name}: non-finite metric {key}')
        if 'delta' not in run:
            problems.append(f'{name}: delta missing {key}')
        runs[key] = run
    expected = {(s, t, a, seed) for s, t in cfg['directions'] for a in cfg['arms'] for seed in cfg['training_seeds']}
    if set(runs) != expected:
        problems.append(f'{name}: run set differs from configuration')
    for s, t in cfg['directions']:
        for a in cfg['arms']:
            values = [runs[s, t, a, seed]['target_test']['mrr'] for seed in cfg['training_seeds'] if (s, t, a, seed) in runs]
            if len(values) > 1 and len(set(values)) == 1:
                problems.append(f'{name}: identical MRR across seeds {s}-{t} {a}')
    return (cfg, report, runs, str(path))

def compare(left, right, direction, arm, seeds, metric, problems, label):
    diffs = []
    for seed in seeds:
        a, b = (left.get((*direction, arm, seed)), right.get((*direction, arm, seed)))
        if a is None or b is None:
            problems.append(f'{label}: missing pair {direction} {arm} {seed}')
            return None
        if a['delta'] != b['delta']:
            problems.append(f'{label}: delta differs {direction} {seed}')
        if pairing_signature(a) != pairing_signature(b) or None in pairing_signature(a):
            problems.append(f'{label}: test set size or initialization differs {direction} {arm} {seed}')
        diffs.append(METRICS[metric](a['target_test']) - METRICS[metric](b['target_test']))
    return interval(diffs)

def question(tag, mode, configs):
    try:
        return _question(tag, mode, configs)
    except Exception as error:
        return dict(state='invalid_input', mode=mode, epsilon=None, delta_min=MRR_THRESHOLD, intervals={}, descriptive={}, arm_means={}, problems=[f'{type(error).__name__}: {error}'], descriptive_problems=[], runner_states_not_adopted={}, inputs={})

def _question(tag, mode, configs):
    problems, desc_problems = ([], [])
    cg, lg = (load(f'{tag}_{k}', mode, problems, configs) for k in ('cg', 'lg'))
    seeds = cg[0]['training_seeds']
    if len(seeds) != 5 or lg[0]['training_seeds'] != seeds or lg[0]['directions'] != cg[0]['directions']:
        problems.append('seed or direction lists differ between cg and lg, or not five seeds')
    if cg[0]['epsilon_budget'] != lg[0]['epsilon_budget']:
        problems.append('epsilon budgets differ between cg and lg')
    directions = [tuple(d) for d in cg[0]['directions']]
    intervals = {}
    for d in directions:
        intervals['-'.join(d)] = compare(cg[2], lg[2], d, 'H1', seeds, 'batch_mrr', problems, tag) if not problems else None
    if problems or any((v is None for v in intervals.values())):
        state = 'invalid_input'
    elif all((v['lower'] >= MRR_THRESHOLD for v in intervals.values())):
        state = 'supported'
    elif any((v['upper'] < MRR_THRESHOLD for v in intervals.values())):
        state = 'not_supported'
    else:
        state = 'undetermined'
    descriptive, means = ({}, {})
    if not problems:
        for arm in ('H1', 'R1'):
            for metric in METRICS:
                for d in directions:
                    descriptive[f'cg-lg/{arm}/{metric}/{"-".join(d)}'] = compare(cg[2], lg[2], d, arm, seeds, metric, desc_problems, tag)
        for kind, data in (('cg', cg), ('lg', lg)):
            for arm in ('R1', 'H1'):
                for d in directions:
                    for m in METRICS:
                        means[f'{kind}/{arm}/{"-".join(d)}/{m}'] = statistics.mean((METRICS[m](data[2][*d, arm, s]['target_test']) for s in seeds))
    return dict(state=state, mode=mode, epsilon=cg[0]['epsilon_budget'], delta_min=MRR_THRESHOLD, intervals=intervals, descriptive=descriptive, arm_means=means, problems=problems, descriptive_problems=desc_problems, runner_states_not_adopted={k: d[1].get('state') for k, d in (('cg', cg), ('lg', lg))}, inputs={k: dict(report=d[3], report_sha256=sha256(d[3]) if Path(d[3]).is_file() else None, config_sha256=sha256(configs[f'{tag}_{k}'])) for k, d in (('cg', cg), ('lg', lg))})

def analyse(output, configs=None):
    configs = configs or CONFIGS
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit('Output directory must be empty')
    results = {q: question(tag, mode, configs) for q, (tag, mode) in QUESTIONS.items()}
    report = dict(questions=results, rule='per question: lower95 >= 0.01 both directions -> supported; upper95 < 0.01 any direction -> not_supported; else undetermined; integrity failure -> invalid_input', multiplicity='none across questions (specified)', script_sha256=sha256(__file__), scope='Paired seeds measure training randomness on fixed splits and a fixed test set.')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    lines = ['# comparison analysis', '']
    for q, r in results.items():
        lines += [f'## {q} (epsilon={r["epsilon"]}, {r["mode"]} mode): **{r["state"]}**', '']
        for d, v in r['intervals'].items():
            if v:
                lines.append(f'- H1 batch MRR {d}: {100 * v["mean"]:+.2f} [{100 * v["lower"]:+.2f}, {100 * v["upper"]:+.2f}] points')
        lines += ['', f'Runner states (not adopted): {r["runner_states_not_adopted"]}', '', '### Descriptive (points x100)', '']
        lines += [f'- {k}: {100 * v["mean"]:+.2f} [{100 * v["lower"]:+.2f}, {100 * v["upper"]:+.2f}]' for k, v in r['descriptive'].items() if v]
        lines += ['', '### Arm means (x100)', ''] + [f'- {k}: {100 * v:.2f}' for k, v in r['arm_means'].items()]
        if r['problems']:
            lines += ['', '### Integrity problems (state invalid_input)', ''] + [f'- {p}' for p in r['problems']]
        if r['descriptive_problems']:
            lines += ['', '### Descriptive-only problems', ''] + [f'- {p}' for p in r['descriptive_problems']]
        lines.append('')
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    report = analyse(parser.parse_args().output)
    print({q: r['state'] for q, r in report['questions'].items()})
if __name__ == '__main__':
    main()
