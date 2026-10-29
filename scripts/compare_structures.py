import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import yaml
REPO = Path(__file__).resolve().parents[1]
CONFIGS = {'cg': 'configs/train/cs_candidate_graph_eps10.yaml', 'lg': 'configs/train/cs_public_groups_eps10.yaml', 'cgcons': 'configs/train/cs_candidate_graph_raw_accountant_eps10.yaml'}
CONFIGS = {k: REPO / v for k, v in CONFIGS.items()}
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

def load(name, problems, configs=None):
    configs = configs or CONFIGS
    cfg = yaml.safe_load(Path(configs[name]).read_text(encoding='utf-8'))
    root = Path(cfg['output_dir']) / 'confirm'
    path = root / 'final_report.json'
    if not path.is_file():
        problems.append(f'{name}: final_report.json missing')
        return (cfg, {}, {}, str(path))
    snapshot = root / 'configuration.json'
    if snapshot.is_file():
        stored = json.loads(snapshot.read_text(encoding='utf-8'))
        if stored.get('mode') != 'confirm' or {k: v for k, v in stored.get('config', {}).items() if k != 'data_dir'} != {k: v for k, v in cfg.items() if k != 'data_dir'}:
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
        runs[key] = run
    expected = {(s, t, a, seed) for s, t in cfg['directions'] for a in cfg['arms'] for seed in cfg['training_seeds']}
    if set(runs) != expected:
        problems.append(f'{name}: run set differs from configuration')
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

def analyse(output, configs=None):
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit('Output directory must be empty')
    problems, desc_problems, data = ([], [], {})
    for name in ('cg', 'lg'):
        data[name] = load(name, problems, configs)
    data['cgcons'] = load('cgcons', desc_problems, configs)
    seeds = data['cg'][0]['training_seeds']
    if len(seeds) != 5 or data['lg'][0]['training_seeds'] != seeds or data['lg'][0]['directions'] != data['cg'][0]['directions']:
        problems.append('seed or direction lists differ between cg and lg, or not five seeds')
    if data['cgcons'][0]['training_seeds'] != seeds:
        desc_problems.append('cgcons seeds differ')
    directions = [tuple(d) for d in data['cg'][0]['directions']]
    runs = {n: data[n][2] for n in data}
    structure_comparison = {}
    for d in directions:
        structure_comparison['-'.join(d)] = compare(runs['cg'], runs['lg'], d, 'H1', seeds, 'batch_mrr', problems, 'structure_comparison') if not problems else None
    if problems or any((v is None for v in structure_comparison.values())):
        state = 'invalid_input'
    elif all((v['lower'] >= MRR_THRESHOLD for v in structure_comparison.values())):
        state = 'supported'
    elif any((v['upper'] < MRR_THRESHOLD for v in structure_comparison.values())):
        state = 'not_supported'
    else:
        state = 'undetermined'
    descriptive = {}
    for label, (left, right), arms in (('cg-lg', ('cg', 'lg'), ('H1', 'R1')), ('cg-cgcons', ('cg', 'cgcons'), ('H1', 'R1'))):
        for arm in arms:
            for metric in METRICS:
                for d in directions:
                    descriptive[f'{label}/{arm}/{metric}/{"-".join(d)}'] = compare(runs[left], runs[right], d, arm, seeds, metric, desc_problems, label)
    means = {f'{n}/{arm}/{"-".join(d)}/{m}': statistics.mean((METRICS[m](runs[n][*d, arm, s]['target_test']) for s in seeds)) for n in runs for arm in ('R1', 'H1') for d in directions for m in METRICS if all(((*d, arm, s) in runs[n] for s in seeds))}
    report = dict(structure_comparison=dict(state=state, delta_min=MRR_THRESHOLD, rule='lower95 >= 0.01 both directions -> supported; upper95 < 0.01 any direction -> not_supported; else undetermined', intervals=structure_comparison), sampler_comparison_candidate_graph=data['cg'][1].get('state'), runner_states_not_adopted={n: data[n][1].get('state') for n in ('lg', 'cgcons')}, descriptive=descriptive, arm_means=means, comparison_problems=problems, descriptive_problems=desc_problems, inputs={n: dict(final_report=data[n][3], final_report_sha256=sha256(data[n][3]) if Path(data[n][3]).is_file() else None, config_sha256=sha256((configs or CONFIGS)[n])) for n in data}, script_sha256=sha256(__file__), scope='Paired seeds measure training randomness on fixed splits and a fixed test set.')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    lines = ['# training analysis', '', f'comparison (H1 batch MRR, cg - lg): **{state}**', '']
    for d, v in structure_comparison.items():
        if v:
            lines.append(f'- {d}: {100 * v["mean"]:+.2f} [{100 * v["lower"]:+.2f}, {100 * v["upper"]:+.2f}] points')
    lines += ['', f'comparison (runner, cg H1 - R1, delta_min 0.02): {report["sampler_comparison_candidate_graph"]}', f'Not adopted: {report["runner_states_not_adopted"]}', '', '## Descriptive (points x100)', '']
    for k, v in descriptive.items():
        if v:
            lines.append(f'- {k}: {100 * v["mean"]:+.2f} [{100 * v["lower"]:+.2f}, {100 * v["upper"]:+.2f}]')
    lines += ['', '## Arm means (x100)', ''] + [f'- {k}: {100 * v:.2f}' for k, v in means.items()]
    if problems:
        lines += ['', '## comparison integrity problems (state invalid_input)', ''] + [f'- {p}' for p in problems]
    if desc_problems:
        lines += ['', '## Descriptive-only problems (do not affect comparison)', ''] + [f'- {p}' for p in desc_problems]
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    print(analyse(parser.parse_args().output)['structure_comparison']['state'])
if __name__ == '__main__':
    main()
