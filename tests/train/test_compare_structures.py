import importlib.util
import json
from pathlib import Path
import pytest
import yaml
REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('comparison', REPO / 'scripts/compare_structures.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)

def setup(tmp_path, h1_gain, drop=None, eps=10.0, cgcons=True, report_edit=None):
    configs = {}
    for name in ('cg', 'lg', 'cgcons'):
        cfg = yaml.safe_load(comparison.CONFIGS[name].read_text(encoding='utf-8'))
        cfg['output_dir'] = str(tmp_path / name)
        path = tmp_path / f'{name}.yaml'
        path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
        configs[name] = path
        if name == 'cgcons' and (not cgcons):
            continue
        root = tmp_path / name / 'confirm'
        root.mkdir(parents=True)
        (root / 'configuration.json').write_text(json.dumps(dict(config=cfg, mode='confirm')))
        runs = []
        for s, t in cfg['directions']:
            for arm in cfg['arms']:
                for i, seed in enumerate(cfg['training_seeds']):
                    if drop == (name, s, arm, seed):
                        continue
                    base = 0.3 + 0.001 * i
                    if name == 'cg' and arm == 'H1':
                        base += h1_gain[f'{s}-{t}'] + 0.002 * (i % 2)
                    runs.append(dict(source=s, target=t, arm=arm, seed=seed, status='complete', target_test_accessed=True, epsilon=eps if name == 'cg' else 10.0, delta=1e-06, eval_seed=cfg['eval_seed'], initial_parameter_sha256=f'init-{seed}', target_test=dict(mrr=base, domain=t, seed=cfg['eval_seed'], split='test', role='final_report', n_relations=10000, batch_size=256, tail_batch_size=16, full_corpus=dict(mrr=base / 5, recall_at_10=base / 3, recall_at_100=base / 2, n=10000))))
        report = dict(state='supported', runs=runs, config_hash=comparison.config_digest(cfg))
        if report_edit:
            report_edit(name, report)
        (root / 'final_report.json').write_text(json.dumps(report))
    return configs

@pytest.mark.parametrize('gain,state', [({'CN-US': 0.03, 'US-CN': 0.03}, 'supported'), ({'CN-US': 0.03, 'US-CN': -0.02}, 'not_supported'), ({'CN-US': 0.03, 'US-CN': 0.009}, 'undetermined')])
def test_decision_states(tmp_path, gain, state):
    report = comparison.analyse(tmp_path / 'out', setup(tmp_path, gain))
    assert report['structure_comparison']['state'] == state

@pytest.mark.parametrize('kwargs', [dict(drop=('lg', 'CN', 'H1', 409)), dict(eps=10.1)])
def test_integrity_failures_make_comparison_invalid(tmp_path, kwargs):
    report = comparison.analyse(tmp_path / 'out', setup(tmp_path, {'CN-US': 0.03, 'US-CN': 0.03}, **kwargs))
    assert report['structure_comparison']['state'] == 'invalid_input' and report['comparison_problems']

def test_missing_control_does_not_affect_comparison(tmp_path):
    report = comparison.analyse(tmp_path / 'out', setup(tmp_path, {'CN-US': 0.03, 'US-CN': 0.03}, cgcons=False))
    assert report['structure_comparison']['state'] == 'supported' and report['descriptive_problems']

def test_direction_of_subtraction(tmp_path):
    report = comparison.analyse(tmp_path / 'out', setup(tmp_path, {'CN-US': 0.03, 'US-CN': 0.03}))
    assert report['structure_comparison']['intervals']['CN-US']['mean'] == pytest.approx(0.03 + 0.002 * 2 / 5)

def _h1(report):
    return next((r for r in report['runs'] if r['arm'] == 'H1'))

@pytest.mark.parametrize('edit', [lambda name, r: name == 'cg' and r.update(config_hash='0' * 64), lambda name, r: name == 'lg' and _h1(r)['target_test'].update(n_relations=123), lambda name, r: name == 'lg' and _h1(r)['target_test']['full_corpus'].update(n=9999), lambda name, r: name == 'cg' and _h1(r).update(initial_parameter_sha256='other'), lambda name, r: name == 'cg' and [_h1(r)['target_test'].pop(k) for k in ('seed', 'split', 'role')]])
def test_integrity_gaps_reject_input(tmp_path, edit):
    report = comparison.analyse(tmp_path / 'out', setup(tmp_path, {'CN-US': 0.03, 'US-CN': 0.03}, report_edit=edit))
    assert report['structure_comparison']['state'] == 'invalid_input' and report['comparison_problems']
