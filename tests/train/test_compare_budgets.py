import importlib.util
import json
from pathlib import Path
import pytest
import yaml
REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('budget_comparison', REPO / 'scripts/compare_budgets.py')
budget_comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget_comparison)
GOOD = {'CN-US': 0.03, 'US-CN': 0.03}

def setup(tmp_path, gains=None, drop=None, eps_over=None, skip=None, flip_flag=None, mutate=None):
    gains = gains or {}
    configs = {}
    for q, (tag, mode) in budget_comparison.QUESTIONS.items():
        for kind in ('cg', 'lg'):
            name = f'{tag}_{kind}'
            cfg = yaml.safe_load(budget_comparison.CONFIGS[name].read_text(encoding='utf-8'))
            cfg['output_dir'] = str(tmp_path / name)
            if flip_flag == name:
                cfg['exploratory'] = not cfg.get('exploratory', False)
            path = tmp_path / f'{name}.yaml'
            path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
            configs[name] = path
            if skip == name:
                continue
            root = tmp_path / name / mode
            root.mkdir(parents=True)
            (root / 'configuration.json').write_text(json.dumps(dict(config=cfg, mode=mode)))
            runs = []
            for s, t in cfg['directions']:
                for arm in cfg['arms']:
                    for i, seed in enumerate(cfg['training_seeds']):
                        if drop == (name, s, arm, seed):
                            continue
                        base = 0.3 + 0.001 * i
                        if kind == 'cg' and arm == 'H1':
                            base += gains.get(q, GOOD)[f'{s}-{t}'] + 0.002 * (i % 2)
                        eps = cfg['epsilon_budget'] * (1.01 if eps_over == name else 1.0)
                        runs.append(dict(source=s, target=t, arm=arm, seed=seed, status='complete', target_test_accessed=True, epsilon=eps, delta=1e-06, eval_seed=cfg['eval_seed'], initial_parameter_sha256=f'init-{seed}', target_test=dict(mrr=base, domain=t, seed=cfg['eval_seed'], split='test', role='final_report', exploratory=mode == 'diagnose', n_relations=10000, batch_size=256, tail_batch_size=16, full_corpus=dict(mrr=base / 5, recall_at_10=base / 3, recall_at_100=base / 2, n=10000))))
            report = dict(state='supported', runs=runs)
            if mutate:
                mutate(name, cfg, root, runs)
            if (root / 'configuration.json').is_file():
                stored = json.loads((root / 'configuration.json').read_text())['config']
                report['config_hash'] = budget_comparison.config_digest(stored)
            if mutate and getattr(mutate, 'report', None):
                mutate.report(name, report)
            (root / budget_comparison.REPORT_FILE[mode]).write_text(json.dumps(report))
    return configs

def states(report):
    return {q: r['state'] for q, r in report['questions'].items()}

@pytest.mark.parametrize('gain,state', [({'CN-US': 0.03, 'US-CN': 0.03}, 'supported'), ({'CN-US': 0.03, 'US-CN': -0.02}, 'not_supported'), ({'CN-US': 0.03, 'US-CN': 0.009}, 'undetermined')])
@pytest.mark.parametrize('q', list(budget_comparison.QUESTIONS))
def test_decision_states_are_independent(tmp_path, q, gain, state):
    report = budget_comparison.analyse(tmp_path / 'out', setup(tmp_path, {q: gain}))
    assert states(report) == {k: state if k == q else 'supported' for k in budget_comparison.QUESTIONS}

@pytest.mark.parametrize('kwargs,q', [(dict(drop=('mat_lg', 'CN', 'H1', 409)), 'materials'), (dict(eps_over='eps4_cg'), 'budget4'), (dict(skip='eps1_lg'), 'budget1'), (dict(flip_flag='eps1_cg'), 'budget1')])
def test_integrity_failures_invalidate_only_their_question(tmp_path, kwargs, q):
    report = budget_comparison.analyse(tmp_path / 'out', setup(tmp_path, **kwargs))
    assert states(report) == {k: 'invalid_input' if k == q else 'supported' for k in budget_comparison.QUESTIONS}
    assert report['questions'][q]['problems']

def test_direction_of_subtraction_and_mode_paths(tmp_path):
    report = budget_comparison.analyse(tmp_path / 'out', setup(tmp_path))
    for q, (tag, mode) in budget_comparison.QUESTIONS.items():
        r = report['questions'][q]
        assert r['mode'] == mode and r['intervals']['CN-US']['mean'] == pytest.approx(0.03 + 0.002 * 2 / 5)
        assert r['inputs']['cg']['report'].endswith(budget_comparison.REPORT_FILE[mode])

def test_all_budgets_run_in_confirm():
    for tag, mode in budget_comparison.QUESTIONS.values():
        for kind in ('cg', 'lg'):
            cfg = yaml.safe_load(budget_comparison.CONFIGS[f'{tag}_{kind}'].read_text(encoding='utf-8'))
            assert (cfg.get('exploratory') is True) == (mode == 'diagnose')

def test_report_files_and_modes():
    assert budget_comparison.REPORT_FILE == {'confirm': 'final_report.json', 'diagnose': 'diagnose_report.json'}
    assert budget_comparison.QUESTIONS == {'materials': ('mat', 'confirm'), 'budget4': ('eps4', 'confirm'), 'budget1': ('eps1', 'confirm')}
    assert budget_comparison.MRR_THRESHOLD == 0.01

def test_interval_matches_scipy():
    from scipy import stats
    diffs = [0.011, 0.025, 0.018, 0.009, 0.031]
    v = budget_comparison.interval(diffs)
    lo, hi = stats.t.interval(0.95, 4, loc=stats.tmean(diffs), scale=stats.sem(diffs))
    assert v['lower'] == pytest.approx(lo, abs=1e-15) and v['upper'] == pytest.approx(hi, abs=1e-15)

def _run(target, s, arm, seed):
    return lambda name, cfg, root, runs: [r for r in runs if (r['source'], r['arm'], r['seed']) == (s, arm, seed) and name == target]

def _edit(target, fn):

    def mutate(name, cfg, root, runs):
        if name == target:
            fn(cfg, root, runs)
    return mutate

def _snapshot(change):

    def fn(cfg, root, runs):
        stored = json.loads((root / 'configuration.json').read_text())
        change(stored)
        (root / 'configuration.json').write_text(json.dumps(stored))
    return fn

def _first(change):

    def fn(cfg, root, runs):
        change(runs[0])
    return fn
INJECTIONS = {'delta differs': lambda cfg, root, runs: next((r for r in runs if r['arm'] == 'H1')).update(delta=2e-06), 'delta missing': _first(lambda r: r.pop('delta')), 'not accessed': _first(lambda r: r.update(target_test_accessed=False)), 'not complete': _first(lambda r: r.update(status='failed')), 'snapshot key differs': _snapshot(lambda c: c['config'].update(learning_rate=123.0)), 'snapshot mode differs': _snapshot(lambda c: c.update(mode='explore')), 'snapshot missing': lambda cfg, root, runs: (root / 'configuration.json').unlink(), 'wrong domain': _first(lambda r: r['target_test'].update(domain='XX')), 'wrong eval seed': _first(lambda r: r.update(eval_seed=1)), 'eval seed missing': _first(lambda r: r['target_test'].pop('seed')), 'split missing': _first(lambda r: r['target_test'].pop('split')), 'role differs': _first(lambda r: r['target_test'].update(role='validation')), 'nan metric': _first(lambda r: r['target_test'].update(mrr=float('nan'))), 'epsilon None': _first(lambda r: r.update(epsilon=None)), 'full corpus missing': _first(lambda r: r['target_test'].pop('full_corpus')), 'duplicate run': lambda cfg, root, runs: runs.append(json.loads(json.dumps(runs[0]))), 'identical seeds': lambda cfg, root, runs: [r['target_test'].update(mrr=0.5) for r in runs if (r['source'], r['arm']) == ('CN', 'R1')], 'test size differs': lambda cfg, root, runs: next((r for r in runs if r['arm'] == 'H1'))['target_test'].update(n_relations=123), 'corpus size differs': lambda cfg, root, runs: next((r for r in runs if r['arm'] == 'H1'))['target_test']['full_corpus'].update(n=9999), 'initialization differs': lambda cfg, root, runs: next((r for r in runs if r['arm'] == 'H1')).update(initial_parameter_sha256='other'), 'initialization missing': lambda cfg, root, runs: next((r for r in runs if r['arm'] == 'H1')).pop('initial_parameter_sha256')}

def _wrong_hash(target):

    def mutate(name, cfg, root, runs):
        pass

    def report(name, value):
        if name == target:
            value['config_hash'] = '0' * 64
    mutate.report = report
    return mutate

@pytest.mark.parametrize('target,q', [('mat_cg', 'materials'), ('eps1_lg', 'budget1')])
def test_report_hash_must_match_snapshot(tmp_path, target, q):
    report = budget_comparison.analyse(tmp_path / 'out', setup(tmp_path, mutate=_wrong_hash(target)))
    assert states(report) == {k: 'invalid_input' if k == q else 'supported' for k in budget_comparison.QUESTIONS}

@pytest.mark.parametrize('label', list(INJECTIONS))
@pytest.mark.parametrize('target,q', [('mat_lg', 'materials'), ('eps1_cg', 'budget1')])
def test_each_integrity_branch_fails_alone(tmp_path, label, target, q):
    report = budget_comparison.analyse(tmp_path / 'out', setup(tmp_path, mutate=_edit(target, INJECTIONS[label])))
    assert states(report) == {k: 'invalid_input' if k == q else 'supported' for k in budget_comparison.QUESTIONS}, label

def test_epsilon_budget_mismatch_between_structures(tmp_path):
    configs = setup(tmp_path)
    cfg = yaml.safe_load(configs['eps4_lg'].read_text(encoding='utf-8'))
    cfg['epsilon_budget'] = 10
    configs['eps4_lg'].write_text(yaml.safe_dump(cfg), encoding='utf-8')
    snap = Path(cfg['output_dir']) / 'confirm' / 'configuration.json'
    snap.write_text(json.dumps(dict(config=cfg, mode='confirm')))
    assert states(budget_comparison.analyse(tmp_path / 'out', configs))['budget4'] == 'invalid_input'

def test_data_dir_only_difference_is_accepted(tmp_path):
    configs = setup(tmp_path, mutate=_edit('mat_cg', _snapshot(lambda c: c['config'].update(data_dir='/elsewhere'))))
    assert states(budget_comparison.analyse(tmp_path / 'out', configs))['materials'] == 'supported'

def test_boundaries_are_inclusive_for_support_and_strict_for_rejection(tmp_path, monkeypatch):
    configs = setup(tmp_path, {'materials': {'CN-US': 0.03, 'US-CN': 0.05}})
    first = budget_comparison.analyse(tmp_path / 'a', configs)['questions']['materials']['intervals']
    lower = min((v['lower'] for v in first.values()))
    monkeypatch.setattr(budget_comparison, 'MRR_THRESHOLD', lower)
    assert budget_comparison.analyse(tmp_path / 'b', configs)['questions']['materials']['state'] == 'supported'
    upper = min((v['upper'] for v in first.values()))
    monkeypatch.setattr(budget_comparison, 'MRR_THRESHOLD', upper)
    assert budget_comparison.analyse(tmp_path / 'c', configs)['questions']['materials']['state'] == 'undetermined'
