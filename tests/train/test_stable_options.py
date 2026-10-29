import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip('opacus')
from quotaguard.data.buckets import PublicBuckets
from quotaguard.train import dp, train
from support_runner import runner
from support_encoder import Encoder, tokenize
from support_dataset import bucket_dataset

def _with_expected(cfg, manifest, value=4.5):
    info = manifest['stable_truncation']['domains']['CN']
    info.update(public_expected_size=value, public_expected_derivation='sum of public probabilities')
    (Path(cfg['data_dir']) / 'manifest.json').write_text(json.dumps(manifest))

@pytest.mark.parametrize('method,R', [('mog', None), ('caseB', 1), ('caseB', 10)])
def test_zero_insertion_uses_bounded_accountant_at_T(bucket_dataset, method, R):
    cfg, _ = bucket_dataset
    cfg.update(noise_calibration=method, zero_insertion=True)
    if method == 'caseB':
        cfg.update(bounded=dict(mode='positive_relative', R=R, tau=0.1), caseB=dict(derive_from_source_train=True))
    prepared = runner.prepare(cfg, 'CN', 'US')
    kwargs = dict(epsilon=10, delta=1 / 6, steps=3, gamma=2 / 6, kdeg=4, tolerance=1e-10)
    if method == 'mog':
        expected = dp.calibrate_mog(**kwargs)
    else:
        assert prepared['caseB']['m_bar'] == 6
        expected = dp.calibrate_caseb(**kwargs, k_neg=1, **prepared['caseB'])
    assert prepared['sigma'] == expected['sigma']
    assert prepared['calibration']['method'].startswith('zero_insertion_')
    assert prepared['calibration']['calibration_T']['steps'] == 3
    assert 0 < prepared['calibration']['epsilon'] <= 10

def test_zero_insertion_training_reports_plain_accountant(bucket_dataset, monkeypatch):
    cfg, _ = bucket_dataset
    cfg.update(zero_insertion=True)
    prepared = runner.prepare(cfg, 'CN', 'US')
    monkeypatch.setattr(train, 'build_model', lambda *a, **k: (Encoder(), tokenize))
    _, result = train.run_training(prepared, 'R1')
    assert not result['accountant'].startswith('stable_')
    assert result['sigma'] == prepared['sigma'] and result['steps'] == 3
    assert result['epsilon'] == pytest.approx(prepared['calibration']['epsilon'], rel=1e-12)

def test_zero_insertion_rejects_non_singleton_pools(bucket_dataset, tmp_path):
    cfg, manifest = bucket_dataset
    root = Path(cfg['data_dir'])
    ids = tuple((f'v{i:02}' for i in range(12)))
    pairs = np.arange(12).reshape(-1, 2)
    pooled = PublicBuckets(ids, pairs, np.array([0, 0, 1, 2, 3, 4]), np.ones(5, dtype=int), 271828, 4, 'synthetic')
    path = root / 'public.npz'
    pooled.save(path)
    info = manifest['stable_truncation']['domains']['CN']
    info.update(pooled.validate(), artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), public_train_size=5)
    content = 'v00\tv01\nv04\tv05\n'
    (root / 'CN/splits/train.tsv').write_text(content)
    manifest['sha256']['CN/splits/train.tsv'] = hashlib.sha256(content.encode()).hexdigest()
    (root / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='singleton pools'):
        runner.prepare(dict(cfg, zero_insertion=True), 'CN', 'US')
    prepared = runner.prepare(cfg, 'CN', 'US')
    assert prepared['calibration']['method'].startswith('stable_')

@pytest.mark.parametrize('method', ['mog', 'caseB'])
def test_public_expected_scale_sets_gamma_and_keeps_capacity_for_case_b(bucket_dataset, method):
    cfg, manifest = bucket_dataset
    _with_expected(cfg, manifest)
    cfg.update(noise_calibration=method, zero_insertion=True, stable_scale='public_expected')
    if method == 'caseB':
        cfg.update(bounded=dict(mode='positive_relative', R=10, tau=0.1), caseB=dict(derive_from_source_train=True))
    prepared = runner.prepare(cfg, 'CN', 'US')
    assert prepared['public_train_size'] == 4.5
    assert prepared['gamma'] == 2 / 4.5 and prepared['delta'] == 1 / 4.5
    assert prepared['gamma'] * prepared['public_train_size'] == pytest.approx(2)
    if method == 'caseB':
        assert prepared['caseB']['m_bar'] == 6
    root = Path(cfg['data_dir'])
    for content in ('v00\tv01\n', ''):
        (root / 'CN/splits/train.tsv').write_text(content)
        manifest['sha256']['CN/splits/train.tsv'] = hashlib.sha256(content.encode()).hexdigest()
        (root / 'manifest.json').write_text(json.dumps(manifest))
        reduced = runner.prepare(cfg, 'CN', 'US')
        for key in ('public_train_size', 'gamma', 'delta', 'sigma', 'stable_public_bounds', 'calibration'):
            assert reduced[key] == prepared[key]

def test_entity_power_delta_and_missing_expected_scale(bucket_dataset):
    cfg, manifest = bucket_dataset
    prepared = runner.prepare(dict(cfg, stable_delta='entity_power'), 'CN', 'US')
    assert prepared['delta'] == 12 ** (-1.1)
    with pytest.raises(ValueError, match='public_expected scale missing'):
        runner.prepare(dict(cfg, stable_scale='public_expected'), 'CN', 'US')

@pytest.mark.parametrize('change', [dict(stable_scale='private_count'), dict(stable_delta='1/n'), dict(zero_insertion='yes')])
def test_reject_invalid_stable_options(bucket_dataset, change):
    cfg, _ = bucket_dataset
    with pytest.raises(ValueError):
        runner.prepare(dict(cfg, **change), 'CN', 'US')

def test_default_public_capacity_layout(bucket_dataset):
    cfg, _ = bucket_dataset
    prepared = runner.prepare(cfg, 'CN', 'US')
    bounds = prepared['stable_public_bounds']
    assert set(bounds) == {'public_train_size', 'public_entities', 'N_min', 'artifact_sha256', 'kdeg', 'seed', 'scheme', 'derivation'}
    assert type(prepared['public_train_size']) is int and prepared['public_train_size'] == 6
    assert prepared['delta'] == 1 / 6
    assert prepared['normalization_source'] == 'fixed public sum of bucket quotas * gamma'
    assert 'stable_capacity' not in prepared

@pytest.mark.parametrize('key,value', [('stable_scale', 'public_expected'), ('stable_delta', 'entity_power'), ('zero_insertion', True)])
def test_public_options_require_stable_truncation(bucket_dataset, key, value):
    from quotaguard.train.stable import validate_stable_options
    cfg, _ = bucket_dataset
    with pytest.raises(ValueError, match='require stable_truncation'):
        validate_stable_options(dict(cfg, stable_truncation=False, **{key: value}))

def test_pooled_structure_with_public_expected_keeps_raw_graph_accounting(bucket_dataset):
    cfg, manifest = bucket_dataset
    _with_expected(cfg, manifest)
    prepared = runner.prepare(dict(cfg, stable_scale='public_expected', stable_delta='entity_power'), 'CN', 'US')
    assert prepared['calibration']['method'].startswith('stable_')
    assert prepared['public_train_size'] == 4.5 and prepared['stable_capacity'] == 6
    assert prepared['delta'] == 12 ** (-1.1)
