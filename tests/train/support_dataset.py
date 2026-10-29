import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from quotaguard.data.buckets import PublicBuckets

@pytest.fixture
def bucket_dataset(tmp_path):
    ids = tuple((f'v{i:02}' for i in range(12)))
    pairs = np.arange(12).reshape(-1, 2)
    buckets = PublicBuckets(ids, pairs, np.arange(6), np.ones(6, dtype=int), 271828, 4, 'synthetic')
    path = tmp_path / 'public.npz'
    buckets.save(path)
    root = tmp_path / 'CN'
    (root / 'splits').mkdir(parents=True)
    (root / 'entities.tsv').write_text(''.join((f'{i}\tpublic title {i}\n' for i in ids)))
    (root / 'splits/train.tsv').write_text('v00\tv01\nv02\tv03\nv04\tv05\n')
    summary = dict(buckets.validate(), artifact=path.name, artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), public_train_size=6, public_entities=12, N_min=9)
    manifest = dict(status='passed', domains={'CN': {}}, sha256={}, stable_truncation=dict(version=1, scheme='synthetic', kdeg=4, seed=271828, domains={'CN': summary}))
    for relative in ('CN/entities.tsv', 'CN/splits/train.tsv'):
        manifest['sha256'][relative] = hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest()
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    cfg = dict(stable_truncation=True, noise_calibration='mog', data_dir=str(tmp_path), seed=29, clip_seed=271828, eval_seed=314159, kdeg=4, expected_batch_size=2, steps=3, epsilon_budget=10, k_neg=1, tau_sampler=0.2, tau_loss=0.1, learning_rate=0.001, refresh_every=100, encode_batch_size=12, C=0.1, device='cpu', model={})
    return (cfg, manifest)
