import copy
import itertools
import json
from pathlib import Path
import shutil
import uuid
import numpy as np
import pytest
import yaml
from scripts import candidate_coverage as diagnostic
from quotaguard.data.buckets import PublicBuckets, construct_buckets, title_neighbors
REPO = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((REPO / 'configs/candidate_coverage.yaml').read_text(encoding='utf-8'))
LEGACY = yaml.safe_load((REPO / CONFIG['legacy_config']).read_text(encoding='utf-8'))
BINS = CONFIG['degree_bins']

def edges(pairs):
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)

def test_undirected_deduplication_and_mutual_requires_reverse_direction():
    neighbors = np.array([[1, 1, 2, -1], [0, 0, -1, -1], [-1, -1, -1, -1], [3, 2, -1, -1]])
    union, mutual = diagnostic.candidate_keys(neighbors, 4)
    assert union.tolist() == [1, 2, 11]
    assert mutual.tolist() == [1]
    assert set(mutual) <= set(union)
    assert diagnostic.candidate_keys(neighbors, 1)[1].tolist() == [1]
    assert diagnostic.candidate_keys(np.array([[1, 1], [-1, -1]]), 2)[1].tolist() == []

def test_neighbor_prefix_does_not_include_later_reverse_edge():
    neighbors = np.array([[1, 2], [2, 0], [-1, -1]])
    small, mutual_small = diagnostic.candidate_keys(neighbors, 1)
    large, mutual_large = diagnostic.candidate_keys(neighbors, 2)
    assert small.tolist() == [1, 5]
    assert mutual_small.tolist() == []
    assert mutual_large.tolist() == [1]
    assert set(small) <= set(large)

def test_coverage_counts_full_universe_including_isolates():
    raw = edges([(1, 0), (1, 2), (2, 3), (2, 4)])
    candidates = np.unique(diagnostic.pair_keys(edges([(0, 1), (2, 4), (4, 5)]), 7))
    result = diagnostic.coverage(raw, 7, candidates, BINS)
    assert result['coverage'] == 0.5
    assert result['covered_relations'] == 2
    assert result['candidate_pairs'] == 3
    assert result['mean_candidates_per_entity'] == pytest.approx(6 / 7)
    assert result['degree_strata']['endpoint_incidence']['1'] == {'total': 3, 'covered': 2, 'coverage': 2 / 3}
    assert result['degree_strata']['endpoint_incidence']['2-4']['total'] == 5
    for method, bins in result['degree_strata'].items():
        assert sum((row['total'] for row in bins.values())) == (8 if method == 'endpoint_incidence' else 4)
        assert sum((row['covered'] for row in bins.values())) == (4 if method == 'endpoint_incidence' else 2)

def test_degree_strata_use_raw_degrees_in_all_four_bins():
    raw, offset = ([], 0)
    for count in (1, 3, 8, 18):
        raw.extend(((offset, offset + i + 1) for i in range(count)))
        offset += count + 1
    raw = edges(raw)
    candidates = np.sort(diagnostic.pair_keys(raw[[0, 1, 4, 12]], offset))
    result = diagnostic.coverage(raw, offset, candidates, BINS)
    maximum = result['degree_strata']['max_endpoint']
    assert [row['total'] for row in maximum.values()] == [1, 3, 8, 18]
    assert [row['covered'] for row in maximum.values()] == [1, 1, 1, 1]
    assert result['degree_strata']['endpoint_incidence']['>16']['total'] == 18

def test_empty_graph_and_empty_candidates_have_explicit_null_denominators():
    result = diagnostic.coverage(edges([]), 3, np.array([], dtype=np.int64), BINS)
    assert result['coverage'] is None
    assert result['mean_candidates_per_entity'] == 0
    assert all((row['coverage'] is None for bins in result['degree_strata'].values() for row in bins.values()))
    assert diagnostic.coverage(edges([(0, 1)]), 3, np.array([], dtype=np.int64), BINS)['coverage'] == 0

@pytest.mark.parametrize('raw', [edges([(0, 0)]), edges([(0, 1), (1, 0)]), edges([(0, 4)])])
def test_invalid_raw_graph_is_rejected(raw):
    with pytest.raises(ValueError):
        diagnostic.coverage(raw, 4, np.array([], dtype=np.int64), BINS)

def test_three_nonzero_loss_terms_partition_dropped_relations():
    raw = edges([(0, 1), (0, 2), (1, 2), (3, 4), (2, 4)])
    buckets = PublicBuckets(tuple('abcdef'), edges([(0, 1), (0, 2), (1, 2)]), np.array([0, 0, 1]), np.array([1, 1]), 271828, 4, 'toy')
    candidates = np.sort(diagnostic.pair_keys(raw[:4], 6))
    result = diagnostic.loss_decomposition(raw, buckets, candidates)
    assert result['retained_relations'] == 2
    assert result['retention'] == 0.4
    assert result['B'] == 2
    assert result['capacity_utilization'] == 1
    assert result['nonisolated_entity_coverage'] == 3 / 5
    assert [loss['count'] for loss in result['losses'].values()] == [1, 1, 1]
    assert result['loss_sum'] == pytest.approx(1 - result['retention'])
    assert result['raw_knn_correction']['count'] == 0

def test_clique_fallback_expansion_is_reported_without_false_knn_identity():
    raw = edges([(0, 1), (0, 2), (1, 2), (2, 3)])
    buckets = PublicBuckets(tuple('abcd'), raw[:3], np.zeros(3, dtype=int), np.array([1]), 271828, 4, 'toy_clique')
    knn = np.sort(diagnostic.pair_keys(raw[[0, 3]], 4))
    result = diagnostic.loss_decomposition(raw, buckets, knn)
    assert result['structure_pairs_outside_knn'] == 2
    assert result['raw_relations_in_structure_outside_knn'] == 2
    assert result['candidate_pairs'] == 4
    assert result['losses']['candidate_generation']['count'] == 0
    assert result['raw_knn_losses']['candidate_generation']['count'] == 2
    assert result['raw_knn_correction']['count'] == -2
    assert result['loss_sum'] == pytest.approx(1 - result['retention'])
    corrected = sum((row['fraction'] for row in result['raw_knn_losses'].values()))
    corrected += result['raw_knn_correction']['fraction']
    assert corrected == pytest.approx(1 - result['retention'])

@pytest.mark.parametrize('scheme', ['text_groups', 'singleton_knn'])
def test_actual_public_constructor_partition_for_every_small_graph(scheme):
    ids = tuple('abcde')
    neighbors = np.array([[1, -1], [0, 2], [1, -1], [-1, -1], [-1, -1]])
    scores = (neighbors >= 0).astype(float)
    buckets = construct_buckets(ids, neighbors, scores, LEGACY, scheme)
    union, _ = diagnostic.candidate_keys(neighbors, 2)
    universe = list(itertools.combinations(range(5), 2))
    for mask in range(1 << len(universe)):
        raw = edges([edge for i, edge in enumerate(universe) if mask >> i & 1])
        result = diagnostic.loss_decomposition(raw, buckets, union)
        assert result['identity_verified']
        if len(raw):
            assert result['loss_sum'] == pytest.approx(1 - result['retention'])
        else:
            assert result['loss_sum'] is None

@pytest.fixture
def synthetic_dir():
    root = REPO / 'tmp' / f'coverage-tests-{uuid.uuid4().hex}'
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        if not root.resolve().is_relative_to((REPO / 'tmp').resolve()):
            raise ValueError('Synthetic cleanup escaped worktree/tmp')
        shutil.rmtree(root)

@pytest.fixture
def synthetic_run(synthetic_dir, monkeypatch):
    tmp_path = synthetic_dir
    cfg = copy.deepcopy(CONFIG)
    monkeypatch.setattr(diagnostic, 'REPO', tmp_path)
    legacy = copy.deepcopy(LEGACY)
    legacy['tfidf'].update(min_df=1, max_df=1.0)
    legacy_path = tmp_path / cfg['legacy_config']
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_text(yaml.safe_dump(legacy), encoding='utf-8')
    for relative in ('src/quotaguard/data/buckets.py', 'src/quotaguard/train/data.py'):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# synthetic provenance placeholder\n', encoding='utf-8')
    for name, spec in cfg['datasets'].items():
        root = tmp_path / Path(spec['root']).name
        manifest = dict(status='passed', domains={'CN': {}, 'US': {}}, sha256={}, splits={})
        for domain in cfg['domains']:
            prefix = f'{name}_{domain}'
            ids = [f'{prefix}_{i}' for i in range(6)]
            texts = ['graph neural networks', 'graph neural learning', 'protein cell biology', 'protein cell networks', 'graph cell learning', 'isolated quantum optics']
            files = {'entities.tsv': ''.join((f'{i}\t{t}\n' for i, t in zip(ids, texts)))}
            if name == 'BIO':
                files['abstracts.tsv'] = ''.join((f'{i}\t{("graph cell" if j % 2 else "")}\n' for j, i in enumerate(ids)))
                files['splits/train.tsv'] = ''.join((f'{ids[a]}\t{ids[b]}\n' for a, b in [(0, 1), (1, 2), (2, 3), (3, 4)]))
                manifest['splits'][domain] = {'train': 4}
            else:
                files['splits/train.tsv'] = 'FORBIDDEN EVALUATION RELATIONS'
            files['relations.tsv'] = 'FORBIDDEN FULL RELATIONS'
            files['splits/test.tsv'] = 'FORBIDDEN TEST RELATIONS'
            files['splits/val.tsv'] = 'FORBIDDEN VALIDATION RELATIONS'
            for relative, payload in files.items():
                path = root / domain / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload.encode('utf-8'))
                manifest['sha256'][f'{domain}/{relative}'] = diagnostic.sha256(path)
        manifest_path = root / 'manifest.json'
        manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
        spec.update(root=str(root), manifest_sha256=diagnostic.sha256(manifest_path))
    cfg_path = tmp_path / 'diagnostic.yaml'
    cfg_path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    opened = []
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        resolved = path.resolve()
        for name, spec in cfg['datasets'].items():
            root = Path(spec['root'])
            if resolved.is_relative_to(root):
                relative = resolved.relative_to(root).as_posix()
                allowed = {'manifest.json', 'CN/entities.tsv', 'US/entities.tsv'}
                if name == 'BIO':
                    allowed |= {'CN/abstracts.tsv', 'US/abstracts.tsv', 'CN/splits/train.tsv', 'US/splits/train.tsv'}
                assert relative in allowed, f'Forbidden read: {name}/{relative}'
                opened.append((name, relative))
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded_open)
    return (cfg, cfg_path, opened)

def update_manifest(cfg, name):
    root = Path(cfg['datasets'][name]['root'])
    path = root / 'manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    for domain in ('CN', 'US'):
        relative = f'{domain}/entities.tsv'
        manifest['sha256'][relative] = diagnostic.sha256(root / relative)
    path.write_text(json.dumps(manifest), encoding='utf-8')
    cfg['datasets'][name]['manifest_sha256'] = diagnostic.sha256(path)

def test_overlap_across_domains_aborts_before_bio_relations_or_neighbors(synthetic_run, monkeypatch):
    cfg, path, opened = synthetic_run
    entity_path = Path(cfg['datasets']['CS']['root']) / 'US/entities.tsv'
    entity_path.write_text('BIO_CN_0\tunused title\n', encoding='utf-8')
    update_manifest(cfg, 'CS')
    path.write_text(yaml.safe_dump(cfg), encoding='utf-8')

    def forbidden(*args, **kwargs):
        pytest.fail('No loading/neighbor/structure work is allowed after overlap')
    monkeypatch.setattr(diagnostic, 'load_domain', forbidden)
    monkeypatch.setattr(diagnostic, 'title_neighbors', forbidden)
    monkeypatch.setattr(diagnostic, 'construct_buckets', forbidden)
    report = diagnostic.run(path)
    assert report['status'] == 'aborted_overlap'
    assert report['boundary']['intersections']['CS']['CN:US'] == 1
    assert report['boundary']['intersections']['CS']['all_domains'] == 1
    assert report['domains'] == {}
    assert all(('splits' not in relative and 'abstracts' not in relative for _, relative in opened))
    output = next(path.parent.glob('runs/coverage'))
    assert all(((output / name).exists() for name in ('report.json', 'report.md', 'MANIFEST.json')))

def test_end_to_end_raw_train_abstracts_manifest_and_single_top256_call(synthetic_run, monkeypatch):
    cfg, path, opened = synthetic_run
    calls = []

    def neighbors(ids, texts, options, **kwargs):
        calls.append((len(ids), options['neighbors']))
        return title_neighbors(ids, texts, options, **kwargs)
    monkeypatch.setattr(diagnostic, 'title_neighbors', neighbors)
    report = diagnostic.run(path)
    assert report['status'] == 'complete', report.get('error')
    assert calls == [(6, 256)] * 4
    assert report['boundary']['passed']
    for domain in cfg['domains']:
        data = report['domains'][domain]
        assert data['raw_train_relations'] == 4
        assert data['isolated_entities'] == 1
        assert data['text_modes']['title_abstract']['abstracts']['missing_rate'] == 0.5
        assert set(data['text_modes']['title']['limits']) == {'16', '64', '256'}
        assert set(data['legacy']) == {'text_groups', 'singleton_knn'}
    output = next(path.parent.glob('runs/coverage'))
    manifest = json.loads((output / 'MANIFEST.json').read_text(encoding='utf-8'))
    for name, relative in opened:
        assert str(Path(cfg['datasets'][name]['root']) / relative) in manifest['sha256']
    assert all((diagnostic.sha256(filename) == expected for filename, expected in manifest['sha256'].items()))
    markdown = (output / 'report.md').read_text(encoding='utf-8')
    assert 'endpoint_incidence' in markdown and 'text_groups' in markdown
    assert report['resources'] and report['traced_peak_bytes'] > 0
    assert all((row['elapsed_seconds'] >= 0 for row in report['resources'].values()))
    with pytest.raises(ValueError, match='overwritten'):
        diagnostic.run(path)

def test_missing_abstract_field_is_reported_and_not_read(synthetic_run):
    cfg, path, opened = synthetic_run
    root = Path(cfg['datasets']['BIO']['root'])
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for domain in cfg['domains']:
        del manifest['sha256'][f'{domain}/abstracts.tsv']
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    cfg['datasets']['BIO']['manifest_sha256'] = diagnostic.sha256(manifest_path)
    path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    report = diagnostic.run(path)
    assert report['status'] == 'complete', report.get('error')
    assert all((data['text_modes']['title_abstract']['status'] == 'abstracts unavailable' for data in report['domains'].values()))
    assert not any(('abstracts' in relative for _, relative in opened))

@pytest.mark.parametrize('fault', ['manifest', 'entities', 'clipped', 'train'])
def test_hash_or_truncated_source_failure_is_closed(synthetic_run, monkeypatch, fault):
    cfg, path, _ = synthetic_run
    root = Path(cfg['datasets']['BIO']['root'])
    if fault == 'manifest':
        cfg['datasets']['BIO']['manifest_sha256'] = '0' * 64
    elif fault == 'clipped':
        manifest_path = root / 'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        manifest['stable_truncation'] = {'version': 1}
        manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
        cfg['datasets']['BIO']['manifest_sha256'] = diagnostic.sha256(manifest_path)
    else:
        relative = 'CN/entities.tsv' if fault == 'entities' else 'CN/splits/train.tsv'
        with (root / relative).open('a', encoding='utf-8') as handle:
            handle.write('corrupt\trow\n')
    path.write_text(yaml.safe_dump(cfg), encoding='utf-8')

    def forbidden(*args, **kwargs):
        pytest.fail('No candidate generation is allowed after invalid input')
    monkeypatch.setattr(diagnostic, 'title_neighbors', forbidden)
    report = diagnostic.run(path)
    assert report['status'] == 'failed'
    assert 'checksum mismatch' in report['error'] or 'untruncated' in report['error']
    assert report['domains'] == {}

def test_explicit_reader_forbids_evaluation_relation_paths(synthetic_run):
    cfg, _, _ = synthetic_run
    inputs = diagnostic.Inputs(cfg)
    for name in ('CS', 'MAT'):
        for relative in ('CN/splits/train.tsv', 'US/relations.tsv', 'CN/abstracts.tsv'):
            with pytest.raises(ValueError, match='Forbidden input'):
                inputs.rows(name, relative)
    with pytest.raises(ValueError, match='Forbidden input'):
        inputs.rows('BIO', 'CN/splits/test.tsv')
