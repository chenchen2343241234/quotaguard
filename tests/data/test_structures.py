import copy
from dataclasses import replace
import itertools
import json
from pathlib import Path
import shutil
import uuid
import numpy as np
import pytest
import yaml
from quotaguard.data.buckets import PublicBuckets, construct_buckets
from quotaguard.data.structures import Candidates, Forest, FixedTruncation, adversarial_checks, bucket_objectives, build_forest, candidate_bmatching, candidates_from_neighbors, evaluate_structure, exact_quota_dp, fit_calibration, forest_marginal_greedy, generate_candidates, legacy_text_groups, pair_keys, poisson_binomial, random_public_groups, stability_check
from scripts import build_structures as run
REPO = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load(run.CONFIG.read_text(encoding='utf-8'))
LEGACY = yaml.safe_load((REPO / 'configs/public_groups.yaml').read_text(encoding='utf-8'))

def tiny_forest(seed):
    rng = np.random.default_rng(seed)
    members = [np.array([i], np.int32) for i in range(4)]
    members += [np.array([0, 1], np.int32), np.array([2, 3], np.int32), np.arange(4, dtype=np.int32)]
    counts = rng.integers(0, 5, 7, dtype=np.int32)
    k, cap = (int(rng.integers(1, 5)), int(rng.integers(0, 9)))
    f = np.array([poisson_binomial(rng.choice([0.0, 0.1, 0.3, 0.7, 1.0], size=count), 4)[1] for count in counts])
    forest = Forest(tuple('abcd'), np.empty((0, 2), np.int32), np.empty(0, np.int32), np.array([-1, -1, -1, -1, 0, 2, 4], np.int32), np.array([-1, -1, -1, -1, 1, 3, 5], np.int32), np.array([4, 4, 5, 5, 6, 6, -1], np.int32), np.concatenate(members), np.r_[0, np.cumsum(list(map(len, members)))], np.array([6], np.int32), np.array([k], np.int32), np.array([cap], np.int32), counts, np.zeros(7, np.int32), seed, 4, cap)
    return (forest, f)

@pytest.mark.parametrize('seed', range(240))
def test_dp_equals_independent_exhaustive_enumeration(seed):
    forest, f = tiny_forest(seed)
    quota, objective = exact_quota_dp(forest, f)
    k, cap = (int(forest.budgets[0]), int(forest.capacities[0]))
    best, best_q, best_cap = (-np.inf, None, None)
    for vector in itertools.product(*(range(min(k, int(c)) + 1) for c in forest.counts)):
        c = sum(vector)
        if c > cap:
            continue
        charge = np.array(vector[:4]) + np.repeat(vector[4:6], 2) + vector[6]
        if np.any(charge > k):
            continue
        value = f[6, vector[6]] + (f[4, vector[4]] + f[0, vector[0]] + f[1, vector[1]]) + (f[5, vector[5]] + f[2, vector[2]] + f[3, vector[3]])
        if value > best or (value == best and (best_cap is None or (c, vector) < (best_cap, best_q))):
            best, best_q, best_cap = (value, vector, c)
    assert objective == pytest.approx(best, abs=2e-12)
    assert quota.sum() <= cap
    assert np.max(quota[:4] + np.repeat(quota[4:6], 2) + quota[6]) <= k
    greedy, greedy_value = forest_marginal_greedy(forest, f)
    assert objective + 2e-12 >= greedy_value

def test_dp_fixed_ties_and_unreachable_states():
    forest, f = tiny_forest(10)
    q, value = exact_quota_dp(forest, np.zeros_like(f))
    assert value == 0 and (not q.any())
    forest = replace(forest, counts=np.array([1, 1, 1, 1, 0, 0, 0], np.int32), capacities=np.array([1], np.int32))
    f[:] = 0
    f[:4, 1:] = 1
    q, value = exact_quota_dp(forest, f)
    assert value == 1 and q.tolist() == [0, 0, 0, 1, 0, 0, 0]

@pytest.mark.parametrize('seed', range(60))
def test_dp_random_topologies_and_multiple_roots(seed):
    rng = np.random.default_rng(seed + 1000)
    n = int(rng.integers(2, 5))
    members = [np.array([i], np.int32) for i in range(n)]
    left, right, parent, active = ([-1] * n, [-1] * n, [-1] * n, list(range(n)))
    stop = 2 if seed % 2 and n > 2 else 1
    while len(active) > stop:
        l, r = map(int, rng.choice(active, 2, replace=False))
        active.remove(l)
        active.remove(r)
        t = len(left)
        left.append(l)
        right.append(r)
        parent.append(-1)
        parent[l] = parent[r] = t
        members.append(np.concatenate((members[l], members[r])))
        active.append(t)
    nodes = len(left)
    counts = rng.integers(0, 3, nodes, dtype=np.int32)
    budgets = np.full(len(active), 2, np.int32)
    capacities = rng.integers(0, 6, len(active), dtype=np.int32)
    forest = Forest(tuple((str(i) for i in range(n))), np.empty((0, 2), np.int32), np.empty(0, np.int32), np.array(left, np.int32), np.array(right, np.int32), np.array(parent, np.int32), np.concatenate(members), np.r_[0, np.cumsum(list(map(len, members)))], np.array(active, np.int32), budgets, capacities, counts, np.zeros(nodes, np.int32), 101, 4, int(capacities.sum()))
    f = np.array([poisson_binomial(rng.random(count), 4)[1] for count in counts])
    descendants = []
    for root in active:
        subset, stack = ([], [root])
        while stack:
            t = stack.pop()
            subset.append(t)
            if left[t] >= 0:
                stack.extend((left[t], right[t]))
        descendants.append(subset)
    best = 0.0
    for vector in itertools.product(*(range(int(c) + 1) for c in counts)):
        if any((sum((vector[t] for t in subset)) > cap for subset, cap in zip(descendants, capacities))):
            continue
        charges = np.zeros(n, int)
        for group, q in zip(members, vector):
            charges[group] += q
        if charges.max() > 2:
            continue
        best = max(best, sum((f[t, q] for t, q in enumerate(vector))))
    q, value = exact_quota_dp(forest, f)
    assert value == pytest.approx(best, abs=2e-12)

@pytest.mark.parametrize('seed', range(30))
def test_poisson_binomial_equals_all_bernoulli_outcomes(seed):
    rng = np.random.default_rng(seed)
    p = rng.choice([0.0, 0.02, 0.3, 0.9, 1.0], size=int(rng.integers(0, 10)))
    k = int(rng.integers(1, 6))
    counts = np.zeros(k + 1)
    values = np.zeros(k + 1)
    for bits in itertools.product((0, 1), repeat=len(p)):
        probability = np.prod([pi if b else 1 - pi for pi, b in zip(p, bits)])
        count = sum(bits)
        counts[min(k, count)] += probability
        values += probability * np.minimum(np.arange(k + 1), count)
    actual, f = poisson_binomial(p, k)
    np.testing.assert_allclose(actual, counts, atol=2e-15)
    np.testing.assert_allclose(f, values, atol=2e-15)

def toy_candidates(n=24, seed=7):
    rng = np.random.default_rng(seed)
    ids = tuple((f'id-{i:04d}' for i in range(n)))
    nn = np.array([rng.permutation([j for j in range(n) if j != i]) for i in range(n)], np.int32)
    return candidates_from_neighbors(ids, nn, np.ones_like(nn, dtype=np.float32), np.full(n, 10), 64)

@pytest.mark.parametrize('g,s,budgets', list(itertools.product((4, 8), (16, 32), ([4], [2, 2], [1, 1, 1, 1]))))
def test_forest_pair_disjoint_budget_root_cap_and_dp_dominates_greedy(g, s, budgets):
    c = toy_candidates()
    forest, expanded = build_forest(c, g, s, budgets, 101)
    assert np.all(np.diff(expanded.keys) > 0)
    assert max((len(forest.entities(t)) for t in forest.roots)) <= s
    assert np.all((forest.assignment < len(forest.left)) & (forest.assignment >= -1))
    f = bucket_objectives(np.linspace(0.01, 0.9, len(expanded.pairs)), forest.assignment, len(forest.left))
    q, value = exact_quota_dp(forest, f)
    greedy, gv = forest_marginal_greedy(forest, f)
    assert value + 1e-12 >= gv
    for vector in (q, greedy):
        buckets = forest.to_buckets(vector)
        if len(buckets.pairs):
            assert buckets.validate()['max_entity_quota'] <= 4
        assert int(buckets.quotas.sum()) <= forest.capacity
        assert len(np.unique(pair_keys(buckets.pairs, len(c.ids)))) == len(buckets.pairs)

@pytest.mark.parametrize('seed', [101, 103, 107, 109, 113])
def test_legacy_embedding_proxy_never_worse_and_exact_legacy_baseline(seed):
    c = toy_candidates(32, seed)
    old = legacy_text_groups(c, LEGACY, seed)
    reference = construct_buckets(c.ids, c.neighbors[:, :64], c.scores[:, :64], dict(LEGACY, seed=seed), 'text_groups')
    np.testing.assert_array_equal(old.pairs, reference.pairs)
    np.testing.assert_array_equal(old.bucket, reference.bucket)
    forest, expanded = build_forest(c, 8, 32, [1, 1, 1, 1], seed)
    p = np.linspace(0.01, 0.4, len(expanded.pairs))
    f = bucket_objectives(p, forest.assignment, len(forest.left))
    quota, proxy = exact_quota_dp(forest, f)
    old_p = p[np.searchsorted(expanded.keys, pair_keys(old.pairs, len(c.ids)))]
    old_f = bucket_objectives(old_p, old.bucket, len(old.quotas))
    assert proxy + 1e-12 >= old_f[:, 1].sum()
    for b in range(len(old.quotas)):
        rows = old.bucket == b
        assignments = forest.assignment[np.searchsorted(expanded.keys, pair_keys(old.pairs[rows], len(c.ids)))]
        assert len(np.unique(assignments)) == 1
    forest.to_buckets(quota)

def test_proportional_capacity_tail_conflict_is_reported_without_rule_change():
    forest, expanded = build_forest(toy_candidates(18), 8, 16, [1, 1, 1, 1], 101)
    assert not forest.legacy_embedded
    assert forest.capacities.sum() == 12
    assert all((cap == 0 for root, cap in zip(forest.roots, forest.capacities) if len(forest.entities(root)) == 2))

def test_candidate_direction_ranks_and_duplicate_not_mutual():
    c = candidates_from_neighbors(tuple('abcd'), np.array([[1, 1, 2], [2, 0, -1], [-1, -1, -1], [3, 2, -1]]), np.ones((4, 3)), [10] * 4, 3)
    assert c.keys.tolist() == [1, 2, 6, 11]
    assert c.ranks.tolist() == [[1, 2], [3, 4], [1, 4], [4, 2]]
    assert c.features().shape == (4, 6)

def test_title_abstract_missing_fallback_and_determinism():
    ids = tuple((f'{i:03d}' for i in range(12)))
    titles = [f'common stem{i % 3} word{i}' for i in range(12)]
    config = dict(tfidf=dict(CFG['candidates']['tfidf'], min_df=1, max_df=1.0), similarity_block_size=3)
    first = generate_candidates(ids, titles, [''] * 12, 'title', 64, config, lambda _: None)
    second = generate_candidates(ids, titles, [''] * 12, 'title_abstract', 64, config, lambda _: None)
    np.testing.assert_array_equal(first.pairs, second.pairs)
    np.testing.assert_array_equal(first.ranks, second.ranks)
    np.testing.assert_allclose(first.cosine, second.cosine)

@pytest.mark.parametrize('capacity', [None, 0, 1, 10, 16])
def test_public_bmatching_degree_constraints_and_zero_replacement(capacity):
    c = toy_candidates()
    b = candidate_bmatching(c, np.linspace(0, 1, len(c.pairs)), 101, capacity=capacity)
    assert np.bincount(b.pairs.ravel(), minlength=len(c.ids)).max(initial=0) <= 4
    assert capacity is None or len(b.pairs) <= capacity
    report = stability_check(b, c.pairs, {'uniform': 24}, 101, candidate_graph=True)
    assert report['status'] == 'valid'
    assert report['statistics']['d_plus']['sampled_max'] == 0

def test_random_groups_budget_and_determinism():
    first = random_public_groups(toy_candidates().ids, 101)
    second = random_public_groups(toy_candidates().ids, 101)
    np.testing.assert_array_equal(first.pairs, second.pairs)
    assert first.validate()['max_entity_quota'] <= 4

def test_three_loss_terms_nonzero_and_fixed_index_agrees_with_legacy_truncate():
    raw = np.array([(0, 1), (0, 2), (1, 2), (3, 4), (2, 4)], np.int32)
    b = PublicBuckets(tuple('abcdef'), raw[:3], np.array([0, 0, 1], np.int32), np.array([1, 1], np.int32), 101, 4, 'toy')
    indexed = FixedTruncation(b, raw)
    np.testing.assert_array_equal(indexed.keep, b.truncate(raw))
    c = np.sort(pair_keys(raw[:4], 6))
    metrics = evaluate_structure(b, raw, c, [0.2, 0.3, 0.4], indexed)
    assert metrics['loss_counts'] == [1, 1, 1]
    assert sum((metrics[k] for k in ('loss_candidate_generation', 'loss_structure_assignment', 'loss_quota'))) == pytest.approx(1 - metrics['retention'])
    assert metrics['proxy_expected_retained'] == pytest.approx(0.44 + 0.4)
    expanded = evaluate_structure(b, raw, c, [0.2, 0.3, 0.4], indexed, raw_knn_keys=np.sort(pair_keys(raw[[0, 3]], 6)))
    assert expanded['generated_pairs_outside_knn'] == 2
    assert expanded['structure_pairs_outside_knn'] == 2

def test_stability_catches_injected_quota_and_exhaustive_neighbors():
    c = toy_candidates(16)
    forest, c = build_forest(c, 4, 16, [4], 101)
    f = bucket_objectives(np.full(len(c.pairs), 0.5), forest.assignment, len(forest.left))
    quota, _ = exact_quota_dp(forest, f)
    b = forest.to_buckets(quota)
    assert stability_check(b, c.pairs, {'uniform': 16}, 101)['status'] == 'valid'
    for u in range(16):
        before = set(map(tuple, c.pairs[b.truncate(c.pairs)]))
        adjacent = c.pairs[np.all(c.pairs != u, axis=1)]
        after = set(map(tuple, adjacent[b.truncate(adjacent)]))
        assert len(before - after) <= 4 and len(after - before) <= len(before - after)
        assert all((u in edge for edge in before - after))
    bad = copy.deepcopy(b)
    bad.quotas[0] = 5
    assert stability_check(bad, c.pairs, {'uniform': 16}, 101)['status'] == 'invalid_input'
    assert adversarial_checks()['status'] == 'valid'
    witness = adversarial_checks()['bucket_bound_attaining_2K']
    assert witness['d_minus'] == witness['d_plus'] == 4

def test_logistic_case_control_prior_correction_and_isotonic_report_only():
    c = toy_candidates(48)
    positive = c.pairs[np.arange(len(c.pairs)) % 25 == 0]
    model = fit_calibration(c, positive, CFG['calibration'], 271828)
    assert model.prior_logit_correction == pytest.approx(np.log(model.report['sampled_negatives'] / model.report['negatives']))
    probability = model.predict(c)
    assert np.all((probability > 0) & (probability < 1))
    original = probability.copy()
    model.report['isotonic_report_only']['probabilities'] = [1, 0]
    np.testing.assert_array_equal(model.predict(c), original)

@pytest.fixture
def workdir():
    path = REPO / 'tmp' / ('structure_test_' + uuid.uuid4().hex)
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path)

def fake_datasets(workdir, n=2):
    cfg = copy.deepcopy(yaml.safe_load(run.DATA_CONFIG.read_text(encoding='utf-8')))
    for name, basename in run.Inputs.NAMES.items():
        root = workdir / basename
        root.mkdir()
        hashes, splits = ({}, {})
        for d in ('CN', 'US'):
            (root / d / 'splits').mkdir(parents=True)
            ids = [f'{name}_{d}_{i:04d}' for i in range(n)]
            entities = ''.join((f'{entity}\tshared alpha word{i % 4}\n' for i, entity in enumerate(ids)))
            for relative, content in ((f'{d}/entities.tsv', entities), (f'{d}/splits/train.tsv', f'{ids[0]}\t{ids[1]}\n')):
                (root / relative).write_bytes(content.encode('utf-8'))
                hashes[relative] = run.sha256(root / relative)
            splits[d] = dict(train=1)
        run.write_json(root / 'manifest.json', dict(status='passed', domains=['CN', 'US'], sha256=hashes, splits=splits))
        cfg['datasets'][name] = dict(root=str(root), manifest_sha256=run.sha256(root / 'manifest.json'))
    return cfg

@pytest.mark.parametrize('stage', [1, 2])
@pytest.mark.parametrize('name,domain', list(itertools.product(('CS', 'MAT'), ('CN', 'US'))))
def test_stage12_private_file_tripwire_even_for_direct_open(workdir, stage, name, domain):
    cfg = fake_datasets(workdir)
    inputs = run.Inputs(cfg, stage)
    path = Path(cfg['datasets'][name]['root']) / domain / 'splits/train.tsv'
    with inputs.guard():
        with pytest.raises(PermissionError):
            inputs.rows(name, f'{domain}/splits/train.tsv')
        with pytest.raises(PermissionError):
            path.read_bytes()
        with pytest.raises(PermissionError):
            open(path, 'rb')
        allowed = 'CN' if stage == 1 else 'US'
        assert len(inputs.rows('BIO', f'{allowed}/splits/train.tsv')) == 1
        wrong = 'US' if stage == 1 else 'CN'
        with pytest.raises(PermissionError):
            inputs.rows('BIO', f'{wrong}/entities.tsv')

def test_stage3_raw_train_available_and_no_val_test_or_path_escape(workdir):
    cfg = fake_datasets(workdir)
    inputs = run.Inputs(cfg, 3)
    with inputs.guard():
        for name, domain in itertools.product(('CS', 'MAT'), ('CN', 'US')):
            public = inputs.public(name, domain)
            assert inputs.relations(name, domain, public[0]).tolist() == [[0, 1]]
        for relative in ('CN/splits/test.tsv', 'CN/splits/val.tsv', '../secret', 'CN/../CN/splits/train.tsv'):
            with pytest.raises(PermissionError):
                inputs.path('CS', relative)

def fake_lock(workdir):
    for stage in (1, 2):
        directory = workdir / f'stage{stage}'
        directory.mkdir()
        if stage == 2:
            run.write_json(directory / 'selection.json', dict(status='LOCKED', config_sha256=run.sha256(run.CONFIG), selected={}))
        run.write_json(directory / 'MANIFEST.json', dict(provenance=run.provenance(), artifacts={p.name: run.sha256(p) for p in directory.iterdir() if p.is_file()}))

def test_stage3_refuses_missing_selection_before_any_dataset_access(workdir):
    with pytest.raises(ValueError, match='selection.json'):
        run.run_stage(3, workdir, CFG, {})
    assert not (workdir / 'stage3').exists()

def test_stage3_refuses_modified_selection_and_missing_seal(workdir):
    fake_lock(workdir)
    assert run.locked_selection(workdir)['status'] == 'LOCKED'
    path = workdir / 'stage2/selection.json'
    path.write_text(path.read_text(encoding='utf-8') + ' ', encoding='utf-8')
    with pytest.raises(ValueError, match='hash changed'):
        run.run_stage(3, workdir, CFG, {})
    assert not (workdir / 'stage3').exists()
    path.unlink()
    with pytest.raises(ValueError):
        run.locked_selection(workdir)

def selection_records():
    rows = []
    for seed in CFG['construction_seeds']:
        rows.append(dict(scheme='legacy_text_groups', seed=seed, capacity_multiplier=1.0, params={}, status='valid', metrics=dict(retention=0.1, nonisolated_coverage=0.5, candidate_pairs=100)))
        for i, params in enumerate(run.design_grid(CFG)):
            rows.append(dict(scheme='dp_forest', seed=seed, capacity_multiplier=1.0, params=params, status='valid', metrics=dict(retention=0.2 + i * 1e-05, nonisolated_coverage=0.5 if i else 0.1, candidate_pairs=100 + i)))
    return rows

def test_selection_coverage_exclusion_all_seeds_ties_and_frontier_invalidity():
    records = selection_records()
    selected = run.select_design(records, CFG)
    assert len(selected['candidates']) == 48
    assert selected['selected'] == run.design_grid(CFG)[1]
    bad = copy.deepcopy(next((r for r in records if r['scheme'] == 'dp_forest' and r['params'] == selected['selected'])))
    bad.update(capacity_multiplier=0.5, status='invalid_input')
    records.append(bad)
    second = run.select_design(records, CFG)
    assert second['selected'] != selected['selected']
    rows = [r for r in records if r['seed'] != CFG['construction_seeds'][-1]]
    with pytest.raises(ValueError):
        run.select_design(rows, CFG)

def test_three_stage_synthetic_pipeline_locks_and_evaluates_only_selected(workdir):
    data_cfg = fake_datasets(workdir, n=32)
    cfg = copy.deepcopy(CFG)
    cfg['candidates'].update(text_modes=['title'], limits=[64])
    cfg['candidates']['tfidf'].update(min_df=1, max_df=1.0)
    cfg['construction_seeds'] = [101, 103]
    cfg['designs']['dp_forest'].update(base_group_size=[8], max_root_size=[16], layer_budgets=[[1, 1, 1, 1]])
    cfg['capacity']['frontier_multipliers'] = [1.0]
    cfg['stability']['deletions'] = {'uniform': 2}
    output = workdir / 'output'
    first = run.run_stage(1, output, cfg, data_cfg)
    assert first['status'] == 'COMPLETE'
    second = run.run_stage(2, output, cfg, data_cfg)
    locked_hash = run.sha256(output / 'stage2/selection.json')
    assert len(run.locked_selection(output)['candidates']) == 1
    third = run.run_stage(3, output, cfg, data_cfg)
    assert third['status'] == 'COMPLETE'
    assert run.sha256(output / 'stage2/selection.json') == locked_hash
    assert {r['domain'] for r in third['records']} == {'CS/CN', 'CS/US', 'MAT/CN', 'MAT/US'}
    assert all((r['seed'] == cfg['deployment_seed'] for r in third['records']))
    assert all((r['params'] == second['selected'] for r in third['records'] if r['scheme'] == 'dp_forest'))
