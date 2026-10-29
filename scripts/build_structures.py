import argparse
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import sys
import threading
import time
import numpy as np
import psutil
import yaml
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from quotaguard.data.structures import Calibration, Candidates, FixedTruncation, adversarial_checks, bucket_objectives, build_forest, candidate_bmatching, candidates_from_neighbors, evaluate_structure, exact_quota_dp, fit_calibration, forest_marginal_greedy, generate_candidates, legacy_text_groups, membership, pair_keys, random_public_groups, stability_check, validate_buckets, kernel
CONFIG = REPO / 'configs/structures.yaml'
DATA_CONFIG = REPO / 'configs/candidate_coverage.yaml'
_ACCESS = ContextVar('structure_file_access', default=None)

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def _audit(event, args):
    guard = _ACCESS.get()
    if event != 'open' or guard is None or (not isinstance(args[0], (str, bytes, os.PathLike))):
        return
    lexical = Path(os.path.abspath(os.fsdecode(args[0])))
    path = lexical.resolve()
    roots, allowed = guard
    inside = any((p == root or p.is_relative_to(root) for p in (lexical, path) for root in roots))
    if inside and (path not in allowed or lexical != path):
        raise PermissionError(f'Forbidden structure stage input: {path}')
sys.addaudithook(_audit)

class Inputs:
    NAMES = dict(BIO='openalex_bio', CS='openalex_cs', MAT='openalex_mat')

    def __init__(self, data_config, stage):
        if stage not in (1, 2, 3):
            raise ValueError('Unknown stage')
        self.stage, self.config = (stage, data_config)
        self.roots = {name: Path(data_config['datasets'][name]['root']).resolve() for name in self.NAMES}
        if any((root.name != self.NAMES[name] for name, root in self.roots.items())):
            raise ValueError('Unexpected dataset basename')
        roots = list(self.roots.values())
        if any((a == b or a in b.parents or b in a.parents for i, a in enumerate(roots) for b in roots[i + 1:])):
            raise ValueError('Dataset roots must be separate')
        self.allowed, self.hashes, self.manifests = (set(), {}, {})
        roles = [('BIO', 'CN' if stage == 1 else 'US')] if stage < 3 else list(itertools.product(('CS', 'MAT'), ('CN', 'US')))
        for name, domain in roles:
            self.allowed.add(self.roots[name] / 'manifest.json')
            for suffix in ('entities.tsv', 'abstracts.tsv', 'splits/train.tsv'):
                self.allowed.add(self.roots[name] / domain / suffix)

    @contextmanager
    def guard(self):
        token = _ACCESS.set((tuple(self.roots.values()), self.allowed))
        try:
            yield self
        finally:
            _ACCESS.reset(token)

    def path(self, name, relative):
        if name not in self.roots:
            raise PermissionError('Unknown dataset')
        path = self.roots[name] / relative
        if path.resolve() != path or path not in self.allowed:
            raise PermissionError(f'Forbidden structure stage input: {name}/{relative}')
        return path

    def manifest(self, name):
        if name not in self.manifests:
            path = self.path(name, 'manifest.json')
            actual = sha256(path)
            if self.config['datasets'][name].get('manifest_sha256') and actual != self.config['datasets'][name]['manifest_sha256']:
                raise ValueError('Dataset manifest hash mismatch')
            data = json.loads(path.read_text(encoding='utf-8'))
            if data.get('status') != 'passed' or data.get('stable_truncation'):
                raise ValueError('structure requires passed, raw untruncated dataset')
            self.hashes[str(path)], self.manifests[name] = (actual, data)
        return self.manifests[name]

    def rows(self, name, relative):
        path, manifest = (self.path(name, relative), self.manifest(name))
        payload = path.read_bytes()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != manifest['sha256'][relative]:
            raise ValueError(f'Dataset checksum mismatch: {name}/{relative}')
        self.hashes[str(path)] = actual
        lines = payload.decode('utf-8').split('\n')
        if lines[-1] == '':
            lines.pop()
        rows = [line.split('\t') for line in lines]
        if any((len(row) != 2 for row in rows)):
            raise ValueError('Expected literal two-column TSV')
        return rows

    def public(self, name, domain):
        rows = self.rows(name, f'{domain}/entities.tsv')
        ids, titles = zip(*rows)
        if ids != tuple(sorted(ids)) or len(set(ids)) != len(ids):
            raise ValueError('Invalid public entity universe')
        relative = f'{domain}/abstracts.tsv'
        abstracts = [''] * len(ids)
        if relative in self.manifest(name)['sha256']:
            rows = self.rows(name, relative)
            if tuple((row[0] for row in rows)) != ids:
                raise ValueError('Abstract IDs differ from fixed universe')
            abstracts = [row[1] for row in rows]
        return (ids, titles, abstracts)

    def relations(self, name, domain, ids):
        rows = self.rows(name, f'{domain}/splits/train.tsv')
        index = {entity: i for i, entity in enumerate(ids)}
        try:
            pairs = np.array([(index[a], index[b]) for a, b in rows], np.int32).reshape(-1, 2)
        except KeyError as error:
            raise ValueError('Unknown raw endpoint') from error
        keys = pair_keys(pairs, len(ids))
        if len(np.unique(keys)) != len(keys):
            raise ValueError('Duplicate raw relation')
        if len(pairs) != self.manifest(name)['splits'][domain]['train']:
            raise ValueError('Raw train count differs from manifest')
        return pairs

    def recheck(self):
        for path, expected in self.hashes.items():
            if sha256(path) != expected:
                raise ValueError('Input changed during stage')

class Resources:

    def __init__(self):
        self.stage_seconds, self.peak = ({}, 0)

    @contextmanager
    def measure(self, name):
        stop, process = (threading.Event(), psutil.Process())
        start, cpu = (time.perf_counter(), time.process_time())

        def sample():
            while True:
                info = process.memory_info()
                self.peak = max(self.peak, info.rss, getattr(info, 'peak_wset', 0))
                if stop.wait(0.01):
                    break
        thread = threading.Thread(target=sample, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join()
            self.stage_seconds[name] = dict(wall_seconds=time.perf_counter() - start, cpu_seconds=time.process_time() - cpu)

def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False, default=lambda v: v.item() if isinstance(v, np.generic) else list(v))
        stream.write('\n')

def provenance():
    paths = [CONFIG, DATA_CONFIG, Path(__file__), REPO / 'src/quotaguard/data/structures.py', REPO / 'src/quotaguard/data/structures_kernel.cpp', REPO / 'src/quotaguard/data/buckets.py', REPO / 'configs/public_groups.yaml']
    return {str(path.relative_to(REPO)): sha256(path) for path in paths}

def verify_manifest(directory):
    manifest = json.loads((directory / 'MANIFEST.json').read_text(encoding='utf-8'))
    for relative, expected in manifest['artifacts'].items():
        path = directory / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f'Sealed artifact hash changed: {path}')
    if manifest['provenance'] != provenance():
        raise ValueError('Configured config/source provenance changed')
    return manifest

def locked_selection(output):
    path = Path(output) / 'stage2' / 'selection.json'
    if not path.is_file():
        raise ValueError('Stage 3 requires locked selection.json')
    manifest = verify_manifest(path.parent)
    if 'selection.json' not in manifest['artifacts']:
        raise ValueError('selection.json has no stage 2 seal')
    verify_manifest(Path(output) / 'stage1')
    previous = manifest.get('inputs', {}).get('stage1/MANIFEST.json')
    if previous is not None and previous != sha256(Path(output) / 'stage1/MANIFEST.json'):
        raise ValueError('Stage 1 provenance seal changed after selection')
    selection = json.loads(path.read_text(encoding='utf-8'))
    if selection['config_sha256'] != sha256(CONFIG) or selection['status'] != 'LOCKED':
        raise ValueError('Invalid selection lock')
    return selection

def finish(directory, report, inputs, resources, extra_inputs=None):
    report['stage_seconds'] = resources.stage_seconds
    report['peak_rss_bytes'] = resources.peak
    report['artifact_bytes_before_report'] = sum((p.stat().st_size for p in directory.rglob('*') if p.is_file()))
    write_json(directory / 'report.json', report)
    lines = ['# Public structure report', '', f'Stage: {report["stage"]}; status: {report["status"]}', '', 'See report.json for numerical definitions and complete measurements.', '', '```json', json.dumps({k: v for k, v in report.items() if k not in ('records', 'models')}, ensure_ascii=False, indent=2, default=lambda v: v.item() if isinstance(v, np.generic) else list(v)), '```', '']
    with (directory / 'report.md').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('\n'.join(lines))
    if inputs:
        inputs.recheck()
    artifacts = {str(p.relative_to(directory)): sha256(p) for p in directory.rglob('*') if p.is_file()}
    write_json(directory / 'MANIFEST.json', dict(provenance=provenance(), inputs=dict(inputs.hashes if inputs else {}, **extra_inputs or {}), artifacts=artifacts, artifact_bytes=sum(((directory / p).stat().st_size for p in artifacts)), created_local=datetime.now().isoformat(), python=sys.version, platform=platform.platform()))

def design_grid(cfg):
    forest = cfg['designs']['dp_forest']
    return [dict(text_mode=mode, limit=limit, base_group_size=g, max_root_size=s, layer_budgets=budgets) for mode, limit, g, s, budgets in itertools.product(cfg['candidates']['text_modes'], cfg['candidates']['limits'], forest['base_group_size'], forest['max_root_size'], forest['layer_budgets'])]

def design_id(params):
    return json.dumps(params, sort_keys=True, separators=(',', ':'))

def select_design(records, cfg):
    expected = cfg['construction_seeds']
    legacy = [r for r in records if r['scheme'] == 'legacy_text_groups' and r['capacity_multiplier'] == 1.0]
    if sorted((r['seed'] for r in legacy)) != sorted(expected):
        raise ValueError('Missing legacy seeds')
    legacy_coverage = np.mean([r['metrics']['nonisolated_coverage'] for r in legacy])
    summaries = []
    for params in design_grid(cfg):
        rows = [r for r in records if r['scheme'] == 'dp_forest' and r['params'] == params and (r['capacity_multiplier'] == 1.0)]
        valid = sorted((r['seed'] for r in rows)) == sorted(expected) and all((r['status'] == 'valid' for r in rows)) and all((r['status'] == 'valid' for r in records if r['params'] == params))
        metric = lambda key: float(np.mean([r['metrics'][key] for r in rows])) if valid else None
        coverage = metric('nonisolated_coverage')
        eligible = valid and coverage >= 0.95 * legacy_coverage
        summaries.append(dict(params=params, status='valid' if valid else 'invalid_input', eligible=eligible, mean_retention=metric('retention'), mean_nonisolated_coverage=coverage, mean_candidate_pairs=metric('candidate_pairs'), records=rows))
    eligible = [s for s in summaries if s['eligible']]
    if not eligible:
        raise ValueError('No valid design meets configured coverage constraint')
    best = max((s['mean_retention'] for s in eligible))
    tied = [s for s in eligible if best - s['mean_retention'] <= cfg['selection']['tie_tolerance_pp'] / 100]
    selected = min(tied, key=lambda s: (-s['mean_nonisolated_coverage'], s['mean_candidate_pairs'], s['params']['max_root_size'], design_id(s['params'])))
    return dict(status='LOCKED', config_sha256=sha256(CONFIG), source='BIO-US raw train only', selected=selected['params'], candidates=summaries, legacy_mean_nonisolated_coverage=float(legacy_coverage), best_mean_retention=best, tied_design_count=len(tied), construction_seeds=expected, deployment_seed=cfg['deployment_seed'])

def read_models(output):
    verify_manifest(output / 'stage1')
    raw = json.loads((output / 'stage1/models.json').read_text(encoding='utf-8'))
    return {key: Calibration(**value) for key, value in raw.items()}

def candidate_variants(cfg, public, resources):
    ids, titles, abstracts = public
    variants = {}
    for mode in cfg['candidates']['text_modes']:
        with resources.measure(f'candidates/{mode}/top{max(cfg["candidates"]["limits"])}'):
            full = generate_candidates(ids, titles, abstracts, mode, max(cfg['candidates']['limits']), cfg['candidates'])
        for limit in cfg['candidates']['limits']:
            variants[f'{mode}:{limit}'] = full if limit == full.limit else candidates_from_neighbors(ids, full.neighbors, full.scores, full.lengths, limit, full.matrix)
    return variants

def resized_forest(forest, capacity):
    weights = np.array([len(forest.entities(t)) * int(k) for t, k in zip(forest.roots, forest.budgets)], np.int64)
    denominator = int(weights.sum())
    caps = capacity * weights // denominator
    remaining = capacity - int(caps.sum())
    order = np.lexsort((forest.roots, -(capacity * weights % denominator)))
    caps[order[:remaining]] += 1
    return replace(forest, capacities=caps.astype(np.int32), capacity=int(capacity))

def save_buckets(buckets, path):
    validate_buckets(buckets)
    np.savez_compressed(path, ids=np.array(buckets.ids), pairs=buckets.pairs, bucket=buckets.bucket, quotas=buckets.quotas, seed=buckets.seed, kdeg=buckets.kdeg, scheme=buckets.scheme)

def record_structure(buckets, candidates, probabilities, relations, cfg, params, seed, scheme, multiplier, directory, resources, domain, existing_artifact=None):
    tag = f'{domain}/{scheme}/{hashlib.sha256(design_id(params).encode()).hexdigest()[:10]}/{seed}/{multiplier}'
    with resources.measure(tag + '/private_metrics'):
        keys, bkeys = (candidates.keys, pair_keys(buckets.pairs, len(buckets.ids)))
        order = np.argsort(keys)
        positions = np.searchsorted(keys[order], bkeys)
        if not np.all(membership(bkeys, keys[order])):
            raise ValueError('Proxy population must include all structure pairs')
        bp = probabilities[order[positions]]
        metrics = evaluate_structure(buckets, relations, candidates.keys, bp, raw_knn_keys=getattr(candidates, 'raw_knn_keys', None))
    stability = None
    with resources.measure(tag + '/stability'):
        if seed == cfg['deployment_seed']:
            stability = stability_check(buckets, relations, cfg['stability']['deletions'], seed, candidate_graph=scheme.startswith('candidate_graph'))
            stability['adversarial'] = adversarial_checks(seed, cfg['kdeg'])
    status = 'valid' if stability is None or (stability['status'] == 'valid' and stability['adversarial']['status'] == 'valid') else 'invalid_input'
    filename = existing_artifact or hashlib.sha256(tag.encode()).hexdigest()[:20] + '.npz'
    if existing_artifact is None:
        with resources.measure(tag + '/serialize'):
            save_buckets(buckets, directory / filename)
    metrics['artifact_bytes'] = (directory / filename).stat().st_size
    metrics['peak_rss_bytes'] = resources.peak
    metrics['stage_seconds'] = {k: v for k, v in resources.stage_seconds.items() if k.startswith(tag)}
    return dict(domain=domain, scheme=scheme, params=params, seed=seed, capacity_multiplier=multiplier, status=status, metrics=metrics, stability=stability, artifact=filename)

def construct_domain(cfg, variants, models, params_list, seeds, directory, resources, domain):
    queue = []
    title = variants['title:64']
    legacy_cfg = yaml.safe_load((REPO / 'configs/public_groups.yaml').read_text(encoding='utf-8'))
    b0 = 4 * ((len(title.ids) + 7) // 8)
    predicted, base_proxy = ({}, {})

    def freeze_proxy(c, p, token, raw_knn_keys=None):
        path = 'proxy_' + hashlib.sha256((domain + '/' + token).encode()).hexdigest()[:20] + '.npz'
        if not (directory / path).exists():
            mask = np.ones(len(c.pairs), dtype=bool) if raw_knn_keys is None else membership(c.keys, raw_knn_keys)
            np.savez_compressed(directory / path, pairs=c.pairs, probabilities=p, knn_mask=mask)
        return path

    def enqueue(b, params, seed, scheme, multiplier, proxy, construction=None):
        stem = f'public_{domain.replace("/", "_")}_{len(queue):05d}.npz'
        save_buckets(b, directory / stem)
        queue.append(dict(bucket_file=stem, proxy_file=proxy, params=params, seed=seed, scheme=scheme, multiplier=multiplier, construction=construction))
    for key, c in variants.items():
        with resources.measure(f'{domain}/predict/{key}'):
            predicted[key] = models[key].predict(c)
            base_proxy[key] = freeze_proxy(c, predicted[key], key)
    for seed in seeds:
        for scheme, buckets in (('legacy_text_groups', legacy_text_groups(title, legacy_cfg, seed)), ('random_public_groups', random_public_groups(title.ids, seed))):
            with resources.measure(f'{domain}/{scheme}/{seed}/proxy_expansion'):
                c = title.extend(buckets.pairs)
                p = models['title:64'].predict(c)
            proxy = freeze_proxy(c, p, f'{scheme}/{seed}', title.keys)
            enqueue(buckets, {}, seed, scheme, 1.0, proxy)
        for key, c in variants.items():
            with resources.measure(f'{domain}/bmatching/{key}/{seed}/construct'):
                p = predicted[key]
                for cap in (None, b0):
                    b = candidate_bmatching(c, p, seed, cfg['kdeg'], cap)
                    enqueue(b, dict(text_mode=key.split(':')[0], limit=c.limit), seed, 'candidate_graph_bmatching_native' if cap is None else 'candidate_graph_bmatching_B0', 1.0, base_proxy[key])
        for params in params_list:
            key = f'{params["text_mode"]}:{params["limit"]}'
            c = variants[key]
            c.legacy_neighbors = title.neighbors
            tag = f'{domain}/forest/{hashlib.sha256(design_id(params).encode()).hexdigest()[:10]}/{seed}'
            with resources.measure(tag + '/construct'):
                forest, expanded = build_forest(c, params['base_group_size'], params['max_root_size'], params['layer_budgets'], seed, b0)
            with resources.measure(tag + '/poisson_binomial'):
                p = predicted[key] if expanded is c else models[key].predict(expanded)
                f = bucket_objectives(p, forest.assignment, len(forest.left), cfg['kdeg'])
                proxy_file = base_proxy[key] if expanded is c else freeze_proxy(expanded, p, f'{key}/legacy/{seed}', c.keys)
            for multiplier in cfg['capacity']['frontier_multipliers']:
                current = resized_forest(forest, int(np.floor(b0 * multiplier)))
                for scheme, solver in (('dp_forest', exact_quota_dp), ('forest_marginal_greedy', forest_marginal_greedy)):
                    with resources.measure(tag + f'/{multiplier}/{scheme}'):
                        q, proxy = solver(current, f)
                        buckets = current.to_buckets(q, scheme)
                    if scheme == 'dp_forest':
                        dp_proxy = proxy
                    elif proxy > dp_proxy + 1e-09:
                        raise AssertionError('Greedy proxy exceeds exact DP')
                    enqueue(buckets, params, seed, scheme, multiplier, proxy_file, dict(nodes=len(forest.left), roots=len(forest.roots), assigned_pairs=int((forest.assignment >= 0).sum()), legacy_quota_vector_feasible_at_main_capacity=forest.legacy_embedded, root_capacity_rule='proportional largest remainder by |S_root| * layer_budget'))
    write_json(directory / f'public_queue_{domain.replace("/", "_")}.json', queue)
    return queue

def evaluate_queue(queue, ids, relations, cfg, directory, resources, domain):
    from quotaguard.data.buckets import PublicBuckets
    records = []
    for item in queue:
        with np.load(directory / item['bucket_file'], allow_pickle=False) as z:
            b = PublicBuckets(ids, z['pairs'], z['bucket'], z['quotas'], int(z['seed']), int(z['kdeg']), str(z['scheme']))
        with np.load(directory / item['proxy_file'], allow_pickle=False) as z:
            c = Candidates(ids, np.empty((0, 0), np.int32), np.empty((0, 0), np.float32), z['pairs'], np.empty(0), np.empty((0, 2), np.int32), np.empty(0), 64)
            p = z['probabilities']
            c.raw_knn_keys = c.keys[z['knn_mask']]
        record = record_structure(b, c, p, relations, cfg, item['params'], item['seed'], item['scheme'], item['multiplier'], directory, resources, domain, item['bucket_file'])
        record['construction'] = item.get('construction')
        records.append(record)
    return records

def gate_report(records, cfg):
    domains = sorted({r['domain'] for r in records})
    deltas = []
    invalid_input = any((r['status'] != 'valid' for r in records))
    for domain in domains:
        rows = [r for r in records if r['domain'] == domain and r['capacity_multiplier'] == 1.0]
        dp = np.mean([r['metrics']['retention'] for r in rows if r['scheme'] == 'dp_forest'])
        legacy = np.mean([r['metrics']['retention'] for r in rows if r['scheme'] == 'legacy_text_groups'])
        groups = {(r['scheme'], design_id(r['params'])) for r in rows if r['scheme'] not in ('dp_forest', 'candidate_graph_bmatching_native')}
        simple = max((np.mean([r['metrics']['retention'] for r in rows if (r['scheme'], design_id(r['params'])) == group]) for group in groups))
        cover = np.mean([r['metrics']['nonisolated_coverage'] for r in rows if r['scheme'] == 'dp_forest'])
        lc = np.mean([r['metrics']['nonisolated_coverage'] for r in rows if r['scheme'] == 'legacy_text_groups'])
        deltas.append(dict(domain=domain, vs_legacy_pp=100 * (dp - legacy), vs_best_simple_pp=100 * (dp - simple), coverage_constraint=bool(cover >= 0.95 * lc)))
    mean_legacy = float(np.mean([d['vs_legacy_pp'] for d in deltas]))
    mean_simple = float(np.mean([d['vs_best_simple_pp'] for d in deltas]))
    gates = cfg['gate_to_training']
    passed = mean_legacy >= gates['vs_legacy_pp'] and mean_simple >= gates['vs_best_simple_baseline_pp'] and all((d['vs_legacy_pp'] >= 0 and d['vs_best_simple_pp'] >= 0 and d['coverage_constraint'] for d in deltas))
    state = 'invalid_input' if invalid_input else 'supported' if passed else 'not_supported' if mean_simple <= 0 else 'undetermined'
    return dict(state=state, domains=deltas, macro_vs_legacy_pp=mean_legacy, macro_vs_simple_pp=mean_simple)

def run_stage(stage, output, cfg, data_cfg):
    resources = Resources()
    selection = locked_selection(output) if stage == 3 else None
    models = read_models(output) if stage > 1 else None
    directory = output / f'stage{stage}'
    directory.mkdir(parents=True, exist_ok=False)
    inputs, report = (Inputs(data_cfg, stage), dict(stage=stage, status='RUNNING'))
    extra = {}
    try:
        with inputs.guard(), resources.measure('total'):
            if stage == 1:
                with resources.measure('load_public_and_BIO_CN_raw'):
                    public = inputs.public('BIO', 'CN')
                    relations = inputs.relations('BIO', 'CN', public[0])
                variants = candidate_variants(cfg, public, resources)
                models = {}
                for key, c in variants.items():
                    with resources.measure('calibration/' + key):
                        models[key] = fit_calibration(c, relations, cfg['calibration'], cfg['seed'])
                write_json(directory / 'models.json', {k: asdict(v) for k, v in models.items()})
                write_json(directory / 'BIO_CN_ids.json', public[0])
                report['models'] = {k: v.report for k, v in models.items()}
            elif stage == 2:
                extra['stage1/MANIFEST.json'] = sha256(output / 'stage1/MANIFEST.json')
                with resources.measure('load_BIO_US_public'):
                    public = inputs.public('BIO', 'US')
                    cn_ids = json.loads((output / 'stage1/BIO_CN_ids.json').read_text(encoding='utf-8'))
                    if set(cn_ids) & set(public[0]):
                        raise ValueError('BIO-CN calibration and BIO-US selection entities overlap')
                variants = candidate_variants(cfg, public, resources)
                queue = construct_domain(cfg, variants, models, design_grid(cfg), cfg['construction_seeds'], directory, resources, 'BIO/US')
                del variants
                relations = inputs.relations('BIO', 'US', public[0])
                report['records'] = evaluate_queue(queue, public[0], relations, cfg, directory, resources, 'BIO/US')
                selection = select_design(report['records'], cfg)
                write_json(directory / 'selection.json', selection)
                write_json(directory / 'BIO_US_ids.json', public[0])
                report['selected'] = selection['selected']
                selected_rows = [r for r in report['records'] if r['scheme'] not in ('dp_forest', 'forest_marginal_greedy') or r['params'] == selection['selected']]
                report['gate'] = gate_report(selected_rows, cfg)
            else:
                extra['stage2/selection.json'] = sha256(output / 'stage2/selection.json')
                bio_ids = set(json.loads((output / 'stage1/BIO_CN_ids.json').read_text(encoding='utf-8')))
                bio_ids.update(json.loads((output / 'stage2/BIO_US_ids.json').read_text(encoding='utf-8')))
                report['records'], report['selection_sha256'] = ([], extra['stage2/selection.json'])
                for name, domain in itertools.product(('CS', 'MAT'), ('CN', 'US')):
                    with resources.measure(f'{name}/{domain}/public'):
                        public = inputs.public(name, domain)
                        if bio_ids & set(public[0]):
                            raise ValueError('Protected entities overlap calibration/selection universe')
                    variants = candidate_variants(cfg, public, resources)
                    queue = construct_domain(cfg, variants, models, [selection['selected']], [cfg['deployment_seed']], directory, resources, f'{name}/{domain}')
                    del variants
                    relations = inputs.relations(name, domain, public[0])
                    report['records'].extend(evaluate_queue(queue, public[0], relations, cfg, directory, resources, f'{name}/{domain}'))
                report['gate'] = gate_report(report['records'], cfg)
                locked_selection(output)
            report['status'] = 'COMPLETE'
    except Exception as error:
        report.update(status='invalid_input', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        finish(directory, report, inputs, resources, extra)
    return report

def synthetic_candidates(cfg, n, m):
    rng = np.random.default_rng(cfg['seed'])
    ids = tuple((f'entity-{i:09d}' for i in range(n)))
    a = rng.integers(0, n, m, dtype=np.int32)
    b = rng.integers(0, n, m, dtype=np.int32)
    half = m // 2
    b[:half] = a[:half] // 32 * 32 + rng.integers(0, 32, half, dtype=np.int32)
    b %= n
    valid = a != b
    keys = np.unique(np.minimum(a[valid], b[valid]).astype(np.int64) * n + np.maximum(a[valid], b[valid]))
    pairs = np.column_stack((keys // n, keys % n)).astype(np.int32)
    cosine = rng.random(len(keys), dtype=np.float32)
    neighbors = np.empty((n, 64), np.int32)
    for j in range(64):
        neighbors[:, j] = (np.arange(n) // 32 * 32 + (np.arange(n) + j + 1) % 32) % n
    c = Candidates(ids, neighbors, np.zeros_like(neighbors, np.float32), pairs, cosine, np.full((len(keys), 2), 65, np.int32), np.full(n, 100, np.int32), 64)
    return (c, rng.uniform(0.001, 0.5, len(keys)))

def preflight(output, cfg, n=200000, m=10000000):
    directory = output / 'preflight'
    directory.mkdir(parents=True, exist_ok=False)
    resources = Resources()
    with resources.measure('compile_kernel'):
        kernel()
    with resources.measure('synthetic_generation'):
        c, probabilities = synthetic_candidates(cfg, n, m)
    observations = []
    for g, s, budgets in itertools.product((4, 8), (16, 32), ([4], [2, 2], [1, 1, 1, 1])):
        tag = f'g{g}_s{s}_k' + ''.join(map(str, budgets))
        with resources.measure(tag + '/forest'):
            forest, expanded = build_forest(c, g, s, budgets, cfg['deployment_seed'], preserve_legacy=False)
        with resources.measure(tag + '/poisson_binomial'):
            f = bucket_objectives(probabilities, forest.assignment, len(forest.left), cfg['kdeg'])
        with resources.measure(tag + '/dp'):
            q, value = exact_quota_dp(forest, f)
        observations.append(dict(base_group_size=g, max_root_size=s, layer_budgets=budgets, assigned_pairs=int((forest.assignment >= 0).sum()), nodes=len(forest.left), positive_quota_nodes=int((q > 0).sum()), proxy_expected_retained=value, forest_seconds=resources.stage_seconds[tag + '/forest'], poisson_binomial_seconds=resources.stage_seconds[tag + '/poisson_binomial'], dp_seconds=resources.stage_seconds[tag + '/dp']))
        print(json.dumps(observations[-1]), flush=True)
        del forest, expanded, f, q
    with resources.measure('bmatching_baseline'):
        candidate_bmatching(c, probabilities, cfg['deployment_seed'], cfg['kdeg'], 4 * ((n + 7) // 8))
    total_cpu = sum((sum((row[name]['cpu_seconds'] for name in ('forest_seconds', 'poisson_binomial_seconds', 'dp_seconds'))) for row in observations))
    baseline_cpu = resources.stage_seconds['bmatching_baseline']['cpu_seconds']
    estimate = total_cpu * 4 * len(cfg['construction_seeds']) + baseline_cpu * 8 * len(cfg['construction_seeds'])
    report = dict(stage='preflight', status='COMPLETE', n=n, raw_candidate_draws=m, unique_candidate_pairs=len(c.pairs), observations=observations, peak_rss_bytes=resources.peak, estimated_stage2_main_dp_and_bmatching_cpu_hours=estimate / 3600, estimate_scope='48 designs x 5 seeds main forest+PB+DP, 8 bmatching baselines x 5; synthetic geometry only; excludes TF-IDF, calibration prediction, marginal greedy, capacity-frontier re-solves, private metrics/stability and serialization', full_stage2_cpu_hours='not bounded by this synthetic preflight; see measured components and exclusions', cpu_threads=cfg['cpu_threads_per_process'])
    finish(directory, report, None, resources)
    return report

def profile_overheads(output, cfg, n=200000, m=10000000):
    directory = output / 'preflight_overheads'
    directory.mkdir(parents=True, exist_ok=False)
    core_path = output / 'preflight/report.json'
    core = json.loads(core_path.read_text(encoding='utf-8'))
    if (core['n'], core['raw_candidate_draws']) != (n, m):
        raise ValueError('Core/overhead synthetic sizes differ')
    resources = Resources()
    with resources.measure('generation'):
        c, probabilities = synthetic_candidates(cfg, n, m)
        rng = np.random.default_rng(cfg['seed'] + 1)
        raw = c.pairs[rng.random(len(c.pairs)) < 0.1]
    print(f'synthetic overheads: {n} entities, {len(c.pairs)} candidates, {len(raw)} raw relations', flush=True)
    with resources.measure('proxy_prediction'):
        model = Calibration([1.0, -0.1, -0.1, 0.2, 0.01, 0.01], -3.0, [0.0] * 6, [1.0] * 6, 0.0, {})
        model.predict(c)
    with resources.measure('representative_forest'):
        forest, _ = build_forest(c, 4, 32, [4], cfg['deployment_seed'], preserve_legacy=False)
        f = bucket_objectives(probabilities, forest.assignment, len(forest.left))
    dp_seconds, greedy_seconds = ([], [])
    for multiplier in cfg['capacity']['frontier_multipliers']:
        current = resized_forest(forest, int(forest.capacity * multiplier))
        tag = str(multiplier)
        with resources.measure('frontier_dp/' + tag):
            q, _ = exact_quota_dp(current, f)
        dp_seconds.append(resources.stage_seconds['frontier_dp/' + tag]['cpu_seconds'])
        with resources.measure('frontier_greedy/' + tag):
            gq, _ = forest_marginal_greedy(current, f)
        greedy_seconds.append(resources.stage_seconds['frontier_greedy/' + tag]['cpu_seconds'])
    q, _ = exact_quota_dp(forest, f)
    with resources.measure('validate_and_materialize_buckets'):
        b = forest.to_buckets(q)
    bp = probabilities[np.searchsorted(c.keys, pair_keys(b.pairs, n))]
    with resources.measure('private_metrics'):
        metrics = evaluate_structure(b, raw, c.keys, bp)
    with resources.measure('stratified_stability'):
        stability = stability_check(b, raw, cfg['stability']['deletions'], cfg['deployment_seed'])
    with resources.measure('serialize_bucket'):
        save_buckets(b, directory / 'synthetic_bucket.npz')
    with resources.measure('serialize_proxy'):
        np.savez_compressed(directory / 'synthetic_proxy.npz', pairs=c.pairs, probabilities=probabilities)
    with resources.measure('load_proxy_and_bucket'):
        with np.load(directory / 'synthetic_proxy.npz') as z:
            z['pairs']
            z['probabilities']
        with np.load(directory / 'synthetic_bucket.npz') as z:
            z['pairs']
            z['bucket']
            z['quotas']
    with resources.measure('legacy_baseline_construct'):
        legacy_text_groups(c, yaml.safe_load((REPO / 'configs/public_groups.yaml').read_text(encoding='utf-8')), cfg['deployment_seed'])
    with resources.measure('random_baseline_construct'):
        random_public_groups(c.ids, cfg['deployment_seed'])
    print('synthetic overheads: numeric/metrics/I/O measured; measuring public TF-IDF', flush=True)
    with resources.measure('synthetic_text_generation'):
        tokens = rng.integers(0, n, (n, 20), dtype=np.int32)
        titles = [' '.join((f'token{int(v)}' for v in row[:8])) for row in tokens]
        abstracts = [' '.join((f'token{int(v)}' for v in row[8:])) for row in tokens]
    candidate_counts = {}
    for mode in cfg['candidates']['text_modes']:
        with resources.measure('tfidf_neighbors/' + mode):
            cc = generate_candidates(c.ids, titles, abstracts, mode, 256, cfg['candidates'], lambda _: None)
        candidate_counts[mode] = len(cc.pairs)
        del cc
        print('synthetic TF-IDF complete: ' + mode, flush=True)
    cpu = lambda name: resources.stage_seconds[name]['cpu_seconds']
    designs, seeds, frontiers = (len(design_grid(cfg)), len(cfg['construction_seeds']), len(cfg['capacity']['frontier_multipliers']))
    structure_count = designs * seeds * frontiers * 2 + 10 * seeds
    stability_count = designs * frontiers * 2 + 10
    core_cpu = sum((sum((row[name]['cpu_seconds'] for name in ('forest_seconds', 'poisson_binomial_seconds', 'dp_seconds'))) for row in core['observations'])) * 4 * seeds
    components = dict(core_forest_pb_main_dp=core_cpu, extra_frontier_dp=designs * seeds * (sum(dp_seconds) - dp_seconds[cfg['capacity']['frontier_multipliers'].index(1.0)]), all_forest_greedy=designs * seeds * sum(greedy_seconds), bmatching_baselines=core['stage_seconds']['bmatching_baseline']['cpu_seconds'] * 8 * seeds, group_baselines=(cpu('legacy_baseline_construct') + cpu('random_baseline_construct')) * seeds, public_validation=cpu('validate_and_materialize_buckets') * structure_count, private_metrics=cpu('private_metrics') * structure_count, stability=cpu('stratified_stability') * stability_count, bucket_serialization=cpu('serialize_bucket') * structure_count, proxy_serialization=cpu('serialize_proxy') * (4 + 6 * seeds), repeated_proxy_loading=cpu('load_proxy_and_bucket') * structure_count, proxy_prediction=cpu('proxy_prediction') * (4 + 10 * seeds), tfidf_and_neighbors=sum((cpu('tfidf_neighbors/' + mode) for mode in cfg['candidates']['text_modes'])))
    estimate = sum(components.values()) / 3600
    report = dict(stage='preflight_overheads', status='COMPLETE', n=n, raw_candidate_draws=m, unique_candidate_pairs=len(c.pairs), synthetic_raw_relations=len(raw), synthetic_text_candidate_pairs=candidate_counts, full_stage2_estimated_cpu_hours=estimate, planning_range_cpu_hours=[estimate, 2 * estimate], estimated_component_cpu_seconds=components, structures_including_frontiers=structure_count, deployment_seed_stability_structures=stability_count, representative_metrics=metrics, representative_stability=stability, caveat='Synthetic extrapolation, not a bound. Main core covers 12 forest families; overheads use g4/s32/[4]. TF-IDF timing uses 8-token titles + 12-token abstracts with uniform token IDs. Real posting density, raw relation counts, title/abstract lengths, disk speed and compatibility expansion change costs. Planning range is 1x-2x measured extrapolation, not a confidence interval. No real grid was run.')
    finish(directory, report, None, resources, {str(core_path): sha256(core_path)})
    return report

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=int, choices=(1, 2, 3))
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--profile-overheads', action='store_true')
    parser.add_argument('--output', type=Path, default=REPO / 'runs/structures')
    parser.add_argument('--synthetic-n', type=int, default=200000)
    parser.add_argument('--synthetic-m', type=int, default=10000000)
    args = parser.parse_args(argv)
    if sum((bool(args.stage), args.preflight, args.profile_overheads)) != 1:
        parser.error('Choose exactly one of --stage, --preflight or --profile-overheads')
    output = args.output.resolve()
    if not output.is_relative_to(REPO / 'runs'):
        parser.error('Output must stay inside this worktree results/')
    cfg = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    data_cfg = yaml.safe_load(DATA_CONFIG.read_text(encoding='utf-8'))
    if len(design_grid(cfg)) != 48:
        raise ValueError('Unexpected configured design grid')
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=cfg['cpu_threads_per_process']):
        if args.preflight or args.profile_overheads:
            method = preflight if args.preflight else profile_overheads
            report = method(output, cfg, args.synthetic_n, args.synthetic_m)
        else:
            report = run_stage(args.stage, output, cfg, data_cfg)
    print(json.dumps({key: report[key] for key in ('stage', 'status')}, ensure_ascii=False))
if __name__ == '__main__':
    main()
