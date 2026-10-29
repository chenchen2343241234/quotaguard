import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time
import tracemalloc
import numpy as np
import yaml
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from quotaguard.data.buckets import PublicBuckets, construct_buckets, title_neighbors
from quotaguard.train.data import load_domain, read_manifest_tsv

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def ratio(numerator, denominator):
    return numerator / denominator if denominator else None

@contextmanager
def stage(resources, name):
    baseline = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    start = time.perf_counter()
    print(name, flush=True)
    try:
        yield
    finally:
        peak = tracemalloc.get_traced_memory()[1]
        resources[name] = dict(elapsed_seconds=time.perf_counter() - start, traced_peak_bytes=peak, traced_peak_increment_bytes=max(0, peak - baseline))

def pair_keys(pairs, n):
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if np.any(pairs < 0) or np.any(pairs >= n):
        raise ValueError('Unknown endpoint')
    canonical = np.sort(pairs, axis=1)
    if np.any(canonical[:, 0] == canonical[:, 1]):
        raise ValueError('Self-loop')
    return canonical[:, 0] * n + canonical[:, 1]

def candidate_keys(neighbors, limit):
    n = len(neighbors)
    targets = np.asarray(neighbors[:, :limit], dtype=np.int64)
    if np.any(targets < -1) or np.any(targets >= n):
        raise ValueError('Invalid neighbor endpoint')
    sources = np.arange(n, dtype=np.int64)[:, None]
    valid = (targets >= 0) & (sources != targets)
    directed = np.unique((sources * n + targets)[valid])
    a, b = (directed // n, directed % n)
    union, counts = np.unique(np.minimum(a, b) * n + np.maximum(a, b), return_counts=True)
    return (union, union[counts == 2])

def membership(values, sorted_keys):
    if not len(sorted_keys):
        return np.zeros(len(values), dtype=bool)
    positions = np.searchsorted(sorted_keys, values)
    return (positions < len(sorted_keys)) & (sorted_keys[np.minimum(positions, len(sorted_keys) - 1)] == values)

def coverage(relations, n, candidates, degree_bins):
    keys = pair_keys(relations, n)
    if len(np.unique(keys)) != len(keys):
        raise ValueError('Duplicate raw relation')
    hits = membership(keys, candidates)
    degree = np.bincount(np.asarray(relations).ravel(), minlength=n)
    endpoint_degree = degree[np.asarray(relations).reshape(-1, 2)]

    def summarize(mask, hit_mask):
        total = int(np.count_nonzero(mask))
        covered = int(np.count_nonzero(mask & hit_mask))
        return dict(total=total, covered=covered, coverage=ratio(covered, total))
    strata = {method: {} for method in ('endpoint_incidence', 'min_endpoint', 'max_endpoint')}
    for spec in degree_bins:

        def in_bin(values):
            return (values >= spec['lower']) & (values <= spec['upper'] if spec['upper'] is not None else True)
        strata['endpoint_incidence'][spec['label']] = summarize(in_bin(endpoint_degree), np.broadcast_to(hits[:, None], endpoint_degree.shape))
        for method, values in (('min_endpoint', endpoint_degree.min(axis=1)), ('max_endpoint', endpoint_degree.max(axis=1))):
            strata[method][spec['label']] = summarize(in_bin(values), hits)
    return dict(relations=len(keys), covered_relations=int(hits.sum()), coverage=ratio(int(hits.sum()), len(keys)), candidate_pairs=len(candidates), mean_candidates_per_entity=ratio(2 * len(candidates), n), degree_strata=strata)

def loss_decomposition(relations, buckets: PublicBuckets, knn_candidates):
    public = buckets.validate()
    n = len(buckets.ids)
    raw_keys = pair_keys(relations, n)
    structure_keys = pair_keys(buckets.pairs, n)
    order = np.argsort(structure_keys)
    sorted_structure = structure_keys[order]
    in_structure = membership(raw_keys, sorted_structure)
    in_knn = membership(raw_keys, knn_candidates)
    in_candidates = in_knn | in_structure
    positions = np.searchsorted(sorted_structure, raw_keys[in_structure])
    occupancy = np.bincount(buckets.bucket[order[positions]], minlength=len(buckets.quotas))
    quota_loss = int(np.maximum(occupancy - buckets.quotas, 0).sum())
    keep_rows = buckets.truncate(relations)
    retained = np.asarray(relations)[keep_rows]
    raw_nonisolated = len(np.unique(np.asarray(relations)))
    covered_nonisolated = len(np.unique(retained))
    losses = dict(candidate_generation=int((~in_candidates).sum()), structure_assignment=int((in_candidates & ~in_structure).sum()), quota=quota_loss)
    total = len(raw_keys)
    if sum(losses.values()) != total - len(keep_rows):
        raise AssertionError('Three-stage loss partition failed')
    outside_structure = ~membership(structure_keys, knn_candidates)
    outside_raw = int((in_structure & ~in_knn).sum())
    raw_knn_losses = dict(candidate_generation=int((~in_knn).sum()), structure_assignment=int((in_knn & ~in_structure).sum()), quota=quota_loss)
    if sum(raw_knn_losses.values()) - outside_raw != total - len(keep_rows):
        raise AssertionError('Raw-kNN correction failed')
    return dict(public_structure=public, retained_relations=len(keep_rows), retention=ratio(len(keep_rows), total), raw_nonisolated_entities=raw_nonisolated, retained_nonisolated_entities=covered_nonisolated, nonisolated_entity_coverage=ratio(covered_nonisolated, raw_nonisolated), B=int(buckets.quotas.sum()), capacity_utilization=ratio(len(keep_rows), int(buckets.quotas.sum())), candidate_universe='C = C_knn(L=legacy neighbors) union C_S', candidate_pairs=len(knn_candidates) + int(outside_structure.sum()), structure_pairs_outside_knn=int(outside_structure.sum()), raw_relations_in_structure_outside_knn=outside_raw, losses={key: dict(count=value, fraction=ratio(value, total)) for key, value in losses.items()}, loss_sum=ratio(sum(losses.values()), total), identity_verified=True, raw_knn_losses={key: dict(count=value, fraction=ratio(value, total)) for key, value in raw_knn_losses.items()}, raw_knn_correction=dict(count=-outside_raw, fraction=ratio(-outside_raw, total)), raw_knn_identity='sum(raw_knn_losses) + raw_knn_correction = 1 - retention')

class Inputs:

    def __init__(self, cfg, hashes=None):
        names = dict(BIO='openalex_bio', CS='openalex_cs', MAT='openalex_mat')
        self.roots, self.manifests = ({}, {})
        self.hashes = hashes if hashes is not None else {}
        for name, basename in names.items():
            spec = cfg['datasets'][name]
            root = Path(spec['root']).resolve()
            if root.name != basename:
                raise ValueError(f'Unexpected {name} dataset root')
            self.roots[name] = root
        roots = list(self.roots.values())
        if any((a == b or a in b.parents or b in a.parents for i, a in enumerate(roots) for b in roots[i + 1:])):
            raise ValueError('Dataset roots must be separate')
        self.domains = tuple(cfg['domains'])
        if set(self.domains) != {'CN', 'US'} or len(self.domains) != 2:
            raise ValueError('Exactly CN and US are required')
        for name in names:
            path = self.path(name, 'manifest.json')
            self.record(path, cfg['datasets'][name].get('manifest_sha256'))
            manifest = json.loads(path.read_text(encoding='utf-8'))
            if manifest.get('status') != 'passed' or not all((domain in manifest['domains'] for domain in self.domains)):
                raise ValueError(f'Invalid {name} manifest')
            if name == 'BIO' and manifest.get('stable_truncation'):
                raise ValueError('BIO must contain original, untruncated train splits')
            self.manifests[name] = manifest

    def path(self, dataset, relative):
        allowed = {'manifest.json'} | {f'{d}/entities.tsv' for d in self.domains}
        if dataset == 'BIO':
            allowed |= {f'{d}/{suffix}' for d in self.domains for suffix in ('abstracts.tsv', 'splits/train.tsv')}
        if relative not in allowed:
            raise ValueError(f'Forbidden input: {dataset}/{relative}')
        root = self.roots[dataset]
        path = root / relative
        if path.resolve() != path or not path.is_relative_to(root):
            raise ValueError('Symlink/path escape is forbidden')
        return path

    def record(self, path, expected=None):
        actual = sha256(path)
        self.hashes[str(path)] = actual
        if expected is not None and actual != expected:
            raise ValueError(f'Manifest checksum mismatch: {path}')

    def rows(self, dataset, relative):
        path = self.path(dataset, relative)
        manifest = self.manifests[dataset]
        rows = read_manifest_tsv(self.roots[dataset], manifest, relative)
        self.record(path, manifest['sha256'][relative])
        return rows

    def recheck(self):
        for path, expected in self.hashes.items():
            if sha256(path) != expected:
                raise ValueError(f'Input changed during diagnosis: {path}')

def check_boundary(inputs):
    ids = {}
    for name in ('BIO', 'CS', 'MAT'):
        ids[name] = {}
        for domain in inputs.domains:
            rows = inputs.rows(name, f'{domain}/entities.tsv')
            domain_ids = tuple((row[0] for row in rows))
            if not domain_ids or len(set(domain_ids)) != len(domain_ids):
                raise ValueError(f'Empty/duplicate entities: {name}/{domain}')
            ids[name][domain] = set(domain_ids)
    intersections = {}
    for target in ('CS', 'MAT'):
        intersections[target] = {f'{bio_domain}:{target_domain}': len(ids['BIO'][bio_domain] & ids[target][target_domain]) for bio_domain in inputs.domains for target_domain in inputs.domains}
        intersections[target]['all_domains'] = len(set.union(*ids['BIO'].values()) & set.union(*ids[target].values()))
    return dict(passed=all((info['all_domains'] == 0 for info in intersections.values())), intersections=intersections, entity_counts={name: {d: len(s) for d, s in domains.items()} for name, domains in ids.items()}, implication='Disjoint entity universes imply disjoint internal relation pairs; CS/MAT relations were not opened.')

def analyze_domain(cfg, legacy, inputs, domain, resources):
    with stage(resources, f'{domain}/load_raw_train'):
        inputs.record(inputs.path('BIO', 'manifest.json'), cfg['datasets']['BIO'].get('manifest_sha256'))
        for relative in (f'{domain}/entities.tsv', f'{domain}/splits/train.tsv'):
            inputs.record(inputs.path('BIO', relative), inputs.manifests['BIO']['sha256'][relative])
        data = load_domain(inputs.roots['BIO'], domain, 'train', role='training')
        if data.ids != tuple(sorted(data.ids)):
            raise ValueError('Public entity order must be sorted by immutable ID')
        raw = data.relations.numpy()
        n = len(data.ids)
        expected = inputs.manifests['BIO']['splits'][domain]['train']
        if len(raw) != expected:
            raise ValueError('Raw train count differs from manifest')
        degree = np.bincount(raw.ravel(), minlength=n)
    result = dict(entities=n, raw_train_relations=len(raw), raw_train_sha256=inputs.manifests['BIO']['sha256'][f'{domain}/splits/train.tsv'], isolated_entities=int((degree == 0).sum()), text_modes={}, legacy={})
    for mode in cfg['text_modes']:
        with stage(resources, f'{domain}/{mode}/text'):
            texts = [str(title) for title in data.titles]
            abstract_info = {}
            if mode == 'title_abstract':
                relative = f'{domain}/abstracts.tsv'
                if relative not in inputs.manifests['BIO']['sha256']:
                    result['text_modes'][mode] = dict(status='abstracts unavailable')
                    continue
                rows = inputs.rows('BIO', relative)
                if tuple((row[0] for row in rows)) != data.ids:
                    raise ValueError('Abstract ID order differs from public universe')
                abstracts = [row[1] for row in rows]
                missing = sum((not text.strip() for text in abstracts))
                abstract_info = dict(missing=missing, missing_rate=ratio(missing, n))
                texts = [title + ' ' + abstract if abstract.strip() else title for title, abstract in zip(texts, abstracts)]
                del rows, abstracts
        with stage(resources, f'{domain}/{mode}/neighbors'):
            neighbor_cfg = dict(legacy, neighbors=max(cfg['candidate_limits']))
            neighbors, scores = title_neighbors(data.ids, texts, neighbor_cfg, progress=lambda message: print(f'{domain}/{mode}: {message}', flush=True))
        del texts
        mode_result = dict(status='complete', abstracts=abstract_info, limits={})
        for limit in cfg['candidate_limits']:
            with stage(resources, f'{domain}/{mode}/L{limit}/coverage'):
                union, mutual = candidate_keys(neighbors, limit)
                mode_result['limits'][str(limit)] = dict(union=coverage(raw, n, union, cfg['degree_bins']), mutual=coverage(raw, n, mutual, cfg['degree_bins']))
                del union, mutual
        if mode == 'title':
            with stage(resources, f'{domain}/legacy/candidates'):
                knn, legacy_mutual = candidate_keys(neighbors, legacy['neighbors'])
                del legacy_mutual
            for scheme in cfg['legacy_schemes']:
                with stage(resources, f'{domain}/legacy/{scheme}/construct'):
                    buckets = construct_buckets(data.ids, neighbors[:, :legacy['neighbors']], scores[:, :legacy['neighbors']], legacy, scheme)
                with stage(resources, f'{domain}/legacy/{scheme}/decomposition'):
                    result['legacy'][scheme] = loss_decomposition(raw, buckets, knn)
                del buckets
            del knn
        result['text_modes'][mode] = mode_result
        del neighbors, scores
    return result

def markdown_report(report):
    lines = ['# Candidate coverage on the auxiliary graph', '', f'Status: {report["status"]}', '', 'See report.json for numerical definitions and complete measurements.', '']
    if 'error' in report:
        lines += [f'Error: {report["error"]}', '']
    boundary = report.get('boundary', {})
    lines += ['## Data scope', '', '```json', json.dumps(boundary, ensure_ascii=False, indent=2), '```', '']

    def display(value):
        return '—' if value is None else f'{value:.6f}'
    for domain, info in report.get('domains', {}).items():
        lines += [f'## {domain}', '', f'Entities: {info["entities"]}; raw training relations: {info["raw_train_relations"]}; isolated entities: {info["isolated_entities"]}。', '', '| Text | L | Type | Pairs | Mean candidates | Coverage |', '|---|---:|---|---:|---:|---:|']
        for mode, data in info['text_modes'].items():
            if data['status'] != 'complete':
                lines += [f'| {mode} | — | abstracts unavailable | — | — | — |']
                continue
            for limit, variants in data['limits'].items():
                for kind, stats in variants.items():
                    lines += [f'| {mode} | {limit} | {kind} | {stats["candidate_pairs"]} | {display(stats["mean_candidates_per_entity"])} | {display(stats["coverage"])} |']
        lines += ['', '### Public group decomposition', '', 'See report.json for numerical definitions and complete measurements.', '', '| Structure | Retention | Nonisolated coverage | B | Utilization | Candidate loss | Assignment loss | Quota loss |', '|---|---:|---:|---:|---:|---:|---:|---:|']
        for scheme, data in info['legacy'].items():
            loss = data['losses']
            lines += [f'| {scheme} | {display(data["retention"])} | {display(data["nonisolated_entity_coverage"])} | {data["B"]} | {display(data["capacity_utilization"])} | {display(loss["candidate_generation"]["fraction"])} | {display(loss["structure_assignment"]["fraction"])} | {display(loss["quota"]["fraction"])} |']
        lines += ['', '### Coverage by raw degree', '', '| Text | L | Type | Stratum | Degree | Covered/total | Coverage |', '|---|---:|---|---|---|---:|---:|']
        for mode, data in info['text_modes'].items():
            for limit, variants in data.get('limits', {}).items():
                for kind, stats in variants.items():
                    for method, bins in stats['degree_strata'].items():
                        for label, counts in bins.items():
                            lines += [f'| {mode} | {limit} | {kind} | {method} | {label} | {counts["covered"]}/{counts["total"]} | {display(counts["coverage"])} |']
        lines += ['']
    lines += ['## Resource usage', '', 'See report.json for numerical definitions and complete measurements.', '', '| Stage | Seconds | Traced peak bytes | Peak increment bytes |', '|---|---:|---:|---:|']
    for name, stats in report['resources'].items():
        lines += [f'| {name} | {stats["elapsed_seconds"]:.3f} | {stats["traced_peak_bytes"]} | {stats["traced_peak_increment_bytes"]} |']
    return '\n'.join(lines) + '\n'

def run(config_path):
    cfg_path = Path(config_path).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    output = Path(cfg['output_dir'].format(date=datetime.now().strftime(cfg['date_format'])))
    if not output.is_absolute():
        output = REPO / output
    output = output.resolve()
    if not output.is_relative_to(REPO / 'runs'):
        raise ValueError('Reports must stay inside this worktree/results')
    if any(((output / filename).exists() for filename in ('report.json', 'report.md', 'MANIFEST.json'))):
        raise ValueError('Existing diagnosis output must not be overwritten')
    output.mkdir(parents=True, exist_ok=True)
    tracemalloc.start()
    started = time.perf_counter()
    report = dict(status='running', started_at_utc=datetime.now(timezone.utc).isoformat(), config=cfg, domains={}, resources={})
    hashes = {str(cfg_path): sha256(cfg_path)}
    inputs = None
    try:
        with stage(report['resources'], 'configuration'):
            if type(cfg['seed']) is not int:
                raise ValueError('Fixed integer seed required')
            if cfg['candidate_limits'] != [16, 64, 256]:
                raise ValueError('structure candidate limits must be [16, 64, 256]')
            if cfg['text_modes'] != ['title', 'title_abstract']:
                raise ValueError('Both text modes are required')
            if cfg['legacy_schemes'] != ['text_groups', 'singleton_knn']:
                raise ValueError('Both legacy schemes are required')
            legacy_path = (REPO / cfg['legacy_config']).resolve()
            legacy_source = yaml.safe_load(legacy_path.read_text(encoding='utf-8'))
            hashes[str(legacy_path)] = sha256(legacy_path)
            keys = ('seed', 'kdeg', 'neighbors', 'group_size', 'rounds', 'star_candidates', 'tfidf', 'similarity_block_size')
            legacy = {key: legacy_source[key] for key in keys}
            if legacy['seed'] != cfg['seed'] or legacy['kdeg'] != 4 or legacy['group_size'] != 8 or (legacy['rounds'] != 4) or (legacy['neighbors'] != 64):
                raise ValueError('Unexpected legacy design parameters')
            report['legacy_public_config'] = legacy
            random.seed(cfg['seed'])
            np.random.seed(cfg['seed'])
            from threadpoolctl import threadpool_limits
            if type(cfg['cpu_threads']) is not int or cfg['cpu_threads'] < 1:
                raise ValueError('cpu_threads must be positive')
            for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
                os.environ[key] = str(cfg['cpu_threads'])
            import torch
            torch.set_num_threads(cfg['cpu_threads'])
            torch.manual_seed(cfg['seed'])
            report['versions'] = {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'scikit-learn', 'torch', 'PyYAML')}
            for path in (Path(__file__), REPO / 'src/quotaguard/data/buckets.py', REPO / 'src/quotaguard/train/data.py'):
                hashes[str(path.resolve())] = sha256(path)
        with stage(report['resources'], 'data_boundary'):
            inputs = Inputs(cfg, hashes=hashes)
            report['boundary'] = check_boundary(inputs)
        if not report['boundary']['passed']:
            report['status'] = 'aborted_overlap'
        else:
            with threadpool_limits(limits=cfg['cpu_threads']):
                for domain in cfg['domains']:
                    report['domains'][domain] = analyze_domain(cfg, legacy, inputs, domain, report['resources'])
            report['status'] = 'complete'
        with stage(report['resources'], 'input_recheck'):
            inputs.recheck()
            for path, expected in hashes.items():
                if sha256(path) != expected:
                    raise ValueError(f'Input changed during diagnosis: {path}')
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        report['elapsed_seconds'] = time.perf_counter() - started
        report['traced_peak_bytes'] = max((r['traced_peak_bytes'] for r in report['resources'].values()), default=0)
        tracemalloc.stop()
        manifest = dict(sha256=hashes, status=report['status'], policy='Only listed inputs were hashed. CS/MAT entities + manifest only; BIO entities, optional abstracts, and raw splits/train.tsv only.')
        (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        (output / 'report.md').write_text(markdown_report(report), encoding='utf-8')
        (output / 'MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    report = run(parser.parse_args().config)
    print(report['status'], flush=True)
    return 0 if report['status'] == 'complete' else 2
if __name__ == '__main__':
    sys.exit(main())
