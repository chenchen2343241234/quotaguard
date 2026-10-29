import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import yaml
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)

def stage3_record(report, domain, scheme, params, field='CS'):
    rows = [r for r in report['records'] if r['domain'] == f'{field}/{domain}' and r['scheme'] == scheme and (r['capacity_multiplier'] == 1.0) and ((r.get('params') or {}) == params)]
    if len(rows) != 1 or rows[0]['status'] != 'valid' or rows[0]['stability']['status'] != 'valid':
        raise ValueError(f'Expected exactly one VALID stage-3 record for {domain}/{scheme}')
    return rows[0]

def build(cfg, kind):
    import torch
    from quotaguard.data.buckets import PublicBuckets
    from quotaguard.train.data import clip_degree, load_domain
    spec = cfg['structures'][kind]
    stage3, source, output = (Path(cfg['structure_dir']), Path(cfg['source_dir']), Path(spec['output_dir']))
    if cfg.get('structure_report_sha256') and sha256(stage3 / 'report.json') != cfg['structure_report_sha256']:
        raise ValueError('structure stage-3 report changed')
    if cfg.get('structure_manifest_sha256') and sha256(stage3 / 'MANIFEST.json') != cfg['structure_manifest_sha256']:
        raise ValueError('structure stage-3 manifest changed')
    sealed = json.loads((stage3 / 'MANIFEST.json').read_text())['artifacts']
    if cfg.get('source_manifest_sha256') and sha256(source / 'manifest.json') != cfg['source_manifest_sha256']:
        raise ValueError('Source manifest mismatch')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output dataset directory must be empty')
    report = json.loads((stage3 / 'report.json').read_text())
    parent = json.loads((source / 'manifest.json').read_text())
    report_path = stage3 / 'report.json'
    public = output / 'public'
    manifest = dict(id=f'quotaguard-training-{kind}', status='building', domains={}, splits={}, sha256={}, source_manifest_sha256=sha256(source / 'manifest.json'), source_dir=str(source), output_dir=str(output), config=cfg, structure_report_sha256=sha256(stage3 / 'report.json'), stable_truncation=dict(version=1, scheme=spec['scheme'], kdeg=cfg['kdeg'], seed=cfg['deployment_seed'], domains={}))
    stats = {}
    public.mkdir(parents=True)
    for domain in cfg['domains']:
        record = stage3_record(report, domain, spec['scheme'], spec.get('params', {}), cfg.get('field', 'CS'))
        src_artifact = stage3 / record['artifact']
        if sealed.get(record['artifact']) != sha256(src_artifact):
            raise ValueError('Structure artifact differs from the structure stage-3 seal')
        buckets = PublicBuckets.load(src_artifact)
        summary = buckets.validate()
        if buckets.kdeg != cfg['kdeg'] or buckets.seed != cfg['deployment_seed']:
            raise ValueError('Unexpected structure parameters')
        previous = manifest['stable_truncation'].setdefault('scheme_label', buckets.scheme)
        if previous != buckets.scheme:
            raise ValueError('Domains use different structure schemes')
        manifest['stable_truncation']['scheme'] = buckets.scheme
        if kind == 'cg':
            sizes = np.bincount(buckets.bucket, minlength=len(buckets.quotas))
            if not (np.all(sizes == 1) and np.all(buckets.quotas == 1)):
                raise ValueError('Candidate graph must consist of singleton pools')
        artifact = public / f'{domain}_{kind}.npz'
        shutil.copyfile(src_artifact, artifact)
        data = load_domain(source, domain, 'train', role='training')
        if data.ids != buckets.ids:
            raise ValueError('Public universe/order differs from source entities')
        raw = data.relations.numpy()
        kept = raw[buckets.truncate(raw)]
        if len(kept) != record['metrics']['actual_retained']:
            raise ValueError('Retained count differs from structure stage-3 record')
        if not torch.equal(clip_degree(torch.from_numpy(kept), len(data.ids), cfg['kdeg'], seed=cfg['clip_seed']), torch.from_numpy(kept)):
            raise ValueError('Runner clipping is not identity on the truncated graph')
        for relative, expected in parent['sha256'].items():
            if not relative.startswith(domain + '/') or relative == f'{domain}/splits/train.tsv':
                continue
            dest = output / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, dest)
            if sha256(dest) != expected:
                raise ValueError('Copied source checksum mismatch')
            manifest['sha256'][relative] = expected
        relative = f'{domain}/splits/train.tsv'
        with (output / relative).open('w', encoding='utf-8', newline='\n') as handle:
            for a, b in kept:
                handle.write(f'{data.ids[a]}\t{data.ids[b]}\n')
        manifest['sha256'][relative] = sha256(output / relative)
        manifest['domains'][domain] = dict(entities=len(data.ids), raw_train=len(raw), retained_train=len(kept))
        manifest['splits'][domain] = dict(parent['splits'][domain], train=len(kept))
        manifest['stable_truncation']['domains'][domain] = dict(summary, artifact=str(artifact.relative_to(output)), artifact_sha256=sha256(artifact), public_train_size=summary['quota_sum'], public_entities=len(data.ids), N_min=len(data.ids) - 3, entities_sha256=parent['sha256'][f'{domain}/entities.tsv'], public_expected_size=float(record['metrics']['proxy_expected_retained']), public_expected_derivation='sum_b E_Q[min(beta_b, N_b)] under the BIO-CN calibrated public proxy; structure stage-3 metric proxy_expected_retained', structure_record=dict(artifact=record['artifact'], scheme=record['scheme'], params=record.get('params')))
        stats[domain] = dict(raw=len(raw), retained=len(kept), retention=len(kept) / len(raw), capacity=summary['quota_sum'], public_expected=float(record['metrics']['proxy_expected_retained']))
        print(f'{kind}/{domain}: retained {len(kept)} of {len(raw)}', flush=True)
    manifest.update(status='passed', retention=stats, checks=dict(public_structure_verified=True, degree_at_most_K=True, runner_clip_identity=True, val_test_byte_identical=True, retained_matches_structure=True, no_target_evaluation=True))
    write_json(output / 'manifest.json', manifest)
    readback(cfg, output, manifest)
    write_json(Path(cfg['report_dir']) / f'manifest_{kind}.json', manifest)
    return stats

def readback(cfg, output, manifest):
    from quotaguard.train.data import load_domain
    from quotaguard.train.stable import public_bounds
    for domain in cfg['domains']:
        data = load_domain(output, domain, 'train', role='training')
        info = manifest['stable_truncation']['domains'][domain]
        if len(data.relations) != manifest['domains'][domain]['retained_train']:
            raise ValueError('Read-back retained count mismatch')
        probe = dict(stable_truncation=True, noise_calibration='mog', kdeg=cfg['kdeg'], data_dir=str(output), source=domain)
        bounds, _ = public_bounds(probe, data)
        if bounds['public_train_size'] != info['quota_sum']:
            raise ValueError('Read-back capacity mismatch')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--kind', choices=('cg', 'lg', 'all'), default='all')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    for kind in ('cg', 'lg') if args.kind == 'all' else (args.kind,):
        build(cfg, kind)
if __name__ == '__main__':
    main()
