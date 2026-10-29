import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml
REPO = Path(__file__).resolve().parents[1]

def read_tsv(path):
    lines = Path(path).read_bytes().decode('utf-8').split('\n')
    if lines[-1] == '':
        lines.pop()
    rows = [line.split('\t') for line in lines]
    if any((len(r) != 2 for r in rows)):
        raise ValueError(f'Expected two-column TSV: {path}')
    return rows

def universe(cfg):
    text = {}
    for name in cfg['domains']:
        for side in ('CN', 'US'):
            base = Path(cfg['datasets_root']) / name / side
            titles = dict(read_tsv(base / 'entities.tsv'))
            abstracts = dict(read_tsv(base / 'abstracts.tsv')) if (base / 'abstracts.tsv').is_file() else {}
            for entity, title in titles.items():
                if entity in text:
                    if text[entity] != (title, abstracts.get(entity, '')):
                        raise ValueError(f'Entity {entity} has conflicting texts')
                    continue
                text[entity] = (title, abstracts.get(entity, ''))
    return text

def child(cfg, n, output):
    import numpy as np
    import resource
    sys.path.insert(0, str(REPO / 'src'))
    from quotaguard.data.structures import Calibration, candidate_bmatching, generate_candidates
    text = universe(cfg)
    ids = sorted(text)
    order = np.random.default_rng(cfg['sample_seed']).permutation(len(ids))
    chosen = sorted((ids[i] for i in order[:n]))
    titles = [text[e][0] for e in chosen]
    abstracts = [text[e][1] for e in chosen]
    del text
    model = Calibration(**json.loads(Path(cfg['models']).read_text(encoding='utf-8'))[cfg['model_key']])
    row = dict(n=len(chosen), with_abstract=sum((bool(a.strip()) for a in abstracts)))
    start = time.perf_counter()
    c = generate_candidates(chosen, titles, abstracts, cfg['text_mode'], cfg['limit'], cfg['candidates'], lambda _: None)
    row['tfidf_knn_seconds'] = time.perf_counter() - start
    t = time.perf_counter()
    p = model.predict(c)
    row['predict_seconds'] = time.perf_counter() - t
    t = time.perf_counter()
    b = candidate_bmatching(c, p, cfg['construction_seed'], cfg['kdeg'])
    row['bmatching_seconds'] = time.perf_counter() - t
    row['total_seconds'] = time.perf_counter() - start
    row['candidate_pairs'] = int(len(c.pairs))
    row['retained_pairs'] = int(len(b.pairs))
    path = Path(output) / f'graph_{len(chosen)}.npz'
    np.savez_compressed(path, pairs=b.pairs)
    row['artifact_bytes'] = path.stat().st_size
    path.unlink()
    row['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    (Path(output) / f'size_{len(chosen)}.json').write_text(json.dumps(row, indent=2) + '\n')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--child', type=int)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    if cfg.get('repeats') != 1:
        raise SystemExit('scaling requires the configured configuration')
    if args.child is not None:
        child(cfg, args.child, args.output)
        return
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit('Output directory must be empty')
    args.output.mkdir(parents=True, exist_ok=True)
    total = len(universe(cfg))
    sizes = [total if s == 'all' else int(s) for s in cfg['sizes']]
    if sorted(sizes) != sizes or sizes[-2] > total:
        raise SystemExit(f'Sizes {sizes} invalid for universe of {total}')
    env = dict(os.environ, **{k: str(cfg['threads']) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')})
    rows = []
    for n in sizes:
        subprocess.run([sys.executable, __file__, '--config', str(args.config), '--output', str(args.output), '--child', str(n)], check=True, env=env)
        rows.append(json.loads((args.output / f'size_{n}.json').read_text()))
        print(rows[-1], flush=True)
    report = dict(universe=total, threads=cfg['threads'], rows=rows, descriptive_only=True, host_note='shared container; GPU jobs running concurrently on other cores')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('scaling DONE', flush=True)
if __name__ == '__main__':
    main()
