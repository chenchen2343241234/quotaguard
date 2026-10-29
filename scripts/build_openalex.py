import argparse
import json
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))
from quotaguard.data.openalex import build_dataset

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=REPO_ROOT / 'configs/data/cs.yaml')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--cache-only', action='store_true')
    args = parser.parse_args()
    manifest = build_dataset(args.config, smoke=args.smoke, cache_only=args.cache_only)
    print(json.dumps({key: manifest[key] for key in ('status', 'mode', 'output_dir', 'elapsed_seconds', 'domains', 'checks', 'splits', 'counts', 'drops', 'full_scale_estimate', 'notes')}, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
