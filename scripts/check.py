"""Compile the package, run synthetic tests and render the numerical figures."""
from pathlib import Path
import py_compile
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    destination = root / 'tmp' / 'compiled'
    destination.mkdir(parents=True, exist_ok=True)
    files = sorted([*root.glob('scripts/*.py'), *root.glob('src/quotaguard/**/*.py'), *root.glob('tests/**/*.py')])
    for number, path in enumerate(files):
        py_compile.compile(str(path), cfile=str(destination / (str(number) + '.pyc')), doraise=True)
    print('Compiled', len(files), 'Python files', flush=True)
    subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q'], cwd=root, check=True)
    subprocess.run([sys.executable, '-B', 'scripts/make_figures.py'], cwd=root, check=True)


if __name__ == '__main__':
    main()
