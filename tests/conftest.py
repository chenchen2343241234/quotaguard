"""Test import paths and optional training dependencies."""
from pathlib import Path
import shutil
import sys
import uuid
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'tests/train')]


@pytest.fixture
def tmp_path():
    """Keep temporary synthetic test inputs inside the repository."""
    directory = ROOT / 'tmp' / 'tests' / uuid.uuid4().hex
    directory.mkdir(parents=True)
    try:
        yield directory
    finally:
        shutil.rmtree(directory)
