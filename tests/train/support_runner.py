import importlib.util
from pathlib import Path
MODULE = Path(__file__).resolve().parents[2] / 'scripts' / 'train.py'
spec = importlib.util.spec_from_file_location('training_runner', MODULE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
