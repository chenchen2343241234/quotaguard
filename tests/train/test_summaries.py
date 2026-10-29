"""Check portable paired statistics against independent interval calculations."""
import copy
import json
from pathlib import Path

import pytest
from scipy import stats

from scripts import summarize_results as tables

ROOT = Path(__file__).resolve().parents[2]


def test_bundled_intervals_match_seed_paired_metrics():
    rows = json.loads((ROOT / 'results/per_run_metrics.json').read_text())
    expected = json.loads((ROOT / 'results/paired_comparisons.json').read_text())
    actual = tables.compare_runs(rows)
    assert actual == expected
    for row in actual:
        lo, hi = stats.t.interval(0.95, 4, loc=stats.tmean(row['differences']), scale=stats.sem(row['differences']))
        assert row['lower95'] == pytest.approx(lo, abs=1e-15)
        assert row['upper95'] == pytest.approx(hi, abs=1e-15)


def test_mismatched_delta_rejects_paired_summary():
    rows = json.loads((ROOT / 'results/per_run_metrics.json').read_text())
    rows = copy.deepcopy(rows)
    rows[0]['delta'] *= 2
    with pytest.raises(ValueError, match='privacy deltas'):
        tables.compare_runs(rows)
