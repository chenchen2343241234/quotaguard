"""Offline checks of portable data configuration and authentication."""
import json
from pathlib import Path

import pytest

from quotaguard.data.openalex import OpenAlexClient, load_config, validate_config, minimum_relations

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('field', ['cs', 'mat', 'bio'])
def test_data_configuration_preserves_split_protocol(field):
    cfg = load_config(ROOT / 'configs' / 'data' / (field + '.yaml'))
    validate_config(cfg)
    assert cfg['seed'] == 17
    assert cfg['splits']['split_version'] == 2
    assert minimum_relations(cfg) == 50000


def test_environment_authentication_does_not_enter_report(tmp_path, monkeypatch):
    cfg = load_config(ROOT / 'configs/data/cs.yaml')
    monkeypatch.delenv('OPENALEX_API_KEY', raising=False)
    anonymous = OpenAlexClient(cfg, tmp_path, cache_only=True)
    assert anonymous.auth['mode'] == 'anonymous'
    monkeypatch.setenv('OPENALEX_API_KEY', 'synthetic-test-value')
    authenticated = OpenAlexClient(cfg, tmp_path, cache_only=True)
    assert authenticated.auth['mode'] == 'api_key'
    assert 'synthetic-test-value' not in json.dumps(authenticated.retrieval_report([]))
    with pytest.raises(FileNotFoundError, match='cached page'):
        authenticated.page({'filter': 'publication_year:2017'})
    assert authenticated.network_attempts == 0
