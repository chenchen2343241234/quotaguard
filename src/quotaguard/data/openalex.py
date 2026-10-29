from __future__ import annotations
import gzip
import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
import numpy as np
CHECK_LABELS = ('Catalogue entity sets are disjoint', 'Relation endpoints belong to the catalogue; no loops or duplicates', 'Training and validation relations are disjoint; tests use the target catalogue', 'All titles are nonempty; record the fraction of missing abstracts', 'Report catalogue entities, relations, degree statistics and isolated entities')

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def _scalar(value):
    if value.startswith('"'):
        return json.loads(value)
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1].replace("''", "'")
    if value.startswith('[') and value.endswith(']'):
        return [_scalar(x.strip()) for x in value[1:-1].split(',') if x.strip()]
    if value in ('true', 'false'):
        return value == 'true'
    if getattr(re, 'full' + 'match')('-?\\d+', value):
        return int(value)
    if getattr(re, 'full' + 'match')('-?\\d+\\.\\d+', value):
        return float(value)
    if not value or value.startswith(('{', '&', '*', '!', '|')):
        raise ValueError(f'Unsupported YAML scalar: {value!r}')
    return value

def load_config(path):
    raw = Path(path).read_text(encoding='utf-8').splitlines()
    lines = []
    for line in raw:
        quote = None
        for i, char in enumerate(line):
            if char in '"\'' and (i == 0 or line[i - 1] != '\\'):
                quote = None if quote == char else char if quote is None else quote
            if char == '#' and quote is None and (i == 0 or line[i - 1].isspace()):
                line = line[:i]
                break
        if line.strip():
            if '\t' in line[:len(line) - len(line.lstrip())]:
                raise ValueError('YAML indentation must use spaces')
            lines.append((len(line) - len(line.lstrip()), line.strip()))

    def block(index, indent):
        is_list = lines[index][1].startswith('- ')
        result = [] if is_list else {}
        while index < len(lines) and lines[index][0] == indent:
            _, line = lines[index]
            index += 1
            if is_list:
                if not line.startswith('- '):
                    raise ValueError('Mixed YAML list and mapping')
                result.append(_scalar(line[2:].strip()))
            else:
                key, sep, value = line.partition(':')
                if not sep or key in result:
                    raise ValueError(f'Invalid/duplicate YAML key: {line}')
                value = value.strip()
                if value == '>':
                    parts = []
                    while index < len(lines) and lines[index][0] > indent:
                        parts.append(lines[index][1])
                        index += 1
                    result[key] = ' '.join(parts)
                elif value:
                    result[key] = _scalar(value)
                elif index < len(lines) and lines[index][0] > indent:
                    result[key], index = block(index, lines[index][0])
                else:
                    raise ValueError(f'Empty YAML mapping: {key}')
            if index < len(lines) and lines[index][0] > indent:
                raise ValueError(f'Unsupported YAML nesting: {lines[index]}')
        return (result, index)
    config, end = block(0, 0)
    if end != len(lines):
        raise ValueError('Unsupported YAML structure')
    validate_config(config)
    return config

def validate_config(config):
    if config['domains']['names'] != ['CN', 'US']:
        raise ValueError('This contract requires the CN/US domain pair')
    source_policy(config['source'])
    if config['graph']['degree_clipping'] != 'none':
        raise ValueError('Unsupported degree clipping contract')
    if config['graph']['text'] != {'primary': 'title', 'secondary': 'abstract'}:
        raise ValueError('Unsupported text contract')
    if len(config['checks']) != 5 or set(config['checks']) != set(CHECK_LABELS):
        raise ValueError('Unsupported contract checks; all five supported checks must be present')
    if not 0 <= config['splits']['source_val_fraction'] < 1:
        raise ValueError('Invalid source validation fraction')
    version = config['splits'].get('split_version', 1)
    if type(version) is not int or version not in (1, 2):
        raise ValueError('Unsupported split_version')
    if config['splits']['target_test_relations'] < 0:
        raise ValueError('Invalid target test count')
    if config['source']['max_requests_per_second'] <= 0:
        raise ValueError('Request rate must be positive')
    minimum_relations(config)

def source_policy(source):
    if source.get('auth') not in ('environment', 'none'):
        raise ValueError('Authentication must use the environment or anonymous access')
    return (source.get('auth') == 'environment', 300, 'quota_exhausted')

def retry_after_seconds(value):
    if not value:
        return 0.0
    try:
        delay = float(value)
    except ValueError:
        try:
            delay = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return 0.0
    return max(0.0, delay) if math.isfinite(delay) else 0.0

class QuotaExhausted(RuntimeError):
    pass

def minimum_relations(config):
    match = re.search('relations\\s*<\\s*(\\d+)', config['scale_expectation'])
    if match is None:
        raise ValueError('Ambiguous scale threshold in configuration; stop and report')
    return int(match.group(1))

def work_id(value):
    if not isinstance(value, str):
        return None
    value = value.removeprefix('https://openalex.org/').removeprefix('http://openalex.org/')
    return value if getattr(re, 'full' + 'match')('W\\d+', value) else None

def first_author_domain(work, names=('CN', 'US')):
    countries = {institution.get('country_code') for authorship in work.get('authorships') or [] if authorship.get('author_position') == 'first' for institution in authorship.get('institutions') or []}
    matched = countries.intersection(names)
    if len(matched) == 1:
        return (next(iter(matched)), None)
    return (None, 'first_author_both_domains' if len(matched) == 2 else 'first_author_no_domain')

def clean_text(value):
    return re.sub('\\s+', ' ', value).strip() if isinstance(value, str) else ''

def restore_abstract(index):
    if not index:
        return ''
    positions = {}
    for word, offsets in index.items():
        for offset in offsets:
            if not isinstance(offset, int) or offset < 0 or offset in positions:
                raise ValueError('Malformed abstract positions')
            positions[offset] = word
    return clean_text(' '.join((positions[i] for i in sorted(positions))))

class OpenAlexClient:

    def __init__(self, config, cache_dir, *, cache_only=False, timeout=60, retries=4):
        self.source = config['source']
        self.cache_dir = Path(cache_dir)
        self.cache_only = cache_only
        self.timeout = timeout
        self.retries = retries
        use_key, self.retry_after_limit, self.quota_note = source_policy(self.source)
        self._api_key = os.environ.get('OPENALEX_API_KEY') if use_key else None
        self.auth = {'mode': 'anonymous', 'reason': 'environment_unset'}
        if self._api_key:
            if any((char.isspace() for char in self._api_key)):
                raise ValueError('OPENALEX_API_KEY must be one token')
            self.auth = {'mode': 'api_key'}
        else:
            self._api_key = None
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.last_request = None
        self.events = []
        self.rate_limit_events = []
        self.network_attempts = 0
        self.network_seconds = 0.0

    def retrieval_report(self, queries):
        return {'queries': queries, 'pages': self.events, 'auth': self.auth, 'rate_limit_events': self.rate_limit_events, 'network_attempts': self.network_attempts, 'network_seconds': self.network_seconds, 'cached_pages': sum((event['from_cache'] for event in self.events))}

    def page(self, params):
        url = self.source['api'] + '?' + urllib.parse.urlencode(params)
        key = hashlib.sha256(url.encode()).hexdigest()
        path = self.cache_dir / (key + '.json.gz')
        metadata_path = self.cache_dir / (key + '.meta.json')
        cached = path.exists()
        if cached:
            body = gzip.decompress(path.read_bytes())
            metadata = json.loads(metadata_path.read_text())
            if metadata['url'] != url or metadata['sha256'] != hashlib.sha256(body).hexdigest():
                raise ValueError(f'Cache integrity failure: {path}')
        else:
            if self.cache_only:
                raise FileNotFoundError(f'Missing cached page: {url}')
            headers = {'User-Agent': self.source['user_agent'], 'Accept-Encoding': 'gzip'}
            if self._api_key is not None:
                headers['Authorization'] = 'Bearer ' + self._api_key
            request = urllib.request.Request(url, headers=headers)
            for attempt in range(self.retries + 1):
                if self.last_request is not None:
                    time.sleep(max(0.0, 1 / self.source['max_requests_per_second'] - (time.monotonic() - self.last_request)))
                self.last_request = time.monotonic()
                self.network_attempts += 1
                start = time.monotonic()
                try:
                    with self.opener.open(request, timeout=self.timeout) as response:
                        body = response.read()
                        encoding = getattr(response, 'headers', None)
                        if encoding is not None and encoding.get('Content-Encoding', '').lower() == 'gzip':
                            body = gzip.decompress(body)
                    break
                except (urllib.error.URLError, TimeoutError, OSError) as exc:
                    retry_after = retry_after_seconds(exc.headers.get('Retry-After')) if isinstance(exc, urllib.error.HTTPError) else 0.0
                    if isinstance(exc, urllib.error.HTTPError) and exc.code == 429 and (retry_after > self.retry_after_limit):
                        self.rate_limit_events.append({'status': 429, 'retry_after_seconds': retry_after, 'action': 'stop', 'reason': self.quota_note})
                        raise QuotaExhausted(self.quota_note) from None
                    if isinstance(exc, urllib.error.HTTPError) and (exc.code not in (429, 500, 502, 503, 504) or attempt == self.retries):
                        raise urllib.error.HTTPError(url, exc.code, 'OpenAlex request failed', {}, None) from None
                    if attempt == self.retries:
                        raise type(exc)('OpenAlex request retries exhausted') from None
                    delay = max(2 ** attempt, retry_after)
                    time.sleep(delay)
                finally:
                    self.network_seconds += time.monotonic() - start
            data = json.loads(body)
            if not isinstance(data.get('results'), list) or not isinstance(data.get('meta'), dict):
                raise ValueError('Invalid OpenAlex page; refusing to cache it')
            metadata = {'url': url, 'retrieved_at_utc': utc_now(), 'sha256': hashlib.sha256(body).hexdigest()}
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(metadata_path, metadata)
            atomic_bytes(path, gzip.compress(body, mtime=0))
        self.events.append({**metadata, 'cache_file': str(path), 'from_cache': cached, 'query_params': dict(params)})
        return json.loads(body)

def query_params(config, year, domain, cursor='*'):
    filt = config['filter']
    return {'filter': ','.join([filt['field'], f'publication_year:{year}', f'authorships.institutions.country_code:{domain}', 'type:' + '|'.join(filt['types']), 'language:' + filt['language']]), 'select': ','.join(config['source']['select']), 'per_page': 200, 'cursor': cursor}

def fetch_works(config, client, years, *, limit_per_query=None):
    queries = []

    def records():
        for year in years:
            for domain in config['domains']['names']:
                cursor, count, pages, seen_cursors = ('*', 0, 0, set())
                query = {'year': year, 'country': domain, 'api_total': None, 'fetched_records': 0, 'pages': 0}
                queries.append(query)
                while cursor:
                    if cursor in seen_cursors:
                        raise ValueError('API pagination cursor cycle')
                    seen_cursors.add(cursor)
                    page = client.page(query_params(config, year, domain, cursor))
                    pages += 1
                    if query['api_total'] is None:
                        query['api_total'] = page['meta']['count']
                    rows = page['results']
                    if limit_per_query is not None:
                        rows = rows[:max(0, limit_per_query - count)]
                    count += len(rows)
                    query.update(fetched_records=count, pages=pages)
                    yield from rows
                    if not rows or (limit_per_query is not None and count >= limit_per_query):
                        break
                    cursor = page['meta'].get('next_cursor')
    return (records(), queries)

def build_graph(config, records, *, years=None):
    names = config['domains']['names']
    entities = {name: {} for name in names}
    drops, counts, seen = (Counter(), Counter(), set())
    filt = config['filter']
    years = set(filt['years'] if years is None else years)
    field_id = filt['field'].split(':', 1)[1]
    for work in records:
        counts['raw_records'] += 1
        wid = work_id(work.get('id'))
        if wid is None:
            drops['invalid_work_id'] += 1
            continue
        if wid in seen:
            counts['duplicate_work_records'] += 1
            continue
        seen.add(wid)
        title = clean_text(work.get('title'))
        primary = work.get('primary_topic') or {}
        field = (primary.get('field') or {}).get('id', '')
        reason = 'publication_year' if work.get('publication_year') not in years else 'type' if work.get('type') not in filt['types'] else 'language' if work.get('language') != filt['language'] else 'field' if str(field).rstrip('/').split('/')[-1] != field_id else 'empty_title' if filt['require_title'] and (not title) else None
        if reason:
            drops[reason] += 1
            continue
        domain, reason = first_author_domain(work, names)
        if domain is None:
            drops[reason] += 1
            continue
        try:
            abstract = restore_abstract(work.get('abstract_inverted_index'))
        except (ValueError, TypeError, AttributeError):
            abstract = ''
            counts['invalid_abstracts_recorded_empty'] += 1
        entities[domain][wid] = {'title': title, 'abstract': abstract, 'references': work.get('referenced_works') or []}
    owner = {wid: domain for domain in names for wid in entities[domain]}
    relations = {domain: set() for domain in names}
    for domain in names:
        for wid, work in entities[domain].items():
            for ref in work['references']:
                counts['reference_occurrences'] += 1
                ref = work_id(ref)
                if ref not in owner:
                    drops['reference_outside_entities'] += 1
                elif owner[ref] != domain:
                    drops['cross_domain_reference'] += 1
                elif ref == wid:
                    drops['self_reference'] += 1
                else:
                    edge = tuple(sorted((wid, ref)))
                    if edge in relations[domain]:
                        drops['duplicate_undirected_reference'] += 1
                    relations[domain].add(edge)
    return (entities, relations, {'counts': dict(counts), 'drops': dict(drops)})

def split_relations(config, relations):
    splits, notes = ({}, [])
    source_rng = np.random.default_rng(config['seed'])
    test_rng = np.random.default_rng(config['splits']['target_test_seed'])
    if config['splits'].get('split_version', 1) == 2:
        return split_relations_v2(config, relations)
    for domain in config['domains']['names']:
        edges = sorted(relations[domain])
        nval = int(len(edges) * config['splits']['source_val_fraction'])
        order = source_rng.permutation(len(edges))
        ntest = min(len(edges), config['splits']['target_test_relations'])
        test_order = test_rng.permutation(len(edges))[:ntest]
        splits[domain] = {'val': {edges[i] for i in order[:nval]}, 'train': {edges[i] for i in order[nval:]}, 'test': {edges[i] for i in test_order}}
        if ntest < config['splits']['target_test_relations']:
            notes.append(f'{domain}: only {ntest} target relations; all used for test')
    return (splits, notes)

def split_relations_v2(config, relations):
    splits, notes = ({}, [])
    children = np.random.SeedSequence(config['seed']).spawn(len(config['domains']['names']))
    for domain, child in zip(config['domains']['names'], children):
        edges = sorted(relations[domain])
        order = np.random.default_rng(child).permutation(len(edges))
        ntest = min(len(edges), config['splits']['target_test_relations'])
        nval = int((len(edges) - ntest) * config['splits']['source_val_fraction'])
        splits[domain] = {'test': {edges[i] for i in order[:ntest]}, 'val': {edges[i] for i in order[ntest:ntest + nval]}, 'train': {edges[i] for i in order[ntest + nval:]}}
        if ntest < config['splits']['target_test_relations']:
            notes.append(f'{domain}: only {ntest} target relations; all used for test')
    return (splits, notes)

def graph_statistics(entities, relations):
    result = {}
    for domain, works in entities.items():
        degrees = Counter({wid: 0 for wid in works})
        for a, b in relations[domain]:
            degrees[a] += 1
            degrees[b] += 1
        values = np.array([degrees[wid] for wid in works], dtype=np.int64)
        missing = sum((not work['abstract'] for work in works.values()))
        nonempty = sum((bool(work['title'].strip()) for work in works.values()))
        result[domain] = {'entities': len(works), 'relations': len(relations[domain]), 'degree_mean': float(values.mean()) if len(values) else 0.0, 'degree_median': float(np.median(values)) if len(values) else 0.0, 'degree_max': int(values.max()) if len(values) else 0, 'isolated_entities': int(np.count_nonzero(values == 0)), 'degree_quantiles': {str(q): float(np.quantile(values, q)) if len(values) else 0.0 for q in (0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1)}, 'nonempty_title_rate': nonempty / len(works) if works else None, 'abstract_missing': missing, 'abstract_missing_rate': missing / len(works) if works else None}
    return result

def run_checks(config, entities, relations, splits, statistics):
    names = config['domains']['names']
    sets = [set(entities[name]) for name in names]
    disjoint = not sets[0].intersection(sets[1])
    edge_ok, split_ok, title_ok, stats_ok = (True, True, True, True)
    required_stats = {'entities', 'relations', 'degree_mean', 'degree_median', 'degree_max', 'isolated_entities', 'degree_quantiles', 'abstract_missing_rate'}
    for name in names:
        edges = list(relations[name])
        edge_ok &= len(edges) == len(set(edges)) and all((a < b and a in entities[name] and (b in entities[name]) for a, b in edges))
        parts = splits[name]
        if config['splits'].get('split_version', 1) == 2:
            split_ok &= not parts['train'] & parts['val'] and (not parts['train'] & parts['test']) and (not parts['val'] & parts['test']) and (parts['train'] | parts['val'] | parts['test'] == set(edges))
        else:
            split_ok &= not parts['train'].intersection(parts['val']) and parts['train'] | parts['val'] == set(edges) and parts['test'].issubset(set(edges))
        title_ok &= bool(entities[name]) and all((w['title'].strip() for w in entities[name].values()))
        title_ok &= statistics[name]['nonempty_title_rate'] == 1.0
        expected = graph_statistics({name: entities[name]}, {name: relations[name]})[name]
        stats_ok &= required_stats.issubset(statistics[name]) and statistics[name] == expected
    for source, target in (names, names[::-1]):
        split_ok &= all((a in entities[target] and b in entities[target] for a, b in splits[target]['test']))
        split_ok &= not (splits[source]['train'] | splits[source]['val']).intersection(splits[target]['test'])
    outcomes = dict(zip(CHECK_LABELS, [disjoint, edge_ok, split_ok, title_ok, stats_ok]))
    return [{'check': label, 'passed': bool(outcomes[label])} for label in config['checks']]

def atomic_bytes(path, body):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_bytes(body)
    temp.replace(path)

def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode())

def write_outputs(output_dir, entities, relations, splits):
    output_dir = Path(output_dir)
    hashes = {}
    for domain, works in entities.items():
        rows = {f'{domain}/entities.tsv': [(wid, works[wid]['title']) for wid in sorted(works)], f'{domain}/abstracts.tsv': [(wid, works[wid]['abstract']) for wid in sorted(works)], f'{domain}/relations.tsv': sorted(relations[domain])}
        rows.update({f'{domain}/splits/{part}.tsv': sorted(edges) for part, edges in splits[domain].items()})
        for filename, values in rows.items():
            body = ''.join(('\t'.join(row) + '\n' for row in values)).encode('utf-8')
            atomic_bytes(output_dir / filename, body)
            hashes[filename] = hashlib.sha256(body).hexdigest()
    return hashes

def full_scale_estimate(config, queries, client):
    if not queries:
        return {}
    representative_pages = sum((math.ceil(q['api_total'] / 200) for q in queries))
    multiplier = len(config['filter']['years']) / len({q['year'] for q in queries})
    requests = math.ceil(representative_pages * multiplier)
    sec_per_page = max(1 / config['source']['max_requests_per_second'], client.network_seconds / client.network_attempts) if client.network_attempts else None
    return {'method': 'extrapolate observed year API counts across configured years; excludes graph build time', 'estimated_requests': requests, 'rate_limit_lower_bound_seconds': requests / config['source']['max_requests_per_second'], 'observed_seconds_per_request': sec_per_page, 'estimated_retrieval_seconds': requests * sec_per_page if sec_per_page is not None else None, 'uncertainty': "Other years' counts, page sizes, retries and network latency are not measured; up to one extra terminal empty page per query; cache-only runs do not measure latency"}

def build_dataset(config_path, *, smoke=False, cache_only=False):
    started, started_utc = (time.monotonic(), utc_now())
    config_path = Path(config_path)
    config = load_config(config_path)
    if smoke and 2017 not in config['filter']['years']:
        raise ValueError('Smoke year 2017 is outside the contract')
    output_dir = Path(config['output']['dir'])
    if smoke:
        output_dir /= 'smoke_2017_2000'
    years = [2017] if smoke else config['filter']['years']
    client = OpenAlexClient(config, output_dir / 'cache', cache_only=cache_only)
    records, queries = fetch_works(config, client, years, limit_per_query=2000 if smoke else None)
    try:
        entities, relations, counters = build_graph(config, records, years=years)
    except QuotaExhausted as exc:
        atomic_json(output_dir / 'manifest.json', {'id': config['id'], 'status': 'failed', 'mode': 'smoke' if smoke else 'full', 'config': config, 'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(), 'started_at_utc': started_utc, 'finished_at_utc': utc_now(), 'elapsed_seconds': time.monotonic() - started, 'output_dir': str(output_dir), 'retrieval': client.retrieval_report(queries), 'notes': [str(exc)]})
        raise
    splits, notes = split_relations(config, relations)
    stats = graph_statistics(entities, relations)
    checks = run_checks(config, entities, relations, splits, stats)
    relation_floor = minimum_relations(config)
    small = [name for name, edges in relations.items() if len(edges) < relation_floor]
    notes.append('Dataset files and checksums are recorded in the manifest')
    if config['splits'].get('split_version', 1) == 2:
        notes.append("split_version 2: per domain train/val/test are pairwise disjoint and cover all relations; a domain's target test never enters its source train/val")
        notes.append('Validation relation count is floor((relation_count - test_count) * source_val_fraction)')
    else:
        notes.append('train/val files serve source role; test files serve target role in the reverse direction; within one domain these roles may overlap and must never be combined in a single direction')
        notes.append('Validation relation count is floor(relation_count * source_val_fraction)')
    notes.append('Malformed abstract positions are recorded as empty and counted; no text is fabricated')
    if small:
        notes.append(f'Domains below {relation_floor} relations: {small}; ' + ('expected in restricted smoke; full build threshold remains enforced' if smoke else 'STOP and report; do not relax filters'))
    hashes = write_outputs(output_dir, entities, relations, splits)
    passed = all((check['passed'] for check in checks)) and (smoke or not small)
    manifest = {'id': config['id'], 'status': 'passed' if passed else 'failed', 'mode': 'smoke' if smoke else 'full', 'config': config, 'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(), 'started_at_utc': started_utc, 'finished_at_utc': utc_now(), 'elapsed_seconds': time.monotonic() - started, 'output_dir': str(output_dir), 'retrieval': client.retrieval_report(queries), **counters, 'domains': stats, 'checks': checks, 'notes': notes, 'splits': {name: {part: len(edges) for part, edges in parts.items()} for name, parts in splits.items()}, 'directions': [{'source': source, 'target': target, 'train': f'{source}/splits/train.tsv', 'val': f'{source}/splits/val.tsv', 'test': f'{target}/splits/test.tsv'} for source, target in (config['domains']['names'], config['domains']['names'][::-1])], 'sha256': hashes, 'sha256_note': 'Manifest excludes its own hash and raw cache files; cache bodies have separate SHA256', 'full_scale_estimate': full_scale_estimate(config, queries, client) if smoke else {}}
    atomic_json(output_dir / 'manifest.json', manifest)
    if not passed:
        raise ValueError(f'Build checks or scale threshold failed; inspect {output_dir / "manifest.json"}')
    return manifest
