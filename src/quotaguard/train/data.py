import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import torch

def read_manifest_tsv(root, manifest, relative):
    payload = (Path(root) / relative).read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest['sha256'][relative]:
        raise ValueError(f'Manifest checksum mismatch: {relative}')
    lines = payload.decode('utf-8').split('\n')
    if lines[-1] == '':
        lines.pop()
    rows = [line.split('\t') for line in lines]
    if any((len(row) != 2 for row in rows)):
        raise ValueError(f'Expected exactly two TSV columns: {relative}')
    return rows
TITLE_ONLY = ('title',)
TEXT_FIELD_OPTIONS = (TITLE_ONLY, ('title', 'abstract'))
DEFAULT_MAX_LENGTH = 32
ABSTRACT_SEPARATOR = ' [SEP] '

def validate_text_options(text_fields=TITLE_ONLY, max_length=DEFAULT_MAX_LENGTH):
    if not isinstance(text_fields, (list, tuple)) or tuple(text_fields) not in TEXT_FIELD_OPTIONS:
        raise ValueError('text_fields must be [title] or [title, abstract]')
    if type(max_length) is not int or not 8 <= max_length <= 512:
        raise ValueError('max_length must be an integer in [8, 512]')
    return (tuple(text_fields), max_length)

def is_missing_abstract(abstract):
    return not abstract.strip()

class _AbstractStore:

    def __init__(self, root, manifest, domain, ids):
        self.root, self.manifest, self.ids = (root, manifest, ids)
        self.relative = f'{domain}/abstracts.tsv'
        self._abstracts = None

    def abstracts(self):
        if self._abstracts is None:
            rows = read_manifest_tsv(self.root, self.manifest, self.relative)
            if tuple((entity for entity, _ in rows)) != self.ids:
                raise ValueError(f'Abstract rows do not match entity order: {self.relative}')
            self._abstracts = tuple((abstract for _, abstract in rows))
        return self._abstracts

    def missing_rate(self):
        abstracts = self.abstracts()
        return sum(map(is_missing_abstract, abstracts)) / len(abstracts)

class EntityTitle(str):

    def __new__(cls, title, store, index):
        self = super().__new__(cls, title)
        self._store, self._index = (store, index)
        return self

    def __reduce__(self):
        return (str, (str(self),))

    @property
    def abstract(self):
        return self._store.abstracts()[self._index]

@dataclass
class DomainData:
    ids: tuple[str, ...]
    titles: tuple[str, ...]
    relations: torch.Tensor
    domain: str
    split: str
    role: str

def load_domain(root, domain, split, *, role):
    allowed = {'training': 'train', 'selection': 'val', 'final_report': 'test'}
    if role not in allowed or split != allowed[role]:
        raise ValueError('Split is forbidden for this role')
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('status') != 'passed' or domain not in manifest['domains']:
        raise ValueError('Dataset manifest did not pass or domain is unknown')

    def read(relative):
        return read_manifest_tsv(root, manifest, relative)
    entities = read(f'{domain}/entities.tsv')
    ids, titles = zip(*entities)
    if len(set(ids)) != len(ids) or not all(titles):
        raise ValueError('Duplicate entities or empty titles')
    if f'{domain}/abstracts.tsv' in manifest['sha256']:
        store = _AbstractStore(root, manifest, domain, ids)
        titles = tuple((EntityTitle(title, store, i) for i, title in enumerate(titles)))
    index = {v: i for i, v in enumerate(ids)}
    edges = read(f'{domain}/splits/{split}.tsv')
    pairs = [(index[a], index[b]) for a, b in edges]
    if any((a == b for a, b in pairs)) or len({tuple(sorted(e)) for e in pairs}) != len(pairs):
        raise ValueError('Self-loop or duplicate relation')
    return DomainData(ids, titles, torch.tensor(pairs, dtype=torch.long).reshape(-1, 2), domain, split, role)

def clip_degree(relations, n_entities, kdeg, *, seed):
    if kdeg < 1:
        raise ValueError('Kdeg must be positive')
    relations = relations.cpu()
    degree = torch.zeros(n_entities, dtype=torch.long)
    keep = []
    order = torch.randperm(len(relations), generator=torch.Generator().manual_seed(seed))
    for i in order.tolist():
        a, b = relations[i].tolist()
        if a == b:
            raise ValueError('Self-loop')
        if degree[a] < kdeg and degree[b] < kdeg:
            keep.append(i)
            degree[a] += 1
            degree[b] += 1
    result = relations[sorted(keep)]
    assert int(torch.bincount(result.flatten(), minlength=n_entities).max()) <= kdeg
    return result

class TextOptionsTokenizer:

    def __init__(self, tokenizer, text_fields, max_length):
        self.tokenizer = tokenizer
        self.text_options = validate_text_options(text_fields, max_length)

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

def text_options(tokenizer):
    options = getattr(tokenizer, 'text_options', None)
    return options if options is not None else (TITLE_ONLY, DEFAULT_MAX_LENGTH)

def entity_texts(titles, text_fields=TITLE_ONLY):
    if tuple(text_fields) == TITLE_ONLY:
        return [str(title) for title in titles]
    validate_text_options(text_fields)
    texts = []
    for title in titles:
        if not isinstance(title, EntityTitle):
            raise ValueError('Abstract text requires titles loaded by load_domain from a dataset whose manifest lists abstracts.tsv')
        abstract = title.abstract
        texts.append(str(title) if is_missing_abstract(abstract) else str(title) + ABSTRACT_SEPARATOR + abstract)
    return texts

def tokenize_titles(titles, tokenizer):
    text_fields, max_length = text_options(tokenizer)
    return tokenizer(entity_texts(titles, text_fields), padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')

def abstract_missing_rates(root, domains):
    rates = {}
    for domain in domains:
        titles = load_domain(root, domain, 'train', role='training').titles
        if not titles or not isinstance(titles[0], EntityTitle):
            raise ValueError(f'Dataset manifest has no {domain}/abstracts.tsv')
        rates[domain] = titles[0]._store.missing_rate()
    return rates

def shuffle_labels(relations, *, seed):
    if type(seed) is not int:
        raise ValueError('label_shuffle_seed must be a fixed integer')
    original = relations.cpu().tolist()
    key = lambda a, b: (min(a, b), max(a, b))
    if any((a == b for a, b in original)) or len({key(a, b) for a, b in original}) != len(original):
        raise ValueError('Label shuffle requires a simple clipped graph')
    order = torch.randperm(len(original), generator=torch.Generator().manual_seed(seed)).tolist()
    pairs = [[a, original[j][1]] for (a, _), j in zip(original, order)]
    counts = Counter((key(a, b) for a, b in pairs))
    stats = dict(seed=seed, relations=len(pairs), initial_self_loops=sum((a == b for a, b in pairs)), initial_duplicate_excess=sum((n - 1 for n in counts.values())), repair_swaps=0, fallback_to_original=False, repair_rule='first legal cyclic two-row swap; original assignment fallback')
    for i, (a, b) in enumerate(pairs):
        old = key(a, b)
        if a != b and counts[old] == 1:
            continue
        counts[old] -= 1
        for offset in range(1, len(pairs)):
            j = (i + offset) % len(pairs)
            c, d = pairs[j]
            other, left, right = (key(c, d), key(a, d), key(c, b))
            counts[other] -= 1
            legal = a != d and c != b and (left != right) and (not counts[left]) and (not counts[right])
            counts[other] += 1
            if legal:
                counts[other] -= 1
                counts[left] += 1
                counts[right] += 1
                pairs[i][1], pairs[j][1] = (d, b)
                stats['repair_swaps'] += 1
                break
        else:
            pairs = original
            stats['fallback_to_original'] = True
            break
    result = torch.tensor(pairs, dtype=relations.dtype, device=relations.device).reshape(-1, 2)
    stats.update(changed_second_endpoints=sum((b != d for (_, b), (_, d) in zip(original, pairs))), retained_original_relations=len({key(a, b) for a, b in original} & {key(a, b) for a, b in pairs}), final_self_loops=sum((a == b for a, b in pairs)), final_duplicate_excess=len(pairs) - len({key(a, b) for a, b in pairs}))
    return (result, stats)
