from collections import Counter
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from .data import load_domain, read_manifest_tsv, tokenize_titles
from .model import ResidualE5, encode_entities
EVALUATION_BATCH_SIZE = 256

@torch.no_grad()
def evaluate_embeddings(embeddings, relations, *, seed, query_embeddings=None):
    if embeddings.ndim != 2 or not embeddings.is_floating_point() or embeddings.shape[1] == 0 or (not torch.isfinite(embeddings).all()) or (embeddings.norm(dim=1) == 0).any():
        raise ValueError('Embeddings must be finite, nonzero floating-point vectors')
    if relations.ndim != 2 or relations.shape[1] != 2 or relations.dtype != torch.long:
        raise ValueError('Relations must be a LongTensor of shape [N, 2]')
    if len(relations) == 0:
        raise ValueError('Cannot evaluate an empty relation split')
    if (relations < 0).any() or (relations >= len(embeddings)).any():
        raise ValueError('Relation entity index is out of range')
    if (relations[:, 0] == relations[:, 1]).any():
        raise ValueError('Self-loop would remove the correct answer')
    vectors = F.normalize(embeddings, dim=1)
    query_vectors = _query_vectors(embeddings, vectors, query_embeddings)
    order = torch.randperm(len(relations), generator=torch.Generator().manual_seed(seed))
    shuffled = relations.detach().cpu()[order]
    reciprocal_sum, top_one = (0.0, 0)
    batch_relation_counts, batch_candidate_counts = ([], [])
    for batch in shuffled.split(EVALUATION_BATCH_SIZE):
        batch = batch.to(vectors.device)
        anchors, positives = batch.unbind(dim=1)
        candidates, correct_columns = torch.unique(positives, sorted=True, return_inverse=True)
        scores = query_vectors[anchors] @ vectors[candidates].T
        correct_scores = scores.gather(1, correct_columns[:, None])
        competitors = (candidates[None, :] != anchors[:, None]) & (candidates[None, :] != positives[:, None])
        ranks = 1 + ((scores >= correct_scores) & competitors).sum(dim=1)
        reciprocal_sum += ranks.double().reciprocal().sum().item()
        top_one += (ranks == 1).sum().item()
        batch_relation_counts.append(len(batch))
        batch_candidate_counts.append(len(candidates))
    return {'mrr': reciprocal_sum / len(relations), 'prec_at_1': top_one / len(relations), 'n_relations': len(relations), 'seed': seed, 'batch_size': EVALUATION_BATCH_SIZE, 'tail_batch_size': len(relations) % EVALUATION_BATCH_SIZE, 'batch_relation_counts': batch_relation_counts, 'batch_candidate_counts': batch_candidate_counts, 'candidate_count_histogram': dict(sorted(Counter(batch_candidate_counts).items()))}

def _query_vectors(embeddings, vectors, query_embeddings):
    if query_embeddings is None:
        return vectors
    if query_embeddings.shape != embeddings.shape or query_embeddings.device != embeddings.device or query_embeddings.dtype != embeddings.dtype or (not torch.isfinite(query_embeddings).all()) or (query_embeddings.norm(dim=1) == 0).any():
        raise ValueError('Query embeddings must match candidate shape/device/dtype and be finite, nonzero')
    return F.normalize(query_embeddings, dim=1)

@torch.no_grad()
def evaluate_full_corpus(embeddings, relations, known_relations, *, k_values=(10, 100), query_block_size=256, candidate_block_size=8192, query_embeddings=None):
    if embeddings.ndim != 2 or not embeddings.is_floating_point() or embeddings.shape[1] == 0 or (not torch.isfinite(embeddings).all()) or (embeddings.norm(dim=1) == 0).any():
        raise ValueError('Embeddings must be finite, nonzero floating-point vectors')
    for name, edges in (('Relations', relations), ('Known relations', known_relations)):
        if edges.ndim != 2 or edges.shape[1] != 2 or edges.dtype != torch.long:
            raise ValueError(f'{name} must be a LongTensor of shape [N, 2]')
        if (edges < 0).any() or (edges >= len(embeddings)).any():
            raise ValueError(f'{name} entity index is out of range')
        if (edges[:, 0] == edges[:, 1]).any():
            raise ValueError(f'Self-loop in {name.lower()}')
    if len(relations) == 0:
        raise ValueError('Cannot evaluate an empty relation split')
    if any((type(size) is not int or size < 1 for size in (query_block_size, candidate_block_size))):
        raise ValueError('Block sizes must be positive integers')
    k_values = tuple(k_values)
    if not k_values or any((type(k) is not int or k < 1 for k in k_values)):
        raise ValueError('k_values must contain positive integers')
    neighbours = {}
    for a, b in known_relations.detach().cpu().tolist():
        neighbours.setdefault(a, set()).add(b)
        neighbours.setdefault(b, set()).add(a)
    vectors = F.normalize(embeddings, dim=1)
    query_vectors = _query_vectors(embeddings, vectors, query_embeddings)
    reciprocal_sum = 0.0
    hits = dict.fromkeys(k_values, 0)
    for batch_cpu in relations.detach().cpu().split(query_block_size):
        batch = batch_cpu.to(vectors.device)
        anchors, positives = batch.unbind(dim=1)
        queries = query_vectors[anchors]
        correct_scores = torch.empty(len(batch), device=vectors.device, dtype=vectors.dtype)
        answer_blocks = sorted(set((batch_cpu[:, 1] // candidate_block_size).tolist()))
        for block_index in answer_blocks:
            start = block_index * candidate_block_size
            stop = min(start + candidate_block_size, len(vectors))
            scores = queries @ vectors[start:stop].T
            rows = ((positives >= start) & (positives < stop)).nonzero().flatten()
            correct_scores[rows] = scores[rows, positives[rows] - start]
        ranks = torch.ones(len(batch), dtype=torch.long, device=vectors.device)
        filtered_rows, filtered_columns = ([], [])
        for row, (a, b) in enumerate(batch_cpu.tolist()):
            excluded = neighbours.get(a, set()) | {a, b}
            filtered_rows.extend([row] * len(excluded))
            filtered_columns.extend(excluded)
        filtered_rows = torch.tensor(filtered_rows, dtype=torch.long, device=vectors.device)
        filtered_columns = torch.tensor(filtered_columns, dtype=torch.long, device=vectors.device)
        for start in range(0, len(vectors), candidate_block_size):
            stop = min(start + candidate_block_size, len(vectors))
            scores = queries @ vectors[start:stop].T
            excluded = (filtered_columns >= start) & (filtered_columns < stop)
            scores[filtered_rows[excluded], filtered_columns[excluded] - start] = -torch.inf
            ranks += (scores >= correct_scores[:, None]).sum(dim=1)
        reciprocal_sum += ranks.double().reciprocal().sum().item()
        for k in hits:
            hits[k] += (ranks <= k).sum().item()
    return {'mrr': reciprocal_sum / len(relations), **{f'recall_at_{k}': hits[k] / len(relations) for k in hits}, 'n': len(relations)}

def _evaluate_split(model, tokenizer, config, *, domain, split, role):
    eval_seed = config['eval_seed']
    data = load_domain(config['data_dir'], domain, split, role=role)
    device = next(model.parameters()).device
    tokens = {k: v.to(device) for k, v in tokenize_titles(data.titles, tokenizer).items()}
    embeddings = encode_entities(model, tokens, config['encode_batch_size'])
    scoring = {'query_embeddings': model.query_embeddings(embeddings).detach()} if isinstance(model, ResidualE5) else {}
    result = evaluate_embeddings(embeddings, data.relations, seed=eval_seed, **scoring)
    if role == 'final_report' and config.get('full_corpus_eval') is True:
        root = Path(config['data_dir'])
        manifest = json.loads((root / 'manifest.json').read_text())
        index = {entity: i for i, entity in enumerate(data.ids)}
        edges = read_manifest_tsv(root, manifest, f'{domain}/relations.tsv')
        known = torch.tensor([(index[a], index[b]) for a, b in edges], dtype=torch.long).reshape(-1, 2)
        result['full_corpus'] = evaluate_full_corpus(embeddings, data.relations, known, query_block_size=config.get('full_corpus_query_block_size', 256), candidate_block_size=config.get('full_corpus_candidate_block_size', 8192), **scoring)
    return dict(result, domain=domain, split=split, role=role)

def evaluate_source_validation(model, tokenizer, config):
    return _evaluate_split(model, tokenizer, config, domain=config['source'], split='val', role='selection')

def evaluate_target_test(model, tokenizer, config, *, final_report=False):
    if final_report is not True:
        raise ValueError('Target test evaluation requires explicit final_report=True')
    return _evaluate_split(model, tokenizer, config, domain=config['target'], split='test', role='final_report')
