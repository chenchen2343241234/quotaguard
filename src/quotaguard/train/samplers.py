import hashlib
import math
import time
import numpy as np
import torch
import torch.nn.functional as F

def generator_for(seed, stream, *identity):
    key = repr((int(seed), stream, *identity)).encode()
    value = int.from_bytes(hashlib.sha256(key).digest()[:8], 'little') % (2 ** 63 - 1)
    return torch.Generator().manual_seed(value)

def _splitmix64(values):
    with np.errstate(over='ignore'):
        values = values + np.uint64(11400714819323198485)
        values = (values ^ values >> 30) * np.uint64(13787848793156543929)
        values = (values ^ values >> 27) * np.uint64(10723151780598845931)
        return values ^ values >> 31

def poisson_batch(relations, gamma, *, seed, step):
    if not 0 < gamma <= 1:
        raise ValueError('gamma must be in (0, 1]')
    pairs = relations.detach().cpu().numpy().astype(np.uint64, copy=False)
    mask = (1 << 64) - 1
    key = _splitmix64(np.uint64(int(seed) & mask))
    key = _splitmix64(key ^ np.uint64(int(step) & mask))
    key = _splitmix64(key ^ pairs[:, 0])
    key = _splitmix64(key ^ pairs[:, 1])
    uniform = ((key >> 12).astype(np.float64) + 0.5) * 2.0 ** (-52)
    return torch.from_numpy(np.flatnonzero(uniform < gamma))

def fixed_permutation(n_entities, *, seed):
    return torch.randperm(n_entities, generator=generator_for(seed, 'permutation'))

def bounded_parameters(bounded):
    if not isinstance(bounded, dict):
        raise ValueError('bounded requires a parameter mapping')
    mode = bounded.get('mode', 'absolute')
    if mode not in ('absolute', 'positive_relative'):
        raise ValueError('Unknown bounded mode')
    required = {'R', 's0', 'tau'} if mode == 'absolute' else {'R', 'tau', 'mode'}
    if not required <= set(bounded) or set(bounded) - required - {'mode'}:
        raise ValueError(f'bounded {mode} requires {sorted(required)}')
    R, s0, tau = (bounded['R'], bounded.get('s0'), bounded['tau'])
    values = (R, s0, tau) if mode == 'absolute' else (R, tau)
    if not all((math.isfinite(x) for x in values)) or R < 1 or tau <= 0:
        raise ValueError('Invalid bounded parameters')
    return (R, s0, tau)

def bounded_probabilities(scores, a, b, bounded):
    R, s0, tau = bounded_parameters(bounded)
    if bounded.get('mode', 'absolute') == 'positive_relative':
        scores = scores.double()
        logits = ((scores - scores[b] + tau * math.log(R)) / tau).clamp(0, math.log(R))
    else:
        logits = ((scores.double() - s0) / tau).clamp(0, math.log(R))
    logits[[a, b]] = -torch.inf
    return (logits.softmax(0), logits)

def temperature_for_bound(R, N_min):
    if not math.isfinite(R) or R <= 1 or type(N_min) is not int or (N_min < 2):
        raise ValueError('Temperature control needs R > 1 and N_min >= 2')
    return 2 / math.log(R)
ANN_METHODS = ('ivf',)
_ANN_KEYS = {'method', 'candidates', 'nprobe', 'rebuild_every', 'nlist', 'kmeans_iters', 'recall_queries'}
_ANN_CACHE = {}

def reset_ann_cache():
    _ANN_CACHE.clear()

def ann_parameters(ann):
    if not isinstance(ann, dict) or not {'method', 'candidates', 'nprobe'} <= set(ann) or set(ann) - _ANN_KEYS:
        raise ValueError(f'ann requires method/candidates/nprobe; allowed keys {sorted(_ANN_KEYS)}')
    if ann['method'] == 'hnsw':
        raise ValueError('ann method hnsw is unavailable (no faiss in p5); use the torch ivf index')
    if ann['method'] not in ANN_METHODS:
        raise ValueError('Unknown ann method')
    params = {'nlist': None, 'kmeans_iters': 10, 'recall_queries': 256, 'rebuild_every': None, **ann}
    for key in ('candidates', 'nprobe', 'kmeans_iters', 'recall_queries'):
        if type(params[key]) is not int or params[key] < 1:
            raise ValueError(f'ann {key} must be a positive integer')
    for key in ('nlist', 'rebuild_every'):
        if params[key] is not None and (type(params[key]) is not int or params[key] < 1):
            raise ValueError(f'ann {key} must be a positive integer')
    if params['candidates'] < 3:
        raise ValueError('ann candidates must be >= 3 (two endpoints are always excluded)')
    return params

def _dot64(vectors, queries):
    return (vectors.double() * queries.double()).sum(-1)

def _synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

class IVFIndex:

    def __init__(self, table, *, nlist, iters, generator):
        n, device = (table.shape[0], table.device)
        self.table = table
        nlist = max(1, min(nlist, n))
        train = torch.randperm(n, generator=generator)[:min(n, 64 * nlist)].to(device)
        sample = table[train]
        centroids = sample[torch.randperm(len(train), generator=generator)[:nlist].to(device)]
        for _ in range(iters):
            assign = (sample @ centroids.T).argmax(1)
            onehot = F.one_hot(assign, nlist).to(sample.dtype)
            sums = onehot.T @ sample
            counts = onehot.sum(0)
            centroids = torch.where(counts[:, None] > 0, F.normalize(sums, dim=-1), centroids)
        self.centroids = centroids
        assign = torch.cat([(table[i:i + 65536] @ centroids.T).argmax(1) for i in range(0, n, 65536)])
        order = torch.argsort(assign, stable=True)
        self.sizes = torch.bincount(assign, minlength=nlist)
        self.order = order
        self.sizes_cpu = self.sizes.cpu().numpy()
        self.offsets_cpu = np.cumsum(self.sizes_cpu) - self.sizes_cpu
        self.nlist = nlist

    def search(self, queries, candidates, nprobe, *, chunk_bytes=2 ** 28):
        n, device = (self.table.shape[0], self.table.device)
        k = min(candidates, n)
        centroid_scores = torch.cat([_dot64(self.centroids[None], queries[s:s + 8, None, :]) for s in range(0, len(queries), 8)])
        ranks = torch.argsort(centroid_scores, dim=1, descending=True, stable=True).cpu().numpy()
        cumulative = np.cumsum(self.sizes_cpu[ranks], axis=1)
        needed = np.minimum((cumulative < candidates).sum(1) + 1, self.nlist)
        probes = np.maximum(needed, min(nprobe, self.nlist))
        totals = cumulative[np.arange(len(ranks)), probes - 1]
        positions = []
        for i, m in enumerate(probes.tolist()):
            lists = ranks[i, :m]
            starts, sizes = (self.offsets_cpu[lists], self.sizes_cpu[lists])
            positions.append(np.repeat(starts - np.cumsum(sizes) + sizes, sizes) + np.arange(sizes.sum()))
        ids_out, scores_out = ([], [])
        dim, i = (self.table.shape[1], 0)
        while i < len(positions):
            j = i + 1
            while j < len(positions) and (j - i + 1) * int(totals[i:j + 1].max()) * dim * 16 <= chunk_bytes:
                j += 1
            width = int(totals[i:j].max())
            padded = np.zeros((j - i, width), dtype=np.int64)
            valid = np.zeros((j - i, width), dtype=bool)
            for row, pos in enumerate(positions[i:j]):
                padded[row, :len(pos)] = pos
                valid[row, :len(pos)] = True
            ids = self.order[torch.from_numpy(padded).to(device)]
            scores = _dot64(self.table[ids], queries[i:j, None, :])
            scores = scores.masked_fill(~torch.from_numpy(valid).to(device), -torch.inf)
            top = scores.topk(k, dim=1)
            ids_out.append(ids.gather(1, top.indices))
            scores_out.append(top.values)
            i = j
        return (torch.cat(ids_out), torch.cat(scores_out), int((needed > min(nprobe, self.nlist)).sum()))

def ann_index_for(embeddings, ann, *, seed, step, diagnostics=None):
    params = ann_parameters(ann)
    key = (int(seed), tuple(sorted(((k, v) for k, v in params.items() if k != 'recall_queries'))), tuple(embeddings.shape), embeddings.dtype, embeddings.device, embeddings._version)
    if _ANN_CACHE.get('embeddings') is embeddings and _ANN_CACHE.get('key') == key:
        return (_ANN_CACHE['index'], False)
    _ANN_CACHE.clear()
    device = embeddings.device
    _synchronize(device)
    started = time.perf_counter()
    table = F.normalize(embeddings.detach(), dim=-1)
    nlist = params['nlist'] or max(1, round(math.sqrt(table.shape[0])))
    index = IVFIndex(table, nlist=nlist, iters=params['kmeans_iters'], generator=generator_for(seed, 'ann_index', step))
    _synchronize(device)
    elapsed = time.perf_counter() - started
    _ANN_CACHE.update(embeddings=embeddings, key=key, index=index)
    if diagnostics is not None:
        diagnostics['ann_builds'] = diagnostics.get('ann_builds', 0) + 1
        diagnostics['ann_build_seconds'] = diagnostics.get('ann_build_seconds', 0.0) + elapsed
        diagnostics['ann_nlist'] = index.nlist
        diagnostics['ann_max_list_size'] = int(index.sizes.max())
    return (index, True)

def ann_recall(index, queries, ids):
    k, table = (ids.shape[1], index.table)
    recalls = []
    for i, query in enumerate(queries):
        scores = torch.cat([_dot64(table[r:r + 32768], query) for r in range(0, len(table), 32768)])
        recalls.append(float(torch.isin(ids[i], scores.topk(k).indices).sum()) / k)
    return recalls

def ann_sampling_distribution(embeddings, anchors, positives, *, tau, ann, seed, step, diagnostics=None):
    params = ann_parameters(ann)
    index, rebuilt = ann_index_for(embeddings, ann, seed=seed, step=step, diagnostics=diagnostics)
    if len(anchors) == 0:
        return ([], index, rebuilt)
    device = index.table.device
    queries = index.table[anchors.to(device)]
    started = time.perf_counter()
    ids, scores, extended = index.search(queries, params['candidates'], params['nprobe'])
    if diagnostics is not None:
        _synchronize(device)
        diagnostics['ann_search_seconds'] = diagnostics.get('ann_search_seconds', 0.0) + time.perf_counter() - started
        diagnostics['ann_relations'] = diagnostics.get('ann_relations', 0) + len(ids)
        diagnostics['ann_candidate_sum'] = diagnostics.get('ann_candidate_sum', 0) + ids.numel()
        diagnostics['ann_extended_relations'] = diagnostics.get('ann_extended_relations', 0) + extended
        if rebuilt:
            chosen = torch.randperm(len(ids), generator=generator_for(seed, 'ann_recall', step))
            chosen = chosen[:params['recall_queries']].sort().values.to(device)
            recall_started = time.perf_counter()
            recalls = ann_recall(index, queries[chosen], ids[chosen])
            _synchronize(device)
            diagnostics['ann_recall_seconds'] = diagnostics.get('ann_recall_seconds', 0.0) + time.perf_counter() - recall_started
            diagnostics['ann_recall_sum'] = diagnostics.get('ann_recall_sum', 0.0) + sum(recalls)
            diagnostics['ann_recall_count'] = diagnostics.get('ann_recall_count', 0) + len(recalls)
            diagnostics['ann_recall_mean'] = diagnostics['ann_recall_sum'] / diagnostics['ann_recall_count']
            diagnostics['ann_recall_min'] = min(diagnostics.get('ann_recall_min', 1.0), min(recalls))
            diagnostics.setdefault('ann_recall_by_refresh', []).append([int(step), sum(recalls) / len(recalls), min(recalls)])
    logits = scores.double() / tau
    endpoints = (ids == anchors.to(device)[:, None]) | (ids == positives.to(device)[:, None])
    probabilities = logits.masked_fill(endpoints, -torch.inf).softmax(1).cpu()
    ids = ids.cpu()
    return (list(zip(ids, probabilities)), index, rebuilt)

def _sample_negatives_ann(anchors, positives, kind, k_neg, *, n_entities, seed, step, embeddings, tau, ann, distribution_stats, probability_maxima, diagnostics):
    if kind != 'hard':
        raise ValueError('ann supports the hard arm only (case A)')
    if embeddings is None or embeddings.shape[0] != n_entities:
        raise ValueError('Hard sampling needs a public entity embedding cache')
    for a, p in zip(anchors.tolist(), positives.tolist()):
        if a == p or not (0 <= a < n_entities and 0 <= p < n_entities):
            raise ValueError('Invalid relation endpoints')
    excluded = diagnostics.get('ann_build_seconds', 0.0) + diagnostics.get('ann_recall_seconds', 0.0) if diagnostics is not None else 0.0
    started = time.perf_counter()
    distributions, _, _ = ann_sampling_distribution(embeddings, anchors, positives, tau=tau, ann=ann, seed=seed, step=step, diagnostics=diagnostics)
    rows = []
    for (ids, probabilities), a, p in zip(distributions, anchors.tolist(), positives.tolist()):
        if not probabilities.sum() > 0:
            raise ValueError('ann candidate set holds only the relation endpoints')
        rng = generator_for(seed, 'negative', step, a, p)
        draws = torch.multinomial(probabilities, k_neg, replacement=True, generator=rng)
        rows.append(ids[draws])
        if distribution_stats is not None or probability_maxima is not None:
            q = probabilities.double()
            maximum = q.max().item()
            if probability_maxima is not None:
                probability_maxima.append(maximum)
            if distribution_stats is not None:
                entropy = -(q[q > 0] * q[q > 0].log()).sum()
                distribution_stats.append((entropy.item(), entropy.exp().item(), maximum))
    if diagnostics is not None:
        excluded = diagnostics.get('ann_build_seconds', 0.0) + diagnostics.get('ann_recall_seconds', 0.0) - excluded
        diagnostics['ann_sample_seconds'] = diagnostics.get('ann_sample_seconds', 0.0) + time.perf_counter() - started - excluded
        diagnostics['ann_sample_calls'] = diagnostics.get('ann_sample_calls', 0) + 1
    return torch.stack(rows) if rows else torch.empty((0, k_neg), dtype=torch.long)

def sample_negatives(anchors, positives, kind, k_neg, *, n_entities, seed, step, embeddings=None, tau=1.0, permutation=None, distribution_stats=None, hard_filter_margin=None, hard_probability=None, diagnostics=None, bounded=None, cosine_bound=False, probability_maxima=None, query_embeddings=None, ann=None, bm25_table=None):
    if kind == 'bm25_topn':
        if bm25_table is None or bm25_table.n_entities != n_entities or type(k_neg) is not int or (k_neg < 1) or (anchors.shape != positives.shape) or (anchors.ndim != 1) or any((x is not None for x in (ann, bounded, hard_filter_margin, hard_probability))) or cosine_bound:
            raise ValueError('Invalid bm25_topn sampler parameters (notion A only)')
        rows = []
        for a, p in zip(anchors.tolist(), positives.tolist()):
            if a == p or not (0 <= a < n_entities and 0 <= p < n_entities):
                raise ValueError('Invalid relation endpoints')
            candidates = bm25_table.candidates(a, p)
            rng = generator_for(seed, 'bm25_topn', step, a, p)
            rows.append(candidates[torch.randint(len(candidates), (k_neg,), generator=rng)])
            if distribution_stats is not None:
                distribution_stats.append((math.log(len(candidates)), len(candidates), 1 / len(candidates)))
            if probability_maxima is not None:
                probability_maxima.append(1 / len(candidates))
        return torch.stack(rows) if rows else torch.empty((0, k_neg), dtype=torch.long)
    if bm25_table is not None:
        raise ValueError('bm25_table requires kind=bm25_topn')
    if ann is not None:
        if bounded is not None or cosine_bound or hard_filter_margin is not None or (hard_probability is not None) or (query_embeddings is not None):
            raise ValueError('ann is case A only: no bounded/caseB, filtering, mixing or query caches')
        if k_neg < 1 or tau <= 0 or n_entities < 3 or (anchors.shape != positives.shape) or (anchors.ndim != 1):
            raise ValueError('Invalid negative sampler parameters')
        return _sample_negatives_ann(anchors, positives, kind, k_neg, n_entities=n_entities, seed=seed, step=step, embeddings=embeddings, tau=tau, ann=ann, distribution_stats=distribution_stats, probability_maxima=probability_maxima, diagnostics=diagnostics)
    if kind not in {'uniform', 'hard', 'hard_perm', 'nodedp_wor', 'nodedp_bwor', 'bounded_hard'} or k_neg < 1 or tau <= 0:
        raise ValueError('Invalid negative sampler parameters')
    if kind == 'bounded_hard':
        R, _, _ = bounded_parameters(bounded)
    minimum = 2 if kind in {'nodedp_wor', 'nodedp_bwor'} else 3
    if n_entities < minimum or anchors.shape != positives.shape or anchors.ndim != 1:
        raise ValueError('Need matching vectors and enough public entities')
    if kind == 'nodedp_wor' and k_neg > n_entities:
        raise ValueError('nodedp_wor requires k_neg <= n_entities')
    if kind == 'nodedp_bwor' and len(anchors) * k_neg > n_entities:
        raise ValueError('nodedp_bwor requires batch*k_neg <= n_entities')
    if hard_filter_margin is not None or hard_probability is not None:
        if kind != 'hard':
            raise ValueError('Filtering/mixing requires the hard arm')
        if hard_filter_margin is not None and (not math.isfinite(hard_filter_margin)):
            raise ValueError('Filter margin must be finite')
        if hard_probability is not None and (not 0 <= hard_probability <= 1):
            raise ValueError('Hard probability must be in [0, 1]')
    table = None
    if query_embeddings is not None and (kind != 'hard' or cosine_bound or bounded is not None or (embeddings is None) or (query_embeddings.shape != embeddings.shape) or (query_embeddings.device != embeddings.device) or (query_embeddings.dtype != embeddings.dtype)):
        raise ValueError('Asymmetric sampling requires matching hard query/candidate caches')
    if kind in {'hard', 'hard_perm', 'bounded_hard'}:
        if embeddings is None or embeddings.shape[0] != n_entities:
            raise ValueError('Hard sampling needs a public entity embedding cache')
        table = embeddings.detach() if query_embeddings is not None else F.normalize(embeddings.detach(), dim=-1)
        if kind == 'hard_perm':
            if permutation is None or not torch.equal(permutation.cpu().sort().values, torch.arange(n_entities)):
                raise ValueError('Invalid fixed public permutation')
            table = table[permutation.to(table.device)]
    if kind == 'nodedp_bwor':
        for a, p in zip(anchors.tolist(), positives.tolist()):
            if a == p or not (0 <= a < n_entities and 0 <= p < n_entities):
                raise ValueError('Invalid relation endpoints')
        rng = generator_for(seed, 'nodedp_batch', step)
        total = len(anchors) * k_neg
        swaps, draws = ({}, [])
        for j in range(total):
            remaining = n_entities - j
            slot = int(torch.randint(remaining, (), generator=rng))
            draws.append(swaps.get(slot, slot))
            swaps[slot] = swaps.get(remaining - 1, remaining - 1)
        nodes = torch.tensor(draws, dtype=torch.long).view(len(anchors), k_neg)
        return (nodes, torch.randint(2, (len(anchors), k_neg), generator=rng))
    rows = []
    endpoints = []
    for a, p in zip(anchors.tolist(), positives.tolist()):
        if a == p or not (0 <= a < n_entities and 0 <= p < n_entities):
            raise ValueError('Invalid relation endpoints')
        if kind == 'nodedp_wor':
            rng = generator_for(seed, 'nodedp', step, a, p)
            swaps, draws = ({}, [])
            for j in range(k_neg):
                remaining = n_entities - j
                slot = int(torch.randint(remaining, (), generator=rng))
                draws.append(swaps.get(slot, slot))
                swaps[slot] = swaps.get(remaining - 1, remaining - 1)
            rows.append(torch.tensor(draws, dtype=torch.long))
            endpoints.append(torch.randint(2, (k_neg,), generator=rng))
            continue
        rng = generator_for(seed, 'bounded' if kind == 'bounded_hard' else 'negative', step, a, p)
        if kind == 'uniform':
            lo, hi = sorted((a, p))
            draws = torch.randint(n_entities - 2, (k_neg,), generator=rng)
            draws += draws >= lo
            draws += draws >= hi
            if probability_maxima is not None:
                probability_maxima.append(1 / (n_entities - 2))
            if bounded is not None and bounded['R'] == 1:
                if distribution_stats is not None:
                    distribution_stats.append((math.log(n_entities - 2), n_entities - 2, 1 / (n_entities - 2)))
                if diagnostics is not None:
                    for key in ('bounded_candidates', 'bounded_lower', 'bounded_upper'):
                        diagnostics[key] = diagnostics.get(key, 0) + n_entities - 2
        else:
            scores = table @ (query_embeddings[a].detach() if query_embeddings is not None else table[a])
            if cosine_bound:
                scores = scores.clamp(-1, 1)
            logits = scores / tau
            fallback = False
            if hard_filter_margin is not None:
                excluded = scores > scores[p] - hard_filter_margin
                excluded[[a, p]] = False
                logits[excluded] = -torch.inf
                fallback = int(excluded.sum()) == n_entities - 2
                if diagnostics is not None:
                    for key, value in {'filter_candidates': n_entities - 2, 'filtered_candidates': int(excluded.sum()), 'filter_relations': 1, 'filter_fallbacks': int(fallback)}.items():
                        diagnostics[key] = diagnostics.get(key, 0) + value
                if fallback:
                    logits.zero_()
            logits[[a, p]] = -torch.inf
            if kind == 'bounded_hard':
                probabilities, bounded_logits = bounded_probabilities(scores, a, p, bounded)
                probabilities = probabilities.cpu()
                if diagnostics is not None:
                    valid = torch.isfinite(bounded_logits)
                    for key, value in {'bounded_candidates': int(valid.sum()), 'bounded_lower': int((bounded_logits[valid] == 0).sum()), 'bounded_upper': int((bounded_logits[valid] == math.log(R)).sum())}.items():
                        diagnostics[key] = diagnostics.get(key, 0) + value
            else:
                probabilities = logits.softmax(dim=0).cpu()
            draws = torch.multinomial(probabilities, k_neg, replacement=True, generator=rng)
            if fallback or hard_probability is not None:
                uniform_rng = generator_for(seed, 'negative_uniform', step, a, p)
                uniform = torch.randint(n_entities - 2, (k_neg,), generator=uniform_rng)
                lo, hi = sorted((a, p))
                uniform += uniform >= lo
                uniform += uniform >= hi
                if fallback:
                    draws = uniform
                if hard_probability is not None:
                    choices = torch.rand(k_neg, generator=generator_for(seed, 'hard_mix', step, a, p)) < hard_probability
                    draws = torch.where(choices, draws, uniform)
                    if diagnostics is not None:
                        diagnostics['mix_slots'] = diagnostics.get('mix_slots', 0) + k_neg
                        diagnostics['mix_hard_slots'] = diagnostics.get('mix_hard_slots', 0) + int(choices.sum())
            if distribution_stats is not None or probability_maxima is not None:
                q = probabilities.detach()
                if distribution_stats is not None or hard_probability is not None:
                    q = q.double()
                if hard_probability is not None:
                    uniform_q = torch.full_like(q, 1 / (n_entities - 2))
                    uniform_q[[a, p]] = 0
                    q = hard_probability * q + (1 - hard_probability) * uniform_q
                maximum = q.max().item()
                if probability_maxima is not None:
                    probability_maxima.append(maximum)
                if distribution_stats is not None:
                    entropy = -(q[q > 0] * q[q > 0].log()).sum()
                    distribution_stats.append((entropy.item(), entropy.exp().item(), maximum))
        rows.append(draws)
    nodes = torch.stack(rows) if rows else torch.empty((0, k_neg), dtype=torch.long)
    if kind == 'nodedp_wor':
        positions = torch.stack(endpoints) if endpoints else torch.empty((0, k_neg), dtype=torch.long)
        return (nodes, positions)
    return nodes
