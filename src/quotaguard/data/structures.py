from dataclasses import dataclass, replace
from functools import lru_cache
import ctypes
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from .buckets import PublicBuckets, construct_buckets, edge_priority, public_priority

def _i(x):
    return np.ascontiguousarray(x, dtype=np.int32)

def _d(x):
    return np.ascontiguousarray(x, dtype=np.float64)

@lru_cache(maxsize=1)
def kernel():
    source = Path(__file__).with_name('structures_kernel.cpp')
    key = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    directory = source.parents[3] / 'tmp' / 'structure_kernel' / key
    directory.mkdir(parents=True, exist_ok=True)
    library = directory / ('structures.dll' if sys.platform == 'win32' else 'structures.so')
    if not library.exists():
        compiler = shutil.which('g++') or shutil.which('clang++')
        if not compiler:
            raise RuntimeError('structure requires g++/clang++ for compiled numeric kernels')
        command = [compiler, '-O3', '-std=c++17', '-shared', str(source), '-o', str(library)]
        command += ['-static-libgcc', '-static-libstdc++'] if sys.platform == 'win32' else ['-fPIC']
        subprocess.run(command, check=True, capture_output=True, text=True)
    dll_directory = None
    if sys.platform == 'win32' and hasattr(os, 'add_dll_directory'):
        compiler = shutil.which('g++') or shutil.which('clang++')
        if compiler:
            dll_directory = os.add_dll_directory(str(Path(compiler).parent))
    lib = ctypes.CDLL(str(library))
    lib._dll_directory = dll_directory
    ip = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    dp = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    bp = np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS')
    c = ctypes.c_int
    signatures = dict(groups=([c, c, c, ip, ip, ip], None), matching=([c, c, c, ip, ip, ip, ip, ip], c), assign=([c, ip, ip, ip, ip, c], None), pb=([c, c, c, ip, dp, dp], None), quota_dp=([c, c, ip, ip, ip, ip, ip, ip, dp, ip], ctypes.c_double), bmatch=([c, c, c, c, ip, ip, bp], None))
    for name, (args, result) in signatures.items():
        getattr(lib, name).argtypes, getattr(lib, name).restype = (args, result)
    return lib

def pair_keys(pairs, n):
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if np.any(pairs < 0) or np.any(pairs >= n):
        raise ValueError('Unknown endpoint')
    a, b = (np.minimum(pairs[:, 0], pairs[:, 1]), np.maximum(pairs[:, 0], pairs[:, 1]))
    if np.any(a == b):
        raise ValueError('Self-loop')
    return a * n + b

def membership(values, sorted_keys):
    if not len(sorted_keys):
        return np.zeros(len(values), dtype=bool)
    p = np.searchsorted(sorted_keys, values)
    return (p < len(sorted_keys)) & (sorted_keys[np.minimum(p, len(sorted_keys) - 1)] == values)

@dataclass
class Candidates:
    ids: tuple
    neighbors: np.ndarray
    scores: np.ndarray
    pairs: np.ndarray
    cosine: np.ndarray
    ranks: np.ndarray
    lengths: np.ndarray
    limit: int
    matrix: object = None

    @property
    def keys(self):
        return self.pairs[:, 0].astype(np.int64) * len(self.ids) + self.pairs[:, 1]

    def features(self, rows=None):
        rows = slice(None) if rows is None else rows
        ranks = self.ranks[rows]
        return np.column_stack((self.cosine[rows], np.log1p(ranks), np.all(ranks <= self.limit, axis=1), np.log1p(self.lengths[self.pairs[rows]]))).astype(np.float64)

    def extend(self, pairs):
        new = np.setdiff1d(pair_keys(pairs, len(self.ids)), self.keys, assume_unique=False)
        if not len(new):
            return self
        endpoints = _i(np.column_stack((new // len(self.ids), new % len(self.ids))))
        cosine = np.zeros(len(new), dtype=np.float32)
        if self.matrix is not None:
            for lo in range(0, len(new), 100000):
                a, b = endpoints[lo:lo + 100000].T
                cosine[lo:lo + len(a)] = np.asarray(self.matrix[a].multiply(self.matrix[b]).sum(axis=1)).ravel()
        keys = np.concatenate((self.keys, new))
        order = np.argsort(keys)
        return replace(self, pairs=_i(np.concatenate((self.pairs, endpoints))[order]), cosine=np.concatenate((self.cosine, cosine))[order], ranks=_i(np.concatenate((self.ranks, np.full((len(new), 2), self.limit + 1)))[order]))

def candidates_from_neighbors(ids, neighbors, scores, lengths, limit, matrix=None):
    ids = tuple(ids)
    if ids != tuple(sorted(ids)) or len(set(ids)) != len(ids):
        raise ValueError('Immutable IDs must be sorted and unique')
    n = len(ids)
    nn, ss = (_i(neighbors[:, :limit]), np.asarray(scores[:, :limit], dtype=np.float32))
    if nn.shape != ss.shape or np.any(nn < -1) or np.any(nn >= n):
        raise ValueError('Invalid public neighbors')
    src = np.broadcast_to(np.arange(n, dtype=np.int32)[:, None], nn.shape)
    rank = np.broadcast_to(np.arange(1, nn.shape[1] + 1, dtype=np.int32), nn.shape)
    valid = (nn >= 0) & (nn != src)
    s, t, ranks, weights = (src[valid], nn[valid], rank[valid], ss[valid])
    directed = s.astype(np.int64) * n + t
    order = np.lexsort((ranks, directed))
    first = np.r_[True, np.diff(directed[order]) != 0] if len(order) else np.zeros(0, bool)
    order = order[first]
    s, t, ranks, weights = (s[order], t[order], ranks[order], weights[order])
    keys = np.minimum(s, t).astype(np.int64) * n + np.maximum(s, t)
    unique, inv = np.unique(keys, return_inverse=True)
    rr = np.full((len(unique), 2), limit + 1, dtype=np.int32)
    rr[inv, (s > t).astype(int)] = ranks
    cosine = np.zeros(len(unique), dtype=np.float32)
    np.maximum.at(cosine, inv, weights)
    return Candidates(ids, nn, ss, _i(np.column_stack((unique // n, unique % n))), cosine, rr, np.asarray(lengths, dtype=np.int32), limit, matrix)

def generate_candidates(ids, titles, abstracts, mode, limit, config, progress=print):
    from sklearn.feature_extraction.text import TfidfVectorizer
    if mode not in ('title', 'title_abstract'):
        raise ValueError('Unknown text mode')
    texts = [title + (' ' + abstract if mode == 'title_abstract' and abstract.strip() else '') for title, abstract in zip(titles, abstracts)]
    options = dict(config['tfidf'])
    options['ngram_range'] = tuple(options['ngram_range'])
    matrix = TfidfVectorizer(dtype=np.float32, **options).fit_transform(texts)
    n = len(ids)
    nn, ss = (np.full((n, limit), -1, np.int32), np.zeros((n, limit), np.float32))
    block, transposed = (config['similarity_block_size'], matrix.T.tocsc())
    for start in range(0, n, block):
        sims = (matrix[start:start + block] @ transposed).tocsr()
        for offset in range(sims.shape[0]):
            lo, hi = sims.indptr[offset:offset + 2]
            idx, val = (sims.indices[lo:hi], sims.data[lo:hi])
            valid = (idx != start + offset) & (val > 0)
            idx, val = (idx[valid], val[valid])
            if len(idx) > limit:
                threshold = np.partition(val, -limit)[-limit]
                valid = val >= threshold
                idx, val = (idx[valid], val[valid])
            order = np.lexsort((idx, -val))[:limit]
            nn[start + offset, :len(order)], ss[start + offset, :len(order)] = (idx[order], val[order])
        if start % (block * 100) == 0:
            progress(f'public {mode} neighbors {start}/{n}')
    return candidates_from_neighbors(ids, nn, ss, [len(t) for t in texts], limit, matrix)

@dataclass
class Calibration:
    coef: list
    intercept: float
    mean: list
    scale: list
    prior_logit_correction: float
    report: dict

    def predict(self, candidates):
        from scipy.special import expit
        out = np.empty(len(candidates.pairs), dtype=np.float64)
        for lo in range(0, len(out), 100000):
            features = candidates.features(slice(lo, lo + 100000))
            out[lo:lo + len(features)] = expit((features - self.mean) / self.scale @ np.asarray(self.coef) + self.intercept + self.prior_logit_correction)
        return out

def fit_calibration(candidates, relations, config, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import brier_score_loss, log_loss
    labels = membership(candidates.keys, np.sort(pair_keys(relations, len(candidates.ids))))
    positive, negative = (np.flatnonzero(labels), np.flatnonzero(~labels))
    if not len(positive) or not len(negative):
        raise ValueError('Calibration needs both positive and negative candidates')
    size = min(len(negative), len(positive) * config['negative_sampling']['pairs_per_positive'])
    sample = np.random.default_rng(seed).choice(negative, size, replace=False)
    rows = np.concatenate((positive, sample))
    x, y = (candidates.features(rows), labels[rows].astype(int))
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=config['regularization_C'], solver='lbfgs', random_state=seed, max_iter=1000).fit(scaler.transform(x), y)
    if not config['negative_sampling']['correct_prior']:
        raise ValueError('structure requires negative-sampling prior correction')
    correction = float(np.log(size / len(negative)))
    result = Calibration(model.coef_[0].tolist(), float(model.intercept_[0]), scaler.mean_.tolist(), scaler.scale_.tolist(), correction, {})
    probability = result.predict(candidates)
    weight = np.where(y == 1, 1.0, len(negative) / size)
    iso = IsotonicRegression(out_of_bounds='clip').fit(x[:, 0], y, sample_weight=weight)
    ip = iso.predict(candidates.cosine)
    bins = []
    for low, high in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (probability >= low) & (probability < high if high < 1 else probability <= 1)
        bins.append(dict(lower=low, upper=high, count=int(mask.sum()), predicted=float(probability[mask].mean()) if mask.any() else None, observed=float(labels[mask].mean()) if mask.any() else None))
    result.report = dict(population_pairs=len(labels), positives=len(positive), negatives=len(negative), sampled_negatives=size, negative_inclusion_probability=size / len(negative), logistic_brier=float(brier_score_loss(labels, probability)), logistic_log_loss=float(log_loss(labels, probability)), reliability_bins=bins, isotonic_report_only=dict(brier=float(brier_score_loss(labels, ip)), thresholds=iso.X_thresholds_.tolist(), probabilities=iso.y_thresholds_.tolist()), diagnostics_scope='BIO-CN in-sample descriptive calibration, not holdout performance')
    return result

def poisson_binomial(probabilities, k):
    p = _d(probabilities)
    if k < 1 or np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid probability/count')
    f = np.zeros((1, k + 1), dtype=np.float64)
    kernel().pb(len(p), 1, k, np.zeros(len(p), np.int32), p, f)
    tails = np.diff(f[0])
    mass = np.r_[1 - tails[0], tails[:-1] - tails[1:], tails[-1]]
    return (mass, f[0])

def bucket_objectives(probabilities, assignment, nodes, k=4):
    probabilities, assignment = (_d(probabilities), _i(assignment))
    if len(probabilities) != len(assignment) or np.any(assignment < -1) or np.any(assignment >= nodes):
        raise ValueError('Invalid candidate assignment')
    if np.any(~np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError('Invalid probabilities')
    f = np.zeros((nodes, k + 1), dtype=np.float64)
    kernel().pb(len(assignment), nodes, k, assignment, probabilities, f)
    return f

@dataclass
class Forest:
    ids: tuple
    pairs: np.ndarray
    assignment: np.ndarray
    left: np.ndarray
    right: np.ndarray
    parent: np.ndarray
    members: np.ndarray
    offsets: np.ndarray
    roots: np.ndarray
    budgets: np.ndarray
    capacities: np.ndarray
    counts: np.ndarray
    layer: np.ndarray
    seed: int
    kdeg: int
    capacity: int
    legacy_embedded: bool = False

    def entities(self, node):
        return self.members[self.offsets[node]:self.offsets[node + 1]]

    def to_buckets(self, quotas, scheme='dp_forest'):
        quotas = _i(quotas)
        if len(quotas) != len(self.left) or np.any(quotas < 0) or np.any(quotas > self.counts):
            raise ValueError('Invalid node quota')
        charge = np.zeros(len(self.ids), dtype=np.int32)
        for node in np.flatnonzero(quotas):
            charge[self.entities(node)] += quotas[node]
        if charge.max(initial=0) > self.kdeg or int(quotas.sum()) > self.capacity:
            raise ValueError('Illegal conservative quota/capacity')
        for root, budget, cap in zip(self.roots, self.budgets, self.capacities):
            stack, total = ([(int(root), int(budget))], 0)
            while stack:
                t, k = stack.pop()
                if quotas[t] > k:
                    raise ValueError('Layer path quota exceeded')
                total += int(quotas[t])
                if self.left[t] >= 0:
                    stack.extend(((int(self.left[t]), k - int(quotas[t])), (int(self.right[t]), k - int(quotas[t]))))
            if total > cap:
                raise ValueError('Root capacity exceeded')
        active = np.flatnonzero(quotas)
        remap = np.full(len(quotas), -1, dtype=np.int32)
        remap[active] = np.arange(len(active), dtype=np.int32)
        rows = np.flatnonzero(self.assignment >= 0)
        rows = rows[quotas[self.assignment[rows]] > 0]
        result = PublicBuckets(self.ids, _i(self.pairs[rows]), _i(remap[self.assignment[rows]]), _i(quotas[active]), self.seed, self.kdeg, scheme)
        validate_buckets(result)
        return result

def validate_buckets(buckets):
    if len(buckets.pairs):
        return buckets.validate()
    if len(buckets.bucket) or len(buckets.quotas) or len(set(buckets.ids)) != len(buckets.ids):
        raise ValueError('Malformed empty structure')
    return dict(entities=len(buckets.ids), candidate_pairs=0, buckets=0, quota_sum=0, max_entity_quota=0)

def public_groups(ids, neighbors, size, seed, round_id, scheme='text_groups'):
    order = _i(sorted(range(len(ids)), key=lambda v: (public_priority(seed, scheme, round_id, ids[v]), ids[v])))
    labels = np.empty(len(ids), np.int32)
    kernel().groups(len(ids), neighbors.shape[1], size, order, _i(neighbors), labels)
    order = np.argsort(labels, kind='stable')
    offsets = np.r_[0, np.flatnonzero(np.diff(labels[order])) + 1, len(ids)]
    return [order[lo:hi].astype(np.int32) for lo, hi in zip(offsets[:-1], offsets[1:])]

def build_forest(candidates, base_group_size, max_root_size, layer_budgets, seed, capacity=None, preserve_legacy=True):
    ids, n = (candidates.ids, len(candidates.ids))
    kdeg = sum(layer_budgets)
    if kdeg != 4 or base_group_size not in (4, 8) or max_root_size not in (16, 32):
        raise ValueError('Outside configured structure forest family')
    if not n:
        raise ValueError('Empty entity universe')
    capacity = 4 * ((n + 7) // 8) if capacity is None else int(capacity)
    if capacity < 0:
        raise ValueError('Negative capacity')
    embed = preserve_legacy and base_group_size == 8 and (tuple(layer_budgets) == (1, 1, 1, 1))
    grouping_nn = getattr(candidates, 'legacy_neighbors', candidates.neighbors[:, :64]) if embed else candidates.neighbors
    layers = [public_groups(ids, grouping_nn, base_group_size, seed, r) for r in range(len(layer_budgets))]
    if embed:
        clique = np.asarray([edge for groups in layers for group in groups for edge in itertools.combinations(group.tolist(), 2)], dtype=np.int32).reshape(-1, 2)
        candidates = candidates.extend(clique)
    pairs, m = (candidates.pairs, len(candidates.pairs))
    score_order = _i(np.lexsort((pairs[:, 1], pairs[:, 0], -candidates.cosine)))
    left, right, parent, members, layer, roots, budgets, leaf_maps = ([], [], [], [], [], [], [], [])
    for r, groups in enumerate(layers):
        nodes = []
        leaf = np.empty(n, np.int32)
        for group in groups:
            t = len(left)
            left.append(-1)
            right.append(-1)
            parent.append(-1)
            members.append(group)
            layer.append(r)
            nodes.append(t)
            leaf[group] = t
        leaf_maps.append(leaf)
        while len(nodes) > 1:
            vgroup = np.empty(n, np.int32)
            sizes = _i([len(members[t]) for t in nodes])
            for i, t in enumerate(nodes):
                vgroup[members[t]] = i
            mate = np.empty(len(nodes), np.int32)
            merged = kernel().matching(len(nodes), m, max_root_size, pairs, score_order, vgroup, sizes, mate)
            if not merged:
                break
            new = []
            for i, t in enumerate(nodes):
                j = mate[i]
                if j < 0:
                    new.append(t)
                elif i < j:
                    u, p = (nodes[j], len(left))
                    left.append(t)
                    right.append(u)
                    parent.append(-1)
                    parent[t] = parent[u] = p
                    members.append(np.concatenate((members[t], members[u])))
                    layer.append(r)
                    new.append(p)
            nodes = sorted(new)
        roots.extend(nodes)
        budgets.extend([layer_budgets[r]] * len(nodes))
    left, right, parent = (_i(left), _i(right), _i(parent))
    assignment = np.full(m, -1, np.int32)
    if embed:
        for leaf in leaf_maps:
            kernel().assign(m, pairs, leaf, parent, assignment, 1)
    for leaf in leaf_maps:
        kernel().assign(m, pairs, leaf, parent, assignment, 0)
    counts = _i(np.bincount(assignment[assignment >= 0], minlength=len(left)))
    roots, budgets = (_i(roots), _i(budgets))
    sizes = np.array([len(members[t]) for t in roots], np.int64)
    weights = sizes * budgets
    denominator = int(weights.sum())
    caps = capacity * weights // denominator
    remainder = capacity - int(caps.sum())
    order = np.lexsort((roots, -(capacity * weights % denominator)))
    caps[order[:remainder]] += 1
    offsets = np.r_[0, np.cumsum([len(group) for group in members])].astype(np.int64)
    old_feasible = embed
    if embed:
        for root, cap in zip(roots, caps):
            stack, required = ([int(root)], 0)
            while stack:
                t = stack.pop()
                if left[t] < 0:
                    required += int(counts[t] > 0)
                else:
                    stack.extend((int(left[t]), int(right[t])))
            old_feasible = old_feasible and required <= cap
    forest = Forest(ids, pairs, assignment, left, right, parent, _i(np.concatenate(members)), offsets, roots, budgets, _i(caps), counts, _i(layer), seed, kdeg, capacity, embed)
    forest.legacy_embedded = bool(old_feasible)
    return (forest, candidates)

def exact_quota_dp(forest, f):
    f = _d(f)
    if f.shape != (len(forest.left), forest.kdeg + 1):
        raise ValueError('Wrong objective table shape')
    quotas = np.zeros(len(forest.left), np.int32)
    leaves = np.ones(len(forest.left), np.int32)
    for t in range(len(leaves)):
        if forest.left[t] >= 0:
            leaves[t] = leaves[forest.left[t]] + leaves[forest.right[t]]
    caps = _i(np.minimum(forest.capacities, leaves[forest.roots] * forest.budgets))
    value = kernel().quota_dp(len(forest.roots), f.shape[1], forest.roots, forest.budgets, caps, forest.left, forest.right, forest.counts, f, quotas)
    if not np.isfinite(value):
        raise ValueError('DP root exceeds compiled exact signature limit (21 nodes, K<=7)')
    return (quotas, float(f[np.arange(len(quotas)), quotas].sum()))

def forest_marginal_greedy(forest, f):
    import heapq
    quotas = np.zeros(len(forest.left), np.int32)
    charge = np.zeros(len(forest.ids), np.int32)
    root_of, root_budget = ({}, {})
    for root, k in zip(forest.roots, forest.budgets):
        stack = [int(root)]
        while stack:
            t = stack.pop()
            root_of[t] = int(root)
            root_budget[t] = int(k)
            if forest.left[t] >= 0:
                stack.extend((int(forest.left[t]), int(forest.right[t])))
    used = {int(root): 0 for root in forest.roots}
    caps = dict(zip(map(int, forest.roots), map(int, forest.capacities)))
    layer_charge = np.zeros((int(forest.layer.max()) + 1, len(charge)), np.int32)
    heap = [(-f[t, 1] / len(forest.entities(t)), t, 1) for t in range(len(quotas)) if forest.counts[t]]
    heapq.heapify(heap)
    while heap:
        neg, t, q = heapq.heappop(heap)
        if -neg <= 0:
            continue
        vertices, root, r = (forest.entities(t), root_of[t], forest.layer[t])
        if used[root] >= caps[root] or np.any(layer_charge[r, vertices] >= root_budget[t]):
            continue
        quotas[t] += 1
        used[root] += 1
        charge[vertices] += 1
        layer_charge[r, vertices] += 1
        if q < min(forest.counts[t], root_budget[t]):
            heapq.heappush(heap, (-(f[t, q + 1] - f[t, q]) / len(vertices), t, q + 1))
    return (quotas, float(f[np.arange(len(quotas)), quotas].sum()))

def legacy_text_groups(candidates, config, seed):
    cfg = dict(config, seed=int(seed), neighbors=64)
    return construct_buckets(candidates.ids, candidates.neighbors[:, :64], candidates.scores[:, :64], cfg, 'text_groups')

def random_public_groups(ids, seed, kdeg=4, group_size=8, rounds=4):
    nn = np.empty((len(ids), 0), np.int32)
    pairs, buckets, number = ([], [], 0)
    for r in range(rounds):
        for group in public_groups(ids, nn, group_size, seed, r, 'random_public_groups'):
            p = list(itertools.combinations(group.tolist(), 2))
            pairs.extend(p)
            buckets.extend([number] * len(p))
            number += 1
    pp = _i(pairs).reshape(-1, 2)
    keys = pair_keys(pp, len(ids))
    _, first = np.unique(keys, return_index=True)
    bb = _i(buckets)[first]
    _, bb = np.unique(bb, return_inverse=True)
    result = PublicBuckets(tuple(ids), pp[first], _i(bb), np.ones(len(np.unique(bb)), np.int32), int(seed), kdeg, 'random_public_groups')
    validate_buckets(result)
    return result

def candidate_bmatching(candidates, probabilities, seed, kdeg=4, capacity=None):
    probabilities = _d(probabilities)
    if len(probabilities) != len(candidates.pairs):
        raise ValueError('Wrong weight count')
    pp, n = (candidates.pairs, len(candidates.ids))
    order = _i(np.lexsort((pp[:, 1], pp[:, 0], -probabilities)))
    accepted = np.zeros(len(pp), np.uint8)
    cap = kdeg * n // 2 if capacity is None else int(capacity)
    kernel().bmatch(len(pp), n, kdeg, cap, pp, order, accepted)
    pp = pp[accepted.astype(bool)]
    result = PublicBuckets(candidates.ids, pp, np.arange(len(pp), dtype=np.int32), np.ones(len(pp), np.int32), int(seed), kdeg, 'candidate_graph_bmatching')
    validate_buckets(result)
    return result

class FixedTruncation:

    def __init__(self, buckets, relations):
        validate_buckets(buckets)
        self.buckets = buckets
        self.relations = _i(relations).reshape(-1, 2)
        n = len(buckets.ids)
        keys = pair_keys(self.relations, n)
        if len(np.unique(keys)) != len(keys):
            raise ValueError('Duplicate private relation')
        public_keys = pair_keys(buckets.pairs, n)
        order = np.argsort(public_keys)
        hit = membership(keys, public_keys[order])
        rows = np.flatnonzero(hit)
        bid = buckets.bucket[order[np.searchsorted(public_keys[order], keys[hit])]]
        ranked = sorted(zip(bid.tolist(), rows.tolist()), key=lambda br: (br[0], edge_priority(buckets.seed, *(buckets.ids[v] for v in self.relations[br[1]])), tuple(sorted((buckets.ids[v] for v in self.relations[br[1]])))))
        self.rows = np.array([row for _, row in ranked], np.int64)
        self.bid = _i([bucket for bucket, _ in ranked])
        sizes = np.bincount(self.bid, minlength=len(buckets.quotas))
        self.offsets = np.r_[0, np.cumsum(sizes)]
        self.keep = np.sort(np.concatenate([self.bucket_rows(b)[:q] for b, q in enumerate(buckets.quotas)]) if len(buckets.quotas) else np.array([], np.int64))
        self.raw_bucket = np.full(len(keys), -1, np.int32)
        self.raw_bucket[self.rows] = self.bid
        self.generated_keys = keys

    def bucket_rows(self, b):
        return self.rows[self.offsets[b]:self.offsets[b + 1]]

    def delete(self, entity, affected):
        removed, added = ([], [])
        for b in affected:
            rows = self.bucket_rows(b)
            q = self.buckets.quotas[b]
            before = rows[:q]
            after = rows[np.all(self.relations[rows] != entity, axis=1)][:q]
            removed.extend(np.setdiff1d(before, after, assume_unique=True))
            added.extend(np.setdiff1d(after, before, assume_unique=True))
        return (np.asarray(removed, np.int64), np.asarray(added, np.int64))

def evaluate_structure(buckets, relations, generated_keys, probabilities, indexed=None, raw_knn_keys=None):
    indexed = indexed or FixedTruncation(buckets, relations)
    keys, n = (indexed.generated_keys, len(buckets.ids))
    active_keys = np.sort(pair_keys(buckets.pairs, n))
    in_active = membership(keys, active_keys)
    in_knn = membership(keys, generated_keys)
    in_generated = in_knn | in_active
    kept = indexed.relations[indexed.keep]
    total, actual, cap = (len(keys), len(kept), int(buckets.quotas.sum()))
    covered = len(np.unique(kept))
    nonisolated = len(np.unique(relations))
    ratio = lambda a, b: a / b if b else None
    losses = [int((~in_generated).sum()), int((in_generated & ~in_active).sum()), int(in_active.sum()) - actual]
    if sum(losses) != total - actual:
        raise AssertionError('Three-stage loss identity failed')
    f = bucket_objectives(probabilities, buckets.bucket, len(buckets.quotas), buckets.kdeg)
    sizes = np.bincount(buckets.bucket, minlength=len(buckets.quotas))
    hist = lambda arr: {str(int(k)): int(v) for k, v in zip(*np.unique(arr, return_counts=True))}
    raw_knn_keys = generated_keys if raw_knn_keys is None else raw_knn_keys
    return dict(retention=ratio(actual, total), nonisolated_coverage=ratio(covered, nonisolated), all_entity_coverage=ratio(covered, n), capacity_B=cap, capacity_utilization=ratio(actual, cap), generated_candidate_coverage=ratio(int(in_generated.sum()), total), active_candidate_coverage=ratio(int(in_active.sum()), total), loss_candidate_generation=ratio(losses[0], total), loss_structure_assignment=ratio(losses[1], total), loss_quota=ratio(losses[2], total), loss_counts=losses, identity_verified=True, proxy_expected_retained=float(f[np.arange(len(f)), buckets.quotas].sum()), actual_retained=actual, raw_relations=total, entities=n, candidate_pairs=len(np.union1d(generated_keys, active_keys)), active_pairs=len(active_keys), structure_pairs_outside_knn=int((~membership(active_keys, raw_knn_keys)).sum()), generated_pairs_outside_knn=int((~membership(generated_keys, raw_knn_keys)).sum()), raw_knn_candidate_coverage=ratio(int(membership(keys, raw_knn_keys).sum()), total), bucket_size_and_quota_histograms=dict(size=hist(sizes), quota=hist(buckets.quotas)))

def stability_check(buckets, relations, deletion_counts, seed, candidate_graph=False):
    errors = []
    try:
        validate_buckets(buckets)
    except ValueError as error:
        return dict(status='invalid_input', violations=[str(error)], samples=0)
    indexed = FixedTruncation(buckets, relations)
    n = len(buckets.ids)
    rawdeg = np.bincount(indexed.relations.ravel(), minlength=n)
    retdeg = np.bincount(indexed.relations[indexed.keep].ravel(), minlength=n)
    incidence = np.unique(np.concatenate((buckets.bucket.astype(np.int64) * n + buckets.pairs[:, 0], buckets.bucket.astype(np.int64) * n + buckets.pairs[:, 1])))
    vertex_order = np.argsort(incidence % n, kind='stable')
    incidence = incidence[vertex_order]
    offsets = np.r_[0, np.cumsum(np.bincount(incidence % n, minlength=n))]
    occupancy = np.diff(indexed.offsets)
    sat_buckets = occupancy >= buckets.quotas
    saturated = np.zeros(n, bool)
    for b in np.flatnonzero(sat_buckets):
        lost = indexed.bucket_rows(b)[buckets.quotas[b]:]
        saturated[indexed.relations[lost].ravel()] = True
    multibucket = np.diff(offsets) >= 2
    rng, used, strata, samples = (np.random.default_rng(seed), set(), {}, [])
    uniform = rng.permutation(n)
    pools = dict(uniform=uniform, high_raw_degree=np.lexsort((np.arange(n), -rawdeg)), high_retained_degree=np.lexsort((np.arange(n), -retdeg)), saturated_bucket=uniform[saturated[uniform]], multi_bucket=uniform[multibucket[uniform]])
    for name, count in deletion_counts.items():
        chosen = []
        for pool in (pools[name], uniform):
            for u in pool:
                u = int(u)
                if len(chosen) >= count:
                    break
                if u not in used:
                    used.add(u)
                    chosen.append(u)
            if len(chosen) >= count:
                break
        strata[name] = chosen
        samples.extend(chosen)
    values = []
    for u in samples:
        affected = incidence[offsets[u]:offsets[u + 1]] // n
        removed, added = indexed.delete(u, affected)
        dm, dp = (len(removed), len(added))
        incident = bool(np.all(np.any(indexed.relations[removed] == u, axis=1)))
        if dm > buckets.kdeg or dp > dm or (not incident) or (candidate_graph and dp != 0):
            errors.append(dict(entity=buckets.ids[u], d_minus=dm, d_plus=dp, removed_incident=incident))
        values.append((dm, dp, dm + dp))
    array = np.array(values).reshape(-1, 3)
    summary = {name: dict(mean=float(array[:, j].mean()) if len(array) else None, p99=float(np.percentile(array[:, j], 99)) if len(array) else None, sampled_max=int(array[:, j].max()) if len(array) else None) for j, name in enumerate(('d_minus', 'd_plus', 'symmetric_difference'))}
    return dict(status='invalid_input' if errors else 'valid', violations=errors, samples=len(samples), strata={name: [buckets.ids[u] for u in chosen] for name, chosen in strata.items()}, isolated_samples=sum((rawdeg[u] == 0 for u in samples)), statistics=summary)

def adversarial_checks(seed=101, k=4):
    filler = [(i, 12 + (k - 1) * i + j) for i in range(12) for j in range(k - 1)]
    path = np.array(filler + [(i, i + 1) for i in range(11)], np.int32)

    def greedy(edges):
        degree, out = (np.zeros(12 * k, int), set())
        for a, b in edges:
            if degree[a] < k and degree[b] < k:
                degree[a] += 1
                degree[b] += 1
                out.add((int(a), int(b)))
        return out
    before, after = (greedy(path), greedy(path[np.all(path != 0, axis=1)]))
    cascade = any((0 not in edge for edge in before - after))
    leaves = list(range(k, 3 * k + 1))
    star = [(0, v) for v in leaves] + [(j, v) for j in range(1, k) for v in leaves]
    star += [(v, 3 * k + 1 + i) for i, v in enumerate(leaves)]

    def topk(edges):
        choices = {v: set() for v in range(5 * k + 2)}
        for a, b in edges:
            choices[a].add(b)
            choices[b].add(a)
        choices = {v: set(sorted(vv)[:k]) for v, vv in choices.items()}
        return {edge for edge in edges if edge[1] in choices[edge[0]] and edge[0] in choices[edge[1]]}
    sb, sa = (topk(star), topk([edge for edge in star if 0 not in edge]))
    pp = np.array([(0, 1 + 3 * i) for i in range(k)] + [(2 + 3 * i, 3 + 3 * i) for i in range(k)], np.int32)
    ids = tuple((f'{i:03d}' for i in range(3 * k + 1)))
    witness_seed = next((s for s in range(100000) if all((edge_priority(s, ids[0], ids[1 + 3 * i]) < edge_priority(s, ids[2 + 3 * i], ids[3 + 3 * i]) for i in range(k)))))
    buckets = PublicBuckets(ids, pp, _i(list(range(k)) * 2), np.ones(k, np.int32), witness_seed, k, '2K_witness')
    index = FixedTruncation(buckets, pp)
    removed, added = index.delete(0, np.arange(k))
    return dict(status='valid' if cascade and len(sa - sb) > len(sb - sa) and (len(removed) == len(added) == k) else 'invalid_input', greedy_path_cascade=dict(detected_nonincident_removal=cascade, d_minus=len(before - after), d_plus=len(after - before)), two_sided_topk_star=dict(d_minus=len(sb - sa), d_plus=len(sa - sb), detected_d_plus_exceeds_d_minus=len(sa - sb) > len(sb - sa)), bucket_bound_attaining_2K=dict(d_minus=len(removed), d_plus=len(added), public_seed=witness_seed))
