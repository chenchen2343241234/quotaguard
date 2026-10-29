import hashlib
import itertools
import json
from dataclasses import dataclass
import numpy as np

def public_priority(seed, *identifiers):
    if type(seed) is not int:
        raise ValueError('A fixed public integer seed is required')
    return hashlib.sha256(json.dumps([seed, *identifiers], ensure_ascii=True, separators=(',', ':')).encode()).digest()

def edge_priority(seed, a, b):
    return public_priority(seed, *sorted((a, b)))

@dataclass
class PublicBuckets:
    ids: tuple
    pairs: np.ndarray
    bucket: np.ndarray
    quotas: np.ndarray
    seed: int
    kdeg: int
    scheme: str

    def validate(self):
        n = len(self.ids)
        if len(set(self.ids)) != n or type(self.seed) is not int or type(self.kdeg) is not int or (self.kdeg < 1) or (self.pairs.shape != (len(self.bucket), 2)) or (not np.issubdtype(self.pairs.dtype, np.integer)) or (not np.issubdtype(self.bucket.dtype, np.integer)) or (not np.issubdtype(self.quotas.dtype, np.integer)) or np.any(self.quotas < 1):
            raise ValueError('Invalid public bucket structure')
        if len(self.pairs) == 0:
            raise ValueError('Empty public candidate universe')
        a, b = self.pairs.T
        if np.any(a < 0) or np.any(b >= n) or np.any(a >= b) or np.any(self.bucket < 0) or np.any(self.bucket >= len(self.quotas)):
            raise ValueError('Invalid candidate pair/bucket')
        keys = a.astype(np.int64) * n + b
        if len(np.unique(keys)) != len(keys):
            raise ValueError('Buckets must be pair-disjoint')
        if len(np.unique(self.bucket)) != len(self.quotas):
            raise ValueError('Do not serialize structurally empty buckets')
        incidence = np.unique(np.concatenate([self.bucket.astype(np.int64) * n + a, self.bucket.astype(np.int64) * n + b]))
        charge = np.bincount(incidence % n, weights=self.quotas[incidence // n], minlength=n)
        if charge.max() > self.kdeg:
            raise ValueError('Per-entity bucket quota sum exceeds K')
        return dict(entities=n, candidate_pairs=len(keys), buckets=len(self.quotas), quota_sum=int(self.quotas.sum()), max_entity_quota=int(charge.max()), entity_quota_histogram={str(i): int((charge == i).sum()) for i in range(self.kdeg + 1)})

    def truncate(self, relations):
        relations = np.asarray(relations, dtype=np.int64).reshape(-1, 2)
        n = len(self.ids)
        if np.any(relations < 0) or np.any(relations >= n):
            raise ValueError('Unknown private endpoint')
        canonical = np.sort(relations, axis=1)
        if np.any(canonical[:, 0] == canonical[:, 1]):
            raise ValueError('Self-loop')
        raw_keys = canonical[:, 0] * n + canonical[:, 1]
        if len(np.unique(raw_keys)) != len(raw_keys):
            raise ValueError('Duplicate private relation')
        keys = self.pairs[:, 0].astype(np.int64) * n + self.pairs[:, 1]
        order = np.argsort(keys)
        pos = np.searchsorted(keys[order], raw_keys)
        hits = (pos < len(keys)) & (keys[order[np.minimum(pos, len(keys) - 1)]] == raw_keys)
        rows = np.flatnonzero(hits)
        buckets = self.bucket[order[pos[hits]]]
        ranked = sorted(zip(buckets.tolist(), rows.tolist()), key=lambda item: (item[0], edge_priority(self.seed, *(self.ids[v] for v in canonical[item[1]])), tuple((self.ids[v] for v in canonical[item[1]]))))
        count, keep = ({}, [])
        for bucket, row in ranked:
            if count.get(bucket, 0) < self.quotas[bucket]:
                keep.append(row)
                count[bucket] = count.get(bucket, 0) + 1
        return np.array(sorted(keep), dtype=np.int64)

    def save(self, path):
        self.validate()
        np.savez_compressed(path, ids=np.array(self.ids), pairs=self.pairs, bucket=self.bucket, quotas=self.quotas, seed=self.seed, kdeg=self.kdeg, scheme=self.scheme)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as p:
            result = cls(tuple(p['ids'].tolist()), p['pairs'], p['bucket'], p['quotas'], int(p['seed']), int(p['kdeg']), str(p['scheme']))
        result.validate()
        return result

def title_neighbors(ids, titles, config, progress=lambda message: None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    if tuple(ids) != tuple(sorted(ids)):
        raise ValueError('Public entities must be ordered by immutable ID')
    options = dict(config['tfidf'])
    options['ngram_range'] = tuple(options['ngram_range'])
    matrix = TfidfVectorizer(dtype=np.float32, **options).fit_transform(titles)
    n, width = (len(ids), config['neighbors'])
    neighbors = np.full((n, width), -1, dtype=np.int32)
    scores = np.zeros((n, width), dtype=np.float32)
    block = config['similarity_block_size']
    transposed = matrix.T.tocsc()
    for start in range(0, n, block):
        sims = (matrix[start:start + block] @ transposed).tocsr()
        for offset in range(sims.shape[0]):
            lo, hi = sims.indptr[offset:offset + 2]
            idx, val = (sims.indices[lo:hi], sims.data[lo:hi])
            valid = (idx != start + offset) & (val > 0)
            idx, val = (idx[valid], val[valid])
            if len(idx) > width:
                threshold = np.partition(val, -width)[-width]
                valid = val >= threshold
                idx, val = (idx[valid], val[valid])
            order = np.lexsort((idx, -val))[:width]
            neighbors[start + offset, :len(order)] = idx[order]
            scores[start + offset, :len(order)] = val[order]
        if start % (block * 100) == 0:
            progress(f'public title neighbors {start}/{n}')
    return (neighbors, scores)

def construct_buckets(ids, neighbors, scores, config, scheme):
    ids = tuple(ids)
    n, seed, kdeg = (len(ids), config['seed'], config['kdeg'])
    pairs, membership, seen = ([], [], set())
    nbuckets = 0

    def add_bucket(candidates):
        nonlocal nbuckets
        fresh = []
        for u, v in candidates:
            a, b = sorted((int(u), int(v)))
            key = a * n + b
            if a != b and key not in seen:
                seen.add(key)
                fresh.append((a, b))
        if fresh:
            pairs.extend(fresh)
            membership.extend([nbuckets] * len(fresh))
            nbuckets += 1
        return fresh

    def public_order(round_id):
        return sorted(range(n), key=lambda v: (public_priority(seed, scheme, round_id, ids[v]), ids[v]))
    if scheme == 'singleton_knn':
        source = np.repeat(np.arange(n, dtype=np.int32), neighbors.shape[1])
        target, weight = (neighbors.ravel(), scores.ravel())
        valid = target >= 0
        a, b = (np.minimum(source[valid], target[valid]), np.maximum(source[valid], target[valid]))
        order = np.lexsort((b, a, -weight[valid]))
        degree = np.zeros(n, dtype=np.int32)
        for i in order:
            u, v = (int(a[i]), int(b[i]))
            if degree[u] < kdeg and degree[v] < kdeg and add_bucket([(u, v)]):
                degree[u] += 1
                degree[v] += 1
    elif scheme == 'out_star':
        incoming = np.zeros(n, dtype=np.int32)
        for u in public_order(0):
            candidates = []
            for v in neighbors[u]:
                v = int(v)
                if v < 0:
                    continue
                if incoming[v] < kdeg - 1 and min(u, v) * n + max(u, v) not in seen:
                    candidates.append((u, v))
                    if len(candidates) == config['star_candidates']:
                        break
            add_bucket(candidates)
            for _, v in candidates:
                incoming[v] += 1
    elif scheme == 'text_groups':
        if config['rounds'] > kdeg or config['group_size'] < 2:
            raise ValueError('Invalid public partition budget')
        for round_id in range(config['rounds']):
            order = public_order(round_id)
            unused = np.ones(n, dtype=bool)
            fallback = 0
            for u in order:
                if not unused[u]:
                    continue
                group = [u]
                unused[u] = False
                for v in neighbors[u]:
                    if v >= 0 and unused[v]:
                        group.append(int(v))
                        unused[v] = False
                        if len(group) == config['group_size']:
                            break
                while len(group) < config['group_size'] and fallback < n:
                    v = order[fallback]
                    fallback += 1
                    if unused[v]:
                        unused[v] = False
                        group.append(v)
                add_bucket(itertools.combinations(group, 2))
    else:
        raise ValueError('Unknown public bucket scheme')
    result = PublicBuckets(ids, np.asarray(pairs, dtype=np.int32).reshape(-1, 2), np.asarray(membership, dtype=np.int32), np.ones(nbuckets, dtype=np.int32), seed, kdeg, scheme)
    result.validate()
    return result
