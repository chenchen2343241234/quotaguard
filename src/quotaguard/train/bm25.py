from collections import defaultdict
import math
import numpy as np
import torch

def bm25_parameters(config):
    topn, params = (config.get('bm25_topn', 100), config.get('bm25', {}))
    if type(topn) is not int or topn < 1:
        raise ValueError('bm25_topn must be a positive integer')
    if not isinstance(params, dict) or set(params) - {'k1', 'b'}:
        raise ValueError('bm25 accepts only k1 and b')
    params = {'k1': 1.2, 'b': 0.75, **params}
    if any((type(v) not in (int, float) or not math.isfinite(v) for v in params.values())) or params['k1'] <= 0 or (not 0 <= params['b'] <= 1):
        raise ValueError('bm25 requires finite k1 > 0 and 0 <= b <= 1')
    return (topn, params)

class BM25TopN:

    def __init__(self, titles, *, topn=100, bm25=None):
        from rank_bm25 import BM25Okapi
        self.topn, params = bm25_parameters({'bm25_topn': topn, 'bm25': {} if bm25 is None else bm25})
        tokens = [str(title).lower().split() for title in titles]
        if len(tokens) < 3 or not any(tokens):
            raise ValueError('BM25 needs >=3 entities and nonempty title tokens')
        index = BM25Okapi(tokens, **params, epsilon=0.25)
        postings = defaultdict(lambda: ([], []))
        for di, frequencies in enumerate(index.doc_freqs):
            for term, frequency in frequencies.items():
                docs, counts = postings[term]
                docs.append(di)
                counts.append(frequency)
        postings = {q: (np.asarray(ids, dtype=np.intp), np.asarray(tf, dtype=np.int_)) for q, (ids, tf) in postings.items()}
        lengths = np.asarray(index.doc_len)
        self.n_entities = len(tokens)
        keep = min(topn + 1, self.n_entities - 1)
        ranking = np.empty((self.n_entities, keep), dtype=np.int64)
        for anchor, query in enumerate(tokens):
            support = np.unique(np.concatenate([postings[q][0] for q in query])) if query else np.empty(0, dtype=np.intp)
            prefix = np.arange(min(self.n_entities, len(support) + keep + 1))
            missing = np.setdiff1d(prefix, support, assume_unique=True)[:keep + 1]
            candidates = np.union1d(support, missing)
            doc_len = lengths[candidates]
            scores = np.zeros(len(candidates), dtype=np.float64)
            if params['b'] == 1 and (doc_len == 0).any():
                scores = np.asarray(index.get_batch_scores(query, candidates.tolist()))
            else:
                for q in query:
                    docs, q_freq = postings[q]
                    positions = np.searchsorted(candidates, docs)
                    scores[positions] += (index.idf.get(q) or 0) * (q_freq * (index.k1 + 1) / (q_freq + index.k1 * (1 - index.b + index.b * doc_len[positions] / index.avgdl)))
            if not np.isfinite(scores).all():
                raise ValueError('BM25 produced nonfinite scores')
            eligible = candidates != anchor
            candidates, scores = (candidates[eligible], scores[eligible])
            cutoff = np.partition(scores, len(scores) - keep)[len(scores) - keep]
            winners = np.flatnonzero(scores >= cutoff)
            order = np.lexsort((candidates[winners], -scores[winners]))[:keep]
            ranking[anchor] = candidates[winners[order]]
        self.ranking = torch.from_numpy(ranking)

    def candidates(self, anchor, positive):
        row = self.ranking[anchor]
        return row[row != positive][:self.topn]
