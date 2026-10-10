"""Pseudo-relevance feedback with the Rocchio algorithm (Manning et al., Sec. 9.1.1).

q_m = alpha * q_0 + beta * (1/|D_r|) * sum_{d in D_r} d      (no negative-feedback term)
D_r is simply the top-k documents of the first-pass ranking (blind feedback, Sec. 9.1.6).
"""
import math
from collections import defaultdict


class RocchioPRF:
    def __init__(self, index, alpha=1.0, beta=0.75, fb_docs=5, fb_terms=20):
        self.index = index
        self.alpha, self.beta = alpha, beta
        self.fb_docs, self.fb_terms = fb_docs, fb_terms

    def _doc_vector(self, d):
        tf = self.index.forward[d]
        vec = {t: (1 + math.log10(f)) * self.index.idf(t) for t, f in tf.items()}
        n = math.sqrt(sum(w * w for w in vec.values())) or 1.0
        return {t: w / n for t, w in vec.items()}

    def expand(self, qtf, top_docs):
        ix = self.index
        q0 = {t: (1 + math.log10(f)) * ix.idf(t) for t, f in qtf.items()}
        n = math.sqrt(sum(w * w for w in q0.values())) or 1.0
        q0 = {t: w / n for t, w in q0.items()}

        centroid = defaultdict(float)
        for d in top_docs:
            for t, w in self._doc_vector(d).items():
                centroid[t] += w / max(len(top_docs), 1)
        # keep only the strongest expansion terms (query-drift control)
        best = dict(sorted(centroid.items(), key=lambda x: -x[1])[: self.fb_terms])

        qm = defaultdict(float)
        for t, w in q0.items():
            qm[t] += self.alpha * w
        for t, w in best.items():
            qm[t] += self.beta * w
        return dict(qm)
