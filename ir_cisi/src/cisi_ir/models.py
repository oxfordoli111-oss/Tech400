"""Ranked retrieval models (Manning et al.): VSM tf-idf (Ch. 6), BM25 (Ch. 11), query LM (Ch. 12).

Every model scores documents term-at-a-time (Sec. 7.1.5): it walks the postings list of each query
term and adds that term's contribution to a per-document accumulator.
A query is a dict {term: weight}; this lets pseudo-relevance feedback (Ch. 9) reuse all models.
"""
import heapq
import math
from collections import Counter


class RetrievalModel:
    name = "base"

    def __init__(self, index):
        self.index = index

    def score(self, qweights):  # -> {doc_id: score}
        raise NotImplementedError


class TfIdfCosine(RetrievalModel):
    """SMART lnc.ltc: log tf, no idf, cosine norm for documents; log tf x idf for queries."""
    name = "tfidf"

    def __init__(self, index):
        super().__init__(index)
        self.norm = {}
        for d, tf in index.forward.items():
            self.norm[d] = math.sqrt(sum((1 + math.log10(f)) ** 2 for f in tf.values())) or 1.0

    def query_vector(self, qtf):
        vec = {t: (1 + math.log10(f)) * self.index.idf(t) for t, f in qtf.items() if self.index.df(t)}
        n = math.sqrt(sum(w * w for w in vec.values())) or 1.0
        return {t: w / n for t, w in vec.items()}

    def score(self, qweights):
        acc = {}
        for t, wq in qweights.items():
            for d, f in self.index.postings.get(t, ()):
                acc[d] = acc.get(d, 0.0) + wq * (1 + math.log10(f)) / self.norm[d]
        return acc


class BM25(RetrievalModel):
    """Okapi BM25 (Sec. 11.4.3) with the non-negative idf variant log(1 + (N-df+.5)/(df+.5))."""
    name = "bm25"

    def __init__(self, index, k1=1.2, b=0.75):
        super().__init__(index)
        self.k1, self.b = k1, b

    def score(self, qweights):
        ix, k1, b = self.index, self.k1, self.b
        acc = {}
        for t, wq in qweights.items():
            df = ix.df(t)
            if not df:
                continue
            idf = math.log(1 + (ix.N - df + 0.5) / (df + 0.5))
            for d, f in ix.postings[t]:
                norm = k1 * (1 - b + b * ix.doc_len[d] / ix.avgdl)
                acc[d] = acc.get(d, 0.0) + wq * idf * f * (k1 + 1) / (f + norm)
        return acc


class QueryLikelihoodJM(RetrievalModel):
    """Unigram query likelihood with Jelinek-Mercer smoothing (Sec. 12.2.2).
    Uses the rank-equivalent form: sum_t log(1 + (1-l)/l * (tf/dl) / (cf/T)); terms absent from a
    document add only a document-independent constant, so only postings are visited."""
    name = "lm_jm"

    def __init__(self, index, lam=0.7):
        super().__init__(index)
        self.lam = lam

    def score(self, qweights):
        ix, lam = self.index, self.lam
        acc = {}
        for t, wq in qweights.items():
            if not ix.df(t):
                continue
            pc = ix.cf[t] / ix.total_tokens
            for d, f in ix.postings[t]:
                acc[d] = acc.get(d, 0.0) + wq * math.log(1 + ((1 - lam) / lam) * (f / ix.doc_len[d]) / pc)
        return acc


class SearchEngine:
    """Query processing: analyse query -> weight terms -> score via model -> top-k with a heap."""

    def __init__(self, index, model, feedback=None):
        self.index, self.model, self.feedback = index, model, feedback

    def analyse(self, text):
        return self.index.pre(text)

    def search(self, text, k=10):
        qtf = Counter(t for t in self.analyse(text) if self.index.df(t))
        if not qtf:
            return []
        qweights = dict(qtf)
        scores = self.model.score(qweights)
        if self.feedback is not None:
            top = [d for d, _ in self._top(scores, self.feedback.fb_docs)]
            qweights = self.feedback.expand(qtf, top)
            scores = self.model.score(qweights)
        return self._top(scores, k)

    @staticmethod
    def _top(scores, k):
        return heapq.nsmallest(k, scores.items(), key=lambda x: (-x[1], x[0]))
