"""Inverted index (Manning et al., Ch. 1 & 4): term -> postings list [(docID, tf), ...]."""
import math
import pickle
from collections import Counter, defaultdict


class InvertedIndex:
    def __init__(self, preprocessor):
        self.pre = preprocessor
        self.postings = defaultdict(list)   # term -> [(doc_id, tf)] sorted by doc_id
        self.doc_len = {}                   # doc_id -> number of indexed tokens
        self.forward = {}                   # doc_id -> Counter(term -> tf)  (used for feedback)
        self.cf = Counter()                 # collection frequency of each term
        self.N = 0
        self.avgdl = 0.0
        self.total_tokens = 0

    # ---- construction ---------------------------------------------------
    def build(self, docs):
        """One pass over the corpus; docs processed in increasing docID order so
        postings stay sorted without a separate sort step (blocked sort-based idea, Sec. 4.2)."""
        for doc_id in sorted(docs):
            tokens = self.pre(docs[doc_id]["text"])
            tf = Counter(tokens)
            self.forward[doc_id] = tf
            self.doc_len[doc_id] = len(tokens)
            for term, f in tf.items():
                self.postings[term].append((doc_id, f))
                self.cf[term] += f
        self.N = len(self.doc_len)
        self.total_tokens = sum(self.doc_len.values())
        self.avgdl = self.total_tokens / self.N
        return self

    # ---- statistics -----------------------------------------------------
    def df(self, term):
        return len(self.postings.get(term, ()))

    def idf(self, term):
        d = self.df(term)
        return math.log10(self.N / d) if d else 0.0

    @property
    def vocab_size(self):
        return len(self.postings)

    def stats(self):
        return {
            "documents": self.N,
            "vocabulary (terms)": self.vocab_size,
            "total tokens": self.total_tokens,
            "avg doc length": round(self.avgdl, 1),
            "total postings": sum(len(p) for p in self.postings.values()),
        }

    # ---- Boolean retrieval (Sec. 1.3): postings intersection / union ----------
    def boolean_and(self, terms):
        lists = sorted((self.postings.get(t, []) for t in terms), key=len)
        if not lists or not lists[0]:
            return []
        result = [d for d, _ in lists[0]]
        for plist in lists[1:]:
            ids, i, j, out = [d for d, _ in plist], 0, 0, []
            while i < len(result) and j < len(ids):
                if result[i] == ids[j]:
                    out.append(result[i]); i += 1; j += 1
                elif result[i] < ids[j]:
                    i += 1
                else:
                    j += 1
            result = out
        return result

    def boolean_or(self, terms):
        return sorted({d for t in terms for d, _ in self.postings.get(t, [])})

    # ---- persistence ----------------------------------------------------
    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
