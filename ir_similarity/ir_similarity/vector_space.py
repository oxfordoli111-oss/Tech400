"""
vector_space.py
----------------
Implements the vector space model from Chapter 6 of "An Introduction to
Information Retrieval" (Manning, Raghavan & Schutze) to compute
similarity scores BETWEEN DOCUMENTS (rather than between a query and a
document, which is the more commonly shown case).

Weighting scheme used (SMART notation "ltc", applied to every document):

    l - logarithmic term frequency:      wf(t,d) = 1 + log(tf(t,d))
    t - inverse document frequency:      idf(t)  = log(N / df(t))
    c - cosine normalization:            divide each document's weight
                                          vector by its Euclidean length

    weight(t,d) = wf(t,d) * idf(t)

Using log-weighted tf (Section 6.4.1) rather than raw tf reflects the
textbook's observation that twenty occurrences of a term don't carry
twenty times the significance of one occurrence. Because every document
vector is normalized to unit length (cosine normalization, Section
6.3.1), the cosine similarity between two documents is simply the dot
product of their normalized weight vectors -- this is what
`cosine_similarity` computes.
"""

import math
from collections import Counter

from preprocessing import preprocess


class VectorSpaceModel:
    def __init__(self, documents, use_stemming=True):
        """
        documents: dict mapping doc_id -> raw text
        """
        self.raw_documents = documents
        self.use_stemming = use_stemming
        self.doc_ids = list(documents.keys())

        self._term_freqs = {}      # doc_id -> Counter(term -> raw tf)
        self._df = Counter()       # term -> number of documents containing it
        self._idf = {}             # term -> idf value
        self._vectors = {}         # doc_id -> {term: normalized weight}

        self._build()

    # ---------- construction ----------

    def _build(self):
        n = len(self.doc_ids)

        # 1. tokenize/preprocess every document and count raw term frequencies
        for doc_id, text in self.raw_documents.items():
            terms = preprocess(text, use_stemming=self.use_stemming)
            self._term_freqs[doc_id] = Counter(terms)

        # 2. document frequency: how many documents contain each term
        for doc_id, counts in self._term_freqs.items():
            for term in counts:
                self._df[term] += 1

        # 3. idf(t) = log(N / df(t))
        for term, df in self._df.items():
            self._idf[term] = math.log(n / df)

        # 4. raw ltc weights, then cosine-normalize each document vector
        for doc_id, counts in self._term_freqs.items():
            weights = {}
            for term, tf in counts.items():
                wf = 1 + math.log(tf) if tf > 0 else 0.0
                weights[term] = wf * self._idf[term]

            norm = math.sqrt(sum(w * w for w in weights.values()))
            if norm > 0:
                weights = {t: w / norm for t, w in weights.items()}
            self._vectors[doc_id] = weights

    # ---------- public API ----------

    def vector(self, doc_id):
        """Return the normalized tf-idf weight vector (as a dict) for a document."""
        return self._vectors[doc_id]

    def idf(self, term):
        term = term.lower()
        return self._idf.get(term, 0.0)

    def top_terms(self, doc_id, n=5):
        """The n terms with the highest tf-idf weight in a document --
        i.e. the terms that most distinguish it from the rest of the
        collection."""
        vec = self._vectors[doc_id]
        return sorted(vec.items(), key=lambda kv: kv[1], reverse=True)[:n]

    def cosine_similarity(self, doc_id1, doc_id2):
        """Cosine similarity between two documents. Because both vectors
        are already unit-length (cosine normalized), this is just their
        dot product -- Equation 6.11 in the textbook."""
        v1 = self._vectors[doc_id1]
        v2 = self._vectors[doc_id2]
        # iterate over the smaller vector for efficiency
        if len(v1) > len(v2):
            v1, v2 = v2, v1
        return sum(weight * v2.get(term, 0.0) for term, weight in v1.items())

    def similarity_matrix(self):
        """Return the full pairwise cosine similarity matrix as a nested
        dict: matrix[doc_a][doc_b] -> similarity score."""
        matrix = {a: {} for a in self.doc_ids}
        for i, a in enumerate(self.doc_ids):
            matrix[a][a] = 1.0
            for b in self.doc_ids[i + 1:]:
                sim = self.cosine_similarity(a, b)
                matrix[a][b] = sim
                matrix[b][a] = sim
        return matrix

    def most_similar(self, doc_id, top_n=3):
        """Rank every other document by similarity to `doc_id` (the
        'more like this' feature described in Section 6.3.1)."""
        scores = [
            (other, self.cosine_similarity(doc_id, other))
            for other in self.doc_ids
            if other != doc_id
        ]
        scores.sort(key=lambda kv: kv[1], reverse=True)
        return scores[:top_n]

    def most_similar_pair(self):
        """The single pair of distinct documents with the highest cosine
        similarity in the whole collection."""
        best_pair, best_score = None, -1.0
        for i, a in enumerate(self.doc_ids):
            for b in self.doc_ids[i + 1:]:
                sim = self.cosine_similarity(a, b)
                if sim > best_score:
                    best_pair, best_score = (a, b), sim
        return best_pair, best_score
