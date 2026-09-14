"""
inverted_index.py
------------------
Builds and stores the inverted index described in Chapter 1 (and
constructed at scale in Chapter 4) of the textbook: for every term in
the dictionary, a postings list of the IDs of documents containing
that term.

The construction here follows the same two logical steps as the
BSBI / SPIMI algorithms in Chapter 4, just done in memory since our
collection is tiny (5-10 documents) rather than requiring disk-based
blocking:

    1. Collect (term, docID) pairs while scanning the collection.
    2. Group the pairs by term to build each postings list.

Postings are stored as Python sets for O(1) membership tests and easy
AND / OR / NOT set operations during Boolean retrieval, then exposed
as sorted lists (the standard, docID-ordered representation used in
the book) via `postings()`.
"""


class InvertedIndex:
    def __init__(self):
        # term -> set of doc_ids  (the postings list for that term)
        self._postings = {}
        self.all_doc_ids = set()

    def add_document(self, doc_id, terms):
        """Add the (term, docID) pairs for one document to the index."""
        self.all_doc_ids.add(doc_id)
        for term in set(terms):
            self._postings.setdefault(term, set()).add(doc_id)

    def postings(self, term):
        """Return the postings list for `term` as a sorted list of doc IDs."""
        return sorted(self._postings.get(term, set()))

    def postings_set(self, term):
        """Return the postings list for `term` as a set (used internally
        by the Boolean retrieval engine for fast set operations)."""
        return self._postings.get(term, set())

    def terms(self):
        return sorted(self._postings.keys())

    def __repr__(self):
        lines = []
        for term in self.terms():
            lines.append(f"{term:15s} -> {self.postings(term)}")
        return "\n".join(lines)
