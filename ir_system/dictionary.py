"""
dictionary.py
-------------
The Dictionary is the classical data structure described in Chapter 3
of the textbook: it holds every distinct vocabulary term (key) found
in the collection, together with a document frequency count for each
term. In this implementation the dictionary is backed by a Python
dict, which behaves like an in-memory hash table. Because our
collection is small, a hash-based dictionary is the right choice; the
textbook notes that search trees (B-trees) become preferable once a
collection is very large, disk-resident, or needs prefix/wildcard
lookups.
"""


class Dictionary:
    def __init__(self):
        # term -> document frequency (number of documents containing term)
        self._doc_freq = {}

    def add_terms(self, doc_id, terms):
        """Register the unique terms of one document in the dictionary."""
        for term in set(terms):
            self._doc_freq[term] = self._doc_freq.get(term, 0) + 1

    def contains(self, term):
        return term in self._doc_freq

    def document_frequency(self, term):
        """Number of documents in the collection that contain `term`."""
        return self._doc_freq.get(term, 0)

    def vocabulary(self):
        """Return the full vocabulary, sorted (as a B-tree walk would)."""
        return sorted(self._doc_freq.keys())

    def size(self):
        return len(self._doc_freq)

    def __len__(self):
        return self.size()

    def __contains__(self, term):
        return self.contains(term)
