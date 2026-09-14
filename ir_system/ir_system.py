"""
ir_system.py
------------
Top level class that ties together the four pieces required by the
assignment brief:

    1. A way to add documents to the system.
    2. A basic dictionary.
    3. An inverted index for the document collection.
    4. Boolean retrieval functions (AND / OR / NOT).

Usage:

    system = IRSystem()
    system.add_document("doc1", "Python is a programming language")
    system.add_document("doc2", "Cats are popular pets")
    system.build_index()
    system.search("python AND language")
"""

import os

from preprocessing import preprocess
from dictionary import Dictionary
from inverted_index import InvertedIndex
from boolean_retrieval import BooleanRetrieval


class IRSystem:
    def __init__(self, use_stemming=True):
        self.use_stemming = use_stemming
        self.documents = {}          # doc_id -> raw text
        self.dictionary = Dictionary()
        self.index = InvertedIndex()
        self._retrieval = None       # built lazily by build_index()

    # ---------- 1. adding documents ----------

    def add_document(self, doc_id, text):
        """Add a single document (given as raw text) to the system."""
        if doc_id in self.documents:
            raise ValueError(f"Document id '{doc_id}' already exists")
        self.documents[doc_id] = text

    def add_documents_from_folder(self, folder_path):
        """Add every .txt file found in `folder_path`. The file name
        (without extension) is used as the document ID."""
        for filename in sorted(os.listdir(folder_path)):
            if filename.lower().endswith(".txt"):
                doc_id = os.path.splitext(filename)[0]
                path = os.path.join(folder_path, filename)
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                self.add_document(doc_id, text)

    # ---------- 2 & 3. dictionary + inverted index ----------

    def build_index(self):
        """Preprocess every document and build the dictionary and the
        inverted index from scratch. This mirrors, on a small scale,
        the two-step 'inversion' process described in Chapter 4:
        first collect (term, docID) pairs, then group them by term."""
        self.dictionary = Dictionary()
        self.index = InvertedIndex()

        for doc_id, text in self.documents.items():
            terms = preprocess(text, use_stemming=self.use_stemming)
            self.dictionary.add_terms(doc_id, terms)
            self.index.add_document(doc_id, terms)

        self._retrieval = BooleanRetrieval(self.index, self.use_stemming)

    # ---------- 4. Boolean retrieval ----------

    def search(self, query):
        """Run a Boolean query (AND / OR / NOT / parentheses) against
        the index and return a sorted list of matching document IDs."""
        if self._retrieval is None:
            raise RuntimeError("Call build_index() before searching")
        return self._retrieval.search(query)

    # ---------- convenience / reporting helpers ----------

    def stats(self):
        return {
            "num_documents": len(self.documents),
            "vocabulary_size": self.dictionary.size(),
        }

    def print_dictionary(self):
        print(f"{'Term':15s} {'Doc. Freq.':>10s}")
        print("-" * 27)
        for term in self.dictionary.vocabulary():
            print(f"{term:15s} {self.dictionary.document_frequency(term):>10d}")

    def print_index(self):
        print(self.index)
