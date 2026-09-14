"""
demo.py
-------
Demonstrates the basic Information Retrieval system end to end:

    1. Loads a small collection of 8 sample documents.
    2. Preprocesses them (tokenize, remove stop words, stem).
    3. Builds the dictionary and inverted index.
    4. Runs a set of Boolean queries (AND, OR, NOT, and combinations)
       and prints the matching document IDs.

Run with:  python demo.py
"""

import os

from ir_system import IRSystem

DOCS_FOLDER = os.path.join(os.path.dirname(__file__), "documents")


def main():
    system = IRSystem(use_stemming=True)
    system.add_documents_from_folder(DOCS_FOLDER)
    system.build_index()

    stats = system.stats()
    print("=" * 60)
    print("COLLECTION LOADED")
    print("=" * 60)
    print(f"Documents indexed : {stats['num_documents']}")
    print(f"Vocabulary size   : {stats['vocabulary_size']}")
    print()

    print("=" * 60)
    print("DICTIONARY (term -> document frequency)")
    print("=" * 60)
    system.print_dictionary()
    print()

    print("=" * 60)
    print("INVERTED INDEX (term -> postings list)")
    print("=" * 60)
    system.print_index()
    print()

    queries = [
        "python",
        "python AND language",
        "python AND NOT language",
        "cat OR dog",
        "python AND (snake OR language)",
        "coffee OR football",
        "machine AND learn",
        "NOT space",
    ]

    print("=" * 60)
    print("BOOLEAN RETRIEVAL RESULTS")
    print("=" * 60)
    for q in queries:
        results = system.search(q)
        print(f"Query: {q!r:38s} -> {results}")


if __name__ == "__main__":
    main()
