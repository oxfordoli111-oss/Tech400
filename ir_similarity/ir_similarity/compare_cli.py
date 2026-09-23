"""
compare_cli.py
--------------
Interactive tool: pick two documents from the collection and see their
cosine similarity plus the terms driving that score. Handy for taking
screenshots for the assignment report.

Run with:  python compare_cli.py
"""

import os

from vector_space import VectorSpaceModel
from demo import load_documents, DOCS_FOLDER


def main():
    documents = load_documents(DOCS_FOLDER)
    vsm = VectorSpaceModel(documents)

    ids = vsm.doc_ids
    print(f"Loaded {len(ids)} documents:")
    for i, doc_id in enumerate(ids, 1):
        print(f"  {i}. {doc_id}")
    print()
    print("Enter two document numbers to compare (e.g. '1 8'), or 'exit' to quit.\n")

    while True:
        raw = input("compare> ").strip()
        if raw.lower() in ("exit", "quit"):
            break
        parts = raw.split()
        if len(parts) != 2:
            print("  Please enter exactly two numbers.")
            continue
        try:
            i, j = int(parts[0]) - 1, int(parts[1]) - 1
            a, b = ids[i], ids[j]
        except (ValueError, IndexError):
            print("  Invalid document number(s).")
            continue

        sim = vsm.cosine_similarity(a, b)
        print(f"  cosine_similarity({a}, {b}) = {sim:.4f}")
        print(f"  Top terms in {a}: {[t for t, _ in vsm.top_terms(a, 5)]}")
        print(f"  Top terms in {b}: {[t for t, _ in vsm.top_terms(b, 5)]}")
        print()


if __name__ == "__main__":
    main()
