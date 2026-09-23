"""
demo.py
-------
Loads the document collection, builds tf-idf vectors, and computes
pairwise cosine similarity between every pair of documents.

Run with:  python demo.py
"""

import os

from vector_space import VectorSpaceModel

DOCS_FOLDER = os.path.join(os.path.dirname(__file__), "documents")


def load_documents(folder):
    """Load each .txt file in `folder` as a document.

    Files may optionally start with a metadata header (lines beginning
    with 'Source:' or 'License:', ending with a blank line) used for
    attribution -- this header is stripped before indexing so it
    doesn't pollute the vocabulary; only the actual article text below
    it is treated as document content.
    """
    documents = {}
    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith(".txt"):
            doc_id = os.path.splitext(filename)[0]
            with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                lines = f.readlines()

            body_start = 0
            for i, line in enumerate(lines):
                if line.strip() == "" and i > 0:
                    body_start = i + 1
                    break
                if not (line.startswith("Source:") or line.startswith("License:")):
                    body_start = i
                    break

            documents[doc_id] = "".join(lines[body_start:])
    return documents


def print_similarity_matrix(vsm):
    ids = vsm.doc_ids
    short = [d[:10] for d in ids]  # shorten labels so the table fits
    header = " " * 12 + "".join(f"{s:>11s}" for s in short)
    print(header)
    matrix = vsm.similarity_matrix()
    for a, sa in zip(ids, short):
        row = f"{sa:12s}" + "".join(f"{matrix[a][b]:11.3f}" for b in ids)
        print(row)


def main():
    documents = load_documents(DOCS_FOLDER)

    vsm = VectorSpaceModel(documents, use_stemming=True)

    print("=" * 70)
    print("COLLECTION LOADED")
    print("=" * 70)
    print(f"Documents: {len(documents)}")
    print()

    print("=" * 70)
    print("PAIRWISE COSINE SIMILARITY MATRIX")
    print("=" * 70)
    print_similarity_matrix(vsm)
    print()

    print("=" * 70)
    print("MOST SIMILAR PAIR OVERALL")
    print("=" * 70)
    pair, score = vsm.most_similar_pair()
    print(f"{pair[0]}  <->  {pair[1]}   (cosine similarity = {score:.3f})")
    print()

    print("=" * 70)
    print("TOP-2 MOST SIMILAR DOCUMENTS, FOR EACH DOCUMENT")
    print("=" * 70)
    for doc_id in vsm.doc_ids:
        similar = vsm.most_similar(doc_id, top_n=2)
        similar_str = ", ".join(f"{d} ({s:.3f})" for d, s in similar)
        print(f"{doc_id:28s} -> {similar_str}")
    print()

    print("=" * 70)
    print("TOP-5 DISTINGUISHING TERMS PER DOCUMENT (highest tf-idf weight)")
    print("=" * 70)
    for doc_id in vsm.doc_ids:
        terms = vsm.top_terms(doc_id, n=5)
        terms_str = ", ".join(f"{t} ({w:.3f})" for t, w in terms)
        print(f"{doc_id:28s} -> {terms_str}")


if __name__ == "__main__":
    main()
