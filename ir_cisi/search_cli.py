"""Interactive / one-shot search over CISI.   python search_cli.py "your query" --model bm25 --prf"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from cisi_ir.loader import load_documents
from cisi_ir.preprocess import Preprocessor
from cisi_ir.index import InvertedIndex
from cisi_ir.models import TfIdfCosine, BM25, QueryLikelihoodJM, SearchEngine
from cisi_ir.feedback import RocchioPRF

ap = argparse.ArgumentParser()
ap.add_argument("query", nargs="?", help="query text (omit for interactive mode)")
ap.add_argument("--data", default="data/CISI.ALL")
ap.add_argument("--model", choices=["tfidf", "bm25", "lm"], default="bm25")
ap.add_argument("--prf", action="store_true", help="Rocchio pseudo-relevance feedback")
ap.add_argument("-k", type=int, default=10)
a = ap.parse_args()

docs = load_documents(a.data)
ix = InvertedIndex(Preprocessor(True, True, True)).build(docs)
model = {"tfidf": TfIdfCosine, "bm25": BM25, "lm": QueryLikelihoodJM}[a.model](ix)
engine = SearchEngine(ix, model, RocchioPRF(ix) if a.prf else None)
print(f"Indexed {ix.N} docs, {ix.vocab_size} terms. Model={a.model} PRF={a.prf}")


def run(q):
    print(f"analysed -> {engine.analyse(q)}")
    for r, (d, s) in enumerate(engine.search(q, a.k), 1):
        print(f"{r:>2}. [{d:>4}] {s:7.3f}  {docs[d]['title'][:80]}")


if a.query:
    run(a.query)
else:
    while True:
        q = input("\nquery> ").strip()
        if not q:
            break
        run(q)
