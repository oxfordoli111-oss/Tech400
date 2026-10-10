# CISI Information Retrieval System

A basic ad-hoc search engine built **from scratch in Python** on the CISI collection
(1,460 documents, 112 queries, 76 queries with relevance judgments).
It implements methods from Chapters 1–12 of Manning, Raghavan & Schütze (2009),
*An Introduction to Information Retrieval*.

| Stage | Method | Book reference |
|---|---|---|
| Preprocessing | tokenise, lowercase, stop-word removal, Porter stemming | Ch. 2 |
| Indexing | inverted index with (docID, tf) postings, df/cf/doc-length statistics | Ch. 1, 4 |
| Retrieval models | TF-IDF cosine (lnc.ltc), Okapi BM25, query likelihood (Jelinek-Mercer) | Ch. 6, 11, 12 |
| Query processing | same analyzer as documents, term-at-a-time scoring, heap top-k, Boolean AND/OR | Ch. 1, 7 |
| Query expansion | Rocchio pseudo-relevance feedback | Ch. 9 |
| Evaluation | P@k, R@k, MAP, nDCG, MRR, 11-point PR curve | Ch. 8 |

## Setup
```bash
pip install -r requirements.txt
# Put CISI.ALL, CISI.QRY, CISI.REL (Kaggle: dmaso01dsta/cisi-a-dataset-for-information-retrieval) in ./data/
python run_experiments.py          # all experiments -> results/
python search_cli.py "automatic indexing of library catalogues" --model bm25 --prf -k 10
python make_snapshots.py           # regenerate report snapshot images
```

## Layout
```
src/cisi_ir/  loader.py preprocess.py index.py models.py feedback.py evaluate.py
run_experiments.py   search_cli.py   make_snapshots.py
results/             CSV tables, figures, logs, snapshots
```

## Key results (76 judged queries, stop + stem pipeline)
| System | MAP | P@10 |
|---|---|---|
| TF-IDF cosine | 0.175 | 0.290 |
| BM25 (k1=1.2, b=0.75) | 0.222 | 0.367 |
| BM25 tuned + Rocchio PRF | 0.249 | 0.380 |

Parameters were tuned on half of the queries (dev) and reported on the other half (test), following Sec. 13.6.


## Dataset

This project uses the **CISI** collection (1,460 documents, 112 queries, relevance judgments),
a public test collection for information retrieval originally from the University of Glasgow
Information Retrieval Group.

- Source: [CISI (a dataset for Information Retrieval) on Kaggle](https://www.kaggle.com/datasets/dmaso01dsta/cisi-a-dataset-for-information-retrieval)
- Original collection: http://ir.dcs.gla.ac.uk/resources/test_collections/cisi/
- The data files are **not included** in this repository. Download them from Kaggle and place
  these three files in a `data/` folder inside `ir_cisi/`:

      ir_cisi/data/CISI.ALL   (documents)
      ir_cisi/data/CISI.QRY   (queries)
      ir_cisi/data/CISI.REL   (relevance judgments)

Citation: HJMASON. (n.d.). *CISI (a dataset for information retrieval)* [Data set]. Kaggle.
https://www.kaggle.com/datasets/dmaso01dsta/cisi-a-dataset-for-information-retrieval