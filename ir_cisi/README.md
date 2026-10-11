# CISI Information Retrieval System

A basic ad-hoc search engine built **from scratch in Python** on the CISI collection
(1,460 documents, 112 queries, 76 queries with relevance judgments).
It implements classical retrieval methods described by Manning, Raghavan & Schütze (2009),
*An Introduction to Information Retrieval*.

| Stage | Method |
|---|---|
| Preprocessing | tokenise, lowercase, stop-word removal, Porter stemming |
| Indexing | inverted index with (docID, tf) postings, df/cf/doc-length statistics |
| Retrieval models | TF-IDF cosine (lnc.ltc), Okapi BM25, query likelihood (Jelinek-Mercer) |
| Query processing | same analyzer as documents, term-at-a-time scoring, heap top-k, Boolean AND/OR |
| Query expansion | Rocchio pseudo-relevance feedback |
| Evaluation | P@k, R@k, MAP, nDCG, MRR, 11-point PR curve |

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

Parameters were tuned on half of the queries (dev) and reported on the other half (test), following Manning et al. (2009).


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

## Week 7: evaluation of the models

`evaluate_models.py` evaluates every system on the 76 judged queries with precision/recall at several cutoffs,
F1, R-precision, MAP, nDCG, MRR, Success@10 and 11-point precision-recall curves. It also computes bootstrap
confidence intervals, paired randomization significance tests, Boolean AND/OR baselines, a per-query failure
analysis, an analysis of query length and a short-query (2-8 terms) simulation.

```bash
python evaluate_models.py          # tables + figures -> results/week7/
python capture_recapture_demo.py   # simulation for the Week 7 discussion (index-size estimation)
python make_snapshots_w7.py        # regenerate the screenshot-style images used in the report
```

| Week 7 file | Purpose |
|---|---|
| `src/cisi_ir/evaluate.py` | metric functions (P@k, R@k, F1, AP, nDCG, R-precision, set metrics, bootstrap CI, randomization test) |
| `evaluate_models.py` | runs all systems and writes `results/week7/*.csv` and figures |
| `capture_recapture_demo.py` | simulates the capture-recapture estimate of relative index size |
| `snapshot_utils.py`, `make_snapshots_w7.py` | create screenshot-style images of code and output |

Headline results (76 judged queries): BM25 + Rocchio feedback MAP 0.249, nDCG@10 0.420, P@1 0.566;
default BM25 MAP 0.222; query likelihood MAP 0.211; TF-IDF cosine MAP 0.175.
