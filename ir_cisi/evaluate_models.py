"""Week 7: detailed evaluation of the CISI retrieval models built in Week 6.

Run:  python evaluate_models.py          (needs data/CISI.ALL, CISI.QRY, CISI.REL)
Output: tables (CSV) and figures in results/week7/, console log of every table.

The script follows the Cranfield / test-collection paradigm: every system answers the
same queries over the same documents, and its ranked list is compared with the
human relevance judgments (qrels) using several complementary metrics.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")                      # draw figures to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from cisi_ir.loader import load_documents, load_queries, load_qrels
from cisi_ir.preprocess import Preprocessor
from cisi_ir.index import InvertedIndex
from cisi_ir.models import TfIdfCosine, BM25, QueryLikelihoodJM, SearchEngine
from cisi_ir.feedback import RocchioPRF
from cisi_ir.evaluate import (full_metrics, set_metrics, interpolated_pr, paired_randomization_test,
                              bootstrap_ci)

OUT = "results/week7"
os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 200)

# ------------------------------------------------------------------ 1. data and index
docs = load_documents("data/CISI.ALL")
queries = load_queries("data/CISI.QRY")
qrels = load_qrels("data/CISI.REL")
judged = sorted(q for q in qrels if q in queries)    # only queries with judgments can be scored
N = len(docs)
index = InvertedIndex(Preprocessor(True, True, True)).build(docs)   # best pipeline from Week 6
params = json.load(open("results/best_params.json"))                # parameters tuned on the dev queries
k1, b, prf = params["best_k1"], params["best_b"], params["prf"]
print(f"{N} documents, {len(judged)} judged queries, vocabulary {index.vocab_size} terms")

# ------------------------------------------------------------------ 2. systems under test
rocchio = RocchioPRF(index, 1.0, prf["beta"], prf["fb_docs"], prf["fb_terms"])
systems = {
    "TF-IDF cosine":      SearchEngine(index, TfIdfCosine(index)),
    "Query LM (JM)":      SearchEngine(index, QueryLikelihoodJM(index, 0.7)),
    "BM25 (default)":     SearchEngine(index, BM25(index, 1.2, 0.75)),
    "BM25 (tuned)":       SearchEngine(index, BM25(index, k1, b)),
    "BM25 tuned + PRF":   SearchEngine(index, BM25(index, k1, b), rocchio),
}
BASE = "BM25 (default)"                                   # reference system for significance tests


def rank_all(engine, term_lists=None):
    """Full ranking (all N documents that match) for every judged query."""
    run = {}
    for q in judged:
        terms = term_lists[q] if term_lists else engine.analyse(queries[q])
        run[q] = [d for d, _ in engine.search_terms(terms, k=N)]
    return run


runs = {name: rank_all(eng) for name, eng in systems.items()}
per_query = {name: pd.DataFrame(full_metrics(run, qrels)).T for name, run in runs.items()}   # rows = queries

# ------------------------------------------------------------------ 3. mean effectiveness table
cols = ["P@1", "P@5", "P@10", "P@20", "R@10", "R@20", "R@50", "R@100", "F1@10", "Rprec", "AP",
        "nDCG@5", "nDCG@10", "nDCG@20", "RR", "Success@10"]
mean_tbl = pd.DataFrame({n: df[cols].mean() for n, df in per_query.items()}).T
mean_tbl = mean_tbl.rename(columns={"AP": "MAP", "RR": "MRR"})
mean_tbl.to_csv(f"{OUT}/ranked_metrics.csv")
# printed as two narrow tables so they stay readable in a terminal / screenshot
print("\n== MEAN METRICS AT FIXED CUTOFFS (76 judged queries) ==\n",
      mean_tbl[["P@1", "P@5", "P@10", "P@20", "R@10", "R@20", "R@50", "R@100"]].round(4).to_string())
print("\n== MEAN SUMMARY METRICS (76 judged queries) ==\n",
      mean_tbl[["F1@10", "Rprec", "MAP", "nDCG@5", "nDCG@10", "nDCG@20", "MRR", "Success@10"]].round(4).to_string())

# 95% bootstrap confidence intervals for the headline metrics
ci_rows = []
for n, df in per_query.items():
    for m, label in [("AP", "MAP"), ("nDCG@10", "nDCG@10"), ("P@10", "P@10")]:
        lo, hi = bootstrap_ci(df[m].tolist())
        ci_rows.append({"system": n, "metric": label, "mean": df[m].mean(), "ci_low": lo, "ci_high": hi})
ci_tbl = pd.DataFrame(ci_rows)
ci_tbl.to_csv(f"{OUT}/confidence_intervals.csv", index=False)
print("\n== 95% BOOTSTRAP CIs ==\n", ci_tbl.round(4).to_string(index=False))

# ------------------------------------------------------------------ 4. dev vs test split (same split as Week 6)
dev_ids, test_ids = judged[0::2], judged[1::2]
split_tbl = pd.DataFrame({n: {"MAP dev": df.loc[dev_ids, "AP"].mean(), "MAP test": df.loc[test_ids, "AP"].mean(),
                              "nDCG@10 test": df.loc[test_ids, "nDCG@10"].mean(),
                              "P@10 test": df.loc[test_ids, "P@10"].mean()} for n, df in per_query.items()}).T
split_tbl.to_csv(f"{OUT}/dev_test_split.csv")
print("\n== DEV / TEST SPLIT ==\n", split_tbl.round(4).to_string())

# ------------------------------------------------------------------ 5. significance tests vs reference
pairs = [("TF-IDF cosine", BASE), ("Query LM (JM)", BASE), ("BM25 (tuned)", BASE), ("BM25 tuned + PRF", BASE),
         ("BM25 tuned + PRF", "BM25 (tuned)")]
sig_rows = []
for a, bname in pairs:
    for m, label in [("AP", "MAP"), ("nDCG@10", "nDCG@10"), ("P@10", "P@10")]:
        x, y = per_query[a][m].tolist(), per_query[bname][m].tolist()
        diff = np.array(x) - np.array(y)
        sig_rows.append({"system": a, "vs": bname, "metric": label, "mean diff": diff.mean(),
                         "wins": int((diff > 1e-9).sum()), "losses": int((diff < -1e-9).sum()),
                         "ties": int((abs(diff) <= 1e-9).sum()), "p-value": paired_randomization_test(x, y)})
sig_tbl = pd.DataFrame(sig_rows)
sig_tbl.to_csv(f"{OUT}/significance.csv", index=False)
print("\n== PAIRED RANDOMIZATION TESTS (two-sided) ==\n", sig_tbl.round(4).to_string(index=False))

# ------------------------------------------------------------------ 6. unranked Boolean baselines
# A Boolean engine returns a SET, so precision/recall/F1 are computed over the whole set.
bool_rows = {"Boolean AND": [], "Boolean OR": []}
for q in judged:
    terms = sorted(set(t for t in index.pre(queries[q]) if index.df(t)))
    bool_rows["Boolean AND"].append(set_metrics(index.boolean_and(terms), qrels[q]))
    bool_rows["Boolean OR"].append(set_metrics(index.boolean_or(terms), qrels[q]))
bool_tbl = pd.DataFrame({n: pd.DataFrame(r).mean() for n, r in bool_rows.items()}).T
bool_tbl["% queries with empty result"] = [100 * np.mean([m["retrieved"] == 0 for m in r]) for r in bool_rows.values()]
bool_tbl.to_csv(f"{OUT}/boolean_baselines.csv")
print("\n== BOOLEAN BASELINES (set-based) ==\n", bool_tbl.round(4).to_string())

# ------------------------------------------------------------------ 7. query-level analysis
ap = pd.DataFrame({n: df["AP"] for n, df in per_query.items()})
ap.to_csv(f"{OUT}/per_query_ap.csv")
qinfo = pd.DataFrame({"terms": {q: len(set(index.pre(queries[q]))) for q in judged},
                      "relevant": {q: len(qrels[q]) for q in judged}})
qinfo["bucket"] = pd.qcut(qinfo["terms"], 3, labels=["short", "medium", "long"])
len_tbl = pd.DataFrame({"queries": qinfo.groupby("bucket", observed=True).size(),
                        "avg terms": qinfo.groupby("bucket", observed=True)["terms"].mean(),
                        "MAP BM25 (tuned)": ap["BM25 (tuned)"].groupby(qinfo["bucket"], observed=True).mean(),
                        "MAP TF-IDF": ap["TF-IDF cosine"].groupby(qinfo["bucket"], observed=True).mean(),
                        "MAP + PRF": ap["BM25 tuned + PRF"].groupby(qinfo["bucket"], observed=True).mean()})
len_tbl.to_csv(f"{OUT}/query_length_analysis.csv")
print("\n== EFFECT OF QUERY LENGTH (terciles by analysed term count) ==\n", len_tbl.round(4).to_string())


def spearman(x, y):
    """Spearman rank correlation = Pearson correlation of the ranks."""
    return float(np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1])


rho_rel = spearman(qinfo["relevant"], ap["BM25 (tuned)"])
rho_len = spearman(qinfo["terms"], ap["BM25 (tuned)"])
print(f"\nSpearman(AP, #relevant docs) = {rho_rel:.3f}   Spearman(AP, #query terms) = {rho_len:.3f}")

# failure analysis: hardest and easiest queries for the tuned BM25 system
fa = qinfo.copy()
fa["AP"] = ap["BM25 (tuned)"]
fa["first_rel_rank"] = [next((i for i, d in enumerate(runs["BM25 (tuned)"][q], 1) if d in qrels[q]), None) for q in judged]
fa["query"] = [queries[q][:42] + "..." for q in judged]
worst, best = fa.sort_values("AP").head(8), fa.sort_values("AP", ascending=False).head(5)
fa.sort_values("AP").to_csv(f"{OUT}/failure_analysis.csv")
print("\n== 8 HARDEST QUERIES (BM25 tuned) ==\n", worst[["AP", "terms", "relevant", "first_rel_rank", "query"]].round(3).to_string())
print("\n== 5 EASIEST QUERIES ==\n", best[["AP", "terms", "relevant", "first_rel_rank", "query"]].round(3).to_string())
print(f"\nQueries with AP < 0.05 (BM25 tuned): {(fa['AP'] < 0.05).sum()} of {len(fa)}; "
      f"queries with no relevant doc in top 10: {int((per_query['BM25 (tuned)']['Success@10'] == 0).sum())}")

# ------------------------------------------------------------------ 8. short-query simulation (web-style queries)
# Web users type about 2-3 keywords (Manning et al., 2009), CISI queries are long sentences.
# We keep only the m highest-idf (most discriminative) terms of each query and re-run the systems.
def shorten(q, m):
    """Keep the m most discriminative (highest-idf) distinct terms of a query. Each kept term
    keeps its original frequency in the query, so m=None reproduces the full query exactly."""
    qtf = {}
    for t in index.pre(queries[q]):
        if index.df(t):
            qtf[t] = qtf.get(t, 0) + 1
    keep = sorted(qtf, key=lambda t: -index.idf(t))
    if m:
        keep = keep[:m]
    return [t for t in keep for _ in range(qtf[t])]


short_rows = []
for m in (2, 3, 5, 8, None):                       # None = full query
    tl = {q: shorten(q, m) for q in judged}
    for name in ("TF-IDF cosine", "Query LM (JM)", "BM25 (tuned)", "BM25 tuned + PRF"):
        pq = pd.DataFrame(full_metrics(rank_all(systems[name], tl), qrels)).T
        short_rows.append({"terms kept": m if m else "all", "system": name, "MAP": pq["AP"].mean(),
                           "P@10": pq["P@10"].mean(), "nDCG@10": pq["nDCG@10"].mean(), "P@1": pq["P@1"].mean()})
short_tbl = pd.DataFrame(short_rows)
short_tbl.to_csv(f"{OUT}/short_query_simulation.csv", index=False)
print("\n== SHORT-QUERY SIMULATION (top-m idf terms) ==\n", short_tbl.round(4).to_string(index=False))

# ------------------------------------------------------------------ 9. figures
names = list(systems)
colors = plt.cm.tab10(np.arange(len(names)))

# 9a grouped bars of headline metrics
fig, ax = plt.subplots(figsize=(8.6, 4.2))
metrics = ["P@10", "R@20", "MAP", "nDCG@10", "MRR"]
w = 0.16
for i, n in enumerate(names):
    ax.bar(np.arange(len(metrics)) + i * w, [mean_tbl.loc[n, m] for m in metrics], w, label=n, color=colors[i])
ax.set_xticks(np.arange(len(metrics)) + 2 * w); ax.set_xticklabels(metrics)
ax.set_ylabel("score"); ax.set_title("Headline metrics (76 judged queries)"); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_headline_metrics.png", dpi=160); plt.close()

# 9b precision at k and recall at rank k curves
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
ks = np.arange(1, 51)
for i, n in enumerate(names):
    pk = [np.mean([sum(d in qrels[q] for d in runs[n][q][:k]) / k for q in judged]) for k in ks]
    rk = [np.mean([sum(d in qrels[q] for d in runs[n][q][:k]) / len(qrels[q]) for q in judged]) for k in ks]
    axs[0].plot(ks, pk, label=n, color=colors[i]); axs[1].plot(ks, rk, label=n, color=colors[i])
axs[0].set_xlabel("rank k"); axs[0].set_ylabel("precision@k"); axs[0].set_title("Precision at k")
axs[1].set_xlabel("rank k"); axs[1].set_ylabel("recall@k"); axs[1].set_title("Recall at k"); axs[0].legend(fontsize=7)
for a_ in axs: a_.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_precision_recall_at_k.png", dpi=160); plt.close()

# 9c 11-point interpolated precision-recall
fig, ax = plt.subplots(figsize=(6, 4.3))
for i, n in enumerate(names):
    pr = np.mean([interpolated_pr(runs[n][q], qrels[q]) for q in judged], axis=0)
    ax.plot(np.linspace(0, 1, 11), pr, marker="o", ms=3, label=n, color=colors[i])
ax.set_xlabel("recall"); ax.set_ylabel("interpolated precision"); ax.set_title("11-point precision-recall"); ax.legend(fontsize=7); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_pr_curve.png", dpi=160); plt.close()

# 9d distribution of per-query AP, and per-query scatter (BM25 default vs PRF)
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].boxplot([ap[n] for n in names], showmeans=True)
axs[0].set_xticks(range(1, len(names) + 1)); axs[0].set_xticklabels([n.replace(" ", "\n", 1) for n in names], fontsize=7)
axs[0].set_ylabel("average precision per query"); axs[0].set_title("Per-query AP distribution")
axs[1].scatter(ap[BASE], ap["BM25 tuned + PRF"], s=18)
axs[1].plot([0, 1], [0, 1], "k--", lw=1)
axs[1].set_xlabel("AP, BM25 (default)"); axs[1].set_ylabel("AP, BM25 tuned + PRF"); axs[1].set_title("Query-by-query comparison")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_per_query_ap.png", dpi=160); plt.close()

# 9e short-query simulation
fig, ax = plt.subplots(figsize=(6.2, 4))
order = ["2", "3", "5", "8", "all"]
for i, n in enumerate(["TF-IDF cosine", "Query LM (JM)", "BM25 (tuned)", "BM25 tuned + PRF"]):
    sub = short_tbl[short_tbl.system == n].copy(); sub["k"] = sub["terms kept"].astype(str)
    sub = sub.set_index("k").loc[order]
    ax.plot(order, sub["MAP"], marker="o", label=n)
ax.set_xlabel("query terms kept (highest idf first)"); ax.set_ylabel("MAP"); ax.set_title("Effect of query length"); ax.legend(fontsize=7); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_short_queries.png", dpi=160); plt.close()

# 9f one compact summary figure (used in the report): headline bars, P@k, PR curve, short queries
fig, axs = plt.subplots(2, 2, figsize=(10, 6.6))
for i, n in enumerate(names):
    axs[0, 0].bar(np.arange(len(metrics)) + i * w, [mean_tbl.loc[n, m] for m in metrics], w, label=n, color=colors[i])
axs[0, 0].set_xticks(np.arange(len(metrics)) + 2 * w); axs[0, 0].set_xticklabels(metrics)
axs[0, 0].set_title("(a) Headline metrics"); axs[0, 0].legend(fontsize=6.5)
for i, n in enumerate(names):
    pk = [np.mean([sum(d in qrels[q] for d in runs[n][q][:k_]) / k_ for q in judged]) for k_ in ks]
    axs[0, 1].plot(ks, pk, label=n, color=colors[i])
    pr = np.mean([interpolated_pr(runs[n][q], qrels[q]) for q in judged], axis=0)
    axs[1, 0].plot(np.linspace(0, 1, 11), pr, marker="o", ms=3, label=n, color=colors[i])
axs[0, 1].set_xlabel("rank k"); axs[0, 1].set_ylabel("precision@k"); axs[0, 1].set_title("(b) Precision at k"); axs[0, 1].grid(alpha=.3)
axs[1, 0].set_xlabel("recall"); axs[1, 0].set_ylabel("interpolated precision"); axs[1, 0].set_title("(c) 11-point precision-recall"); axs[1, 0].grid(alpha=.3)
for n in ["TF-IDF cosine", "Query LM (JM)", "BM25 (tuned)", "BM25 tuned + PRF"]:
    sub = short_tbl[short_tbl.system == n].copy(); sub["k"] = sub["terms kept"].astype(str)
    axs[1, 1].plot(order, sub.set_index("k").loc[order, "MAP"], marker="o", label=n)
axs[1, 1].set_xlabel("query terms kept (highest idf first)"); axs[1, 1].set_ylabel("MAP"); axs[1, 1].set_title("(d) Effect of query length"); axs[1, 1].grid(alpha=.3)
axs[1, 1].legend(fontsize=6.5)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_summary.png", dpi=150); plt.close()

json.dump({"rho_relevant": rho_rel, "rho_terms": rho_len}, open(f"{OUT}/correlations.json", "w"))
print("\nDone. Files written to", OUT)
