"""Run all experiments for the Week 6 CISI assignment and write tables/figures to results/."""
import itertools
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from cisi_ir.loader import load_documents, load_queries, load_qrels
from cisi_ir.preprocess import Preprocessor
from cisi_ir.index import InvertedIndex
from cisi_ir.models import TfIdfCosine, BM25, QueryLikelihoodJM, SearchEngine
from cisi_ir.feedback import RocchioPRF
from cisi_ir.evaluate import evaluate_run

DATA, OUT = "data", "results"
os.makedirs(OUT, exist_ok=True)

docs = load_documents(f"{DATA}/CISI.ALL")
queries = load_queries(f"{DATA}/CISI.QRY")
qrels = load_qrels(f"{DATA}/CISI.REL")
judged = sorted(q for q in qrels if q in queries)
N = len(docs)
print(f"{N} documents | {len(queries)} queries | {len(judged)} judged queries | "
      f"{sum(len(v) for v in qrels.values())} relevance pairs")

# development / test split (Sec. 13.6: tune on held-out data, report once on the rest)
dev_ids = judged[0::2]
test_ids = judged[1::2]


def make_run(engine, qids):
    return {q: [d for d, _ in engine.search(queries[q], k=N)] for q in qids}


def score(engine, qids):
    run = make_run(engine, qids)
    return evaluate_run({q: run[q] for q in qids}, {q: qrels[q] for q in qids})


# ---------------------------------------------------------------- 1. preprocessing & index stats
configs = {
    "A: tokenise only": Preprocessor(True, False, False),
    "B: + stop words": Preprocessor(True, True, False),
    "C: + stop + Porter stem": Preprocessor(True, True, True),
}
indexes, stats_rows = {}, []
for name, pre in configs.items():
    ix = InvertedIndex(pre).build(docs)
    indexes[name] = ix
    stats_rows.append({"preprocessing": name, **ix.stats()})
stats_df = pd.DataFrame(stats_rows)
stats_df.to_csv(f"{OUT}/index_stats.csv", index=False)
print("\nINDEX STATISTICS\n", stats_df.to_string(index=False))

# ---------------------------------------------------------------- 2. model x preprocessing (all judged queries)
rows, curves = [], {}
for cname, ix in indexes.items():
    models = {"TF-IDF cosine (lnc.ltc)": TfIdfCosine(ix), "BM25 (k1=1.2, b=0.75)": BM25(ix),
              "Query LM (JM, λ=0.7)": QueryLikelihoodJM(ix)}
    for mname, model in models.items():
        m, _, pr = score(SearchEngine(ix, model), judged)
        rows.append({"preprocessing": cname, "model": mname, **m})
        curves[(cname, mname)] = pr
res_df = pd.DataFrame(rows)
res_df.to_csv(f"{OUT}/model_comparison.csv", index=False)
cols = ["preprocessing", "model", "MAP", "P@5", "P@10", "R@10", "nDCG@10", "MRR"]
print("\nMODEL x PREPROCESSING (all judged queries)\n", res_df[cols].round(4).to_string(index=False))

# ---------------------------------------------------------------- 3. BM25 tuning on dev, report on test
ix = indexes["C: + stop + Porter stem"]
grid_k1 = [0.6, 0.9, 1.2, 1.5, 1.8, 2.2]
grid_b = [0.25, 0.5, 0.75, 0.9, 1.0]
heat = pd.DataFrame(index=grid_b, columns=grid_k1, dtype=float)
for k1, b in itertools.product(grid_k1, grid_b):
    heat.loc[b, k1] = score(SearchEngine(ix, BM25(ix, k1, b)), dev_ids)[0]["MAP"]
best_b, best_k1 = heat.stack().idxmax()
print(f"\nBM25 tuning on DEV queries ({len(dev_ids)}): best k1={best_k1}, b={best_b}, dev MAP={heat.loc[best_b, best_k1]:.4f}")
heat.to_csv(f"{OUT}/bm25_dev_grid.csv")

# ---------------------------------------------------------------- 4. Pseudo-relevance feedback (Rocchio)
final_rows = []
systems = {
    "TF-IDF cosine": SearchEngine(ix, TfIdfCosine(ix)),
    "BM25 default": SearchEngine(ix, BM25(ix)),
    f"BM25 tuned (k1={best_k1}, b={best_b})": SearchEngine(ix, BM25(ix, best_k1, best_b)),
}
# tune PRF on dev as well
prf_grid = list(itertools.product([3, 5, 10], [10, 20, 40], [0.5, 1.0]))
best_prf, best_prf_map = None, -1
for fd, ft, beta in prf_grid:
    eng = SearchEngine(ix, BM25(ix, best_k1, best_b), RocchioPRF(ix, 1.0, beta, fd, ft))
    mp = score(eng, dev_ids)[0]["MAP"]
    if mp > best_prf_map:
        best_prf, best_prf_map = (fd, ft, beta), mp
fd, ft, beta = best_prf
print(f"PRF tuning on DEV: fb_docs={fd}, fb_terms={ft}, beta={beta}, dev MAP={best_prf_map:.4f}")
systems[f"BM25 tuned + Rocchio PRF (k={fd}, m={ft}, β={beta})"] = SearchEngine(
    ix, BM25(ix, best_k1, best_b), RocchioPRF(ix, 1.0, beta, fd, ft))

pr_final = {}
for name, eng in systems.items():
    m_test, _, pr_t = score(eng, test_ids)
    m_all, _, pr_a = score(eng, judged)
    final_rows.append({"system": name, "MAP (test)": m_test["MAP"], "P@10 (test)": m_test["P@10"],
                       "nDCG@10 (test)": m_test["nDCG@10"], "MAP (all 76)": m_all["MAP"],
                       "P@10 (all 76)": m_all["P@10"], "R@10 (all 76)": m_all["R@10"]})
    pr_final[name] = pr_a
final_df = pd.DataFrame(final_rows)
final_df.to_csv(f"{OUT}/final_systems.csv", index=False)
print("\nFINAL SYSTEMS\n", final_df.round(4).to_string(index=False))

# ---------------------------------------------------------------- 5. figures
fig, ax = plt.subplots(figsize=(8, 4.2))
piv = res_df.pivot(index="preprocessing", columns="model", values="MAP")
piv.plot(kind="bar", ax=ax, rot=15)
ax.set_ylabel("MAP"); ax.set_title("MAP by preprocessing pipeline and retrieval model (76 judged queries)")
ax.legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{OUT}/fig_map_models.png", dpi=160); plt.close()

fig, ax = plt.subplots(figsize=(6.5, 4.5))
for name, pr in pr_final.items():
    ax.plot([i / 10 for i in range(11)], pr, marker="o", ms=3, label=name)
ax.set_xlabel("Recall"); ax.set_ylabel("Interpolated precision"); ax.set_title("11-point interpolated precision-recall")
ax.legend(fontsize=7); ax.grid(alpha=.3); plt.tight_layout(); plt.savefig(f"{OUT}/fig_pr_curves.png", dpi=160); plt.close()

fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(heat.values.astype(float), aspect="auto", cmap="viridis", origin="lower")
ax.set_xticks(range(len(grid_k1))); ax.set_xticklabels(grid_k1)
ax.set_yticks(range(len(grid_b))); ax.set_yticklabels(grid_b)
ax.set_xlabel("k1"); ax.set_ylabel("b"); ax.set_title("BM25 dev-set MAP")
for i in range(len(grid_b)):
    for j in range(len(grid_k1)):
        ax.text(j, i, f"{heat.values[i, j]:.3f}", ha="center", va="center", color="w", fontsize=7)
plt.colorbar(im); plt.tight_layout(); plt.savefig(f"{OUT}/fig_bm25_grid.png", dpi=160); plt.close()

# ---------------------------------------------------------------- 6. sample query walk-through
eng = systems[list(systems)[2]]
qid = judged[4]
lines = [f"Query {qid}: {queries[qid]}", f"Analysed terms: {eng.analyse(queries[qid])}",
         f"Relevant docs ({len(qrels[qid])}): {sorted(qrels[qid])[:15]}...", "", "Top 10 (BM25 tuned):"]
for r, (d, s) in enumerate(eng.search(queries[qid], 10), 1):
    flag = "REL" if d in qrels[qid] else "   "
    lines.append(f"{r:>2}. doc {d:<5} score={s:6.3f} {flag}  {docs[d]['title'][:70]}")
open(f"{OUT}/sample_query.txt", "w").write("\n".join(lines))
print("\n" + "\n".join(lines))
json.dump({"best_k1": best_k1, "best_b": best_b, "prf": {"fb_docs": fd, "fb_terms": ft, "beta": beta},
           "dev_queries": len(dev_ids), "test_queries": len(test_ids)}, open(f"{OUT}/best_params.json", "w"), indent=2)
