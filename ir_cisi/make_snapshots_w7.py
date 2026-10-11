"""Create the screenshot-style images used in the Week 7 report (results/week7/snapshots/)."""
import os
from snapshot_utils import render, block

OUT = "results/week7/snapshots"
os.makedirs(OUT, exist_ok=True)
log = "results/week7/run_log_w7.txt"

# console output
render(block(log, "== MEAN METRICS AT FIXED", "== 95% BOOTSTRAP"), f"{OUT}/s1_mean_metrics.png", "python evaluate_models.py  (mean metrics)", 14)
render(block(log, "== 95% BOOTSTRAP", "== DEV / TEST"), f"{OUT}/s2_confidence.png", "python evaluate_models.py  (bootstrap confidence intervals)", 14)
render(block(log, "== PAIRED RANDOMIZATION", "== BOOLEAN"), f"{OUT}/s3_significance.png", "python evaluate_models.py  (significance tests)", 14)
render(block(log, "== BOOLEAN BASELINES", "== 8 HARDEST"), f"{OUT}/s4_boolean_length.png", "python evaluate_models.py  (Boolean baselines, query length)", 14)
render(block(log, "== 8 HARDEST", "== SHORT-QUERY"), f"{OUT}/s5_failures.png", "python evaluate_models.py  (failure analysis)", 13)
render(block(log, "== SHORT-QUERY", "Done."), f"{OUT}/s6_short_queries.png", "python evaluate_models.py  (short-query simulation)", 14)
render(open("results/week7/capture_recapture_log.txt").read(), f"{OUT}/s7_capture_recapture.png", "python capture_recapture_demo.py", 14)

# code listings
render(block("src/cisi_ir/evaluate.py", "def average_precision", "def reciprocal_rank"), f"{OUT}/c1_average_precision.png", "src/cisi_ir/evaluate.py  (average precision)", 14)
render(block("src/cisi_ir/evaluate.py", "def ndcg_at_k", "def interpolated_pr"), f"{OUT}/c2_ndcg.png", "src/cisi_ir/evaluate.py  (nDCG@k)", 14)
render(block("src/cisi_ir/evaluate.py", "def set_metrics", "def full_metrics"), f"{OUT}/c3_set_metrics.png", "src/cisi_ir/evaluate.py  (set-based precision / recall / F1)", 14)
render(block("src/cisi_ir/evaluate.py", "def paired_randomization_test", "def bootstrap_ci"), f"{OUT}/c4_randomization.png", "src/cisi_ir/evaluate.py  (paired randomization test)", 14)
render(block("evaluate_models.py", "def rank_all", "# ---------------------------------------------------------------- 3.") if False else
       block("evaluate_models.py", "def rank_all", "# ------------------------------------------------------------------ 3."), f"{OUT}/c5_run_systems.png", "evaluate_models.py  (run every system)", 14)
render(block("evaluate_models.py", "def shorten", "short_rows = []"), f"{OUT}/c6_shorten.png", "evaluate_models.py  (short-query simulation)", 14)
print("snapshots written")
