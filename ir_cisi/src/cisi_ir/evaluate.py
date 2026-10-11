"""Evaluation of ranked retrieval (Manning et al., 2009): P@k, R@k, MAP, nDCG, MRR, 11-pt PR curve."""
import math


def precision_at_k(ranked, rel, k):
    # fraction of the first k results that are relevant
    return sum(1 for d in ranked[:k] if d in rel) / k


def recall_at_k(ranked, rel, k):
    # fraction of ALL relevant documents that appear in the first k results
    return sum(1 for d in ranked[:k] if d in rel) / len(rel) if rel else 0.0


def average_precision(ranked, rel):
    hits, total = 0, 0.0
    for i, d in enumerate(ranked, 1):          # i = rank, starting at 1
        if d in rel:                           # only ranks holding a relevant document count
            hits += 1
            total += hits / i                  # precision at this rank
    # dividing by |Rel| (not by hits) penalises relevant documents that were never retrieved
    return total / len(rel) if rel else 0.0


def reciprocal_rank(ranked, rel):
    for i, d in enumerate(ranked, 1):
        if d in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked, rel, k):
    # binary gain: a relevant document at rank i adds 1/log2(i+1), so early ranks count more
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], 1) if d in rel)
    # ideal ranking puts min(|Rel|, k) relevant documents first; dividing by it gives a 0-1 score
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel), k) + 1))
    return dcg / idcg if idcg else 0.0


def interpolated_pr(ranked, rel, levels=11):
    """Interpolated precision at recall 0.0, 0.1, ..., 1.0."""
    points, hits = [], 0
    for i, d in enumerate(ranked, 1):
        if d in rel:
            hits += 1
            points.append((hits / len(rel), hits / i))
    out = []
    for j in range(levels):
        r = j / (levels - 1)
        out.append(max([p for rec, p in points if rec >= r], default=0.0))
    return out


def evaluate_run(run, qrels, ks=(5, 10)):
    """run: {qid: [doc ids ranked]}; returns mean metrics over queries that have judgments."""
    qids = [q for q in qrels if q in run]
    per_query = {}
    for q in qids:
        r, rel = run[q], qrels[q]
        m = {"AP": average_precision(r, rel), "MRR": reciprocal_rank(r, rel)}
        for k in ks:
            m[f"P@{k}"] = precision_at_k(r, rel, k)
            m[f"R@{k}"] = recall_at_k(r, rel, k)
            m[f"nDCG@{k}"] = ndcg_at_k(r, rel, k)
        per_query[q] = m
    keys = next(iter(per_query.values())).keys()
    mean = {k: sum(v[k] for v in per_query.values()) / len(per_query) for k in keys}
    mean["MAP"] = mean.pop("AP")
    pr = [interpolated_pr(run[q], qrels[q]) for q in qids]
    mean_pr = [sum(p[i] for p in pr) / len(pr) for i in range(11)]
    return mean, per_query, mean_pr


# =====================================================================================
# Week 7 additions: more metrics, set-based (unranked) evaluation, significance testing
# =====================================================================================
import random


def f1_at_k(ranked, rel, k):
    """Harmonic mean of precision@k and recall@k (balanced F-measure)."""
    p, r = precision_at_k(ranked, rel, k), recall_at_k(ranked, rel, k)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def r_precision(ranked, rel):
    """Precision at rank R, where R = number of relevant documents for the query.
    At this cutoff precision and recall are equal, so it is a single-number summary
    that adapts to how many relevant documents the query has."""
    R = len(rel)
    return sum(1 for d in ranked[:R] if d in rel) / R if R else 0.0


def success_at_k(ranked, rel, k):
    """1 if at least one relevant document is in the top k, else 0 (a 'hit rate')."""
    return 1.0 if any(d in rel for d in ranked[:k]) else 0.0


def set_metrics(retrieved, rel):
    """Precision, recall and F1 for an UNRANKED result set (used for Boolean retrieval)."""
    retrieved = set(retrieved)
    tp = len(retrieved & rel)                       # relevant documents actually retrieved
    p = tp / len(retrieved) if retrieved else 0.0   # fraction of the result set that is relevant
    r = tp / len(rel) if rel else 0.0               # fraction of all relevant docs that were found
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "F1": f, "retrieved": len(retrieved)}


def full_metrics(run, qrels):
    """Return a DataFrame-friendly dict {qid: {metric: value}} with every ranked metric.
    `run` maps query id -> full ranked list of doc ids; only judged queries are scored."""
    out = {}
    for q in sorted(qrels):
        if q not in run:
            continue
        r, rel = run[q], qrels[q]
        m = {"AP": average_precision(r, rel), "RR": reciprocal_rank(r, rel), "Rprec": r_precision(r, rel)}
        for k in (1, 3, 5, 10, 20):
            m[f"P@{k}"] = precision_at_k(r, rel, k)
        for k in (10, 20, 50, 100):
            m[f"R@{k}"] = recall_at_k(r, rel, k)
        for k in (5, 10, 20):
            m[f"nDCG@{k}"] = ndcg_at_k(r, rel, k)
        m["F1@10"] = f1_at_k(r, rel, 10)
        m["Success@10"] = success_at_k(r, rel, 10)
        out[q] = m
    return out


def paired_randomization_test(a, b, n_perm=20000, seed=7):
    """Two-sided paired randomization (sign-flip) test on per-query scores a[i], b[i].
    Under the null hypothesis the two systems are exchangeable, so the sign of each
    per-query difference is arbitrary; we flip signs at random and count how often
    the mean difference is at least as extreme as the observed one."""
    rng = random.Random(seed)                          # fixed seed -> reproducible p-values
    diffs = [x - y for x, y in zip(a, b)]              # per-query difference (system 1 - 2)
    observed = abs(sum(diffs) / len(diffs))            # observed mean difference
    extreme = 0
    for _ in range(n_perm):
        # under the null hypothesis the sign of every difference is arbitrary: flip each at random
        s = sum(d if rng.random() < 0.5 else -d for d in diffs) / len(diffs)
        if abs(s) >= observed - 1e-12:                 # shuffled difference at least as extreme?
            extreme += 1
    return (extreme + 1) / (n_perm + 1)             # add-one so p is never exactly 0


def bootstrap_ci(values, n_boot=10000, seed=11, alpha=0.05):
    """Percentile bootstrap confidence interval for the mean of per-query scores."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(alpha / 2 * n_boot)], means[int((1 - alpha / 2) * n_boot) - 1]
