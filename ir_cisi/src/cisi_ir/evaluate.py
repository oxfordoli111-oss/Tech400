"""Evaluation of ranked retrieval (Manning et al., Ch. 8): P@k, R@k, MAP, nDCG, MRR, 11-pt PR curve."""
import math


def precision_at_k(ranked, rel, k):
    return sum(1 for d in ranked[:k] if d in rel) / k


def recall_at_k(ranked, rel, k):
    return sum(1 for d in ranked[:k] if d in rel) / len(rel) if rel else 0.0


def average_precision(ranked, rel):
    hits, total = 0, 0.0
    for i, d in enumerate(ranked, 1):
        if d in rel:
            hits += 1
            total += hits / i
    return total / len(rel) if rel else 0.0


def reciprocal_rank(ranked, rel):
    for i, d in enumerate(ranked, 1):
        if d in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked, rel, k):
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], 1) if d in rel)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel), k) + 1))
    return dcg / idcg if idcg else 0.0


def interpolated_pr(ranked, rel, levels=11):
    """Interpolated precision at recall 0.0, 0.1, ..., 1.0 (Sec. 8.4)."""
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
