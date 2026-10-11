"""Week 7 discussion: simulate the capture-recapture estimate of relative index size.

Engine A indexes 50,000 pages, engine B 30,000, and 15,000 pages are in both. That is exactly
30% of A's pages (x = 0.3) and 50% of B's pages (y = 0.5), as in the discussion question.
We draw uniform random samples from each index, test membership in the other index, and check
that the estimate |A|/|B| ~ y/x recovers the true ratio 5/3.
"""
import random
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = set(range(0, 50_000))            # pages in A's index
B = set(range(35_000, 65_000))       # pages in B's index (overlap = ids 35,000..49,999 = 15,000 pages)
print(f"|A|={len(A):,}  |B|={len(B):,}  |A and B|={len(A & B):,}")
print(f"true x = |A and B|/|A| = {len(A & B) / len(A):.2f}   true y = |A and B|/|B| = {len(A & B) / len(B):.2f}")
print(f"exact y/x = {(len(A & B) / len(B)) / (len(A & B) / len(A)):.4f}   true |A|/|B| = {len(A) / len(B):.4f}")

listA, listB = sorted(A), sorted(B)
rng = random.Random(42)


def estimate(sample_size):
    """One capture-recapture experiment: sample pages uniformly from each index and test them."""
    x = sum(p in B for p in rng.sample(listA, sample_size)) / sample_size   # fraction of A's sample found in B
    y = sum(p in A for p in rng.sample(listB, sample_size)) / sample_size   # fraction of B's sample found in A
    return y / x if x else float("nan")                                     # |A|/|B| ~ y/x


for n in (100, 500, 2000):
    est = [estimate(n) for _ in range(2000)]
    est.sort()
    print(f"sample size {n:>4} per engine: mean estimate {statistics.mean(est):.3f}, "
          f"95% of estimates in [{est[int(.025 * len(est))]:.3f}, {est[int(.975 * len(est))]:.3f}]")

# figure: overlap bars + distribution of the estimates for 500 samples
est500 = [estimate(500) for _ in range(2000)]
fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.6))
axs[0].barh(["B (30,000)", "A (50,000)"], [30000, 50000], color=["#fca5a5", "#93c5fd"])
axs[0].barh(["B (30,000)", "A (50,000)"], [15000, 15000], color="#6b7280", label="pages in both (15,000)")
axs[0].set_xlabel("pages in index"); axs[0].set_title("15,000 shared pages = 30% of A, 50% of B"); axs[0].legend(fontsize=8)
axs[1].hist(est500, bins=30, color="#a5b4fc")
axs[1].axvline(5 / 3, color="k", ls="--", label="true ratio 5/3 = 1.667")
axs[1].set_xlabel("estimate y/x of |A|/|B| (500 samples per engine)"); axs[1].set_title("2,000 simulated experiments"); axs[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("results/week7/fig_capture_recapture.png", dpi=160)
