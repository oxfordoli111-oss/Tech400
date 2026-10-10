import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(10, 5.2)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 52)

def box(x, y, w, h, t, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec="#333", lw=1.1))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8.5)

def arrow(x1, y1, x2, y2, t=None):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", lw=1.3, color="#333"))
    if t: ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.5, t, ha="center", fontsize=7.5, style="italic")

ax.text(25, 50, "OFFLINE: indexing", ha="center", fontweight="bold", fontsize=10)
ax.text(70, 50, "ONLINE: query processing", ha="center", fontweight="bold", fontsize=10)
box(2, 36, 20, 8, "CISI.ALL\n1,460 documents\n(title + abstract)", "#dbeafe")
box(2, 21, 20, 9, "Preprocessing\ntokenise · lowercase\nstop words · Porter stem", "#fde68a")
box(2, 5, 20, 11, "Inverted index\nterm → [(docID, tf)]\ndf · cf · doc length\navgdl · forward index", "#bbf7d0")
arrow(12, 36, 12, 30.5); arrow(12, 21, 12, 16.5)

box(40, 36, 20, 8, "User query\n(CISI.QRY / CLI)", "#dbeafe")
box(40, 21, 20, 9, "Same analyzer\n(identical pipeline)", "#fde68a")
box(66, 21, 30, 9, "Retrieval model\nTF-IDF cosine · BM25 · Query LM\nterm-at-a-time scoring", "#fecaca")
box(66, 5, 30, 11, "Rocchio pseudo-relevance\nfeedback (top-k docs →\nexpanded weighted query)", "#e9d5ff")
box(40, 5, 20, 11, "Ranked list\n(heap top-k)\nscore · docID · title", "#bbf7d0")
arrow(50, 36, 50, 30.5); arrow(60, 25.5, 66, 25.5)
arrow(81, 21, 81, 16.5, "top-k"); arrow(66, 10.5, 66, 24, None)
arrow(75, 21, 60, 13, "final scores")
arrow(22, 10.5, 40, 25, None)
ax.text(30, 20, "postings\nlookup", fontsize=7.5, style="italic", ha="center")
ax.text(50, 1, "Evaluation (Ch. 8): P@k · R@k · MAP · nDCG · MRR · 11-pt PR using CISI.REL judgments", ha="center", fontsize=8.5,
        bbox=dict(boxstyle="round", fc="#f1f5f9", ec="#64748b"))
plt.tight_layout(); plt.savefig("results/fig_architecture.png", dpi=170)
