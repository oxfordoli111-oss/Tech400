"""Render code/console excerpts as images for the report (swap for real screenshots if preferred)."""
from PIL import Image, ImageDraw, ImageFont
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def render(text, path, title, size=15):
    f = ImageFont.truetype(FONT, size)
    lines = text.rstrip().split("\n")
    w = max(int(f.getlength(l)) for l in lines + [title]) + 40
    h = (len(lines) + 2) * (size + 6) + 30
    img = Image.new("RGB", (w, h), "#1e1e2e")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w, 28], fill="#313244")
    for i, c in enumerate(["#f38ba8", "#f9e2af", "#a6e3a1"]):
        d.ellipse([10 + i * 18, 8, 20 + i * 18, 18], fill=c)
    d.text((80, 6), title, font=ImageFont.truetype(FONT, 12), fill="#cdd6f4")
    y = 42
    for l in lines:
        d.text((20, y), l, font=f, fill="#cdd6f4"); y += size + 6
    img.save(path)


def grab(path, start, end):
    return "".join(open(path).readlines()[start - 1:end])

log = open("results/run_log.txt").read().split("\n")
idx = log.index("INDEX STATISTICS")
render("\n".join(log[idx:idx + 6]), "results/snapshots/s1_index_stats.png", "python run_experiments.py  (index statistics)", 14)
m = log.index("MODEL x PREPROCESSING (all judged queries)")
render("\n".join(log[m:m + 11]), "results/snapshots/s2_model_comparison.png", "python run_experiments.py  (model comparison)", 13)
f = log.index("FINAL SYSTEMS")
render("\n".join(log[f - 3:f + 6]), "results/snapshots/s3_final_systems.png", "python run_experiments.py  (tuning + final systems)", 13)
render(open("results/sample_query.txt").read(), "results/snapshots/s4_sample_query.png", "sample query walk-through", 14)
render(open("results/cli_demo.txt").read(), "results/snapshots/s5_cli.png", 'python search_cli.py "automatic indexing of library catalogues" -k 5', 15)
render(grab("src/cisi_ir/preprocess.py", 18, 42), "results/snapshots/c1_preprocess.png", "src/cisi_ir/preprocess.py", 14)
render(grab("src/cisi_ir/index.py", 19, 33), "results/snapshots/c2_index_build.png", "src/cisi_ir/index.py  (build)", 14)
render(grab("src/cisi_ir/models.py", 45, 64), "results/snapshots/c3_bm25.png", "src/cisi_ir/models.py  (BM25, term-at-a-time)", 14)
render(grab("src/cisi_ir/feedback.py", 22, 40), "results/snapshots/c4_rocchio.png", "src/cisi_ir/feedback.py  (Rocchio PRF)", 14)
print("ok")
