"""Helpers that turn code or console text into 'screenshot' images for the reports."""
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"   # monospace font bundled with most Linux installs


def render(text, path, title, size=14):
    """Draw `text` on a dark terminal-style window and save it as a PNG."""
    f = ImageFont.truetype(FONT, size)
    lines = text.rstrip().split("\n")
    w = max(int(f.getlength(l)) for l in lines + [title]) + 40
    h = (len(lines) + 2) * (size + 6) + 30
    img = Image.new("RGB", (w, h), "#1e1e2e")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w, 28], fill="#313244")
    for i, c in enumerate(["#f38ba8", "#f9e2af", "#a6e3a1"]):          # fake window buttons
        d.ellipse([10 + i * 18, 8, 20 + i * 18, 18], fill=c)
    d.text((80, 6), title, font=ImageFont.truetype(FONT, 12), fill="#cdd6f4")
    y = 42
    for l in lines:
        d.text((20, y), l, font=f, fill="#cdd6f4")
        y += size + 6
    img.save(path)


def block(path, start, end=None, extra=0):
    """Return the lines of `path` from the first line containing `start` up to (not including)
    the first later line containing `end`; `extra` adds more lines after the end marker."""
    lines = open(path, encoding="utf-8").read().split("\n")
    i = next(k for k, l in enumerate(lines) if start in l)
    j = next(k for k in range(i + 1, len(lines)) if end in lines[k]) if end else len(lines)
    return "\n".join(lines[i:j + extra])
