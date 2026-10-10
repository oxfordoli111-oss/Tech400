"""Parsers for the CISI collection (SMART format): CISI.ALL, CISI.QRY, CISI.REL."""
import re
from collections import defaultdict

_FIELD = re.compile(r"^\.([A-Z])\s*(\d+)?\s*$")


def _parse_smart(path):
    """Yield dicts {id, T, A, W, ...} for each '.I n' record in a SMART file."""
    rec, field = None, None
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.rstrip("\r\n")
            m = _FIELD.match(line)
            if m:
                tag, num = m.group(1), m.group(2)
                if tag == "I":
                    if rec:
                        yield rec
                    rec, field = {"id": int(num)}, None
                else:
                    field = tag
                    rec.setdefault(field, [])
                continue
            if rec is not None and field is not None:
                rec[field].append(line)
    if rec:
        yield rec


def load_documents(path):
    """Return {doc_id: {'title', 'author', 'text'}}; text = title + abstract."""
    docs = {}
    for r in _parse_smart(path):
        title = " ".join(r.get("T", [])).strip()
        abstract = " ".join(r.get("W", [])).strip()
        docs[r["id"]] = {
            "title": title,
            "author": " ".join(r.get("A", [])).strip(),
            "text": f"{title} {abstract}".strip(),
        }
    return docs


def load_queries(path):
    return {r["id"]: " ".join(r.get("W", [])).strip() for r in _parse_smart(path)}


def load_qrels(path):
    """CISI.REL lines: query_id doc_id 0 0.000. Every listed pair is relevant."""
    qrels = defaultdict(set)
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 2:
                qrels[int(parts[0])].add(int(parts[1]))
    return dict(qrels)
