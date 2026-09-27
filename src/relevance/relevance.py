import re

KEYWORDS = [
    "flood", "flooding", "strike", "fire", "disruption",
    "shutdown", "delay", "regulation", "regulatory",
    "shipment", "cargo", "factory", "import", "supply"
]

def calculate_relevance(text):
    text = str(text or "").lower()
    hits = sum(1 for k in KEYWORDS if re.search(r"\b" + re.escape(k) + r"\b", text))
    score = min(hits / 5.0, 1.0)
    return {"score": round(score, 3), "label": hits > 0}
