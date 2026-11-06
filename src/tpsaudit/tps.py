"""Triple-Proxy Score re-implemented from the manuscript (Eq. 1-2).

TPS(y) = 0.4*Sg + 0.4*Sr + 0.2*Sl
  Sg: clipped cosine(response, context)   Sr: clipped cosine(response, query)
  Sl: share of unique response keywords (stopwords and tokens < 4 characters removed) found in the context.

IMPORTANT: this is a re-implementation. Run scripts/check_tps_reproduction.py to confirm it reproduces
the scores stored with your pairs before trusting any new number (context embedding details matter).
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd

_STOP_FILE = Path(__file__).resolve().parents[2] / "data" / "stopwords_id.txt"
_TOKEN = re.compile(r"[a-zA-Z0-9]+")

def load_stopwords(path=None):
    p = Path(path) if path else _STOP_FILE
    return set(w.strip().lower() for w in p.read_text(encoding="utf8").split() if w.strip())

def keywords(text, stop, min_len=4):
    return {t for t in _TOKEN.findall((text or "").lower()) if len(t) >= min_len and t not in stop}

def lexical(response, context, stop, min_len=4):
    ty = keywords(response, stop, min_len)
    if not ty:
        return 0.0
    return len(ty & keywords(context, stop, min_len)) / len(ty)

def _cos_clip(a, b):
    return np.clip((a * b).sum(1), 0.0, 1.0)

class Scorer:
    """embed: callable(list[str]) -> L2-normalised ndarray. Default loads sentence-transformers lazily."""
    def __init__(self, embed=None, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 weights=None, stopwords=None, min_kw_len=4, device=None):
        self.w = weights or {"sg": 0.4, "sr": 0.4, "sl": 0.2}
        self.stop = stopwords if stopwords is not None else load_stopwords()
        self.min_len = min_kw_len
        if embed is None:
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer(model_name, device=device)
            embed = lambda xs: m.encode(list(xs), normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        self.embed = embed

    def score(self, responses, queries, contexts):
        r, q, c = (self.embed(list(x)) for x in (responses, queries, contexts))
        sg, sr = _cos_clip(r, c), _cos_clip(r, q)
        sl = np.array([lexical(a, b, self.stop, self.min_len) for a, b in zip(responses, contexts)])
        tps = self.w["sg"] * sg + self.w["sr"] * sr + self.w["sl"] * sl
        return pd.DataFrame({"sg": sg, "sr": sr, "sl": sl, "tps": tps})
