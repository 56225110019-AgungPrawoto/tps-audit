"""NLI check and DAHL-style atomic-fact check, with every rule pre-specified in configs/default.yaml.

NLI check:    response -> sentences (split on . ! ? followed by space; sentences < 3 words ignored);
              premise = retrieved context cut into overlapping token windows; p_entail(sentence) = max over windows;
              a sentence is supported when p_entail >= entail_threshold; a response is flagged when the share of
              supported sentences < supported_share_min.
DAHL-style:   an open LLM splits the response into atomic facts (prompt below); each fact is verified with the same NLI model
              against the retrieved context; flagged when supported share < supported_share_min; responses with no
              extractable facts are not flagged and are counted separately.
Differences from DAHL (check against Seo et al. before writing the paper): evidence is the retrieved
document pages (no external knowledge base), the verifier is a local NLI model, and the fact extractor is an open 7-9B model.
"""
import re
import numpy as np
import pandas as pd
from .judge import parse_json

SPLIT = re.compile(r"(?<=[.!?])\s+")

def sentences(text, min_words=3):
    return [s.strip() for s in SPLIT.split((text or "").strip()) if len(s.split()) >= min_words]

class NLI:
    def __init__(self, model_id, window_tokens=300, stride=150, device=None):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu"); self.model.to(self.dev)
        lab = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.ent = lab["entailment"]; self.win, self.stride = window_tokens, stride
    def _windows(self, context):
        ids = self.tok(context or "", add_special_tokens=False).input_ids
        if len(ids) <= self.win: return [context or ""]
        return [self.tok.decode(ids[i:i + self.win]) for i in range(0, max(1, len(ids) - self.win + self.stride), self.stride)]
    def p_entail(self, context, hyps, batch=16):
        wins = self._windows(context); best = np.zeros(len(hyps))
        for w in wins:
            for i in range(0, len(hyps), batch):
                h = hyps[i:i + batch]
                enc = self.tok([w] * len(h), h, return_tensors="pt", truncation="only_first", max_length=512, padding=True).to(self.dev)
                with self.torch.no_grad(): p = self.model(**enc).logits.softmax(-1)[:, self.ent].cpu().numpy()
                best[i:i + len(h)] = np.maximum(best[i:i + len(h)], p)
        return best

def nli_flags(nli, df, entail_threshold=0.5, supported_share_min=0.5):
    """df: response_id, context, response."""
    rows = []
    for r in df.itertuples():
        ss = sentences(r.response)
        if not ss: rows.append((r.response_id, np.nan, np.nan, 0)); continue
        sup = float((nli.p_entail(r.context, ss) >= entail_threshold).mean())
        rows.append((r.response_id, sup, float(sup < supported_share_min), len(ss)))
    return pd.DataFrame(rows, columns=["response_id", "nli_supported_share", "nli_flag", "n_sentences"])

FACT_SYSTEM = "Anda memecah jawaban menjadi fakta atomik. Balas hanya dengan JSON."
FACT_TMPL = ('Pecah jawaban berikut menjadi daftar fakta atomik: satu klaim verifikasi sederhana per item, tanpa kata ganti, tanpa opini. '
             'Jika tidak ada klaim faktual, kembalikan daftar kosong.\n\nJAWABAN:\n{a}\n\nFormat: {{"facts": ["...", "..."]}}')

def extract_facts(backend, responses, max_facts=15):
    reqs = [[{"role": "system", "content": FACT_SYSTEM}, {"role": "user", "content": FACT_TMPL.format(a=a)}] for a in responses]
    out = []
    for t in backend.generate(reqs, max_new_tokens=300):
        p = parse_json(t); f = p.get("facts", []) if isinstance(p, dict) else []
        out.append([x for x in f if isinstance(x, str) and x.strip()][:max_facts])
    return out

def dahl_style_flags(nli, backend, df, entail_threshold=0.5, supported_share_min=0.5):
    facts = extract_facts(backend, list(df.response))
    rows = []
    for r, fs in zip(df.itertuples(), facts):
        if not fs: rows.append((r.response_id, np.nan, np.nan, 0)); continue
        sup = float((nli.p_entail(r.context, fs) >= entail_threshold).mean())
        rows.append((r.response_id, sup, float(sup < supported_share_min), len(fs)))
    return pd.DataFrame(rows, columns=["response_id", "dahl_supported_share", "dahl_flag", "n_facts"])
