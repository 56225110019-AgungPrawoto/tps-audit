"""Triple-Proxy Score exactly as implemented in the original notebooks.

Two near-identical implementations exist in the original code and are reproduced separately:
  * Stage 2  compute_sfinal()          : scores the candidates when the dataset is built
  * Stage 4  calculate_DAHL_score()    : scores model responses at evaluation ("DAHL proxy" in that notebook = TPS in the paper)
They differ in how the context string is built (see docs/CODE_vs_PAPER.md, items 4 and 5).

The sentence embedder is injected (`embed`) so the logic can be unit-tested without a GPU/model download.
`embed(list[str]) -> np.ndarray` must return L2-normalised rows (SentenceTransformer.encode(..., normalize_embeddings=True)).
"""
import re
from pathlib import Path
import numpy as np

STOP_FILE = Path(__file__).resolve().parents[2] / "data" / "stopwords_id.txt"
STOP_WORDS_ID = set(w.strip() for w in STOP_FILE.read_text(encoding="utf8").split() if w.strip())   # 89 words (verified)
MIN_TOKEN_LEN = 4
W_SG, W_SR, W_LEX = 0.4, 0.4, 0.2
HALLUCINATION_THRESHOLD, DIM1_MIN, DIM2_MIN = 0.50, 0.35, 0.20          # Stage 4 [FIX-THRESHOLD-2/-DIM1-5/-GROUNDING-3]
MAX_CONTEXT_CHARS_PER_SOURCE = 1200

def tokenize_for_lexical(text, stop=STOP_WORDS_ID, min_len=MIN_TOKEN_LEN):
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {t for t in tokens if len(t) >= min_len and t not in stop}

def lexical_overlap(response, context):
    tr, tc = tokenize_for_lexical(response), tokenize_for_lexical(context)
    return len(tr & tc) / len(tr) if tr else 0.0

def default_embedder(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu"):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name, device=device)
    f = lambda xs: m.encode(list(xs), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    f.model = m
    return f

# ------------------------------------------------------------------ Stage 2 (dataset construction)
def build_dataset_context(pages, get_text, max_chars=MAX_CONTEXT_CHARS_PER_SOURCE):
    """Stage 2 build_context(): pages = [{pdf_name, page_number(0-based), score, ...}], get_text(page)->str.
    The header is added AFTER truncation; pages with no text are skipped; blocks are joined by a blank line."""
    blocks = []
    for i, p in enumerate(pages, start=1):
        text = get_text(p)
        if not text:
            continue
        text = text[:max_chars]
        hal = (p["page_number"] + 1) if p.get("page_number") is not None else "?"
        blocks.append(f"[Sumber {i}: {p['pdf_name']}.pdf - Halaman {hal} | skor={p['score']:.3f}]\n{text}")
    return "\n\n".join(blocks)

def compute_sfinal(response, context, query, embed):
    e = embed([response, context, query])
    sg = float(max(0.0, np.dot(e[0], e[1]))); sr = float(max(0.0, np.dot(e[0], e[2])))
    lex = lexical_overlap(response, context)
    return {"sem_grounding": round(sg, 4), "sem_relevance": round(sr, 4), "lexical": round(lex, 4),
            "s_final": round(W_SG * sg + W_SR * sr + W_LEX * lex, 4)}

def select_dpo_pair(candidates, context, query, embed, margin_threshold=0.15):
    """Stage 2 select_dpo_pair(): chosen = best grounded candidate, rejected = (the single) ungrounded candidate."""
    scored = [{**c, **compute_sfinal(c["text"], context, query, embed)} for c in candidates]
    w = [c for c in scored if c["with_context"]]; wo = [c for c in scored if not c["with_context"]]
    if not w or not wo:
        return None
    best, worst = max(w, key=lambda x: x["s_final"]), min(wo, key=lambda x: x["s_final"])
    margin = best["s_final"] - worst["s_final"]
    return None if margin < margin_threshold else {"chosen": best, "rejected": worst, "margin": round(margin, 4)}

# ------------------------------------------------------------------ Stage 4 (evaluation scoring)
def eval_score(response, query, contexts, embed):
    """Stage 4 calculate_DAHL_score(). contexts = [{filename, text, ...}] (Stage 4 precomputed_contexts).
    NOTE: the context string is the page texts joined by a single space, WITHOUT the '[Sumber ...]' headers."""
    if not response or len(response.strip()) < 5:
        return 0.0, {}
    er, eq = embed([response, query])
    dim1 = max(0.0, float(np.dot(er, eq)))
    if contexts:
        proxy = " ".join(c["text"] for c in contexts if c.get("text"))
        if not proxy.strip():
            proxy = " ".join(c["filename"].replace("_", " ").split(".")[0] for c in contexts)
        ec = embed([proxy])[0]
        dim2 = max(0.0, float(np.dot(er, ec)))
        dim3 = lexical_overlap(response, proxy)
    else:
        dim2 = dim3 = 0.0
    score = round(0.40 * dim1 + 0.40 * dim2 + 0.20 * dim3, 4)
    return score, {"dim1_relevance": round(dim1, 4), "dim2_grounding": round(dim2, 4), "dim3_lexical": round(dim3, 4)}

def hallucination_flag(response, score, details, filenames):
    """Stage 4 hallucination_audit(): a response is flagged if ANY of:
       empty response | score is None | score < 0.50 | dim1_relevance < 0.35 | (has contexts and dim2_grounding < 0.20).
    Returns (flagged: bool, reasons: list[str]). The paper must call this the 'TPS flag' only if it describes this rule."""
    reasons = []
    if not response:
        reasons.append("empty")
    if score is None:
        reasons.append("score_none")
    elif score < HALLUCINATION_THRESHOLD:
        reasons.append("score<0.50")
    if details.get("dim1_relevance", 1.0) < DIM1_MIN:
        reasons.append("relevance<0.35")
    if filenames and details.get("dim2_grounding", 1.0) < DIM2_MIN:
        reasons.append("grounding<0.20")
    return (len(reasons) > 0), reasons

def build_eval_context_text(contexts, prompts):
    """Stage 4: the text placed in the SYSTEM message (header + blocks) or the 'no context' sentence."""
    if not contexts:
        return prompts["stage4_no_context"]
    blocks = []
    for i, c in enumerate(contexts, start=1):
        hal = (c["page_number"] + 1) if c.get("page_number") is not None else "?"
        blocks.append(prompts["stage4_context_block"].format(i=i, pdf_name=c["pdf_name"], page=hal, score=c["score"], text=c["text"]))
    return prompts["stage4_context_header"] + "\n\n".join(blocks)

def context_truncation_report(embed, texts, name="context"):
    """MiniLM embeds only its first `max_seq_length` word pieces (128 by default for this model, VERIFY with the printed value).
    Reports how much of each text the embedder actually sees; relevant to Reviewer 2 (context truncation affects the TPS)."""
    m = getattr(embed, "model", None)
    if m is None:
        return {}
    limit = m.max_seq_length; tok = m.tokenizer
    n = np.array([len(tok(t, add_special_tokens=True)["input_ids"]) for t in texts])
    return {f"{name}_max_seq_length": int(limit), f"{name}_tokens_mean": float(n.mean()), f"{name}_share_truncated_%": float(100 * (n > limit).mean()),
            f"{name}_visible_fraction_mean": float(np.minimum(1.0, limit / n).mean())}
