"""NLI check and "DAHL-style" atomic-fact check, as implemented in the original Stage 4 notebook.

The atomic block below (splitter, de-duplication, checker, scoring) is copied VERBATIM from the original
notebook (Cell 9B); only the imports at the top are new. Differences from the DAHL paper (Seo et al., 2024):
  * evidence = the retrieved document text (first 4,000 characters), not an external knowledge source;
  * splitter AND checker are the same local Qwen2-7B-Instruct (4-bit), greedy, with the base model used as the FIXED checker
    for both baseline and DPO responses (Cell 19) -- same family as the generator, so NOT independent of it;
  * the checker answers true / false / unknown per statement; 'unknown' units are dropped; a response whose units are all
    unknown (or whose label count mismatches) gets score None and is flagged.
The NLI check (original) feeds ONE concatenated string "premise </s> hypothesis" to a cross-encoder text-classification
pipeline instead of a (premise, hypothesis) pair, and sees only the first 512 characters of the concatenated context and the
first 256 characters of the response. `mode="pair"` below is the standard pair input, run as a sensitivity check.
"""
import re
try:
    import torch
except ImportError:          # CPU-only tests
    torch = None
from .tps import tokenize_for_lexical



# ============================================================== NLI (Stage 4, Cell 20)
NLI_MODEL_ID = "cross-encoder/nli-deberta-v3-small"
NLI_ENTAIL_THRESHOLD = 0.50
NLI_MAX_CHARS = 512

def load_nli_pipeline(model_id=NLI_MODEL_ID):
    from transformers import pipeline
    return pipeline("text-classification", model=model_id, device=-1, truncation=True, max_length=512, top_k=None)

def _truncate_pair(context_text, response):
    half = NLI_MAX_CHARS // 2
    return (context_text[:NLI_MAX_CHARS] if context_text else ""), (response[:half] if response else "")

def nli_flags(nli_pipe, items, mode="original", batch_size=8, threshold=NLI_ENTAIL_THRESHOLD):
    """items: list of (context_text, response). mode: 'original' (single string with '</s>') | 'pair' (premise/hypothesis pair).
    Returns list of dicts with nli_label, nli_entail_score, nli_contradict_score, nli_hallucination."""
    out = [None] * len(items); batch_idx, batch_in = [], []
    for i, (ctx, resp) in enumerate(items):
        resp = (resp or "").strip()
        if not resp or len(resp) < 5 or not (ctx or "").strip():
            out[i] = {"nli_label": "UNKNOWN", "nli_entail_score": 0.0, "nli_contradict_score": 0.0, "nli_hallucination": True}
            continue
        c, r = _truncate_pair(ctx, resp)
        batch_idx.append(i); batch_in.append(f"{c} </s> {r}" if mode == "original" else {"text": c, "text_pair": r})
    preds = []
    for s in range(0, len(batch_in), batch_size):
        try:
            preds.extend(nli_pipe(batch_in[s:s + batch_size]))
        except Exception:
            preds.extend([None] * len(batch_in[s:s + batch_size]))
    for i, p in zip(batch_idx, preds):
        if p is None:
            out[i] = {"nli_label": "UNKNOWN", "nli_entail_score": 0.0, "nli_contradict_score": 0.0, "nli_hallucination": True}; continue
        m = {x["label"].upper(): x["score"] for x in p}
        best = max(m, key=m.get); ent = m.get("ENTAILMENT", 0.0); con = m.get("CONTRADICTION", 0.0)
        out[i] = {"nli_label": best, "nli_entail_score": round(ent, 4), "nli_contradict_score": round(con, 4),
                  "nli_hallucination": bool(best == "CONTRADICTION" or ent < threshold)}
    return out

# ============================================================== DAHL-style atomic check (Stage 4, Cell 9B, VERBATIM)
NONCOMMITTAL_PATTERNS = [
    r"tidak (dapat|bisa) (saya )?jawab",
    r"tidak (ada|tersedia) informasi",
    r"saya tidak (tahu|mengetahui)",
    r"tidak dapat ditemukan",
    r"maaf,? saya tidak",
    r"i don'?t know",
    r"cannot be answered",
    r"no information (is )?available",
]
NONCOMMITTAL_RE = re.compile("|".join(NONCOMMITTAL_PATTERNS), re.IGNORECASE)


def is_noncommittal(unit: str) -> bool:
    return bool(NONCOMMITTAL_RE.search(unit.strip()))


# --- Splitter prompt (adaptasi Figure 9 paper) ---
SPLITTER_PROMPT_TEMPLATE = """Pecah teks di bawah menjadi kalimat-kalimat yang lebih kecil, \
masing-masing hanya memuat SATU fakta tunggal. Hindari penggunaan kata \
ganti seperti "ini", "itu", "tersebut" — sebutkan rujukannya secara \
eksplisit. Pisahkan setiap kalimat hasil dengan baris baru. Jangan \
menambahkan penomoran, penjelasan, atau komentar apa pun.

TEKS:
{response}

KALIMAT TERPISAH:"""


def split_into_atomic_units(response: str, model, tokenizer, max_new_tokens: int = 400) -> list:
    """Step1 — Splitter. Memecah respons menjadi atomic units via LLM."""
    if not response or len(response.strip()) < 5:
        return []

    prompt = SPLITTER_PROMPT_TEMPLATE.format(response=response.strip())
    messages = [{"role": "user", "content": prompt}]
    text_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        text_input, return_tensors="pt", truncation=True, max_length=2048
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # deterministik untuk Splitter (konsisten dgn paper)
            pad_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(
        out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()
    del inputs, out

    units = []
    for line in raw.split("\n"):
        line = re.sub(r"^\s*[\-\*\d]+[\.\)]?\s*", "", line).strip()
        if len(line) >= 5:
            units.append(line)
    return units


def dedup_atomic_units(units: list, jaccard_threshold: float = 0.85) -> list:
    """Hapus atomic unit yang informasinya duplikat (paper: 'sentences
    containing identical information are removed')."""
    kept, kept_tokens = [], []
    for u in units:
        t = tokenize_for_lexical(u)
        is_dup = any(
            (len(t & kt) / len(t | kt) if (t | kt) else 0) >= jaccard_threshold
            for kt in kept_tokens
        )
        if not is_dup:
            kept.append(u)
            kept_tokens.append(t)
    return kept


# --- Checker (grounded ke konteks dokumen yang di-retrieve ColPali) ---
CHECKER_PROMPT_TEMPLATE = """Berikut adalah KONTEKS DOKUMEN yang menjadi satu-satunya sumber \
kebenaran. Untuk setiap STATEMENT yang diberi nomor, tentukan apakah \
statement tersebut DIDUKUNG oleh konteks (jawab "true"), DIBANTAH/TIDAK \
SESUAI dengan konteks (jawab "false"), atau TIDAK DAPAT ditentukan dari \
konteks yang ada (jawab "unknown"). Jangan gunakan pengetahuan di luar \
konteks yang diberikan. Jawab HANYA dengan format:
1. true/false/unknown
2. true/false/unknown
... (dan seterusnya, satu baris per statement, tanpa penjelasan lain)

KONTEKS DOKUMEN:
{context}

STATEMENTS:
{numbered_units}

JAWABAN:"""


def check_atomic_units_against_context(
    units: list, context_text: str, model, tokenizer, max_new_tokens: int = 300
) -> list:
    """Step2 — Checker. Cek faktualitas setiap atomic unit terhadap
    konteks dokumen yang di-retrieve."""
    if not units:
        return []
    if not context_text or not context_text.strip():
        return ["unknown"] * len(units)

    numbered = "\n".join(f"{i+1}. {u}" for i, u in enumerate(units))
    prompt = CHECKER_PROMPT_TEMPLATE.format(
        context=context_text[:4000], numbered_units=numbered
    )
    messages = [{"role": "user", "content": prompt}]
    text_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        text_input, return_tensors="pt", truncation=True, max_length=3072
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # deterministik untuk Checker
            pad_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(
        out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()
    del inputs, out

    parsed = {}
    for line in raw.split("\n"):
        m = re.match(r"\s*(\d+)[\.\)]\s*(true|false|unknown)", line, re.IGNORECASE)
        if m:
            parsed[int(m.group(1))] = m.group(2).lower()

    # [Step3 paper] jika jumlah label tidak cocok dengan jumlah unit,
    # respons dianggap tidak bisa dinilai (discrepancy → exclude)
    if len(parsed) != len(units):
        return ["unknown"] * len(units)

    return [parsed[i + 1] for i in range(len(units))]


def calculate_dahl_score_atomic(
    response: str, contexts: list, model, tokenizer,
    checker_model=None, checker_tokenizer=None,
):
    """
    Step1–3 lengkap → DAHL Score (persis metodologi paper) untuk satu respons.

    Parameters
    ----------
    checker_model, checker_tokenizer : optional
        Kalau diisi, dipakai untuk Splitter & Checker SEBAGAI GANTI
        `model`/`tokenizer` (yang dipakai generate jawaban). Tujuannya:
        supaya model yang dievaluasi tidak menilai jawabannya sendiri —
        pakai 1 model fixed yang sama untuk cek Baseline maupun DPO
        (lihat CELL 19). Kalau None, fallback ke `model`/`tokenizer`
        (model menilai dirinya sendiri).

    Returns
    -------
    (score: float | None, details: dict)
        score None jika seluruh unit noncommittal/unknown — respons
        tersebut dikeluarkan dari rata-rata akhir, sesuai paper.
    """
    if not response or len(response.strip()) < 5:
        return None, {"reason": "empty_response", "atomic_units": [], "labels": []}

    cm = checker_model if checker_model is not None else model
    ct = checker_tokenizer if checker_tokenizer is not None else tokenizer

    context_text = (
        "\n\n".join(c["text"] for c in contexts if c.get("text")) if contexts else ""
    )

    units = split_into_atomic_units(response, cm, ct)
    units = [u for u in units if not is_noncommittal(u)]
    units = dedup_atomic_units(units)

    if not units:
        return None, {
            "reason": "no_atomic_units_after_filtering",
            "atomic_units": [],
            "labels": [],
        }

    labels = check_atomic_units_against_context(units, context_text, cm, ct)

    valid = [(u, l) for u, l in zip(units, labels) if l in ("true", "false")]
    if not valid:
        return None, {
            "reason": "all_units_unknown",
            "atomic_units": units,
            "labels": labels,
        }

    n_true = sum(1 for _, l in valid if l == "true")
    score = round(n_true / len(valid), 4)

    return score, {
        "atomic_units"  : units,
        "labels"        : labels,
        "n_total_units" : len(units),
        "n_valid_units" : len(valid),
        "n_true"        : n_true,
    }




DAHL_HALLUCINATION_THRESHOLD = 0.50

def dahl_flag(response, score, threshold=DAHL_HALLUCINATION_THRESHOLD):
    """Stage 4 dahl_hallucination_audit(): flagged if the response is empty, the score is None, or the score < threshold."""
    return bool((not response) or score is None or score < threshold)
