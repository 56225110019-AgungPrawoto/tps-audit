"""Independent LLM-judge validation of preference labels and of model responses.

Design (pre-specified, see docs/PROTOCOL.md):
  * judge models come from a different family than the generator (Qwen2) and are run greedy;
  * every pair is judged twice, in both answer orders, with the retrieved context as the only evidence;
  * the final judgement is 'chosen' / 'rejected' only if both orders agree, otherwise 'inconsistent';
  * rubric: factual support, completeness, relevance, language quality (1-5), plus overall preference;
  * answer length is explicitly NOT a criterion; agreement is also reported by length tercile.
"""
import json, re
import numpy as np
import pandas as pd

SYSTEM = ("Anda adalah penilai independen. Nilailah jawaban HANYA berdasarkan KONTEKS yang diberikan. "
          "Jawaban yang menyatakan hal yang tidak didukung atau bertentangan dengan konteks (misalnya tenggat, syarat, atau aturan yang salah) "
          "mendapat nilai dukungan faktual rendah. Panjang jawaban BUKAN kriteria. Jawaban yang baik menggunakan bahasa Indonesia. "
          "Balas hanya dengan satu objek JSON.")

PAIR_TMPL = """KONTEKS:
{context}

PERTANYAAN:
{question}

JAWABAN A:
{a}

JAWABAN B:
{b}

Beri nilai 1-5 untuk setiap jawaban pada: factual_support (dukungan faktual oleh konteks), completeness (kelengkapan), relevance (relevansi), language (kualitas bahasa Indonesia).
Lalu pilih jawaban yang lebih baik secara keseluruhan: "A", "B", atau "tie".
Format JSON: {{"A": {{"factual_support": n, "completeness": n, "relevance": n, "language": n}}, "B": {{"factual_support": n, "completeness": n, "relevance": n, "language": n}}, "preferred": "A|B|tie", "reason": "satu kalimat"}}"""

SINGLE_TMPL = """KONTEKS:
{context}

PERTANYAAN:
{question}

JAWABAN:
{a}

Nilai 1-5: factual_support, completeness, relevance, language. Format JSON: {{"factual_support": n, "completeness": n, "relevance": n, "language": n, "reason": "satu kalimat"}}"""

_JSON = re.compile(r"\{.*\}", re.S)

def parse_json(text):
    m = _JSON.search(text or "")
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception:
        try: return json.loads(re.sub(r",\s*}", "}", m.group(0)))
        except Exception: return None

def _msgs(user): return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]

def judge_pairs(backend, df, max_context_chars=6000):
    """df columns: pair_id, question, context, chosen, rejected. Returns one row per pair."""
    reqs = []
    for r in df.itertuples():
        ctx = (r.context or "")[:max_context_chars]
        reqs.append(_msgs(PAIR_TMPL.format(context=ctx, question=r.question, a=r.chosen, b=r.rejected)))   # order 1: chosen = A
        reqs.append(_msgs(PAIR_TMPL.format(context=ctx, question=r.question, a=r.rejected, b=r.chosen)))   # order 2: chosen = B
    raw = backend.generate(reqs, max_new_tokens=300)
    rows = []
    for i, r in enumerate(df.itertuples()):
        p1, p2 = parse_json(raw[2 * i]), parse_json(raw[2 * i + 1])
        v1 = _verdict(p1, chosen_is="A"); v2 = _verdict(p2, chosen_is="B")
        rows.append({"pair_id": r.pair_id, "judge": backend.name, "verdict_order1": v1, "verdict_order2": v2,
                     "final": _final(v1, v2),
                     "fs_chosen": _fs(p1, "A", p2, "B"), "fs_rejected": _fs(p1, "B", p2, "A"),
                     "parse_fail": int(p1 is None) + int(p2 is None)})
    return pd.DataFrame(rows)

def _verdict(p, chosen_is):
    if not p or "preferred" not in p: return "parse_error"
    v = str(p["preferred"]).strip().upper()
    if v == "TIE": return "tie"
    if v not in ("A", "B"): return "parse_error"
    return "chosen" if v == chosen_is else "rejected"

def _final(v1, v2):
    if v1 == v2 and v1 in ("chosen", "rejected", "tie"): return v1
    return "inconsistent"

def _fs(p1, k1, p2, k2):
    vals = []
    for p, k in ((p1, k1), (p2, k2)):
        try: vals.append(float(p[k]["factual_support"]))
        except Exception: pass
    return float(np.mean(vals)) if vals else np.nan

def summarize_pairs(res, pairs):
    """Agreement of one judge with the automatic labels, overall and by stratum."""
    m = res.merge(pairs[["pair_id", "run", "margin", "chosen", "rejected"]], on="pair_id")
    m["clen"] = m.chosen.str.split().str.len(); n = len(m)
    cons = m[m.final.isin(["chosen", "rejected"])]
    out = {"judge": m.judge.iloc[0], "n_pairs": n,
           "agree_chosen_%": 100 * float((m.final == "chosen").mean()),
           "disagree_rejected_%": 100 * float((m.final == "rejected").mean()),
           "tie_%": 100 * float((m.final == "tie").mean()),
           "inconsistent_%": 100 * float((m.final == "inconsistent").mean()),
           "agree_among_consistent_%": 100 * float((cons.final == "chosen").mean()) if len(cons) else np.nan,
           "position_bias_%": 100 * float(((m.verdict_order1 == "chosen") != (m.verdict_order2 == "chosen")).mean()),
           "parse_failures": int(m.parse_fail.sum()),
           "mean_factual_support_chosen": float(m.fs_chosen.mean()), "mean_factual_support_rejected": float(m.fs_rejected.mean())}
    strata = {}
    for name, col in (("run", m.run), ("length_tercile_chosen", pd.qcut(m.clen.rank(method="first"), 3, labels=["short", "mid", "long"])),
                      ("margin_band", pd.cut(m.margin, [0.15, 0.20, 0.25, 10], right=False))):
        strata[name] = m.groupby(col, observed=True).final.apply(lambda s: round(100 * float((s == "chosen").mean()), 1)).to_dict()
    out["agree_chosen_by_stratum_%"] = {k: {str(a): b for a, b in v.items()} for k, v in strata.items()}
    return out

def judge_agreement(res_a, res_b):
    """Judge-vs-judge agreement (substitute for inter-annotator agreement)."""
    from .analyses import kappa
    m = res_a.merge(res_b, on="pair_id", suffixes=("_a", "_b"))
    both = m[m.final_a.isin(["chosen", "rejected"]) & m.final_b.isin(["chosen", "rejected"])]
    return {"n_both_consistent": int(len(both)), "raw_agreement_%": 100 * float((both.final_a == both.final_b).mean()) if len(both) else np.nan,
            "cohen_kappa": float(kappa(both.final_a, both.final_b)) if len(both) else np.nan,
            "both_agree_with_label_%": 100 * float(((m.final_a == "chosen") & (m.final_b == "chosen")).mean())}

def judge_responses(backend, df, flag_max_fs=2, max_context_chars=6000):
    """Single-answer grading of model responses. df: response_id, question, context, response.
    A response is flagged when factual_support <= flag_max_fs (pre-specified)."""
    reqs = [_msgs(SINGLE_TMPL.format(context=(r.context or "")[:max_context_chars], question=r.question, a=r.response)) for r in df.itertuples()]
    raw = backend.generate(reqs, max_new_tokens=200)
    fs = []
    for t in raw:
        p = parse_json(t)
        try: fs.append(float(p["factual_support"]))
        except Exception: fs.append(np.nan)
    out = df[["response_id"]].copy(); out["judge_fs"] = fs
    out["judge_flag"] = (out.judge_fs <= flag_max_fs).astype(float); out.loc[out.judge_fs.isna(), "judge_flag"] = np.nan
    return out
