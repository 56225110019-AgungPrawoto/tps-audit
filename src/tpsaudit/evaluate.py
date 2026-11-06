"""Assemble the long response table (all arms, all runs, all checks) and run the clustered statistics."""
import json
import numpy as np
import pandas as pd
from . import analyses, tps
from .audit import has_zh, zh_share, words

def standardize_existing(df):
    """Stage 4 evaluation_per_sample_epoch{N}.csv (model, run already set by data_io.load_existing_runs) -> standard columns.
    query_id = position within (model, run); Stage 4 writes every model in dataset order."""
    d = df.copy()
    d["query_id"] = d.groupby(["model", "run"]).cumcount()
    d["hallucination_flag"] = d.hallucination_flag.astype(str).str.lower().eq("true")
    for c in ("dahl_score", "dim1_relevance", "dim2_grounding", "dim3_lexical"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["response"] = d.response.fillna("").astype(str)
    return d[["query_id", "run", "model", "response", "dahl_score", "dim1_relevance", "dim2_grounding", "dim3_lexical", "hallucination_flag"]]

def finish(df):
    d = df.copy()
    d["words"] = d.response.map(words); d["has_zh"] = d.response.map(has_zh).astype(int); d["zh_share"] = d.response.map(zh_share)
    d["flagged"] = d.hallucination_flag.astype(int)                    # primary: composite Stage-4 flag (see tps.hallucination_flag)
    d["flag_score_lt_050"] = (d.dahl_score.fillna(0) < 0.50).astype(int)  # secondary: score alone
    return d

def recheck_flags(df):
    """The stored composite flag must equal the rule recomputed from the stored scores (consistency check of the original run)."""
    bad = 0
    for r in df.itertuples():
        det = {"dim1_relevance": r.dim1_relevance, "dim2_grounding": r.dim2_grounding}
        det = {k: v for k, v in det.items() if pd.notna(v)}
        f, _ = tps.hallucination_flag(r.response, None if pd.isna(r.dahl_score) else r.dahl_score, det, ["x"])
        bad += int(f != bool(r.hallucination_flag))
    return {"n": len(df), "mismatches": bad}

def add_rouge(df, eval_df):
    """ROUGE-1 F1 with the rouge-score package, use_stemmer=False (as in Stage 4 Cell 10)."""
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=False)
    ref = dict(zip(eval_df.query_id, eval_df.reference))
    d = df.copy(); d["rouge1"] = [sc.score(ref[q], r)["rouge1"].fmeasure if ref.get(q) else np.nan for q, r in zip(d.query_id, d.response)]
    return d

def summarize(df, arms, flag_cols=("flagged", "flag_score_lt_050", "nli_flag_original", "nli_flag_pair", "dahl_flag", "judge_flag"),
              n_perm=100000, n_boot=3000):
    """Per arm vs baseline: rates, exact counts, clustered tests, language drift, response length."""
    out = {}
    for arm in arms:
        sub = df[df.model.isin(["baseline", arm])]
        res = {"language": {"baseline_zh": int(df[df.model == "baseline"].has_zh.sum()), "arm_zh": int(df[df.model == arm].has_zh.sum()),
                            "predominantly_zh_baseline": int((df[df.model == "baseline"].zh_share > .5).sum()),
                            "predominantly_zh_arm": int((df[df.model == arm].zh_share > .5).sum()), "n_per_model": int((df.model == arm).sum())},
               "mean_words": {"baseline": float(df[df.model == "baseline"].words.mean()), "arm": float(df[df.model == arm].words.mean())}, "rates": {}}
        for c in flag_cols:
            if c in sub and sub[c].notna().any():
                res["rates"][c] = {m: float(sub[sub.model == m][c].mean() * 100) for m in ("baseline", arm)}
        if "rouge1" in sub:
            res["rouge1"] = {m: float(sub[sub.model == m].rouge1.mean()) for m in ("baseline", arm)}
        res["paired_flag"] = analyses.paired(sub[["query_id", "run", "model", "flagged", "words"]], n_perm=n_perm, n_boot=n_boot, treat=arm)
        nz = sub[sub.has_zh == 0]
        ok = nz.groupby(["query_id", "run"]).model.nunique(); keep = ok[ok == 2].index
        if len(keep):
            nz = nz.set_index(["query_id", "run"]).loc[keep].reset_index()
            res["without_chinese_n"] = {m: int((nz.model == m).sum()) for m in ("baseline", arm)}
            res["paired_flag_without_chinese"] = analyses.paired(nz[["query_id", "run", "model", "flagged", "words"]], n_perm=n_perm, n_boot=n_boot, treat=arm)
        out[arm] = res
    return out

def jsonable(o):
    if isinstance(o, dict): return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    return o

def save(o, path):
    with open(path, "w", encoding="utf8") as f: json.dump(jsonable(o), f, indent=2, ensure_ascii=False)
