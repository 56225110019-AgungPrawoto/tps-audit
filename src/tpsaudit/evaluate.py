"""Assemble all response-level flags and run the clustered statistics for every trained arm."""
import json
import numpy as np
import pandas as pd
from . import analyses

def add_tps(df, eval_df, scorer, flag_threshold=0.50):
    m = df.merge(eval_df[["query_id", "question", "context"]], on="query_id")
    s = scorer.score(m.response, m.question, m.context)
    m = pd.concat([m.reset_index(drop=True), s], axis=1)
    m["flagged"] = (m.tps < flag_threshold).astype(int)      # primary: TPS flag rate
    return m

def summarize(df, arms, flag_cols=("flagged", "nli_flag", "dahl_flag", "judge_flag"), n_perm=100000, n_boot=3000):
    """Per arm vs baseline: rates, exact counts, clustered tests (TPS flag), language drift counts."""
    out = {}
    for arm in arms:
        sub = df[df.model.isin(["baseline", arm])]
        res = {"language": {"baseline_zh": int(df[df.model == "baseline"].has_zh.sum()), "arm_zh": int(df[df.model == arm].has_zh.sum()),
                            "n_per_model": int((df.model == arm).sum())},
               "mean_words": {"baseline": float(df[df.model == "baseline"].words.mean()), "arm": float(df[df.model == arm].words.mean())},
               "rates": {}}
        for c in flag_cols:
            if c in sub and sub[c].notna().any():
                res["rates"][c] = {m: float(sub[sub.model == m][c].mean() * 100) for m in ("baseline", arm)}
        res["paired_tps_flag"] = analyses.paired(sub[["query_id", "run", "model", "flagged", "words"]], n_perm=n_perm, n_boot=n_boot, treat=arm)
        nz = sub[sub.has_zh == 0]                                # excluding responses with Chinese characters
        pairs_ok = nz.groupby(["query_id", "run"]).model.nunique(); keep = pairs_ok[pairs_ok == 2].index
        nz = nz.set_index(["query_id", "run"]).loc[keep].reset_index()
        res["without_chinese_n"] = {m: int((nz.model == m).sum()) for m in ("baseline", arm)}
        if len(nz): res["paired_tps_flag_without_chinese"] = analyses.paired(nz[["query_id", "run", "model", "flagged", "words"]], n_perm=n_perm, n_boot=n_boot, treat=arm)
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

def rouge1_f(a, b):
    """Simple unigram-overlap F1 on lower-cased word tokens (not the rouge-score package; say so in the paper)."""
    import re
    from collections import Counter
    ta, tb = Counter(re.findall(r"\w+", (a or "").lower())), Counter(re.findall(r"\w+", (b or "").lower()))
    ov = sum((ta & tb).values())
    if not ov: return 0.0
    p, r = ov / sum(ta.values()), ov / sum(tb.values())
    return 2 * p * r / (p + r)

def add_rouge(df, eval_df):
    if "reference" not in eval_df: return df
    m = df.merge(eval_df[["query_id", "reference"]], on="query_id")
    m["rouge1"] = [rouge1_f(a, b) for a, b in zip(m.response, m.reference)]
    return m
