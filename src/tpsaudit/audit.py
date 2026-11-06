"""Dataset audit: language contamination, duplicates, length, ablation, repairs, selection report."""
import re
import numpy as np
import pandas as pd
from scipy import stats as sps

ZH = re.compile(r"[一-鿿]")           # CJK Unified Ideographs (state this range in the paper)
LETTERS = re.compile(r"[A-Za-z一-鿿]")

def has_zh(text): return bool(ZH.search(text or ""))

def zh_share(text):
    """Share of letters (Latin + CJK) that are Chinese; 'predominantly Chinese' is share > 0.5."""
    letters = LETTERS.findall(text or "")
    return (len(ZH.findall(text or "")) / len(letters)) if letters else 0.0

def words(text): return len((text or "").split())

def language_report(pairs):
    c = pairs.chosen.map(has_zh); r = pairs.rejected.map(has_zh)
    both = pd.concat([pairs.chosen[c], pairs.rejected[r]])
    return {"chosen_with_zh": int(c.sum()), "rejected_with_zh": int(r.sum()),
            "pairs_zh_either_side": int((c | r).sum()), "pairs_zh_chosen_only_filter_removes": int(c.sum()),
            "contaminated_answers_total": int(len(both)),
            "predominantly_chinese_answers": int((both.map(zh_share) > 0.5).sum())}

def duplicate_report(pairs, q="question", code="question_code", run="run"):
    n = len(pairs); distinct = pairs[q].str.strip().str.lower().nunique()
    grp = pairs.assign(_k=pairs[q].str.strip().str.lower()).groupby("_k").size()
    rep = {"pairs": n, "distinct_question_texts": int(distinct), "excess_records": int(n - distinct),
           "duplicated_question_groups": int((grp > 1).sum())}
    if code in pairs and run in pairs:
        ck = pairs.groupby(code)[run].nunique()
        rep["codes_shared_by_both_runs"] = int((ck > 1).sum())
        rep["pairs_with_colliding_code"] = int(pairs[code].isin(ck[ck > 1].index).sum())
    if "pair_id" in pairs:
        rep["duplicate_pair_ids"] = int(pairs.pair_id.duplicated().sum())
    return rep

def length_report(pairs):
    cw, rw = pairs.chosen.map(words), pairs.rejected.map(words)
    out = {"chosen_words_mean": cw.mean(), "rejected_words_mean": rw.mean(),
           "pairs_chosen_longer": int((cw > rw).sum()), "share_chosen_longer_%": 100 * float((cw > rw).mean())}
    if "margin" in pairs:
        out["corr_margin_vs_chosen_len"] = float(np.corrcoef(pairs.margin, cw)[0, 1])
    if "run" in pairs:
        out["chosen_words_by_run"] = cw.groupby(pairs.run).mean().round(1).to_dict()
        out["rejected_words_by_run"] = rw.groupby(pairs.run).mean().round(1).to_dict()
    return out

def high_scoring_rejected(pairs, thr=0.50):
    return {"rejected_tps_ge_thr": int((pairs.rej_tps >= thr).sum()), "chosen_tps_lt_thr": int((pairs.chosen_tps < thr).sum())}

def repair(pairs, language="chosen_or_rejected", dedupe=True):
    """Repairs used for the 'clean' training arm. language: 'chosen' | 'chosen_or_rejected' | None."""
    d = pairs.copy()
    if language:
        bad = d.chosen.map(has_zh) if language == "chosen" else (d.chosen.map(has_zh) | d.rejected.map(has_zh))
        d = d[~bad]
    if dedupe:
        d = d.assign(_k=d.question.str.strip().str.lower()).sort_values("margin", ascending=False).drop_duplicates("_k").drop(columns="_k")
    return d.sort_index()

def select_pairs(df, n, mode="random", seed=42):
    """Pre-specified selection rule. random: seeded simple random sample; max_margin; all."""
    if mode == "all" or n is None or n >= len(df): return df.copy()
    if mode == "random": return df.sample(n=n, random_state=seed)
    if mode == "max_margin": return df.nlargest(n, "margin")
    raise ValueError(mode)

def selection_report(full, sub):
    """Compare the training subset with the full set (reviewer 2: characteristics of the 234)."""
    rows = []
    feats = {"chosen_words": lambda d: d.chosen.map(words), "rejected_words": lambda d: d.rejected.map(words)}
    if "margin" in full: feats["margin"] = lambda d: d.margin
    for k, f in feats.items():
        a, b = f(full), f(sub)
        rows.append({"feature": k, "full_mean": a.mean(), "subset_mean": b.mean(), "mann_whitney_p": sps.mannwhitneyu(a, b).pvalue})
    if "run" in full:
        for r in full.run.unique():
            rows.append({"feature": f"share_run_{r}", "full_mean": (full.run == r).mean(), "subset_mean": (sub.run == r).mean(), "mann_whitney_p": np.nan})
    return pd.DataFrame(rows)
