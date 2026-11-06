#!/usr/bin/env python3
"""Analyses requested in the ICIMCIS 2026 review round, run from files you already have.

Subcommands (each prints the numbers that fill the [[ ]] placeholders in the revised manuscript):

  ablation   Renormalized scoring ablation + ranking reversals      -> Table VI, Section V-A
  decontam   Semantic overlap of eval questions with training ones  -> Section V-D, clean set
  paired     Counts, discordant pairs, query-clustered permutation
             test, cluster bootstrap, Mantel-Haenszel by length     -> Section V-D, Section VI
  sample     Blinded, stratified human-audit sample + answer key    -> Section V-C
  agree      Confirmation rate (Wilson CI) and Cohen's kappa        -> Section V-C
  selftest   Run every step on synthetic data (checks the code)

Input columns (CSV, UTF-8):
  ablation : pair_id, run, margin, chosen_sg, chosen_sr, chosen_sl, rej_sg, rej_sr, rej_sl
  decontam : --train csv with column question ; --eval csv with columns query_id, question
  paired   : query_id, run, model (baseline|dpo), flagged (0/1), words (int), has_zh (0/1; optional)
  sample   : pair_id, run, margin, question, chosen, rejected
  agree    : pair_id, rater1, rater2, [adjudicated]   (values A or B = which answer preferred),
             plus --key from `sample` to map A/B back to chosen/rejected
"""
import argparse, math, sys
import numpy as np
import pandas as pd

RNG = np.random.default_rng(2026)
W = dict(sg=0.4, sr=0.4, sl=0.2)


# ---------------------------------------------------------------- ablation
def ablation(df, thr=0.15):
    """Scoring ablation on stored component scores. For each removed dimension three variants are reported:
       proportional : remaining weights rescaled to sum to 1 (0.667/0.333 ...)
       equal_split  : remaining weight shared equally (0.5/0.5), as in the original Stage 2 notebook (NO_SG/NO_SR/NO_LEX)
       unadjusted   : remaining weights left as they are (score range shrinks; threshold not comparable)
    Reported: pairs still >= thr, ranking reversals (reduced margin < 0), ties (== 0), mean margin."""
    def score(prefix, w): return sum(w[k] * df[f"{prefix}_{k}"] for k in w)
    rows = []
    def add(name, variant, w):
        d = score("chosen", w) - score("rej", w)
        rows.append((name, variant, {k: round(v, 3) for k, v in w.items()}, int((d >= thr).sum()), int((d < 0).sum()), int((d == 0).sum()),
                     float(100 * (d < 0).mean()), float(d.mean())))
    add("none (full TPS)", "full", W)
    for drop in ("sg", "sr", "sl"):
        keep = {k: v for k, v in W.items() if k != drop}; tot = sum(keep.values())
        add(f"w/o {drop}", "proportional", {k: v / tot for k, v in keep.items()})
        add(f"w/o {drop}", "equal_split", {k: 1 / len(keep) for k in keep})
        add(f"w/o {drop}", "unadjusted", keep)
    return pd.DataFrame(rows, columns=["removed", "variant", "weights", "pairs>=thr", "reversed", "tied", "reversed_%", "mean_margin"])


# ---------------------------------------------------------------- decontam
def decontam(train_q, eval_df, tau=0.80, model="paraphrase-multilingual-MiniLM-L12-v2", emb=None):
    if emb is None:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(model)
        emb = lambda xs: m.encode(list(xs), normalize_embeddings=True)
    A = emb(eval_df["question"]); B = emb(train_q)
    S = A @ B.T
    best = S.max(1); arg = S.argmax(1)
    res = eval_df.copy()
    res["max_cos"] = best; res["nearest_train"] = [train_q[i] for i in arg]
    res["flag_overlap"] = res["max_cos"] >= tau
    return res.sort_values("max_cos", ascending=False)


# ---------------------------------------------------------------- paired stats
def mh_or(df, strata_col, treat="dpo"):
    """Mantel-Haenszel OR of a flag, treat vs baseline, across strata."""
    num = den = 0.0
    for _, g in df.groupby(strata_col):
        a = ((g.model == treat) & (g.flagged == 1)).sum(); b = ((g.model == treat) & (g.flagged == 0)).sum()
        c = ((g.model == "baseline") & (g.flagged == 1)).sum(); d = ((g.model == "baseline") & (g.flagged == 0)).sum()
        n = a + b + c + d
        if n:
            num += a * d / n; den += b * c / n
    return num / den if den else float("nan")


def paired(df, n_perm=100000, n_boot=3000, treat="dpo"):
    out = {}
    df = df[df.model.isin(["baseline", treat])]
    base = df[df.model == "baseline"].set_index(["query_id", "run"]).sort_index()
    dpo = df[df.model == treat].set_index(["query_id", "run"]).sort_index()
    assert base.index.equals(dpo.index), "baseline and dpo must cover identical (query_id, run) pairs"
    b = base.flagged.values; d = dpo.flagged.values
    out["n_pairs"] = len(b)
    out["flagged_baseline"] = int(b.sum()); out["flagged_dpo"] = int(d.sum())
    out["baseline_only"] = int(((b == 1) & (d == 0)).sum()); out["dpo_only"] = int(((b == 0) & (d == 1)).sum())
    out["both"] = int(((b == 1) & (d == 1)).sum()); out["neither"] = int(((b == 0) & (d == 0)).sum())
    # query-level sums of (baseline - dpo); sign-flip permutation keeps the runs of a query together
    qs = pd.Series(b - d, index=base.index).groupby(level=0).sum().values.astype(float)
    obs = qs.sum()
    signs = RNG.choice([-1.0, 1.0], size=(n_perm, len(qs)))
    perm = np.abs((signs * qs).sum(1))
    out["perm_stat(b-d flags)"] = float(obs)
    out["p_two_sided_cluster_perm"] = float((np.sum(perm >= abs(obs)) + 1) / (n_perm + 1))
    # cluster bootstrap of the flag-rate difference and of the (length-adjusted) MH odds ratio
    df = df.copy()
    df["tercile"] = pd.qcut(df.words, 3, labels=False, duplicates="drop")
    qids = df.query_id.unique(); groups = {q: g for q, g in df.groupby("query_id")}
    diffs, ors, ors_adj = [], [], []
    for _ in range(n_boot):
        samp = RNG.choice(qids, len(qids), replace=True)
        g = pd.concat([groups[q] for q in samp])
        diffs.append(g[g.model == treat].flagged.mean() - g[g.model == "baseline"].flagged.mean())
        g = g.assign(all=0)
        ors.append(mh_or(g, "all", treat)); ors_adj.append(mh_or(g, "tercile", treat))
    ci = lambda x: tuple(np.nanpercentile(x, [2.5, 97.5]).round(3))
    out["rate_diff_dpo_minus_base"] = float(df[df.model == treat].flagged.mean() - df[df.model == "baseline"].flagged.mean())
    out["rate_diff_CI95"] = ci(diffs)
    out["MH_OR_unadjusted"] = float(mh_or(df.assign(all=0), "all", treat)); out["MH_OR_unadjusted_CI95"] = ci(ors)
    out["MH_OR_length_adjusted"] = float(mh_or(df, "tercile", treat)); out["MH_OR_length_adjusted_CI95"] = ci(ors_adj)
    return out


# ---------------------------------------------------------------- audit sample
def sample(df, n=100, seed=7):
    r = np.random.default_rng(seed)
    df = df[df.margin >= 0.15].copy()
    df["band"] = pd.cut(df.margin, [0.15, 0.20, 0.25, 10], right=False, labels=["0.15-0.20", "0.20-0.25", ">=0.25"])
    df["stratum"] = df.run.astype(str) + "|" + df.band.astype(str)
    take = []
    for s, g in df.groupby("stratum", observed=True):
        k = max(1, round(n * len(g) / len(df)))
        take.append(g.sample(min(k, len(g)), random_state=int(r.integers(1e9))))
    s = pd.concat(take)
    if len(s) < n:   # top up after rounding so exactly n pairs are drawn
        rest = df[~df.pair_id.isin(s.pair_id)]
        s = pd.concat([s, rest.sample(min(n - len(s), len(rest)), random_state=seed)])
    s = s.sample(frac=1, random_state=seed).head(n).reset_index(drop=True)
    flip = r.random(len(s)) < 0.5
    s["answer_A"] = np.where(flip, s.rejected, s.chosen); s["answer_B"] = np.where(flip, s.chosen, s.rejected)
    s["chosen_is"] = np.where(flip, "B", "A")
    blind = s[["pair_id", "question", "answer_A", "answer_B"]].copy()
    for c in ["A_factual_support_1to5", "B_factual_support_1to5", "A_completeness_1to5", "B_completeness_1to5",
              "A_relevance_1to5", "B_relevance_1to5", "A_language_1to5", "B_language_1to5", "preferred_A_or_B_or_tie"]:
        blind[c] = ""
    key = s[["pair_id", "run", "margin", "chosen_is"]]
    return blind, key


# ---------------------------------------------------------------- agreement
def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)

def kappa(a, b):
    a = np.asarray(a); b = np.asarray(b); cats = sorted(set(a) | set(b))
    po = np.mean(a == b); pe = sum(np.mean(a == c) * np.mean(b == c) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")

def agree(df, key):
    m = df.merge(key, on="pair_id")
    final = m["adjudicated"] if "adjudicated" in m and m["adjudicated"].notna().any() else m["rater1"]
    if "adjudicated" in m:
        final = m["adjudicated"].fillna(m["rater1"].where(m["rater1"] == m["rater2"]))
    conf = (final == m["chosen_is"]).sum(); n = final.notna().sum()
    lo, hi = wilson(int(conf), int(n))
    return {"n_rated": int(n), "confirmed": int(conf), "confirmed_%": 100 * conf / n, "wilson95_%": (100 * lo, 100 * hi),
            "cohen_kappa_r1_r2": kappa(m.rater1, m.rater2), "raw_agreement_%": 100 * float(np.mean(m.rater1 == m.rater2))}


# ---------------------------------------------------------------- selftest
def selftest():
    n = 693
    syn = pd.DataFrame({"pair_id": range(n), "run": np.where(np.arange(n) < 86, "initial", "extended")})
    for p, mu in (("chosen", (0.59, 0.79, 0.64)), ("rej", (0.44, 0.65, 0.16))):
        for k, m_ in zip(("sg", "sr", "sl"), mu):
            syn[f"{p}_{k}"] = np.clip(RNG.normal(m_, 0.1, n), 0, 1)
    syn["margin"] = sum(W[k] * (syn[f"chosen_{k}"] - syn[f"rej_{k}"]) for k in W)
    print(ablation(syn[syn.margin >= 0.15]).to_string(index=False))
    q = ["Apa syarat sidang skripsi?", "Kapan batas cuti akademik?"]
    toy = {"Apa syarat sidang skripsi?": [1, 0], "Bagaimana syarat untuk sidang tugas akhir?": [0.95, 0.1],
           "Kapan batas cuti akademik?": [0, 1], "Di mana perpustakaan?": [0.3, 0.3]}
    emb = lambda xs: np.array([np.array(toy[x]) / np.linalg.norm(toy[x]) for x in xs])
    print(decontam(q, pd.DataFrame({"query_id": [1, 2], "question": ["Bagaimana syarat untuk sidang tugas akhir?", "Di mana perpustakaan?"]}), emb=emb).to_string(index=False))
    rows = []
    for qi in range(200):
        for run in range(3):
            w = int(RNG.integers(15, 110))
            rows.append((qi, run, "baseline", int(RNG.random() < 0.163), w)); rows.append((qi, run, "dpo", int(RNG.random() < 0.130), w - 5))
    pr = pd.DataFrame(rows, columns=["query_id", "run", "model", "flagged", "words"])
    for k, v in paired(pr, n_perm=5000, n_boot=200).items(): print(f"  {k}: {v}")
    pairs = syn.assign(question="q", chosen="c", rejected="r")
    blind, key = sample(pairs, n=100)
    print("sample", blind.shape, key.chosen_is.value_counts().to_dict())
    r1 = np.where(RNG.random(len(key)) < 0.8, key.chosen_is, np.where(key.chosen_is == "A", "B", "A"))
    r2 = np.where(RNG.random(len(key)) < 0.85, r1, np.where(r1 == "A", "B", "A"))
    print(agree(pd.DataFrame({"pair_id": key.pair_id, "rater1": r1, "rater2": r2}), key))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("ablation"); a.add_argument("csv"); a.add_argument("--thr", type=float, default=0.15)
    d = sp.add_parser("decontam"); d.add_argument("--train"); d.add_argument("--eval"); d.add_argument("--tau", type=float, default=0.80)
    p = sp.add_parser("paired"); p.add_argument("csv")
    s = sp.add_parser("sample"); s.add_argument("csv"); s.add_argument("--n", type=int, default=100)
    g = sp.add_parser("agree"); g.add_argument("csv"); g.add_argument("--key")
    sp.add_parser("selftest")
    x = ap.parse_args()
    if x.cmd == "ablation": print(ablation(pd.read_csv(x.csv), x.thr).to_string(index=False))
    elif x.cmd == "decontam":
        r = decontam(list(pd.read_csv(x.train).question), pd.read_csv(x.eval), x.tau)
        print(r.to_string(index=False)); r.to_csv("decontam_report.csv", index=False)
    elif x.cmd == "paired":
        for k, v in paired(pd.read_csv(x.csv)).items(): print(f"{k}: {v}")
    elif x.cmd == "sample":
        b, k = sample(pd.read_csv(x.csv), x.n); b.to_csv("audit_blinded.csv", index=False); k.to_csv("audit_key.csv", index=False)
        print("wrote audit_blinded.csv (give to annotators) and audit_key.csv (keep private)")
    elif x.cmd == "agree": print(agree(pd.read_csv(x.csv), pd.read_csv(x.key)))
    else: selftest()
