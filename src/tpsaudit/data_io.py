"""Loaders that read the ORIGINAL files produced by the four notebooks, so nothing has to be converted by hand."""
import json, os
import numpy as np, pandas as pd

STAGE2_MAP = {  # original column -> standard column
    "kode_pertanyaan": "question_code", "pertanyaan": "question", "sumber_file": "source_file", "context_retrieved": "context",
    "chosen": "chosen", "rejected": "rejected", "sem_grounding_chosen": "chosen_sg", "sem_relevance_chosen": "chosen_sr",
    "lexical_chosen": "chosen_sl", "sem_grounding_rejected": "rej_sg", "sem_relevance_rejected": "rej_sr", "lexical_rejected": "rej_sl",
    "score_chosen": "chosen_tps", "score_rejected": "rej_tps", "score_margin": "margin", "label_chosen": "label_chosen",
    "temp_chosen": "temp_chosen", "temp_rejected": "temp_rejected", "timestamp": "timestamp", "topik": "topic"}
NUMERIC = ["chosen_sg", "chosen_sr", "chosen_sl", "rej_sg", "rej_sr", "rej_sl", "chosen_tps", "rej_tps", "margin", "temp_chosen", "temp_rejected"]

def load_stage2_pairs(path, initial_run_n_pairs=86):
    """Read dpo_dataset_output.csv (Stage 2). Row order = processing order. `run` is INFERRED: the first
    `initial_run_n_pairs` rows are the initial run (175 questions, 86 pairs) -- check with run_boundary_report()."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False).rename(columns=STAGE2_MAP)
    for c in NUMERIC:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.insert(0, "pair_id", [f"P{i:04d}" for i in range(len(df))])
    df["run"] = np.where(np.arange(len(df)) < initial_run_n_pairs, "initial", "extended")
    return df

def run_boundary_report(df):
    """Print timestamps around the inferred run boundary so the assumption can be checked."""
    if "timestamp" not in df:
        return {}
    ts = pd.to_datetime(df.timestamp, errors="coerce")
    out = {k: (str(ts[df.run == k].min()), str(ts[df.run == k].max())) for k in df.run.unique()}
    out["rows_with_decreasing_timestamp"] = int((ts.diff().dt.total_seconds() < 0).sum())
    return out

EVAL_COLUMN_MAP = {"kode pertanyaan": "kode", "topik": "topik", "pertanyaan": "prompt", "sumber file": "sumber_file", "jawaban (ground truth)": "reference"}

def load_eval_csv(path_or_url):
    """Stage 4 Cell 5. query_id = row position (the 'kode' column may not be unique)."""
    df = pd.read_csv(path_or_url)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={c: EVAL_COLUMN_MAP[c.strip().lower()] for c in df.columns if c.strip().lower() in EVAL_COLUMN_MAP})
    for c in ("kode", "topik", "sumber_file", "reference"):
        if c not in df:
            df[c] = ""
    df["prompt"] = df["prompt"].fillna("").astype(str); df["reference"] = df["reference"].fillna("").astype(str)
    df = df[df["prompt"].str.strip() != ""].reset_index(drop=True)
    df.insert(0, "query_id", range(len(df)))
    return df

def load_contexts(path):
    """precomputed_contexts.json from Stage 4 (list of per-query lists of {filename,pdf_name,page_number,text,score})."""
    return json.load(open(path, encoding="utf8"))

def eval_context_string(contexts):
    """Concatenated page text as used by judges/NLI (no headers)."""
    return " ".join(c.get("text", "") for c in contexts if c.get("text"))

def load_existing_runs(out_dir, epochs=(1, 2, 3), label_map=None):
    """Load evaluation_per_sample_epoch{N}.csv written by Stage 4. model_label -> arm name."""
    label_map = label_map or {"Baseline (pre-DPO)": "baseline", "DPO Model (post-DPO)": "dpo_orig"}
    frames = []
    for e in epochs:
        p = os.path.join(out_dir, f"evaluation_per_sample_epoch{e}.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        d["run"] = e; d["model"] = d.model_label.map(label_map)
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
