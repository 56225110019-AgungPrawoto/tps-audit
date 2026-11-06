#!/usr/bin/env python3
"""Generate notebooks/revision_colab.ipynb  (usage: python scripts/make_notebook.py notebooks/revision_colab.ipynb)."""
import json, sys
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)})

md("""
# ICIMCIS 2026 revision: one-click Colab (T4 GPU runtime)
**Use:** fill the form fields in cell 0, then *Runtime -> Run all*. Everything is resumable (results are written to Google Drive and skipped on re-run).

| Phase | Needs | What it gives |
|---|---|---|
| A. Stored-output analyses (no new model runs) | your existing Stage 2 CSV, Stage 4 `evaluation_per_sample_epoch{1,2,3}.csv`, eval question CSV | dataset audit, ablation + rank flips, **234 calibration**, decontamination, clustered statistics, language counts, length-adjusted OR |
| B. LLM-judge label validation (**GPU**, set `RUN_JUDGES`) | HF token with Gemma-2 and Llama-3.1 licences accepted | agreement of two independent judges with the 693 labels |
| C. Clean-pool retraining + regeneration (**GPU, hours**, `RUN_RETRAIN`, default OFF) | `precomputed_contexts.json` | tests whether removing contaminated pairs changes the drift |

Not tested on a GPU when this package was written: phases B and C. Phase A is covered by unit tests with mocked models. If a cell fails, send me the traceback.
""")

code("""
#@title 0. Parameters and setup
BASE_DRIVE   = "/content/drive/MyDrive/Thesis-Agung_Prawoto-56225110019"  #@param {type:"string"}
PAIRS_CSV    = "DPO/dpo_dataset_output.csv"          #@param {type:"string"}
EVAL_OUT_DIR = "OUTPUT_EVAL"                         #@param {type:"string"}
EVAL_CSV     = ""                                    #@param {type:"string"}
REPO_URL     = "https://github.com/<account>/tps-audit"  #@param {type:"string"}
RUN_JUDGES   = True    #@param {type:"boolean"}
RUN_RETRAIN  = False   #@param {type:"boolean"}
import os, subprocess, sys
from google.colab import drive; drive.mount('/content/drive')
if not os.path.exists('/content/tps-audit'):
    subprocess.run(['git', 'clone', REPO_URL, '/content/tps-audit'], check=True)
os.chdir('/content/tps-audit'); sys.path.insert(0, 'src')
!pip -q install -r requirements-colab.txt
OUT = f"{BASE_DRIVE}/REVISION_RUN"; os.makedirs(OUT, exist_ok=True)
try:
    from google.colab import userdata; os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')   # key icon in the sidebar
except Exception as e: print('No HF_TOKEN secret (needed for phases B/C):', e)
""")

code("""
#@title 1. Load stored files and check them
import json, yaml, numpy as np, pandas as pd
from tpsaudit import config, audit, analyses, data_io, evaluate, selection
cfg = config.load(); prompts = yaml.safe_load(open('prompts/generation_prompts.yaml', encoding='utf8'))
config.snapshot(cfg, OUT)
pairs = data_io.load_stage2_pairs(f"{BASE_DRIVE}/{PAIRS_CSV}", cfg['data']['initial_run_n_pairs'])
print(len(pairs), 'pairs;', data_io.run_boundary_report(pairs))
evald = data_io.load_eval_csv(EVAL_CSV) if EVAL_CSV else None
print('eval questions:', None if evald is None else len(evald))
runs = data_io.load_existing_runs(f"{BASE_DRIVE}/{EVAL_OUT_DIR}")
print('stored evaluation rows:', len(runs), runs.groupby(['model', 'run']).size().to_dict() if len(runs) else '')
""")

md("## Phase A: analyses on stored outputs (no GPU needed)")
code("""
#@title A1. Dataset audit and renormalized ablation (Sec. V-A/V-B, Table VI)
A = {'language': audit.language_report(pairs), 'duplicates': audit.duplicate_report(pairs), 'length': audit.length_report(pairs),
     'high_scoring_rejected': audit.high_scoring_rejected(pairs, 0.50),
     'repairs': {'chosen_only': len(audit.repair(pairs, 'chosen', False)), 'either_side': len(audit.repair(pairs, 'chosen_or_rejected', False)),
                 'dedupe_only': len(audit.repair(pairs, None, True)), 'both': len(audit.repair(pairs, 'chosen_or_rejected', True))}}
abl = analyses.ablation(pairs, 0.15); abl.to_csv(f'{OUT}/ablation.csv', index=False)
print(abl.to_string(index=False)); evaluate.save(A, f'{OUT}/audit.json'); print(json.dumps(evaluate.jsonable(A), indent=1))
""")
code("""
#@title A2. Which pairs were the 234? (token-length calibration, CPU)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(cfg['training']['base_model'])
render, count = selection.qwen_counters(tok)
L = selection.token_lengths(pairs, prompts, render, count); L.to_csv(f'{OUT}/token_lengths.csv', index=False)
cal = selection.calibrate(L, cfg['training']['dpo']['max_length'], target=234); cal.to_csv(f'{OUT}/selection_calibration.csv', index=False)
print(cal.to_string(index=False)); print('steps for 234 examples:', selection.expected_steps(234), '| for 693:', selection.expected_steps(693))
print('If no rule matches, report only: 693 loaded, 59 steps, 234 examples. Do NOT claim a rule.')
""")
code("""
#@title A3. Decontamination of the evaluation questions against all 693 pool questions
if evald is None: print('Set EVAL_CSV to run this cell')
else:
    rep = analyses.decontam(list(pairs.question), evald.rename(columns={'prompt': 'question'}), tau=cfg['decontamination']['tau'], model=cfg['decontamination']['embed_model'])
    rep.to_csv(f'{OUT}/decontam_report.csv', index=False)
    print(rep.head(25)[['query_id', 'max_cos', 'flag_overlap', 'question', 'nearest_train']].to_string(index=False)); print('flagged:', int(rep.flag_overlap.sum()))
""")
md("**Manual step:** read the flagged rows and the ones just below the threshold; list any extra paraphrases in `EXTRA_DROP` and re-run the next cell.")
code("""
EXTRA_DROP = []   # query_ids to drop after manual review
if evald is not None:
    drop = set(rep[rep.flag_overlap].query_id) | set(EXTRA_DROP)
    clean_ids = sorted(set(evald.query_id) - drop)
    json.dump({'tau': cfg['decontamination']['tau'], 'dropped': sorted(map(int, drop)), 'n_clean': len(clean_ids)}, open(f'{OUT}/clean_eval.json', 'w'))
    print(len(drop), 'dropped ->', len(clean_ids), 'clean queries')
""")
code("""
#@title A4. Clustered statistics on the stored baseline vs original-DPO runs (queries = clusters)
if len(runs) == 0: print('No stored runs found')
else:
    df = evaluate.finish(evaluate.standardize_existing(runs))
    print('flag-rule consistency:', evaluate.recheck_flags(df))
    if evald is not None: df = evaluate.add_rouge(df, evald)
    R = {}
    sets = {'full': set(df.query_id)} | ({'clean': set(clean_ids)} if evald is not None else {})
    for label, ids in sets.items():
        R[label] = evaluate.summarize(df[df.query_id.isin(ids)], ['dpo_orig'], n_perm=100000, n_boot=3000); R[label]['n_queries'] = len(ids)
    evaluate.save(R, f'{OUT}/results_stored.json'); df.to_csv(f'{OUT}/stored_runs_standardized.csv', index=False)
    print(json.dumps(evaluate.jsonable(R), indent=1)[:6000])
""")

md("## Phase B: independent LLM judges for all 693 preference labels (GPU)")
code("""
#@title B. Judge all pairs, both answer orders, two model families
import gc
if not RUN_JUDGES: print('RUN_JUDGES is off')
else:
    import torch
    from tpsaudit import judge
    from tpsaudit.llm import make_backend
    judged = {}
    for spec in cfg['judges']:
        f = f"{OUT}/judge_pairs_{spec['name']}.csv"
        if os.path.exists(f): judged[spec['name']] = pd.read_csv(f); continue
        be = make_backend(spec)
        res = judge.judge_pairs(be, pairs[['pair_id', 'question', 'context', 'chosen', 'rejected']]); res.to_csv(f, index=False); judged[spec['name']] = res
        be.close(); del be; gc.collect(); torch.cuda.empty_cache()
    J = {n: judge.summarize_pairs(r, pairs) for n, r in judged.items()}
    names = list(judged)
    if len(names) >= 2: J['judge_vs_judge'] = judge.judge_agreement(judged[names[0]], judged[names[1]])
    evaluate.save(J, f'{OUT}/judge_pairs_summary.json'); print(json.dumps(evaluate.jsonable(J), indent=1))
""")

md("""
## Phase C (optional, hours): retrain on the cleaned version of the SAME 234 pairs, then regenerate
Needs `precomputed_contexts.json` in the eval output folder and a calibrated rule from A2 (set `RULE`). It answers whether the Chinese / repeated-question pairs cause the drift. Skip it if time is short: the manuscript then says the cause is undetermined.
""")
code("""
#@title C. Clean-subset arm (train + 3 generation runs)
RULE = ""   #@param {type:"string"}   # a rule name from A2 whose survivors == 234
if not RUN_RETRAIN: print('RUN_RETRAIN is off')
elif RULE not in set(cal[cal.matches_log].rule): print('Set RULE to a calibrated rule that matches 234 (see A2).')
else:
    from tpsaudit.train_dpo import train_arm
    from tpsaudit.generate import generate_all
    sub = selection.survivors(pairs, L, RULE, cfg['training']['dpo']['max_length'])
    clean, removed = audit.minimal_clean(sub); removed.to_csv(f'{OUT}/removed_from_234.csv', index=False)
    print(len(sub), '->', len(clean), 'after removing', len(removed))
    ad = f'{OUT}/train_clean/adapter'
    if not os.path.exists(ad): train_arm(cfg, prompts, clean, f'{OUT}/train_clean', seed=cfg['training']['seed'])
    ctx = data_io.load_contexts(f"{BASE_DRIVE}/{EVAL_OUT_DIR}/precomputed_contexts.json")
    from tpsaudit import tps
    embed = tps.default_embedder(cfg['scoring']['embed_model'], device='cuda')
    new = generate_all(cfg, prompts, evald, ctx, {'clean': ad}, embed, OUT, include_baseline=False); new.to_csv(f'{OUT}/responses_clean.csv', index=False)
    allr = pd.concat([df[df.model != 'clean'], evaluate.finish(evaluate.standardize_existing(new))], ignore_index=True)
    R2 = evaluate.summarize(allr, ['dpo_orig', 'clean']); evaluate.save(R2, f'{OUT}/results_clean_arm.json'); print(json.dumps(evaluate.jsonable(R2), indent=1)[:5000])
""")
code("""
#@title Package results for download
!cd "{OUT}" && zip -qr /content/tps-audit-results.zip . && ls -lh /content/tps-audit-results.zip
from google.colab import files; files.download('/content/tps-audit-results.zip')
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
