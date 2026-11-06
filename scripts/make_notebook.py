import json, sys
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)})

md("""
# Reproduce and extend the ICIMCIS 2026 audit (Google Colab, free T4)

Runs every analysis requested in the review round from one place, with **all settings pre-specified in `configs/default.yaml`** and a config snapshot saved next to each result:

| Step | What it produces | Manuscript location |
|---|---|---|
| 1 | TPS reproduction check on the stored pairs | sanity check |
| 2 | Dataset audit: language, duplicates, length, renormalized ablation + ranking reversals | Sec. V-A, V-B, Table VI |
| 3 | **Independent LLM-judge validation of all 693 labels** (2 judges, both answer orders) | Sec. V-C |
| 4 | Decontamination of the evaluation set (semantic overlap) | Sec. V-D |
| 5 | Training arms: `raw` vs `clean` (+ seeds), selection report | Sec. V-D, VI |
| 6 | Generation: baseline vs each arm, 3 sampling runs | Sec. V-D |
| 7 | Flags: TPS, NLI, DAHL-style, LLM-judge | Sec. V-D |
| 8 | Clustered statistics, length/language adjustment, drift counts | Sec. VI |

**Resumable:** each stage writes to Google Drive and is skipped when its output already exists. Use *Runtime → Disconnect and delete runtime* between heavy stages if memory is tight.
**Inputs you must provide** in `data/`: `pairs.csv` and `eval_queries.csv` (schemas are checked in step 0).
Rough T4 time: judging 2×~1 h, training ~15 min per arm-seed, generation ~30–45 min, flags ~1–2 h. These are estimates; the code was not run on a GPU when this package was written.
""")

code("""
#@title 0. Setup (run once, then restart the runtime if Colab asks)
from google.colab import drive; drive.mount('/content/drive')
REPO_URL = "https://github.com/<your-account>/tps-audit"     #@param {type:"string"}
import os, subprocess
if not os.path.exists('/content/tps-audit'):
    subprocess.run(['git','clone',REPO_URL,'/content/tps-audit'],check=True)
%cd /content/tps-audit
!pip -q install -r requirements-colab.txt
""")
code("""
# Hugging Face token for gated judge models (accept the Gemma / Llama licences on the model pages first)
from google.colab import userdata
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')      # add it under the key icon in the left sidebar
""")
code("""
#@title 1. Config, paths, data checks
import sys, json, yaml, numpy as np, pandas as pd
sys.path.insert(0,'src')
from tpsaudit import config, audit, analyses, judge, nli_dahl, evaluate, tps
from tpsaudit.llm import make_backend
cfg = config.load('configs/default.yaml')
prompts = yaml.safe_load(open('prompts/generation_prompts.yaml', encoding='utf8'))
OUT = '/content/drive/MyDrive/tps-audit-run'; os.makedirs(OUT, exist_ok=True)
config.snapshot(cfg, OUT)
pairs = pd.read_csv(cfg['paths']['pairs']); evald = pd.read_csv(cfg['paths']['eval'])
need_p = {'pair_id','run','question','context','chosen','rejected','chosen_sg','chosen_sr','chosen_sl','rej_sg','rej_sr','rej_sl','chosen_tps','rej_tps','margin'}
need_e = {'query_id','question','context'}
assert need_p <= set(pairs.columns), f"pairs.csv is missing {need_p - set(pairs.columns)}"
assert need_e <= set(evald.columns), f"eval_queries.csv is missing {need_e - set(evald.columns)}"
print(len(pairs), 'pairs;', len(evald), 'evaluation queries')
""")
code("""
#@title 2a. TPS reproduction check (must be ~0 before you trust new scores)
sc = tps.Scorer(model_name=cfg['scoring']['embed_model'], weights=cfg['scoring']['weights'])
c = sc.score(pairs.chosen, pairs.question, pairs.context); r = sc.score(pairs.rejected, pairs.question, pairs.context)
print('max |diff| chosen', (c.tps - pairs.chosen_tps).abs().max(), '| rejected', (r.tps - pairs.rej_tps).abs().max())
print('pairs still >= 0.15:', int(((c.tps - r.tps) >= 0.15).sum()), 'of', len(pairs))
""")
code("""
#@title 2b. Dataset audit (Sec. V-A, V-B, Table VI)
A = {}
A['language'] = audit.language_report(pairs)
A['duplicates'] = audit.duplicate_report(pairs)
A['length'] = audit.length_report(pairs)
A['high_scoring_rejected'] = audit.high_scoring_rejected(pairs, cfg['scoring']['flag_threshold'])
A['repairs'] = {'language_chosen_only': len(audit.repair(pairs,'chosen',False)), 'language_either': len(audit.repair(pairs,'chosen_or_rejected',False)),
                'dedupe_only': len(audit.repair(pairs,None,True)), 'both': len(audit.repair(pairs,'chosen_or_rejected',True))}
abl = analyses.ablation(pairs, cfg['scoring']['margin_threshold'])       # renormalized and unadjusted variants
abl.to_csv(f'{OUT}/ablation.csv', index=False); print(abl.to_string(index=False))
evaluate.save(A, f'{OUT}/audit.json'); print(json.dumps(evaluate.jsonable(A), indent=1))
""")
md("""
## 3. Independent LLM-judge validation of the preference labels
Two judges from model families different from the generator (Qwen2). Each pair is judged in **both answer orders** with the retrieved context as the only evidence; the verdict counts only if both orders agree. The summary reports agreement with the automatic label, position bias, agreement by length tercile (guards against a length preference of the judge), and judge–judge agreement.
""")
code("""
#@title 3. Judge all pairs (resumable, one CSV per judge)
import gc, torch
judged = {}
for spec in cfg['judges']:
    f = f"{OUT}/judge_pairs_{spec['name']}.csv"
    if os.path.exists(f): judged[spec['name']] = pd.read_csv(f); continue
    be = make_backend(spec)
    res = judge.judge_pairs(be, pairs[['pair_id','question','context','chosen','rejected']]); res['judge'] = spec['name']
    res.to_csv(f, index=False); judged[spec['name']] = res
    be.close(); del be; gc.collect(); torch.cuda.empty_cache()
J = {n: judge.summarize_pairs(r, pairs) for n, r in judged.items()}
names = list(judged)
if len(names) >= 2: J['judge_vs_judge'] = judge.judge_agreement(judged[names[0]], judged[names[1]])
evaluate.save(J, f'{OUT}/judge_pairs_summary.json'); print(json.dumps(evaluate.jsonable(J), indent=1))
""")
code("""
#@title 4. Decontaminate the evaluation set against ALL 693 pool questions
rep = analyses.decontam(list(pairs.question), evald, tau=cfg['decontamination']['tau'], model=cfg['decontamination']['embed_model'])
rep.to_csv(f'{OUT}/decontam_report.csv', index=False)
print(rep.head(25)[['query_id','max_cos','flag_overlap','question','nearest_train']].to_string(index=False))
print('flagged at tau:', int(rep.flag_overlap.sum()))
""")
md("**Manual step:** read the flagged rows above (and the rows just below the threshold). Put any extra paraphrases you find into `EXTRA_DROP`, then run the next cell.")
code("""
EXTRA_DROP = []                                   # e.g. [17, 98]  query_ids to remove after manual review
drop = set(rep[rep.flag_overlap].query_id) | set(EXTRA_DROP)
clean_ids = sorted(set(evald.query_id) - drop)
json.dump({'tau': cfg['decontamination']['tau'], 'dropped': sorted(map(int, drop)), 'n_clean': len(clean_ids), 'unit': cfg['decontamination']['unit']},
          open(f'{OUT}/clean_eval.json','w'))
print(len(drop), 'dropped ->', len(clean_ids), 'clean queries')
""")
code("""
#@title 5. Build training arms and the selection report
S = cfg['training']['selection']; rep_cfg = cfg['training']['clean_repairs']
pools = {'raw': pairs, 'clean': audit.repair(pairs, rep_cfg['language'], rep_cfg['dedupe'])}
arms = {}                                          # name -> training dataframe
sel_reports = {}
for arm in cfg['training']['arms']:
    for seed in S['seeds']:
        sub = audit.select_pairs(pools[arm], S['n_train'], S['mode'], seed)
        arms[f'{arm}_s{seed}'] = sub
        sel_reports[f'{arm}_s{seed}'] = audit.selection_report(pools[arm], sub)
        sub[['pair_id']].to_csv(f'{OUT}/train_ids_{arm}_s{seed}.csv', index=False)     # publish these ids
        print(arm, seed, len(pools[arm]), '->', len(sub), 'pairs;', 'chosen/rejected Chinese in subset:',
              int(sub.chosen.map(audit.has_zh).sum()), int(sub.rejected.map(audit.has_zh).sum()))
pd.concat({k: v for k, v in sel_reports.items()}).to_csv(f'{OUT}/selection_report.csv')
""")
code("""
#@title 6. Train (LoRA-DPO) — one adapter per arm and seed
summ = {}
from tpsaudit.train_dpo import train_arm
for name, sub in arms.items():
    d = f'{OUT}/train_{name}'
    if os.path.exists(f'{d}/adapter'): summ[name] = json.load(open(f'{d}/train_summary.json')); continue
    summ[name] = train_arm(cfg, prompts, sub, d, seed=int(name.split('_s')[-1]))
    print(name, {k: v for k, v in summ[name].items() if k != 'log_history'})
evaluate.save({k: {a: b for a, b in v.items() if a != 'log_history'} for k, v in summ.items()}, f'{OUT}/train_summaries.json')
""")
code("""
#@title 7. Generate responses (baseline vs every adapter, 3 sampling runs, full evaluation set)
from tpsaudit.generate import generate_all
f = f'{OUT}/responses.csv'
if os.path.exists(f): resp = pd.read_csv(f)
else: resp = generate_all(cfg, prompts, evald, {n: f'{OUT}/train_{n}/adapter' for n in arms}, f)
print(resp.groupby('model').size())
""")
code("""
#@title 8a. TPS flags (primary) and NLI flags
df = evaluate.add_tps(resp, evald, sc, cfg['scoring']['flag_threshold'])
df = evaluate.add_rouge(df, evald)              # only if eval_queries.csv has a 'reference' column
df = df.merge(evald[['query_id']], on='query_id')
nc = cfg['checks']['nli']
nli = nli_dahl.NLI(nc['model'], nc['window_tokens'], nc['stride'])
nf = nli_dahl.nli_flags(nli, df[['response_id','context','response']], nc['entail_threshold'], nc['supported_share_min'])
df = df.merge(nf, on='response_id'); df.to_csv(f'{OUT}/flags_tps_nli.csv', index=False)
""")
code("""
#@title 8b. DAHL-style flags and LLM-judge response flags (use judge A as extractor/grader; resumable)
f = f'{OUT}/flags_all.csv'
if os.path.exists(f): df = pd.read_csv(f)
else:
    be = make_backend(cfg['judges'][0])
    dc = cfg['checks']['dahl_style']
    dfl = nli_dahl.dahl_style_flags(nli, be, df[['response_id','context','response']], dc['entail_threshold'], dc['supported_share_min'])
    jr = judge.judge_responses(be, df.assign(response_id=df.response_id)[['response_id','question','context','response']], cfg['checks']['judge_response_flag_max_factual_support'])
    df = df.merge(dfl, on='response_id').merge(jr, on='response_id'); df.to_csv(f, index=False)
    be.close(); del be; gc.collect(); torch.cuda.empty_cache()
cols = [c for c in ['flagged','nli_flag','dahl_flag','judge_flag','has_zh','rouge1','words'] if c in df]
print(df.groupby('model')[cols].mean().round(3))
""")
code("""
#@title 9. Statistics: full evaluation set and clean evaluation set (queries as clusters)
R = {}
for label, ids in (('full', set(evald.query_id)), ('clean', set(clean_ids))):
    sub = df[df.query_id.isin(ids)]
    R[label] = evaluate.summarize(sub, list(arms), n_perm=100000, n_boot=3000)
    R[label]['n_queries'] = len(ids)
evaluate.save(R, f'{OUT}/results.json')
for label in R:
    for arm in arms:
        r = R[label][arm]; p = r['paired_tps_flag']
        print(label, arm, 'TPS flag % (base/arm):', {k: round(v, 1) for k, v in r['rates']['flagged'].items()},
              '| flagged', p['flagged_baseline'], 'vs', p['flagged_dpo'], 'of', p['n_pairs'],
              '| b/c', p['baseline_only'], p['dpo_only'], '| perm p', round(p['p_two_sided_cluster_perm'], 4),
              '| OR adj', round(p['MH_OR_length_adjusted'], 2), p['MH_OR_length_adjusted_CI95'],
              '| zh', r['language']['baseline_zh'], '->', r['language']['arm_zh'])
""")
md("""
## 10. Numbers for the manuscript placeholders
Everything is in `results.json`, `audit.json`, `ablation.csv`, `judge_pairs_summary.json`, `selection_report.csv`, `train_summaries.json`, `decontam_report.csv`, `clean_eval.json`. 
**These runs replace the earlier v4 downstream numbers.** Report the new run throughout (abstract, Tables VII–VIII, Section VI); do not mix them with v4 values.
Drift attribution: compare the `raw_s*` and `clean_s*` rows for Chinese-character counts (`language.arm_zh`). If `clean` also drifts, the contaminated pairs are not the (only) cause.
""")
code("""
#@title Package results for download
!cd {OUT} && zip -qr /content/tps-audit-results.zip . && ls -lh /content/tps-audit-results.zip
from google.colab import files; files.download('/content/tps-audit-results.zip')
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
