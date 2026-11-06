# TPS audit: reproducibility package

Code for the ICIMCIS 2026 paper *Adapting an Automated Preference Dataset Pipeline to Document-Grounded Academic Question Answering in Indonesian: A Methodological Audit*.
It re-implements the Triple-Proxy Score (TPS), audits the resulting preference pairs, validates the labels with independent LLM judges,
trains LoRA-DPO adapters on a `raw` and a `clean` version of the pairs, and evaluates them with clustered statistics.

## Layout
```
configs/default.yaml        every threshold, hyperparameter and model name (pre-specified)
prompts/                    generation prompts (replace the template text with the exact original prompts)
src/tpsaudit/               tps.py scoring · audit.py · judge.py · nli_dahl.py · train_dpo.py · generate.py · evaluate.py · analyses.py
notebooks/reproduce_colab.ipynb   the whole experiment, resumable, free Colab T4
scripts/check_tps_reproduction.py re-score stored pairs and compare with stored TPS
tests/                      CPU tests with mocked models (pytest)
docs/PROTOCOL.md            pre-specified protocol and safeguards
DATA.md                     data schema and release terms
```

## Quick start
```bash
pip install -r requirements.txt && python -m pytest -q tests          # CPU: 7 tests, no GPU needed
python scripts/check_tps_reproduction.py data/pairs.csv                # run first: must reproduce stored scores
```
Full experiment: open `notebooks/reproduce_colab.ipynb` in Colab (GPU: T4), set `REPO_URL`, add `HF_TOKEN` as a Colab secret, put `pairs.csv` and
`eval_queries.csv` in `data/`, and run the cells in order. Results land in `Drive/tps-audit-run/` with a `config_snapshot.json`.

## What each result file feeds
| file | paper |
|---|---|
| `audit.json`, `ablation.csv` | Sec. V-A/B, Table VI (renormalized ablation, ranking reversals) |
| `judge_pairs_*.csv`, `judge_pairs_summary.json` | Sec. V-C label validation |
| `decontam_report.csv`, `clean_eval.json` | Sec. V-D clean evaluation set |
| `train_ids_*.csv`, `selection_report.csv`, `train_summaries.json` | Sec. V-D selection and training setup |
| `responses.csv`, `flags_all.csv`, `results.json` | Sec. V-D, VI statistics |

## Honest limitations of this package
- `tps.py` is a re-implementation from the paper's equations. Verify it with `check_tps_reproduction.py`; adapt how the context is embedded if scores differ.
- `prompts/generation_prompts.yaml` is a template until you paste the original prompts.
- The library pins in `requirements-colab.txt` and the GPU stages (judging, DPO, generation, NLI) were written against the TRL 0.12.2 / PEFT 0.13.2 APIs and checked against their source, but **not executed on a GPU** by the author of this package. The CPU tests cover the logic with mocked models.
- Judges are small open LLMs; see `docs/PROTOCOL.md` for what the validation does and does not show.
- Retrieval (ColPali, Qdrant) is not included; export its text context into `eval_queries.csv`.

## Licence
Code: MIT. Data and document text: see `DATA.md`.
