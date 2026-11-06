# Pre-specified protocol (write this down before running; copy the numbers into the paper)

## Why a judge-based validation
Automatic labels from a similarity score need an independent check. Because the TPS and the labels share the same
signals, the check must not reuse them. We use LLM judges from model families other than the generator (Qwen2),
greedy decoding, the retrieved context as the only evidence, and both answer orders. This is a validation of label
*agreement with an independent judge*, not proof of factual correctness.

## Safeguards (state each in the paper)
1. Different families: `judgeA` Gemma-2-9B-it, `judgeB` Llama-3.1-8B-Instruct (configurable).
2. Both orders; a verdict counts only if both orders agree; the share `inconsistent` is the position-bias indicator.
3. Rubric: factual support, completeness, relevance, language quality (1–5) and an overall preference; length is not a criterion.
4. Agreement is reported by length tercile, run and margin band. LLM judges tend to prefer longer answers; chosen answers here are
   shorter, so agreement in the short tercile is the conservative number.
5. Judge–judge agreement (Cohen's κ) replaces inter-annotator agreement.
6. Limits to state: judges are small open models, Indonesian ability varies, they see only the stored context (which may be
   truncated at 1,200 characters per source), and they share failure modes with each other. A small human spot-check
   (e.g. 30 pairs by one domain expert) is the cheapest way to calibrate the judges; report it if you do it.

## Other pre-specified rules (all in `configs/default.yaml`)
- TPS flag: TPS < 0.50. NLI flag: share of NLI-supported sentences < 0.5 (entailment ≥ 0.5, context windows of 300 tokens).
  DAHL-style flag: share of NLI-supported atomic facts < 0.5. Judge flag: factual_support ≤ 2.
- Training: 234 pairs by seeded simple random sampling from each arm's pool (`raw`: all 693; `clean`: language filter on either side + deduplication),
  LoRA r=16, DPO β=0.1, lr 5e-5, effective batch size 4 (59 steps), reference = frozen base model.
- Evaluation: 200 queries × 3 sampling runs; analysis on the full set and on the clean set (decontaminated, τ = 0.80 plus manual review).
- Statistics: queries are clusters. Sign-flip permutation test and bootstrap resample queries (the three runs stay together).
- Drift attribution: compare `raw` and `clean` arms (controlled comparison); with one seed per arm treat it as exploratory.

Thresholds were fixed before looking at results. If you change one, say so and report the original as well.
