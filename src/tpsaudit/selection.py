"""Which of the 693 pairs did the original DPO run actually train on?

FACT FROM THE ORIGINAL LOG (Stage 3, saved output): `CSV Stage 2 dibaca: 693 baris -> 693 valid`, `Dataset num_rows: 693`,
then a tokenization step "Dropping fully truncated examples from train dataset", and `TrainOutput(global_step=59)`.
With per_device_batch=1 and grad_accum=4 (effective batch 4), 59 optimizer steps correspond to 234 examples (ceil(234/4) = 59), not 693
(693 examples would give 174 steps). The paper's "234 examples" therefore is NOT a deliberate sample: it is the number of pairs
that survived the trainer's length filter (max_length = 1024 tokens, prompt + completion; the prompt carries up to three 1,200-character pages).

HYPOTHESIS (to verify with `calibrate`, it is NOT assumed): the survivors are the pairs whose chat-formatted prompt is shorter than
max_length (a prompt of >= 1024 tokens leaves nothing of the completion, i.e. a "fully truncated" example). The exact rule depends on the
TRL version that ran. `calibrate` evaluates several candidate rules on the real CSV and reports which one yields exactly the observed count.
If none does, `survivors` must not be used; say in the paper only what the log shows (693 loaded, 59 steps, 234 examples).

All functions take injected callables so they run without downloading a tokenizer (see tests); `qwen_counters(tokenizer)` builds the real ones.
"""
import math
import pandas as pd

def to_messages(row, prompts):
    """Stage 3 to_trl_conversational(): the same grounded prompt for chosen and rejected."""
    user = prompts["stage3_user"].format(context=row["context"], query=row["question"])
    return [{"role": "system", "content": prompts["stage3_system"]}, {"role": "user", "content": user}]

def qwen_counters(tok):
    """(render, count) for a Hugging Face tokenizer. Uses tokenize=False + re-tokenization so it works across transformers versions."""
    render = lambda msgs, gen: tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen)
    count = lambda text: len(tok(text, add_special_tokens=False)["input_ids"])
    return render, count

def token_lengths(df, prompts, render, count):
    """One row per pair: tokens of the prompt (with generation prompt), and of the completion = full chat text minus prompt text."""
    rows = []
    for r in df.to_dict("records"):
        m = to_messages(r, prompts); p = render(m, True)
        n_p = count(p)
        n_c = count(render(m + [{"role": "assistant", "content": r["chosen"]}], False)) - count(render(m, False))
        n_r = count(render(m + [{"role": "assistant", "content": r["rejected"]}], False)) - count(render(m, False))
        rows.append({"pair_id": r.get("pair_id"), "n_prompt": n_p, "n_chosen": max(n_c, 0), "n_rejected": max(n_r, 0)})
    return pd.DataFrame(rows)

def candidate_rules(L):
    return {
        "prompt < L": lambda d: d.n_prompt < L,
        "prompt <= L": lambda d: d.n_prompt <= L,
        "prompt + chosen <= L": lambda d: d.n_prompt + d.n_chosen <= L,
        "prompt + rejected <= L": lambda d: d.n_prompt + d.n_rejected <= L,
        "prompt + chosen <= L and prompt + rejected <= L": lambda d: (d.n_prompt + d.n_chosen <= L) & (d.n_prompt + d.n_rejected <= L),
        "prompt + longer completion < L": lambda d: d.n_prompt + d[["n_chosen", "n_rejected"]].max(axis=1) < L,
    }

def calibrate(lengths, max_length=1024, target=234):
    """Count survivors under each candidate rule and mark the ones that equal the observed number of examples."""
    rows = []
    for name, f in candidate_rules(max_length).items():
        n = int(f(lengths).sum())
        rows.append({"rule": name, "survivors": n, "difference_to_target": n - target, "matches_log": n == target})
    return pd.DataFrame(rows)

def survivors(df, lengths, rule, max_length=1024):
    keep = candidate_rules(max_length)[rule](lengths).values
    return df[keep].copy()

def expected_steps(n, per_device_batch=1, grad_accum=4, epochs=1):
    return math.ceil(n / (per_device_batch * grad_accum)) * epochs

def steps_to_examples(steps, per_device_batch=1, grad_accum=4, epochs=1):
    """Range of example counts compatible with an observed number of optimizer steps."""
    b = per_device_batch * grad_accum
    per_epoch = steps / epochs
    return (int((per_epoch - 1) * b + 1), int(per_epoch * b))

def truncation_profile(lengths, keep_mask, max_length=1024):
    """Among the surviving pairs: how many still lose part of the chosen / rejected completion to the max_length cut."""
    d = lengths[keep_mask]
    return {"n": int(len(d)), "chosen_cut": int(((d.n_prompt + d.n_chosen) > max_length).sum()),
            "rejected_cut": int(((d.n_prompt + d.n_rejected) > max_length).sum()),
            "prompt_tokens_mean": float(d.n_prompt.mean()), "prompt_tokens_mean_all": float(lengths.n_prompt.mean())}
