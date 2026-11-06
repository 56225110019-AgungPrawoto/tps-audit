"""Response generation = Stage 4 generate_responses_llm(), for baseline and any number of adapters on ONE loaded base model.
Same prompt layout, sampling settings, input truncation and scoring as the original. Differences (docs/DEVIATIONS.md): optional seed,
and the baseline is the adapter-disabled model instead of a separately loaded copy (mathematically identical)."""
import json, os
import pandas as pd
from . import tps

def load_base_with_adapters(base_id, adapters, hf_token=None):
    """adapters: {name: path}. Returns (model, tokenizer). Mirrors Stage 4 loading: 4-bit NF4 + double quant, use_cache=False."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dt, bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(base_id, quantization_config=bnb, torch_dtype=dt, low_cpu_mem_usage=True, device_map="auto",
                                                token=hf_token, trust_remote_code=True)
    base.config.use_cache = False
    tok = AutoTokenizer.from_pretrained(base_id, token=hf_token, trust_remote_code=True); tok.padding_side = "left"
    names = list(adapters)
    if not names:
        return base.eval(), tok
    model = PeftModel.from_pretrained(base, adapters[names[0]], adapter_name=names[0])
    for n in names[1:]:
        model.load_adapter(adapters[n], adapter_name=n)
    return model.eval(), tok

def build_messages(prompts, contexts, query):
    system = prompts["stage4_system_prefix"] + "\n\n" + tps.build_eval_context_text(contexts, prompts)
    return [{"role": "system", "content": system}, {"role": "user", "content": query}]

def generate_run(model, tok, eval_df, all_contexts, prompts, embed, seed=None, arm=None, run=None, ckpt=None, save_every=5):
    """One pass over the evaluation set. Resumes from `ckpt` (json list). Returns list of records."""
    import torch
    s = prompts["stage4_sampling"]
    if seed is not None:
        torch.manual_seed(seed)
    results = json.load(open(ckpt)) if ckpt and os.path.exists(ckpt) else []
    done = {r["query_id"] for r in results}
    eos = [tok.eos_token_id]; im = tok.convert_tokens_to_ids("<|im_end|>")
    if im is not None and im != tok.unk_token_id and im not in eos:
        eos.append(im)
    for i, ex in enumerate(eval_df.itertuples()):
        if ex.query_id in done:
            continue
        ctxs = all_contexts[i]
        text = tok.apply_chat_template(build_messages(prompts, ctxs, ex.prompt), tokenize=False, add_generation_prompt=True)
        inp = tok(text, return_tensors="pt", truncation=True, max_length=s["input_max_length"]).to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=s["max_new_tokens"], do_sample=True, temperature=s["temperature"], top_p=s["top_p"],
                                 repetition_penalty=s["repetition_penalty"], eos_token_id=eos, pad_token_id=tok.eos_token_id)
        n = inp.input_ids.shape[1]
        resp = tok.decode(out[0][n:], skip_special_tokens=True).strip() if out.shape[1] > n else ""
        score, det = tps.eval_score(resp, ex.prompt, ctxs, embed)
        flag, why = tps.hallucination_flag(resp, score, det, [c["filename"] for c in ctxs])
        results.append({"query_id": int(ex.query_id), "kode": ex.kode, "model": arm, "run": run, "response": resp, "dahl_score": score,
                        "dim1_relevance": det.get("dim1_relevance"), "dim2_grounding": det.get("dim2_grounding"), "dim3_lexical": det.get("dim3_lexical"),
                        "hallucination_flag": bool(flag), "flag_reasons": "|".join(why), "seed": seed})
        del inp, out
        if ckpt and len(results) % save_every == 0:
            json.dump(results, open(ckpt, "w", encoding="utf8"), ensure_ascii=False)
    if ckpt:
        json.dump(results, open(ckpt, "w", encoding="utf8"), ensure_ascii=False)
    return results

def generate_all(cfg, prompts, eval_df, all_contexts, adapters, embed, out_dir, hf_token=None, include_baseline=False):
    """Every adapter (and the baseline if include_baseline), for each of n_runs seeds. Checkpoints per (arm, run) so a dropped Colab session resumes."""
    import torch
    model, tok = load_base_with_adapters(cfg["training"]["base_model"], adapters, hf_token)
    n_runs = cfg["evaluation"]["n_runs"]; seeds = cfg["evaluation"]["run_seeds"]; frames = []
    for run in range(1, n_runs + 1):
        for arm in (["baseline"] if include_baseline else []) + list(adapters):
            ckpt = os.path.join(out_dir, f"responses_{arm}_run{run}.json")
            if arm == "baseline" and adapters:
                with model.disable_adapter():
                    res = generate_run(model, tok, eval_df, all_contexts, prompts, embed, seeds[run - 1], arm, run, ckpt)
            else:
                if arm != "baseline":
                    model.set_adapter(arm)
                res = generate_run(model, tok, eval_df, all_contexts, prompts, embed, seeds[run - 1], arm, run, ckpt)
            frames.append(pd.DataFrame(res))
    del model; import gc; gc.collect(); torch.cuda.empty_cache()
    return pd.concat(frames, ignore_index=True)
