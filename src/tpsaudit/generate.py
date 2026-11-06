"""Generate evaluation responses for the baseline and every trained adapter (same prompts, same seeds)."""
import pandas as pd
from .train_dpo import build_prompt
from .audit import has_zh, zh_share, words

def generate_all(cfg, prompts, eval_df, adapters, out_csv):
    """adapters: {arm_name: adapter_dir}. Baseline = the adapter-disabled model. Returns long dataframe."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    t = cfg["training"]; e = cfg["evaluation"]
    tok = AutoTokenizer.from_pretrained(t["base_model"]); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    base = AutoModelForCausalLM.from_pretrained(t["base_model"], quantization_config=q, device_map="auto", torch_dtype=torch.float16)
    names = list(adapters)
    model = PeftModel.from_pretrained(base, adapters[names[0]], adapter_name=names[0]) if names else base
    for n in names[1:]: model.load_adapter(adapters[n], adapter_name=n)
    model.eval()
    prompts_txt = [build_prompt(tok, prompts, r.question, r.context) for r in eval_df.itertuples()]
    rows = []
    def run(arm, seed, run_id):
        torch.manual_seed(seed)
        for i in range(0, len(prompts_txt), 8):
            enc = tok(prompts_txt[i:i + 8], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            with torch.no_grad():
                g = model.generate(**enc, do_sample=True, temperature=e["temperature"], top_p=e["top_p"],
                                   max_new_tokens=e["max_new_tokens"], pad_token_id=tok.pad_token_id)
            for j, txt in enumerate(tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)):
                r = eval_df.iloc[i + j]
                rows.append({"query_id": r.query_id, "run": run_id, "model": arm, "response": txt.strip()})
    for run_id, seed in enumerate(e["run_seeds"][:e["n_runs"]], start=1):
        if names:
            with model.disable_adapter(): run("baseline", seed, run_id)
            for n in names: model.set_adapter(n); run(n, seed, run_id)
        else: run("baseline", seed, run_id)
    df = pd.DataFrame(rows)
    df["words"] = df.response.map(words); df["has_zh"] = df.response.map(has_zh).astype(int); df["zh_share"] = df.response.map(zh_share)
    df["response_id"] = df.model + "|" + df.query_id.astype(str) + "|" + df.run.astype(str)
    df.to_csv(out_csv, index=False)
    return df
