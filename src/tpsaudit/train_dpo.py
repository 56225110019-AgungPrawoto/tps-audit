"""LoRA-DPO training = Stage 3 notebook, parametrised. Every value comes from configs/default.yaml (copied from Stage 3).
Differences from the original, all listed in docs/DEVIATIONS.md: explicit `seed`, one output dir per arm, save_strategy='no'."""
import gc, json
from pathlib import Path
import pandas as pd

def prepare_training_frame(df):
    """Stage 3 load_dpo_csv(): strip the four columns and drop rows with an empty field."""
    d = df.copy()
    for c in ("context", "question", "chosen", "rejected"):
        d[c] = d[c].astype(str).str.strip()
    return d[(d.context != "") & (d.question != "") & (d.chosen != "") & (d.rejected != "")].reset_index(drop=True)

def to_trl_conversational(row, prompts):
    """Stage 3 to_trl_conversational(): the SAME grounded prompt is used for chosen AND rejected."""
    user = prompts["stage3_user"].format(context=row["context"], query=row["question"])
    return {"prompt": [{"role": "system", "content": prompts["stage3_system"]}, {"role": "user", "content": user}],
            "chosen": [{"role": "assistant", "content": row["chosen"]}], "rejected": [{"role": "assistant", "content": row["rejected"]}]}

def expected_steps(n, per_device_batch, grad_accum, epochs=1):
    import math
    return math.ceil(n / (per_device_batch * grad_accum)) * epochs

def train_arm(cfg, prompts, train_df, out_dir, seed=42, hf_token=None):
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import DPOConfig, DPOTrainer
    t = cfg["training"]; d = t["dpo"]; lc = t["lora"]; out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df = prepare_training_frame(train_df)
    ds = Dataset.from_list([to_trl_conversational(r, prompts) for r in df.to_dict("records")])
    dtype = torch.bfloat16
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    tok = AutoTokenizer.from_pretrained(t["base_model"], token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(t["base_model"], quantization_config=bnb, device_map="auto", token=hf_token, low_cpu_mem_usage=True)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"], target_modules=lc["target_modules"], bias=lc["bias"], task_type="CAUSAL_LM")
    args = DPOConfig(output_dir=str(out), beta=d["beta"], learning_rate=d["learning_rate"], per_device_train_batch_size=d["per_device_batch"],
                     gradient_accumulation_steps=d["grad_accum"], num_train_epochs=d["epochs"], optim=d["optimizer"], lr_scheduler_type=d["lr_scheduler"],
                     warmup_ratio=d["warmup_ratio"], max_length=d["max_length"], bf16=True, fp16=False, gradient_checkpointing=True,
                     logging_steps=5, save_strategy="no", report_to="none", seed=seed)
    kw = dict(model=model, args=args, train_dataset=ds, peft_config=lora)     # no ref_model: reference = adapter disabled
    try:
        trainer = DPOTrainer(tokenizer=tok, **kw)
    except TypeError:
        trainer = DPOTrainer(processing_class=tok, **kw)
    res = trainer.train()
    trainer.save_model(str(out)); tok.save_pretrained(str(out))
    info = {"n_examples": len(ds), "optimizer_steps": int(trainer.state.global_step), "expected_steps": expected_steps(len(ds), d["per_device_batch"], d["grad_accum"], d["epochs"]),
            "train_loss": float(res.training_loss), "effective_batch_size": d["per_device_batch"] * d["grad_accum"], "seed": seed,
            "log_history": trainer.state.log_history}
    (out / "train_summary.json").write_text(json.dumps(info, indent=2))
    del trainer, model; gc.collect(); torch.cuda.empty_cache()
    return info
