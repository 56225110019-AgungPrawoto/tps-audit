"""LoRA-DPO training = Stage 3 notebook, parametrised. Every value comes from configs/default.yaml (copied from Stage 3).
Differences from the original, all listed in docs/DEVIATIONS.md: explicit `seed`, one output dir per arm, checkpoints every 10 steps (resumable)."""
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
    # GPU without native bf16 (Colab T4, compute capability 7.5): use fp16 instead and turn off the TRL kernel path that needs capability >= 8.0.
    # The original run used bf16 (Stage 3); this fallback is listed in docs/DEVIATIONS.md.
    bf16_ok = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    dtype = torch.bfloat16 if bf16_ok else torch.float16
    if not bf16_ok:
        try:
            import trl.kernels.chunked_logprob as _k; _k._MM_OUT_DTYPE = False
        except Exception:
            pass
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    tok = AutoTokenizer.from_pretrained(t["base_model"], token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(t["base_model"], quantization_config=bnb, device_map="auto", token=hf_token, low_cpu_mem_usage=True, torch_dtype=dtype)   # explicit: newer transformers default to the checkpoint dtype (bf16)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"], target_modules=lc["target_modules"], bias=lc["bias"], task_type="CAUSAL_LM")
    base = dict(output_dir=str(out), beta=d["beta"], learning_rate=d["learning_rate"], per_device_train_batch_size=d["per_device_batch"],
                gradient_accumulation_steps=d["grad_accum"], num_train_epochs=d["epochs"], optim=d["optimizer"], lr_scheduler_type=d["lr_scheduler"],
                max_length=d["max_length"], bf16=bf16_ok, fp16=not bf16_ok, gradient_checkpointing=True,
                logging_steps=5, save_strategy="steps", save_steps=10, save_total_limit=2, report_to="none", seed=seed)   # checkpoints so a dropped Colab session can resume
    try:
        args = DPOConfig(warmup_ratio=d["warmup_ratio"], **base)        # older transformers (used in the original run, deprecated there)
    except TypeError:
        args = DPOConfig(warmup_steps=d["warmup_ratio"], **base)        # newer transformers: a float < 1 is read as a ratio of total steps
    kw = dict(model=model, args=args, train_dataset=ds, peft_config=lora)     # no ref_model: reference = adapter disabled
    try:
        trainer = DPOTrainer(tokenizer=tok, **kw)
    except TypeError:
        trainer = DPOTrainer(processing_class=tok, **kw)
    if not bf16_ok:      # fp16 GradScaler cannot unscale bf16/fp16 gradients: keep every trainable (LoRA) parameter in fp32
        for prm in trainer.model.parameters():
            if prm.requires_grad and prm.dtype != torch.float32:
                prm.data = prm.data.float()
    ckpts = sorted(out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    res = trainer.train(resume_from_checkpoint=str(ckpts[-1]) if ckpts else None)
    trainer.save_model(str(out)); tok.save_pretrained(str(out))
    info = {"n_examples": len(ds), "optimizer_steps": int(trainer.state.global_step), "expected_steps": expected_steps(len(ds), d["per_device_batch"], d["grad_accum"], d["epochs"]),
            "train_loss": float(res.training_loss), "effective_batch_size": d["per_device_batch"] * d["grad_accum"], "seed": seed,
            "log_history": trainer.state.log_history}
    (out / "train_summary.json").write_text(json.dumps(info, indent=2))
    del trainer, model; gc.collect(); torch.cuda.empty_cache()
    return info
