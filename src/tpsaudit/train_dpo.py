"""LoRA-DPO training with every setting taken from the config (TRL 0.12.2 API; see requirements-colab.txt)."""
import json
from pathlib import Path
import pandas as pd

def build_prompt(tok, prompts, question, context=None):
    user = (prompts["user_grounded"].format(context=context, question=question) if context
            else prompts["user_ungrounded"].format(question=question))
    msgs = [{"role": "system", "content": prompts["system_grounded"]}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def to_dpo_records(df, tok, prompts, grounded_prompt=True):
    """Prompt = grounded prompt (with retrieved context) so that training matches evaluation."""
    recs = []
    for r in df.itertuples():
        ctx = r.context if grounded_prompt and isinstance(getattr(r, "context", None), str) and r.context else None
        recs.append({"prompt": build_prompt(tok, prompts, r.question, ctx),
                     "chosen": r.chosen.strip() + tok.eos_token, "rejected": r.rejected.strip() + tok.eos_token})
    return recs

def train_arm(cfg, prompts, train_df, out_dir, seed):
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import DPOTrainer, DPOConfig
    t = cfg["training"]; d = t["dpo"]; out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(t["base_model"]); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16) if t["load_in_4bit"] else None
    model = AutoModelForCausalLM.from_pretrained(t["base_model"], quantization_config=q, device_map="auto", torch_dtype=torch.float16)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=d["gradient_checkpointing"])
    lc = LoraConfig(r=t["lora"]["r"], lora_alpha=t["lora"]["alpha"], lora_dropout=t["lora"]["dropout"],
                    target_modules=t["lora"]["target_modules"], task_type="CAUSAL_LM")
    ds = Dataset.from_list(to_dpo_records(train_df, tok, prompts))
    args = DPOConfig(output_dir=str(out), num_train_epochs=d["epochs"], per_device_train_batch_size=d["per_device_batch"],
                     gradient_accumulation_steps=d["grad_accum"], learning_rate=d["learning_rate"], lr_scheduler_type=d["lr_scheduler"],
                     warmup_ratio=d["warmup_ratio"], beta=d["beta"], max_length=d["max_length"], max_prompt_length=d["max_prompt_length"],
                     optim=d["optimizer"], gradient_checkpointing=d["gradient_checkpointing"], fp16=True, bf16=False,
                     logging_steps=5, save_strategy="no", report_to="none", seed=seed, remove_unused_columns=False)
    # ref_model=None with a PEFT config: the reference policy is the base model with the adapter disabled
    trainer = DPOTrainer(model=model, ref_model=None, args=args, train_dataset=ds, processing_class=tok, peft_config=lc)
    res = trainer.train()
    trainer.model.save_pretrained(str(out / "adapter"))
    info = {"n_examples": len(ds), "optimizer_steps": int(trainer.state.global_step), "train_loss": float(res.training_loss),
            "effective_batch_size": d["per_device_batch"] * d["grad_accum"], "seed": seed,
            "log_history": trainer.state.log_history}
    (out / "train_summary.json").write_text(json.dumps(info, indent=2))
    del trainer, model; import gc; gc.collect(); torch.cuda.empty_cache()
    return info
