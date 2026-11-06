"""Generation backends used by the judges and the DAHL-style fact extractor.
All backends expose .generate(list_of_chat_messages, max_new_tokens) -> list[str] (greedy, temperature 0)."""
import json, os, time

class MockBackend:
    def __init__(self, fn): self.fn = fn; self.name = "mock"
    def generate(self, batch, max_new_tokens=300): return [self.fn(m) for m in batch]

class HFBackend:
    """Local open model in 4-bit (fits a free Colab T4 for 7-9B models)."""
    def __init__(self, model_id, load_4bit=True, batch_size=4, max_input_tokens=3500):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        self.name, self.bs, self.max_in = model_id, batch_size, max_input_tokens
        self.tok = AutoTokenizer.from_pretrained(model_id); self.tok.padding_side = "left"
        if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16) if load_4bit else None
        self.model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=q, device_map="auto",
                                                          torch_dtype=torch.float16).eval()
        self.torch = torch
    def _prompt(self, msgs):
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:   # templates without a system role (e.g. Gemma): merge it into the user turn
            sys_ = "\n".join(m["content"] for m in msgs if m["role"] == "system")
            user = "\n".join(m["content"] for m in msgs if m["role"] != "system")
            return self.tok.apply_chat_template([{"role": "user", "content": sys_ + "\n\n" + user}], tokenize=False, add_generation_prompt=True)
    def generate(self, batch, max_new_tokens=300):
        out = []
        for i in range(0, len(batch), self.bs):
            texts = [self._prompt(m) for m in batch[i:i + self.bs]]
            enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_in, add_special_tokens=False).to(self.model.device)
            with self.torch.no_grad():
                g = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tok.pad_token_id)
            out += self.tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        return [o.strip() for o in out]
    def close(self):
        import gc; del self.model; gc.collect(); self.torch.cuda.empty_cache()

class OpenAICompatBackend:
    """Any OpenAI-compatible chat endpoint. Key is read from an environment variable, never stored."""
    def __init__(self, model, base_url="https://api.openai.com/v1", key_env="JUDGE_API_KEY", sleep=0.0):
        self.name, self.base, self.key_env, self.sleep = model, base_url.rstrip("/"), key_env, sleep
    def generate(self, batch, max_new_tokens=300):
        import requests
        out = []
        for msgs in batch:
            r = requests.post(self.base + "/chat/completions", timeout=120,
                              headers={"Authorization": "Bearer " + os.environ[self.key_env]},
                              json={"model": self.name, "messages": msgs, "temperature": 0, "max_tokens": max_new_tokens})
            r.raise_for_status(); out.append(r.json()["choices"][0]["message"]["content"].strip()); time.sleep(self.sleep)
        return out

def make_backend(spec):
    if spec["backend"] == "hf": return HFBackend(spec["model"], batch_size=spec.get("batch_size", 4))
    if spec["backend"] == "openai_compat":
        return OpenAICompatBackend(spec["model"], spec.get("base_url", "https://api.openai.com/v1"), spec.get("key_env", "JUDGE_API_KEY"))
    raise ValueError(spec["backend"])
