import yaml, json, hashlib, platform, sys, time
from pathlib import Path

def load(path="configs/default.yaml"):
    with open(path, encoding="utf8") as f:
        return yaml.safe_load(f)

def snapshot(cfg, out_dir):
    """Write the exact configuration and environment next to every result (reviewer: record config per run)."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    env = {"python": sys.version, "platform": platform.platform(), "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for pkg in ("torch", "transformers", "trl", "peft", "accelerate", "bitsandbytes", "sentence_transformers"):
        try:
            env[pkg] = __import__(pkg).__version__
        except Exception:
            env[pkg] = None
    blob = json.dumps(cfg, sort_keys=True)
    (out / "config_snapshot.json").write_text(json.dumps({"config": cfg, "env": env, "config_sha256": hashlib.sha256(blob.encode()).hexdigest()}, indent=2, ensure_ascii=False))
