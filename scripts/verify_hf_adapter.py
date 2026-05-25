"""Verify the pushed adapter downloads cleanly from HuggingFace.

From a fresh snapshot_download (no local references), confirm:
  - the three files download
  - adapter_config.json parses as a PEFT config
  - adapter_model.safetensors opens and contains LoRA tensors

Full inference verification (load gpt-oss-20b + adapter, generate) requires
the ~14 GB base model. It is skipped by default; pass --inference to attempt
it (needs sufficient disk + memory).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

ROOT = Path(__file__).resolve().parents[1]
console = Console()

REPO_ID = "meftun/gpt-oss-20b-uav-doctrine"
SYSTEM_PROMPT = (
    "You are an expert assistant on Unmanned Aerial Vehicle (UAV) combat "
    "doctrine. Provide accurate, concise answers grounded in US, NATO, and "
    "allied doctrinal publications."
)


def main() -> int:
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", default=REPO_ID)
    p.add_argument("--inference", action="store_true",
                   help="Attempt full base-model + adapter generation (needs ~14GB + GPU/CPU memory)")
    args = p.parse_args()

    from huggingface_hub import snapshot_download
    from peft import PeftConfig
    from safetensors import safe_open

    token = os.environ.get("HF_TOKEN")

    console.rule("snapshot_download (fresh)")
    local = snapshot_download(repo_id=args.repo_id, token=token)
    local_dir = Path(local)
    files = {p.name: p.stat().st_size for p in local_dir.iterdir() if p.is_file()}
    console.print(f"downloaded to: {local_dir}")
    for name, size in sorted(files.items()):
        console.print(f"  {name}: {size/1e6:.2f} MB")

    for needed in ("adapter_config.json", "adapter_model.safetensors", "README.md"):
        if needed not in files:
            raise SystemExit(f"STOP: {needed} missing from snapshot")

    console.rule("PEFT config parse")
    cfg = PeftConfig.from_pretrained(local_dir)
    config_summary = {
        "peft_type": str(cfg.peft_type),
        "base_model": cfg.base_model_name_or_path,
        "r": cfg.r,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "task_type": str(cfg.task_type),
        "target_modules": sorted(cfg.target_modules) if not isinstance(cfg.target_modules, str) else cfg.target_modules,
    }
    console.print_json(data=config_summary)

    console.rule("safetensors structure")
    st = local_dir / "adapter_model.safetensors"
    with safe_open(str(st), framework="pt") as f:
        keys = list(f.keys())
    lora_keys = [k for k in keys if "lora_A" in k or "lora_B" in k]
    console.print(f"tensors: {len(keys)}, lora tensors: {len(lora_keys)}")
    if not lora_keys:
        raise SystemExit("STOP: no LoRA tensors in downloaded safetensors")

    report = {
        "adapter_url": f"https://huggingface.co/{args.repo_id}",
        "downloaded_files": {k: round(v / 1e6, 2) for k, v in files.items()},
        "config": config_summary,
        "n_tensors": len(keys),
        "n_lora_tensors": len(lora_keys),
        "inference_check": "skipped (base model ~14GB)",
    }

    if args.inference:
        console.rule("inference check (base model + adapter)")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            tok = AutoTokenizer.from_pretrained(cfg.base_model_name_or_path, token=token)
            model = AutoModelForCausalLM.from_pretrained(
                cfg.base_model_name_or_path, torch_dtype="auto", device_map="auto", token=token
            )
            model = PeftModel.from_pretrained(model, local_dir)
            model.eval()
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Say the single word 'hello' and nothing else."},
            ]
            inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(inputs, max_new_tokens=64, temperature=0.3, do_sample=True)
            text = tok.decode(out[0, inputs.shape[1]:], skip_special_tokens=True)
            console.print(f"generation: {text[:200]!r}")
            report["inference_check"] = {"ok": True, "sample": text[:200]}
        except Exception as exc:
            console.print(f"[yellow]inference check failed/skipped: {type(exc).__name__}: {exc}[/yellow]")
            report["inference_check"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    out_path = ROOT / "checkpoints" / "phase_a_qa" / "v4" / "hf_verify.json"
    out_path.write_text(json.dumps(report, indent=2))
    console.print(f"\n[green]VERIFY OK[/green] — report at {out_path}")
    console.print_json(data=report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
