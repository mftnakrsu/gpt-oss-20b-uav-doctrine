"""Push the v4 LoRA adapter to huggingface.co/meftun/gpt-oss-20b-uav-doctrine.

Uploads ONLY the three adapter files — never training data, journals, or
secrets:
  adapter_model.safetensors
  adapter_config.json
  README.md (the model card)

Public repo, Apache-2.0 (matches the gpt-oss-20b base license).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

ROOT = Path(__file__).resolve().parents[1]
console = Console()

REPO_ID = "meftun/gpt-oss-20b-uav-doctrine"
ADAPTER_DIR = ROOT / "checkpoints" / "phase_a_qa" / "v4" / "adapter"
ALLOWED = {"adapter_model.safetensors", "adapter_config.json", "README.md"}
COMMIT_MESSAGE = "Initial Phase A v4 LoRA adapter + model card"


def main() -> int:
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", default=REPO_ID)
    p.add_argument("--private", action="store_true",
                   help="Create private (default: public, per project decision)")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("STOP: HF_TOKEN not set")

    from huggingface_hub import HfApi

    # Guard: ensure exactly the three expected files exist, nothing else.
    present = {p.name for p in ADAPTER_DIR.iterdir() if p.is_file()}
    missing = ALLOWED - present
    extra = present - ALLOWED
    if missing:
        raise SystemExit(f"STOP: adapter dir missing files: {missing}")
    if extra:
        console.print(f"[yellow]note: extra files present, will NOT be uploaded "
                      f"(allow_patterns restricts): {extra}[/yellow]")

    api = HfApi(token=token)
    whoami = api.whoami()
    console.print(f"authenticated as: {whoami.get('name')}")

    console.rule("create_repo")
    url = api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    console.print(f"repo: {url}")

    console.rule("upload_folder (3 files only)")
    commit_info = api.upload_folder(
        repo_id=args.repo_id,
        folder_path=str(ADAPTER_DIR),
        allow_patterns=sorted(ALLOWED),  # hard allow-list — nothing else uploads
        commit_message=COMMIT_MESSAGE,
    )
    console.print(f"commit: {commit_info}")

    console.rule("verify repo contents")
    files = api.list_repo_files(repo_id=args.repo_id)
    console.print(f"files in repo: {sorted(files)}")
    for needed in ALLOWED:
        if needed not in files:
            raise SystemExit(f"STOP: {needed} not found in repo after push")

    public_url = f"https://huggingface.co/{args.repo_id}"
    commit_hash = getattr(commit_info, "oid", None) or getattr(commit_info, "commit_id", None)
    console.print(f"\n[green]PUSH OK[/green]")
    console.print(f"public URL: {public_url}")
    console.print(f"commit hash: {commit_hash}")

    # Persist push metadata for the journal writer.
    (ADAPTER_DIR.parent / "hf_push.json").write_text(
        __import__("json").dumps({
            "repo_id": args.repo_id,
            "public_url": public_url,
            "commit_hash": str(commit_hash),
            "files": sorted(files),
            "private": args.private,
            "commit_message": COMMIT_MESSAGE,
        }, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
