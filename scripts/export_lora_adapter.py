"""Download the v4 LoRA adapter from Tinker and convert it to PEFT format.

Why this is not a one-liner:
  tinker_cookbook.weights.download() works, but the default Tinker SDK
  transport (pyqwest, a Rust HTTP client) resets the long-running HTTP/2
  stream while the server assembles the checkpoint archive. We force
  httpx's default transport (HTTP/1.1) before fetching the signed URL,
  which survives the wait. The actual archive download is a plain GCS
  signed-URL GET (transport-independent).

Flow:
  1. publish the checkpoint (idempotent) so it is downloadable
  2. fetch the signed archive URL (pyqwest disabled)
  3. download + safe-extract the tar to raw_tinker/
  4. convert raw_tinker/ → PEFT adapter/ via
     tinker_cookbook.weights.build_lora_adapter (needs base-model config)
  5. verify adapter_model.safetensors + adapter_config.json

Outputs:
  checkpoints/phase_a_qa/v4/raw_tinker/   (Tinker-native adapter)
  checkpoints/phase_a_qa/v4/adapter/      (PEFT-format adapter — pushed to HF)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import tinker  # noqa: E402

console = Console()

V4_SAMPLER_URL = (
    "tinker://d79c263a-1e99-57b9-b09e-584fc222f3da:train:0/sampler_weights/final-step-00994"
)
BASE_MODEL = "openai/gpt-oss-20b"

V4_DIR = ROOT / "checkpoints" / "phase_a_qa" / "v4"
RAW_DIR = V4_DIR / "raw_tinker"
ADAPTER_DIR = V4_DIR / "adapter"


def _service_client_no_pyqwest(timeout: float = 900.0) -> tinker.ServiceClient:
    """ServiceClient pinned to httpx default transport (no pyqwest).

    pyqwest resets the stream during long archive assembly; httpx HTTP/1.1
    waits it out.
    """
    sc = tinker.ServiceClient(timeout=timeout)
    holder = sc.holder
    holder._constructor_kwargs["_use_pyqwest"] = False
    holder._client_pools = {}
    return sc


def _safe_extract_tar(archive_path: Path, extract_dir: Path) -> None:
    """Extract a tar, rejecting symlinks/hardlinks and path traversal."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path) as tf:
        for member in tf.getmembers():
            target = (extract_dir / member.name).resolve()
            if not str(target).startswith(str(extract_dir.resolve())):
                raise RuntimeError(f"unsafe path in archive: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"link in archive rejected: {member.name}")
        tf.extractall(extract_dir)


def download_raw_adapter(tinker_path: str, out_dir: Path) -> None:
    rc_sc = _service_client_no_pyqwest()
    rc = rc_sc.create_rest_client()

    console.print("publishing checkpoint (idempotent)…")
    try:
        rc.publish_checkpoint_from_tinker_path(tinker_path).result()
    except Exception as exc:  # publish is best-effort; archive fetch is the real gate
        console.print(f"[yellow]publish warning: {type(exc).__name__}: {exc}[/yellow]")

    console.print("fetching signed archive URL (pyqwest disabled)…")
    resp = rc.get_checkpoint_archive_url_from_tinker_path(tinker_path).result()
    url = resp.url
    console.print(f"archive host: {url.split('?')[0].split('/')[2]}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        console.print("downloading archive…")
        urllib.request.urlretrieve(url, str(tmp_path))
        size_mb = tmp_path.stat().st_size / 1e6
        console.print(f"downloaded {size_mb:.1f} MB; extracting…")
        _safe_extract_tar(tmp_path, out_dir)
    finally:
        tmp_path.unlink(missing_ok=True)

    files = sorted(p.name for p in out_dir.rglob("*") if p.is_file())
    console.print(f"extracted files: {files}")


def _prefetch_base_keys() -> tuple[dict, set[str]]:
    """Fetch ONLY config.json + the safetensors index (a few hundred KB),
    not the 13.76 GB of weights.

    The gpt_oss adapter conversion needs the base model's *parameter key
    names* (for attention key remapping + profile detection) but never the
    tensor data — its MergeProfile sets no fused_projection_map, so
    get_model_state_shapes is never called. We read the key names out of
    model.safetensors.index.json instead of downloading the shards.
    """
    import os
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")
    cfg_path = hf_hub_download(BASE_MODEL, "config.json", token=token)
    idx_path = hf_hub_download(BASE_MODEL, "model.safetensors.index.json", token=token)
    config = json.loads(Path(cfg_path).read_text())
    index = json.loads(Path(idx_path).read_text())
    keys = set(index["weight_map"].keys())
    return config, keys


def convert_to_peft(raw_dir: Path, adapter_dir: Path) -> None:
    import tinker_cookbook.weights._adapter as adapter_mod
    from tinker_cookbook.weights._adapter import build_lora_adapter

    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)

    candidates = [raw_dir] + [p.parent for p in raw_dir.rglob("adapter_model.safetensors")]
    adapter_src = next((c for c in candidates if (c / "adapter_model.safetensors").exists()), raw_dir)
    console.print(f"adapter source dir: {adapter_src}")

    config, model_keys = _prefetch_base_keys()
    console.print(f"base-model keys from index: {len(model_keys)} "
                  f"(arch={config.get('architectures')}) — no weight download")

    # Patch the three base-model accessors so build_lora_adapter uses the
    # lightweight index-derived keys instead of snapshot_download'ing 13.76 GB.
    orig_resolve = adapter_mod.resolve_model_dir
    orig_load_cfg = adapter_mod.load_config_dict
    orig_keys = adapter_mod.get_model_state_keys
    orig_shapes = adapter_mod.get_model_state_shapes
    adapter_mod.resolve_model_dir = lambda base_model: Path("/__virtual_base__")
    adapter_mod.load_config_dict = lambda *a, **k: config
    adapter_mod.get_model_state_keys = lambda model_dir: model_keys

    def _no_shapes(model_dir):
        raise AssertionError(
            "get_model_state_shapes called — gpt_oss should not need tensor shapes"
        )

    adapter_mod.get_model_state_shapes = _no_shapes
    try:
        build_lora_adapter(
            base_model=BASE_MODEL,
            adapter_path=str(adapter_src),
            output_path=str(adapter_dir),
        )
    finally:
        adapter_mod.resolve_model_dir = orig_resolve
        adapter_mod.load_config_dict = orig_load_cfg
        adapter_mod.get_model_state_keys = orig_keys
        adapter_mod.get_model_state_shapes = orig_shapes

    # build_lora_adapter writes base_model_name_or_path from the base_model
    # arg; ensure it is the canonical HF id (not a local path), for publishing.
    cfg_path = adapter_dir / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["base_model_name_or_path"] = BASE_MODEL
    cfg_path.write_text(json.dumps(cfg, indent=2))
    console.print(f"PEFT adapter written to {adapter_dir}")


def verify_adapter(adapter_dir: Path) -> dict:
    from safetensors import safe_open

    st_path = adapter_dir / "adapter_model.safetensors"
    cfg_path = adapter_dir / "adapter_config.json"
    if not st_path.exists():
        raise SystemExit(f"STOP: missing {st_path}")
    if not cfg_path.exists():
        raise SystemExit(f"STOP: missing {cfg_path}")

    cfg = json.loads(cfg_path.read_text())
    required = ["base_model_name_or_path", "r", "lora_alpha", "target_modules", "peft_type"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise SystemExit(f"STOP: adapter_config.json missing keys: {missing}")

    keys = []
    with safe_open(str(st_path), framework="pt") as f:
        keys = list(f.keys())
    lora_keys = [k for k in keys if "lora_A" in k or "lora_B" in k]
    if not lora_keys:
        raise SystemExit("STOP: no lora_A/lora_B tensors found in safetensors")

    size_mb = st_path.stat().st_size / 1e6
    if not (50 <= size_mb <= 800):
        console.print(f"[yellow]warning: adapter size {size_mb:.1f} MB outside expected 50-800 MB[/yellow]")

    summary = {
        "safetensors_size_mb": round(size_mb, 1),
        "n_tensors": len(keys),
        "n_lora_tensors": len(lora_keys),
        "base_model": cfg.get("base_model_name_or_path"),
        "r": cfg.get("r"),
        "lora_alpha": cfg.get("lora_alpha"),
        "peft_type": cfg.get("peft_type"),
        "task_type": cfg.get("task_type"),
        "n_target_modules": len(cfg.get("target_modules", [])) if isinstance(cfg.get("target_modules"), list) else "n/a",
        "target_modules_sample": (cfg.get("target_modules", [])[:6]
                                  if isinstance(cfg.get("target_modules"), list) else cfg.get("target_modules")),
    }
    return summary


def main() -> int:
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-download", action="store_true",
                   help="Reuse existing raw_tinker/ (skip Tinker fetch)")
    p.add_argument("--skip-convert", action="store_true",
                   help="Only download; do not run PEFT conversion")
    args = p.parse_args()

    if not args.skip_download:
        console.rule("Download raw Tinker adapter")
        download_raw_adapter(V4_SAMPLER_URL, RAW_DIR)
    else:
        console.print(f"[yellow]--skip-download: reusing {RAW_DIR}[/yellow]")

    if args.skip_convert:
        console.print("[yellow]--skip-convert: stopping after download[/yellow]")
        return 0

    console.rule("Convert to PEFT format")
    convert_to_peft(RAW_DIR, ADAPTER_DIR)

    console.rule("Verify")
    summary = verify_adapter(ADAPTER_DIR)
    console.print_json(data=summary)
    (V4_DIR / "export_summary.json").write_text(json.dumps(summary, indent=2))
    console.print("[green]adapter export verified[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
