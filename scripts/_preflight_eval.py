"""Pre-flight for Phase A judge-based eval — checks Tinker v4 + baseline + Azure.

One-shot script intended to run before scripts/eval_phase_a.py. Verifies:
  1. .env populated (TINKER_API_KEY, AZURE_API_KEY)
  2. test.jsonl and held_out_source.jsonl exist with expected sizes
  3. Tinker SamplingClient for v4 sampler URL — one short completion
  4. Tinker SamplingClient for baseline (no adapter) — one short completion
  5. Azure GPT-5.4 chat completion — "Reply OK."

Prints PASS/FAIL per check. Exit 0 only if all pass.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

V4_SAMPLER_URL = "tinker://d79c263a-1e99-57b9-b09e-584fc222f3da:train:0/sampler_weights/final-step-00994"
BASE_MODEL = "openai/gpt-oss-20b"
RENDERER_NAME = "gpt_oss_no_sysprompt"

TRIVIAL_PROMPT_MSGS = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Say the single word 'hello' and nothing else."},
]


def check_env() -> bool:
    load_dotenv()
    ok = True
    for k in ("TINKER_API_KEY", "AZURE_API_KEY",
              "AZURE_OPENAI_LLM_BASE_URL", "AZURE_OPENAI_LLM_DEPLOYMENT"):
        v = os.environ.get(k, "")
        marker = "PASS" if len(v) > 5 else "FAIL"
        print(f"[{marker}] env {k}: len={len(v)}")
        ok = ok and len(v) > 5
    return ok


def check_data() -> bool:
    ok = True
    for path, expected in [
        (ROOT / "data" / "processed" / "qa" / "test.jsonl", 1875),
        (ROOT / "data" / "processed" / "qa" / "held_out_source.jsonl", 15),
    ]:
        n = sum(1 for _ in path.open()) if path.exists() else 0
        good = path.exists() and n == expected
        marker = "PASS" if good else "FAIL"
        print(f"[{marker}] data {path.name}: exists={path.exists()} n={n} expected={expected}")
        ok = ok and good
    return ok


def _sample_tinker(model_path: str | None, label: str) -> bool:
    import tinker
    from tinker import SamplingParams
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    try:
        tokenizer = get_tokenizer(BASE_MODEL)
        renderer = get_renderer(RENDERER_NAME, tokenizer, model_name=BASE_MODEL)
        service = tinker.ServiceClient()
        sc = service.create_sampling_client(model_path=model_path, base_model=BASE_MODEL)
        prompt = renderer.build_generation_prompt(TRIVIAL_PROMPT_MSGS)
        params = SamplingParams(
            max_tokens=16, temperature=0.3, stop=renderer.get_stop_sequences()
        )
        fut = sc.sample(prompt=prompt, num_samples=1, sampling_params=params)
        resp = fut.result()
        tokens = resp.sequences[0].tokens
        parsed, _ = renderer.parse_response(tokens)
        content = parsed.get("content", "")
        if not isinstance(content, str):
            content = "".join(
                p.get("text") or p.get("thinking") or ""
                for p in content if isinstance(p, dict)
            )
        ok = len(content.strip()) > 0
        marker = "PASS" if ok else "FAIL"
        print(f"[{marker}] tinker sample ({label}): content={content[:60]!r}")
        return ok
    except Exception as exc:
        print(f"[FAIL] tinker sample ({label}): {type(exc).__name__}: {exc}")
        return False


def check_tinker_v4() -> bool:
    return _sample_tinker(V4_SAMPLER_URL, "v4")


def check_tinker_baseline() -> bool:
    return _sample_tinker(None, "baseline")


async def _azure_call() -> tuple[bool, str]:
    from finetune_slm.data.teacher import AzureLLMClient
    client = AzureLLMClient()
    try:
        content, usage = await client.chat_completion(
            [{"role": "user", "content": "Reply with only the two letters: OK."}],
            temperature=0.3,
            max_tokens=16,
        )
        return True, content.strip()
    finally:
        await client.close()


def check_azure() -> bool:
    try:
        ok, content = asyncio.run(_azure_call())
        marker = "PASS" if ok else "FAIL"
        print(f"[{marker}] azure gpt-5.4: content={content[:60]!r}")
        return ok
    except Exception as exc:
        print(f"[FAIL] azure gpt-5.4: {type(exc).__name__}: {exc}")
        return False


def main() -> int:
    print("=== Pre-flight: Phase A judge eval ===")
    results = [
        ("env", check_env()),
        ("data", check_data()),
        ("tinker_baseline", check_tinker_baseline()),
        ("tinker_v4", check_tinker_v4()),
        ("azure", check_azure()),
    ]
    print()
    for name, ok in results:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    all_ok = all(ok for _, ok in results)
    print(f"\nOverall: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
