"""Phase A SFT evaluation + sample generations.

Inputs:
  --sampler-path  Tinker sampler weights URL (from
                  outputs/phase_a_qa_run_summary.json → final_sampler_path)
  --config        configs/tinker/phase_a_qa/run_v1.yaml (for base_model + renderer)

Outputs:
  outputs/phase_a_qa_test_results.jsonl              (one row per test example)
  outputs/phase_a_qa_held_out_source_results.jsonl   (one row per held-out example)
  outputs/phase_a_qa_samples.jsonl                   (15 generations)
  outputs/phase_a_qa_eval_summary.json               (aggregate metrics for journal)

Loss is computed per example via compute_logprobs(prompt+target) over the
assistant-tokens-only weight mask. Token accuracy is the per-token fraction
where argmax(logprobs) (approximated by the highest-likelihood token at each
position) matches the target. Since SamplingClient.compute_logprobs returns
*per-position teacher-forced* logprobs for the actual target tokens, we can
recover per-token NLL but not exact argmax without sampling. We therefore
report mean NLL as the primary loss metric and token-NLL exp-1 as a proxy
for likelihood quality.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import tinker  # noqa: E402
from tinker import SamplingParams  # noqa: E402
from tinker_cookbook.renderers import get_renderer  # noqa: E402
from tinker_cookbook.renderers.base import TrainOnWhat  # noqa: E402
from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa: E402

console = Console()

DEFAULT_CONFIG = ROOT / "configs" / "tinker" / "phase_a_qa" / "run_v1.yaml"

TEST_PATH = ROOT / "data" / "processed" / "qa" / "test.jsonl"
HELD_OUT_PATH = ROOT / "data" / "processed" / "qa" / "held_out_source.jsonl"

SAMPLE_N = 15
SAMPLE_SEED = 42
GEN_TEMPERATURE = 0.3
GEN_MAX_TOKENS = 300


def variant_paths(variant: str | None) -> dict[str, Path]:
    """Return the path set for a given variant ('v1', 'v2', ... or None=baseline)."""
    if variant:
        prefix = variant
    else:
        prefix = "phase_a_qa"
    return {
        "run_summary": ROOT / "outputs" / f"{prefix}_run_summary.json",
        "test_results": ROOT / "outputs" / f"{prefix}_test_results.jsonl",
        "held_out_results": ROOT / "outputs" / f"{prefix}_held_out_source_results.jsonl",
        "samples": ROOT / "outputs" / f"{prefix}_samples.jsonl",
        "eval_summary": ROOT / "outputs" / f"{prefix}_eval_summary.json",
    }


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


async def compute_loss_for_example(
    sampling_client: tinker.SamplingClient,
    renderer,
    example: dict,
) -> dict:
    """Teacher-force the assistant turn and return per-example loss stats."""
    model_input, weights = renderer.build_supervised_example(
        example["messages"], train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE
    )
    # compute_logprobs returns the logprob of each *next* token in the prompt.
    fut = sampling_client.compute_logprobs(model_input)
    logprobs = fut.result() if hasattr(fut, "result") else await fut.result_async()
    # Some SDK versions return a list; some a ConcurrentFuture-wrapped object.
    if hasattr(logprobs, "result"):
        logprobs = logprobs.result()

    weights_list = weights.tolist() if hasattr(weights, "tolist") else list(weights)
    # logprobs is per-position over the prompt; weights mask which positions count.
    # The two must align — if SDK returns length+1 (prompt+next), slice.
    n = min(len(logprobs), len(weights_list))
    total_lp = 0.0
    total_w = 0.0
    counted = 0
    for i in range(n):
        if weights_list[i] and logprobs[i] is not None:
            total_lp += float(logprobs[i]) * float(weights_list[i])
            total_w += float(weights_list[i])
            counted += 1
    if total_w == 0:
        return {
            "pair_id": example["pair_id"],
            "source_filename": example["source_filename"],
            "question_type": example["question_type"],
            "loss_tokens": 0,
            "mean_nll": float("nan"),
        }
    return {
        "pair_id": example["pair_id"],
        "source_filename": example["source_filename"],
        "question_type": example["question_type"],
        "loss_tokens": counted,
        "mean_nll": -total_lp / total_w,
    }


def aggregate_loss(results: list[dict]) -> dict:
    valid = [r for r in results if not math.isnan(r["mean_nll"])]
    if not valid:
        return {"n": 0, "mean_nll": float("nan")}
    total_w = sum(r["loss_tokens"] for r in valid)
    if total_w == 0:
        return {"n": len(valid), "mean_nll": float("nan")}
    weighted = sum(r["mean_nll"] * r["loss_tokens"] for r in valid)
    by_type: dict[str, list[float]] = defaultdict(list)
    for r in valid:
        by_type[r["question_type"]].append(r["mean_nll"])
    type_means = {
        qt: {"n": len(xs), "mean_nll": sum(xs) / len(xs)} for qt, xs in by_type.items()
    }
    return {
        "n": len(valid),
        "mean_nll": weighted / total_w,
        "by_question_type": type_means,
    }


async def evaluate_split(
    sampling_client,
    renderer,
    examples: list[dict],
    out_path: Path,
    label: str,
) -> dict:
    results: list[dict] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"eval ({label})", total=len(examples))
        for ex in examples:
            try:
                row = await compute_loss_for_example(sampling_client, renderer, ex)
            except Exception as exc:
                row = {
                    "pair_id": ex["pair_id"],
                    "source_filename": ex["source_filename"],
                    "question_type": ex["question_type"],
                    "loss_tokens": 0,
                    "mean_nll": float("nan"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            results.append(row)
            progress.advance(task)
    write_jsonl(out_path, results)
    return aggregate_loss(results)


async def generate_samples(
    sampling_client,
    renderer,
    examples: list[dict],
) -> list[dict]:
    rng = random.Random(SAMPLE_SEED)
    picks = rng.sample(examples, min(SAMPLE_N, len(examples)))
    samples: list[dict] = []
    params = SamplingParams(
        max_tokens=GEN_MAX_TOKENS,
        temperature=GEN_TEMPERATURE,
        stop=renderer.get_stop_sequences(),
    )
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("generate", total=len(picks))
        for ex in picks:
            messages = ex["messages"]
            prompt = renderer.build_generation_prompt(messages[:-1])
            fut = sampling_client.sample(
                prompt=prompt, num_samples=1, sampling_params=params
            )
            resp = fut.result() if hasattr(fut, "result") else await fut.result_async()
            tokens = resp.sequences[0].tokens
            parsed, _ = renderer.parse_response(tokens)
            content = parsed["content"]
            if not isinstance(content, str):
                # gpt-oss may return content as parts list
                content = "".join(
                    p.get("text") or p.get("thinking") or ""
                    for p in content
                    if isinstance(p, dict)
                )
            samples.append(
                {
                    "pair_id": ex["pair_id"],
                    "source_filename": ex["source_filename"],
                    "question_type": ex["question_type"],
                    "user_question": messages[1]["content"],
                    "reference_answer": messages[2]["content"],
                    "model_generation": content,
                }
            )
            progress.advance(task)
    return samples


async def main_async(args: argparse.Namespace) -> int:
    load_dotenv()
    cfg = load_yaml(args.config)
    base_model = cfg.get("base_model", "openai/gpt-oss-20b")
    renderer_name = cfg.get("renderer_name", "gpt_oss_no_sysprompt")

    paths = variant_paths(args.variant)
    sampler_path = args.sampler_path
    if not sampler_path:
        if paths["run_summary"].exists():
            data = json.loads(paths["run_summary"].read_text())
            sampler_path = data.get("final_sampler_path")
    if not sampler_path:
        raise SystemExit(
            f"STOP: no --sampler-path given and no final_sampler_path in "
            f"{paths['run_summary']}"
        )
    variant_tag = args.variant or "baseline"
    console.print(f"variant: {variant_tag}")
    console.print(f"sampler: {sampler_path}")
    console.print(f"base_model: {base_model}; renderer: {renderer_name}")

    tokenizer = get_tokenizer(base_model)
    renderer = get_renderer(renderer_name, tokenizer, model_name=base_model)

    service = tinker.ServiceClient()
    sampling_client = service.create_sampling_client(
        model_path=sampler_path, base_model=base_model
    )

    console.rule(f"[{variant_tag}] Evaluating test.jsonl")
    test_examples = read_jsonl(TEST_PATH)
    test_agg = await evaluate_split(
        sampling_client, renderer, test_examples, paths["test_results"], f"test/{variant_tag}"
    )
    console.print(f"test aggregate: {test_agg}")

    console.rule(f"[{variant_tag}] Evaluating held_out_source.jsonl")
    held_examples = read_jsonl(HELD_OUT_PATH)
    held_agg = await evaluate_split(
        sampling_client, renderer, held_examples, paths["held_out_results"],
        f"held-out/{variant_tag}"
    )
    console.print(f"held-out aggregate: {held_agg}")

    console.rule(f"[{variant_tag}] Generating 15 samples")
    samples = await generate_samples(sampling_client, renderer, test_examples)
    write_jsonl(paths["samples"], samples)
    console.print(f"wrote {len(samples)} samples")

    paths["eval_summary"].parent.mkdir(parents=True, exist_ok=True)
    paths["eval_summary"].write_text(
        json.dumps(
            {
                "variant": variant_tag,
                "sampler_path": sampler_path,
                "base_model": base_model,
                "renderer_name": renderer_name,
                "test": test_agg,
                "held_out_source": held_agg,
                "n_samples": len(samples),
            },
            indent=2,
        )
    )
    console.print(f"summary: {paths['eval_summary']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG), type=Path,
    )
    parser.add_argument("--sampler-path", default=None, type=str,
                        help="Override; if omitted, read from {variant}_run_summary.json")
    parser.add_argument("--variant", default=None,
                        help="Variant name (v1/v2/v3/v4 etc.); namespaces all outputs")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
