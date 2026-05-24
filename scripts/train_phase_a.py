"""Phase A SFT on openai/gpt-oss-20b via Tinker.

Reads:
  configs/tinker/phase_a_qa/run_v1.yaml   (YAML config; Python constants below
                                           are the authoritative source — YAML
                                           is a human-readable mirror)
  data/processed/qa/train.jsonl           (15,905 ChatML examples)
  data/processed/qa/val.jsonl             (935 ChatML examples)

Writes:
  checkpoints/phase_a_qa/state-<step>     (Tinker save_state for resume)
  checkpoints/phase_a_qa/sampler-final    (save_weights_for_sampler URL)
  outputs/phase_a_qa_metrics.jsonl        (one JSON line per training step)
  outputs/phase_a_qa_run_summary.json     (final summary, consumed by journal)

Guardrails (hard):
  $15 cost cap (placeholder $/M-token rate from ADR 0001)
  2-hour walltime cap
  Divergence: if smoothed train_nll > 1.5 * min-so-far for 50 consecutive steps

Cancel signals (SIGINT, SIGTERM): write final state + summary, exit 130.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from rich.console import Console

# Make sure local src is importable.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import tinker  # noqa: E402
from tinker import AdamParams  # noqa: E402
from tinker_cookbook.renderers import get_renderer  # noqa: E402
from tinker_cookbook.supervised.common import compute_mean_nll  # noqa: E402
from tinker_cookbook.supervised.data import FromConversationFileBuilder  # noqa: E402
from tinker_cookbook.supervised.types import (  # noqa: E402
    ChatDatasetBuilderCommonConfig,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa: E402

console = Console()


# =============================================================================
# Configuration (Python constants are authoritative; YAML is a mirror)
# =============================================================================


@dataclass
class TrainConfig:
    # model
    base_model: str = "openai/gpt-oss-20b"
    renderer_name: str = "gpt_oss_no_sysprompt"

    # LoRA — Tinker only exposes rank + which module groups to train
    lora_rank: int = 16
    lora_train_mlp: bool = True
    lora_train_attn: bool = True
    lora_train_unembed: bool = True

    # optimizer
    learning_rate: float = 1.0e-4
    warmup_steps: int = 100
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1.0e-8

    # data
    train_path: str = "data/processed/qa/train.jsonl"
    val_path: str = "data/processed/qa/val.jsonl"
    batch_size: int = 32
    max_seq_length: int = 512

    # training cadence
    num_epochs: int = 2
    eval_every_n_steps: int = 50
    save_every_n_steps: int = 100
    log_every_n_steps: int = 10

    # determinism
    seed: int = 42

    # guardrails (cost cap is a safety net, not a budget — project has
    # abundant Tinker credits; cap catches pathological cases only)
    hard_cost_cap_usd: float = 200.0
    hard_walltime_cap_seconds: int = 7200
    divergence_streak_threshold: int = 50
    divergence_ratio: float = 1.5
    placeholder_usd_per_million_tokens: float = 0.40

    # I/O
    log_path: str = "checkpoints/phase_a_qa"
    metrics_path: str = "outputs/phase_a_qa_metrics.jsonl"
    summary_path: str = "outputs/phase_a_qa_run_summary.json"
    wandb_project: str = "finetune-slm"
    wandb_name: str = "phase_a_qa_v1"


def load_config_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def merge_config(cli_args: argparse.Namespace, yaml_cfg: dict[str, Any]) -> TrainConfig:
    """YAML overrides defaults; CLI overrides YAML.

    --variant <name> namespaces all output paths so multiple variants of the
    same base config can run concurrently without clobbering each other.
    """
    cfg = TrainConfig()
    for key, value in yaml_cfg.items():
        if hasattr(cfg, key) and value is not None:
            setattr(cfg, key, value)

    # axis overrides (single-axis sweep support)
    if cli_args.lora_rank is not None:
        cfg.lora_rank = cli_args.lora_rank
    if cli_args.num_epochs is not None:
        cfg.num_epochs = cli_args.num_epochs
    if cli_args.learning_rate is not None:
        cfg.learning_rate = cli_args.learning_rate
    if cli_args.batch_size is not None:
        cfg.batch_size = cli_args.batch_size
    if cli_args.max_steps is not None:
        cfg._max_steps_override = cli_args.max_steps  # type: ignore[attr-defined]

    # variant namespacing — keeps multi-variant runs from clobbering paths
    if cli_args.variant:
        v = cli_args.variant
        cfg.log_path = f"checkpoints/phase_a_qa/{v}"
        cfg.metrics_path = f"outputs/{v}_metrics.jsonl"
        cfg.summary_path = f"outputs/{v}_run_summary.json"
        cfg.wandb_name = f"phase_a_qa_{v}"

    return cfg


# =============================================================================
# Run state — accumulated metrics and guardrail bookkeeping
# =============================================================================


@dataclass
class RunState:
    started_at: float = field(default_factory=time.monotonic)
    started_iso: str = ""

    # progress
    step: int = 0
    epoch: int = 0
    last_eval_step: int = -1
    last_save_step: int = -1
    final_state_path: str = ""
    final_sampler_path: str = ""

    # losses
    train_losses: list[float] = field(default_factory=list)
    smoothed_train_losses: list[float] = field(default_factory=list)
    val_losses: list[tuple[int, float]] = field(default_factory=list)  # (step, loss)

    # divergence detection
    smoothed_window: deque = field(default_factory=lambda: deque(maxlen=10))
    min_smoothed_loss: float = float("inf")
    divergence_streak: int = 0

    # accounting
    tokens_processed: int = 0
    loss_tokens_processed: int = 0
    estimated_cost_usd: float = 0.0

    # outcome
    abort_reason: str = ""
    exit_code: int = 0


# =============================================================================
# Utilities
# =============================================================================


def atomic_write(path: Path, content: str) -> None:
    """Write text atomically: write to .tmp then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    tmp.rename(path)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


def warmup_decay_lr(step: int, peak_lr: float, warmup: int, total_steps: int) -> float:
    """Linear warmup → linear decay to 0."""
    if step < warmup:
        return peak_lr * (step + 1) / max(1, warmup)
    if total_steps <= warmup:
        return peak_lr
    remaining = total_steps - warmup
    progress = (step - warmup) / remaining
    return peak_lr * max(0.0, 1.0 - progress)


def compute_cost(state: RunState, cfg: TrainConfig) -> float:
    return state.tokens_processed * cfg.placeholder_usd_per_million_tokens / 1_000_000


def check_guardrails(state: RunState, cfg: TrainConfig) -> str | None:
    """Return abort_reason string if any guardrail trips, else None."""
    elapsed = time.monotonic() - state.started_at
    if elapsed > cfg.hard_walltime_cap_seconds:
        return f"walltime cap hit ({elapsed:.0f}s > {cfg.hard_walltime_cap_seconds}s)"
    cost = compute_cost(state, cfg)
    if cost > cfg.hard_cost_cap_usd:
        return f"cost cap hit (${cost:.2f} > ${cfg.hard_cost_cap_usd})"
    if state.divergence_streak >= cfg.divergence_streak_threshold:
        return (
            f"divergence: smoothed loss above {cfg.divergence_ratio}× min "
            f"for {state.divergence_streak} consecutive steps"
        )
    return None


def update_divergence(state: RunState, train_nll: float, cfg: TrainConfig) -> None:
    state.smoothed_window.append(train_nll)
    smoothed = sum(state.smoothed_window) / len(state.smoothed_window)
    state.smoothed_train_losses.append(smoothed)
    if smoothed < state.min_smoothed_loss:
        state.min_smoothed_loss = smoothed
        state.divergence_streak = 0
        return
    if smoothed > state.min_smoothed_loss * cfg.divergence_ratio:
        state.divergence_streak += 1
    else:
        state.divergence_streak = 0


# =============================================================================
# Evaluation
# =============================================================================


async def evaluate(
    training_client: tinker.TrainingClient,
    val_dataset,
    max_batches: int | None = None,
) -> tuple[float, int]:
    """Run val_dataset through forward (no backward) and return (mean_nll, n_batches).

    We use forward_backward_async because TrainingClient.forward doesn't expose
    logprobs in this SDK version; we just don't apply optim_step.
    """
    total_weighted = 0.0
    total_weights = 0.0
    n_batches = min(len(val_dataset), max_batches) if max_batches else len(val_dataset)

    futures = []
    batches = []
    for i in range(n_batches):
        batch = val_dataset.get_batch(i)
        # forward only — no optim_step, so weights are not updated.
        f = await training_client.forward_backward_async(batch, loss_fn="cross_entropy")
        futures.append(f)
        batches.append(batch)

    for f, batch in zip(futures, batches):
        out = await f.result_async()
        logprobs = [x["logprobs"] for x in out.loss_fn_outputs]
        weights = [d.loss_fn_inputs["weights"] for d in batch]
        for lp, w in zip(logprobs, weights):
            lp_t = lp.to_torch()
            w_t = w.to_torch()
            total_weighted += float(lp_t.dot(w_t))
            total_weights += float(w_t.sum())

    if total_weights == 0:
        return float("nan"), n_batches
    return -total_weighted / total_weights, n_batches


# =============================================================================
# Checkpointing
# =============================================================================


async def save_state_atomic(
    training_client: tinker.TrainingClient,
    cfg: TrainConfig,
    state: RunState,
    label: str,
) -> str:
    """Save Tinker state and persist its checkpoint URL atomically."""
    fut = await training_client.save_state_async(label)
    ckpt = await fut.result_async()
    ckpt_url = ckpt.path
    out_dir = Path(cfg.log_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    pointer = out_dir / f"state-{label}.json"
    atomic_write(
        pointer,
        json.dumps(
            {
                "label": label,
                "step": state.step,
                "epoch": state.epoch,
                "tokens_processed": state.tokens_processed,
                "estimated_cost_usd": state.estimated_cost_usd,
                "checkpoint_url": ckpt_url,
            },
            indent=2,
        ),
    )
    return ckpt_url


async def save_sampler_weights(
    training_client: tinker.TrainingClient,
    cfg: TrainConfig,
    label: str,
) -> str:
    fut = await training_client.save_weights_for_sampler_async(label)
    ckpt = await fut.result_async()
    out_dir = Path(cfg.log_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    pointer = out_dir / f"sampler-{label}.json"
    atomic_write(
        pointer,
        json.dumps({"label": label, "checkpoint_url": ckpt.path}, indent=2),
    )
    return ckpt.path


# =============================================================================
# Pre-flight
# =============================================================================


def preflight(cfg: TrainConfig) -> None:
    load_dotenv()
    key = os.environ.get("TINKER_API_KEY") or ""
    if len(key) <= 30:
        raise SystemExit(
            "STOP: TINKER_API_KEY missing or too short (need length > 30). "
            "Set it in .env and re-run."
        )
    for path in (cfg.train_path, cfg.val_path):
        p = Path(path)
        if not p.exists() or p.stat().st_size == 0:
            raise SystemExit(f"STOP: data file missing or empty: {p}")
    Path(cfg.log_path).mkdir(parents=True, exist_ok=True)
    Path(cfg.metrics_path).parent.mkdir(parents=True, exist_ok=True)


# =============================================================================
# wandb (optional, graceful skip)
# =============================================================================


class _NullLogger:
    def log(self, *args, **kwargs):
        pass

    def finish(self, *args, **kwargs):
        pass


def init_wandb(cfg: TrainConfig):
    if not os.environ.get("WANDB_API_KEY"):
        return _NullLogger()
    try:
        import wandb

        run = wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_name,
            config=asdict(cfg),
        )

        class _Logger:
            def log(self, metrics, step=None):
                wandb.log(metrics, step=step)

            def finish(self):
                run.finish()

        return _Logger()
    except Exception as exc:
        console.print(f"[yellow]wandb init failed: {exc}; continuing without it[/yellow]")
        return _NullLogger()


# =============================================================================
# Main training loop
# =============================================================================


async def run_training(cfg: TrainConfig, args: argparse.Namespace) -> RunState:
    state = RunState(
        started_iso=time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
    )

    console.rule("Loading tokenizer + renderer")
    tokenizer = get_tokenizer(cfg.base_model)
    renderer = get_renderer(cfg.renderer_name, tokenizer, model_name=cfg.base_model)
    console.print(f"Tokenizer: {cfg.base_model}; renderer: {cfg.renderer_name}")

    console.rule("Loading data")
    common_cfg = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=cfg.base_model,
        renderer_name=cfg.renderer_name,
        max_length=cfg.max_seq_length,
        batch_size=cfg.batch_size,
        train_on_what="last_assistant_message",
    )
    train_builder = FromConversationFileBuilder(
        file_path=cfg.train_path,
        common_config=common_cfg,
    )
    val_builder = FromConversationFileBuilder(
        file_path=cfg.val_path,
        common_config=common_cfg,
    )
    train_dataset, _ = train_builder()
    val_dataset, _ = val_builder()
    n_train_batches = len(train_dataset)
    n_val_batches = len(val_dataset)
    total_steps = n_train_batches * cfg.num_epochs
    if hasattr(cfg, "_max_steps_override") and cfg._max_steps_override is not None:
        total_steps = min(total_steps, cfg._max_steps_override)
    console.print(
        f"train batches: {n_train_batches:,}, val batches: {n_val_batches:,}, "
        f"total planned steps: {total_steps:,}"
    )

    console.rule("Creating training client")
    service = tinker.ServiceClient()
    training_client = service.create_lora_training_client(
        base_model=cfg.base_model,
        rank=cfg.lora_rank,
        seed=cfg.seed,
        train_mlp=cfg.lora_train_mlp,
        train_attn=cfg.lora_train_attn,
        train_unembed=cfg.lora_train_unembed,
    )
    info = training_client.get_info()
    console.print(f"Training client up. Server info: {info}")

    # Resume?
    resume_state_url = None
    if args.resume:
        pointers = sorted(Path(cfg.log_path).glob("state-step-*.json"))
        if pointers:
            data = json.loads(pointers[-1].read_text())
            resume_state_url = data["checkpoint_url"]
            state.step = data["step"]
            state.epoch = data["epoch"]
            state.tokens_processed = data.get("tokens_processed", 0)
            console.print(
                f"[yellow]Resuming from {pointers[-1].name} at step {state.step}[/yellow]"
            )
    elif any(Path(cfg.log_path).glob("state-step-*.json")):
        raise SystemExit(
            f"STOP: existing checkpoints under {cfg.log_path}. "
            "Pass --resume to continue, or move/delete the directory for a fresh run."
        )

    if resume_state_url is not None:
        load_fut = await training_client.load_state_async(resume_state_url)
        await load_fut.result_async()

    logger = init_wandb(cfg)

    # Cancellation handling — store signal in shared mutable so we can detect it.
    cancellation = {"signum": None}

    def _on_signal(signum, _frame):
        if cancellation["signum"] is None:
            console.print(f"\n[red]signal {signum}; will abort after current step[/red]")
            cancellation["signum"] = signum

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # Run loop.
    console.rule(f"Training loop — up to {total_steps:,} steps")
    for epoch in range(state.epoch, cfg.num_epochs):
        train_dataset.set_epoch(seed=cfg.seed + epoch)
        for batch_idx in range(n_train_batches):
            if state.step >= total_steps:
                break

            # Per-step LR
            lr = warmup_decay_lr(
                state.step, cfg.learning_rate, cfg.warmup_steps, total_steps
            )
            adam = AdamParams(
                learning_rate=lr,
                beta1=cfg.adam_beta1,
                beta2=cfg.adam_beta2,
                eps=cfg.adam_eps,
                weight_decay=cfg.weight_decay,
                grad_clip_norm=cfg.gradient_clip,
            )

            batch = train_dataset.get_batch(batch_idx)

            t0 = time.monotonic()
            fwd_bwd_fut = await training_client.forward_backward_async(
                batch, loss_fn="cross_entropy"
            )
            optim_fut = await training_client.optim_step_async(adam)
            fwd_bwd_out = await fwd_bwd_fut.result_async()
            await optim_fut.result_async()
            step_seconds = time.monotonic() - t0

            # Loss
            logprobs = [x["logprobs"] for x in fwd_bwd_out.loss_fn_outputs]
            weights = [d.loss_fn_inputs["weights"] for d in batch]
            train_nll = compute_mean_nll(logprobs, weights)
            state.train_losses.append(train_nll)
            update_divergence(state, train_nll, cfg)

            # Token accounting
            batch_tokens = sum(d.model_input.length for d in batch)
            batch_loss_tokens = sum(sum(d.loss_fn_inputs["weights"].data) for d in batch)
            state.tokens_processed += batch_tokens
            state.loss_tokens_processed += int(batch_loss_tokens)
            state.estimated_cost_usd = compute_cost(state, cfg)

            metrics = {
                "step": state.step,
                "epoch": epoch,
                "batch_idx": batch_idx,
                "train_nll": train_nll,
                "smoothed_nll": state.smoothed_train_losses[-1],
                "lr": lr,
                "batch_tokens": batch_tokens,
                "batch_loss_tokens": batch_loss_tokens,
                "tokens_total": state.tokens_processed,
                "estimated_cost_usd": state.estimated_cost_usd,
                "step_seconds": round(step_seconds, 3),
            }
            append_jsonl(Path(cfg.metrics_path), metrics)
            logger.log(metrics, step=state.step)

            if state.step % cfg.log_every_n_steps == 0:
                console.print(
                    f"step {state.step:>5}  ep {epoch}  bi {batch_idx:>4}  "
                    f"lr {lr:.2e}  nll {train_nll:.4f}  "
                    f"smooth {state.smoothed_train_losses[-1]:.4f}  "
                    f"tok {state.tokens_processed:>10,}  "
                    f"$ {state.estimated_cost_usd:.4f}  "
                    f"t {step_seconds:.2f}s"
                )

            state.step += 1
            state.epoch = epoch

            # Periodic eval
            if (
                cfg.eval_every_n_steps > 0
                and state.step % cfg.eval_every_n_steps == 0
                and state.step != state.last_eval_step
            ):
                console.print(f"[cyan]eval @ step {state.step}…[/cyan]")
                val_nll, _ = await evaluate(training_client, val_dataset)
                state.val_losses.append((state.step, val_nll))
                state.last_eval_step = state.step
                logger.log({"val_nll": val_nll}, step=state.step)
                console.print(f"[cyan]val_nll @ step {state.step}: {val_nll:.4f}[/cyan]")

            # Periodic save
            if (
                cfg.save_every_n_steps > 0
                and state.step % cfg.save_every_n_steps == 0
                and state.step != state.last_save_step
            ):
                console.print(f"[blue]save state @ step {state.step}…[/blue]")
                await save_state_atomic(training_client, cfg, state, f"step-{state.step:05d}")
                state.last_save_step = state.step

            # Guardrails
            abort = check_guardrails(state, cfg)
            if abort:
                state.abort_reason = abort
                state.exit_code = 2
                console.print(f"[red]ABORT: {abort}[/red]")
                break
            if cancellation["signum"] is not None:
                state.abort_reason = f"signal {cancellation['signum']}"
                state.exit_code = 130
                break

        if state.abort_reason or cancellation["signum"] is not None:
            break
        if state.step >= total_steps:
            break

    console.rule("Final eval + checkpoint")
    # Final val pass
    try:
        val_nll, _ = await evaluate(training_client, val_dataset)
        state.val_losses.append((state.step, val_nll))
        console.print(f"final val_nll: {val_nll:.4f}")
    except Exception as exc:
        console.print(f"[yellow]final eval failed: {exc}[/yellow]")

    # Final state + sampler weights (always, even on abort)
    try:
        state.final_state_path = await save_state_atomic(
            training_client, cfg, state, f"final-step-{state.step:05d}"
        )
        state.final_sampler_path = await save_sampler_weights(
            training_client, cfg, f"final-step-{state.step:05d}"
        )
        console.print(f"final sampler path: {state.final_sampler_path}")
    except Exception as exc:
        console.print(f"[red]final save failed: {exc}[/red]")

    logger.finish()
    return state


def write_run_summary(cfg: TrainConfig, state: RunState) -> None:
    payload = {
        "config": asdict(cfg),
        "started_iso": state.started_iso,
        "ended_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "wall_seconds": time.monotonic() - state.started_at,
        "step": state.step,
        "epoch": state.epoch,
        "tokens_processed": state.tokens_processed,
        "loss_tokens_processed": state.loss_tokens_processed,
        "estimated_cost_usd": state.estimated_cost_usd,
        "min_smoothed_train_nll": (
            min(state.smoothed_train_losses) if state.smoothed_train_losses else None
        ),
        "final_train_nll": state.train_losses[-1] if state.train_losses else None,
        "final_smoothed_train_nll": (
            state.smoothed_train_losses[-1] if state.smoothed_train_losses else None
        ),
        "val_losses": state.val_losses,
        "final_val_nll": state.val_losses[-1][1] if state.val_losses else None,
        "abort_reason": state.abort_reason,
        "exit_code": state.exit_code,
        "final_state_path": state.final_state_path,
        "final_sampler_path": state.final_sampler_path,
    }
    atomic_write(Path(cfg.summary_path), json.dumps(payload, indent=2))
    console.print(f"summary written to {cfg.summary_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/tinker/phase_a_qa/run_v1.yaml",
        type=Path,
    )
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Run pre-flight only; do not create the training client.")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override total steps (smoke testing).")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override config batch_size.")
    parser.add_argument("--variant", default=None,
                        help="Variant name; namespaces all output paths "
                             "(checkpoints/phase_a_qa/<v>, outputs/<v>_*.json, "
                             "wandb_name=phase_a_qa_<v>).")
    parser.add_argument("--lora-rank", type=int, default=None,
                        help="Override LoRA rank.")
    parser.add_argument("--num-epochs", type=int, default=None,
                        help="Override number of epochs.")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Override peak learning rate.")
    args = parser.parse_args()

    yaml_cfg = load_config_yaml(args.config)
    cfg = merge_config(args, yaml_cfg)

    console.rule("Pre-flight")
    preflight(cfg)
    console.print("[green]Pre-flight passed.[/green]")

    if args.dry_run:
        console.print("[yellow]--dry-run set; not launching training.[/yellow]")
        return 0

    state = asyncio.run(run_training(cfg, args))
    write_run_summary(cfg, state)

    if state.exit_code:
        console.print(f"[red]EXIT {state.exit_code}: {state.abort_reason}[/red]")
        return state.exit_code
    console.print("[green]Training complete.[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
