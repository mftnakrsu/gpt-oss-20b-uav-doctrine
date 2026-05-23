"""Pre-flight Azure connection tests.

Three sanity checks before the Q&A pilot:

  1. LLM chat completion — round-trip on the gpt-5.4 deployment.
  2. Embeddings — call returns a 1536-dim vector.
  3. Embedding cosine similarity — semantic sanity: 'doctrine' query
     scores higher against a doctrine sentence than against a food
     sentence. (Replaces the originally-spec'd Cohere rerank test —
     see docs/journal/05a-qa-pilot.md for rationale.)

All three must PASS to proceed to the pilot. Never prints the API key.

Run::

    uv run python scripts/preflight_azure.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Load .env before importing the client modules so they pick up the values.
load_dotenv(PROJECT_ROOT / ".env")

from rich.console import Console  # noqa: E402

console = Console()

REQUIRED_ENV_VARS = (
    "AZURE_API_KEY",
    "AZURE_OPENAI_LLM_BASE_URL",
    "AZURE_OPENAI_LLM_DEPLOYMENT",
    "AZURE_OPENAI_LLM_API_VERSION",
    "AZURE_OPENAI_EMBEDDING_BASE_URL",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    "AZURE_RERANKER_BASE_URL",
    "AZURE_RERANKER_DEPLOYMENT",
    "AZURE_RERANKER_PATH",
)


def check_env_vars() -> bool:
    console.print("[bold]Env var presence check[/bold] (values not echoed)")
    all_set = True
    for k in REQUIRED_ENV_VARS:
        if os.environ.get(k):
            console.print(f"  {k}: [green]set[/green]")
        else:
            console.print(f"  {k}: [red]MISSING[/red]")
            all_set = False
    return all_set


async def test_llm() -> tuple[bool, str | None]:
    from finetune_slm.data.teacher import AzureLLMClient

    client = AzureLLMClient()
    try:
        content, usage = await client.chat_completion(
            messages=[{"role": "user", "content": "Reply with exactly the word OK."}],
            max_tokens=20,
        )
        ok = "ok" in content.strip().lower()
        console.print(f"  Response: {content.strip()!r}")
        console.print(f"  Model:    {client.deployment}")
        console.print(
            f"  Usage:    prompt={usage['prompt_tokens']}, "
            f"completion={usage['completion_tokens']}, total={usage['total_tokens']}"
        )
        return ok, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    finally:
        await client.close()


async def test_embedding_dim() -> tuple[bool, str | None]:
    from finetune_slm.data.embedder import AzureEmbeddingClient

    client = AzureEmbeddingClient()
    try:
        vecs = await client.embed(["hello"])
        dim = len(vecs[0]) if vecs else 0
        console.print(f"  Embedding length: {dim}")
        console.print(f"  Model:            {client.deployment}")
        ok = dim == client.expected_dim
        return ok, None if ok else f"expected dim {client.expected_dim}, got {dim}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    finally:
        await client.close()


async def test_embedding_similarity() -> tuple[bool, str | None]:
    """Semantic sanity: doctrine query closer to doctrine sentence than to pizza."""
    from finetune_slm.data.embedder import AzureEmbeddingClient, cosine_similarity

    client = AzureEmbeddingClient()
    try:
        texts = [
            "What is military doctrine?",
            "Doctrine is a body of fundamental military principles guiding operations.",
            "Pizza is a popular Italian food made with dough and toppings.",
        ]
        vecs = await client.embed(texts)
        q, doc_doctrine, doc_pizza = vecs
        sim_doctrine = cosine_similarity(q, doc_doctrine)
        sim_pizza = cosine_similarity(q, doc_pizza)
        console.print(f"  cos(query, doctrine sentence): {sim_doctrine:.4f}")
        console.print(f"  cos(query, pizza sentence):    {sim_pizza:.4f}")
        ok = sim_doctrine > sim_pizza
        return ok, None if ok else "doctrine sentence did not outrank pizza sentence"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    finally:
        await client.close()


async def main() -> int:
    if not check_env_vars():
        console.print("\n[bold red]STOP[/bold red] — required env vars missing.")
        return 1

    tests = [
        ("Test 1: GPT-5.4 chat completion", test_llm),
        ("Test 2: Embedding returns 1536-dim vector", test_embedding_dim),
        ("Test 3: Embedding semantic sanity (doctrine > pizza)", test_embedding_similarity),
    ]
    results: list[tuple[str, bool, str | None]] = []
    for name, fn in tests:
        console.print(f"\n[bold]{name}[/bold]")
        ok, err = await fn()
        color = "green" if ok else "red"
        label = "PASS" if ok else "FAIL"
        console.print(f"  Result: [{color}]{label}[/{color}]")
        if err:
            console.print(f"  Error:  {err}")
        results.append((name, ok, err))

    all_pass = all(ok for _, ok, _ in results)
    summary_color = "green" if all_pass else "red"
    summary_label = "PASS" if all_pass else "FAIL"
    console.print(f"\n[bold]Overall: [{summary_color}]{summary_label}[/{summary_color}][/bold]")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
