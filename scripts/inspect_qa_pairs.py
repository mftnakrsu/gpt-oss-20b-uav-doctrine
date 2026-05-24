"""Phase 1 inspection of data/interim/qa_pairs.jsonl.

Read-only. Reports counts, schema completeness, pair_id uniqueness,
and held-out source presence. STOPs (exits 1) if a hard rule is violated.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

QA_PATH = Path("data/interim/qa_pairs.jsonl")
HELD_OUT_PREFIXES = ("crs_2026_law_enforcement_cuas", "faa_notice_uas_lost_link")
REQUIRED_FIELDS = ("pair_id", "chunk_id", "source_filename", "question", "answer", "question_type")


def source_folder(source_filename: str) -> str:
    return source_filename.split("/", 1)[0]


def held_out_match(source_filename: str) -> str | None:
    folder = source_folder(source_filename)
    for prefix in HELD_OUT_PREFIXES:
        if folder == prefix or folder.startswith(prefix + "_"):
            return prefix
    return None


def main() -> int:
    if not QA_PATH.exists():
        print(f"STOP: {QA_PATH} not found", file=sys.stderr)
        return 1

    pairs: list[dict] = []
    with QA_PATH.open() as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                pairs.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"STOP: invalid JSON on line {line_num}: {exc}", file=sys.stderr)
                return 1

    print(f"Total pairs loaded: {len(pairs)}")
    if len(pairs) != 18730:
        print(f"WARN: expected 18730 pairs, got {len(pairs)}")

    # Group by source_filename
    by_source = Counter(p["source_filename"] for p in pairs)
    print("\nPairs per source (descending):")
    width = max(len(s) for s in by_source)
    for src, count in by_source.most_common():
        print(f"  {src:<{width}}  {count:>6}")

    # Held-out check (prefix match on first path segment)
    print("\nHeld-out source candidates (prefix match):")
    held_out_counts: dict[str, int] = {prefix: 0 for prefix in HELD_OUT_PREFIXES}
    held_out_filenames: dict[str, set[str]] = {prefix: set() for prefix in HELD_OUT_PREFIXES}
    for pair in pairs:
        match = held_out_match(pair["source_filename"])
        if match is not None:
            held_out_counts[match] += 1
            held_out_filenames[match].add(pair["source_filename"])

    for prefix, count in held_out_counts.items():
        files = sorted(held_out_filenames[prefix])
        print(f"  {prefix}: {count} pairs  (files: {files})")
        if count == 0:
            print(f"STOP: held-out source '{prefix}' has zero pairs", file=sys.stderr)
            return 1

    # pair_id uniqueness
    ids = [p["pair_id"] for p in pairs]
    dup_counts = Counter(ids)
    dups = [pid for pid, n in dup_counts.items() if n > 1]
    if dups:
        print(f"STOP: {len(dups)} duplicate pair_ids; first 5: {dups[:5]}", file=sys.stderr)
        return 1
    print(f"\npair_id uniqueness: OK ({len(ids)} unique)")

    # Schema completeness sample (3 random)
    rng = random.Random(0)
    sample = rng.sample(pairs, 3)
    print("\nRandom schema spot-check (3 pairs):")
    for pair in sample:
        missing = [f for f in REQUIRED_FIELDS if not pair.get(f)]
        status = "OK" if not missing else f"MISSING={missing}"
        print(f"  {pair['pair_id']}  {status}")

    # Full schema sweep
    bad = []
    for pair in pairs:
        missing = [f for f in REQUIRED_FIELDS if not pair.get(f)]
        if missing:
            bad.append((pair.get("pair_id", "<no-id>"), missing))
    if bad:
        print(f"\nSTOP: {len(bad)} pairs have empty required fields; first 5: {bad[:5]}", file=sys.stderr)
        return 1
    print("\nAll pairs pass schema completeness check.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
