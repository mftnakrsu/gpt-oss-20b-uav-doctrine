"""Discovery + paired-manifest loading for the doctrine corpus.

The current corpus layout is one source directory per document, each
containing exactly one PDF and one ``manifest.json``::

    data/raw/<source_id>/<doc>.pdf
    data/raw/<source_id>/manifest.json

We also accept the slightly more general ``<doc>.manifest.json`` naming
that other ingest pipelines emit, since the second CLI session may
adopt that convention.

PDFs are the training content. Manifests are provenance metadata —
attached to every chunk that comes from the paired PDF, but never
tokenized as training tokens. ``discover_training_pdfs`` therefore
only returns ``.pdf`` paths.
"""

from __future__ import annotations

import json
from pathlib import Path


def discover_training_pdfs(raw_dir: Path) -> list[Path]:
    """Walk ``raw_dir`` recursively and return all ``.pdf`` paths,
    deterministically sorted.

    The walk is recursive so a sibling ``data/raw/expanded/`` directory
    (or any future sub-tree) is picked up without any caller changes.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        return []
    return sorted(p for p in raw_dir.rglob("*.pdf") if p.is_file())


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_paired_manifest(pdf_path: Path) -> dict:
    """Find and load the manifest paired with ``pdf_path``.

    Search order, all in the PDF's parent directory:
      1. ``manifest.json``
      2. ``<pdf_stem>.manifest.json``
      3. any ``*.manifest.json`` (first match, deterministic via sort)

    Returns ``{}`` when no manifest is found — never raises.
    """
    pdf_path = Path(pdf_path)
    parent = pdf_path.parent

    # 1. canonical
    candidate = parent / "manifest.json"
    if candidate.exists():
        return _read_json(candidate)

    # 2. stem-prefixed
    candidate = parent / f"{pdf_path.stem}.manifest.json"
    if candidate.exists():
        return _read_json(candidate)

    # 3. any *.manifest.json sibling
    for candidate in sorted(parent.glob("*.manifest.json")):
        return _read_json(candidate)

    return {}


def list_non_training_files(raw_dir: Path) -> list[Path]:
    """Return any files under ``raw_dir`` that are neither a PDF nor a
    recognised manifest. Used to populate the journal's "skipped files"
    audit so we never silently drop content.
    """
    raw_dir = Path(raw_dir)
    out: list[Path] = []
    for p in sorted(raw_dir.rglob("*")):
        if not p.is_file():
            continue
        name = p.name.lower()
        if p.suffix.lower() == ".pdf":
            continue
        if name == "manifest.json" or name.endswith(".manifest.json"):
            continue
        out.append(p)
    return out
