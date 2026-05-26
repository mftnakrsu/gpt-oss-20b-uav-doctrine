# GitHub public release

_Hand-written. This turn turned the local/private repo into a polished public
GitHub repository and is the public-release milestone for Phase A. Date: 2026-05-26._

## Secret audit results

**Clean — no secrets in git history or the working tree.** Scans (values were
never printed, only presence/locations):

- Azure key prefix `BGzb8ir5`: **0 commits**.
- Value-format tokens (`hf_…`, `sk-…`, `AKIA…`, `ghp_…`, `github_pat_…`) across
  all commits (per-commit `git grep -Il`): **none**.
- A real `.env` was **never committed**; only `.env.example` (placeholders only)
  and `deploy/hf_space/.env.example` are tracked.
- `.env` is in `.gitignore`. Working-tree value scan: clean.

No history rewrite was needed for secrets.

## Repo cleanup

- **`.gitignore`** — replaced the blanket `data/processed/` ignore with
  `data/processed/*` + a re-include of `data/processed/qa/` (git can't re-include
  a child of an excluded dir, so the parent is globbed instead). Added
  `.env.local`, `mlruns/`, `**/.archive_url`, `feedback_*.md`. Committed with the
  final Q&A splits (`data/processed/qa/*.jsonl`, 18,730 pairs / ~16 MB, regular
  git — not LFS).
- **Git LFS** holds **PDFs only** (58 → **50** after the copyrighted-PDF
  exclusion below). The 895 MB v4 adapter is **not** in the repo (it lives on
  HuggingFace; `checkpoints/` is git-ignored).

## README rewrite summary

Full public-facing overhaul (~300 insertions): shields badges, "Try it now"
(live demo), "What this is", a results table, a **Mermaid** pipeline diagram
(renders natively on GitHub), use- and reproduce-quickstarts, project structure,
the four methodology findings, an honest-limitations section, reproducibility,
a cost table, citation, and acknowledgments. Every figure is journal-sourced.

**Charts:** `scripts/make_charts.py` (committed, reproducible; matplotlib added
to the dev extra) generates five PNGs in `docs/images/` from the journal-verified
numbers — win rates, per-axis scores, validation NLL curve, answer length
(v4 ~49 vs baseline ~297 tokens), and cost by phase — embedded in the README.

## examples/ directory

16 verbatim v4-vs-baseline comparison files (4 per question type) + an index
`README.md`, sourced from `docs/journal/09` and the generation JSONL in
`outputs/phase_a_eval/`. Distribution: **12 v4 wins / 4 ties / 0 baseline wins**
(honest — there are no baseline wins in the journal's formatted samples). Nothing
fabricated.

## docs reorganization

- `docs/README.md` (new) — index of the two ADRs + a 14-row journal table.
- `scripts/README.md` (new) — run-order index of the pipeline scripts + utilities.

## LICENSE

Switched **MIT → Apache 2.0** (canonical text, "Copyright 2026 Meftun Akarsu"),
per the project decision and to match the HuggingFace model/Space and the
`gpt-oss-20b` base model. `NOTICE` updated: corpus count corrected to 58, and the
copyrighted expansion analyses are flagged as **not redistributed** (manifest +
source URL only). The redistributed doctrine PDFs retain their per-source terms.

## GitHub repo

- **Renamed** `mftnakrsu/uav-doctrine-slm` → **`mftnakrsu/gpt-oss-20b-uav-doctrine`**
  (so GitHub + HF model + HF Space share one name; the old URL auto-redirects).
  In-code repo links (`scripts/ui_app.py`, `deploy/hf_space/app.py`, the Space
  `README.md`) were updated and the **HF Space was re-deployed**.
- **Copyrighted-PDF exclusion + history purge.** 8 third-party analysis PDFs
  (3 ICRC, 1 CNAS, 3 RUSI, 1 LIIA) were `git rm`'d from the tree (manifests kept),
  then **purged from all history with `git filter-repo`** and force-pushed
  (`b4345c6`). Verified: those PDFs appear in **no** commit; 50 PDFs + all 58
  manifests remain; `v0.1.0` tag preserved.
- **Tag** `v0.1.0` (annotated). **Topics** + homepage (the HF Space) set.
- **Visibility: made public** — `github.com/mftnakrsu/gpt-oss-20b-uav-doctrine`.

## Cost note

No new model/API spend this turn — packaging only. (Project total remains
~$55–62, dominated by Azure GPT-5.4; see the README cost table and journals
05b/09/10. Tinker LoRA cost is an ADR-0001 placeholder.)

## Anything weird

- **`git push` returned HTTP 400 / "unexpected disconnect reading sideband
  packet"** on the first attempt, while the 166 MB LFS push succeeded — i.e. not
  a bandwidth issue. Fixed with `git config http.postBuffer 524288000` +
  `http.version HTTP/1.1` (the smart-HTTP RPC / HTTP-2 sideband issue).
- **`git filter-repo` rewrote every commit hash from the corpus-expansion commit
  onward.** Git commit hashes cited in earlier journals/reports are now stale;
  the narrative is unaffected. (HuggingFace commit hashes are unrelated and
  unchanged.)
- **The repo already existed as a private `uav-doctrine-slm`** (pushed earlier by
  the parallel Gradio-UI session). This turn consolidated naming, cleaned, and
  published it — rather than creating from scratch as the prompt assumed.
- **Orphaned LFS objects.** The 8 removed PDFs' LFS blobs are now unreferenced on
  the remote (GitHub does not serve unreferenced LFS objects to clones and GCs
  them over time); pruned locally with `git lfs prune`.
- A stray `qwen` repo topic (left from an earlier project state) was removed.
