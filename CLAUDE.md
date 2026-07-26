# German Wiki — Project Context

## What this is
Personal German-learning wiki. Ingest images/text → extract concepts →
merge into a node graph → study via situational + morphological clusters.
Single user, single machine, runs locally. See SPEC.md for full design.

Full design spec: docs/SPEC.md
Architecture decisions: docs/decisions.md — append new ADRs here, don't re-litigate old ones.

## Non-negotiable design rules
- Nodes are Markdown + YAML frontmatter in /nodes, tracked in git. Files are the source of truth.
- SQLite (sqlite-vec) is a DERIVED index, always rebuildable from /nodes. Never authoritative.
- NOTHING auto-writes to /nodes. Every merge/classification goes through a review queue I approve.
- Raw extracted text is immutable in /raw. Canonical node body is a derived view.
- Merges must never add information absent from the sources. Flag uncertainty, don't invent.

## Stack (do not add to this without asking)
Python 3.11, Typer CLI, Pydantic v2, SQLite + sqlite-vec, LangGraph for the pipeline, pytest.
NO Docker, NO auth, NO cloud services, NO web frontend until I explicitly ask.
CLI-first. The review command is the only UI needed for the first several slices.

## Model / cost rules
- All model calls go through ONE swappable interface using the OpenAI-compatible client.
  Provider + model configured PER PIPELINE STEP in config, never hardcoded.
- Default provider: Z.AI (GLM). Fallback provider: DeepSeek. Both OpenAI-compatible — base URL swap only.
- Embeddings: ALWAYS local via sentence-transformers (multilingual-e5-small). Never an API.
- CEFR level anchors: rules + wordlists, no LLM.
- Cache EVERY model call by content hash on disk. Never re-call on identical input.
  This is the primary cost control — tuning re-runs must be free.
- Structure prompts with fixed content FIRST (system prompt, schema, few-shot),
  variable content LAST, to maximize provider cache hits.
- Log token counts + estimated cost per call to /logs. Surface a running total.

## Git
- Do NOT run git commands (commit, add, push, checkout, branch, reset, etc.).
  Stage and commit are mine. When a checkpoint is ready, tell me what to commit
  and suggest a message; I run the git commands myself.

## How I work
- Build in vertical slices. One feature working end-to-end before starting the next.
- SHOW ME THE PLAN before writing code for any new module. Wait for my approval.
- Small, reviewable commits with clear messages.
- Explain tradeoffs and let me decide; don't silently pick.
- Write pytest tests alongside code, not after.