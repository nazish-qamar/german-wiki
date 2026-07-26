# Architecture Decisions

Append-only log. One entry per decision that would otherwise get re-litigated.
Format: what was decided, why, and what was rejected.

---

## ADR-001 — Markdown files as source of truth, SQLite as derived index

**Decided:** Nodes live as Markdown + YAML frontmatter in `/nodes`, tracked in git.
SQLite (with sqlite-vec) is a rebuildable index, never authoritative.

**Why:** Merges are the risky operation in this system. Git gives free diffs, revert,
and history for every merge — the single best safeguard against merge drift and
hallucinated content. Files also stay readable in Obsidian and outlive the app.

**Rejected:** SQLite/Postgres as primary store — faster queries, but no diff, no revert,
and the notes become hostage to the app.

---

## ADR-002 — Standalone app, not an Obsidian plugin

**Decided:** FastAPI/CLI app with its own storage layer; Markdown files remain
Obsidian-readable but Obsidian is not the platform.

**Why:** The custom grid views, dual-axis clustering, and human-in-loop review queue
push past what the Obsidian plugin API does comfortably. The LangGraph pipeline has to
run outside Obsidian regardless.

**Rejected:** Obsidian plugin (free mobile + editor + graph view, but constrained UI).
Mitigated by keeping files as plain Markdown so the vault can still be opened in
Obsidian for graph view.

---

## ADR-003 — Nothing auto-writes to /nodes

**Decided:** Every merge and classification goes through a review queue requiring
explicit approval. LangGraph `interrupt()` at the adjudication node.

**Why:** The system's value is entirely in judgment calls — what's a duplicate, what
merges, what level something is. Those must be watched and tuned. An unreviewed
pipeline silently corrupts study notes, which is worse than no notes.

**Revisit when:** merge acceptance rate is consistently >95% over several hundred
items. Then consider auto-accepting above a confidence threshold — still never SAME/
OVERLAP deletions.

---

## ADR-004 — Embeddings always local, generation via cheap API

**Decided:** sentence-transformers (multilingual-e5-small) locally for all embeddings.
Generation (extraction, adjudication, vision) via OpenAI-compatible cheap APIs,
default Z.AI/GLM with DeepSeek as fallback.

**Why:** Embeddings are the highest-volume call and trivially solved offline — zero
cost, no rate limit, works offline. Generation quality matters more and cheap APIs
cost ~$1–3 lifetime at this volume, which is below the effort threshold of self-hosting.

**Rejected:** Local generation via Ollama — free, but costs VRAM, setup time, and
slower tuning iteration for savings measured in single dollars.

---

## ADR-005 — Cache every model call by content hash

**Decided:** Disk cache keyed on content hash, wrapping every model call. Built in
slice 2, before any tuning begins.

**Why:** The tuning phase re-runs the pipeline on the same sources dozens of times.
Uncached, that multiplies token spend by ~50×. This single mechanism, not model
selection, is what keeps the project inside budget.

---

## ADR-006 — Aggressive merge bias

**Decided:** Tune the OVERLAP threshold low, so more candidates merge into existing
nodes rather than spawning new ones.

**Why:** The failure mode for a study wiki is fragmentation — 15,000 disconnected
atoms with no learning structure. Dense nodes teach the connections German rewards.
Node count is a dial, and the right end of it is "fewer, richer."

**Guardrail:** max 5–8 candidate nodes extracted per source. Exceeding it signals the
extractor is atomizing rather than conceptualizing.

---

## ADR-007 — Register/theme tags are open, normalized vocabularies (not enums)

**Decided:** The two clustering axes — `register` and `themes` (SPEC §6) — are open
`list[str]`, not enums. Every tag value passes through one normalizer (strip →
lowercase → alias-map from `vocab/aliases.yaml`) before it is stored. The canonical
known-sets live in git-tracked text files, one tag per line: `vocab/registers.txt` and
`vocab/themes.txt`. Aliases fold spelling/language variants onto a canonical tag and are
human-curated — the pipeline reads `aliases.yaml` but never writes it.

By contrast, `type` / `cefr` / `status` / `family_transparency` stay strict enums: small,
stable, syllabus-fixed vocabularies where a typo should fail loudly.

**Learning happens only at the ingest boundary, not on every write.** Normalization
(strip/lower/alias) runs on every write, but *appending an unknown value to the known-set*
is opt-in via `learn=True`, passed only by the reviewed pipeline as it admits genuinely
new material. An ordinary re-save or a merge round-trip (`write_node` default `learn=False`)
normalizes tags but never grows the vocabulary — so the known-sets only ever expand through
the same human-approved gate as `/nodes` writes (ADR-003), never as a silent side effect of
touching an existing node. When learning does fire, the new tag is appended (append-only, no
re-sort) and a warning is logged to `logs/gw.log`; it never raises.

**Why:** These axes are *discovered from material* and unbounded by design — new domains
(`arzt`, `amt`, `café`, …) appear as ingestion proceeds, and §6.2 expands register into
several sub-dimensions later. A closed enum would reject real data and force constant model
edits. Normalizing-and-warning keeps tags consistent (no `Küche`/`kueche`/` kitchen ` drift)
while staying open. Gating *growth* of the vocabulary to the ingest boundary keeps the
known-sets an auditable, reviewed artifact rather than accumulating typos from every write.

**Rejected:**
- *Enum-validating register/themes* — catches typos but fights the "discover from material"
  goal and needs a code change for every new tag.
- *Free-text tags with no normalization* — no store to append to, but `Küche` vs `küche` vs
  `kitchen` fragments the very clusters the axes exist to create.
- *Learning on every write* — simpler, but any round-trip of a node with a stray tag would
  silently enshrine it in the vocabulary, defeating the audit trail.

---

## ADR-008 — Model layer: cache key, step gating, and cost accounting

**Decided:** Slice 2's model layer (`german_wiki.llm`) resolves provider+model per pipeline
step from `config/models.yaml`, wraps every call in a content-hash disk cache, and appends
one JSONL record per call. Six decisions inside that are worth not re-litigating:

**1. The cache key excludes the pipeline step.** It covers the request as the provider sees
it — provider, model, messages, temperature, max_tokens, response_format, seed — plus a
`prompt_version` escape hatch and a key-schema version `v`. Two steps issuing a
byte-identical request *should* share an entry (ADR-005 says never re-call on identical
input), and renaming a step must not cost money. The step is stored in the entry for
debugging, just not hashed. Credentials, base URL, timeouts and retry counts are excluded
for the same reason: they don't change the response.

**2. Steps carry `status: active | planned`.** The config documents the whole SPEC §9
routing plan, but only `active` steps resolve — routing to a `planned` one raises. This
lets a later slice's config sit in the repo, reviewed and visible, without being reachable.
`status` defaults to `planned`, so a step becomes callable only by opting in.

**3. `kind: api | local` on providers is ADR-004's enforcement mechanism.** `embeddings`
is `provider: local` from day one, and `complete()` refuses any step whose provider kind
isn't `api`. That check lives at the call site, not in resolution, and is independent of
`status` — so activating embeddings in slice 4 still cannot turn it into an API call.
Tested against an *active* local step, since a planned one would trip the earlier gate and
leave the guard unexercised.

**4. Unpriced models are never estimated.** Only known-free models carry a rate
(`glm-4.5-flash` at explicit zeros, so "known free" stays distinguishable from "unknown").
A model absent from the table logs its token counts and `cost_usd: null`, plus one warning
per `(provider, model)`. A hardcoded per-1M figure goes stale silently and corrupts the
running total; an honest gap does not. Rates are config, so enabling a paid model is a YAML
edit.

**5. The running total is derived by summing `logs/llm_usage.jsonl`, which is git-tracked.**
A persisted counter would be a second source of truth that drifts and needs a repair path,
and it could not live in `data/index.db` — `rebuild_schema` drops every table on each
`gw reindex`, silently zeroing it. The ledger is tracked because spend history is a durable
record; it is append-only, so diffs are pure additions. `logs/gw.log` stays ignored, which
is why `.gitignore` uses `logs/*` plus a negation rather than `logs/`.

**6. No automatic provider failover yet.** Swapping is manual: a config edit, `provider=`,
`fallback=True`, or `GW_LLM_PROVIDER`. Auto-failover raises questions — does the failover
call share a cache key, does it double-log, what counts as fatal — better answered against
a real failure than guessed at now.

**Also:** `llm/__init__.py` is the only import surface; internals are `_`-prefixed and the
rest of the app, `cli.py` included, imports only the public names. A test asserts `__all__`
and greps for boundary violations.

**Rejected:**
- *Hashing the step into the cache key* — would make a rename cost real money for no gain.
- *A persisted cost counter* — drift plus a repair path, for a sum that takes milliseconds.
- *Guessing prices for paid models* — a wrong rate is worse than a visible gap.
- *Enforcing "embeddings are local" by convention* — ADR-004 deserves a check that fires.

---

## ADR-009 — Ingestion stages into `/queue`; `gw promote` is the only writer to `/nodes`

**Decided:** `gw ingest` writes complete node files to `queue/<source_id>/<node_id>.md` and
never touches `/nodes`. `gw promote <source_id>` is the sole path into `/nodes`. Manual
review sits between them this slice; rejecting a candidate is deleting its queue file.

**Why:** SPEC §11 calls slice 3 "new nodes", but ADR-003 and CLAUDE.md say nothing
auto-writes to `/nodes`, and the review CLI is slice 5. Extraction assigns `type`, `cefr`,
`register` and `themes` — precisely the classifications ADR-003 gates. The queue satisfies
both: material flows end-to-end now, and the approval gate exists two slices before the
machinery that will automate it. Slice 5 wraps *this seam* with LangGraph `interrupt()`;
it does not replace it.

**Queued files are real nodes, not an intermediate format.** Each loads through
`storage.load_node` unchanged, so review needs nothing but an editor, and promote gets
validation for free.

**`/queue` is gitignored.** Candidates live there minutes to hours between ingest and
promote-or-reject. Tracking them would make every ingest a batch of adds and every promote a
batch of deletes — churn that records no meaningful history, since a rejected candidate is
one you decided *not* to keep. The promoted nodes in `/nodes` are the diffable record, which
is the whole point of the gate; `/raw` is the durable provenance behind them.

**Promote is not a file move.** It loads (validating any hand-edit made during review),
writes via `write_node(..., learn=True)`, unlinks the queue entry on success, then
reindexes. That `learn=True` is **the only one in the codebase** — ADR-007 says the tag
known-sets grow through the same human-approved gate as `/nodes` writes, and this is that
gate. Ingestion stages with `learn=False`. An AST-based test asserts the uniqueness, so a
second learner cannot appear unnoticed.

**Nothing is ever overwritten.** A node id colliding with `/nodes` or `/queue` gets a
numeric suffix at ingest; at promote, an id that already exists in `/nodes` is refused and
left queued. Refusals are per-file, so one bad candidate never blocks the rest. There is no
dedup at all in this slice (SPEC §11) — two sources describing the same concept produce two
nodes, and slice 4 is what detects that.

**Source ids are `<YYYYMMDD>-<slug>-<content hash prefix>`.** Hashing the *content*, not the
filename, makes re-ingesting the same text detectable. `/raw` holds two files per source:
`<id>.txt` byte-verbatim and immutable (written *before* extraction, so provenance never
depends on the model succeeding) and `<id>.json` for metadata. The sidecar's
**`content_sha256` is stored full-length** — the filename's short prefix is a human handle,
this is slice 5's tier-1 exact-duplicate key (SPEC §3.1), and backfilling it later would
mean re-reading every raw file. A short-prefix collision widens the prefix rather than
overwriting, compared against stored bytes so it works even when a prior ingest left no
sidecar.

**`finish_reason == "length"` is a failed extraction, not an empty result.** GLM-4.5 spends
completion tokens on reasoning before emitting content, so a tight cap returns well-formed
*empty* output. `complete()` is parse-free by design, so extraction owns the check; the
error carries `reasoning_content` so the truncation is inspectable. That field is
capture-only: never parsed, never in the cache key, never written to `/nodes` or `/raw` —
but it *is* in the cache payload, because a truncated call is itself cached and a re-run
would otherwise report the failure with an empty context.

**Every machine-assigned CEFR level is marked `cefr_basis: llm:extraction…`.** `Node.cefr`
is required and SPEC §5 says zero-shot LLM CEFR is unreliable, so the level is provisional
by construction. The marker makes `grep` find every one when slice 6 lands the wordlist and
grammar anchors.

**Rejected:**
- *Writing straight to `/nodes` with git revert as the safety net* — fast, but sets aside a
  rule CLAUDE.md marks non-negotiable for two slices.
- *A summary-only preview with a `--write` flag* — approving a table is not approving the
  content; the queue lets you read the actual file.
- *`shutil.move` on promote* — skips validation, and leaves ADR-007's vocabulary growth with
  no gate to fire at.
- *Failing an extraction that exceeds the 5–8 cap* — ADR-006 treats the overage as a signal
  the extractor is atomizing; warn and keep the first 8 rather than discard the source.