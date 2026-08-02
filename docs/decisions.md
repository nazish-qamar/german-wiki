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

---

## ADR-010 — Duplicate detection: exact Jaccard, cached vectors, report-only

**Decided:** Slice 4 detects duplicates across three tiers and reports them. It writes
nothing to `/nodes` or `/queue`; acting on a finding is slice 5.

**Tier 1 is exact, NOT MinHash — a deliberate departure from SPEC §3.1's wording.**
Exact duplicates are `sha256` of normalized text; near-exact is *true* Jaccard over 5-character
shingles, with a length-ratio prefilter (`J(A,B) ≤ min/max`, so a size mismatch rules a pair
out without touching the sets). MinHash exists to approximate Jaccard sublinearly at a scale
SPEC §3.3 explicitly says this corpus will not reach — "low tens of thousands". At that size
the real similarity is directly affordable, and computing it avoids LSH band/row tuning and
approximation error in both directions.

**This is recorded so the MinHash reference is not later "restored" as a perceived gap.**
SPEC §3.1 still says MinHash; that line is over-specified for the actual scale. If the corpus
ever grows past where the prefilter keeps the pair sweep comfortable, revisit — the prefilter
is the seam an LSH stage would slot into.

**Embeddings are cached on disk, keyed by (model, exact model input)** — ADR-005's principle
applied to a local model. Local embeddings cost no money but cost time, and a full re-encode is
the difference between `gw embed` being instant and being a coffee break. **The model name is
in the key**, so two embedding models never collide and can be A/B'd against one corpus. The
key covers the *exact* string sent to the model, `query: ` prefix included.

**Cache and vector table are deliberately distinct.** The cache is recompute-avoidance and
survives `gw reindex`; the sqlite-vec table is query structure and is dropped by it. That split
is precisely what lets `gw reindex` repopulate vectors **without loading a model** — the
run-freely property ADR-001 depends on. A fresh clone must not download 470MB to run `gw list`.

**The governing principle, since it came up twice in one slice:** the cache key must contain
everything that affects the vector. Two consequences fall out of that, and both were hit here:

1. Changing *what* is embedded (`embed_text`) changes the key, so every cached vector is
   orphaned. Happened during calibration; cost was one `gw cache clear --kind embeddings`.
2. Changing *how* the vector is transformed after encoding has the same effect — which is why
   mean-centering is expensive. The mean is **corpus-global**, so it would have to enter the
   key, and then *every node addition* orphans the entire cache.

The general rule: a transform depending only on a single node's own text keeps the key stable
and the cache useful; a transform depending on the corpus makes the key corpus-dependent and
the cache nearly worthless. **Prefer stateless-per-node transforms.** Worth checking against
any future candidate — TF-IDF weighting, whitening, dimensionality reduction fitted on the
corpus, and centroid subtraction all fall on the expensive side of that line.

- `gw reindex` — rebuilds scalar tables, reloads cached vectors, **never computes**.
- `gw embed` — explicit and heavy; reports "N new, M from cache", which is the embedding-layer
  equivalent of slice 2's `call_count == 1` proof.
- `gw dupes` — lazily embeds what is missing, so it never fails on a cold cache.

**`EMBEDDING_DIM` is pinned in `db.py`, not in `models.yaml`.** A vec0 column needs its width
at CREATE time, and `StepSettings`/`ResolvedStep` are both `extra="forbid"` and describe routing,
not storage layout. `embed/_model.py` asserts the loaded model reports that width **at load,
before any encode**, so a mismatched model can never produce a vector that reaches the column,
where sqlite-vec might reject it loudly or accept it quietly.

**The `query: ` prefix is mandatory.** multilingual-e5 requires `query: `/`passage: `, and its
own guidance is `query: ` on both sides for symmetric similarity — which node-to-node comparison
is. It is applied in `embed_text` rather than left to call sites.

**Two normalizations, deliberately not shared.** Tier 1 lowercases aggressively because it asks
"is this the same text?". Embedding preserves case because German capitalizes nouns and the
model uses that signal.

**The embedded text is the German title plus a de-scaffolded body — measured, not guessed.**
SPEC §3.1's thresholds (0.75–0.92) do not survive contact with multilingual-e5, whose cosine
scores compress into a narrow high band. The first implementation embedded
`"{title_de} — {title_en}\n\n{body_md}"` and flagged **all six pairs of four unrelated seed
nodes** at 0.86–0.90 — a 100% false-positive rate, which would have handed slice 5 an
adjudication queue containing everything: exactly what §3.1's cheap tiers exist to prevent.

Diagnosis: the margin between the weakest true duplicate (0.9137) and the strongest unrelated
pair (0.8980) was 0.016, because the string was dominated by structure every node shares.
Eight variants were measured against the seed corpus plus a real ingested pair
(`um-hilfe-bitten` vs the queued `hoefliche-bitten-im-buero`), scoring
`min(related) − max(unrelated)`:

| variant | margin |
|---|---|
| German + English title, full raw body (original) | +0.0378 |
| German title only, full raw body | +0.0298 |
| German + English title, stripped body | +0.0480 |
| **German title only, stripped body** | **+0.0559** |
| German title only, stripped and truncated | +0.018 … +0.037 |
| German title alone | +0.0175 |

Two counter-intuitive results worth not rediscovering. Stripping Markdown — the `## Examples`
heading, table pipes, `[alltag]` register tags — nearly doubles the margin, because otherwise
the model spends capacity on scaffolding every node carries. And **appending the English title
hurts**: the German-English pairing is itself a shared structure that pulls unrelated nodes
together. Truncating hurts too; the body carries real signal, so all of it is kept.

Consequence: the vector no longer contains `title_en`, so an English query matches less well.
Correct here (dedup compares German node to German node); slice 9's RAG chat may want its own
text builder.

**Thresholds are calibrated to that measurement, not to SPEC's numbers**: `GRAY_LOW = 0.87`
(above the 0.8635 unrelated ceiling), `GRAY_HIGH = 0.95` (above the 0.9194 weakest true
duplicate, so genuine paraphrases reach the LLM rather than being decided by a constant), and
`NEAR_EXACT_JACCARD = 0.85`. SPEC §3.3 calls the threshold the node-count dial; this is where
that dial lives, in one edit. After the fix the four seeds report **zero** pairs, and the real
ingested source reports three defensible ones with the true duplicate ranked top.

The test asserts the *relationship* — `GRAY_LOW` above the measured unrelated ceiling and below
the weakest duplicate — rather than a literal pair, so recalibrating on more material updates a
comment instead of breaking a test. **Nine unrelated pairs is thin evidence; revisit once real
material accumulates.** Changing `embed_text` changes the cache key, so stale vectors are
simply never read again; `gw cache clear --kind embeddings` reclaims the space.

**Report-only means two different layers.** `/nodes`, `/queue` and the vocab files are
byte-identical after a run. `data/index.db` is **not** — detection embeds lazily and the derived
index gains vectors, which ADR-001 makes freely rebuildable and `.gitignore` already excludes.
The test asserts both directions, because asserting the DB is unchanged would be wrong.

**Rejected:**
- *MinHash + LSH* — approximation and tuning for a scale this project will not reach.
- *A `dimension:` key in `models.yaml`* — the width belongs with the DDL that consumes it.
- *Recomputing embeddings on every reindex* — makes a structural command pay a model cost.
- *Reusing the `live` pytest marker* — `live` means paid and networked; the model test is free
  and offline. Merging them would make one impossible to run without the other.
- *Sharing a cache base module with `llm/_cache.py`* — different payloads, key material and
  corruption semantics; rule of three, extract if a third cache appears.
- *Recalibrating thresholds into the original 0.016 band* — a knife-edge the next batch of
  material would invalidate. Fixing the input widened the margin 3.5× instead.

**Carried into slice 5 — the gray zone holds two different questions.** SPEC §3.1 frames
adjudication as `SAME | OVERLAP | DISTINCT`, which assumes every flagged pair is a
*redundancy* question. Real output from the first ingested source says otherwise. Of three
gray-zone pairs, the weakest was `um-hilfe-bitten` ↔ `verben-mit-praepositionen` at 0.882 —
not a duplicate at all, but *bitten **um** + Akkusativ*, which is a verb-preposition
combination and therefore a `governs` relation (SPEC §4.2).

So high similarity means "these are connected", and *how* they are connected is the thing
adjudication has to decide. Slice 5's outcome set needs a fourth branch:

- `SAME` → discard, append source id and any new examples (§3.2)
- `OVERLAP` → merge (§4.1)
- `DISTINCT-but-related` → **propose a typed edge** (§4.2), write nothing to the bodies
- `DISTINCT` → leave alone

Without that fourth outcome, every genuine relation gets mis-answered as a merge question and
either fragments the wiki or corrupts a node body.

**All four outcomes route through review. `interrupt()` fires on link proposals exactly as it
does on merge proposals.** A proposed typed edge is a reviewed write, not a side effect —
`DISTINCT-but-related` is an *adjudication result awaiting approval*, in the same queue and
under the same gate as `OVERLAP`.

This is the load-bearing assumption, and stating it is what separates a reading of §4.3 from a
loophole. §4.3 says relations are a batched background pass, not an ingest-time inference. The
reason slice 5 may propose an edge from the adjudication pass anyway is *only* that the
proposal is reviewed — detection has already paid for the neighbour search, so the candidate
pair is in hand, and what §4.3 protects against is **auto-accepting** it. If a `governs` edge
could land without approval while a merge required it, §4.3 would be violated after all, and
ADR-003's "nothing auto-writes to /nodes" with it. An edge changes how the graph is traversed
and how §5.1 priority scores compute; it is not a lesser write than a body edit.

(§4.3's "auto-accept > 0.9, queue the rest" applies to the *later* batched relation-inference
pass, once merge acceptance is trustworthy per ADR-003's revisit condition — not to slice 5.)
- *Mean-centering the vectors to counter e5's anisotropy* — the principled fix for the narrow
  band, and held in reserve rather than built. The blocking reason is **cache coherence**, not
  just statefulness: the corpus mean shifts every time a node is added, so vectors cached
  against an old mean silently drift from query vectors computed against a new one. That turns
  a stateless transform into one whose cached output has a hidden dependency on corpus
  composition — the mean would have to enter the cache key, and every addition would invalidate
  everything. Trimming the input kept the whole pipeline stateless and was enough. Pay that
  complexity only if the margin closes again at volume.

---

## ADR-011 — Slice 5: proposals as durable state, four outcomes, and two unequal drift guards

**Decided:** Slice 5 turns slice 4's gray-zone list into reviewed decisions. Seven things
inside it are worth not re-litigating.

**1. The durable state between adjudication and review is a FILE, not a paused graph.**
`interrupt()` needs a checkpointer, but nothing here needs a *paused graph* to survive the
process. `gw adjudicate` runs to `interrupt()`, writes one Markdown-with-frontmatter
proposal per proposed decision into `/proposals`, and exits. `gw review` reads those files
minutes or days later. So `InMemorySaver` is sufficient and no new dependency is needed.

Re-deriving a proposal's context on the apply pass is free, because adjudication is
content-hash cached (ADR-005) and embeddings are cached (ADR-010) — **the cache is the
checkpointer**. A `SqliteSaver` was rejected: it adds a dependency, puts pending human
decisions in an opaque binary store inside `data/`, which ADR-001 declares freely
deletable, and creates a second source of truth for "what is pending" that would drift
from the files and the index.

`/proposals` is gitignored exactly like `/queue`. A proposal is resolved by
approval-or-rejection and deleted either way, so the directory only ever holds pending
work. The durable record is the commit to `/nodes` that approval produces, plus
`logs/decisions.jsonl`.

> **AMENDED in slice 7 — the proposal body is authoritative only when there is no staged
> node file.**
>
> As originally written, this section said the proposal body is the hand-editable content
> and "what you edited is exactly what gets written". That is right for a **merge**, which
> has no other file. It is wrong for a **create backed by `candidate_path`**, and the
> difference was a silent data-loss bug.
>
> A staged create has *two* editable artifacts holding duplicate content: the queue file
> (ADR-009: "review needs nothing but an editor") and the proposal body (this ADR). They
> can diverge, and `apply_create` resolved the divergence in favour of the proposal —
> overwriting the loaded staged node's body with its own copy, then `unlink()`ing the
> staged file. A reviewer's edit was therefore **discarded and destroyed, with no copy
> left anywhere**, while the command reported success.
>
> **The rule now: for any proposal backed by `candidate_path`, that file is authoritative
> for display, edit, and write; the proposal never carries a duplicate body.** So
> `gw review` renders the staged file's *current* content, `apply_create` promotes that
> file, and staged-create proposals are written with `body_md: ""`. Merges keep the
> original rule unchanged, because there is no second artifact to disagree with.
>
> The bug was latent from slice 5 and first reachable in slice 7 — `gw families` is the
> first code that stages a node *and* writes a proposal carrying the same body. Regression
> test: `test_cli_grid.py::test_a_hand_edit_to_a_staged_node_is_what_lands` edits only the
> staged file and asserts that edit reaches `/nodes`.
>
> The general principle, worth more than the specific fix: **two artifacts holding the
> same content is the bug, not the tie-break rule that resolves them.** Eliminating the
> duplicate is what makes "what you edited is what lands" true rather than conditional.

**2. Merges and links share one queue, one format, one command.** ADR-010 says a proposed
edge sits "in the same queue and under the same gate as OVERLAP", and a second,
lighter-feeling queue for links is precisely the loophole §4.3's reinterpretation has to
avoid. `kind` is a field, not a directory.

**3. The gate is topological, not conventional.** The graph has **no edge** from
`adjudicate` to any apply node; `adjudicate` ends at `END`. The only way into `route` is to
enter the graph already holding a human decision, which only `gw review` does. So even a
resume of the propose pass cannot reach a write. AST tests pin `interrupt()` to the
adjudication node and pin the apply functions to a single caller.

**Routing is implemented exactly once.** `_apply.apply_merge/link/create/discard` is the
implementation; the graph's four nodes are thin wrappers, and `gw review` drives the graph
rather than a parallel copy. Two implementations is how one of them quietly stops honouring
the gate.

**4. Everything writes through the slice-3 promote seam.** `write_approved` was factored
out of `promote_source` and is now the single `write_node(..., learn=True)` call site *and*
the create-vs-overwrite precondition (`expect_exists`). Two responsibilities on one
function, so the ADR-007 test was **tightened from file-level to function-level**: the old
assertion (*the call is in `ingest/_promote.py`*) would have stayed green whether the call
lived in `promote_source` or `write_approved` — it could not see the refactor at all, and
so no longer pinned what it existed to pin.

**5. Adjudication runs on `glm-4.5-flash` during tuning, not SPEC §9's `glm-4.6`.** Flash
is free and explicitly zero-priced, and slice 5 is the token-hungriest phase. The switch is
a one-line YAML edit; the model name is in the cache key, so it re-adjudicates every pair
on 4.6 — one paid pass after free tuning. **Add 4.6's real price at that moment**, checked
live (ADR-008 §4 forbids a guessed rate, and a paid model left unpriced defeats cost
tracking entirely). Consequence: flash verdicts are pipeline development, not trusted
production merges, so every decision record stores the deciding `provider`/`model` and
"which merges came from flash?" stays a grep.

***§5 addendum (measured, 2026-07-31).*** The paragraph above was written as a prediction.
It has now been tested. `glm-4.5-flash` was run against two real gray-zone pairs from the
first ingested source and produced defensible-sounding but incorrect verdicts on **both**:

- **0.882 — `um-hilfe-bitten` ↔ `verben-mit-praepositionen`:** answered `DISTINCT`
  ("A focuses on politeness levels in requests for help, while B covers verb-preposition
  combinations requiring specific cases"), missing that *bitten **um*** is itself one of
  B's verb-preposition combinations. Correct answer: `governs`.
- **0.900 — `verben-mit-praepositionen` ↔ `wechselpraepositionen`:** answered
  `same_family`, which per SPEC §4.2 means a **shared root or stem**
  (*stellen/stehen/setzen/sitzen*). These two share none. A plausible-sounding but
  definitionally wrong relation *type*.

Both failures are the same shape: confident, well-phrased, and wrong in a way a reviewer
skimming "yes, these are related" could wave through. The second is the more instructive,
because a typed-edge system's whole value is that the *type* carries meaning — §4.2 exists
precisely because generic backlinks produce a useless hairball, and a mislabelled edge is
worse than no edge, since §5.1's priority scores and the §6.1 morphological clusters both
read the type as a fact.

**So the 4.6 switch is a correctness requirement, not a cost trade.** The risk is not
obviously-bad output a human catches; it is plausibly-wrong relation types that erode the
pedagogical meaning of the graph while every individual proposal looks reasonable. Anyone
reaching for "flash is free and seems fine, let's keep it for verdicts too" should read the
two bullets above first — that is why they are recorded here rather than only in a test.

`tests/test_merge_live.py` pins this: the ADR-010 `governs` verdict is asserted **strictly**
and marked `xfail(strict=True, raises=AssertionError)` while the configured adjudication
model is in `TUNING_MODELS`, keyed off `resolve_step("adjudication").model` so the
strictness tracks config automatically. Switching to 4.6 flips it to a hard assertion with
no test edit. `raises=AssertionError` matters: a bare `xfail` would swallow a rate-limit or
truncation error and report the same XFAIL as a wrong verdict — the test would look like it
had confirmed this claim while never reaching an assertion. Same principle as §6's ledger
read: "unknown" must never be indistinguishable from "checked, as expected".

**6. The two SPEC §12.1 drift guards bite with deliberately different force.**

*Unsourced example sentences → flag.* Example lines in a regenerated body are checked
against A, B and their `/raw` texts, and misses are marked ⚠ in the review diff. The
check is fuzzy — legitimate paraphrase produces non-verbatim examples that are not drift —
so a hard refusal would false-positive and get routed around. The human gate (ADR-003) is
the real guard; this aims attention. The check runs on the `## Examples` section only:
prose is legitimately rewritten on merge, but a fabricated example sentence is a fact you
would memorize.

*Regeneration cap → hard refuse.* An exact integer with no false positives, guarding the
one drift a reviewer structurally **cannot** see: the reviewer judges one diff at a time
and never sees cumulative divergence across many merges. A capped node emits a `MANUAL`
proposal rather than dead-ending. The cap and `/raw` immutability are two halves of one
defence — the cap stops drift accumulating, and raw-immutability is what lets a capped node
be *re-derived from its sources* rather than re-merged from its already-drifted state.

Only `OVERLAP` counts toward the cap. `SAME` appends provenance and new example lines
mechanically, with no model call and no re-encoding, so it is not a regeneration.

**`logs/decisions.jsonl` is therefore a different class of artifact from
`logs/llm_usage.jsonl`.** ADR-008 §5 tracks the cost ledger because spend history is a
durable record; losing it costs a statistic. This one is the *authoritative regeneration
count*, so losing it would disarm a safety guard. Two consequences: it is git-tracked (a
second `.gitignore` negation), and **`merge_count` raises rather than returning 0** when it
cannot read the file. A missing ledger must never be indistinguishable from "never merged".

That forces a three-state read, because a strict "missing → always refuse" would dead-end a
fresh clone, where the ledger is legitimately absent:

| ledger | node | result |
|---|---|---|
| readable | any | authoritative count; `version` is not consulted |
| missing / corrupt | `version` unset or 1 | proceed — never merged, nothing was lost |
| missing / corrupt | `version` > 1 | **refuse**, and name `git restore` as the fix |

`Node.version` is a **tripwire, never the count**. It is hand-editable, so it cannot be
trusted to *permit* a merge — but it can be trusted to *forbid* one, since a wrong value in
that direction costs only a refusal. One corrupt line poisons the whole read rather than
being skipped, because skipping would silently undercount into the same fail-open.

The asymmetry is deliberate: the "don't re-propose an already-decided pair" lookup handles
an unreadable ledger *permissively*, because forgetting a rejection costs one redundant
question while forgetting a merge would let the cap fail open.

**7. OVERLAP regeneration demotes status: `stable` → `reviewed`. SAME and links leave
status untouched.** Status is the trust signal, and OVERLAP is a lossy machine re-encoding
(§12.1) — the node's body was rewritten and only confirmed as a *diff*, not re-vetted
whole, so `stable` would overclaim.

This mirrors the cap exactly: the operation that counts toward `MAX_REGENERATIONS` is the
same operation that demotes status. **"OVERLAP is the drift event", applied to both
guards.** That symmetry is structural rather than coincidental — `_apply._status_after`
keys off `_ledger.REGENERATING_OUTCOMES`, the same frozenset the cap counts, so the two
guards cannot silently disagree after a future edit to either one. A test asserts the
biconditional directly.

Demotion **caps at `reviewed`, never forces `draft`**: a regeneration should stop a node
claiming more than it re-earned, not undo the review it already had. `draft` and `reviewed`
nodes are already at or below that ceiling and are unaffected.

**Also:** the graph's unit is one candidate, which has one *fate* plus zero or more typed
edges — so `DISTINCT_RELATED` yields a `create` **and** a `link`, and creates are applied
before links so an edge never dangles. Candidates are compared against `/nodes`, not
against each other. The `duplicate` band (≥ `GRAY_HIGH`) resolves to SAME with **no model
call** but still produces a proposal: confidence saves the call, not the human gate. The
few-shot exemplars deliberately exclude ADR-010's `um-hilfe-bitten` ↔
`verben-mit-praepositionen` pair, because that pair is the live test's assertion and
teaching it would make the test measure prompt recall instead of generalization.

**Rejected:**
- *`langgraph-checkpoint-sqlite` + `SqliteSaver`* — a dependency and a second source of
  truth, to persist a paused graph nothing needs to resume.
- *A separate queue or an auto-accept threshold for link proposals* — the exact loophole
  ADR-010 §4.3 warns about.
- *Implementing routing twice, once as graph nodes and once as review helpers* — how one
  copy stops honouring the gate.
- *Hard-refusing unsourced examples* — fuzzy check, so it would false-positive on
  legitimate reformatting and get worked around.
- *Reading the regeneration count from `Node.version`* — hand-editable and absent on the
  seed nodes; sound as a tripwire, unsound as an authority.
- *Treating a missing ledger as count 0* — the fail-open this whole section exists to
  prevent.
- *Retaining `stable` through a regeneration* — makes status a sticky label that survives
  the very operation it should respond to.
- *Demoting all the way to `draft` on regeneration* — throws away a review that did happen,
  and would make every merge feel like starting over.

---

## ADR-012 — Node ids carry real German; source ids stay ASCII

**Decided:** The two identifier types deliberately differ.

- **Node ids** preserve umlauts and ß: `Wechselpräpositionen` → `wechselpräpositionen`.
- **Source ids** stay transliterated ASCII: `test-büro.txt` → `20260726-test-buero-<hash>`.

`ingest/_raw.py` therefore exposes two slug functions rather than one. `node_slug`
NFC-normalizes and keeps the letters; `slugify` transliterates (ä→ae, ß→ss) and strips
remaining diacritics. `_nodes.node_id_for` uses the first, `_raw.resolve_source_id` the
second.

**Why they differ, rather than one convention winning.** They are not the same kind of
name:

A **node id is human-facing**. It is the filename, and ADR-002 keeps these files plain
Markdown specifically so the vault opens in Obsidian for graph view — where Obsidian
labels every note by its *filename*. So the id is what you read in the sidebar and on
every graph node. It is also the value in `links: target:`, and the argument you type
into `gw review --proposal` and `gw promote`. Content was always proper UTF-8 (titles,
bodies, examples, tags, `/raw`); only the id was transliterated, which made the graph the
one place the wiki looked less German than it is. In a toolchain that is UTF-8 end to end,
that bought nothing.

A **source id is an opaque machine handle**. It names a file in `/raw`, which SPEC §1.2
makes immutable and append-only, and it already carries a content hash (`…-90458c3d`) that
marks it as machine-generated. Nobody reads it as German. More decisively: because `/raw`
is append-only, its filenames are *historical records* and are never renamed — so changing
the convention would not migrate the existing ones, it would only leave `/raw` permanently
mixed. ASCII there is not a compromise, it is the correct answer for a name that can never
be revised.

**NFC normalization is what makes umlaut ids safe, and it is load-bearing.** `ä` has two
encodings — precomposed U+00E4 and decomposed U+0061 U+0308 — identical on screen,
different bytes, therefore different filenames. macOS stores filenames NFD while Linux and
Windows use NFC. Without normalizing, the same title arriving in the other form would
create a *second node for a word that already has one*: exactly the fragmentation ADR-006
exists to prevent. `node_slug` normalizes to NFC before slugging, which removes the hazard
rather than dodging it — and dodging it was the strongest argument for ASCII ids in the
first place.

**Migration was done in one atomic step**, at 5 nodes, because a half-applied rename would
leave `wechselpraepositionen` and `wechselpräpositionen` coexisting as two nodes for one
concept. Verified before renaming: zero `links: target:` and zero `source_ids` entries in
either the live corpus or the frozen fixtures referenced a changing slug — every existing
link target is a forward reference to a node not yet written. That is why this cost two
file renames and no link rewrites; it would not have stayed that cheap.

The embedding cache was unaffected: ADR-010 keys it on `(model, embed_text)`, and
`embed_text` is built from `title_de` plus the body, never the id. `gw reindex` restored
all five vectors.

**A future session must not "fix" the inconsistency.** It is deliberate, and both
directions of `fix` break something: transliterating node ids throws away the German in
the Obsidian graph and in every link target, while un-transliterating source ids means
renaming files in an append-only store whose whole guarantee is that they are never
touched. `tests/test_ingest_raw.py::test_the_two_slug_functions_deliberately_disagree`
asserts the divergence directly so a well-meaning unification fails loudly.

**Rejected:**
- *One slug function for both* — simpler, but it forces the wrong answer on one of the two
  identifier types whichever way it goes.
- *Umlaut source ids* — would rename nothing that already exists (append-only `/raw`) and
  so would only fragment the convention.
- *Keeping ASCII node ids* — the status quo, and defensible only while the NFC hazard was
  unhandled. Once normalized, its remaining cost was a graph view that spells your own
  notes wrong.
- *Deferring the migration* — the rename is free only while no `links: target:` points at
  a changing slug. That window closes as soon as the wiki cross-references itself.

---

## ADR-013 — Slice 6: CEFR is rules-first, and the wordlist ships as a seam with no data

**Decided:** SPEC §5's three signals, in strict precedence, plus §6.2's register
vocabulary. Seven things worth not re-litigating.

**1. The lexical anchor ships as working code with no wordlist.** `vocab/cefr/a1.txt …
c2.txt`, one lemma per line — the shape a real Goethe or DWDS list drops straight into.
Missing files, empty files and blank lines are all "no signal", never an error.

**No hand-written starter list.** A CEFR list written from recollection is exactly the
unreliable per-item judgment SPEC §5 introduces the rules approach to *replace*, and it
would put unearned confidence into `cefr_basis` where nobody could audit it. The project
already applies this rule to model prices (ADR-008 §4: a wrong rate is worse than a
visible gap); a wrong level is worse, because it silently reorders study.

**The data is gitignored, the directory is not.** Real lists carry licensing this repo
cannot redistribute, so `.gitignore` excludes `vocab/cefr/*.txt` while
`vocab/cefr/README.md` stays tracked — which is *also* what keeps the directory present on
a fresh clone. **No `.gitkeep` and no `!` negation are needed**, because `*.txt` does not
match `README.md`. Verified: `git check-ignore` catches `a1.txt`, and `git add -An
vocab/cefr/` adds only the README.

Tests read `tests/fixtures/cefr/` (committed), never `vocab/cefr/`. A test reading the real
directory would pass only on a machine that happens to have a list installed.

**Consequence, stated plainly:** until a real list exists, only *title-anchored* grammar
nodes level from rules. On the live corpus that is one node of five; the other targets
reach the tiebreak. Adding a list later is zero code change.

**2. Grammar hits are graded by where they matched, and this is load-bearing.** SPEC §5's
table is a lookup, but *matching a node to a row* is an interpretation. The naive
whole-node keyword sweep mislevels, and the live corpus proves it: `verben-mit-präpositionen`
is B1 and its body mentions *Akkusativ*, which §5 puts at A2 — a flat sweep proposes
**B1 → A2** and relabels a B1 rule as beginner material.

So a hit in `title_de` says what the node *is about* and decides; a hit in `body_md` says
what it *mentions* and is weak evidence. A body-only hit that disagrees with the node's
current level goes to the tiebreak rather than overriding it. Title hits shadow body hits
entirely rather than being weighted — averaging them would produce a level neither signal
supports.

Two matching details with tests: **`Konjunktiv II` is protected from `Konjunktiv I` by a
negative lookahead, not by list ordering** (ordering works only until someone re-sorts the
table); and matching anchors at the start of a word but not the end, so `Akkusativobjekt`
counts as *Akkusativ* while `Plusquamperfekt` correctly matches nothing, since it is not in
§5's table.

**3. `gw relevel` writes through the `/proposals` queue as a fifth proposal kind.** Not
directly to `/nodes`, and **no `--write` flag** — that is the batch-write-unreviewed pattern
ADR-009 already rejected. Determinism of the lookup does not exempt it: ADR-003 gates
writes to `/nodes`, not uncertain judgments, and `cefr` drives SPEC §5.1's priority score,
so a bad relevel reorders what you learn. If high-volume re-levelling later needs speed,
add batch *approval* on top of the queue (approve-all-where-`basis: rules`), never a bypass.

`relevel` is ordered **after `merge`** in `KIND_ORDER`, because a merge archives its loser
and re-levelling an archived node would fail the `expect_exists` check.

**No status demotion.** ADR-011 §7 demotes `stable → reviewed` on OVERLAP because the
*body* was re-encoded and the reviewer only confirmed a diff of it. A relevel rewrites two
frontmatter fields and the reviewer saw the entire change, so trust is intact — same
reasoning as an approved link.

**4. `cefr_basis` is `signal:detail(LEVEL)`, semicolon-separated.** Formalizing the loose
convention the hand-authored seeds already used. The tiebreak records *what it was shown*,
not merely what it said:

```
grammar:wechselpräposition(A2)
grammar:wechselpräposition(A2); goethe:a2(helfen)
llm:tiebreak(B1); grammar:akkusativ(A2,body); lexical:none
```

**`llm:tiebreak` stays greppable on purpose** — it is the successor to slice 3's
`llm:extraction` marker (ADR-009), so `grep -l 'cefr_basis: llm:tiebreak' nodes/` remains
the one command that finds the least-grounded levels in the wiki. This matters more than it
looks: a tiebreak level is indistinguishable from a grammar-anchored one in the file unless
you read the basis, and until a wordlist exists every pure-vocabulary node has one.

**A derived `llm:tiebreak` basis is NOT a placeholder**, so re-running does not re-target
it. A *missing* basis is, because SPEC §5 says always store one.

**5. Hand-authored bases are not touched by default.** `freq:high; goethe:A1(waschen)` on a
seed node is human judgment, and recording it is the entire reason `cefr_basis` exists.
`--all` overrides this and is opt-in for exactly that reason.

**An ABSENT basis and a PLACEHOLDER basis are different facts, and conflating them inverts
the slice.** Both are *targeted* — SPEC §5 says always store a basis, so a node without one
has an unexplained level either way. But only one of them licenses changing the level:

- `cefr_basis: llm:extraction…` — the **level itself** was a machine guess, so the tiebreak
  may move it.
- `cefr_basis` absent **and both anchors silent** — only the *explanation* is missing. The
  level may well be a human's, and a tiebreak running on `grammar:none; lexical:none` has
  strictly **less** information than whoever set it. So the level is kept and the basis is
  recorded as `human:seed`. **The tiebreak explains; it never moves.**

This was caught by running the slice against the live corpus, not by reasoning: `prefix-an`
is a hand-authored A2 seed with no basis, and the first implementation proposed **A2 → B1**
purely because the explanation field was empty. Overwriting a human's level on zero
evidence is precisely what a rules-first design exists to prevent, and the review gate
catching it is not good enough — it would recur on every seed node and train the reviewer
to skim.

The protection is narrow on purpose: it applies only when the rules are *silent*. A real
grammar anchor still outranks an unexplained level, and `human:seed` is not a placeholder,
so re-running converges instead of re-deriving forever.

**6. The tiebreak runs on free `glm-4.5-flash`, never the paid `glm-4.6`.** Adjudication is
a per-pair judgment worth paying for; a tiebreak can fire on *every* node lacking a grammar
match. It is also the least-grounded signal, which argues for spending less on it, not
more. Same limited judgment quality ADR-011 §5 measured on adjudication applies here — the
mitigation is the wordlist, which anchors vocabulary nodes and stops them reaching the
tiebreak at all.

**7. Register gains SPEC §6.2's dimensions; examples are NOT parsed out of `body_md`.**
`vocab/registers.txt` gains formality (`du-ebene`, `sie-ebene`, `neutral`), domain
(`behörde`, `akademisch`, `medien`), mode (`gesprochen`, `geschrieben`) and regional
(`de`, `at`, `ch`), with aliases folding spelling variants. Behördendeutsch gets its own
domain tag, which §6.2 calls out as a genuinely distinct sublanguage rather than a
formality level.

**`Example.register` stays reserved and unpopulated, deliberately.** Parsing the
`## Examples` bullets out of `body_md` into the structured field is a body-format migration
with a dual-source-of-truth question — and slice 5's merge regeneration rewrites `body_md`
knowing nothing about an `examples:` field, so the two would drift on the first merge. It
belongs in its own slice. Node-level tags are the interim mechanism; the empty field is
intentional, not forgotten.

**`schriftlich → geschrieben` was deliberately NOT aliased.** §6.2 names the mode axis
`gesprochen`/`geschrieben`, and `schriftlich` is the same concept under a different word —
but two existing nodes already carry `schriftlich`, and an alias rewrites their tags on the
next write. That is a change to hand-authored material, so it is the author's call rather
than a side effect of adding vocabulary. Both remain valid tags until someone decides.

**No second `learn=True` path.** Ingest stages with `learn=False`; the vocabulary grows only
at `write_approved` (ADR-007). The existing AST test pins this and stayed green unchanged
through the whole slice.

**Also:** the extraction prompt bumped to `extract@2` to name §6.2's dimensions. That
orphans two cached *extraction* entries on free flash and **provably does not touch the
paid glm-4.6 adjudication cache** — prompt versions are per-`Prompt` module constants and
the messages differ entirely, verified by recomputing an adjudication cache key across the
bump and getting a byte-identical hash (ADR-008 §1 is what makes per-step invalidation
independent).

**Priority scoring (SPEC §5.1) is deferred.** SPEC §11 already places it at slice 9+, and
its `freq_rank_score` term depends on the wordlist deferred in point 1. Building it now
would ship a formula with a constant standing in for its main input.

**Rejected:**
- *A hand-written starter wordlist* — unauditable guesses in the one field that exists to
  make levels auditable.
- *Committing a real Goethe/DWDS list* — licensing this repo cannot redistribute.
- *`gw relevel --write`* — the batch-write-unreviewed pattern ADR-009 turned down, and the
  level it writes drives study order.
- *Folding re-levelling into `gw reindex`* — reindex rebuilds the derived index and must
  never write to `/nodes` (ADR-001).
- *A flat keyword sweep over title + body* — demonstrably downgrades `verben-mit-präpositionen`
  from B1 to A2 on a passing mention.
- *Ordering the grammar table so `Konjunktiv II` precedes `Konjunktiv I`* — works until the
  table is re-sorted; a negative lookahead does not.
- *Parsing examples into `Node.examples` this slice* — dual source of truth, and slice 5's
  regeneration would drift from it immediately.

---

## ADR-014 — Slice 7: the grid reads human assertions, and withholds what it cannot know

**Decided:** SPEC §7's root × prefix grid ships as a **view plus on-demand analysis**.
Seven decisions inside it are worth not re-litigating.

**1. Three prefix inventories, not two.** Separable and inseparable are the obvious split
and they are not enough. A third class exists whose separability is not recoverable from
text at all:

    úmfahren   run over      separable
    umfáhren   drive around  inseparable

Identical spelling; the difference is stress, and stress is not written. So `um durch über
unter wider wieder hinter voll` are a **variable** class, and a variable-prefix verb yields
**no segmentation and no grid cell** — not a cell carrying a caveat.

The distinction matters: a grid saying *"here is a family, caveat"* still teaches the
family, while a grid saying nothing prompts you to decide. Stress-homographs are exactly
where a silent segmentation does §7.4-shaped damage. **The withholding is structural** —
membership in `VARIABLE` *is* the refusal — rather than a flag downstream, because a flag
is something a later change can quietly start ignoring. `classify()` tests `VARIABLE`
first, since several of its members are also legitimately in `SEPARABLE`.

One escape hatch, and it is human: an explicit `separable:` on a node overrides the
inventory. That is §7.4's *"the node holds the truth; the grid only predicts a guess"*,
applied to segmentation rather than to meaning.

**2. Columns come from human assertions, not from segmentation — and this is the
non-obvious one.**

The natural design derives grid columns by segmenting words. It cannot work here, and the
reason is circular: `_segment` refuses `ankommen` because the corpus has no `kommen` node
(rule 3 below), so a segmentation-driven grid could **never display the very links you
wrote**. Corpus roots alone are no better — today they yield a 1×1 grid whose only cell is
the non-word `anwaschen`.

So an implied column comes from stripping a prefix node's *own* morpheme from its *own*
`same_family` targets. `prefix-an` already asserts that `ankommen` belongs to it; the grid
reads that claim and derives the column `kommen`. **That is reading a human's assertion,
not guessing** — which is why it is allowed to be looser than segmentation is.

If someone later "simplifies" this into calling `segment()`, the grid goes blank against a
corpus full of forward references, and the cause will not be obvious.

**3. Segmentation requires corpus evidence.** A split is proposed only when the residual
stem already exists as a `root:` or a lemma. `verstehen` does not split, because there is
no `stehen` node — which is SPEC §7.4's trap (`verstehen` ≠ `ver-` + `stehen`) shut by a
mechanical rule rather than by a judgment call. Prefix-shaped starts are never blindly
stripped.

The constraint also means the grid grows *as you study* instead of speculating about
vocabulary you have never met, and it keeps `gw families` from proposing edges into empty
space. `no-corpus-evidence` resolves itself the day the stem gets a node; only
`variable-stress` needs a person, which is why `Withheld.needs_human` distinguishes them.

**4. A human may dangle; the pipeline may not.** Every link in the corpus currently points
at a node that does not exist, and that is the feature (SPEC §7.3: *"Empty cells are words
you haven't learned yet"*).

- A hand-authored dangling link is an **intention**. `storage` does not validate target
  existence, and `gw families` never re-proposes an existing edge even when it dangles —
  turning your roadmap into a chore list would defeat it.
- A pipeline-created dangling link is a **bug**. Slice 5's `apply_link` keeps its refusal.

Same artifact, opposite rules, keyed on who wrote it — the same shape as ADR-007's
`learn=True`, where human promotion may grow the vocabulary and automated ingest may not.

Auto-creating stub nodes for dangling targets was rejected outright: it fills the gap the
feature exists to detect, turns missing nodes into existing-looking stubs, and floods
`/nodes` with material SPEC §3.4 says should not be nodes.

**5. Dangling-ness is computed at read time, never stored.** The grid asks "does this
target exist *right now*?". There is no flag anywhere.

The consequence is the point: the day you write `nodes/ankommen.md`, its cell reads
`learned` with **no migration, no flag to clear, no repair step**. A stored flag would need
updating when the target appeared, and that fixup is exactly the maintenance this design
refuses. A test creates the target of a dangling link and asserts the cell flips with
nothing else done.

**6. Transparency is the one model call, on the free step, pinned by a test.** Segmentation
is mechanical; whether a family is a scaffold or a coincidence is semantic and not
rule-decidable (§7.4). So `family_transparency` costs one `glm-4.5-flash` call and nothing
else does — asserted by a test that no second module in `morph/` calls `complete`.

It runs on **free flash, never the paid `glm-4.6`**: it is a per-candidate call that fires
as verbs arrive, and a wrong answer marks a grid cell rather than corrupting a node body.
The choice is asserted *against config*, so flipping adjudication to a paid model (as slice
6 did) cannot drag this along with it — the cheap-by-default choice is enforced, not merely
configured.

`family_transparency` is written to `/nodes`, so it routes through review like every other
classification (ADR-003). A family that already declares one is **never re-judged**, so a
model verdict cannot overwrite yours.

**7. A grid cell spells out a word only when something vouches for it.** This is a
rendering rule and it is load-bearing, which is why it is here rather than left as a
detail someone improves away.

The grid computes a full cross-product, and most of that product is not German. `an-` ×
`waschen` yields **`anwaschen`**, which is not a word. Printing it in a study tool reads as
*"learn this"*. At 1×3 that is a curiosity; at 10 prefixes × 20 roots it would be roughly
140 invented vocabulary items burying the real signal — and the tool **cannot** distinguish
`abkommen` (real) from `anwaschen` (not) without a dictionary it does not have.

So `learned`, `identified` and `irregular` print the word (a lemma, your own link, or a
node's transparency vouches for each). `gap` prints the glyph alone. The row and column
headers already say what the cell is, and composing it yourself is the exercise.

A future change reading *"show the composed word for completeness"* looks like an
improvement and is not: it re-introduces the non-word problem at whatever scale the corpus
has then reached.

**Also: SPEC §7.2's ingest-time auto-creation is deferred.** §7.2 wants a new verb linked
to its root and prefix nodes automatically, "creating either if absent". That is a write
path whose correctness rests on segmentation and transparency being trustworthy, and the
corpus holds one family node and one prefix node — too thin to tune an automatic
edge-writer against. Every earlier slice earned its automation by being watched first:
detection before merge (4→5), queue before `/nodes` (3), propose before apply (5, 6). When
the judgment is proven it can arrive as auto-accept-high-transparency / review-the-rest,
which is also ADR-003's own revisit condition.

**And one guard this slice exposed.** Adding `apply_morphology` left the suite **green**
while the new write path was unguarded: `test_only_the_graph_drives_the_apply_functions`
asserts which *files* call an apply function, and `merge/_graph.py` already qualified via
the other five. The registry was hand-maintained, so it could only catch omissions someone
remembered to prevent. It is now derived from `_apply.py` via AST, and every `Kind` is
asserted to have a route — a kind without one falls through the router to `discard`, which
reports success while writing nothing.

**Rejected:**
- *Two prefix inventories with a "variable" flag* — a flag annotates; membership withholds.
- *Deriving grid columns by segmenting* — circular; the grid would go blank against exactly
  the corpus it exists to display.
- *Storing a `dangling` boolean* — creates a fixup step every time a target appears.
- *Auto-creating stub nodes for dangling targets* — fills the gap the feature detects.
- *Re-proposing links that already exist but dangle* — turns a roadmap into a chore list.
- *Letting transparency run on whatever `defaults` points at* — it fires per candidate, so
  an accidental paid-model drag is a real cost leak.
- *Spelling out every cell "for completeness"* — invents vocabulary the tool cannot verify.
- *Wiring §7.2's ingest hook now* — automation before observation, which no prior slice did.
