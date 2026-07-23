# German Wiki — Design Specification

Personal German-learning wiki. Ingest images/text → extract concepts → merge into a
node graph → study via situational and morphological clusters.

Single user, single machine, local-first. This document is the authoritative design.
`CLAUDE.md` holds the working rules; this holds the *what and why*.

---

## 1. Core Data Model

### 1.1 The Concept Node

The atomic unit is a **Concept Node** — not a note, not a page. A node is one
learnable thing: a grammar rule, a word family, a fixed expression, a register pattern.

```yaml
Node:
  id: str                      # slug, unique, stable, filename stem
  title_de: str
  title_en: str
  type: grammar | vocab | phrase | pattern | culture
  cefr: A1 | A2 | B1 | B2 | C1 | C2      # single value, not a range — force precision
  cefr_basis: str              # why this level, e.g. "grammar:passiv" or "freq:rank_2400"
  register: [alltag, büro, formell, umgangssprachlich, schriftlich]   # multi-label
  themes: [küche, büro, arzt, amt, café, ...]                          # situational, multi-label
  body_md: str                 # canonical explanation (the Markdown body)
  examples:
    - de: str
      en: str
      source_id: str
      register: [str]
  links:
    - target: str              # node id
      relation: str            # see §4.2
      confidence: float
  source_ids: [str]            # provenance → /raw
  confidence: float            # 0–1, LLM's certainty on merge/classification
  status: draft | reviewed | stable
  version: int
  updated_at: datetime
```

Type-specific optional fields:

```yaml
# type: pattern (prefix nodes)
separable: bool                        # trennbar vs untrennbar
family_transparency: high | drifted | opaque

# type: vocab (word families)
lemmas: [str]                          # members of the family
root: str                              # shared stem
```

### 1.2 Storage

**Markdown files with YAML frontmatter, in `/nodes`, tracked in git.**

Rationale:
- **Diffable** — you can see exactly what a merge changed
- **Revertable** — a bad merge is `git revert`
- **Greppable** — no query language needed for "where did I mention X"
- **Portable** — opens in Obsidian for free graph view, no lock-in
- **Durable** — outlives this app

**SQLite + sqlite-vec is a DERIVED INDEX.** It holds embeddings and enables fast
querying. It is always rebuildable from `/nodes` via `gw reindex`. It is never the
source of truth. If the DB and the files disagree, the files win.

**`/raw` holds immutable extracted text**, one file per ingested source. The canonical
node body is a *derived view* over raw sources. This is what makes merge drift (§8.1)
recoverable.

### 1.3 Layer summary

| Layer | Location | Mutable? | Authoritative? |
|---|---|---|---|
| Raw extractions | `/raw` | No — append only | Yes, for provenance |
| Canonical nodes | `/nodes` | Yes, via review | Yes, for content |
| Index | `data/*.db` | Rebuilt freely | No |

---

## 2. Ingestion Pipeline

```
Input (image / PDF / text / audio)
  → OCR / ASR
  → Extraction: LLM emits structured JSON of candidate concepts
  → Embedding + retrieval: find top-k similar existing nodes
  → Adjudication: SAME | OVERLAP | DISTINCT
  → Review queue (human approval)
  → Write + relink
```

**Key design choice:** don't extract *notes*, extract *claims*. One textbook page may
yield 6 discrete concepts. The extractor outputs a list of candidate concepts with a
proposed title each — never one undifferentiated blob.

**Extraction cap:** maximum 5–8 candidate nodes per source. If a single page tries to
emit 20, the extractor is atomizing rather than conceptualizing. The cap forces it to
pick genuinely distinct concepts, and kills fragmentation at the source (§3.3).

---

## 3. Feature 1 — Redundancy Removal

### 3.1 Three-tier detection, cheap → expensive

1. **Exact / near-exact** — normalized text hash, MinHash over shingles. Catches
   copy-paste duplicates. Free.
2. **Semantic** — embed candidate, cosine against existing corpus. Threshold ~0.85
   → candidate duplicate. Free (local embeddings).
3. **LLM adjudication** — only for the 0.75–0.92 gray zone:

```
Are these the same learnable concept?
A: {existing}
B: {new}
Answer: SAME | OVERLAP | DISTINCT
If OVERLAP, state what B adds that A lacks.
```

Because tiers 1–2 are free and handle the majority, the LLM fires on roughly 10–20%
of candidates. Redundancy detection is mostly *not* an LLM problem.

### 3.2 Outcomes

- **SAME** → discard B's body; append its `source_id` and any *new* example sentences to A
- **OVERLAP** → route to merge (§4.1)
- **DISTINCT** → create new node

**Never silently delete.** Discarded duplicates go to `/_merged` with a pointer to the
surviving node. You will want to audit this heavily in the first few hundred items.

### 3.3 The threshold is the node-count dial

The OVERLAP threshold directly controls how many nodes you end up with:

- **Merge aggressively** → fewer, richer, denser nodes
- **Merge conservatively** → more, thinner nodes

For a personal study wiki, bias hard toward **aggressive**. Dense nodes teach the
connections German rewards; thin nodes just relocate disorganization.

### 3.4 Granularity: what earns a node

| Earns its own node | Lives inside another node |
|---|---|
| A word **family** (shared stem) | Individual inflected forms |
| A grammar **rule** | Every example sentence of it |
| A **pattern** (e.g. request across registers) | A single word that fits an existing family |
| A concept you'd **link to or review** | A word that is just "a noun meaning X" |

**Test:** a node is something you'd want to review as a unit, or link to from elsewhere.
If a candidate will never be linked and never reviewed alone, it's an example or an
attribute — not a node.

Expected scale: ~2,000–4,000 vocab nodes (families, not words), 300–500 grammar rules,
a few thousand patterns. **Low tens of thousands total.** SQLite doesn't notice.

---

## 4. Feature 2 — Merging & Interconnection

### 4.1 Merging

On OVERLAP, **regenerate** rather than concatenate:

```
Write one canonical explanation covering both A and B.
Preserve every distinct example sentence.
Preserve any exception or edge case mentioned in either.
Do NOT add information not present in either source.
Output: merged body + a one-line changelog.
```

The "do not add" constraint is critical. Hallucinated grammar rules are the worst
failure mode for a study tool — you'd be memorizing fiction. Store the changelog;
`git diff` supplies the rest.

### 4.2 Typed relations

Generic backlinks produce a useless hairball. Use **typed** edges:

| Relation | Example |
|---|---|
| `contrasts_with` | Konjunktiv I ↔ Konjunktiv II |
| `prerequisite_for` | Akkusativ → Wechselpräpositionen |
| `formal_variant_of` | möchten → hätte gern |
| `same_family` | stellen / stehen / setzen / sitzen |
| `false_friend_of` | bekommen ↔ become |
| `governs` | verb → its case or preposition |
| `exception_to` | irregular member of a rule |

`prerequisite_for` is the highest-value edge: it lets you topologically sort the wiki
into an actual learning path.

### 4.3 Relation inference as a background pass

Do **not** infer relations at ingest time. Batch every N new nodes: embed-retrieve
neighbors, ask the LLM to propose typed edges with confidence. Auto-accept > 0.9,
queue the rest for review.

---

## 5. Feature 3 — CEFR Leveling

Zero-shot LLM CEFR judgment is inconsistent across sessions. Combine three signals:

1. **Lexical anchor** — check lemma against a wordlist. Goethe-Institut publishes
   official A1/A2/B1 vocab lists. Use DWDS frequency classes or a frequency dictionary
   for higher levels. Frequency rank is a strong CEFR proxy. *No LLM.*
2. **Grammar anchor** — hardcoded map. This is a known, stable syllabus; there's no
   reason to infer it:

   | Structure | Level |
   |---|---|
   | Präsens, Nominativ | A1 |
   | Perfekt, Akkusativ, Wechselpräpositionen | A2 |
   | Passiv, Konjunktiv II (basic), Relativsätze | B1 |
   | Konjunktiv II (full), Genitiv, erweiterte Infinitive | B2 |
   | Partizipialattribute, Nominalstil, Konjunktiv I | C1 |

   *No LLM.*
3. **LLM tiebreak** — only with signals 1 and 2 already in context, and only when
   they conflict or are absent.

Always store `cefr_basis` alongside `cefr`. When you disagree with a label, you need
to see what drove it.

### 5.1 Priority scoring

```
priority = freq_rank_score
         × (1 if cefr <= target_level else decay)
         × (1 + n_dependents)
```

Nodes that unlock many others (high `prerequisite_for` out-degree) get boosted. This
yields a genuine "learn this next" queue rather than an arbitrary list.

---

## 6. Feature 4 — Register & Theme (the two clustering axes)

### 6.1 Two orthogonal axes — do NOT force into one hierarchy

**Axis 1 — Thematic (situational):** küche, büro, arzt, amt, café. *Where/when you'd use it.*

**Axis 2 — Morphological (structural):** shared root/stem. *How the word is built.*

The same word lives in both. `abwaschen` is kitchen (thematic) **and** a `waschen` +
`ab-` member (morphological). Folders would force a choice and lose one. Tags and
typed links give both for free:

- Query by `themes:` tag → situational cluster
- Traverse `same_family` edges → morphological cluster

No new machinery: theme is a tag (this section), root-family is a relation (§4.2).

### 6.2 Register dimensions

Tag each **example sentence** as well as the node — register lives in usage, not in
the lemma.

- **Formality**: du-Ebene / Sie-Ebene / neutral
- **Domain**: Alltag, Büro/Beruf, Behörde/Amt, Akademisch, Medien
- **Mode**: gesprochen / geschrieben
- **Regional**: DE / AT / CH

**Behördendeutsch deserves its own domain tag.** It is a genuinely distinct sublanguage
(Nominalstil, passive constructions, *hiermit*, *Antragsteller*) and causes the most
practical pain.

### 6.3 Register pairs — a first-class node type

The same intent across formality levels:

> "Kannst du mir helfen?" → "Könnten Sie mir bitte behilflich sein?"
> → "Ich wäre Ihnen dankbar, wenn Sie … könnten."

This is the highest-value structure for the daily-vs-office distinction, because it
teaches the *mapping* rather than two disconnected facts. Model as `type: pattern`,
not as on-demand generation.

---

## 7. Feature 5 — Morphological Grids (trennbare Verben)

The single highest-leverage structure in German vocabulary. A **root × prefix grid**
turns dozens of scattered verbs into one learnable system:

| Prefix | + kommen | + machen | + stellen |
|---|---|---|---|
| **an-** | ankommen (arrive) | anmachen (turn on) | anstellen (hire / queue) |
| **auf-** | aufkommen (arise) | aufmachen (open) | aufstellen (set up) |
| **aus-** | auskommen (get by) | ausmachen (turn off / matter) | ausstellen (issue / exhibit) |
| **ab-** | abkommen (deviate) | abmachen (agree) | abstellen (turn off / park) |
| **vor-** | vorkommen (occur) | — | vorstellen (introduce / imagine) |

Learn the prefix's directional logic once (`an-` = toward/on, `aus-` = out/off,
`auf-` = up/open, `ab-` = away/off) and each new root's family becomes semi-predictable.

### 7.1 Two structural requirements

- **Prefix meaning is itself a node** (`type: pattern`). `an-` links to *every* verb
  using it — learn the prefix once, it pays off across the vocabulary.
- **The separability rule rides on the prefix node**, not on each verb. One rule, many
  beneficiaries. Note stress-dependent pairs: *úmfahren* (run over, separable) vs
  *umfáhren* (drive around, inseparable).

### 7.2 Ingestion behavior

When a separable verb arrives, the merge step links it to **both** its root node and
its prefix node, creating either if absent. The grid self-assembles from ordinary
ingestion — no separate authoring step.

### 7.3 Grid view = gap detection

Rendering family nodes as the root × prefix matrix makes the "learn 3, get 15 free"
payoff visible. **Empty cells are words you haven't learned yet** — the view doubles
as gap detection.

### 7.4 Caution: prefix logic is semi-regular

`verstehen` has nothing to do with `stehen` + directional `ver-`. `bekommen` ≠
`be-` + `kommen` in any useful sense. Meanings have drifted centuries past transparency.

The family link is a **learning scaffold, not an etymological claim.** Two safeguards:

- `family_transparency: high | drifted | opaque` on each family. High-transparency
  families you learn as a grid; opaque ones you learn as standalone words that merely
  *look* related.
- **The node holds the truth; the grid only predicts a guess.** When they diverge, the
  node wins and the grid marks the cell irregular — itself a useful "watch out" signal.

---

## 8. Additional Features

### 8.1 High value, low effort

- **Ingest queue with review UI.** Every merge/classification lands in a diff view you
  approve or reject. Do NOT auto-commit early on. This feature determines whether you
  trust the wiki.
- **Provenance links.** Every claim points back to its source image/page. Non-negotiable
  for a study tool — when something looks wrong you must be able to check the original.
- **Gap detection.** Diff your wiki against a reference syllabus for your target level
  → "you have no notes on Relativsätze im Genitiv." Turns passive storage into an
  active curriculum.
- **Anki / FSRS export.** Nodes → cards; let an existing scheduler do spaced repetition.
  Do not build your own SRS. Cloze-delete the example sentences.

### 8.2 High value, more effort

- **Personal error log.** Feed corrected writing/speaking mistakes in as a source type.
  Cross-reference against nodes → "7 Wechselpräposition errors." Weights priority by
  *your* actual weaknesses rather than generic frequency. Highest-leverage feature after
  the review UI.
- **Word-family clustering.** German rewards this enormously —
  `sprechen / Sprache / Gespräch / besprechen / Ansprache / versprechen`. Auto-cluster
  by stem, then explain each prefix's semantic drift. One node teaches ten words.
- **Chat over the wiki.** RAG against your own nodes. Answers from your notes, and flags
  when you have nothing on a topic → triggers a gap node.

### 8.3 Nice to have

- **Immersion capture** — browser extension / share-sheet target that clips German text
  from the wild into the ingest queue. Removes the friction that kills these projects.
- **TTS on examples** for pronunciation.
- **Weekly digest** of what changed and what to review.

---

## 9. Stack

| Layer | Choice | Why |
|---|---|---|
| Storage | Markdown + git | Diffable, revertable, portable, Obsidian-compatible |
| Index | SQLite + sqlite-vec | One file, no server, ample at this scale |
| Embeddings | **Local** — sentence-transformers, multilingual-e5-small | Free, offline, must be multilingual for DE/EN |
| Vision / OCR | GLM-5.2 or GLM-4.1V (Z.AI) | Vision + text on one provider |
| Extraction | GLM-4.5-Flash (free tier) or DeepSeek V4 Flash | Cheap, structured output |
| Merge adjudication | GLM-4.7 or DeepSeek V4 Pro | Quality-sensitive step; still pennies |
| Pipeline | LangGraph | Human-in-loop via `interrupt()` |
| CLI | Typer + Rich | Review UI is a CLI for the first several slices |
| Tests | pytest | |

**Provider swappability:** both Z.AI and DeepSeek expose OpenAI-compatible endpoints.
One client, base-URL swap. Model+provider configured **per pipeline step**, never
hardcoded.

---

## 10. Cost Model

Budget: coffee-money, worst case one cheap dinner, one-time.

Realistic lifetime ingestion (~3,000 sources, ~2K in / 500 out each) on cheap models:
**~$1–3 total**, before free tiers. GLM-4.5-Flash is $0. DeepSeek grants 5M free
tokens on signup. Expected actual spend: **≈ zero.**

**Caching is the real cost control, not model choice.** The token-hungriest phase is
tuning, where you re-run the pipeline on the same 100 sources dozens of times. Cache
every model call by content hash → re-runs cost $0. Without it, a $3 project becomes $50.

Secondary lever: put fixed prompt content (system prompt, schema, few-shot) **first**
and variable content last, to maximize provider prompt-cache hits.

---

## 11. Build Order

Vertical slices. Each end-to-end and reviewed before the next begins.

| # | Slice |
|---|---|
| 1 | Node layer: model, MD+YAML read/write, SQLite index, `reindex`, `list` |
| 2 | Model interface (swappable) + **disk cache** + token/cost logging |
| 3 | Text ingestion → extraction → new nodes (no merging, no dedup) |
| 4 | Local embeddings + duplicate **detection only** (report, no writes) |
| 5 | LangGraph merge pipeline + `interrupt()` + `gw review` CLI |
| 6 | CEFR anchors + register/theme tagging |
| 7 | Prefix/root nodes, `same_family` links, grid view |
| 8 | Vision ingestion |
| 9+ | Priority scoring, Anki export, gap detection, error log, RAG chat |

Slice 2 must come immediately after 1 — the cache has to exist before tuning burns tokens.

**Ship 1–5 before anything else.** The whole system's value depends on merge quality,
and that can't be tuned without watching it run on real material.

**No web frontend until the CLI pipeline is validated.** Don't build UI over a pipeline
you haven't seen work.

---

## 12. Risks

### 12.1 Merge drift

Every regeneration is a lossy re-encoding. After ten merges a node can quietly diverge
from anything in its sources.

Mitigations:
- Keep original extracted text immutable in `/raw`; treat the canonical body as a
  derived view
- Cap regenerations per node
- Periodically re-verify a node against its source texts

### 12.2 Over-engineering the wiki instead of learning German

The notorious failure mode of language-learning tooling.

**Rule: no new features until the current version has been used daily for two weeks.**