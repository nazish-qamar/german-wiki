# German Wiki

A personal German-learning wiki that builds itself from what you're studying, but only
with your approval.

Inspired by Andrej Karpathy's LLM Wiki pattern, with one deliberate change: here the LLM
proposes, but nothing reaches the wiki without human review.

Give it a text file, a photo of a textbook page, or a PDF. A language model pulls out
the learnable concepts (grammar rules, word families, fixed phrases). Each one is checked
against what the wiki already knows. You then decide whether it becomes a new note,
merges into an existing one, gets linked to related notes, or is thrown away. **Nothing
is written to the wiki without your approval.**

The notes are plain Markdown files with YAML frontmatter in [`nodes/`](nodes/), tracked
in git. You can read them on GitHub, edit them in any editor, and see every change in a
diff. Once notes exist, you can:

- **Study by situation.** Each note is tagged with register (`büro`, `formell`,
  `umgangssprachlich`, …) and theme (`amt`, `küche`, …), so you can pull up everything
  you need for, say, writing to the Ausländerbehörde.
- **Study by word family.** `gw grid` shows a root × prefix grid
  (*waschen → abwaschen, aufwaschen…*). The empty cells are what to learn next.
- **Follow the gaps.** `gw gaps` lists notes that are linked to but not written yet.
- **Export to Anki.** `gw anki` writes an `.apkg`. Re-exporting updates existing cards
  instead of duplicating them, so your review history is kept.

It runs on one machine. It's a CLI with no server, no account and no web UI.

## Engineering highlights

- **1,190+ tests** (pytest). Tests that call real model APIs or load the local embedding
  model are opt-in, so the default run is offline and free.
- **An architecture decision log.** [`docs/decisions.md`](docs/decisions.md) has 16 ADRs,
  each recording what was decided, the alternatives, and why. One example: why Anki note
  GUIDs come from node identity, so a re-export keeps your review history.
- **Cost-aware model routing.** Every pipeline step picks its own provider and model in
  [`config/models.yaml`](config/models.yaml):
  - Most steps (extraction, CEFR tiebreak, word-family transparency, OCR) run on free
    GLM "flash" models.
  - The one paid model, `glm-4.6`, is used only for adjudication. Flash was used to tune
    it and got real borderline pairs wrong; the config comment records why.
  - Most steps fall back to DeepSeek if Z.AI fails, and changing a model is a one-line
    config edit.
  - Every call is cached on disk by a hash of the full request (model name included), so
    re-running on the same input costs nothing, while switching models re-runs cleanly.
  - Prices are stored only when checked against the provider's own pricing page, with
    the date. A model without a checked price is reported as "unpriced" instead of
    getting a guessed cost.
  - `gw cost` shows tokens and spend per step and in total.

## Quick start

Requires Python 3.11+.

```bash
git clone https://github.com/nazish-qamar/german-wiki.git
cd german-wiki
pip install -e .

cp .env.example .env        # then put your key in it: ZAI_API_KEY=...
```

Extraction and merging use Z.AI's GLM models by default, with DeepSeek as a fallback
(set `DEEPSEEK_API_KEY` too if you want it). Both are configured per pipeline step in
[`config/models.yaml`](config/models.yaml). Embeddings always run **locally**
(`multilingual-e5-small`, downloaded on first use). Your notes are never sent to an
embedding API.

Build the local index once. It's derived from `nodes/` and can be rebuilt at any time:

```bash
gw reindex
gw embed
```

Then feed it something you're studying:

```bash
gw ingest -f notes.txt          # also accepts .png/.jpg (OCR, shown to you to correct first) and .pdf
gw queue                        # what was extracted, waiting for review
gw adjudicate <source-id>       # compare each candidate against existing notes and propose create / merge / link
gw review                       # approve or reject each proposal; this is the only step that writes to nodes/
```

You can review a proposal by hand before approving it: it's a Markdown file in
`proposals/`, and whatever it says when you approve is what gets written.

Other commands: `gw list`, `gw grid`, `gw gaps`, `gw families`, `gw anki`, and
`gw cost` (token usage and spend so far). `gw --help` lists everything.

**Cost:** `gw cost --by step` shows what you've spent so far. See
[Engineering highlights](#engineering-highlights) for how routing and caching keep it low.

## Example

Input: a short text about a working day in an office, which ends with:

> Viele Verben mit Präposition sind hier wichtig: teilnehmen an (+ Dativ), denken an
> (+ Akkusativ), warten auf (+ Akkusativ).

`gw ingest` → `gw adjudicate` → `gw review` produced this new note,
[`nodes/verben-mit-präpositionen.md`](nodes/verben-mit-präpositionen.md):

```markdown
---
id: verben-mit-präpositionen
title_de: Verben mit Präpositionen
title_en: Verbs with specific prepositions
type: grammar
cefr: B1
cefr_basis: llm:tiebreak(B1); grammar:akkusativ(A2,body); lexical:none
register:
- büro
- formell
themes:
- büro
source_ids:
- 20260726-test-buero-90458c3d
confidence: 0.9
status: draft
---

Some German verbs require specific prepositions, which in turn determine the case of
the following noun. The text highlights three important examples:

- teilnehmen an (+ Dativ): to participate in
- denken an (+ Akkusativ): to think of/about
- warten auf (+ Akkusativ): to wait for

## Examples
- Ich nehme an der Besprechung teil. (I participate in the meeting.)
- Ich denke an die neuen Projekte. (I think about the new projects.)
- Ich warte auf die Entscheidung vom Chef. (I'm waiting for the decision from the boss.)
```

Things to notice:

- **Everything comes from the source.** The three verbs and their cases are the ones the
  text listed, and the examples are its own sentences rewritten in first-person word
  order. Merges are held to the same rule: they may not add information that isn't in
  the sources.
- **`source_ids` points back to the original text,** which is kept unchanged so any note
  can be checked against what it came from. Source texts are kept locally in `raw/`,
  which is gitignored, so the one for this example isn't in the repo.
- **`cefr_basis` explains the level.** It's computed from grammar rules and wordlists
  first. The model is only asked when those disagree or have nothing to say
  (`llm:tiebreak`).
- **`status: draft`**: it was written by the pipeline and approved, but you haven't
  edited it yet. The hand-written seed notes are `stable`.

The same text also contained the polite-request phrases (*Könnten Sie mir bitte
helfen?*). The wiki already had [`um-hilfe-bitten`](nodes/um-hilfe-bitten.md), so
adjudication proposed a **merge** instead of a duplicate note. When a merge is approved,
the losing note is archived in `_merged/` with a record of the decision rather than
deleted.

## How it fits together

```
text / image / PDF
   │  gw ingest        extract concepts (LLM, cached)  →  raw/ (immutable) + queue/
   ▼
candidates
   │  gw adjudicate    local embeddings find neighbours, LLM judges same / overlap / related / new
   ▼
proposals/
   │  gw review        you approve or reject  ← the only door into nodes/
   ▼
nodes/*.md  ──►  SQLite + sqlite-vec index (derived; `gw reindex` rebuilds it)
            ──►  grid / gaps / Anki export
```

- **`nodes/` is the source of truth.** The SQLite index is only a cache of it.
- **One model interface** using the OpenAI-compatible client. Provider and model are set
  per step in config, so switching providers is a config change, not a code change.
- Built with Python, Typer, Pydantic v2, LangGraph, sentence-transformers, SQLite +
  sqlite-vec and pytest.

The full design is in [`docs/SPEC.md`](docs/SPEC.md), and the reasoning behind each
decision is in [`docs/decisions.md`](docs/decisions.md).

## Development

```bash
pip install -e ".[dev]"
pytest
```

Tests that call a real model API are skipped unless `GW_LIVE_TESTS=1` is set. Tests that
load the local embedding model need `GW_MODEL_TESTS=1`.

## License

[MIT](LICENSE)
