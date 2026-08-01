# CEFR wordlists — the lexical anchor (SPEC §5, signal 1)

This directory is **deliberately empty of data** on a fresh clone. The code that reads it
ships; the lists themselves do not.

## Format

One lemma per line, lowercase, UTF-8. Blank lines and `#` comments are ignored.
Files are named for the level they define:

```
a1.txt  a2.txt  b1.txt  b2.txt  c1.txt  c2.txt
```

A lemma is looked up in level order, lowest first, so a word appearing in both `a1.txt`
and `b1.txt` resolves to A1 — you learn it at the earlier level.

Example (`a1.txt`):

```
# Goethe-Institut A1 Wortliste
waschen
haus
trinken
```

## Why the data is not in git

Real wordlists — Goethe-Institut's A1/A2/B1 Wortlisten, DWDS frequency classes, or a
frequency dictionary — carry licensing this repo cannot redistribute. `.gitignore` excludes
`vocab/cefr/*.txt` so anything you drop here stays local and untracked. This README is
tracked, which is also what keeps the directory present on a fresh clone.

**Check it worked:** after copying a list in, `git status --short vocab/` must print
nothing.

## Where to get one

- **Goethe-Institut** publishes official A1 / A2 / B1 Wortlisten as PDFs on goethe.de.
  Extract the lemma column; the level is the file you save it as.
- **DWDS** (dwds.de) publishes frequency classes; SPEC §5 notes frequency rank is a strong
  CEFR proxy for levels above B1, where no official list exists.

## What happens without it

The lexical anchor contributes **nothing** and never errors — a missing directory, a
missing file, and an empty file are all "no signal". Levels then come from the grammar map
(SPEC §5, signal 2) and, where that is silent, the LLM tiebreak (signal 3).

That matters for one category in particular: **pure-vocabulary nodes have no grammar
structure to match**, so until a list exists they fall through to the tiebreak and are the
least-grounded levels in the wiki. They are findable:

```bash
grep -l 'cefr_basis: llm:tiebreak' nodes/
```

Dropping a real list here and re-running `gw relevel` anchors them, with no code change.
