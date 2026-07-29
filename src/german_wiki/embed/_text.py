"""Text preparation for the two detection tiers (SPEC §3.1).

**Two normalizations live here, for two different jobs. Do not conflate them.**

``normalize_for_hash`` is aggressive -- NFKC, lowercase, collapsed whitespace --
because tier 1 asks "is this literally the same text?", where casing and spacing
are noise.

``embed_text`` preserves natural case, because tier 2 asks "does this mean the
same thing?", and German capitalizes nouns: lowercasing throws away a signal the
model was trained on.

Tier-1 near-exact detection is **exact Jaccard over character shingles, not
MinHash** -- a deliberate departure from SPEC §3.1's wording, recorded in ADR-010.
MinHash approximates Jaccard to stay sublinear at a scale SPEC §3.3 says this
corpus will not reach (low tens of thousands of nodes). Computing the real
similarity is directly affordable here and avoids both LSH tuning and
approximation error.
"""

from __future__ import annotations

import re
import unicodedata

from ..models import Node

# Character n-gram width. 5 is small enough to survive word-level edits and large
# enough that unrelated German prose does not share many shingles by chance.
SHINGLE_SIZE = 5

# multilingual-e5 requires a "query: " or "passage: " prefix; the model's own
# guidance is to use "query: " on BOTH sides for symmetric similarity, which is
# what node-to-node comparison is. Omitting it measurably degrades results, so it
# is applied here rather than left to each call site to remember.
E5_PREFIX = "query: "


def normalize_for_hash(text: str) -> str:
    """NFKC + lowercase + collapsed whitespace. Tier-1 identity only."""
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def strip_markdown(text: str) -> str:
    """Drop the scaffolding every node shares, keeping the prose.

    Headings (``## Examples``), table rows, bullet markers, emphasis and the
    ``[alltag, gesprochen]`` register tags are structure, not meaning. Every node
    carries them, so they raise the similarity floor between *unrelated* nodes --
    see ``embed_text`` for the measurements.
    """
    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "|")):
            continue
        if set(stripped) <= set("-|: "):  # table rules and horizontal lines
            continue
        stripped = re.sub(r"^[-*+]\s+", "", stripped)  # bullet markers
        stripped = re.sub(r"\*\*(.+?)\*\*", r"\1", stripped)  # bold
        stripped = re.sub(r"\*(.+?)\*", r"\1", stripped)  # italic
        stripped = re.sub(r"`(.+?)`", r"\1", stripped)  # inline code
        stripped = re.sub(r"\[(.+?)\]", "", stripped)  # register tags
        kept.append(stripped.strip())
    return " ".join(part for part in kept if part)


def embed_text(node: Node) -> str:
    """The exact string handed to the embedding model, natural case preserved.

    **German title plus the de-scaffolded body -- deliberately NOT the English
    title, and deliberately not truncated.** Measured on the seed corpus plus a
    real ingested pair, as the margin between the weakest true duplicate and the
    strongest unrelated pair (bigger is more room for a threshold):

    ==========================================  =========
    variant                                      margin
    ==========================================  =========
    German + English title, full raw body        +0.0378
    German title only, full raw body             +0.0298
    German + English title, stripped body        +0.0480
    **German title only, stripped body**         **+0.0559**
    German title only, stripped and truncated    +0.018..+0.037
    German title alone                           +0.0175
    ==========================================  =========

    Two lessons, both counter-intuitive enough to be worth recording. Stripping
    Markdown nearly doubles the margin, because multilingual-e5 otherwise spends
    capacity on the ``## Examples`` heading and table pipes that *every* node has.
    And appending the English title *hurts*: the German-English pairing is itself
    a shared structure, so it pulls unrelated nodes together. Truncating hurts too
    -- the body carries real signal, so all of it is kept.

    Consequence for later slices: the vector no longer contains the English title,
    so an English query would match less well. Fine here (dedup compares German
    node against German node); slice 9's RAG chat may want a different text for
    its own index.
    """
    body = strip_markdown(node.body_md)
    title = node.title_de.strip()
    return f"{E5_PREFIX}{title}. {body}".strip() if body else f"{E5_PREFIX}{title}"


def shingles(text: str, size: int = SHINGLE_SIZE) -> set[str]:
    """Character n-grams over normalized text."""
    normalized = normalize_for_hash(text)
    if len(normalized) <= size:
        return {normalized} if normalized else set()
    return {normalized[i : i + size] for i in range(len(normalized) - size + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    """True |A∩B| / |A∪B| -- not an estimate."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    return intersection / (len(a) + len(b) - intersection)


def could_reach(a: set[str], b: set[str], threshold: float) -> bool:
    """Cheap upper-bound prefilter: can this pair possibly reach ``threshold``?

    ``J(A,B) = |A∩B| / |A∪B| ≤ min(|A|,|B|) / max(|A|,|B|)``, because the
    intersection cannot exceed the smaller set and the union cannot be smaller
    than the larger one. So a size ratio below the threshold rules the pair out
    without touching the sets.

    Sound by construction: it can only skip pairs whose true Jaccard is already
    below the threshold, never one that would have passed.
    """
    if not a or not b:
        return False
    smaller, larger = sorted((len(a), len(b)))
    return smaller / larger >= threshold
