"""The immutable raw store: source ids and provenance (SPEC §1.2, §1.3).

``/raw`` holds the extracted text exactly as ingested. It is append-only and
authoritative for provenance: a node's ``body_md`` is a *derived view*, and when a
merge drifts (SPEC §12.1) the raw text is what you re-verify against. So the
``.txt`` is written byte-for-byte with no normalization, and every piece of
metadata lives in a sidecar rather than polluting it.

Two files per source, sharing a stem::

    raw/20260726-notes-a1b2c3d4.txt     verbatim source bytes
    raw/20260726-notes-a1b2c3d4.json    metadata

The stem is the ``source_id`` written into each node's ``source_ids``.

Ordering is deliberate: the ``.txt`` is written before extraction runs, the
sidecar after it succeeds. The raw record is never contingent on the model
working. A ``.txt`` with no sidecar is an incomplete ingest -- detectable, and
free to re-run because the model call is cached (ADR-005).
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from .. import config

# Characters German readers expect to survive as digraphs rather than be stripped.
_TRANSLITERATIONS = {
    "ä": "ae",
    "ö": "oe",
    "ü": "ue",
    "ß": "ss",
    "Ä": "Ae",
    "Ö": "Oe",
    "Ü": "Ue",
}

# Prefix lengths tried when a short hash collides. Practically never past the first.
_HASH_WIDTHS = (8, 12, 16, 24, 32, 64)

SIDECAR_KEYS = frozenset(
    {
        "source_id",
        "original_path",
        "content_sha256",
        "ingested_at",
        "chars",
        "model",
        "provider",
        "candidate_count",
    }
)


def content_hash(text: str) -> str:
    """Full sha256 of the source text -- the real identity of a source.

    Slice 5 queries the stored copy of this for exact-duplicate detection
    (SPEC §3.1 tier 1, the free tier that catches copy-paste before any embedding
    or LLM call fires). That is why the sidecar keeps all 64 characters while the
    filename carries only a short prefix: the prefix is a human-readable handle,
    this is the key. Truncating the stored value would mean re-reading every raw
    file to backfill it later.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _slug(text: str, *, max_length: int, fallback: str) -> str:
    """Shared core: lowercase, non-alphanumerics to hyphens, collapse and trim.

    ``str.isalnum()`` is already true for ``ä``/``ö``/``ü``/``ß``, so what separates the
    two public slug functions below is only what they do *before* this runs.
    """
    out = [ch if ch.isalnum() else "-" for ch in text.lower()]
    slug = "-".join(part for part in "".join(out).split("-") if part)
    return slug[:max_length].strip("-") or fallback


def slugify(text: str, *, max_length: int = 40, fallback: str = "quelle") -> str:
    """Lowercase **ASCII** slug, German-aware (ä->ae, ß->ss). For SOURCE ids only.

    Source ids are opaque machine provenance handles -- they carry a content hash and
    name a file in ``/raw``, which SPEC §1.2 makes immutable and append-only. Nobody
    reads them as German, nobody renames them, and keeping them ASCII avoids ever
    having to (ADR-012).

    Node ids took the opposite decision and use ``node_slug``; see ADR-012 for why the
    two identifier types deliberately differ.
    """
    for source, replacement in _TRANSLITERATIONS.items():
        text = text.replace(source, replacement)
    # Strip any remaining diacritics rather than dropping the letters entirely.
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return _slug(text, max_length=max_length, fallback=fallback)


def node_slug(text: str, *, max_length: int = 60, fallback: str = "konzept") -> str:
    """Lowercase slug preserving real German. For NODE ids (ADR-012).

    ``Wechselpräpositionen`` -> ``wechselpräpositionen``, not ``wechselpraepositionen``.
    A node id is human-facing: it is the filename Obsidian shows as the note's name in
    the sidebar and graph, the value that appears in ``links: target:``, and what you
    type into ``gw review``. Transliterating it loses the actual word for no benefit in
    a UTF-8 toolchain.

    **NFC normalization is the load-bearing line here.** ``ä`` has two encodings --
    precomposed U+00E4 and decomposed U+0061 U+0308 -- which are indistinguishable on
    screen but are different bytes, hence different filenames. macOS stores filenames
    NFD while Linux and Windows use NFC, so without this a title arriving in one form
    would spawn a second node for a word that already has one. That is exactly the
    fragmentation ADR-006 exists to prevent, and it is the reason ASCII slugs were
    defensible in the first place; normalizing removes the hazard rather than dodging it.
    """
    return _slug(
        unicodedata.normalize("NFC", text), max_length=max_length, fallback=fallback
    )


def raw_paths(source_id: str, *, raw_dir: Path | str | None = None) -> tuple[Path, Path]:
    """``(text_path, sidecar_path)`` for a source id."""
    root = Path(raw_dir) if raw_dir is not None else config.RAW_DIR
    return root / f"{source_id}.txt", root / f"{source_id}.json"


def resolve_source_id(
    text: str,
    original: Path | str,
    *,
    raw_dir: Path | str | None = None,
    today: date | None = None,
) -> tuple[str, bool]:
    """Return ``(source_id, is_new)`` for this content.

    The id is ``<YYYYMMDD>-<slug of filename>-<content hash prefix>``. Hashing the
    *content* rather than the filename means re-ingesting the same text under any
    name resolves to the same id, so duplicates are detectable.

    A short-hash collision -- two different sources sharing a date, a slug *and* a
    hash prefix -- extends the prefix instead of overwriting. The comparison is
    against the stored bytes, not the sidecar, so it still works when a previous
    extraction failed and left a ``.txt`` with no sidecar.
    """
    stamp = (today or datetime.now(UTC).date()).strftime("%Y%m%d")
    stem = slugify(Path(original).stem)
    digest = content_hash(text)

    for width in _HASH_WIDTHS:
        candidate = f"{stamp}-{stem}-{digest[:width]}"
        text_path, _ = raw_paths(candidate, raw_dir=raw_dir)
        if not text_path.exists():
            return candidate, True
        if hashlib.sha256(text_path.read_bytes()).hexdigest() == digest:
            return candidate, False  # same content, already ingested
        # Different content behind the same prefix: widen and try again.

    # Unreachable: an equal full digest means equal content, handled above.
    raise ValueError(f"could not resolve a free source id for {original}")


def read_source(path: Path | str) -> str:
    """Read an input file as UTF-8, failing loudly rather than mangling bytes."""
    path = Path(path)
    try:
        return path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path} is not valid UTF-8 text: {exc}") from exc


def store_raw(text: str, source_id: str, *, raw_dir: Path | str | None = None) -> Path:
    """Write the verbatim source text. Immutable: never overwrites an existing file."""
    text_path, _ = raw_paths(source_id, raw_dir=raw_dir)
    if text_path.exists():
        return text_path
    text_path.parent.mkdir(parents=True, exist_ok=True)
    # write_bytes, not write_text: no newline translation on Windows, so the file
    # is byte-identical to what was ingested.
    text_path.write_bytes(text.encode("utf-8"))
    return text_path


def write_sidecar(
    source_id: str,
    *,
    text: str,
    original: Path | str,
    model: str,
    provider: str,
    candidate_count: int,
    raw_dir: Path | str | None = None,
    ingested_at: datetime | None = None,
) -> Path:
    """Write the provenance sidecar. Called only after a successful extraction."""
    _, sidecar_path = raw_paths(source_id, raw_dir=raw_dir)
    meta: dict[str, Any] = {
        "source_id": source_id,
        "original_path": str(original),
        "content_sha256": content_hash(text),  # full digest: slice-5 dedup key
        "ingested_at": (ingested_at or datetime.now(UTC)).isoformat(),
        "chars": len(text),
        "model": model,
        "provider": provider,
        "candidate_count": candidate_count,
    }
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return sidecar_path


def read_sidecar(source_id: str, *, raw_dir: Path | str | None = None) -> dict[str, Any] | None:
    """The sidecar for a source, or ``None`` if the ingest never completed."""
    _, sidecar_path = raw_paths(source_id, raw_dir=raw_dir)
    if not sidecar_path.is_file():
        return None
    return json.loads(sidecar_path.read_text(encoding="utf-8"))


def read_raw_text(source_id: str, *, raw_dir: Path | str | None = None) -> str | None:
    """The verbatim source text behind a node, or ``None`` if it is not on disk.

    This is the read side of SPEC §12.1's re-verification anchor: a node body is a
    derived view, and ``/raw`` is what you check it against when a merge looks like it
    has drifted. Slice 5's merge guard uses it to ask whether a regenerated example
    sentence actually came from somewhere. Read-only -- nothing outside ingestion ever
    writes to ``/raw``.
    """
    text_path, _ = raw_paths(source_id, raw_dir=raw_dir)
    if not text_path.is_file():
        return None
    return text_path.read_bytes().decode("utf-8")
