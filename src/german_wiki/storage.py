"""Read/write Concept Nodes as Markdown + YAML frontmatter (SPEC §1.2).

Files in ``/nodes`` are the source of truth. Reading is lossless into the ``Node``
model; writing emits a stable, unicode-preserving frontmatter block in SPEC §1.1
key order, omitting absent (``None``) fields so a file round-trips without gaining
fields it never had. ``register``/``themes`` values are normalized on write via
``vocab``. Byte-identical round-trip is not a goal; semantic round-trip is.

Nothing here auto-writes to ``/nodes`` — ``write_node`` is called only by tests and
(later) the reviewed pipeline. ``reindex``/``list`` never call it.
"""

from __future__ import annotations

from pathlib import Path

import frontmatter

from . import config, vocab
from .models import Node

# Frontmatter key order (SPEC §1.1). body_md is the Markdown body, not a key.
_FRONTMATTER_ORDER = [
    "id",
    "title_de",
    "title_en",
    "type",
    "cefr",
    "cefr_basis",
    "register",
    "themes",
    "separable",
    "family_transparency",
    "root",
    "lemmas",
    "examples",
    "links",
    "source_ids",
    "confidence",
    "status",
    "version",
    "updated_at",
]


def load_node(path: Path | str) -> Node:
    """Parse a single ``.md`` file into a ``Node``.

    Enforces SPEC §1.1: ``id`` must equal the filename stem.
    """
    path = Path(path)
    post = frontmatter.load(str(path))
    meta = dict(post.metadata)
    meta["body_md"] = post.content
    node = Node.model_validate(meta)
    if node.id != path.stem:
        raise ValueError(
            f"node id {node.id!r} does not match filename stem {path.stem!r} ({path})"
        )
    return node


def load_all_nodes(nodes_dir: Path | str | None = None) -> list[Node]:
    """Load every ``*.md`` node in ``nodes_dir``, sorted by id."""
    nodes_dir = Path(nodes_dir) if nodes_dir is not None else config.NODES_DIR
    nodes = [load_node(p) for p in sorted(nodes_dir.glob("*.md"))]
    return sorted(nodes, key=lambda n: n.id)


def node_to_frontmatter(node: Node, *, vocab_dir: Path | str | None = None) -> dict:
    """Build the ordered frontmatter mapping for ``node`` (absent fields omitted)."""
    meta: dict = {}
    meta["id"] = node.id
    meta["title_de"] = node.title_de
    meta["title_en"] = node.title_en
    meta["type"] = node.type
    meta["cefr"] = node.cefr

    if node.cefr_basis is not None:
        meta["cefr_basis"] = node.cefr_basis

    meta["register"] = [
        n for r in node.register if (n := vocab.normalize("register", r, vocab_dir=vocab_dir))
    ]
    if node.themes is not None:
        meta["themes"] = [
            n for t in node.themes if (n := vocab.normalize("themes", t, vocab_dir=vocab_dir))
        ]

    if node.separable is not None:
        meta["separable"] = node.separable
    if node.family_transparency is not None:
        meta["family_transparency"] = node.family_transparency
    if node.root is not None:
        meta["root"] = node.root
    if node.lemmas is not None:
        meta["lemmas"] = node.lemmas

    if node.examples is not None:
        meta["examples"] = [
            {k: v for k, v in ex.model_dump().items() if v is not None} for ex in node.examples
        ]
    if node.links:
        meta["links"] = [
            {k: v for k, v in link.model_dump().items() if v is not None} for link in node.links
        ]

    meta["source_ids"] = list(node.source_ids)
    if node.confidence is not None:
        meta["confidence"] = node.confidence
    meta["status"] = node.status
    if node.version is not None:
        meta["version"] = node.version
    if node.updated_at is not None:
        meta["updated_at"] = node.updated_at.isoformat()

    # Emit in canonical order (any stray key appended after, deterministically).
    ordered = {k: meta[k] for k in _FRONTMATTER_ORDER if k in meta}
    for k in meta:
        if k not in ordered:
            ordered[k] = meta[k]
    return ordered


def dumps_node(node: Node, *, vocab_dir: Path | str | None = None) -> str:
    """Serialize ``node`` to the full Markdown-with-frontmatter string."""
    meta = node_to_frontmatter(node, vocab_dir=vocab_dir)
    post = frontmatter.Post(node.body_md, **meta)
    text = frontmatter.dumps(
        post,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=1000,
    )
    return text.rstrip("\n") + "\n"


def write_node(node: Node, path: Path | str, *, vocab_dir: Path | str | None = None) -> None:
    """Write ``node`` to ``path`` (LF line endings, UTF-8)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_node(node, vocab_dir=vocab_dir), encoding="utf-8", newline="\n")
