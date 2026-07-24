"""Pydantic models for Concept Nodes (SPEC §1.1).

Design notes:
- ``type`` / ``cefr`` / ``status`` / ``family_transparency`` are strict enums:
  small, stable vocabularies where a typo should fail loudly.
- ``register`` and ``themes`` are open ``list[str]`` — discovered from material,
  normalized (not enum-validated) on write. See ``vocab.py``.
- ``extra='forbid'`` rejects unknown frontmatter *keys* so schema drift in a
  hand-authored file surfaces immediately. It does not constrain list *values*.
- Optional fields default to ``None`` = "absent", so serialization can omit them
  and round-trip a file without inventing fields. A present-but-empty list (e.g.
  ``themes: []``) stays ``[]`` and is preserved.
"""

from __future__ import annotations

import warnings
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

NodeType = Literal["grammar", "vocab", "phrase", "pattern", "culture"]
CEFR = Literal["A1", "A2", "B1", "B2", "C1", "C2"]
Status = Literal["draft", "reviewed", "stable"]
FamilyTransparency = Literal["high", "drifted", "opaque"]

# The `register` field on Example/Node intentionally shadows BaseModel.register
# (the ABCMeta virtual-subclass hook, which this project never uses). Silence the
# purely-cosmetic pydantic warning for that one field name.
warnings.filterwarnings("ignore", message=r'Field name "register".*', category=UserWarning)


class Link(BaseModel):
    """A typed, directed edge to another node (SPEC §4.2)."""

    model_config = ConfigDict(extra="forbid")

    target: str
    relation: str
    confidence: float | None = None


class Example(BaseModel):
    """A worked example sentence (SPEC §1.1).

    Present in the schema for completeness; in slice 1 examples live in the
    Markdown body (``body_md``) and are not parsed out into this structure.
    """

    model_config = ConfigDict(extra="forbid")

    de: str
    en: str | None = None
    source_id: str | None = None
    register: list[str] | None = None


class Node(BaseModel):
    """The atomic learnable unit (SPEC §1.1)."""

    model_config = ConfigDict(extra="forbid")

    # --- required (present in every seed) ---
    id: str
    title_de: str
    title_en: str
    type: NodeType
    cefr: CEFR
    status: Status

    # --- core multi-label fields (always emitted; may be empty) ---
    register: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    links: list[Link] = Field(default_factory=list)

    # --- optional; None == absent, omitted on write ---
    cefr_basis: str | None = None
    themes: list[str] | None = None
    confidence: float | None = None
    version: int | None = None
    updated_at: datetime | None = None
    examples: list[Example] | None = None

    # --- type-specific optionals ---
    separable: bool | None = None
    family_transparency: FamilyTransparency | None = None
    root: str | None = None
    lemmas: list[str] | None = None

    # --- the canonical Markdown explanation (the body, not frontmatter) ---
    body_md: str = ""
