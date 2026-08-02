"""The decision ledger: one JSONL record per human decision (ADR-011).

Written by ``_apply``, read by ``_regenerate`` for the SPEC §12.1 regeneration cap and
by the CLI for the "have I already decided this pair?" check. It lives in its own module
because two consumers need it and neither should import the other.

**This file is load-bearing, which makes it a different class of artifact from
``logs/llm_usage.jsonl``.** ADR-008 §5 tracks the cost ledger in git because spend
history is a durable record; losing it costs a statistic. This one is the *authoritative
regeneration count* behind the merge cap, so losing it would disarm a safety guard. Hence
two consequences:

- it is git-tracked (``.gitignore`` carries a second negation for it), so a wipe is
  recoverable with ``git restore``; and
- ``merge_count`` raises rather than returning ``0`` when it cannot read the file. A
  missing ledger must never be indistinguishable from "this node has never been merged"
  -- that is precisely the fail-open the cap exists to prevent. Deciding what to do about
  an unknown count is the caller's job (see ``_regenerate.check_cap``).

Same reasoning applies within the file: **one unparseable line poisons the whole read**
rather than being skipped. Skipping it would silently undercount, which lands back in the
same fail-open.

``Node.version`` is deliberately not used as the count. It is hand-editable and absent on
the seed nodes, so it cannot be authoritative -- but it makes a sound *tripwire*, because
a wrong value there can only ever cause a refusal.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .. import config

# The outcome that actually re-encodes a body. SAME appends provenance and new example
# lines mechanically -- no model call, no re-encoding -- so it is NOT a regeneration and
# does not count toward the cap. SPEC §12.1's concern is specifically that "every
# regeneration is a lossy re-encoding".
REGENERATING_OUTCOMES = frozenset({"OVERLAP"})

# `relevel` (slice 6) rewrites cefr + cefr_basis and nothing else. It is a proposal kind
# rather than its own command-with-a-write because ADR-003 gates writes to /nodes, not
# uncertain judgments -- and matching a node to a SPEC §5 grammar row is an interpretation
# that can be wrong, even though the level lookup that follows is a table read.
#
# `morphology` (slice 7) is the same shape one slice on: root / lemmas / separable /
# family_transparency, and nothing else. The transparency half is an outright model
# judgment (SPEC §7.4), so it could never have been anything but reviewed.
Kind = Literal["merge", "link", "create", "discard", "relevel", "morphology"]


class LedgerUnreadable(RuntimeError):
    """The ledger is missing or corrupt, so any count derived from it is unknown."""


class Decision(BaseModel):
    """One human decision, durable. Append-only; never rewritten."""

    model_config = ConfigDict(extra="forbid")

    decision_id: str
    proposal_id: str
    decided_at: str
    approved: bool
    kind: Kind
    outcome: str
    winner: str | None = None
    loser: str | None = None
    source_id: str | None = None
    similarity: float | None = None
    tier: str | None = None
    confidence: float | None = None
    relation: str | None = None
    direction: str | None = None
    # "llm" or "threshold" -- whether a model produced this verdict or the >= GRAY_HIGH
    # band did. A threshold verdict still went through review (ADR-003); it just cost
    # nothing to reach.
    basis: str = "llm"
    # Which model decided it. ADR-011: adjudication runs on the free glm-4.5-flash while
    # slice 5 is being tuned, and those verdicts are pipeline development rather than
    # trusted production merges -- so "which decisions came from flash?" stays a grep.
    provider: str | None = None
    model: str | None = None
    changelog: str | None = None
    flags: list[str] = Field(default_factory=list)


def _path(decisions_log: Path | str | None) -> Path:
    return Path(decisions_log) if decisions_log is not None else config.DECISIONS_LOG_PATH


def append(decision: Decision, *, decisions_log: Path | str | None = None) -> Path:
    """Append one record. Append-only, so diffs stay pure additions."""
    path = _path(decisions_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(decision.model_dump(), ensure_ascii=False) + "\n")
    return path


def read_all(*, decisions_log: Path | str | None = None) -> list[Decision]:
    """Every record, or raise ``LedgerUnreadable``.

    Raises on a missing file too. "Absent" and "empty" are different facts here: an
    empty ledger says nothing has been decided, a missing one says the record is gone.
    """
    path = _path(decisions_log)
    if not path.is_file():
        raise LedgerUnreadable(f"no decision ledger at {path}")

    decisions = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            decisions.append(Decision.model_validate(json.loads(line)))
        except Exception as exc:  # json or pydantic
            raise LedgerUnreadable(
                f"{path}:{lineno} is not a valid decision record: {exc}. Refusing to "
                "skip it -- a partial read would undercount and silently weaken the "
                "regeneration cap. Restore the file with `git restore`."
            ) from exc
    return decisions


def exists(*, decisions_log: Path | str | None = None) -> bool:
    return _path(decisions_log).is_file()


def merge_count(node_id: str, *, decisions_log: Path | str | None = None) -> int:
    """How many approved regenerations this node has survived (SPEC §12.1).

    Raises ``LedgerUnreadable`` rather than returning 0 when the ledger cannot be read.
    """
    return sum(
        1
        for d in read_all(decisions_log=decisions_log)
        if d.approved and d.outcome in REGENERATING_OUTCOMES and d.winner == node_id
    )


def decided_pairs(*, decisions_log: Path | str | None = None) -> set[tuple[str, str]]:
    """Ordered id pairs already decided, so a rejected pair is not re-proposed forever."""
    pairs = set()
    for d in read_all(decisions_log=decisions_log):
        if d.winner and d.loser:
            pairs.add((d.winner, d.loser) if d.winner <= d.loser else (d.loser, d.winner))
    return pairs


def new_decision_id(proposal_id: str, *, now: datetime | None = None) -> str:
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{proposal_id}"
