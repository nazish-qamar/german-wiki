"""The merge pipeline: gray-zone pairs in, reviewed decisions out (SPEC §11 slice 5).

This package is the **only** import surface for adjudication, merging and review;
internals are ``_``-prefixed, matching the ``llm``, ``ingest`` and ``embed`` packages.

Slice 4 detected duplicates and reported them. This slice decides what they *are*, and
ADR-010 is why there are four answers rather than SPEC §3.1's three: a pair flagged at
0.882 turned out to be a ``governs`` relation, not a redundancy. So:

- ``SAME``             -> fold into the existing node, keep provenance and new examples
- ``OVERLAP``          -> regenerate one canonical body (SPEC §4.1)
- ``DISTINCT_RELATED`` -> create the node **and** propose a typed edge (SPEC §4.2)
- ``DISTINCT``         -> create the node

**All four route through review**, links exactly as much as merges (ADR-010). The
propose pass cannot write: ``adjudicate`` ends at ``interrupt()`` and the graph has no
edge from there to any apply node. Approval enters the graph from the other side, and
every write goes through ``ingest.write_approved`` -- the slice-3 promote seam, still the
one door into ``/nodes``.

Two commands, two artifacts::

    gw adjudicate <source>   ->  proposals/*.md      (pending, gitignored)
    gw review                ->  nodes/, _merged/, logs/decisions.jsonl
"""

from __future__ import annotations

from ._adjudicate import Adjudication, AdjudicationError, adjudicate
from ._apply import ApplyError, ApplyResult, review_order
from ._graph import Context, ProposeResult, apply_decision, build_graph, propose_for_source
from ._ledger import Decision, LedgerUnreadable, decided_pairs, merge_count, read_all
from ._proposal import (
    FLAG_CAP,
    FLAG_LEDGER_MISSING,
    FLAG_LOW_CONFIDENCE,
    FLAG_UNSOURCED,
    Proposal,
    delete_proposal,
    list_proposals,
    load_proposal,
    now_iso,
    proposal_id,
    write_proposal,
)
from ._regenerate import MAX_REGENERATIONS, CapCheck, MergedBody, check_cap, regenerate

__all__ = [
    "FLAG_CAP",
    "FLAG_LEDGER_MISSING",
    "FLAG_LOW_CONFIDENCE",
    "FLAG_UNSOURCED",
    "MAX_REGENERATIONS",
    "Adjudication",
    "AdjudicationError",
    "ApplyError",
    "ApplyResult",
    "CapCheck",
    "Context",
    "Decision",
    "LedgerUnreadable",
    "MergedBody",
    "Proposal",
    "ProposeResult",
    "adjudicate",
    "apply_decision",
    "build_graph",
    "check_cap",
    "decided_pairs",
    "delete_proposal",
    "list_proposals",
    "load_proposal",
    "merge_count",
    "now_iso",
    "proposal_id",
    "propose_for_source",
    "read_all",
    "regenerate",
    "review_order",
    "write_proposal",
]
