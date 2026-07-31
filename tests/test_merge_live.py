"""ADR-010's worked example, against a real provider. Skipped unless GW_LIVE_TESTS=1.

The rest of the merge suite is offline and free by construction. This file is the only
adjudication test that opens a socket.

    GW_LIVE_TESTS=1 pytest -m live

What it asserts is the finding that reshaped this slice. ``um-hilfe-bitten`` and
``verben-mit-präpositionen`` sit at cosine 0.882 -- squarely in the gray zone -- and
SPEC §3.1's three outcomes would force that pair to be answered as a redundancy question:
merge two unrelated concepts, or call them DISTINCT and lose the connection entirely. It
is neither. It is *bitten **um** + Akkusativ*, a verb-preposition combination, which
SPEC §4.2 calls a ``governs`` relation.

**The few-shot exemplars deliberately do not contain this pair** (asserted offline in
test_merge_adjudicate.py). Shot 3 teaches the same *shape* -- a fixed expression versus
the grammar rule it obeys -- with entirely different content, so passing here means the
model generalized rather than recalled an answer out of its own prompt.

Nothing is written: this calls ``adjudicate`` directly, which returns a verdict and has
no write path at all.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from german_wiki import config, storage
from german_wiki.llm import resolve_step
from german_wiki.merge import adjudicate

LIVE = os.environ.get("GW_LIVE_TESTS") == "1"

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not LIVE, reason="set GW_LIVE_TESTS=1 to run live model API tests"),
]

# Models whose verdicts are pipeline development rather than production merges.
#
# See ADR-011 §5 and its measured addendum for the evidence: flash got BOTH real
# gray-zone pairs wrong -- DISTINCT where the answer is `governs`, and `same_family`
# where SPEC §4.2 requires a shared stem and there is none. Recovering the first is the
# acceptance criterion for the glm-4.6 switch, so it is asserted strictly there and
# marked an expected failure here.
TUNING_MODELS = {"glm-4.5-flash"}


def _configured_model() -> str:
    return resolve_step("adjudication").model

# The A side is a seed node; the B side is the candidate slice 4 flagged against it.
#
# B is looked up in /nodes first and /queue second, because it legitimately lives in
# either: staged before review, promoted after. Pinning it to /queue alone meant this
# test silently started skipping the moment the candidate was approved -- and a gated
# test that always skips is a test that never runs, which is worse than a failing one
# because nothing reports it.
NODE_A = config.NODES_DIR / "um-hilfe-bitten.md"
NODE_B_NAME = "verben-mit-präpositionen.md"
B_LOCATIONS = [
    config.NODES_DIR / NODE_B_NAME,
    config.QUEUE_DIR / "20260726-test-buero-90458c3d" / NODE_B_NAME,
]


@pytest.fixture
def pair():
    if not NODE_A.is_file():
        pytest.skip(f"seed node missing: {NODE_A}")
    b = next((p for p in B_LOCATIONS if p.is_file()), None)
    if b is None:
        pytest.skip(
            f"{NODE_B_NAME} is in neither /nodes nor /queue (looked in "
            f"{[str(p) for p in B_LOCATIONS]}). Re-create it with "
            "`gw ingest -f test-büro.txt --force` (the extraction call is cached)."
        )
    return storage.load_node(NODE_A), storage.load_node(b)


def test_live_the_0882_pair_never_merges(pair, tmp_path: Path) -> None:
    """The floor, asserted on every model: this pair must not be folded together.

    Merging *Um Hilfe bitten* into *Verben mit Präpositionen* would corrupt two study
    notes at once, and it is the failure SPEC §3.1's three-outcome framing invites --
    with no DISTINCT_RELATED branch, a strong similarity score has nowhere to go except
    "merge". Whether the connection is additionally *recorded* is the next test's
    question; not merging is this one's, and it is not negotiable on any model.
    """
    a, b = pair
    verdict, response = adjudicate(
        a, b, cache_dir=tmp_path / "cache", usage_log=tmp_path / "usage.jsonl"
    )

    assert verdict.outcome in ("DISTINCT", "DISTINCT_RELATED"), (
        f"{verdict.outcome} would merge two unrelated concepts: {verdict.reason!r}"
    )
    assert response.finish_reason != "length"
    # A four-outcome answer with no confidence is a verdict you cannot triage.
    assert 0.0 <= verdict.confidence <= 1.0


def test_live_the_0882_pair_is_a_governs_relation(pair, tmp_path: Path, request) -> None:
    """ADR-010's worked example, and the acceptance criterion for the glm-4.6 switch.

    *bitten **um** + Akkusativ* is a verb-preposition combination, so the pair is a
    ``governs`` relation (SPEC §4.2) rather than either a duplicate or two unconnected
    concepts. Answering DISTINCT is not *wrong* so much as lossy: the wiki keeps both
    nodes but never records why they turned up next to each other, which is the
    fragmentation ADR-006 says to bias against.

    Marked xfail on a tuning model (ADR-011 §5) rather than weakened, so the assertion
    stays exactly as strict as it should be and simply starts passing when the model is
    switched. ``strict=True`` cuts both ways deliberately: if flash ever *does* get this
    right, the XPASS forces the ADR's claim about it to be revisited instead of quietly
    going stale.
    """
    model = _configured_model()
    if model in TUNING_MODELS:
        request.applymarker(
            pytest.mark.xfail(
                strict=True,
                # Only a WRONG VERDICT is the expected failure. Without `raises`, xfail
                # swallows every exception -- so a 429, a dropped connection or a
                # truncated response would all report XFAIL, and the test would look
                # like it confirmed the ADR's claim about flash while never actually
                # reaching an assertion. Same shape as the ledger's fail-safe read:
                # "unknown" must not be indistinguishable from "checked, as expected".
                raises=AssertionError,
                reason=(
                    f"{model} is the free tuning model (ADR-011 §5); it answers DISTINCT "
                    "here. Recovering the governs verdict is what the glm-4.6 switch buys."
                ),
            )
        )

    a, b = pair
    verdict, _ = adjudicate(
        a, b, cache_dir=tmp_path / "cache", usage_log=tmp_path / "usage.jsonl"
    )

    assert verdict.outcome == "DISTINCT_RELATED", (
        f"expected the ADR-010 finding, got {verdict.outcome}: {verdict.reason!r}"
    )
    assert verdict.relation == "governs", f"got relation {verdict.relation!r}: {verdict.reason!r}"
    assert verdict.direction in ("a_to_b", "b_to_a")


def test_live_an_obviously_unrelated_pair_is_distinct(pair, tmp_path: Path) -> None:
    """The negative control: without one, the test above passes on a model that always
    answers DISTINCT_RELATED."""
    a, _ = pair
    unrelated = storage.load_node(config.NODES_DIR / "familie-waschen.md")
    verdict, _ = adjudicate(
        a, unrelated, cache_dir=tmp_path / "cache", usage_log=tmp_path / "usage.jsonl"
    )
    assert verdict.outcome in ("DISTINCT", "DISTINCT_RELATED")
    assert verdict.outcome != "SAME"


def test_live_re_adjudicating_the_same_pair_is_free(pair, tmp_path: Path) -> None:
    """ADR-005 against a real provider, and the premise ADR-011 rests on.

    ``gw review`` can re-derive a proposal's context days later at zero cost, which is
    why the durable state is a file rather than a paused graph.
    """
    a, b = pair
    common = {"cache_dir": tmp_path / "cache", "usage_log": tmp_path / "usage.jsonl"}
    first_verdict, first = adjudicate(a, b, **common)
    second_verdict, second = adjudicate(a, b, **common)

    assert first.cached is False
    assert second.cached is True
    assert second_verdict == first_verdict
