"""Combining the three signals into a level and a ``cefr_basis`` (SPEC §5).

SPEC §5's ordering is a **precedence rule, not a vote**: rules first, and the LLM only
"when they conflict or are absent". This module is where that is enforced, so the tiebreak
cannot quietly become the default path.

The decision table, in order:

===========================  ======================  ================================
grammar                      lexical                 result
===========================  ======================  ================================
title-anchored               absent, or agrees       grammar decides. No model call.
title-anchored               present and DISAGREES   tiebreak
body-only                    agrees                  lexical decides (grammar corroborates)
body-only                    absent, agrees w/ current  keep current level, record basis
body-only                    absent, differs from current  **tiebreak**
absent                       present                 lexical decides. No model call.
absent                       absent                  tiebreak
===========================  ======================  ================================

The fifth row is the one that earns its keep. ``verben-mit-präpositionen`` is B1 and its
body mentions *Akkusativ*; treating that body hit as authoritative would relabel it A2 and
push a B1 rule into beginner material. Sending it to the tiebreak instead costs one free
call and keeps a human in the loop, because ``cefr`` drives SPEC §5.1's study order.

``cefr_basis`` is written as ``signal:detail(LEVEL)``, semicolons between contributing
signals — formalizing the loose convention the hand-authored seeds already use
(``grammar:wechselpraeposition``, ``freq:high; goethe:A1(waschen)``). ``llm:tiebreak``
stays greppable on purpose (ADR-009's habit, one slice on).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ..llm import ChatClient
from ..logutil import get_logger
from ..models import CEFR, Node
from . import _grammar, _lexical, _tiebreak
from ._grammar import GrammarHit

logger = get_logger(__name__)

# The marker that says "this level is not rules-grounded". Successor to slice 3's
# PROVISIONAL_CEFR ("llm:extraction"); kept greppable for the same reason.
TIEBREAK_MARKER = "llm:tiebreak"

# Recorded when a node's level was set by a human and the rules have nothing to say about
# it. It explains the level; it never replaces one.
HUMAN_SEED_MARKER = "human:seed"

# Bases beginning with any of these are machine placeholders that slice 6 exists to
# replace. A hand-authored basis matches none of them and is left alone.
PLACEHOLDER_PREFIXES = ("llm:extraction",)


class LevelResult(BaseModel):
    """A derived level, with the evidence that produced it."""

    model_config = ConfigDict(extra="forbid")

    cefr: CEFR | None  # None == undecidable without a model call that was not allowed
    basis: str
    used_tiebreak: bool = False
    grammar_summary: str = "none"
    lexical_summary: str = "none"

    @property
    def rules_grounded(self) -> bool:
        return self.cefr is not None and not self.used_tiebreak


def _grammar_summary(hits: list[GrammarHit], decided: tuple[CEFR, list[GrammarHit]] | None) -> str:
    if not hits or decided is None:
        return "no structure from the syllabus map matched"
    level, winners = decided
    where = winners[0].where
    names = ", ".join(sorted({h.structure for h in winners}))
    return f"{names} ({level}) matched in the {where.upper()}"


def _basis_grammar(decided: tuple[CEFR, list[GrammarHit]]) -> str:
    level, winners = decided
    names = "+".join(sorted({h.structure for h in winners}))
    suffix = "" if winners[0].where == "title" else ",body"
    return f"grammar:{names}({level}{suffix})"


def derive_level(
    node: Node,
    *,
    cefr_dir: Path | str | None = None,
    allow_llm: bool = True,
    client: ChatClient | None = None,
    settings_path: Path | str | None = None,
    cache_dir: Path | str | None = None,
    usage_log: Path | str | None = None,
    env: dict[str, str] | None = None,
) -> LevelResult:
    """Derive ``(cefr, cefr_basis)`` for one node. Writes nothing.

    ``allow_llm=False`` makes this pure rules: a case that would need the tiebreak returns
    ``cefr=None`` instead of calling out. That is what keeps the offline test suite honest
    — it cannot accidentally depend on a model — and it is also the mode a future
    ``--no-llm`` flag would use.
    """
    hits = _grammar.grammar_anchor(node.title_de, node.body_md)
    decided = _grammar.strongest(hits)
    title_anchored = _grammar.is_title_anchored(hits)

    lex = _lexical.lexical_anchor(node.title_de, lemmas=node.lemmas, cefr_dir=cefr_dir)
    lex_summary = (
        f"{lex.lemma} is {lex.level} in {lex.source}"
        if lex
        else ("no wordlist installed" if not _lexical.available(cefr_dir) else "not in any list")
    )
    gram_summary = _grammar_summary(hits, decided)

    def _result(cefr: CEFR | None, basis: str, used_tiebreak: bool = False) -> LevelResult:
        return LevelResult(
            cefr=cefr,
            basis=basis,
            used_tiebreak=used_tiebreak,
            grammar_summary=gram_summary,
            lexical_summary=lex_summary,
        )

    # --- rules first ---
    if decided is not None and title_anchored:
        level, _ = decided
        if lex is None or lex.level == level:
            basis = _basis_grammar(decided)
            if lex is not None:
                basis = f"{basis}; {lex.source}({lex.lemma})"
            return _result(level, basis)
        # Title grammar and the wordlist disagree -- exactly SPEC §5's "conflict".
        logger.info(
            "cefr signals conflict for %s: grammar says %s, %s says %s",
            node.id,
            level,
            lex.source,
            lex.level,
        )

    elif decided is not None:  # body-only hits: weak evidence
        level, _ = decided
        if lex is not None and lex.level == level:
            return _result(level, f"{lex.source}({lex.lemma}); {_basis_grammar(decided)}")
        if lex is None and level == node.cefr:
            # Corroborates what the node already claims, so nothing is being changed on
            # the strength of a passing mention.
            return _result(level, _basis_grammar(decided))

    elif lex is not None:  # no grammar structure, but the word is listed
        return _result(lex.level, f"{lex.source}({lex.lemma})")

    # --- a human set this level and the rules have nothing to add ---
    #
    # "No basis" and "a machine-placeholder basis" are DIFFERENT FACTS, and conflating
    # them inverts the point of the slice. A missing basis means *nobody recorded why this
    # level was chosen*; it says nothing about whether the level is right. When both
    # anchors are silent the tiebreak has strictly LESS information than whoever set the
    # level, so letting it move the level would overwrite a human judgment because the
    # explanation field happened to be empty.
    #
    # Caught on the live corpus: `prefix-an` is a hand-authored A2 seed with no basis, and
    # an ungrounded tiebreak proposed A2 -> B1 on `grammar:none; lexical:none`.
    #
    # So here the tiebreak *explains* rather than *moves*. A placeholder basis
    # (`llm:extraction`) does not qualify -- that level was itself a machine guess, so
    # there is no human judgment to protect.
    if decided is None and lex is None and is_absent(node.cefr_basis):
        return _result(node.cefr, HUMAN_SEED_MARKER)

    # --- signals conflicted or were absent: SPEC §5 signal 3 ---
    if not allow_llm:
        return _result(None, "unresolved: rules gave no answer and the tiebreak was disabled")

    verdict, _response = _tiebreak.tiebreak(
        title_de=node.title_de,
        title_en=node.title_en,
        node_type=node.type,
        body_md=node.body_md,
        grammar=gram_summary,
        lexical=lex_summary,
        client=client,
        settings_path=settings_path,
        cache_dir=cache_dir,
        usage_log=usage_log,
        env=env,
    )
    # Record what the tiebreak was shown, not just what it said -- a level you cannot
    # audit is the thing SPEC §5 objects to about zero-shot CEFR in the first place.
    parts = [f"{TIEBREAK_MARKER}({verdict.cefr})"]
    parts.append(_basis_grammar(decided) if decided else "grammar:none")
    parts.append(f"{lex.source}({lex.lemma})" if lex else "lexical:none")
    return _result(verdict.cefr, "; ".join(parts), used_tiebreak=True)


def is_absent(cefr_basis: str | None) -> bool:
    """No basis was ever recorded — as opposed to a machine placeholder being recorded.

    The two are deliberately distinguished (see ``derive_level``): a placeholder means the
    *level* was a guess, while an absent basis means only the *explanation* is missing and
    the level may well be a human's.
    """
    return cefr_basis is None or not cefr_basis.strip()


def is_placeholder(cefr_basis: str | None) -> bool:
    """Whether a basis needs re-deriving at all.

    Absent counts, because SPEC §5 says "always store ``cefr_basis``" — a node without one
    has an unexplained level. But *targeting* a node is not the same as being allowed to
    move its level; ``derive_level`` draws that second line via ``is_absent``.
    """
    if is_absent(cefr_basis):
        return True
    return cefr_basis.strip().startswith(PLACEHOLDER_PREFIXES)
