"""Node model validation against the four seed files and their variance."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from german_wiki import storage
from german_wiki.models import Node

from conftest import SEED_IDS


@pytest.mark.parametrize("sid", SEED_IDS)
def test_seed_validates(seed_paths, sid):
    node = storage.load_node(seed_paths[sid])
    assert node.id == sid
    assert node.body_md.strip()  # body preserved


def test_prefix_an_absent_optionals(seed_nodes):
    """prefix-an omits cefr_basis/themes/confidence -> they parse as None."""
    n = seed_nodes["prefix-an"]
    assert n.cefr_basis is None
    assert n.themes is None
    assert n.confidence is None
    assert n.separable is True  # type-specific field that IS present
    assert n.family_transparency == "high"


def test_present_empty_vs_absent_themes(seed_nodes):
    """wechsel has `themes: []` (present-empty); prefix-an has none (absent)."""
    assert seed_nodes["wechselpraepositionen"].themes == []
    assert seed_nodes["prefix-an"].themes is None


def test_vocab_family_fields(seed_nodes):
    n = seed_nodes["familie-waschen"]
    assert n.root == "waschen"
    assert "abwaschen" in n.lemmas
    assert n.links and n.links[0].target


BASE = {
    "id": "x",
    "title_de": "X",
    "title_en": "X",
    "type": "vocab",
    "cefr": "A1",
    "status": "draft",
}


@pytest.mark.parametrize(
    "field,bad",
    [("type", "verb"), ("cefr", "A0"), ("status", "final"), ("family_transparency", "clear")],
)
def test_enum_rejects_bad_value(field, bad):
    with pytest.raises(ValidationError):
        Node.model_validate({**BASE, field: bad})


def test_extra_key_forbidden():
    """Unknown frontmatter *keys* raise (schema drift surfaces loudly)."""
    with pytest.raises(ValidationError):
        Node.model_validate({**BASE, "titel_de": "typo"})


def test_open_tag_values_do_not_raise():
    """register/themes accept arbitrary values — never enum-validated."""
    node = Node.model_validate(
        {**BASE, "register": ["some-new-register"], "themes": ["some-new-theme"]}
    )
    assert node.register == ["some-new-register"]
    assert node.themes == ["some-new-theme"]
