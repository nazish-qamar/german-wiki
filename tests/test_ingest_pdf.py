"""PDF text layer: what is read, what is refused, and one source per page."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from german_wiki.ingest import _pdf, _raw
from german_wiki.ingest._pdf import PdfError, extract_pages

TODAY = date(2026, 8, 3)


def _make_pdf(path: Path, pages: list[str]) -> Path:
    """Write a real PDF whose pages carry actual text operators.

    Built byte by byte rather than with ``PdfWriter``, which can add blank pages but not
    author text content streams. The point is to exercise **pypdf's real extraction** --
    a stubbed reader would test nothing about the thing that can actually fail. Keeping
    the generator here rather than committing a binary fixture also means each test's
    input is readable in the test itself.
    """
    path.write_bytes(_pdf_bytes(pages))
    return path


def _pdf_bytes(pages: list[str]) -> bytes:
    """A minimal, valid multi-page PDF whose pages carry real text operators."""
    objects: list[bytes] = []

    def obj(body: str | bytes) -> int:
        objects.append(body.encode("latin-1") if isinstance(body, str) else body)
        return len(objects)

    font = obj("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids: list[int] = []
    for text in pages:
        escaped = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET" if text else ""
        content = obj(
            f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream".encode("latin-1")
        )
        page_ids.append(
            obj(
                f"<< /Type /Page /Parent 999 0 R /MediaBox [0 0 595 842] "
                f"/Resources << /Font << /F1 {font} 0 R >> >> /Contents {content} 0 R >>"
            )
        )
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    pages_id = obj(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>")
    catalog = obj(f"<< /Type /Catalog /Pages {pages_id} 0 R >>")

    objects[:] = [o.replace(b"999 0 R", f"{pages_id} 0 R".encode()) for o in objects]

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets[1:]:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog} 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n"
    ).encode()
    return bytes(out)


@pytest.fixture
def two_page_pdf(tmp_path: Path) -> Path:
    return _make_pdf(
        tmp_path / "kapitel3.pdf",
        ["Die Wechselpraepositionen stehen mit Akkusativ oder Dativ.",
         "Im Buero sagt man hoeflich Koennten Sie mir bitte helfen."],
    )


# --- reading the text layer ---


def test_a_text_layer_pdf_reads_without_any_model_call(two_page_pdf) -> None:
    """The whole point of this path: the text is already in the file, so it is free.

    OCR'ing it would be slower, cost money, and be *worse* -- vision can misread what is
    sitting there as data.
    """
    extraction = extract_pages(two_page_pdf)
    assert len(extraction.pages) == 2
    assert "Wechselpraepositionen" in extraction.pages[0].text
    assert "hoeflich" in extraction.pages[1].text


def test_pages_are_numbered_as_a_reader_sees_them(two_page_pdf) -> None:
    assert [p.number for p in extract_pages(two_page_pdf).pages] == [1, 2]


def test_an_unreadable_file_fails_clearly(tmp_path: Path) -> None:
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf at all")
    with pytest.raises(PdfError, match="could not read"):
        extract_pages(broken)


# --- scanned refusal ---


def test_a_pdf_with_no_text_layer_is_refused_with_the_next_step(tmp_path: Path) -> None:
    """Refused, not silently ingested empty -- and the message says what to do."""
    scanned = _make_pdf(tmp_path / "scan.pdf", ["", "", ""])
    with pytest.raises(PdfError) as exc:
        extract_pages(scanned)

    message = str(exc.value)
    assert "looks scanned" in message
    assert "gw ingest -f" in message  # points at the image path, which does OCR


def test_a_mostly_empty_pdf_is_treated_as_scanned(tmp_path: Path) -> None:
    mostly = _make_pdf(tmp_path / "mostly.pdf", ["Ein einziger Satz mit genug Text darin.", "", "", ""])
    with pytest.raises(PdfError, match="looks scanned"):
        extract_pages(mostly)


def test_a_partial_pdf_extracts_what_exists_and_flags_the_rest(tmp_path: Path, caplog) -> None:
    """The interesting middle case: a scanned plate inside a typeset chapter.

    Refusing the whole document would be wrong, and ingesting the empty page silently
    would look like success while its content vanished.
    """
    partial = _make_pdf(
        tmp_path / "gemischt.pdf",
        ["Erste Seite mit ausreichend Text fuer die Schwelle.",
         "",
         "Dritte Seite mit ausreichend Text fuer die Schwelle."],
    )
    extraction = extract_pages(partial)

    assert [p.number for p in extraction.with_text] == [1, 3]
    assert extraction.empty_pages == [2]


def test_a_noise_only_page_counts_as_empty(tmp_path: Path) -> None:
    """A stray page number is not a text layer."""
    noisy = _make_pdf(
        tmp_path / "noise.pdf",
        ["Eine richtige Seite mit genuegend Text fuer die Schwelle.", "42"],
    )
    assert extract_pages(noisy).empty_pages == [2]


# --- one source per page ---


def test_each_page_becomes_its_own_source(two_page_pdf, tmp_raw: Path) -> None:
    """SPEC §8.1 wants provenance at the page, and the extraction cap is per source --
    a 20-page chapter as one source would blow ADR-006's 5-8 cap and lose most of it."""
    extraction = extract_pages(two_page_pdf)
    ids = [
        _raw.resolve_source_id(
            page.text, two_page_pdf, raw_dir=tmp_raw, today=TODAY, page=page.number
        )[0]
        for page in extraction.with_text
    ]

    assert len(set(ids)) == 2
    assert "-p1-" in ids[0] and "-p2-" in ids[1]
    assert all(sid.startswith("20260803-kapitel3-") for sid in ids)


def test_two_identical_pages_still_get_distinct_ids(tmp_raw: Path) -> None:
    """The page marker earning its place: identical pages hash the same.

    Content hashing alone would collide them, and the second page would silently resolve
    to the first's id -- looking like an already-ingested duplicate.
    """
    same = "Dieselbe Seite zweimal, mit genuegend Text."
    first, _ = _raw.resolve_source_id(same, Path("doppelt.pdf"), raw_dir=tmp_raw, today=TODAY, page=4)
    second, _ = _raw.resolve_source_id(same, Path("doppelt.pdf"), raw_dir=tmp_raw, today=TODAY, page=5)

    assert first != second
    assert first.split("-")[-1] == second.split("-")[-1]  # same content hash...
    assert "-p4-" in first and "-p5-" in second  # ...distinguished only by the page


def test_re_ingesting_the_same_page_is_detected(two_page_pdf, tmp_raw: Path) -> None:
    page = extract_pages(two_page_pdf).pages[0]
    first, is_new = _raw.resolve_source_id(
        page.text, two_page_pdf, raw_dir=tmp_raw, today=TODAY, page=1
    )
    assert is_new
    _raw.store_raw(page.text, first, raw_dir=tmp_raw)

    again, is_new_again = _raw.resolve_source_id(
        page.text, two_page_pdf, raw_dir=tmp_raw, today=TODAY, page=1
    )
    assert again == first
    assert is_new_again is False


def test_is_pdf_matches_on_suffix() -> None:
    assert _pdf.is_pdf("a.pdf") and _pdf.is_pdf(Path("A.PDF"))
    assert not _pdf.is_pdf("a.png")
