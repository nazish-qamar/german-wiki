"""PDF text layer → one source per page (SPEC §2's PDF input, §8.1's provenance).

**A PDF is two input types wearing one extension.**

A *text-based* PDF already contains its text; ``pypdf`` reads it losslessly and it goes
straight into the existing extraction path with **no vision call and no cost**. Most
textbook and Goethe material is this. Sending it to a vision model would be slower, more
expensive and strictly *worse* -- OCR can misread what is sitting there as data.

A *scanned* PDF has no text layer, only pictures of pages. Reading it needs rasterization,
which needs PyMuPDF (a ~20 MB binary wheel) or poppler (a system dependency). Neither is
taken on: the PDF is **refused with an actionable message** pointing at the image path
slice 8 also builds, so scanned material is reachable in one manual step rather than being
impossible (ADR-015).

**Every page is its own source.** Two independent reasons, either sufficient:

- SPEC §8.1 says provenance points at the "source image/**page**" -- a node citing a
  40-page PDF cannot be checked against anything; one citing page 12 can.
- The extraction cap is *per source*. SPEC §2 and ADR-006 cap a source at 5-8 candidates,
  and ADR-006's guardrail keeps the first 8 and warns. A 20-page chapter as one source
  would therefore either silently drop most of the chapter or squeeze it into 8 thin
  nodes -- the atomizing failure the cap exists to catch, inverted.

**A page with no text is flagged, never ingested silently.** An empty source looks like a
successful ingest while the content is quietly lost, which is the worst of both outcomes.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ..logutil import get_logger

logger = get_logger(__name__)

PDF_SUFFIX = ".pdf"

# --- The dials. Both are re-exported from `ingest` (`embed.GRAY_LOW`'s convention). ---
#
# **Both were guessed, not measured.** They were chosen before this ran on a single real
# German textbook PDF, so unlike SPEC's own numbers they carry no evidence behind them.
# Expect to move them; that is why they are named, commented and on the public surface
# rather than inline literals somewhere in `extract_pages`.

# Below this a "text layer" is noise -- a stray page number, a scanner watermark, a
# ligature artifact -- not content. Tuned low on purpose: the cost of treating a thin
# page as scanned is one clear refusal, while the cost of ingesting noise as if it were
# a page of German is a source that produces nonsense candidates.
#
# **Watch for legitimately thin pages**: chapter dividers, image-heavy layouts, exercise
# pages that are mostly blanks to fill in. Those are real content pages carrying little
# text, and they will land in `empty_pages` here -- harmless on its own (they are only
# skipped and reported), but see the ratio below for where it does bite.
MIN_PAGE_CHARS = 40

# What fraction of pages must carry text before the document counts as text-based rather
# than scanned. A mixed PDF (a scanned plate inside a typeset chapter) is common and
# should not be refused wholesale -- the empty pages are flagged instead.
#
# **This is where thin pages actually hurt.** A workbook that is half exercise pages
# could tip below the ratio and be refused *entirely*, even though its prose pages read
# fine. If that happens the answer is to lower this (or MIN_PAGE_CHARS), not to work
# around it -- a wrongly refused document is visible and recoverable, which is the
# failure direction worth erring toward.
MIN_TEXT_PAGE_RATIO = 0.5


class PdfError(RuntimeError):
    """The PDF cannot be ingested as-is. The message says what to do about it."""


class PdfPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    number: int  # 1-based, matching what a reader sees
    text: str

    @property
    def has_text(self) -> bool:
        return len(self.text.strip()) >= MIN_PAGE_CHARS


class PdfExtraction(BaseModel):
    """Every page of one document, with the empty ones called out."""

    model_config = ConfigDict(extra="forbid")

    pages: list[PdfPage] = Field(default_factory=list)

    @property
    def with_text(self) -> list[PdfPage]:
        return [p for p in self.pages if p.has_text]

    @property
    def empty_pages(self) -> list[int]:
        """Page numbers that yielded nothing -- probably scanned, possibly just blank."""
        return [p.number for p in self.pages if not p.has_text]

    @property
    def looks_scanned(self) -> bool:
        if not self.pages:
            return True
        return len(self.with_text) / len(self.pages) < MIN_TEXT_PAGE_RATIO


def is_pdf(path: Path | str) -> bool:
    return Path(path).suffix.lower() == PDF_SUFFIX


SCANNED_MESSAGE = (
    "{name} has no usable text layer — it looks scanned.\n"
    "\n"
    "Reading scanned pages needs rasterization, which this project deliberately has not "
    "taken on (a ~20 MB binary wheel or a system dependency, for a case that may be "
    "rare). Export the pages as images and ingest those — the image path runs OCR:\n"
    "\n"
    "    gw ingest -f seite-01.png\n"
)


def extract_pages(path: Path | str) -> PdfExtraction:
    """Read the text layer page by page. No model call, no cost.

    Raises ``PdfError`` when the file is unreadable or carries no usable text layer.
    """
    path = Path(path)
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - declared in pyproject
        raise PdfError(f"pypdf is required to read {path.name}: {exc}") from exc

    try:
        reader = PdfReader(str(path))
        pages = [
            PdfPage(number=index, text=(page.extract_text() or ""))
            for index, page in enumerate(reader.pages, start=1)
        ]
    except PdfError:
        raise
    except Exception as exc:
        raise PdfError(f"could not read {path.name} as a PDF: {exc}") from exc

    extraction = PdfExtraction(pages=pages)
    if extraction.looks_scanned:
        raise PdfError(SCANNED_MESSAGE.format(name=path.name))

    if extraction.empty_pages:
        # Loud, because the alternative is a source that ingested "successfully" while
        # part of the document went missing.
        logger.warning(
            "%s: %d page(s) yielded no text and were skipped — probably scanned: %s. "
            "Export those pages as images and ingest them separately.",
            path.name,
            len(extraction.empty_pages),
            ", ".join(str(n) for n in extraction.empty_pages),
        )
    return extraction
