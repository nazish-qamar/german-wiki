"""Ingestion orchestration: file -> raw -> extraction -> queue.

The whole flow is deliberately linear and writes nothing to ``/nodes``. Ordering
carries meaning:

1. Read and resolve the source id from the **content**, so a re-ingest is detected.
2. Write the raw ``.txt`` -- before the model runs, so the provenance record never
   depends on the model succeeding.
3. Extract. A failure here leaves the raw file with no sidecar: an incomplete
   ingest, free to retry because the call is cached (ADR-005).
4. Stage complete node files in ``/queue``.
5. Write the sidecar last, recording what the extraction actually produced.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ..llm import ChatClient
from ..models import Node
from . import _nodes, _pdf, _raw, _vision
from ._extract import extract


class IngestResult(BaseModel):
    """What one ``gw ingest`` produced. Nothing here has touched ``/nodes``."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    already_ingested: bool
    raw_path: Path
    queue_paths: list[Path]
    nodes: list[Node]
    cached: bool = False
    cost_usd: float | None = None
    # Slice 8: whether this text came from a camera, and whether you corrected it.
    ocr: bool = False
    ocr_edited: bool = False


class OcrRejected(RuntimeError):
    """You declined the transcription at the checkpoint. Nothing was frozen.

    Not an error in the pipeline -- an answer. The image stays in ``/raw`` so a retry is
    one command, and no ``.txt`` was written, so the bad transcription never became the
    §12.1 anchor.
    """


# Called with the model's transcription and the image it came from. Returns the text to
# freeze -- possibly corrected -- or None to abort.
#
# A callback rather than a prompt inside this function: `ingest_file` is a library entry
# point, and burying a terminal interaction in it would make the checkpoint impossible to
# test offline and impossible to drive from anywhere but a TTY.
Confirm = Callable[[str, Path], str | None]


def ingest_file(
    path: Path | str,
    *,
    force: bool = False,
    client: ChatClient | None = None,
    raw_dir: Path | str | None = None,
    queue_dir: Path | str | None = None,
    nodes_dir: Path | str | None = None,
    vocab_dir: Path | str | None = None,
    settings_path: Path | str | None = None,
    cache_dir: Path | str | None = None,
    usage_log: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    now: datetime | None = None,
    today: date | None = None,
    confirm: Confirm | None = None,
) -> IngestResult:
    """Ingest one source -- a text file or an image -- into the review queue.

    **An image takes a different route to the same place.** Its bytes are stored in
    ``/raw`` *before any model call* (slice 3's rule: the raw record never depends on the
    model succeeding), then OCR'd, then checkpointed via ``confirm`` before the
    transcription is frozen. From there it is the ordinary text path -- extraction knows
    nothing about cameras.

    The checkpoint exists because ``/raw`` is immutable **and** is §12.1's
    re-verification anchor. An OCR error frozen there is worse than a bad node: it
    corrupts the reference you would use to *detect* the bad node. German OCR fails in
    ways that survive a glance (``Bäckerei`` -> ``Backerei``), so it is checked once
    while checking is still cheap.

    Raises ``ExtractionError`` if the model returned nothing usable -- notably on
    ``finish_reason == "length"``, which looks like an empty result but is a truncation.
    ``VisionError`` for the same failure one step earlier, and ``OcrRejected`` if you
    decline the transcription.
    """
    path = Path(path)
    if _vision.is_image(path):
        return _ingest_image(
            path,
            force=force,
            client=client,
            raw_dir=raw_dir,
            queue_dir=queue_dir,
            nodes_dir=nodes_dir,
            vocab_dir=vocab_dir,
            settings_path=settings_path,
            cache_dir=cache_dir,
            usage_log=usage_log,
            env=env,
            now=now,
            today=today,
            confirm=confirm,
        )

    text = _raw.read_source(path)
    source_id, is_new = _raw.resolve_source_id(text, path, raw_dir=raw_dir, today=today)
    raw_path, _ = _raw.raw_paths(source_id, raw_dir=raw_dir)

    if not is_new and not force:
        # Same content already in /raw. Report what is still queued for it, if
        # anything -- an empty list means it was already promoted or rejected.
        queued = _nodes.list_queue(queue_dir=queue_dir).get(source_id, [])
        return IngestResult(
            source_id=source_id,
            already_ingested=True,
            raw_path=raw_path,
            queue_paths=queued,
            nodes=[],
        )

    _raw.store_raw(text, source_id, raw_dir=raw_dir)

    candidates, response = extract(
        text,
        client=client,
        settings_path=settings_path,
        cache_dir=cache_dir,
        usage_log=usage_log,
        env=dict(env) if env is not None else None,
    )

    nodes = _nodes.to_nodes(
        candidates,
        source_id=source_id,
        nodes_dir=nodes_dir,
        queue_dir=queue_dir,
        now=now,
    )
    queue_paths = (
        _nodes.write_queue(nodes, source_id, queue_dir=queue_dir, vocab_dir=vocab_dir)
        if nodes
        else []
    )

    _raw.write_sidecar(
        source_id,
        text=text,
        original=path,
        model=response.model,
        provider=response.provider,
        candidate_count=len(candidates),
        raw_dir=raw_dir,
    )

    return IngestResult(
        source_id=source_id,
        already_ingested=False,
        raw_path=raw_path,
        queue_paths=queue_paths,
        nodes=nodes,
        cached=response.cached,
        cost_usd=response.cost_usd,
    )


def _ingest_image(
    path: Path,
    *,
    force: bool,
    client: ChatClient | None,
    raw_dir: Path | str | None,
    queue_dir: Path | str | None,
    nodes_dir: Path | str | None,
    vocab_dir: Path | str | None,
    settings_path: Path | str | None,
    cache_dir: Path | str | None,
    usage_log: Path | str | None,
    env: Mapping[str, str] | None,
    now: datetime | None,
    today: date | None,
    confirm: Confirm | None,
) -> IngestResult:
    """Image -> OCR -> checkpoint -> the ordinary text path.

    Ordering carries the same meaning it does for text, one artifact along:

    1. the **image** lands in ``/raw`` first, before any model call -- it is the true
       source, and a failed or rejected OCR must leave it in place, retryable;
    2. OCR runs (paid, cached by the image's own bytes);
    3. you accept or correct the transcription;
    4. only then is the ``.txt`` frozen, and the sidecar written last.
    """
    data = path.read_bytes()
    suffix = path.suffix.lower()
    source_id, is_new = _raw.resolve_source_id(
        data, path, raw_dir=raw_dir, today=today, artifact_suffix=suffix
    )
    text_path, _ = _raw.raw_paths(source_id, raw_dir=raw_dir)
    image_path = _raw.raw_artifact_path(source_id, suffix, raw_dir=raw_dir)

    # `is_new` tracks the IMAGE. A previously aborted checkpoint leaves the image stored
    # with no .txt beside it, which is an incomplete ingest and freely resumable -- so
    # only a finished one counts as already ingested.
    if not is_new and text_path.exists() and not force:
        queued = _nodes.list_queue(queue_dir=queue_dir).get(source_id, [])
        return IngestResult(
            source_id=source_id,
            already_ingested=True,
            raw_path=image_path,
            queue_paths=queued,
            nodes=[],
        )

    _raw.store_binary(data, source_id, suffix, raw_dir=raw_dir)

    transcription, vision_response = _vision.transcribe(
        path,
        client=client,
        settings_path=settings_path,
        cache_dir=cache_dir,
        usage_log=usage_log,
        env=dict(env) if env is not None else None,
    )

    text = transcription if confirm is None else confirm(transcription, image_path)
    if text is None or not text.strip():
        raise OcrRejected(
            f"transcription of {path.name} was not accepted; nothing was written to "
            f"/raw beyond the image itself ({image_path.name}). Re-run to try again -- "
            "the OCR call is cached, so a retry costs nothing."
        )
    edited = text.strip() != transcription.strip()

    _raw.store_raw(text, source_id, raw_dir=raw_dir)

    candidates, response = extract(
        text,
        client=client,
        settings_path=settings_path,
        cache_dir=cache_dir,
        usage_log=usage_log,
        env=dict(env) if env is not None else None,
    )

    nodes = _nodes.to_nodes(
        candidates, source_id=source_id, nodes_dir=nodes_dir, queue_dir=queue_dir, now=now
    )
    queue_paths = (
        _nodes.write_queue(nodes, source_id, queue_dir=queue_dir, vocab_dir=vocab_dir)
        if nodes
        else []
    )

    _raw.write_sidecar(
        source_id,
        text=text,
        original=path,
        model=response.model,
        provider=response.provider,
        candidate_count=len(candidates),
        raw_dir=raw_dir,
        artifact_suffix=suffix,
        ocr_model=vision_response.model,
        ocr_provider=vision_response.provider,
        # The digest of what the MODEL said, so a corrected transcription stays auditable
        # against the image without needing a third file.
        ocr_sha256=_raw.content_hash(transcription),
        ocr_edited=edited,
    )

    return IngestResult(
        source_id=source_id,
        already_ingested=False,
        raw_path=image_path,
        queue_paths=queue_paths,
        nodes=nodes,
        cached=response.cached and vision_response.cached,
        cost_usd=(response.cost_usd or 0.0) + (vision_response.cost_usd or 0.0),
        ocr=True,
        ocr_edited=edited,
    )


class PdfIngestResult(BaseModel):
    """What one PDF produced -- **one source per page**, plus what was skipped."""

    model_config = ConfigDict(extra="forbid")

    document: str  # the source id of the stored PDF itself
    document_path: Path
    pages: list[IngestResult] = []
    skipped_pages: list[int] = []

    @property
    def source_ids(self) -> list[str]:
        return [p.source_id for p in self.pages]

    @property
    def cost_usd(self) -> float:
        return sum(p.cost_usd or 0.0 for p in self.pages)


def ingest_pdf(
    path: Path | str,
    *,
    force: bool = False,
    client: ChatClient | None = None,
    raw_dir: Path | str | None = None,
    queue_dir: Path | str | None = None,
    nodes_dir: Path | str | None = None,
    vocab_dir: Path | str | None = None,
    settings_path: Path | str | None = None,
    cache_dir: Path | str | None = None,
    usage_log: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    now: datetime | None = None,
    today: date | None = None,
) -> PdfIngestResult:
    """Ingest a text-layer PDF as one source per page. No vision call, no cost.

    Raises ``PdfError`` when the document is scanned -- with a message pointing at the
    image path, which does run OCR.

    Each page is its own source because SPEC §8.1 wants provenance at the page *and*
    because the extraction cap is per source: a 20-page chapter ingested whole would
    blow ADR-006's 5-8 cap and silently lose most of itself.

    The PDF is stored once beside the pages, so §12.1 still has something to
    re-verify a drifted node against, and each page's sidecar names it.
    """
    path = Path(path)
    extraction = _pdf.extract_pages(path)  # raises PdfError on a scanned document

    data = path.read_bytes()
    doc_id, _ = _raw.resolve_source_id(
        data, path, raw_dir=raw_dir, today=today, artifact_suffix=_pdf.PDF_SUFFIX
    )
    document_path = _raw.store_binary(data, doc_id, _pdf.PDF_SUFFIX, raw_dir=raw_dir)

    results: list[IngestResult] = []
    for page in extraction.with_text:
        source_id, is_new = _raw.resolve_source_id(
            page.text, path, raw_dir=raw_dir, today=today, page=page.number
        )
        text_path, _ = _raw.raw_paths(source_id, raw_dir=raw_dir)

        if not is_new and not force:
            queued = _nodes.list_queue(queue_dir=queue_dir).get(source_id, [])
            results.append(
                IngestResult(
                    source_id=source_id,
                    already_ingested=True,
                    raw_path=text_path,
                    queue_paths=queued,
                    nodes=[],
                )
            )
            continue

        _raw.store_raw(page.text, source_id, raw_dir=raw_dir)
        candidates, response = extract(
            page.text,
            client=client,
            settings_path=settings_path,
            cache_dir=cache_dir,
            usage_log=usage_log,
            env=dict(env) if env is not None else None,
        )
        nodes = _nodes.to_nodes(
            candidates, source_id=source_id, nodes_dir=nodes_dir, queue_dir=queue_dir, now=now
        )
        queue_paths = (
            _nodes.write_queue(nodes, source_id, queue_dir=queue_dir, vocab_dir=vocab_dir)
            if nodes
            else []
        )
        _raw.write_sidecar(
            source_id,
            text=page.text,
            original=path,
            model=response.model,
            provider=response.provider,
            candidate_count=len(candidates),
            raw_dir=raw_dir,
            source_document=doc_id,
            page=page.number,
        )
        results.append(
            IngestResult(
                source_id=source_id,
                already_ingested=False,
                raw_path=text_path,
                queue_paths=queue_paths,
                nodes=nodes,
                cached=response.cached,
                cost_usd=response.cost_usd,
            )
        )

    return PdfIngestResult(
        document=doc_id,
        document_path=document_path,
        pages=results,
        skipped_pages=extraction.empty_pages,
    )
