"""Chat attachments: ephemeral evidence from a file asked about directly.

The UI's paperclip lets a question carry a document or an image, the way a
chat assistant accepts an upload. The attachment is extracted with exactly the
processors ingestion uses (text layer and tables for documents, OCR and vision
for images), its chunks enter the reranking pool as evidence for this turn,
and everything downstream - abstention gate, coverage and entity guards,
synthesis, citations, history - treats them like any other evidence. So an
attachment answer is refused when the attachment does not actually contain the
answer, and every claim still resolves to a citable block.

What attachments are NOT: they are never indexed, never persisted and never
visible to later questions. The bytes live in a temporary directory for the
duration of one extraction and are deleted in a `finally`; the only trace left
is the citation in the recorded turn, which carries the filename and an
excerpt resolved at record time. A question that should search the attachment
again can ingest it from Sources instead, which is the durable path.

Audio is deliberately not accepted here: transcription is the expensive
modality, and an upload that quietly costs minutes belongs to the ingest queue
with its rate bucket, not to a chat round trip.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from config import settings
from rag_pipeline import DOCUMENT_EXTENSIONS, IMAGE_EXTENSIONS
from schemas import RetrievalResult

ATTACHMENT_EXTENSIONS = DOCUMENT_EXTENSIONS | IMAGE_EXTENSIONS
MAX_ATTACHMENT_FILES = 5


class AttachmentError(ValueError):
    """One attachment could not be accepted; the message names the file."""


def validate_attachment(name: str, size: int) -> None:
    suffix = Path(name or "").suffix.lower()
    if suffix not in ATTACHMENT_EXTENSIONS:
        raise AttachmentError(
            f"'{name}' is not a supported attachment. Attach documents "
            f"({', '.join(sorted(DOCUMENT_EXTENSIONS))}) or images "
            f"({', '.join(sorted(IMAGE_EXTENSIONS))}); audio and other "
            "formats belong in Sources > ingest."
        )
    if size > settings.max_file_bytes:
        raise AttachmentError(
            f"'{name}' is larger than the {settings.max_file_bytes // (1024 * 1024)} "
            "MB attachment limit."
        )


def _scrub(item, tmp: Path) -> None:
    """Remove the temporary path from anything that would leave with it.

    Processors record where they read a file (image crops, metadata); for an
    attachment that path is a temp directory that ceases to exist when the
    turn ends, so letting it reach a citation or the history database would
    store a server path that reads as nonsense later.
    """
    prefix = str(tmp)
    if item.location.image_path and prefix in str(item.location.image_path):
        item.location.image_path = None
    for key, value in list((item.metadata or {}).items()):
        if isinstance(value, str) and prefix in value:
            item.metadata[key] = Path(value).name


def extract_attachments(pipeline, files: list[tuple[str, bytes]]):
    """Extract uploaded attachments into rerank-pool evidence.

    Returns (evidence, filenames, warnings): `evidence` is a list of
    RetrievalResult ready to join the corpus candidates, `filenames` maps the
    ephemeral source id to the name the user actually uploaded (the temp file
    name is an implementation detail), `warnings` carries processor warnings
    prefixed with the file they belong to.

    Raises AttachmentError for the first file that cannot be accepted; the
    caller turns that into a 400 that names the file and the reason.
    """
    if not files:
        raise AttachmentError("No attachment was supplied.")
    if len(files) > MAX_ATTACHMENT_FILES:
        raise AttachmentError(
            f"At most {MAX_ATTACHMENT_FILES} attachments may accompany one "
            f"question; {len(files)} were supplied."
        )

    evidence: list[RetrievalResult] = []
    filenames: dict[str, str] = {}
    warnings: list[str] = []
    seen_items: set[str] = set()
    tmp = Path(tempfile.mkdtemp(prefix="eai-attach-"))
    try:
        for index, (name, data) in enumerate(files):
            validate_attachment(name, len(data))
            suffix = Path(name).suffix.lower()
            destination = tmp / f"attach-{index}{suffix}"
            destination.write_bytes(data)
            processor = pipeline.route(destination)
            if processor is None:  # unreachable after validate, kept honest
                raise AttachmentError(f"'{name}' has no processor.")
            result = processor(str(destination))
            if not result.success or not result.items:
                raise AttachmentError(
                    f"'{name}' could not be read: "
                    f"{result.error or 'no extractable content.'}"
                )
            for item in result.items:
                _scrub(item, tmp)
                if item.item_id in seen_items:
                    continue  # the same file attached twice adds nothing
                seen_items.add(item.item_id)
                evidence.append(
                    RetrievalResult(item=item, score=1.0, rank=len(evidence))
                )
            filenames[result.source.source_id] = name
            warnings.extend(f"{name}: {warning}" for warning in result.warnings)
        if not evidence:
            raise AttachmentError(
                "The attachments produced no readable text or image content."
            )
        return evidence, filenames, warnings
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
