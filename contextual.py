"""
contextual.py -- situating blurbs, so a chunk can be found by what it is about.

Anthropic's contextual retrieval (September 2024) in the smallest form that is
still the whole idea: at ingest, one local LLM call per chunk writes a sentence or
two saying where that chunk sits in its document, and that sentence is prepended to
the text that gets embedded and lexically indexed. A chunk that reads "87 percent"
on its own is hard to find from a question about budget; a chunk that reads "Q4
review of the Alpha programme, reporting budget utilisation, 87 percent" is not.

The chunk itself is never touched, and that is the part worth being strict about.
The blurb lives in `metadata["context"]`, `ContentItem.content` stays exactly what
was extracted, and every display, citation, excerpt and guard in this system reads
`content`. So a blurb cannot reach a user's screen: the worst a bad blurb can do is
mis-rank retrieval, which is what `evaluation.py` measures. A design that let the
model's own words into the cited text would turn a retrieval aid into a
hallucination path.

Two limits are stated rather than hidden. One local LLM call per chunk makes
ingest slower by roughly the generation time of the corpus, so this is off by
default and is a decision to make per corpus, not a free switch. And the calls run
one at a time: the pipeline is synchronous end to end, and the local server this
talks to is started with a single parallel slot, so concurrent requests would queue
rather than overlap.
"""
from __future__ import annotations

import logging
import re
import time

from config import settings
from schemas import ContentItem

logger = logging.getLogger(__name__)

# Where the blurb is stored. Metadata rather than a new column on `items`: the
# store inserts sources positionally and the schema is versioned by what the
# extractors produce, so an annotation that only retrieval reads belongs with the
# other annotations (OCR confidence, speaker, table shape).
CONTEXT_KEY = "context"

PROMPT = """You situate one fragment inside the document it came from, so that a search engine can find the fragment from a question about the document.

<document filename="{filename}">
{document_text}
</document>

<fragment>
{chunk_text}
</fragment>

Write one or two short sentences, at most {max_tokens} tokens, saying where this fragment sits in the document: which section or topic it belongs to, what claim or figure it carries, and its key terms. Be specific and terse. Do not paraphrase the fragment. Do not state anything that is not in the document. Output only the sentences, with no preamble and no quotation marks."""

# A small model asked for a situating sentence sometimes answers the fragment
# instead, or refuses, or announces itself. None of those is a blurb, and indexing
# one would put the model's own words into the retrieval text -- so they are
# dropped and the chunk is indexed as itself, which is what happened before this
# module existed.
_NOT_A_BLURB = re.compile(
    r"^\s*(i\s+(?:cannot|can't|am\s+unable)|as\s+an?\s+(?:ai|language\s+model)"
    r"|sorry|apolog|i'?m\s+sorry|there\s+is\s+no\s+(?:document|information)"
    r"|fragment:|document:|answer:|\[|\{)",
    re.IGNORECASE,
)


def truncate_document(text: str, max_chars: int) -> str:
    """Head and tail of the document, with the middle marked as missing.

    A blurb is only as good as the document context it was written against, and the
    context has to fit the model's window along with the chunk and the reply. Head
    and tail rather than head alone: a title, an abstract and a summary table sit at
    the front of most documents, and the conclusion -- which is what a fragment is
    usually supporting -- sits at the end.
    """
    text = text or ""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    half = max(1, max_chars // 2)
    return f"{text[:half]}\n\n[… {len(text) - 2 * half} characters omitted …]\n\n{text[-half:]}"


def build_prompt(document_text: str, chunk_text: str, filename: str,
                 max_tokens: int, max_chars: int) -> str:
    """The single-prompt form of the request. `ollama_generate` takes one prompt,
    so the instruction and the two documents share it."""
    return PROMPT.format(
        filename=filename or "unknown",
        document_text=truncate_document(document_text, max_chars),
        chunk_text=(chunk_text or "").strip(),
        max_tokens=max_tokens,
    )


def clean_blurb(reply: str, chunk_text: str, max_chars: int) -> str:
    """Turn a model reply into a blurb worth indexing, or into nothing.

    Nothing is a real outcome and the common one at 0.5B: an empty reply, a refusal,
    a preamble, or a "blurb" that only repeats the fragment. Each of those is
    dropped rather than indexed, because a blurb that restates the chunk adds
    tokens to the embedded text without adding a way to find it.
    """
    blurb = " ".join(str(reply or "").split()).strip()
    if not blurb:
        return ""
    blurb = blurb.strip("\"'“”‘’").strip()
    if not blurb or _NOT_A_BLURB.match(blurb):
        return ""
    chunk = " ".join(str(chunk_text or "").split()).lower()
    if chunk and blurb.lower() in chunk:
        return ""
    if max_chars > 0 and len(blurb) > max_chars:
        cut = blurb[:max_chars]
        # Cut at a word, never mid-token: a truncated blurb is still indexed text.
        if " " in cut:
            cut = cut.rsplit(" ", 1)[0]
        blurb = cut.rstrip(" ,;:.") + " …"
    return blurb


def blurb_for(document_text: str, chunk_text: str, filename: str, generate=None,
              errors: list[str] | None = None) -> str:
    """One blurb, or "" -- this never raises into an ingest.

    `generate` is injectable so the prompt, the cleaning and the failure path are
    all testable without a model, and so a caller can supply a different backend.
    `errors` collects why a call failed, because "0 of 35 chunks got a blurb" tells
    an operator something is wrong but not what: a model that is down and a model
    that answers with refusals need different fixes.
    """
    if not (chunk_text or "").strip():
        return ""
    generate = generate or _default_generate
    try:
        reply = generate(
            build_prompt(
                document_text,
                chunk_text,
                filename,
                settings.contextual_max_tokens,
                settings.contextual_document_chars,
            ),
            settings.contextual_max_tokens,
        )
    except Exception as exc:
        logger.warning("contextual blurb failed for %s: %s", filename, exc)
        if errors is not None:
            errors.append(f"{type(exc).__name__}: {exc}")
        return ""
    return clean_blurb(reply, chunk_text, settings.contextual_max_chars)


def _default_generate(prompt: str, max_tokens: int) -> str:
    from llm import ollama_generate

    return ollama_generate(prompt, max_tokens=max_tokens)


def document_text_for(items: list[ContentItem]) -> str:
    """The document a chunk sits in, assembled from the chunks themselves.

    Re-reading the file would mean extracting it twice and would disagree with the
    items whenever extraction is lossy -- OCR'd pages, transcript segments, a table
    lifted into a grid. The items are the document as this system understands it, so
    they are what the blurb is written against.
    """
    return "\n\n".join((item.content or "").strip() for item in items if (item.content or "").strip())


def annotate_items(items: list[ContentItem], filename: str, generate=None,
                   warnings: list[str] | None = None) -> tuple[int, float]:
    """Write `metadata["context"]` on each item. Returns (blurbs written, seconds).

    Items are mutated in place and `content` is never read as writable: the caller
    goes on to embed and persist them, and the persisted metadata is what makes the
    blurb survive a restart instead of being recomputed on every query.
    """
    started = time.perf_counter()
    if not items or len(items) < settings.contextual_min_items:
        # One chunk is the whole document, so there is nothing to situate it in: the
        # blurb could only paraphrase the text it is attached to.
        return 0, time.perf_counter() - started

    document = document_text_for(items)
    written = 0
    missed = 0
    errors: list[str] = []
    for item in items:
        blurb = blurb_for(document, item.content or "", filename, generate=generate,
                          errors=errors)
        if blurb:
            metadata = dict(item.metadata or {})
            metadata[CONTEXT_KEY] = blurb
            item.metadata = metadata
            written += 1
        else:
            missed += 1

    elapsed = time.perf_counter() - started
    if missed and warnings is not None:
        reason = f" ({errors[0]})" if errors else ""
        warnings.append(
            f"contextual retrieval: {written} of {len(items)} chunks got a situating "
            f"blurb{reason}; the other {missed} are indexed as themselves"
        )
    logger.info(
        "contextual retrieval: %d/%d blurbs for %s in %.1fs",
        written, len(items), filename, elapsed,
    )
    return written, elapsed


def indexed_text(item: ContentItem) -> str:
    """The text that gets embedded and lexically indexed for one item.

    The blurb leads, so a chunk's own words are still at the start of what a reranker
    or a BM25 window sees after the first sentence, and a chunk with no blurb -- every
    chunk, when the feature is off -- returns its content unchanged. This function is
    the only place the two are combined, which is what keeps the display path honest:
    citations, excerpts and guards all read `content` directly.
    """
    blurb = (item.metadata or {}).get(CONTEXT_KEY) if item is not None else None
    content = (item.content or "") if item is not None else ""
    if not blurb:
        return content
    return f"{blurb}\n{content}"
