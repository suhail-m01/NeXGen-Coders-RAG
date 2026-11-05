"""Parent/child chunking: retrieve small, answer big.

The measured failure this exists for
-----------------------------------
Structure-aware chunking improved every retrieval metric -- precision .3115 ->
.324, MRR .6146 -> .6719, citation accuracy .4135 -> .4917, forbidden leaks
.1875 -> .125 -- and still lost the gate, because answer correctness fell .8125
-> .6458. Retrieval found the right material more often and the model answered
worse. The reason is arithmetic, not ranking: a four-row metrics table becomes
four separate chunks, ``mmr_max_per_source`` admits two chunks per file, so the
question "which quarter used the most budget?" reaches the model as two rows out
of four and it guesses. Three queries failed that way (a table row, a page whose
neighbouring sentence carried the answer, a slide title separated from its body).

The fix is the standard small-to-big arrangement, and it needs no new storage
concept: ``ContentItem.parent_id`` was already in the schema and unused.

* children stay the unit of retrieval and reranking, so the fine granularity that
  improved precision and MRR is kept;
* a parent is the contiguous run of children that forms one complete unit of
  reading -- a whole table, a whole section, a whole transcript -- capped at
  ``parent_max_chars``;
* at query time the selected children are collapsed into their parents, so the
  model reads the complete table while the citation still names the section, page
  or timestamp range the parent covers.

Everything here is inert unless parents exist, and parents are only built when
``chunking_mode == "structure"``, so the fixed-chunking baseline is untouched.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from config import settings
from schemas import ContentItem, RerankedResult, SourceLocation, derive_id

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ingestion: build parents from children
# ---------------------------------------------------------------------------

def _group_key(item: ContentItem) -> str:
    """The heading a child belongs to, or '' when the document has none.

    Only the top level of the heading path is used: a parent is a section a
    reader would recognise ("Open Risks", "Table 1"), not every sub-bullet under
    it. Audio and image children have no heading and group by their own run.
    """
    path = item.metadata.get("heading_path") or ()
    if isinstance(path, str):
        return path
    return path[0] if path else ""


def _parent_location(children: list[ContentItem], heading: str) -> SourceLocation:
    """A location that is true of the whole parent, not just its first child.

    A page number is kept only when every child is on that page -- a parent
    spanning pages 2 to 4 that claims "Page 2" would send a reader to the wrong
    place. Audio keeps its range, widened to cover every child, because a
    timestamp citation is the whole point of transcript evidence.
    """
    first, last = children[0].location, children[-1].location
    pages = {c.location.page_number for c in children if c.location.page_number is not None}
    starts = [
        c.location.timestamp_start for c in children
        if c.location.timestamp_start is not None
    ]
    ends = [
        c.location.timestamp_end for c in children
        if c.location.timestamp_end is not None
    ]
    return SourceLocation(
        page_number=next(iter(pages)) if len(pages) == 1 else None,
        section=heading or first.section,
        timestamp_start=min(starts) if starts else None,
        timestamp_end=max(ends) if ends else None,
        image_path=first.image_path,
    )


def assign_parents(
    items: list[ContentItem],
    max_chars: int | None = None,
) -> tuple[list[ContentItem], list[ContentItem]]:
    """Group one document's children into parents.

    Returns ``(children, parents)``. Children are the same objects with
    ``parent_id`` set; a child that would be its own whole parent keeps
    ``parent_id=None``, because storing a parent identical to its only child
    doubles the text and buys nothing.
    """
    cap = settings.parent_max_chars if max_chars is None else max_chars
    if not items:
        return [], []

    groups: list[list[ContentItem]] = []
    current: list[ContentItem] = []
    current_key: str | None = None
    current_chars = 0
    for item in items:
        key = _group_key(item)
        size = len(item.content or "") + 2
        if current and (key != current_key or current_chars + size > cap):
            groups.append(current)
            current, current_chars = [], 0
        current.append(item)
        current_key = key
        current_chars += size
    if current:
        groups.append(current)

    parents: list[ContentItem] = []
    for index, group in enumerate(groups):
        if len(group) == 1:
            continue  # the child is already the whole unit of reading
        source_id = group[0].source_id
        parent_id = derive_id(source_id, "parent", index, group[0].item_id)
        heading = _group_key(group[0])
        parent = ContentItem(
            item_id=parent_id,
            source_id=source_id,
            modality=group[0].modality,
            content="\n\n".join(child.content for child in group),
            location=_parent_location(group, heading),
            parent_id=None,
            vector_id=None,          # never embedded, never retrieved directly
            metadata={
                "is_parent": True,
                "group_index": index,
                "child_ids": [child.item_id for child in group],
                "child_count": len(group),
                "heading_path": list(group[0].metadata.get("heading_path") or ()),
                "source_type": group[0].metadata.get("source_type", ""),
            },
        )
        parents.append(parent)
        for child in group:
            child.parent_id = parent_id
    return items, parents


# ---------------------------------------------------------------------------
# Query time: collapse selected children into their parents
# ---------------------------------------------------------------------------

@dataclass
class ExpansionDetail:
    """What the collapse did, for the technical-detail trail."""

    children: int = 0
    parents: int = 0
    kept_as_child: int = 0
    reason: str = ""  # expanded | no_parents | ""

    @property
    def message(self) -> str:
        if self.reason == "expanded":
            extra = (
                f"; {self.kept_as_child} item(s) left as retrieved because the "
                f"parent did not fit the context budget"
                if self.kept_as_child
                else ""
            )
            return (
                f"parent expansion: {self.children} retrieved chunk(s) collapsed "
                f"into {self.parents} complete section(s){extra}"
            )
        return ""


def _block_text(parent: ContentItem, child: ContentItem) -> str:
    """The context block for a selected child: its own text, then its section.

    Order matters and was measured. Handing the model the whole transcript alone
    cost an audio question that fixed chunking answers correctly: the retrieved
    segment said "classification accuracy reached 99.4%", the section continued
    into milestones and latency, and the model answered from the far end of the
    block instead of the passage that matched. So the passage that was actually
    retrieved leads, and the complete section follows as labelled context -- the
    table a row came from, the paragraph a sentence sits in -- which is what stops
    a small model guessing at what surrounds the fragment it was shown.
    """
    child_text = (child.content or "").strip()
    parent_text = (parent.content or "").strip()
    if not child_text or child_text == parent_text:
        return parent_text
    where = parent.location.human() or "the same source"
    return f"{child_text}\n\nComplete section ({where}):\n{parent_text}"


def expand_to_parents(
    store,
    results: list[RerankedResult],
    char_budget: int | None = None,
) -> tuple[list[RerankedResult], ExpansionDetail]:
    """Replace selected children with the complete sections they came from.

    Order is preserved: the first child of a parent decides where the parent
    sits, so the period targeting that ran just before this still holds, and a
    parent inherits the best score among its children, so confidence and the
    abstention decision are unchanged. When a parent would not fit the remaining
    context budget the child is kept instead -- a smaller block beats dropping
    the evidence, and beats truncating a table mid-row, which is what
    ``format_evidence`` would otherwise do.
    """
    budget = settings.max_context_chars if char_budget is None else char_budget
    if not results:
        return results, ExpansionDetail()
    if not any(getattr(result.item, "parent_id", None) for result in results):
        return results, ExpansionDetail(
            children=len(results), reason="no_parents"
        )

    cache: dict[str, ContentItem | None] = {}
    out: list[RerankedResult] = []
    used = 0
    children_collapsed = 0
    parents_emitted = 0
    kept_as_child = 0
    placed: dict[str, int] = {}

    for result in results:
        parent_id = getattr(result.item, "parent_id", None)
        if not parent_id:
            out.append(result)
            used += len(result.item.content or "")
            continue

        if parent_id in placed:
            # Another child of a parent already in the list: record it and drop
            # the duplicate, which is the whole point of the collapse.
            position = placed[parent_id]
            merged = out[position].item.metadata.setdefault("expanded_from", [])
            if result.item.item_id not in merged:
                merged.append(result.item.item_id)
            passages = out[position].item.metadata.setdefault("retrieved_texts", [])
            if (result.item.content or "") not in passages:
                passages.append(result.item.content or "")
            children_collapsed += 1
            continue

        if parent_id not in cache:
            try:
                cache[parent_id] = store.get_parent(parent_id)
            except Exception as exc:                      # never fail a query
                logger.warning("parent lookup failed for %s: %s", parent_id, exc)
                cache[parent_id] = None
        parent = cache[parent_id]
        if parent is None:
            out.append(result)
            used += len(result.item.content or "")
            continue

        block_text = _block_text(parent, result.item)
        if used + len(block_text) > budget and used > 0:
            out.append(result)                             # keep the child
            used += len(result.item.content or "")
            kept_as_child += 1
            continue

        parent = parent.model_copy(deep=True)
        parent.content = block_text
        parent.metadata["expanded_from"] = [result.item.item_id]
        parent.metadata["retrieved_child"] = result.item.location.human()
        # The passages that actually matched, kept separate from the section text
        # added around them. Guards that ask "what did the evidence say?" must
        # read these, not the whole block: a section that spans 2023 and 2024
        # would otherwise let an answer about the wrong year look supported.
        parent.metadata["retrieved_texts"] = [result.item.content or ""]
        out.append(
            RerankedResult(
                item=parent,
                retrieval_score=result.retrieval_score,
                rerank_score=result.rerank_score,
                rank=result.rank,
            )
        )
        placed[parent_id] = len(out) - 1
        used += len(block_text)
        children_collapsed += 1
        parents_emitted += 1

    if not parents_emitted:
        return results, ExpansionDetail(
            children=len(results), reason="no_parents"
        )
    return out, ExpansionDetail(
        children=children_collapsed,
        parents=parents_emitted,
        kept_as_child=kept_as_child,
        reason="expanded",
    )
