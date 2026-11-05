from __future__ import annotations

import argparse
import shutil
from pathlib import Path

CITATION_OLD = '''            if not hits:
                # Fall back to the single best-matching evidence item so the
                # sentence still resolves to something the user can inspect.
                hits = [int(np.argmax(similarities))]
            supported.update(hits)
            sentence_map[sentence] = hits
'''

CITATION_NEW = '''            if not hits:
                # Do not attach a weak, nearest-neighbour citation merely to
                # make an unsupported sentence look grounded.
                sentence_map[sentence] = []
                continue
            supported.update(hits)
            sentence_map[sentence] = hits
'''

PIPELINE_IMPORT_OLD = "        from citations import build_citations\n"
PIPELINE_IMPORT_NEW = (
    "        from citations import build_citations\n"
    "        from grounding_audit import build_grounding_ledger\n"
)
PIPELINE_CALL_OLD = (
    "        citations, _sentence_map = build_citations(answer, final_evidence, filenames)\n"
    "        timings[\"citations_ms\"] = (time.perf_counter() - mark) * 1000\n"
)
PIPELINE_CALL_NEW = (
    "        citations, _sentence_map = build_citations(answer, final_evidence, filenames)\n"
    "        ledger = build_grounding_ledger(answer, _sentence_map)\n"
    "        timings[\"citations_ms\"] = (time.perf_counter() - mark) * 1000\n"
    "\n"
    "        # Fail closed: if any answer sentence has no qualifying citation,\n"
    "        # do not return it as a grounded answer.\n"
    "        if ledger.status != \"complete\":\n"
    "            logger.warning(\n"
    "                \"Grounding audit rejected answer: %s/%s sentences cited\",\n"
    "                ledger.supported_sentences,\n"
    "                ledger.total_sentences,\n"
    "            )\n"
    "            timings[\"total_ms\"] = (time.perf_counter() - started) * 1000\n"
    "            return QueryResponse(\n"
    "                answer=ABSTENTION_MESSAGE,\n"
    "                citations=[],\n"
    "                retrieved_items=reranked,\n"
    "                relationships=relationships,\n"
    "                abstained=True,\n"
    "                confidence=confidence,\n"
    "                error_code=ErrorCode.INSUFFICIENT_EVIDENCE,\n"
    "                query_representation=representation,\n"
    "                latency_ms=timings,\n"
    "            )\n"
)

def replace_once(content: str, old: str, new: str, label: str) -> str:
    count = content.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one occurrence of {label}, found {count}. "
            "No files were changed. Check that this is the expected repository version."
        )
    return content.replace(old, new, 1)

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply strict sentence-level citation enforcement to NeXGen RAG."
    )
    parser.add_argument(
        "--project",
        default=".",
        help="Path to the NeXGen-Coders-RAG project root (default: current directory).",
    )
    args = parser.parse_args()
    root = Path(args.project).resolve()
    citations_path = root / "citations.py"
    pipeline_path = root / "rag_pipeline.py"
    audit_path = root / "grounding_audit.py"

    for path in (citations_path, pipeline_path, audit_path):
        if not path.is_file():
            raise SystemExit(f"Required file not found: {path}")

    citations = citations_path.read_text(encoding="utf-8")
    pipeline = pipeline_path.read_text(encoding="utf-8")

    # Build all changes in memory first. A mismatch aborts before writing.
    updated_citations = replace_once(
        citations, CITATION_OLD, CITATION_NEW, "citation fallback block"
    )
    updated_pipeline = replace_once(
        pipeline, PIPELINE_IMPORT_OLD, PIPELINE_IMPORT_NEW, "citation import"
    )
    updated_pipeline = replace_once(
        updated_pipeline, PIPELINE_CALL_OLD, PIPELINE_CALL_NEW, "citation call block"
    )

    # Back up originals before replacing them.
    for path in (citations_path, pipeline_path):
        backup = path.with_suffix(path.suffix + ".before_grounding_enforcement.bak")
        if not backup.exists():
            shutil.copy2(path, backup)

    citations_path.write_text(updated_citations, encoding="utf-8")
    pipeline_path.write_text(updated_pipeline, encoding="utf-8")
    print("Grounding enforcement patch applied.")
    print("Backups created with .before_grounding_enforcement.bak suffix.")
    print("Next: run the tests described in docs/INTEGRATION.md.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
