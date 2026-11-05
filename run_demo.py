"""Live demo driver for Evidence AI (Multi-Modal-RAG repo).

Ingests the synthetic Project Alpha corpus (PDF/DOCX/CSV/PNG/WAV),
then runs grounded queries + one question that must be refused.
"""
import json
import time

from rag_pipeline import get_pipeline
from schemas import QueryRequest

pipe = get_pipeline()
print("HEALTH:", json.dumps(pipe.health(), indent=1)[:600])

t0 = time.time()
results = []
for d in ["data/documents", "data/images", "data/audio"]:
    results += pipe.ingest_directory(d)
print(f"\n=== INGESTION ({time.time()-t0:.1f}s) ===")
for r in results:
    print(f"- {r.source.filename}: success={r.success} items={len(r.items)} "
          f"relationships={len(r.relationships)} error={r.error}")
    for it in r.items[:3]:
        loc = str(it.location)
        print(f"    item[{it.modality}] loc={loc} text={it.content[:90]!r}")

QUESTIONS = [
    "How many records did the ingestion platform process in 2024, and what classification accuracy was reached?",
    "What was the budget utilisation, and how many milestones were completed by how many engineers?",
    "What is the main outstanding risk for Project Alpha?",
    "What is the current stock price of the company and who is its CEO?",  # must abstain
]

for q in QUESTIONS:
    t0 = time.time()
    resp = pipe.query(QueryRequest(query=q))
    print(f"\n=== QUERY ({time.time()-t0:.1f}s) ===")
    print("Q:", q)
    print("ABSTAINED:", resp.abstained, "| confidence:", resp.confidence,
          "| error:", resp.error_code)
    print("ANSWER:", resp.answer)
    for c in resp.citations:
        print("  CITE:", c.model_dump())
    print("  latency_ms:", {k: round(v) for k, v in resp.latency_ms.items()})
