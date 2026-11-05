"""observability.py -- what the pipeline did, and what build did it.

Two artefacts, both cheap and both offline:

``stage_trace`` turns the per-stage timings the pipeline already records into an
ordered list of stages with what each one moved. Timings alone answer "where did
the time go"; they do not answer "which gate refused this question", "did MMR
change the selection", or "was the answer completed from the evidence". Those
are the questions a reviewer of an evidence system asks, so each stage carries a
note built from the item counts the pipeline reports, and a stage that did not
run is listed as skipped rather than silently absent -- a guard that never fired
and a guard that is switched off must be distinguishable.

``build_manifest`` describes the exact configuration and code version behind a
result: git commit, models, retrieval and chunking knobs, every guard flag and
threshold, auth mode, index counts and dataset version. It is embedded in each
evaluation report, served at ``GET /manifest``, written by
``scripts/manifest.py``, and printed by the health endpoint, so a number in a
report can always be traced to the build that produced it.

Neither function touches a model or the network: both are safe to call from a
health check.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from config import settings

# ---------------------------------------------------------------------------
# Stage trace
# ---------------------------------------------------------------------------

# Pipeline order. The timing key is what rag_pipeline records; None means the
# stage is a decision rather than a measured span.
STAGES: tuple[tuple[str, str, str | None], ...] = (
    ("representation", "Query representation", "representation_ms"),
    ("retrieval", "Hybrid retrieval (dense + BM25, RRF)", "retrieval_ms"),
    ("rerank", "Cross-encoder reranking", "rerank_ms"),
    ("abstention_gate", "Per-modality abstention gate", None),
    ("mmr", "MMR diversity re-selection", "mmr_ms"),
    ("period_targeting", "Period targeting (reorder)", "period_targeting_ms"),
    ("parent_expansion", "Parent/child expansion", "parent_expansion_ms"),
    ("relationships", "Relationship expansion", "relationships_ms"),
    ("coverage_guard", "Coverage guard (pre-generation)", "coverage_ms"),
    ("entity_guard", "Entity guard (pre-generation)", "entity_guard_ms"),
    ("generation", "Answer generation", "generation_ms"),
    ("repetition_guard", "Repetition guard", "repetition_guard_ms"),
    ("temporal_guard", "Temporal guard", "temporal_guard_ms"),
    ("synthesis", "Evidence completion", "synthesis_ms"),
    ("citations", "Citation building", "citations_ms"),
)

# Which config flag decides whether a stage can run at all.
STAGE_FLAGS: dict[str, str] = {
    "mmr": "mmr_enabled",
    "period_targeting": "period_targeting_enabled",
    "parent_expansion": "parent_expansion_enabled",
    "coverage_guard": "coverage_guard_enabled",
    "entity_guard": "entity_guard_enabled",
    "repetition_guard": "repetition_guard_enabled",
    "temporal_guard": "temporal_guard_enabled",
    "synthesis": "synthesis_completion_enabled",
}


def _note(stage: str, counts: dict) -> str:
    """What this stage moved, in item counts."""
    if stage == "retrieval":
        retrieved = counts.get("retrieved")
        return f"{retrieved} candidate item(s)" if retrieved is not None else ""
    if stage == "rerank":
        pool, ranked = counts.get("rerank_pool"), counts.get("reranked")
        if pool is not None and ranked is not None:
            return f"{pool} scored, top {ranked} kept"
        return ""
    if stage == "abstention_gate":
        score, threshold = counts.get("best_score"), counts.get("threshold")
        if score is None or threshold is None:
            return ""
        verdict = "passed" if score >= threshold else "would refuse"
        return f"best rerank score {score:+.3f} vs {threshold:+.1f} -> {verdict}"
    if stage == "mmr":
        selected = counts.get("mmr_selected")
        cap = counts.get("mmr_cap")
        if selected is None:
            return ""
        return f"{selected} item(s) selected, cap {cap} per source"
    if stage == "parent_expansion":
        blocks = counts.get("evidence_blocks")
        if blocks is None:
            return ""
        sections = counts.get("parent_sections")
        if not sections:
            return f"{blocks} context block(s); no parent sections in this index"
        return (
            f"{blocks} context block(s), {sections} completed from its section"
        )
    if stage == "relationships":
        final = counts.get("final_evidence")
        return f"{final} evidence item(s) to the model" if final is not None else ""
    if stage == "citations":
        cites = counts.get("citations")
        return f"{cites} citation(s)" if cites is not None else ""
    return ""


def stage_trace(
    timings: dict | None,
    counts: dict | None = None,
    detail: str = "",
) -> list[dict]:
    """Ordered stage list for one query: what ran, how long, what it moved.

    `detail` is the technical detail of the decisive stage (the guard that
    refused, the aid that completed the answer); it is attached to the trace as a
    whole rather than guessed onto a stage, because the pipeline records one
    detail per query.
    """
    timings = timings or {}
    counts = counts or {}
    trace: list[dict] = []
    for stage, label, timing_key in STAGES:
        flag = STAGE_FLAGS.get(stage)
        enabled = True if flag is None else bool(getattr(settings, flag, True))
        ran = True if timing_key is None else timing_key in timings
        entry = {
            "stage": stage,
            "label": label,
            "ms": round(timings[timing_key], 1) if ran and timing_key else None,
            "ran": bool(ran and enabled),
            "enabled": enabled,
            "note": _note(stage, counts) if ran else "",
        }
        if not enabled:
            entry["note"] = f"disabled by {flag}"
        elif not ran and timing_key:
            entry["note"] = "not reached"
        trace.append(entry)

    total = timings.get("total_ms")
    if total is not None:
        trace.append({
            "stage": "total",
            "label": "End to end",
            "ms": round(total, 1),
            "ran": True,
            "enabled": True,
            "note": counts.get("outcome", ""),
        })
    if detail:
        trace.append({
            "stage": "decision",
            "label": "Decisive check",
            "ms": None,
            "ran": True,
            "enabled": True,
            "note": detail,
        })
    return trace


def format_stage_trace(trace: list[dict]) -> str:
    """Fixed-width text rendering for logs, the CLI and the UI expander."""
    if not trace:
        return ""
    width = max(len(entry["label"]) for entry in trace)
    lines: list[str] = []
    for entry in trace:
        state = "" if entry["ran"] else ("off" if not entry["enabled"] else "-")
        timing = f"{entry['ms']:>8.1f} ms" if entry["ms"] is not None else " " * 11
        lines.append(
            f"  {entry['label']:<{width}}  {timing}  {state:<3} {entry['note']}".rstrip()
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Run manifest
# ---------------------------------------------------------------------------

def git_commit(repo: Path | None = None) -> str:
    """Short commit hash, or 'unknown' when git or the repository is absent."""
    directory = Path(repo or Path(__file__).resolve().parent)
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(directory), capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def git_dirty(repo: Path | None = None) -> bool | None:
    """True when tracked files differ from HEAD; None when git is unavailable."""
    directory = Path(repo or Path(__file__).resolve().parent)
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(directory), capture_output=True, text=True, timeout=10,
        )
        return bool(out.stdout.strip())
    except Exception:
        return None


def _disk_free_mb(path: Path) -> float | None:
    try:
        usage = os.statvfs(str(path))
        return round(usage.f_bavail * usage.f_frsize / (1024 * 1024), 1)
    except Exception:
        return None


def _ram_mb() -> dict:
    """Total and available RAM in MB, from /proc when present."""
    info: dict[str, float | None] = {"total": None, "available": None}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            value = rest.strip().split(" ")[0]
            if key == "MemTotal":
                info["total"] = round(int(value) / 1024, 1)
            elif key == "MemAvailable":
                info["available"] = round(int(value) / 1024, 1)
    except Exception:
        pass
    return info


def build_manifest(index_stats: dict | None = None) -> dict:
    """Everything needed to reproduce a result, without probing any model.

    `index_stats` is accepted pre-built so a caller that already has the store
    open does not open it twice; when omitted the manifest simply leaves the
    index section out rather than loading FAISS from a health check.
    """
    from auth import PERMISSIONS

    dataset_version = None
    dataset_path = settings.evaluation_dir / "dataset" / "queries.json"
    try:
        import json

        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        dataset_version = dataset.get("dataset_version")
    except Exception:
        dataset_version = None

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "application": {
            "name": "Evidence AI (Multi-Modal RAG)",
            "git_commit": git_commit(),
            "git_dirty": git_dirty(),
        },
        "platform": {
            "python": sys.version.split(" ")[0],
            "system": f"{platform.system()} {platform.release()}",
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "ram_mb": _ram_mb(),
            "disk_free_mb": _disk_free_mb(Path(settings.index_dir).parent),
        },
        "models": {
            "llm": settings.ollama_model,
            "embedding": settings.embedding_model,
            "embedding_dim": settings.embedding_dim,
            "reranker": settings.reranker_model,
            "whisper": settings.whisper_model_size,
            "inference_backend": settings.inference_backend,
        },
        "retrieval": {
            "top_k": settings.top_k,
            "rerank_top_k": settings.rerank_top_k,
            "hybrid_enabled": settings.hybrid_enabled,
            "mmr_enabled": settings.mmr_enabled,
            "mmr_lambda": settings.mmr_lambda,
            "mmr_max_per_source": settings.mmr_max_per_source,
            "mmr_max_per_source_structure": settings.mmr_max_per_source_structure,
            "max_context_items": settings.max_context_items,
            "max_context_chars": settings.max_context_chars,
            "max_relationship_expansion": settings.max_relationship_expansion,
        },
        "chunking": {
            "mode": settings.chunking_mode,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "min_chunk_chars": settings.min_chunk_chars,
            "structure_min_chunk_chars": settings.structure_min_chunk_chars,
            "parent_expansion_enabled": settings.parent_expansion_enabled,
            "parent_max_chars": settings.parent_max_chars,
        },
        "guards": {
            "abstention_threshold": settings.abstention_threshold,
            "abstention_threshold_image": settings.abstention_threshold_image,
            "coverage_guard_enabled": settings.coverage_guard_enabled,
            "entity_guard_enabled": settings.entity_guard_enabled,
            "temporal_guard_enabled": settings.temporal_guard_enabled,
            "period_targeting_enabled": settings.period_targeting_enabled,
            "numeric_guard_enabled": settings.numeric_guard_enabled,
            "numeric_guard_min_digits": settings.numeric_guard_min_digits,
            "repetition_guard_enabled": settings.repetition_guard_enabled,
            "synthesis_completion_enabled": settings.synthesis_completion_enabled,
        },
        "generation": {
            "temperature": settings.ollama_temperature,
            "seed": settings.ollama_seed,
            "num_ctx": settings.ollama_num_ctx,
            "num_predict": settings.ollama_num_predict,
            "timeout_s": settings.ollama_timeout,
        },
        "security": {
            "api_auth_enabled": settings.api_auth_enabled,
            "token_prefix": "eai_",
            "roles": sorted(PERMISSIONS),
        },
        "evaluation": {
            "dataset_version": dataset_version,
            "baseline_present": (settings.evaluation_dir / "baseline.json").exists(),
        },
    }
    if index_stats is not None:
        manifest["index"] = index_stats
    return manifest


def manifest_text(manifest: dict) -> str:
    """Human-readable rendering, for the CLI and the demo runbook."""
    lines: list[str] = [
        f"Evidence AI run manifest  ({manifest['generated_at']})",
        f"  commit      {manifest['application']['git_commit']}"
        f"{' (uncommitted changes)' if manifest['application']['git_dirty'] else ''}",
        f"  platform    {manifest['platform']['system']} / python "
        f"{manifest['platform']['python']} / {manifest['platform']['cpu_count']} cpu",
        f"  models      llm={manifest['models']['llm']} "
        f"embed={manifest['models']['embedding']} rerank={manifest['models']['reranker']}",
        f"  chunking    mode={manifest['chunking']['mode']} "
        f"size={manifest['chunking']['chunk_size']} "
        f"parents={'on' if manifest['chunking']['parent_expansion_enabled'] else 'off'}",
        f"  retrieval   top_k={manifest['retrieval']['top_k']} "
        f"rerank_top_k={manifest['retrieval']['rerank_top_k']} "
        f"mmr={manifest['retrieval']['mmr_enabled']} "
        f"(lambda {manifest['retrieval']['mmr_lambda']})",
        f"  generation  temp={manifest['generation']['temperature']} "
        f"seed={manifest['generation']['seed']} "
        f"num_ctx={manifest['generation']['num_ctx']} "
        f"num_predict={manifest['generation']['num_predict']}",
        f"  guards      "
        + ", ".join(
            f"{name.replace('_enabled', '')}={'on' if value else 'off'}"
            for name, value in manifest["guards"].items()
            if name.endswith("_enabled")
        ),
        f"  thresholds  text={manifest['guards']['abstention_threshold']} "
        f"image={manifest['guards']['abstention_threshold_image']}",
        f"  security    api_auth={'on' if manifest['security']['api_auth_enabled'] else 'off'} "
        f"roles={','.join(manifest['security']['roles'])}",
        f"  evaluation  dataset v{manifest['evaluation']['dataset_version']} "
        f"baseline={'present' if manifest['evaluation']['baseline_present'] else 'missing'}",
    ]
    if "index" in manifest:
        index = manifest["index"]
        lines.append(
            f"  index       sources={index.get('sources')} items={index.get('items')} "
            f"parents={index.get('parents', 0)} relationships={index.get('relationships')} "
            f"vectors={index.get('vectors')}"
        )
    return "\n".join(lines)


def write_manifest(path: str | Path, index_stats: dict | None = None) -> Path:
    """Write the manifest as JSON and return the path."""
    import json

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(index_stats=index_stats)
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return target
