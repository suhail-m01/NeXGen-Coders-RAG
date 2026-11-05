"""chat_history.py -- persistent sessions with full retrieval traces.

Sessions and messages live in SQLite, outside the index, so rebuilding the index
never destroys a conversation.

Why the trace is stored and not just the answer text: an evidence tool has to
answer "why did you say that?" days later, when the index may have been rebuilt
and the original retrieval is no longer reproducible. Each assistant message
therefore stores the evidence it actually used -- ranks, both scores, filenames,
locations, excerpts, speaker labels, OCR confidence -- plus citations, timings
and any figures the numeric guard could not verify. A past answer can be
re-opened and inspected without re-running the query, and the inspection does
not depend on the index still holding those items.

Adapted from ORION's ChatDB (sessions + messages, one JSON blob), with the audit
fields promoted to real columns so they can be queried, and export writers added
for handing a session to a reviewer.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from config import settings

logger = logging.getLogger("evidence_ai.chat_history")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id    TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    backend       TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    message_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    abstained  INTEGER NOT NULL DEFAULT 0,
    error_code TEXT,
    confidence REAL,
    -- Figures the numeric guard could not verify, as a JSON list.
    unverified TEXT NOT NULL DEFAULT '[]',
    citations  TEXT NOT NULL DEFAULT '[]',
    trace      TEXT NOT NULL DEFAULT '{}',
    timings    TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC);
"""

_JSON_COLUMNS = ("unverified", "citations", "trace", "timings")
_JSON_DEFAULTS = {"unverified": [], "citations": [], "trace": {}, "timings": {}}

_history: "ChatHistory | None" = None
_history_lock = threading.Lock()


def _now() -> str:
    """ISO-8601 UTC, second resolution. Stored as text so exports are readable
    and sortable without a datetime parser."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def title_from(question: str, limit: int = 60) -> str:
    """Session title from its first question: one line, truncated on a word
    boundary, never an empty string."""
    text = " ".join((question or "").split())
    if not text:
        return "Untitled session"
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return f"{cut}…"


# ---------------------------------------------------------------------------
# Trace construction
# ---------------------------------------------------------------------------

def build_trace(request, response, filenames: dict[str, str]) -> dict:
    """Snapshot everything needed to re-inspect one answer later.

    `filenames` maps source_id -> filename (the pipeline already builds this);
    resolving names here rather than at replay time is the whole point -- the
    trace stays readable even if the source is later deleted.
    """
    evidence = []
    for result in response.retrieved_items:
        item = result.item
        metadata = item.metadata or {}
        evidence.append({
            "rank": result.rank,
            "item_id": item.item_id,
            "source_id": item.source_id,
            "filename": filenames.get(item.source_id, "unknown"),
            "modality": item.modality.value,
            "location": item.location.human(),
            "retrieval_score": round(result.retrieval_score, 4),
            "rerank_score": round(result.rerank_score, 4),
            "excerpt": item.content[: settings.max_excerpt_chars],
            # Modality-specific provenance that a reviewer will ask about.
            "speaker": metadata.get("speaker"),
            "ocr_confidence": metadata.get("ocr_confidence"),
            "ocr_unreliable": metadata.get("ocr_unreliable"),
        })

    return {
        "query": request.query,
        "query_modality": request.query_modality.value,
        "inference_mode": request.inference_mode.value,
        "top_k": request.top_k,
        "rerank_top_k": request.rerank_top_k,
        "representation": (response.query_representation or "")[:2000],
        "evidence": evidence,
        "relationships": [
            {
                "type": relationship.relationship_type.value,
                "confidence": relationship.confidence,
                "source_item_id": relationship.source_item_id,
                "target_item_id": relationship.target_item_id,
            }
            for relationship in response.relationships[: settings.max_relationship_expansion]
        ],
    }


def build_citation_records(response) -> list[dict]:
    return [
        {
            "citation_id": citation.citation_id,
            "filename": citation.filename,
            "modality": citation.modality.value,
            "location": citation.location.human(),
            "item_id": citation.item_id,
            "source_id": citation.source_id,
            "excerpt": citation.excerpt,
        }
        for citation in response.citations
    ]


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class ChatHistory:
    """SQLite-backed sessions and messages. One connection per process, guarded
    by a lock: Streamlit reruns on every interaction and may do so from more
    than one thread."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path or settings.history_db)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # ---- sessions ---------------------------------------------------------

    def create_session(self, title: str = "New session", backend: str | None = None) -> str:
        session_id = uuid.uuid4().hex
        stamp = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions VALUES (?,?,?,?,?,?)",
                (session_id, title, backend, stamp, stamp, 0),
            )
            self._conn.commit()
        return session_id

    def list_sessions(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def get_session(self, session_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return dict(row) if row else None

    def rename_session(self, session_id: str, title: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                (title, _now(), session_id),
            )
            self._conn.commit()

    def delete_session(self, session_id: str) -> None:
        """Messages go with it: the foreign key is ON DELETE CASCADE."""
        with self._lock:
            self._conn.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,)
            )
            self._conn.commit()

    # ---- messages ---------------------------------------------------------

    def add_user_message(self, session_id: str, question: str) -> str:
        return self._add(session_id, "user", question)

    def add_assistant_message(
        self,
        session_id: str,
        response,
        *,
        trace: dict | None = None,
        citations: list[dict] | None = None,
        unverified: list[str] | None = None,
    ) -> str:
        """Persist one answer with its audit trail."""
        return self._add(
            session_id,
            "assistant",
            response.answer,
            abstained=bool(response.abstained),
            error_code=response.error_code.value if response.error_code else None,
            confidence=response.confidence,
            unverified=unverified or [],
            citations=citations if citations is not None else build_citation_records(response),
            trace=trace or {},
            timings=response.latency_ms or {},
        )

    def _add(self, session_id: str, role: str, content: str, **fields) -> str:
        message_id = uuid.uuid4().hex
        stamp = _now()
        row = {
            "abstained": 0,
            "error_code": None,
            "confidence": None,
            "unverified": [],
            "citations": [],
            "trace": {},
            "timings": {},
        }
        row.update(fields)
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    message_id, session_id, role, content, stamp,
                    int(bool(row["abstained"])), row["error_code"], row["confidence"],
                    json.dumps(row["unverified"]), json.dumps(row["citations"]),
                    json.dumps(row["trace"]), json.dumps(row["timings"]),
                ),
            )
            # The first question names the session; every message touches it so
            # the sidebar list stays in recency order.
            self._conn.execute(
                """UPDATE sessions
                      SET message_count = message_count + 1,
                          updated_at = ?,
                          title = CASE WHEN message_count = 0 AND ? = 'user'
                                       THEN ? ELSE title END
                    WHERE session_id = ?""",
                (stamp, role, title_from(content), session_id),
            )
            self._conn.commit()
        return message_id

    def messages(self, session_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,),
            ).fetchall()
        out = []
        for row in rows:
            message = dict(row)
            message["abstained"] = bool(message["abstained"])
            for column in _JSON_COLUMNS:
                try:
                    message[column] = json.loads(message[column] or "null")
                except json.JSONDecodeError:
                    logger.warning("Corrupt %s JSON in message %s", column, message["message_id"])
                    message[column] = _JSON_DEFAULTS[column]
            out.append(message)
        return out

    def stats(self) -> dict:
        with self._lock:
            sessions = self._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            messages = self._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            flagged = self._conn.execute(
                "SELECT COUNT(*) FROM messages WHERE unverified != '[]'"
            ).fetchone()[0]
        return {
            "sessions": sessions,
            "messages": messages,
            "answers_with_unverified_figures": flagged,
            "db_path": str(self.db_path),
        }

    def close(self) -> None:
        with self._lock:
            self._conn.commit()
            self._conn.close()

    # ---- exports ----------------------------------------------------------

    def export_markdown(self, session_id: str) -> str:
        """A session as a reviewable document: every claim, its citations, the
        evidence behind it and the timings."""
        session = self.get_session(session_id) or {}
        messages = self.messages(session_id)
        lines = [
            f"# Evidence AI session — {session.get('title', 'Untitled')}",
            "",
            f"- Session ID: `{session_id}`",
            f"- Backend: {session.get('backend') or 'unknown'}",
            f"- Created (UTC): {session.get('created_at', '')}",
            f"- Last activity (UTC): {session.get('updated_at', '')}",
            f"- Messages: {len(messages)}",
            "",
        ]
        for message in messages:
            stamp = message["created_at"]
            if message["role"] == "user":
                lines += [f"## Question — {stamp}", "", message["content"], ""]
                continue

            lines += [f"## Answer — {stamp}", ""]
            if message["abstained"]:
                lines += [
                    f"**Abstained** (`{message['error_code'] or 'insufficient_evidence'}`).",
                    "",
                ]
            if message["unverified"]:
                lines += [
                    "**Unverified figure(s):** " + ", ".join(message["unverified"]),
                    "",
                ]
            lines += [message["content"], ""]

            if message["citations"]:
                lines += ["### Citations", ""]
                for citation in message["citations"]:
                    where = citation.get("location") or ""
                    lines.append(
                        f"- [{citation['citation_id']}] {citation['filename']}"
                        f" · {citation['modality']}"
                        + (f" · {where}" if where else "")
                    )
                lines.append("")

            evidence = (message["trace"] or {}).get("evidence", [])
            if evidence:
                lines += [
                    "### Evidence retrieved",
                    "",
                    "| Rank | Source | Modality | Location | Retrieval | Rerank |",
                    "| --- | --- | --- | --- | --- | --- |",
                ]
                for item in evidence:
                    lines.append(
                        f"| {item['rank']} | {item['filename']} | {item['modality']} | "
                        f"{item.get('location') or '-'} | {item['retrieval_score']} | "
                        f"{item['rerank_score']} |"
                    )
                lines.append("")

            timings = message["timings"] or {}
            if timings:
                lines += [
                    "### Stage timings (ms)",
                    "",
                    *[f"- {stage}: {value:.0f}" for stage, value in timings.items()],
                    "",
                ]
        return "\n".join(lines)

    def export_csv(self, session_id: str) -> str:
        """One row per message, flat enough for a spreadsheet or an audit
        review. Evidence is summarised rather than exploded, so a session stays
        readable at a glance."""
        session = self.get_session(session_id) or {}
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow([
            "session_id", "session_title", "backend", "created_at", "role",
            "abstained", "error_code", "confidence", "unverified_numbers",
            "cited_files", "evidence_files", "content",
        ])
        for message in self.messages(session_id):
            trace = message["trace"] or {}
            writer.writerow([
                session_id,
                session.get("title", ""),
                session.get("backend") or "",
                message["created_at"],
                message["role"],
                int(message["abstained"]),
                message["error_code"] or "",
                "" if message["confidence"] is None else f"{message['confidence']:.4f}",
                "; ".join(message["unverified"]),
                "; ".join(c["filename"] for c in message["citations"]),
                "; ".join(
                    f"#{item['rank']} {item['filename']}"
                    for item in trace.get("evidence", [])
                ),
                message["content"],
            ])
        return buffer.getvalue()


def get_history() -> ChatHistory:
    """Process-wide singleton, mirroring get_pipeline()."""
    global _history
    with _history_lock:
        if _history is None:
            _history = ChatHistory()
        return _history
