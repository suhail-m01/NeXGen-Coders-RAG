"""
vector_store.py -- persistent local storage.

    FAISS   IndexFlatIP over L2-normalized 384-d vectors (inner product ==
            cosine). Exact search: no training step, no recall loss, nothing
            to tune, and faster than an approximate index at this scale.
    SQLite  sources, content items, and deterministic relationships. It is
            the authority mapping a FAISS row ordinal back to a ContentItem.

Both are written on every ingest, so the index survives a restart.
Everything is local; no remote vector database is involved.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path

import numpy as np

from config import settings
from schemas import (
    ContentItem,
    Modality,
    Relationship,
    RelationshipType,
    Source,
    SourceLocation,
    SourceType,
)

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    source_id   TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    source_type TEXT NOT NULL,
    file_path   TEXT NOT NULL,
    mime_type   TEXT,
    file_size   INTEGER,
    collection  TEXT NOT NULL DEFAULT 'general'
);

CREATE TABLE IF NOT EXISTS items (
    item_id   TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    modality  TEXT NOT NULL,
    content   TEXT NOT NULL,
    location  TEXT NOT NULL,
    parent_id TEXT,
    vector_id INTEGER UNIQUE,
    metadata  TEXT NOT NULL,
    FOREIGN KEY (source_id) REFERENCES sources(source_id)
);

CREATE TABLE IF NOT EXISTS relationships (
    relationship_id   TEXT PRIMARY KEY,
    source_item_id    TEXT NOT NULL,
    target_item_id    TEXT NOT NULL,
    relationship_type TEXT NOT NULL,
    confidence        REAL,
    metadata          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_items_source    ON items(source_id);
CREATE INDEX IF NOT EXISTS idx_items_vector    ON items(vector_id);
CREATE INDEX IF NOT EXISTS idx_rel_source_item ON relationships(source_item_id);
CREATE INDEX IF NOT EXISTS idx_rel_target_item ON relationships(target_item_id);
"""


class VectorStore:
    """FAISS + SQLite behind one interface."""

    def __init__(self, index_path: Path | None = None, db_path: Path | None = None):
        self.index_path = Path(index_path or settings.faiss_path)
        self.db_path = Path(db_path or settings.db_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._conn.commit()
        self._index = self._load_or_create_index()

    # ---- index lifecycle -------------------------------------------------

    def _migrate(self) -> None:
        """Add columns that post-date an existing database.

        Indexes are rebuilt often enough that a migration path feels like
        overkill, but chat history and a curated corpus are not: an index built
        before collections existed must keep working rather than crash on
        INSERT arity.
        """
        columns = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(sources)")
        }
        if "collection" not in columns:
            logger.info("Migrating sources table: adding 'collection' column")
            self._conn.execute(
                "ALTER TABLE sources ADD COLUMN collection TEXT NOT NULL DEFAULT 'general'"
            )
        # Created here, not in SCHEMA: on a legacy database SCHEMA runs before
        # the ALTER, and an index on a column that does not exist yet is an
        # error rather than a no-op.
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sources_collection ON sources(collection)"
        )

    def _load_or_create_index(self):
        import faiss

        dim = settings.embedding_dim
        if self.index_path.exists():
            try:
                index = faiss.read_index(str(self.index_path))
                if index.d != dim:
                    logger.warning(
                        "Index dim %s != configured %s; rebuilding empty index.",
                        index.d, dim,
                    )
                    return faiss.IndexFlatIP(dim)
                return index
            except Exception as exc:
                logger.error("Failed to read FAISS index, starting fresh: %s", exc)
        return faiss.IndexFlatIP(dim)

    def save(self) -> None:
        import faiss

        with self._lock:
            faiss.write_index(self._index, str(self.index_path))
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.commit()
            self._conn.close()

    # ---- writes -----------------------------------------------------------

    def add_source(self, source: Source) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO sources VALUES (?,?,?,?,?,?,?)",
                (
                    source.source_id, source.filename, source.source_type.value,
                    source.file_path, source.mime_type, source.file_size,
                    source.collection,
                ),
            )
            self._conn.commit()

    def delete_source(self, source_id: str) -> int:
        """Remove a source, its items, and every relationship touching them.

        Returns the number of items removed (0 if the source was unknown).

        The index is rebuilt rather than pruned: IndexFlat stores vectors
        positionally (add_items derives vector_id from ntotal), so removing rows
        in place would shift every later vector and silently map searches to the
        wrong items. Reconstructing the survivors keeps the mapping exact for a
        cost linear in the remaining item count.

        The original file on disk is left alone, and chat history keeps its
        stored trace -- a deleted source can still be audited after the fact.
        """
        import faiss

        with self._lock:
            victim = self._conn.execute(
                "SELECT item_id, vector_id FROM items WHERE source_id = ?",
                (source_id,),
            ).fetchall()
            known = self._conn.execute(
                "SELECT 1 FROM sources WHERE source_id = ?", (source_id,)
            ).fetchone()
            if not victim and not known:
                return 0

            dropped = {
                row["vector_id"] for row in victim if row["vector_id"] is not None
            }
            survivors = sorted(
                (
                    (row["item_id"], row["vector_id"])
                    for row in self._conn.execute(
                        "SELECT item_id, vector_id FROM items"
                    ).fetchall()
                    if row["vector_id"] is not None and row["vector_id"] not in dropped
                ),
                key=lambda pair: pair[1],
            )

            vectors = [
                self._index.reconstruct(int(vector_id)) for _, vector_id in survivors
            ]
            rebuilt = faiss.IndexFlatIP(settings.embedding_dim)
            if vectors:
                rebuilt.add(
                    np.ascontiguousarray(np.vstack(vectors), dtype=np.float32)
                )
            self._index = rebuilt

            item_ids = [row["item_id"] for row in victim]
            if item_ids:
                placeholders = ",".join("?" * len(item_ids))
                self._conn.execute(
                    f"DELETE FROM relationships WHERE source_item_id IN ({placeholders}) "
                    f"OR target_item_id IN ({placeholders})",
                    item_ids + item_ids,
                )
                self._conn.execute(
                    f"DELETE FROM items WHERE item_id IN ({placeholders})", item_ids
                )
            self._conn.execute("DELETE FROM sources WHERE source_id = ?", (source_id,))

            for new_id, (item_id, _) in enumerate(survivors):
                self._conn.execute(
                    "UPDATE items SET vector_id = ? WHERE item_id = ?",
                    (new_id, item_id),
                )
            self._conn.commit()
            self.save()
            logger.info(
                "Deleted source %s: %d item(s) removed, %d renumbered",
                source_id, len(item_ids), len(survivors),
            )
            return len(item_ids)

    def source_exists(self, source_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM sources WHERE source_id = ?", (source_id,)
        ).fetchone()
        return row is not None

    def add_items(self, items: list[ContentItem], vectors: np.ndarray) -> list[ContentItem]:
        """Append vectors to FAISS and persist items with their row ordinals.
        Items whose item_id already exists are skipped, making re-ingestion of
        an identical file idempotent."""
        if not items:
            return []
        if len(items) != len(vectors):
            raise ValueError(
                f"items ({len(items)}) and vectors ({len(vectors)}) length mismatch"
            )

        with self._lock:
            existing = {
                row["item_id"]
                for row in self._conn.execute(
                    "SELECT item_id FROM items WHERE item_id IN "
                    f"({','.join('?' * len(items))})",
                    [i.item_id for i in items],
                )
            }
            fresh_items, fresh_vectors = [], []
            for item, vector in zip(items, vectors):
                if item.item_id not in existing:
                    fresh_items.append(item)
                    fresh_vectors.append(vector)

            if not fresh_items:
                return []

            start = self._index.ntotal
            self._index.add(np.ascontiguousarray(
                np.vstack(fresh_vectors), dtype=np.float32
            ))

            stored: list[ContentItem] = []
            for offset, item in enumerate(fresh_items):
                item.vector_id = start + offset
                self._conn.execute(
                    "INSERT OR REPLACE INTO items VALUES (?,?,?,?,?,?,?,?)",
                    (
                        item.item_id, item.source_id, item.modality.value,
                        item.content, item.location.model_dump_json(),
                        item.parent_id, item.vector_id, json.dumps(item.metadata),
                    ),
                )
                stored.append(item)
            self._conn.commit()
            self.save()
            return stored

    def add_relationships(self, relationships: list[Relationship]) -> None:
        if not relationships:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO relationships VALUES (?,?,?,?,?,?)",
                [
                    (
                        r.relationship_id, r.source_item_id, r.target_item_id,
                        r.relationship_type.value, r.confidence,
                        json.dumps(r.metadata),
                    )
                    for r in relationships
                ],
            )
            self._conn.commit()

    # ---- reads -------------------------------------------------------------

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[ContentItem, float]]:
        """Cosine search. Returns (item, score) ordered best-first."""
        if self._index.ntotal == 0:
            return []
        vector = np.ascontiguousarray(
            query_vector.reshape(1, -1), dtype=np.float32
        )
        scores, indices = self._index.search(vector, min(top_k, self._index.ntotal))

        out: list[tuple[ContentItem, float]] = []
        for score, vector_id in zip(scores[0], indices[0]):
            if vector_id < 0:
                continue
            item = self.get_item_by_vector_id(int(vector_id))
            if item is not None:
                out.append((item, float(score)))
        return out

    def _row_to_item(self, row: sqlite3.Row) -> ContentItem:
        return ContentItem(
            item_id=row["item_id"],
            source_id=row["source_id"],
            modality=Modality(row["modality"]),
            content=row["content"],
            location=SourceLocation.model_validate_json(row["location"]),
            parent_id=row["parent_id"],
            vector_id=row["vector_id"],
            metadata=json.loads(row["metadata"]),
        )

    def get_item_by_vector_id(self, vector_id: int) -> ContentItem | None:
        row = self._conn.execute(
            "SELECT * FROM items WHERE vector_id = ?", (vector_id,)
        ).fetchone()
        return self._row_to_item(row) if row else None

    def get_item(self, item_id: str) -> ContentItem | None:
        row = self._conn.execute(
            "SELECT * FROM items WHERE item_id = ?", (item_id,)
        ).fetchone()
        return self._row_to_item(row) if row else None

    def get_items_by_source(self, source_id: str) -> list[ContentItem]:
        rows = self._conn.execute(
            "SELECT * FROM items WHERE source_id = ? ORDER BY vector_id", (source_id,)
        ).fetchall()
        return [self._row_to_item(r) for r in rows]

    def all_items(self) -> list[ContentItem]:
        """Every indexed item, in vector_id order. Used by the lexical leg."""
        rows = self._conn.execute(
            "SELECT * FROM items ORDER BY vector_id"
        ).fetchall()
        return [self._row_to_item(r) for r in rows]

    def items_signature(self) -> tuple[int, int]:
        """Cheap change detection for cached derived indexes (e.g. BM25).

        (row count, highest rowid) changes on any insert or delete, which is
        all a cache needs; re-hashing every document would cost far more than
        the index it protects.
        """
        row = self._conn.execute(
            "SELECT COUNT(*), COALESCE(MAX(rowid), 0) FROM items"
        ).fetchone()
        return (int(row[0]), int(row[1]))

    def get_source(self, source_id: str) -> Source | None:
        row = self._conn.execute(
            "SELECT * FROM sources WHERE source_id = ?", (source_id,)
        ).fetchone()
        if not row:
            return None
        return self._source_from_row(row)

    @staticmethod
    def _source_from_row(row) -> Source:
        return Source(
            source_id=row["source_id"],
            filename=row["filename"],
            source_type=SourceType(row["source_type"]),
            file_path=row["file_path"],
            mime_type=row["mime_type"],
            file_size=row["file_size"],
            collection=row["collection"] or "general",
        )

    def list_sources(self, collection: str | None = None) -> list[Source]:
        """Every source, or just those in one collection."""
        if collection:
            rows = self._conn.execute(
                "SELECT * FROM sources WHERE collection = ? ORDER BY filename",
                (collection,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM sources ORDER BY filename"
            ).fetchall()
        return [self._source_from_row(row) for row in rows]

    def sources_by_filename(self, filename: str) -> list[Source]:
        """Every indexed version of one filename.

        A source_id is a content hash, so editing a file and re-ingesting it
        produces a *second* source. Without this lookup the stale version stays
        indexed and keeps being cited next to the new one.
        """
        rows = self._conn.execute(
            "SELECT * FROM sources WHERE filename = ? ORDER BY rowid", (filename,)
        ).fetchall()
        return [self._source_from_row(row) for row in rows]

    def duplicate_filenames(self) -> dict[str, list[Source]]:
        """Filenames with more than one indexed version, newest last."""
        out: dict[str, list[Source]] = {}
        for source in self.list_sources():
            out.setdefault(source.filename, []).append(source)
        return {name: versions for name, versions in out.items() if len(versions) > 1}

    def list_collections(self) -> list[dict]:
        """Collections present in the index, with source counts."""
        rows = self._conn.execute(
            "SELECT collection, COUNT(*) AS sources FROM sources "
            "GROUP BY collection ORDER BY collection"
        ).fetchall()
        return [{"collection": row["collection"], "sources": row["sources"]} for row in rows]

    def source_ids_in_collection(self, collection: str) -> set[str]:
        rows = self._conn.execute(
            "SELECT source_id FROM sources WHERE collection = ?", (collection,)
        ).fetchall()
        return {row["source_id"] for row in rows}

    def get_relationships_for_items(self, item_ids: list[str]) -> list[Relationship]:
        """Deterministic relationships touching any of the given items."""
        if not item_ids:
            return []
        placeholders = ",".join("?" * len(item_ids))
        rows = self._conn.execute(
            f"SELECT * FROM relationships WHERE source_item_id IN ({placeholders}) "
            f"OR target_item_id IN ({placeholders})",
            item_ids + item_ids,
        ).fetchall()
        return [
            Relationship(
                relationship_id=r["relationship_id"],
                source_item_id=r["source_item_id"],
                target_item_id=r["target_item_id"],
                relationship_type=RelationshipType(r["relationship_type"]),
                confidence=r["confidence"],
                metadata=json.loads(r["metadata"]),
            )
            for r in rows
        ]

    # ---- stats --------------------------------------------------------------

    def stats(self) -> dict:
        item_count = self._conn.execute("SELECT COUNT(*) c FROM items").fetchone()["c"]
        source_count = self._conn.execute("SELECT COUNT(*) c FROM sources").fetchone()["c"]
        rel_count = self._conn.execute(
            "SELECT COUNT(*) c FROM relationships"
        ).fetchone()["c"]
        by_modality = {
            row["modality"]: row["c"]
            for row in self._conn.execute(
                "SELECT modality, COUNT(*) c FROM items GROUP BY modality"
            )
        }
        return {
            "sources": source_count,
            "items": item_count,
            "relationships": rel_count,
            "vectors": int(self._index.ntotal),
            "by_modality": by_modality,
            "index_path": str(self.index_path),
            "db_path": str(self.db_path),
        }
