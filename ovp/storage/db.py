"""PostgreSQL + pgvector storage.

Schema highlights
* ``media``      – one row per uploaded image or video
* ``entities``   – detections (image) / tracks (video) with boxes, mask polygons, VLM output,
                   verification result, identity link; a generated ``tsvector`` column for FTS
* ``embeddings`` – one row per (entity, embedding space). Spaces have different dimensions
                   (CLIP, text, OSNet, ...), so the column is an untyped ``vector`` and every
                   space gets its own partial HNSW expression index created on demand.
* ``persons``    – identity clusters with a centroid per Re-ID space
* ``qa``         – persisted follow-up questions

The legacy schema (``sessions``/``entities`` with ``initial_analysis``) is migrated in place
on first start; the old tables are kept as ``legacy_*``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
from typing import Any, Iterable, Optional, Sequence

import asyncpg
import numpy as np

from ..config import get_settings

log = logging.getLogger("ovp.db")

SCHEMA_VERSION = "2"


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(5)}"


def vec_literal(v: Sequence[float] | np.ndarray) -> str:
    arr = np.asarray(v, dtype=np.float32).reshape(-1)
    return "[" + ",".join(f"{x:.6g}" for x in arr.tolist()) + "]"


def parse_vec(s: Any) -> Optional[np.ndarray]:
    if s is None:
        return None
    if isinstance(s, np.ndarray):
        return s
    return np.asarray(json.loads(s), dtype=np.float32)


def like_escape(s: str) -> str:
    """Escape LIKE wildcards so user input matches literally (default ESCAPE character '\\')."""
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def lit(s: str) -> str:
    """SQL string literal (used for embedding-space names so partial indexes are usable)."""
    return "'" + s.replace("'", "''") + "'"


ENTITY_SELECT = """
SELECT e.*, p.label AS person_label, COALESCE(p.is_watchlist, FALSE) AS person_is_watchlist,
       sp.label AS suggested_label, COALESCE(sp.is_watchlist, FALSE) AS suggested_is_watchlist
FROM entities e
LEFT JOIN persons p ON p.id = e.person_id
LEFT JOIN persons sp ON sp.id = e.suggested_person_id
"""

MEDIA_SELECT = """
SELECT m.*,
  (SELECT COUNT(*) FROM entities e WHERE e.media_id = m.id AND e.stage <> 'rejected') AS entity_count,
  (SELECT COUNT(DISTINCT e.person_id) FROM entities e WHERE e.media_id = m.id AND e.person_id IS NOT NULL) AS person_count
FROM media m
"""

PERSON_SELECT = """
SELECT p.*,
  (SELECT COUNT(DISTINCT e.media_id) FROM entities e WHERE e.person_id = p.id) AS media_count,
  (SELECT MIN(e.created_at) FROM entities e WHERE e.person_id = p.id) AS first_seen,
  (SELECT MAX(e.created_at) FROM entities e WHERE e.person_id = p.id) AS last_seen
FROM persons p
"""

ENTITY_COLUMNS = (
    "id", "media_id", "label", "concept", "score", "x1", "y1", "x2", "y2", "polygon", "mask_area", "crop_path",
    "description", "attributes", "verified", "verify_reason", "stage", "track_id", "t_start", "t_end",
    "frame_index", "trajectory",
)


class Database:
    def __init__(self, url: Optional[str] = None):
        self.url = url or get_settings().database_url
        self.pool: Optional[asyncpg.Pool] = None
        self._vector_indexes: set[str] = set()

    # ── lifecycle ─────────────────────────────────────────────────────────────
    async def connect(self) -> None:
        if self.pool is not None:
            return
        conn = await asyncpg.connect(self.url)
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        finally:
            await conn.close()

        async def _init(c: asyncpg.Connection) -> None:
            for t in ("json", "jsonb"):
                await c.set_type_codec(t, encoder=lambda v: json.dumps(v, default=_json_default), decoder=json.loads, schema="pg_catalog")

        self.pool = await asyncpg.create_pool(self.url, min_size=2, max_size=12, command_timeout=120, init=_init)
        await self._migrate()

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()
            self.pool = None

    # ── schema ────────────────────────────────────────────────────────────────
    async def _migrate(self) -> None:
        assert self.pool
        async with self.pool.acquire() as c:
            async with c.transaction():
                await c.execute("SELECT pg_advisory_xact_lock(727274)")
                legacy = await c.fetchval(
                    "SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'entities' AND column_name = 'initial_analysis')"
                )
                if legacy:
                    log.warning("legacy schema detected – migrating to schema v%s", SCHEMA_VERSION)
                    for t in ("entities", "persons", "sessions"):
                        exists = await c.fetchval("SELECT to_regclass($1) IS NOT NULL", f"public.{t}")
                        if exists:
                            await c.execute(f"ALTER TABLE {t} RENAME TO legacy_{t}")
                await self._create_schema(c)
                if legacy:
                    await self._copy_legacy(c)
                await c.execute(
                    "INSERT INTO ovp_meta(key, value) VALUES ('schema_version', $1) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                    SCHEMA_VERSION,
                )
            rows = await c.fetch("SELECT DISTINCT space, vector_dims(vec) AS dim FROM embeddings")
        for r in rows:
            await self.ensure_vector_index(r["space"], r["dim"])

    async def _create_schema(self, c: asyncpg.Connection) -> None:
        await c.execute(
            """
            CREATE TABLE IF NOT EXISTS ovp_meta (key TEXT PRIMARY KEY, value TEXT);

            CREATE TABLE IF NOT EXISTS media (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                name TEXT,
                path TEXT NOT NULL,
                thumb_path TEXT,
                width INTEGER, height INTEGER,
                duration REAL, fps REAL, frame_count INTEGER,
                query TEXT, plan JSONB, options JSONB,
                status TEXT NOT NULL DEFAULT 'done',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS persons (
                id TEXT PRIMARY KEY,
                label TEXT,
                is_watchlist BOOLEAN NOT NULL DEFAULT FALSE,
                notes TEXT,
                space TEXT,
                centroid vector,
                sighting_count INTEGER NOT NULL DEFAULT 0,
                representative_entity_id TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                media_id TEXT NOT NULL REFERENCES media(id) ON DELETE CASCADE,
                label TEXT NOT NULL,
                concept TEXT,
                score REAL NOT NULL,
                x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL,
                polygon JSONB, mask_area INTEGER,
                crop_path TEXT,
                description TEXT, attributes JSONB,
                verified BOOLEAN, verify_reason TEXT,
                stage TEXT NOT NULL,
                track_id INTEGER, t_start REAL, t_end REAL, frame_index INTEGER, trajectory JSONB,
                person_id TEXT REFERENCES persons(id) ON DELETE SET NULL,
                match_status TEXT, match_score REAL,
                suggested_person_id TEXT REFERENCES persons(id) ON DELETE SET NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                tsv tsvector GENERATED ALWAYS AS (
                    setweight(to_tsvector('english', coalesce(label, '') || ' ' || coalesce(concept, '')), 'A') ||
                    setweight(to_tsvector('english', coalesce(description, '')), 'B')
                ) STORED
            );

            CREATE TABLE IF NOT EXISTS embeddings (
                entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                space TEXT NOT NULL,
                vec vector NOT NULL,
                PRIMARY KEY (entity_id, space)
            );

            CREATE TABLE IF NOT EXISTS qa (
                id BIGSERIAL PRIMARY KEY,
                media_id TEXT REFERENCES media(id) ON DELETE CASCADE,
                entity_id TEXT REFERENCES entities(id) ON DELETE CASCADE,
                scope TEXT NOT NULL DEFAULT 'crop',
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS ovp_media_created ON media(created_at DESC);
            CREATE INDEX IF NOT EXISTS ovp_entities_media ON entities(media_id);
            CREATE INDEX IF NOT EXISTS ovp_entities_person ON entities(person_id);
            CREATE INDEX IF NOT EXISTS ovp_entities_label ON entities(label);
            CREATE INDEX IF NOT EXISTS ovp_entities_tsv ON entities USING gin(tsv);
            CREATE INDEX IF NOT EXISTS ovp_entities_pending ON entities(match_status) WHERE match_status = 'pending';
            CREATE INDEX IF NOT EXISTS ovp_embeddings_space ON embeddings(space);
            CREATE INDEX IF NOT EXISTS ovp_qa_entity ON qa(entity_id);
            CREATE INDEX IF NOT EXISTS ovp_persons_space ON persons(space);
            """
        )

    async def _copy_legacy(self, c: asyncpg.Connection) -> None:
        space = "reid:osnet:osnet_x1_0"
        await c.execute(
            """
            INSERT INTO media (id, kind, name, path, width, height, created_at)
            SELECT id, 'image', id || '.jpg', full_image_path, image_width, image_height, created_at FROM legacy_sessions
            ON CONFLICT DO NOTHING
            """
        )
        has_persons = await c.fetchval("SELECT to_regclass('public.legacy_persons') IS NOT NULL")
        if has_persons:
            person_cols = await self._columns(c, "legacy_persons")
            # installs without pgvector only kept the JSON copy of the centroid
            centroid = "centroid::text::vector" if "centroid" in person_cols else "centroid_json::vector" if "centroid_json" in person_cols else "NULL"
            await c.execute(
                f"""
                INSERT INTO persons (id, label, is_watchlist, notes, space, centroid, sighting_count, representative_entity_id, created_at, updated_at)
                SELECT id, label, is_watchlist, notes, '{space}', {centroid}, sighting_count, representative_entity_id, created_at, updated_at
                FROM legacy_persons ON CONFLICT DO NOTHING
                """
            )
        cols = await self._columns(c, "legacy_entities")
        person = "person_id" if "person_id" in cols else "NULL"
        mscore = "match_score" if "match_score" in cols else "NULL"
        mstatus = "match_status" if "match_status" in cols else "NULL"
        await c.execute(
            f"""
            INSERT INTO entities (id, media_id, label, score, x1, y1, x2, y2, crop_path, description, stage,
                                  person_id, match_score, match_status, created_at)
            SELECT id, session_id, class_name, confidence, box_x1, box_y1, box_x2, box_y2, crop_image_path, initial_analysis,
                   CASE stage WHEN 'vlm_full' THEN 'described' WHEN 'yolo_only' THEN 'detected' ELSE 'low_confidence' END,
                   {person if has_persons else 'NULL'}, {mscore}, {mstatus}, created_at
            FROM legacy_entities WHERE session_id IN (SELECT id FROM media)
            ON CONFLICT DO NOTHING
            """
        )
        if "reid_embedding_json" in cols:
            await c.execute(
                f"""
                INSERT INTO embeddings (entity_id, space, vec)
                SELECT id, '{space}', reid_embedding_json::vector FROM legacy_entities
                WHERE reid_embedding_json IS NOT NULL AND id IN (SELECT id FROM entities)
                ON CONFLICT DO NOTHING
                """
            )
        for pid in [r["id"] for r in await c.fetch("SELECT id FROM persons")]:
            await self._refresh_person(c, pid)  # counts, centroids and representatives from the copied rows
        n = await c.fetchval("SELECT COUNT(*) FROM entities")
        log.warning("legacy migration copied %s entities (legacy tables kept as legacy_*)", n)

    @staticmethod
    async def _columns(c: asyncpg.Connection, table: str) -> set[str]:
        rows = await c.fetch("SELECT column_name FROM information_schema.columns WHERE table_schema = 'public' AND table_name = $1", table)
        return {r["column_name"] for r in rows}

    async def ensure_vector_index(self, space: str, dim: int) -> None:
        """HNSW index for one (space, dimension). The predicate includes the dimension, so a space
        whose model changed size (e.g. ``reid:clip`` after switching CLIP models) keeps working."""
        key = f"{space}|{dim}"
        if key in self._vector_indexes or not dim:
            return
        digest = hashlib.sha1(key.encode()).hexdigest()[:12]
        assert self.pool
        try:
            async with self.pool.acquire() as c:
                await c.execute(f"DROP INDEX IF EXISTS ovp_emb_{digest}")  # v2 index without the dimension predicate
                await c.execute(
                    f"CREATE INDEX IF NOT EXISTS ovp_embd_{digest} ON embeddings USING hnsw ((vec::vector({int(dim)})) vector_cosine_ops) "
                    f"WHERE space = {lit(space)} AND vector_dims(vec) = {int(dim)}"
                )
        except Exception as e:  # pgvector < 0.5 has no HNSW → sequential scan still works
            log.warning("could not create vector index for %s: %s", space, e)
        self._vector_indexes.add(key)

    # ── media ─────────────────────────────────────────────────────────────────
    async def insert_media(self, **f: Any) -> None:
        cols = list(f)
        assert self.pool
        await self.pool.execute(
            f"INSERT INTO media ({', '.join(cols)}) VALUES ({', '.join(f'${i + 1}' for i in range(len(cols)))})",
            *f.values(),
        )

    async def update_media(self, media_id: str, **f: Any) -> None:
        if not f:
            return
        sets = ", ".join(f"{k} = ${i + 2}" for i, k in enumerate(f))
        assert self.pool
        await self.pool.execute(f"UPDATE media SET {sets} WHERE id = $1", media_id, *f.values())

    async def get_media(self, media_id: str) -> Optional[asyncpg.Record]:
        assert self.pool
        return await self.pool.fetchrow(MEDIA_SELECT + " WHERE m.id = $1", media_id)

    async def get_media_many(self, ids: Sequence[str]) -> list[asyncpg.Record]:
        assert self.pool
        return await self.pool.fetch(MEDIA_SELECT + " WHERE m.id = ANY($1::text[])", list(ids))

    # Library listings, counts, stats and scans cover finished media only ('pending', 'error' and
    # 'cancelled' rows are hidden); list_media also shows media still being analysed.
    async def list_media(self, limit: int = 50, offset: int = 0, kind: Optional[str] = None, ids: Optional[Sequence[str]] = None) -> list[asyncpg.Record]:
        where, args = ["m.status IN ('done', 'processing')"], []
        if kind:
            args.append(kind)
            where.append(f"m.kind = ${len(args)}")
        if ids is not None:
            args.append(list(ids))
            where.append(f"m.id = ANY(${len(args)}::text[])")
        args += [limit, offset]
        assert self.pool
        return await self.pool.fetch(
            MEDIA_SELECT + f" WHERE {' AND '.join(where)} ORDER BY m.created_at DESC LIMIT ${len(args) - 1} OFFSET ${len(args)}", *args
        )

    async def count_media(self, kind: Optional[str] = None) -> int:
        assert self.pool
        if kind:
            return await self.pool.fetchval("SELECT COUNT(*) FROM media WHERE kind = $1 AND status = 'done'", kind)
        return await self.pool.fetchval("SELECT COUNT(*) FROM media WHERE status = 'done'")

    async def all_media_ids(self, kind: Optional[str] = None, limit: Optional[int] = None) -> list[str]:
        assert self.pool
        q = "SELECT id FROM media WHERE status = 'done'" + (" AND kind = $1" if kind else "") + " ORDER BY created_at DESC"
        if limit:
            q += f" LIMIT {int(limit)}"
        rows = await (self.pool.fetch(q, kind) if kind else self.pool.fetch(q))
        return [r["id"] for r in rows]

    async def fail_interrupted_media(self) -> int:
        """Mark media left 'processing' by a previous run (crash or shutdown) as 'error'."""
        assert self.pool
        r = await self.pool.execute("UPDATE media SET status = 'error' WHERE status = 'processing'")
        return int(r.split()[-1])

    async def delete_media(self, media_id: str) -> Optional[dict[str, Any]]:
        """Delete media; returns file paths to remove. Person clusters are recomputed."""
        assert self.pool
        async with self.pool.acquire() as c:
            async with c.transaction():
                m = await c.fetchrow("SELECT path, thumb_path FROM media WHERE id = $1", media_id)
                if m is None:
                    return None
                ents = await c.fetch("SELECT crop_path, person_id FROM entities WHERE media_id = $1", media_id)
                await c.execute("DELETE FROM media WHERE id = $1", media_id)
                pids = {r["person_id"] for r in ents if r["person_id"]}
                for pid in pids:
                    await self._refresh_person(c, pid)
        return {"paths": [m["path"], m["thumb_path"], *[r["crop_path"] for r in ents]], "persons": list(pids)}

    # ── entities ──────────────────────────────────────────────────────────────
    async def insert_entities(self, rows: list[dict[str, Any]], embeddings: list[tuple[str, str, np.ndarray]]) -> None:
        if not rows:
            return
        assert self.pool
        cols = ENTITY_COLUMNS
        async with self.pool.acquire() as c:
            async with c.transaction():
                await c.executemany(
                    f"INSERT INTO entities ({', '.join(cols)}) VALUES ({', '.join(f'${i + 1}' for i in range(len(cols)))})",
                    [tuple(r.get(k) for k in cols) for r in rows],
                )
                if embeddings:
                    await c.executemany(
                        "INSERT INTO embeddings (entity_id, space, vec) VALUES ($1, $2, $3::vector) "
                        "ON CONFLICT (entity_id, space) DO UPDATE SET vec = EXCLUDED.vec",
                        [(eid, space, vec_literal(v)) for eid, space, v in embeddings],
                    )
        for space, dim in {(s, len(np.asarray(v).reshape(-1))) for _, s, v in embeddings}:
            await self.ensure_vector_index(space, dim)

    async def upsert_embeddings(self, embeddings: list[tuple[str, str, np.ndarray]]) -> None:
        if not embeddings:
            return
        assert self.pool
        await self.pool.executemany(
            "INSERT INTO embeddings (entity_id, space, vec) VALUES ($1, $2, $3::vector) ON CONFLICT (entity_id, space) DO UPDATE SET vec = EXCLUDED.vec",
            [(eid, space, vec_literal(v)) for eid, space, v in embeddings],
        )
        for space, dim in {(s, len(np.asarray(v).reshape(-1))) for _, s, v in embeddings}:
            await self.ensure_vector_index(space, dim)

    async def get_entity(self, entity_id: str) -> Optional[asyncpg.Record]:
        assert self.pool
        return await self.pool.fetchrow(ENTITY_SELECT + " WHERE e.id = $1", entity_id)

    async def get_entities(self, ids: Sequence[str]) -> list[asyncpg.Record]:
        assert self.pool
        return await self.pool.fetch(ENTITY_SELECT + " WHERE e.id = ANY($1::text[])", list(ids))

    async def media_entities(self, media_id: str) -> list[asyncpg.Record]:
        assert self.pool
        return await self.pool.fetch(ENTITY_SELECT + " WHERE e.media_id = $1 ORDER BY e.t_start NULLS FIRST, e.score DESC", media_id)

    async def media_entities_many(self, media_ids: Sequence[str]) -> list[asyncpg.Record]:
        assert self.pool
        return await self.pool.fetch(ENTITY_SELECT + " WHERE e.media_id = ANY($1::text[]) ORDER BY e.media_id, e.t_start NULLS FIRST, e.score DESC", list(media_ids))

    async def update_entity(self, entity_id: str, **f: Any) -> None:
        if not f:
            return
        sets = ", ".join(f"{k} = ${i + 2}" for i, k in enumerate(f))
        assert self.pool
        await self.pool.execute(f"UPDATE entities SET {sets} WHERE id = $1", entity_id, *f.values())

    async def delete_entity(self, entity_id: str) -> Optional[str]:
        assert self.pool
        async with self.pool.acquire() as c:
            async with c.transaction():
                row = await c.fetchrow("DELETE FROM entities WHERE id = $1 RETURNING crop_path, person_id", entity_id)
                if row is None:
                    return None
                if row["person_id"]:
                    await self._refresh_person(c, row["person_id"])
                return row["crop_path"] or ""

    async def get_embedding(self, entity_id: str, space: str) -> Optional[np.ndarray]:
        assert self.pool
        v = await self.pool.fetchval("SELECT vec::text FROM embeddings WHERE entity_id = $1 AND space = $2", entity_id, space)
        return parse_vec(v)

    async def get_embeddings(self, space: str, entity_ids: Optional[Sequence[str]] = None, only_persons: bool = False) -> list[tuple[str, np.ndarray, str, Optional[str], Optional[str]]]:
        """(entity_id, vec, media_id, person_id, match_status) for a space."""
        q = (
            "SELECT b.entity_id, b.vec::text AS vec, e.media_id, e.person_id, e.match_status FROM embeddings b "
            "JOIN entities e ON e.id = b.entity_id WHERE b.space = $1"
        )
        args: list[Any] = [space]
        if entity_ids is not None:
            args.append(list(entity_ids))
            q += " AND b.entity_id = ANY($2::text[])"
        if only_persons:
            q += " AND e.label = 'person' AND e.stage <> 'rejected'"
        assert self.pool
        rows = await self.pool.fetch(q, *args)
        return [(r["entity_id"], parse_vec(r["vec"]), r["media_id"], r["person_id"], r["match_status"]) for r in rows]  # type: ignore[misc]

    async def spaces(self) -> list[tuple[str, int, int]]:
        assert self.pool
        rows = await self.pool.fetch("SELECT space, MIN(vector_dims(vec)) AS dim, COUNT(*) AS n FROM embeddings GROUP BY space")
        return [(r["space"], r["dim"], r["n"]) for r in rows]

    # ── search primitives ─────────────────────────────────────────────────────
    @staticmethod
    def _filters(filters: dict[str, Any], args: list[Any], alias: str = "e") -> str:
        clauses = [f"{alias}.stage <> 'rejected'"]
        if filters.get("media_id"):
            args.append(filters["media_id"])
            clauses.append(f"{alias}.media_id = ${len(args)}")
        if filters.get("media_ids"):
            args.append(list(filters["media_ids"]))
            clauses.append(f"{alias}.media_id = ANY(${len(args)}::text[])")
        if filters.get("person_id"):
            args.append(filters["person_id"])
            clauses.append(f"{alias}.person_id = ${len(args)}")
        if filters.get("labels"):
            args.append([x.lower() for x in filters["labels"]])
            clauses.append(f"lower({alias}.label) = ANY(${len(args)}::text[])")
        if filters.get("kind"):
            args.append(filters["kind"])
            clauses.append(f"EXISTS (SELECT 1 FROM media mm WHERE mm.id = {alias}.media_id AND mm.kind = ${len(args)})")
        return " AND ".join(clauses)

    async def fts(self, text: str, limit: int, filters: dict[str, Any]) -> list[tuple[str, float]]:
        args: list[Any] = [text]
        where = self._filters(filters, args)
        args.append(limit)
        assert self.pool
        rows = await self.pool.fetch(
            f"""
            WITH q AS (SELECT websearch_to_tsquery('english', $1) AS a, replace(plainto_tsquery('english', $1)::text, '&', '|')::tsquery AS b)
            SELECT e.id, GREATEST(ts_rank_cd(e.tsv, q.a) * 2, ts_rank_cd(e.tsv, q.b)) AS rank
            FROM entities e, q
            WHERE (e.tsv @@ q.a OR e.tsv @@ q.b) AND {where}
            ORDER BY rank DESC LIMIT ${len(args)}
            """,
            *args,
        )
        return [(r["id"], float(r["rank"])) for r in rows]

    async def label_match(self, terms: list[str], limit: int, filters: dict[str, Any]) -> list[tuple[str, float]]:
        if not terms:
            return []
        args: list[Any] = [[t.lower() for t in terms], [f"%{like_escape(t.lower())}%" for t in terms]]
        where = self._filters(filters, args)
        args.append(limit)
        assert self.pool
        rows = await self.pool.fetch(
            f"""
            SELECT e.id, CASE WHEN lower(e.label) = ANY($1::text[]) OR lower(coalesce(e.concept, '')) = ANY($1::text[]) THEN 1.0 ELSE 0.6 END AS s
            FROM entities e
            WHERE (lower(e.label) LIKE ANY($2::text[]) OR lower(coalesce(e.concept, '')) LIKE ANY($2::text[])) AND {where}
            ORDER BY s DESC, e.score DESC LIMIT ${len(args)}
            """,
            *args,
        )
        return [(r["id"], float(r["s"])) for r in rows]

    async def knn(self, space: str, vec: np.ndarray, limit: int, filters: dict[str, Any], min_sim: float = -1.0) -> list[tuple[str, float]]:
        dim = int(np.asarray(vec).reshape(-1).shape[0])
        args: list[Any] = [vec_literal(vec)]
        where = self._filters(filters, args)
        args.append(limit)
        assert self.pool
        rows = await self.pool.fetch(
            f"""
            SELECT b.entity_id AS id, 1 - ((b.vec::vector({dim})) <=> $1::vector({dim})) AS sim
            FROM embeddings b JOIN entities e ON e.id = b.entity_id
            WHERE b.space = {lit(space)} AND vector_dims(b.vec) = {dim} AND {where}
            ORDER BY (b.vec::vector({dim})) <=> $1::vector({dim}) LIMIT ${len(args)}
            """,
            *args,
        )
        return [(r["id"], float(r["sim"])) for r in rows if float(r["sim"]) >= min_sim]

    # ── persons ───────────────────────────────────────────────────────────────
    async def match_persons(self, space: str, vec: np.ndarray, k: int = 8, exclude: Sequence[str] = ()) -> list[tuple[str, float]]:
        dim = int(np.asarray(vec).reshape(-1).shape[0])
        assert self.pool
        rows = await self.pool.fetch(
            f"""
            SELECT id, 1 - ((centroid::vector({dim})) <=> $1::vector({dim})) AS sim FROM persons
            WHERE space = $2 AND centroid IS NOT NULL AND vector_dims(centroid) = {dim} AND NOT (id = ANY($3::text[]))
            ORDER BY (centroid::vector({dim})) <=> $1::vector({dim}) LIMIT $4
            """,
            vec_literal(vec), space, list(exclude), k,
        )
        return [(r["id"], float(r["sim"])) for r in rows]

    async def create_person(self, space: str, centroid: Optional[np.ndarray], representative: Optional[str], label: Optional[str] = None, conn: Optional[asyncpg.Connection] = None) -> str:
        pid = new_id("p")
        c = conn or self.pool
        assert c is not None
        await c.execute(
            "INSERT INTO persons (id, label, space, centroid, representative_entity_id) VALUES ($1, $2, $3, $4::vector, $5)",
            pid, label, space, vec_literal(centroid) if centroid is not None else None, representative,
        )
        return pid

    async def media_sightings(self, entity_ids: Sequence[str]) -> list[asyncpg.Record]:
        """``entity_ids`` plus the person-linked sightings that share a media item with them
        (id, person_id and the timing columns used to tell whether two sightings co-occur)."""
        assert self.pool
        return await self.pool.fetch(
            """
            SELECT e.id, e.person_id, e.t_start, e.t_end, e.trajectory FROM entities e
            WHERE e.media_id IN (SELECT media_id FROM entities WHERE id = ANY($1::text[]))
              AND (e.id = ANY($1::text[]) OR (e.person_id IS NOT NULL AND e.stage <> 'rejected'))
            """,
            list(entity_ids),
        )

    async def link_entity(self, entity_id: str, person_id: Optional[str], status: Optional[str], score: Optional[float], suggested: Optional[str] = None) -> None:
        assert self.pool
        async with self.pool.acquire() as c:
            async with c.transaction():
                old = await c.fetchval("SELECT person_id FROM entities WHERE id = $1", entity_id)
                await c.execute(
                    "UPDATE entities SET person_id = $2, match_status = $3, match_score = $4, suggested_person_id = $5 WHERE id = $1",
                    entity_id, person_id, status, score, suggested,
                )
                for pid in {old, person_id} - {None}:
                    await self._refresh_person(c, pid)  # type: ignore[arg-type]

    async def refresh_person(self, person_id: str) -> None:
        assert self.pool
        async with self.pool.acquire() as c:
            await self._refresh_person(c, person_id)

    async def _refresh_person(self, c: asyncpg.Connection, person_id: str) -> None:
        """Recompute centroid (pgvector avg), count and representative; drop empty unnamed clusters."""
        p = await c.fetchrow("SELECT space, label, is_watchlist, notes FROM persons WHERE id = $1", person_id)
        if p is None:
            return
        n = await c.fetchval("SELECT COUNT(*) FROM entities WHERE person_id = $1", person_id)
        if n == 0 and not p["label"] and not p["is_watchlist"] and not p["notes"]:
            await c.execute("DELETE FROM persons WHERE id = $1", person_id)
            return
        dim = await c.fetchval(
            "SELECT vector_dims(b.vec) FROM embeddings b JOIN entities e ON e.id = b.entity_id "
            "WHERE e.person_id = $1 AND b.space = $2 ORDER BY e.created_at DESC LIMIT 1",
            person_id, p["space"],
        )
        # without embeddings in the person's space (e.g. a named person whose sightings were all
        # deleted) the last centroid is kept so the identity can still be matched
        centroid_sql = (
            f"(SELECT AVG(b.vec::vector({int(dim)})) FROM embeddings b JOIN entities e ON e.id = b.entity_id "
            f"WHERE e.person_id = $1 AND b.space = $2 AND vector_dims(b.vec) = {int(dim)})"
            if dim else "centroid"
        )
        await c.execute(
            f"""
            UPDATE persons SET
              centroid = {centroid_sql},
              sighting_count = (SELECT COUNT(*) FROM entities WHERE person_id = $1),
              representative_entity_id = (
                SELECT e.id FROM entities e WHERE e.person_id = $1
                ORDER BY (e.x2 - e.x1) * (e.y2 - e.y1) * e.score DESC LIMIT 1),
              updated_at = NOW()
            WHERE id = $1
            """,
            *((person_id, p["space"]) if dim else (person_id,)),
        )

    async def get_person(self, person_id: str) -> Optional[asyncpg.Record]:
        assert self.pool
        return await self.pool.fetchrow(PERSON_SELECT + " WHERE p.id = $1", person_id)

    async def get_persons(self, ids: Sequence[str]) -> list[asyncpg.Record]:
        assert self.pool
        return await self.pool.fetch(PERSON_SELECT + " WHERE p.id = ANY($1::text[])", list(ids))

    async def list_persons(self, watchlist: bool = False, q: Optional[str] = None, limit: int = 100, offset: int = 0, order: str = "recent") -> tuple[list[asyncpg.Record], int]:
        where, args = ["TRUE"], []
        if watchlist:
            where.append("p.is_watchlist")
        if q:
            args.append(f"%{like_escape(q.lower())}%")
            where.append(f"(lower(coalesce(p.label, '')) LIKE ${len(args)} OR lower(coalesce(p.notes, '')) LIKE ${len(args)} OR p.id LIKE ${len(args)})")
        w = " AND ".join(where)
        assert self.pool
        total = await self.pool.fetchval(f"SELECT COUNT(*) FROM persons p WHERE {w}", *args)
        order_sql = "p.sighting_count DESC, p.updated_at DESC" if order == "sightings" else "p.is_watchlist DESC, (p.label IS NULL), p.updated_at DESC"
        args += [limit, offset]
        rows = await self.pool.fetch(PERSON_SELECT + f" WHERE {w} ORDER BY {order_sql} LIMIT ${len(args) - 1} OFFSET ${len(args)}", *args)
        return rows, int(total)

    async def find_persons_by_label(self, label: str) -> list[asyncpg.Record]:
        """Persons whose label equals ``label`` (case-insensitive, surrounding spaces ignored)."""
        assert self.pool
        return await self.pool.fetch(PERSON_SELECT + " WHERE lower(p.label) = lower(trim($1)) ORDER BY p.sighting_count DESC", label)

    async def update_person(self, person_id: str, label: Optional[str] = None, notes: Optional[str] = None, is_watchlist: Optional[bool] = None) -> bool:
        """Update the given fields; ``None`` leaves a field unchanged and an empty string clears
        ``label``/``notes``. Returns whether the person exists."""
        f: dict[str, Any] = {k: (v or None) for k, v in (("label", label), ("notes", notes)) if v is not None}
        if is_watchlist is not None:
            f["is_watchlist"] = is_watchlist
        if not f:
            return await self.get_person(person_id) is not None
        sets = ", ".join(f"{k} = ${i + 2}" for i, k in enumerate(f))
        assert self.pool
        r = await self.pool.execute(f"UPDATE persons SET {sets}, updated_at = NOW() WHERE id = $1", person_id, *f.values())
        return r.endswith("1")

    async def delete_person(self, person_id: str) -> bool:
        assert self.pool
        async with self.pool.acquire() as c:
            async with c.transaction():
                await c.execute("UPDATE entities SET person_id = NULL, match_status = NULL, match_score = NULL WHERE person_id = $1", person_id)
                r = await c.execute("DELETE FROM persons WHERE id = $1", person_id)
                return r.endswith("1")

    async def merge_persons(self, keep: str, other: str) -> bool:
        if keep == other:
            return False
        assert self.pool
        async with self.pool.acquire() as c:
            async with c.transaction():
                rows = await c.fetch("SELECT id, label, notes, is_watchlist FROM persons WHERE id = ANY($1::text[])", [keep, other])
                if len(rows) != 2:
                    return False
                o = next(r for r in rows if r["id"] == other)
                await c.execute("UPDATE entities SET person_id = $1, match_status = 'manual' WHERE person_id = $2", keep, other)
                await c.execute("UPDATE entities SET suggested_person_id = $1 WHERE suggested_person_id = $2", keep, other)
                await c.execute(
                    "UPDATE persons SET label = COALESCE(label, $2), notes = COALESCE(notes, $3), is_watchlist = is_watchlist OR $4 WHERE id = $1",
                    keep, o["label"], o["notes"], o["is_watchlist"],
                )
                await c.execute("DELETE FROM persons WHERE id = $1", other)
                await self._refresh_person(c, keep)
        return True

    async def person_sightings(self, person_id: str, limit: int = 500) -> list[asyncpg.Record]:
        assert self.pool
        return await self.pool.fetch(ENTITY_SELECT + " WHERE e.person_id = $1 ORDER BY e.created_at DESC LIMIT $2", person_id, limit)

    async def pending_entities(self, limit: int = 200) -> list[asyncpg.Record]:
        assert self.pool
        return await self.pool.fetch(ENTITY_SELECT + " WHERE e.match_status = 'pending' ORDER BY e.created_at DESC LIMIT $1", limit)

    async def apply_clustering(self, assignments: dict[str, Optional[str]], new_clusters: list[list[str]], space: str) -> int:
        """Persist a re-clustering: ``assignments`` entity→existing person; ``new_clusters`` lists of entity ids."""
        assert self.pool
        changed = 0
        async with self.pool.acquire() as c:
            async with c.transaction():
                touched: set[str] = set()
                cur = {r["id"]: r["person_id"] for r in await c.fetch("SELECT id, person_id FROM entities WHERE id = ANY($1::text[])", list(assignments) + [e for cl in new_clusters for e in cl])}
                for eid, pid in assignments.items():
                    if cur.get(eid) != pid:
                        changed += 1
                        touched |= {x for x in (cur.get(eid), pid) if x}
                        await c.execute("UPDATE entities SET person_id = $2, match_status = 'matched', match_score = NULL, suggested_person_id = NULL WHERE id = $1", eid, pid)
                for cl in new_clusters:
                    pid = await self.create_person(space, None, cl[0], conn=c)
                    touched.add(pid)
                    for eid in cl:
                        touched |= {cur[eid]} if cur.get(eid) else set()
                        changed += 1
                    await c.execute(
                        "UPDATE entities SET person_id = $2, match_status = CASE WHEN array_length($1::text[], 1) > 1 THEN 'matched' ELSE 'new' END, match_score = NULL, suggested_person_id = NULL WHERE id = ANY($1::text[])",
                        cl, pid,
                    )
                for pid in touched:
                    await self._refresh_person(c, pid)
        return changed

    # ── Q&A ───────────────────────────────────────────────────────────────────
    async def add_qa(self, media_id: str, entity_id: Optional[str], scope: str, question: str, answer: str) -> None:
        assert self.pool
        await self.pool.execute("INSERT INTO qa (media_id, entity_id, scope, question, answer) VALUES ($1, $2, $3, $4, $5)", media_id, entity_id, scope, question, answer)

    async def list_qa(self, entity_id: Optional[str] = None, media_id: Optional[str] = None) -> list[asyncpg.Record]:
        assert self.pool
        if entity_id:
            return await self.pool.fetch("SELECT * FROM qa WHERE entity_id = $1 ORDER BY created_at", entity_id)
        return await self.pool.fetch("SELECT * FROM qa WHERE media_id = $1 AND entity_id IS NULL ORDER BY created_at", media_id)

    # ── stats ─────────────────────────────────────────────────────────────────
    async def stats(self) -> dict[str, Any]:
        assert self.pool
        row = await self.pool.fetchrow(
            """
            SELECT (SELECT COUNT(*) FROM media WHERE status = 'done') AS media,
                   (SELECT COUNT(*) FROM media WHERE kind = 'image' AND status = 'done') AS images,
                   (SELECT COUNT(*) FROM media WHERE kind = 'video' AND status = 'done') AS videos,
                   (SELECT COUNT(*) FROM entities WHERE stage <> 'rejected') AS entities,
                   (SELECT COUNT(*) FROM persons) AS persons,
                   (SELECT COUNT(*) FROM persons WHERE is_watchlist) AS watchlist,
                   (SELECT COUNT(*) FROM entities WHERE match_status = 'pending') AS pending
            """
        )
        labels = await self.pool.fetch("SELECT label, COUNT(*) AS n FROM entities WHERE stage <> 'rejected' GROUP BY label ORDER BY n DESC LIMIT 50")
        return {**dict(row), "labels": [{"label": r["label"], "count": r["n"]} for r in labels]}

    async def ping(self) -> bool:
        try:
            assert self.pool
            await self.pool.fetchval("SELECT 1")
            return True
        except Exception:
            return False


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


_db: Optional[Database] = None


async def get_db() -> Database:
    global _db
    if _db is None:
        _db = Database()
        await _db.connect()
    return _db


def reset_db(db: Optional[Database] = None) -> None:
    global _db
    _db = db


def rows_by_id(rows: Iterable[asyncpg.Record]) -> dict[str, asyncpg.Record]:
    return {r["id"]: r for r in rows}
