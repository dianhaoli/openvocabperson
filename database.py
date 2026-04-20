# database.py
"""
Database layer using PostgreSQL with asyncpg for async operations.
Handles all database operations including hybrid search with pgvector.

Features:
    - Session and entity CRUD operations
    - Full-text search on analysis text (PostgreSQL tsvector)
    - Vector similarity search on image embeddings (pgvector)
    - Hybrid search combining text and vector scores with weighted scoring

Hybrid Search Architecture:
    1. Text Search: PostgreSQL full-text search with ts_rank scoring
    2. Vector Search: pgvector cosine similarity (1 - distance)
    3. Hybrid Score: weighted_score = (text_weight * text_score) + (vector_weight * vector_score)

Requirements:
    - PostgreSQL 15+
    - pgvector extension: CREATE EXTENSION vector;
"""

import os
import uuid
from pathlib import Path
from typing import Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import asyncpg
from asyncpg.pool import Pool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Default embedding dimension (matches Qwen2.5-VL hidden size)
DEFAULT_EMBEDDING_DIM = 1536
# OSNet person Re-ID feature dimension
REID_EMBEDDING_DIM = 512


@dataclass
class SessionRecord:
    """Database record for a session."""
    session_id: str
    created_at: float
    full_image_path: str
    image_width: int
    image_height: int


@dataclass
class EntityRecord:
    """Database record for an entity."""
    object_id: str
    session_id: str
    class_name: str
    confidence: float
    box_x1: int
    box_y1: int
    box_x2: int
    box_y2: int
    crop_image_path: str
    initial_analysis: Optional[str]
    stage: str
    created_at: float
    has_embedding: bool = False  # Whether embedding exists
    person_id: Optional[str] = None
    person_label: Optional[str] = None
    person_is_watchlist: bool = False
    match_score: Optional[float] = None
    match_status: Optional[str] = None


@dataclass
class PersonRecord:
    """Persistent person / suspect cluster."""
    person_id: str
    label: Optional[str]
    is_watchlist: bool
    notes: Optional[str]
    sighting_count: int
    representative_entity_id: Optional[str]
    representative_crop_path: Optional[str]
    created_at: float
    updated_at: float


@dataclass
class SearchResult:
    """
    Result from hybrid search with scoring breakdown.
    
    Attributes:
        entity: The matched EntityRecord
        text_score: Score from full-text search (0-1, higher = better match)
        vector_score: Cosine similarity score (0-1, higher = more similar)
        hybrid_score: Combined weighted score
    """
    entity: EntityRecord
    text_score: float = 0.0
    vector_score: float = 0.0
    hybrid_score: float = 0.0


class Database:
    """Async PostgreSQL database manager."""
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:postgres@localhost:5432/vision_analysis"
        )
        self.pool: Optional[Pool] = None
    
    async def connect(self):
        """Create connection pool."""
        if self.pool is None:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            await self._init_schema()
    
    async def close(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None

    def _entity_from_row(self, row) -> EntityRecord:
        """Build EntityRecord from entities row (optionally joined with persons)."""
        ms = row.get("match_score")
        return EntityRecord(
            object_id=row["id"],
            session_id=row["session_id"],
            class_name=row["class_name"],
            confidence=row["confidence"],
            box_x1=row["box_x1"],
            box_y1=row["box_y1"],
            box_x2=row["box_x2"],
            box_y2=row["box_y2"],
            crop_image_path=row["crop_image_path"],
            initial_analysis=row["initial_analysis"],
            stage=row["stage"],
            created_at=row["created_at"].timestamp(),
            has_embedding=row.get("embedding_json") is not None,
            person_id=row.get("person_id"),
            person_label=row.get("person_label"),
            person_is_watchlist=bool(row.get("person_is_watchlist", False)),
            match_score=float(ms) if ms is not None else None,
            match_status=row.get("match_status"),
        )
    
    async def _init_schema(self):
        """
        Initialize database schema with pgvector support.
        
        Creates tables for sessions and entities, plus indexes for:
        - Regular lookups (session_id, class_name, created_at)
        - Full-text search (GIN index on analysis text)
        - Vector similarity search (HNSW index on embeddings)
        """
        async with self.pool.acquire() as conn:
            # Enable pgvector extension (requires superuser or extension already installed)
            try:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                print("   pgvector extension enabled")
            except Exception as e:
                print(f"   Warning: Could not enable pgvector: {e}")
                print("   Run: CREATE EXTENSION vector; as superuser")
            
            # Sessions table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    full_image_path TEXT NOT NULL,
                    image_width INTEGER NOT NULL,
                    image_height INTEGER NOT NULL
                )
            """)
            
            # Entities table with embedding column
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    class_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    box_x1 INTEGER NOT NULL,
                    box_y1 INTEGER NOT NULL,
                    box_x2 INTEGER NOT NULL,
                    box_y2 INTEGER NOT NULL,
                    crop_image_path TEXT NOT NULL,
                    initial_analysis TEXT,
                    stage TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    CONSTRAINT fk_session FOREIGN KEY (session_id) 
                        REFERENCES sessions(id) ON DELETE CASCADE
                )
            """)
            
            # Add embedding column if it doesn't exist
            # Using TEXT to store embedding as JSON array (portable, no pgvector required)
            # If pgvector is available, we also create a vector column
            try:
                await conn.execute("""
                    ALTER TABLE entities 
                    ADD COLUMN IF NOT EXISTS embedding_json TEXT
                """)
            except Exception:
                pass  # Column may already exist
            
            # Try to add vector column for pgvector (optional, for fast similarity search)
            try:
                await conn.execute(f"""
                    ALTER TABLE entities 
                    ADD COLUMN IF NOT EXISTS embedding vector({DEFAULT_EMBEDDING_DIM})
                """)
                self._has_pgvector = True
                print("   Vector column created (pgvector)")
            except Exception as e:
                self._has_pgvector = False
                print(f"   Using JSON embeddings (pgvector not available: {e})")
            
            # Indexes for performance
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_entities_session 
                ON entities(session_id)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_entities_class 
                ON entities(class_name)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_created 
                ON sessions(created_at DESC)
            """)
            
            # Full-text search index (GIN for fast text matching)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_entities_analysis_fts 
                ON entities USING gin(to_tsvector('english', coalesce(initial_analysis, '')))
            """)
            
            # Vector similarity index (HNSW for fast approximate nearest neighbor)
            if self._has_pgvector:
                try:
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_entities_embedding_hnsw
                        ON entities USING hnsw (embedding vector_cosine_ops)
                        WITH (m = 16, ef_construction = 64)
                    """)
                    print("   HNSW vector index created")
                except Exception as e:
                    print(f"   Warning: Could not create HNSW index: {e}")
            
            # Persistent person identities (Re-ID clusters / suspects)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS persons (
                    id TEXT PRIMARY KEY,
                    label TEXT,
                    is_watchlist BOOLEAN NOT NULL DEFAULT FALSE,
                    notes TEXT,
                    sighting_count INTEGER NOT NULL DEFAULT 0,
                    representative_entity_id TEXT,
                    centroid_json TEXT,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
                )
            """)
            
            for col_sql in (
                "ALTER TABLE entities ADD COLUMN IF NOT EXISTS person_id TEXT",
                "ALTER TABLE entities ADD COLUMN IF NOT EXISTS match_score REAL",
                "ALTER TABLE entities ADD COLUMN IF NOT EXISTS match_status TEXT",
                "ALTER TABLE entities ADD COLUMN IF NOT EXISTS reid_embedding_json TEXT",
            ):
                try:
                    await conn.execute(col_sql)
                except Exception:
                    pass
            
            if self._has_pgvector:
                try:
                    await conn.execute(f"""
                        ALTER TABLE persons 
                        ADD COLUMN IF NOT EXISTS centroid vector({REID_EMBEDDING_DIM})
                    """)
                except Exception as e:
                    print(f"   Warning: persons.centroid vector: {e}")
                try:
                    await conn.execute(f"""
                        ALTER TABLE entities 
                        ADD COLUMN IF NOT EXISTS reid_embedding vector({REID_EMBEDDING_DIM})
                    """)
                except Exception as e:
                    print(f"   Warning: entities.reid_embedding vector: {e}")
            
            try:
                await conn.execute("""
                    ALTER TABLE entities
                    ADD CONSTRAINT fk_entities_person 
                    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE SET NULL
                """)
            except Exception:
                pass
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_entities_person 
                ON entities(person_id)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_persons_watchlist 
                ON persons(is_watchlist) WHERE is_watchlist = TRUE
            """)
            
            if self._has_pgvector:
                try:
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_persons_centroid_hnsw
                        ON persons USING hnsw (centroid vector_cosine_ops)
                        WITH (m = 16, ef_construction = 64)
                    """)
                    print("   HNSW index on persons.centroid created")
                except Exception as e:
                    print(f"   Warning: Could not create persons HNSW index: {e}")
    
    async def create_session(
        self,
        session_id: str,
        full_image_path: str,
        image_width: int,
        image_height: int,
        created_at: Optional[datetime] = None
    ):
        """Create a new session record."""
        if created_at is None:
            created_at = datetime.now()
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO sessions (id, created_at, full_image_path, image_width, image_height)
                VALUES ($1, $2, $3, $4, $5)
            """, session_id, created_at, str(full_image_path), image_width, image_height)
    
    async def create_entity(
        self,
        object_id: str,
        session_id: str,
        class_name: str,
        confidence: float,
        box: tuple,
        crop_image_path: str,
        initial_analysis: Optional[str],
        stage: str,
        created_at: Optional[datetime] = None
    ):
        """Create a new entity record."""
        if created_at is None:
            created_at = datetime.now()
        
        x1, y1, x2, y2 = box
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO entities (
                    id, session_id, class_name, confidence,
                    box_x1, box_y1, box_x2, box_y2,
                    crop_image_path, initial_analysis, stage, created_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """, object_id, session_id, class_name, confidence,
                x1, y1, x2, y2, str(crop_image_path), initial_analysis, stage, created_at)
    
    async def get_session(self, session_id: str) -> Optional[SessionRecord]:
        """Get a session by ID."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM sessions WHERE id = $1", session_id
            )
            
            if row is None:
                return None
            
            return SessionRecord(
                session_id=row["id"],
                created_at=row["created_at"].timestamp(),
                full_image_path=row["full_image_path"],
                image_width=row["image_width"],
                image_height=row["image_height"]
            )
    
    async def get_entity(self, object_id: str) -> Optional[EntityRecord]:
        """Get an entity by ID (includes person label when linked)."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT e.*, p.label AS person_label,
                       COALESCE(p.is_watchlist, FALSE) AS person_is_watchlist
                FROM entities e
                LEFT JOIN persons p ON p.id = e.person_id
                WHERE e.id = $1
                """,
                object_id,
            )
            
            if row is None:
                return None
            
            return self._entity_from_row(row)
    
    async def get_session_entities(self, session_id: str) -> List[EntityRecord]:
        """Get all entities for a session."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT e.*, p.label AS person_label,
                       COALESCE(p.is_watchlist, FALSE) AS person_is_watchlist
                FROM entities e
                LEFT JOIN persons p ON p.id = e.person_id
                WHERE e.session_id = $1
                ORDER BY e.created_at
                """,
                session_id,
            )
            
            return [self._entity_from_row(row) for row in rows]
    
    async def list_sessions(self, limit: int = 100, offset: int = 0) -> List[SessionRecord]:
        """List recent sessions."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM sessions 
                ORDER BY created_at DESC 
                LIMIT $1 OFFSET $2
            """, limit, offset)
            
            return [SessionRecord(
                session_id=row["id"],
                created_at=row["created_at"].timestamp(),
                full_image_path=row["full_image_path"],
                image_width=row["image_width"],
                image_height=row["image_height"]
            ) for row in rows]
    
    async def search_entities_by_class(
        self, 
        class_name: str, 
        limit: int = 100
    ) -> List[EntityRecord]:
        """Search entities by class name."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM entities 
                WHERE class_name = $1 
                ORDER BY created_at DESC 
                LIMIT $2
            """, class_name, limit)
            
            return [self._entity_from_row(row) for row in rows]
    
    async def search_entities_by_text(
        self, 
        search_text: str, 
        limit: int = 100
    ) -> List[EntityRecord]:
        """Full-text search on analysis text."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM entities 
                WHERE to_tsvector('english', coalesce(initial_analysis, '')) 
                      @@ plainto_tsquery('english', $1)
                ORDER BY created_at DESC 
                LIMIT $2
            """, search_text, limit)
            
            return [self._entity_from_row(row) for row in rows]
    
    # ══════════════════════════════════════════════════════════════════════════════
    # EMBEDDING OPERATIONS
    # ══════════════════════════════════════════════════════════════════════════════
    
    def _embedding_to_str(self, embedding: np.ndarray) -> str:
        """Convert numpy embedding to PostgreSQL vector string format."""
        return '[' + ','.join(map(str, embedding.tolist())) + ']'
    
    def _embedding_to_json(self, embedding: np.ndarray) -> str:
        """Convert numpy embedding to JSON string for storage."""
        import json
        return json.dumps(embedding.tolist())
    
    def _json_to_embedding(self, json_str: str) -> np.ndarray:
        """Convert JSON string back to numpy array."""
        import json
        return np.array(json.loads(json_str), dtype=np.float32)
    
    async def update_entity_embedding(
        self,
        object_id: str,
        embedding: np.ndarray
    ) -> bool:
        """
        Store embedding for an entity.
        
        Args:
            object_id: Entity ID
            embedding: Numpy array of embedding values
            
        Returns:
            True if successful
        """
        async with self.pool.acquire() as conn:
            # Always store as JSON (portable)
            embedding_json = self._embedding_to_json(embedding)
            
            # If pgvector is available, also store as vector
            if getattr(self, '_has_pgvector', False):
                embedding_str = self._embedding_to_str(embedding)
                await conn.execute("""
                    UPDATE entities 
                    SET embedding_json = $1, embedding = $2::vector
                    WHERE id = $3
                """, embedding_json, embedding_str, object_id)
            else:
                await conn.execute("""
                    UPDATE entities 
                    SET embedding_json = $1
                    WHERE id = $2
                """, embedding_json, object_id)
            
            return True
    
    async def get_entity_embedding(self, object_id: str) -> Optional[np.ndarray]:
        """Get embedding for an entity."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT embedding_json FROM entities WHERE id = $1",
                object_id
            )
            if row and row["embedding_json"]:
                return self._json_to_embedding(row["embedding_json"])
            return None

    async def update_entity_reid_embedding(self, object_id: str, reid_embedding: np.ndarray) -> bool:
        """Store OSNet Re-ID embedding (512-D) for an entity."""
        emb = np.asarray(reid_embedding, dtype=np.float32).reshape(-1)
        embedding_json = self._embedding_to_json(emb)
        async with self.pool.acquire() as conn:
            if getattr(self, "_has_pgvector", False):
                embedding_str = self._embedding_to_str(emb)
                await conn.execute(
                    """
                    UPDATE entities
                    SET reid_embedding_json = $1, reid_embedding = $2::vector
                    WHERE id = $3
                    """,
                    embedding_json,
                    embedding_str,
                    object_id,
                )
            else:
                await conn.execute(
                    """
                    UPDATE entities SET reid_embedding_json = $1 WHERE id = $2
                    """,
                    embedding_json,
                    object_id,
                )
        return True

    async def _recompute_person_centroid(self, conn, person_id: str) -> None:
        rows = await conn.fetch(
            """
            SELECT reid_embedding_json FROM entities
            WHERE person_id = $1 AND reid_embedding_json IS NOT NULL
            """,
            person_id,
        )
        if not rows:
            await conn.execute(
                """
                UPDATE persons
                SET sighting_count = 0, centroid_json = NULL, centroid = NULL, updated_at = NOW()
                WHERE id = $1
                """,
                person_id,
            )
            return
        vecs = np.stack([self._json_to_embedding(r["reid_embedding_json"]) for r in rows], axis=0)
        mean = np.mean(vecs, axis=0).astype(np.float32)
        nrm = float(np.linalg.norm(mean)) + 1e-8
        mean = mean / nrm
        mean_json = self._embedding_to_json(mean)
        n = len(rows)
        if getattr(self, "_has_pgvector", False):
            mean_str = self._embedding_to_str(mean)
            await conn.execute(
                """
                UPDATE persons
                SET sighting_count = $2, centroid_json = $3, centroid = $4::vector, updated_at = NOW()
                WHERE id = $1
                """,
                person_id,
                n,
                mean_json,
                mean_str,
            )
        else:
            await conn.execute(
                """
                UPDATE persons
                SET sighting_count = $2, centroid_json = $3, updated_at = NOW()
                WHERE id = $1
                """,
                person_id,
                n,
                mean_json,
            )

    async def match_person(
        self,
        reid_embedding: np.ndarray,
        top_k: int = 8,
    ) -> List[Tuple[str, float]]:
        """Return (person_id, cosine_similarity) best-first."""
        q = np.asarray(reid_embedding, dtype=np.float32).reshape(-1)
        q = q / (float(np.linalg.norm(q)) + 1e-8)
        async with self.pool.acquire() as conn:
            if getattr(self, "_has_pgvector", False):
                qstr = self._embedding_to_str(q)
                rows = await conn.fetch(
                    """
                    SELECT id,
                           1 - (centroid <=> $1::vector) AS similarity
                    FROM persons
                    WHERE centroid IS NOT NULL
                    ORDER BY centroid <=> $1::vector
                    LIMIT $2
                    """,
                    qstr,
                    top_k,
                )
                return [(row["id"], float(row["similarity"])) for row in rows]
            rows = await conn.fetch(
                "SELECT id, centroid_json FROM persons WHERE centroid_json IS NOT NULL"
            )
            scored: List[Tuple[str, float]] = []
            for row in rows:
                c = self._json_to_embedding(row["centroid_json"])
                sim = float(np.dot(q, c))
                scored.append((row["id"], sim))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]

    async def create_person(
        self,
        label: Optional[str],
        is_watchlist: bool,
        initial_embedding: np.ndarray,
        representative_entity_id: str,
        notes: Optional[str] = None,
    ) -> str:
        """Insert a new person cluster; sighting_count updated when entities link."""
        person_id = str(uuid.uuid4())[:12]
        emb = np.asarray(initial_embedding, dtype=np.float32).reshape(-1)
        emb = emb / (float(np.linalg.norm(emb)) + 1e-8)
        emb_json = self._embedding_to_json(emb)
        async with self.pool.acquire() as conn:
            if getattr(self, "_has_pgvector", False):
                emb_str = self._embedding_to_str(emb)
                await conn.execute(
                    """
                    INSERT INTO persons (
                        id, label, is_watchlist, notes, sighting_count,
                        representative_entity_id, centroid_json, centroid, created_at, updated_at
                    )
                    VALUES ($1, $2, $3, $4, 0, $5, $6, $7::vector, NOW(), NOW())
                    """,
                    person_id,
                    label,
                    is_watchlist,
                    notes,
                    representative_entity_id,
                    emb_json,
                    emb_str,
                )
            else:
                await conn.execute(
                    """
                    INSERT INTO persons (
                        id, label, is_watchlist, notes, sighting_count,
                        representative_entity_id, centroid_json, created_at, updated_at
                    )
                    VALUES ($1, $2, $3, $4, 0, $5, $6, NOW(), NOW())
                    """,
                    person_id,
                    label,
                    is_watchlist,
                    notes,
                    representative_entity_id,
                    emb_json,
                )
        return person_id

    async def link_entity_to_person(
        self,
        object_id: str,
        person_id: str,
        match_score: Optional[float],
        match_status: str,
    ) -> None:
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    UPDATE entities
                    SET person_id = $1, match_score = $2, match_status = $3
                    WHERE id = $4
                    """,
                    person_id,
                    match_score,
                    match_status,
                    object_id,
                )
                await self._recompute_person_centroid(conn, person_id)

    async def assign_detection_to_person(
        self,
        object_id: str,
        reid_embedding: np.ndarray,
        match_threshold: float,
        review_threshold: float,
    ) -> dict[str, Any]:
        """
        Auto-assign entity to person cluster using Re-ID thresholds.

        Returns dict with keys: person_id, person_label, is_watchlist, match_score, match_status
        """
        from person_matcher import decide_match

        matches = await self.match_person(reid_embedding, top_k=8)
        decision = decide_match(matches, match_threshold, review_threshold)

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                if decision.status == "matched" and decision.best_person_id:
                    await conn.execute(
                        """
                        UPDATE entities
                        SET person_id = $1, match_score = $2, match_status = $3
                        WHERE id = $4
                        """,
                        decision.best_person_id,
                        decision.best_score,
                        "matched",
                        object_id,
                    )
                    await self._recompute_person_centroid(conn, decision.best_person_id)
                    prow = await conn.fetchrow(
                        "SELECT label, is_watchlist FROM persons WHERE id = $1",
                        decision.best_person_id,
                    )
                    return {
                        "person_id": decision.best_person_id,
                        "person_label": prow["label"] if prow else None,
                        "is_watchlist": bool(prow["is_watchlist"]) if prow else False,
                        "match_score": float(decision.best_score) if decision.best_score is not None else None,
                        "match_status": "matched",
                    }

                if decision.status == "new":
                    pid = str(uuid.uuid4())[:12]
                    emb = np.asarray(reid_embedding, dtype=np.float32).reshape(-1)
                    emb = emb / (float(np.linalg.norm(emb)) + 1e-8)
                    emb_json = self._embedding_to_json(emb)
                    if getattr(self, "_has_pgvector", False):
                        emb_str = self._embedding_to_str(emb)
                        await conn.execute(
                            """
                            INSERT INTO persons (
                                id, label, is_watchlist, notes, sighting_count,
                                representative_entity_id, centroid_json, centroid, created_at, updated_at
                            )
                            VALUES ($1, NULL, FALSE, NULL, 0, $2, $3, $4::vector, NOW(), NOW())
                            """,
                            pid,
                            object_id,
                            emb_json,
                            emb_str,
                        )
                    else:
                        await conn.execute(
                            """
                            INSERT INTO persons (
                                id, label, is_watchlist, notes, sighting_count,
                                representative_entity_id, centroid_json, created_at, updated_at
                            )
                            VALUES ($1, NULL, FALSE, NULL, 0, $2, $3, NOW(), NOW())
                            """,
                            pid,
                            object_id,
                            emb_json,
                        )
                    await conn.execute(
                        """
                        UPDATE entities
                        SET person_id = $1, match_score = NULL, match_status = 'new'
                        WHERE id = $2
                        """,
                        pid,
                        object_id,
                    )
                    await self._recompute_person_centroid(conn, pid)
                    return {
                        "person_id": pid,
                        "person_label": None,
                        "is_watchlist": False,
                        "match_score": None,
                        "match_status": "new",
                    }

                await conn.execute(
                    """
                    UPDATE entities
                    SET person_id = NULL, match_score = $1, match_status = 'pending'
                    WHERE id = $2
                    """,
                    float(decision.best_score) if decision.best_score is not None else None,
                    object_id,
                )
                return {
                    "person_id": None,
                    "person_label": None,
                    "is_watchlist": False,
                    "match_score": float(decision.best_score) if decision.best_score is not None else None,
                    "match_status": "pending",
                }

    def _person_from_row(self, row) -> PersonRecord:
        rep_path = None
        if row.get("representative_crop_path"):
            rep_path = row["representative_crop_path"]
        elif row.get("representative_entity_id"):
            rep_path = None
        return PersonRecord(
            person_id=row["id"],
            label=row.get("label"),
            is_watchlist=bool(row.get("is_watchlist", False)),
            notes=row.get("notes"),
            sighting_count=int(row.get("sighting_count", 0)),
            representative_entity_id=row.get("representative_entity_id"),
            representative_crop_path=rep_path,
            created_at=row["created_at"].timestamp(),
            updated_at=row["updated_at"].timestamp(),
        )

    async def list_persons(
        self,
        watchlist_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> List[PersonRecord]:
        async with self.pool.acquire() as conn:
            if watchlist_only:
                rows = await conn.fetch(
                    """
                    SELECT p.*, e.crop_image_path AS representative_crop_path
                    FROM persons p
                    LEFT JOIN entities e ON e.id = p.representative_entity_id
                    WHERE p.is_watchlist = TRUE
                    ORDER BY p.updated_at DESC
                    LIMIT $1 OFFSET $2
                    """,
                    limit,
                    offset,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT p.*, e.crop_image_path AS representative_crop_path
                    FROM persons p
                    LEFT JOIN entities e ON e.id = p.representative_entity_id
                    ORDER BY p.updated_at DESC
                    LIMIT $1 OFFSET $2
                    """,
                    limit,
                    offset,
                )
            return [self._person_from_row(r) for r in rows]

    async def get_person(self, person_id: str) -> Optional[PersonRecord]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT p.*, e.crop_image_path AS representative_crop_path
                FROM persons p
                LEFT JOIN entities e ON e.id = p.representative_entity_id
                WHERE p.id = $1
                """,
                person_id,
            )
            if row is None:
                return None
            return self._person_from_row(row)

    async def get_person_sightings(self, person_id: str) -> List[EntityRecord]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT e.*, p.label AS person_label,
                       COALESCE(p.is_watchlist, FALSE) AS person_is_watchlist
                FROM entities e
                LEFT JOIN persons p ON p.id = e.person_id
                WHERE e.person_id = $1
                ORDER BY e.created_at DESC
                """,
                person_id,
            )
            return [self._entity_from_row(r) for r in rows]

    async def update_person(
        self,
        person_id: str,
        label: Optional[str] = None,
        is_watchlist: Optional[bool] = None,
        notes: Optional[str] = None,
    ) -> bool:
        if label is None and is_watchlist is None and notes is None:
            return False
        async with self.pool.acquire() as conn:
            cur = await conn.fetchrow(
                "SELECT label, is_watchlist, notes FROM persons WHERE id = $1",
                person_id,
            )
            if cur is None:
                return False
            nl = label if label is not None else cur["label"]
            nw = is_watchlist if is_watchlist is not None else cur["is_watchlist"]
            nn = notes if notes is not None else cur["notes"]
            row = await conn.fetchrow(
                """
                UPDATE persons
                SET label = $1, is_watchlist = $2, notes = $3, updated_at = NOW()
                WHERE id = $4
                RETURNING id
                """,
                nl,
                nw,
                nn,
                person_id,
            )
            return row is not None

    async def merge_persons(self, keep_id: str, merge_id: str) -> bool:
        if keep_id == merge_id:
            return False
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                k = await conn.fetchrow("SELECT id FROM persons WHERE id = $1", keep_id)
                m = await conn.fetchrow("SELECT id FROM persons WHERE id = $1", merge_id)
                if not k or not m:
                    return False
                await conn.execute(
                    "UPDATE entities SET person_id = $1 WHERE person_id = $2",
                    keep_id,
                    merge_id,
                )
                await conn.execute("DELETE FROM persons WHERE id = $1", merge_id)
                await self._recompute_person_centroid(conn, keep_id)
                rep = await conn.fetchrow(
                    "SELECT id FROM entities WHERE person_id = $1 ORDER BY confidence DESC LIMIT 1",
                    keep_id,
                )
                if rep:
                    await conn.execute(
                        "UPDATE persons SET representative_entity_id = $2 WHERE id = $1",
                        keep_id,
                        rep["id"],
                    )
        return True

    async def reassign_entity(self, object_id: str, new_person_id: str) -> bool:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT person_id FROM entities WHERE id = $1", object_id
            )
            if row is None:
                return False
            newp = await conn.fetchrow("SELECT id FROM persons WHERE id = $1", new_person_id)
            if not newp:
                return False
            old_pid = row["person_id"]
            async with conn.transaction():
                await conn.execute(
                    """
                    UPDATE entities
                    SET person_id = $1, match_status = 'matched', match_score = NULL
                    WHERE id = $2
                    """,
                    new_person_id,
                    object_id,
                )
                if old_pid and old_pid != new_person_id:
                    n_old = await conn.fetchval(
                        "SELECT COUNT(*) FROM entities WHERE person_id = $1", old_pid
                    )
                    if n_old == 0:
                        await conn.execute("DELETE FROM persons WHERE id = $1", old_pid)
                    else:
                        await self._recompute_person_centroid(conn, old_pid)
                        r0 = await conn.fetchrow(
                            "SELECT id FROM entities WHERE person_id = $1 ORDER BY confidence DESC LIMIT 1",
                            old_pid,
                        )
                        if r0:
                            await conn.execute(
                                "UPDATE persons SET representative_entity_id = $2 WHERE id = $1",
                                old_pid,
                                r0["id"],
                            )
                await self._recompute_person_centroid(conn, new_person_id)
                r1 = await conn.fetchrow(
                    "SELECT id FROM entities WHERE person_id = $1 ORDER BY confidence DESC LIMIT 1",
                    new_person_id,
                )
                if r1:
                    await conn.execute(
                        "UPDATE persons SET representative_entity_id = $2 WHERE id = $1",
                        new_person_id,
                        r1["id"],
                    )
        return True

    async def search_persons_by_embedding(
        self,
        reid_embedding: np.ndarray,
        limit: int = 20,
        min_similarity: float = 0.0,
    ) -> List[Tuple[PersonRecord, float]]:
        matches = await self.match_person(reid_embedding, top_k=limit)
        out: List[Tuple[PersonRecord, float]] = []
        for pid, sim in matches:
            if sim < min_similarity:
                continue
            p = await self.get_person(pid)
            if p:
                out.append((p, sim))
        return out

    async def delete_person(self, person_id: str) -> bool:
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "UPDATE entities SET person_id = NULL, match_status = NULL, match_score = NULL WHERE person_id = $1",
                    person_id,
                )
                r = await conn.execute("DELETE FROM persons WHERE id = $1", person_id)
                return "DELETE 1" in r
    
    # ══════════════════════════════════════════════════════════════════════════════
    # HYBRID SEARCH
    # ══════════════════════════════════════════════════════════════════════════════
    
    async def search_by_vector(
        self,
        query_embedding: np.ndarray,
        limit: int = 20,
        min_similarity: float = 0.0
    ) -> List[Tuple[EntityRecord, float]]:
        """
        Vector similarity search using cosine distance.
        
        Args:
            query_embedding: Query embedding vector
            limit: Max results
            min_similarity: Minimum cosine similarity (0-1)
            
        Returns:
            List of (EntityRecord, similarity_score) tuples
        """
        async with self.pool.acquire() as conn:
            if getattr(self, '_has_pgvector', False):
                # Use pgvector's fast similarity search
                embedding_str = self._embedding_to_str(query_embedding)
                rows = await conn.fetch("""
                    SELECT *,
                           1 - (embedding <=> $1::vector) as similarity
                    FROM entities
                    WHERE embedding IS NOT NULL
                      AND (1 - (embedding <=> $1::vector)) >= $2
                    ORDER BY embedding <=> $1::vector
                    LIMIT $3
                """, embedding_str, min_similarity, limit)
            else:
                # Fallback: load all embeddings and compute similarity in Python
                rows = await conn.fetch("""
                    SELECT * FROM entities 
                    WHERE embedding_json IS NOT NULL
                """)
                
                # Compute similarities
                results = []
                for row in rows:
                    emb = self._json_to_embedding(row["embedding_json"])
                    similarity = float(np.dot(query_embedding, emb))  # Assuming normalized
                    if similarity >= min_similarity:
                        results.append((row, similarity))
                
                # Sort and limit
                results.sort(key=lambda x: x[1], reverse=True)
                rows = [(r[0], r[1]) for r in results[:limit]]
                
                # Return early for fallback
                return [
                    (self._entity_from_row(row), sim)
                    for row, sim in rows
                ]
            
            return [
                (self._entity_from_row(row), row["similarity"])
                for row in rows
            ]
    
    async def search_text_with_score(
        self,
        search_text: str,
        limit: int = 100
    ) -> List[Tuple[EntityRecord, float]]:
        """
        Full-text search with relevance scoring.
        
        Args:
            search_text: Search query
            limit: Max results
            
        Returns:
            List of (EntityRecord, text_score) tuples
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT *,
                       ts_rank(
                           to_tsvector('english', coalesce(initial_analysis, '')),
                           plainto_tsquery('english', $1)
                       ) as text_rank
                FROM entities 
                WHERE to_tsvector('english', coalesce(initial_analysis, '')) 
                      @@ plainto_tsquery('english', $1)
                ORDER BY text_rank DESC
                LIMIT $2
            """, search_text, limit)
            
            # Normalize text scores to 0-1 range
            max_score = max((row["text_rank"] for row in rows), default=1.0) or 1.0
            
            return [
                (self._entity_from_row(row), row["text_rank"] / max_score)
                for row in rows
            ]
    
    async def hybrid_search(
        self,
        text_query: Optional[str] = None,
        image_embedding: Optional[np.ndarray] = None,
        text_weight: float = 0.5,
        vector_weight: float = 0.5,
        limit: int = 20,
        min_score: float = 0.1
    ) -> List[SearchResult]:
        """
        Hybrid search combining text and vector similarity.
        
        The search works in three modes:
        1. Text only: text_query provided, no embedding
        2. Image only: embedding provided, no text
        3. Hybrid: both provided, scores are combined with weights
        
        Scoring formula:
            hybrid_score = (text_weight * text_score) + (vector_weight * vector_score)
        
        Args:
            text_query: Text search query (optional)
            image_embedding: Image embedding vector (optional)
            text_weight: Weight for text score (0-1)
            vector_weight: Weight for vector score (0-1)
            limit: Maximum results to return
            min_score: Minimum hybrid score threshold
            
        Returns:
            List of SearchResult objects sorted by hybrid_score descending
        """
        if not text_query and image_embedding is None:
            return []
        
        # Collect results from both search types
        text_results = {}  # object_id -> (entity, text_score)
        vector_results = {}  # object_id -> (entity, vector_score)
        
        # Text search
        if text_query:
            text_matches = await self.search_text_with_score(text_query, limit * 2)
            for entity, score in text_matches:
                text_results[entity.object_id] = (entity, score)
        
        # Vector search
        if image_embedding is not None:
            vector_matches = await self.search_by_vector(image_embedding, limit * 2)
            for entity, score in vector_matches:
                vector_results[entity.object_id] = (entity, score)
        
        # Combine results
        all_ids = set(text_results.keys()) | set(vector_results.keys())
        
        combined = []
        for obj_id in all_ids:
            # Get entity from either source
            if obj_id in text_results:
                entity, text_score = text_results[obj_id]
            else:
                entity, _ = vector_results[obj_id]
                text_score = 0.0
            
            if obj_id in vector_results:
                _, vector_score = vector_results[obj_id]
            else:
                vector_score = 0.0
            
            # Compute hybrid score
            # Normalize weights to sum to 1
            total_weight = text_weight + vector_weight
            if total_weight > 0:
                norm_text = text_weight / total_weight
                norm_vector = vector_weight / total_weight
            else:
                norm_text = norm_vector = 0.5
            
            # If only one source has results, use that score directly
            if text_query and image_embedding is not None:
                hybrid_score = (norm_text * text_score) + (norm_vector * vector_score)
            elif text_query:
                hybrid_score = text_score
            else:
                hybrid_score = vector_score
            
            if hybrid_score >= min_score:
                combined.append(SearchResult(
                    entity=entity,
                    text_score=text_score,
                    vector_score=vector_score,
                    hybrid_score=hybrid_score
                ))
        
        # Sort by hybrid score
        combined.sort(key=lambda x: x.hybrid_score, reverse=True)
        
        return combined[:limit]

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its entities (cascade)."""
        async with self.pool.acquire() as conn:
            # Get image paths before deleting (for cleanup)
            session = await self.get_session(session_id)
            entities = await self.get_session_entities(session_id)
            
            if not session:
                return False
            
            # Delete from database (entities cascade automatically)
            result = await conn.execute(
                "DELETE FROM sessions WHERE id = $1", session_id
            )
            
            return "DELETE 1" in result
    
    async def delete_entity(self, object_id: str) -> bool:
        """Delete a single entity; recomputes affected person cluster."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT person_id FROM entities WHERE id = $1", object_id
            )
            pid = row["person_id"] if row else None
            result = await conn.execute("DELETE FROM entities WHERE id = $1", object_id)
            ok = "DELETE 1" in result
            if ok and pid:
                n = await conn.fetchval(
                    "SELECT COUNT(*) FROM entities WHERE person_id = $1", pid
                )
                if n == 0:
                    await conn.execute("DELETE FROM persons WHERE id = $1", pid)
                else:
                    await self._recompute_person_centroid(conn, pid)
                    rep = await conn.fetchrow(
                        """
                        SELECT id FROM entities
                        WHERE person_id = $1
                        ORDER BY confidence DESC NULLS LAST
                        LIMIT 1
                        """,
                        pid,
                    )
                    if rep:
                        await conn.execute(
                            """
                            UPDATE persons SET representative_entity_id = $2, updated_at = NOW()
                            WHERE id = $1
                            """,
                            pid,
                            rep["id"],
                        )
            return ok
    
    async def list_sessions_with_entity_count(
        self, 
        limit: int = 50, 
        offset: int = 0
    ) -> List[dict]:
        """List sessions with entity counts for the history view."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT 
                    s.id,
                    s.created_at,
                    s.full_image_path,
                    s.image_width,
                    s.image_height,
                    COUNT(e.id) as entity_count
                FROM sessions s
                LEFT JOIN entities e ON e.session_id = s.id
                GROUP BY s.id
                ORDER BY s.created_at DESC
                LIMIT $1 OFFSET $2
            """, limit, offset)
            
            return [{
                "session_id": row["id"],
                "created_at": row["created_at"].timestamp(),
                "full_image_path": row["full_image_path"],
                "image_width": row["image_width"],
                "image_height": row["image_height"],
                "entity_count": row["entity_count"],
            } for row in rows]
    
    async def get_session_count(self) -> int:
        """Get total number of sessions."""
        async with self.pool.acquire() as conn:
            return await conn.fetchval("SELECT COUNT(*) FROM sessions")


# Global database instance
_db_instance: Optional[Database] = None


async def get_db() -> Database:
    """Get or create database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
        await _db_instance.connect()
    return _db_instance

