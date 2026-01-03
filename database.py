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
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import asyncpg
from asyncpg.pool import Pool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Default embedding dimension (matches Qwen2.5-VL hidden size)
DEFAULT_EMBEDDING_DIM = 1536


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
        """Get an entity by ID."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM entities WHERE id = $1", object_id
            )
            
            if row is None:
                return None
            
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
                created_at=row["created_at"].timestamp()
            )
    
    async def get_session_entities(self, session_id: str) -> List[EntityRecord]:
        """Get all entities for a session."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM entities WHERE session_id = $1 ORDER BY created_at",
                session_id
            )
            
            return [EntityRecord(
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
                created_at=row["created_at"].timestamp()
            ) for row in rows]
    
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
            
            return [EntityRecord(
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
                created_at=row["created_at"].timestamp()
            ) for row in rows]
    
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
            
            return [EntityRecord(
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
                created_at=row["created_at"].timestamp()
            ) for row in rows]
    
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
                    (EntityRecord(
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
                        has_embedding=True
                    ), sim)
                    for row, sim in rows
                ]
            
            return [
                (EntityRecord(
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
                    has_embedding=True
                ), row["similarity"])
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
                (EntityRecord(
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
                    has_embedding=row.get("embedding_json") is not None
                ), row["text_rank"] / max_score)
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
        """Delete a single entity."""
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM entities WHERE id = $1", object_id
            )
            return "DELETE 1" in result
    
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

