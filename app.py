#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI Backend for Hierarchical Vision Analysis Pipeline

Endpoints:
    GET  /                         - Serve frontend
    POST /analyze                  - Analyze image, return JSON results with object IDs
    POST /analyze/full-stream      - Analyze with SSE streaming progress
    POST /object/{id}/ask          - Ask follow-up question about specific entity
    GET  /health                   - Health check
    
    # History Management
    GET  /api/sessions             - List all sessions
    GET  /api/session/{id}         - Get session details
    DELETE /api/session/{id}       - Delete session
    
    # Search (NEW)
    POST /api/search               - Hybrid search (text + image)
    GET  /api/search/text          - Text-only search
    POST /api/search/image         - Image similarity search

Usage:
    uvicorn app:app --reload --port 8000
"""

import asyncio
import base64
import io
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from pipeline import (
    ANALYSIS_PROMPT,
    HierarchicalPipeline,
    PipelineConfig,
    create_pipeline,
    CropInfo,
    Detection,
)
from database import get_db, SearchResult
from storage_utils import save_session_image, save_crop_image, load_image
from embeddings import QwenEmbeddingExtractor, create_embedding_extractor


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EntityData:
    """Stored data for a detected entity (person)."""
    object_id: str
    class_name: str
    confidence: float
    box: tuple  # (x1, y1, x2, y2)
    crop_image: Image.Image
    initial_analysis: Optional[str]
    stage: str


@dataclass
class SessionData:
    """Session data containing full image and detected entities."""
    session_id: str
    full_image: Image.Image
    entities: Dict[str, EntityData] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class AskRequest(BaseModel):
    """Request body for follow-up questions."""
    question: str
    use_full_scene: bool = False  # If True, analyze full image instead of crop


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL STATE
# ══════════════════════════════════════════════════════════════════════════════

pipeline: Optional[HierarchicalPipeline] = None
embedding_extractor: Optional[QwenEmbeddingExtractor] = None


# ══════════════════════════════════════════════════════════════════════════════
# VLM PROMPTING
# ══════════════════════════════════════════════════════════════════════════════

def build_followup_prompt(question: str, initial_analysis: Optional[str] = None, use_full_scene: bool = False) -> str:
    """
    Build a grounded VLM prompt for follow-up questions about a person entity.
    
    The prompt explicitly:
    - States the entity type is a person
    - Grounds answers in visible evidence only
    - Instructs uncertainty when unsure
    - Adjusts context for full scene vs crop analysis
    """
    prior_context = f"Prior analysis: {initial_analysis}" if initial_analysis else "No prior analysis available."
    
    if use_full_scene:
        # Full scene prompt - broader context
        prompt = f"""You are analyzing a full scene image. A specific person was previously detected and analyzed.

Context:
- {prior_context}

User question about the scene or person:
{question}

Instructions:
- Consider the full scene context when answering.
- If the question is about a specific person, use the prior analysis as reference.
- Base answers strictly on visible evidence in the image.
- If uncertain, say so explicitly.
- Be concise and direct."""
    else:
        # Crop-only prompt - focused on the person
        prompt = f"""You are analyzing a specific person detected in an image.

Context:
- Entity type: person
- {prior_context}

User question:
{question}

Instructions:
- Focus only on the selected person and their immediate surroundings.
- Base answers strictly on visible evidence in the image.
- If uncertain or the answer is not clearly visible, say so explicitly.
- Be concise and direct."""
    
    return prompt


# ══════════════════════════════════════════════════════════════════════════════
# LIFESPAN
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize pipeline, embedding extractor, and database on startup."""
    global pipeline, embedding_extractor
    
    print("Starting Vision Analysis API...")
    
    # Initialize database
    db = await get_db()
    print("Database connected")
    
    # Initialize pipeline
    config = PipelineConfig(
        use_quantization=True,
        use_torch_compile=False,
        max_new_tokens=100,  # Allow longer responses for Q&A
        batch_size=4,
    )
    pipeline = HierarchicalPipeline(config)
    
    print("Loading models (this may take a minute)...")
    await pipeline.initialize()
    
    # Initialize embedding extractor (reuses VLM, no extra model load)
    print("Initializing embedding extractor...")
    embedding_extractor = create_embedding_extractor(pipeline)
    print(f"   Embedding dimension: {embedding_extractor.get_embedding_dim()}")
    
    print("API ready!")
    
    yield
    
    # Cleanup on shutdown
    if pipeline:
        pipeline.cleanup()
    
    # Close database
    db_instance = await get_db()
    await db_instance.close()
    print("API shutdown complete.")


# ══════════════════════════════════════════════════════════════════════════════
# APP SETUP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Entity-Centric Vision Analysis API",
    description="YOLO detection + VLM analysis with follow-up Q&A",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def image_to_base64(img: Image.Image, format: str = "JPEG") -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


async def load_upload_image(file: UploadFile) -> Image.Image:
    """Load and validate uploaded image."""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")




def generate_object_id() -> str:
    """Generate a unique object ID."""
    return str(uuid.uuid4())[:8]


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>API Running</h1><p>Frontend not found.</p>")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    db = await get_db()
    async with db.pool.acquire() as conn:
        session_count = await conn.fetchval("SELECT COUNT(*) FROM sessions")
    
    return {
        "status": "healthy",
        "pipeline_ready": pipeline is not None and pipeline.initialized,
        "total_sessions": session_count,
    }


@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
):
    """
    Analyze an uploaded image and persist to database.
    
    Returns results with stable object_ids for each detection.
    Stores session data for follow-up questions.
    """
    try:
        if not pipeline or not pipeline.initialized:
            raise HTTPException(status_code=503, detail="Pipeline not ready")
        
        image = await load_upload_image(file)
        analysis_prompt = prompt if prompt else ANALYSIS_PROMPT
        
        start_time = time.perf_counter()
        try:
            results = pipeline.analyze(image, prompt=analysis_prompt)
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Pipeline analysis failed: {error_details}")
            raise HTTPException(
                status_code=500,
                detail=f"Analysis failed: {str(e)}"
            )
        elapsed = time.perf_counter() - start_time
        
        # Get database instance
        db = await get_db()
        
        # Create session ID
        session_id = str(uuid.uuid4())[:12]
        
        # Save full image to disk
        try:
            full_image_path = save_session_image(image, session_id)
        except Exception as e:
            print(f"Failed to save session image: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save session image: {str(e)}"
            )
        
        # Save session metadata to database
        try:
            await db.create_session(
                session_id=session_id,
                full_image_path=str(full_image_path),
                image_width=image.width,
                image_height=image.height,
            )
        except Exception as e:
            print(f"Failed to create session in database: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create session: {str(e)}"
            )
        
        # Build response
        response_data = {
            "success": True,
            "session_id": session_id,
            "elapsed_seconds": round(elapsed, 2),
            "num_detections": len(results),
            "image_width": image.width,
            "image_height": image.height,
            "results": [],
        }
        
        # Process each detection result
        for result in results:
            object_id = generate_object_id()
            
            # Save crop image to disk
            try:
                crop_image_path = save_crop_image(result.crop_info.crop_image, object_id)
            except Exception as e:
                print(f"Failed to save crop image for {object_id}: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to save crop image: {str(e)}"
                )
            
            # Save entity metadata to database
            try:
                await db.create_entity(
                    object_id=object_id,
                    session_id=session_id,
                    class_name=result.crop_info.detection.class_name,
                    confidence=result.crop_info.detection.confidence,
                    box=result.crop_info.detection.box,
                    crop_image_path=str(crop_image_path),
                    initial_analysis=result.analysis_text,
                    stage=result.stage.value,
                )
            except Exception as e:
                print(f"Failed to create entity in database: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to save entity to database: {str(e)}"
                )
            
            # Store embedding from pipeline result (no separate extraction needed)
            if result.embedding is not None:
                try:
                    await db.update_entity_embedding(object_id, result.embedding)
                except Exception as e:
                    print(f"Warning: Failed to store embedding for {object_id}: {e}")
            
            # Use URL instead of base64 encoding (much faster)
            crop_image_url = f"/api/image/crop/{object_id}"
            
            response_data["results"].append({
                "object_id": object_id,
                "index": result.crop_info.detection.index,
                "class": result.crop_info.detection.class_name,
                "confidence": round(result.crop_info.detection.confidence, 3),
                "box": result.crop_info.detection.box,  # (x1, y1, x2, y2)
                "stage": result.stage.value,
                "analysis": result.analysis_text,
                "reason": result.reason,
                "crop_image": crop_image_url,
            })
        
        return response_data
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Catch any other unexpected exceptions
        import traceback
        error_details = traceback.format_exc()
        print(f"Unexpected error in analyze endpoint: {error_details}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.post("/object/{object_id}/ask")
async def ask_about_object(
    object_id: str,
    request: AskRequest,
):
    """
    Ask a follow-up question about a specific detected entity.
    
    The VLM analyzes the person's crop with grounded prompting.
    """
    if not pipeline or not pipeline.initialized:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    
    # Get database instance
    db = await get_db()
    
    # Load entity from database
    entity = await db.get_entity(object_id)
    if not entity:
        raise HTTPException(
            status_code=404,
            detail=f"Object {object_id} not found."
        )
    
    # Load session if needed
    session = None
    if request.use_full_scene:
        session = await db.get_session(entity.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
    
    # Build prompt
    prompt = build_followup_prompt(
        question=request.question,
        initial_analysis=entity.initial_analysis,
        use_full_scene=request.use_full_scene,
    )
    
    # Choose image: full scene or crop
    if request.use_full_scene and session:
        image_to_analyze = load_image(Path(session.full_image_path))
        # Resize if needed
        max_size = 1024
        if max(image_to_analyze.size) > max_size:
            ratio = max_size / max(image_to_analyze.size)
            new_size = (int(image_to_analyze.width * ratio), int(image_to_analyze.height * ratio))
            image_to_analyze = image_to_analyze.resize(new_size, Image.LANCZOS)
    else:
        image_to_analyze = load_image(Path(entity.crop_image_path))
    
    # Run VLM inference
    start_time = time.perf_counter()
    
    try:
        crop_info = CropInfo(
            detection=Detection(
                index=0,
                box=(entity.box_x1, entity.box_y1, entity.box_x2, entity.box_y2),
                confidence=entity.confidence,
                class_name=entity.class_name,
                class_id=0,
            ),
            crop_image=image_to_analyze,
            expanded_box=(entity.box_x1, entity.box_y1, entity.box_x2, entity.box_y2),
        )
        
        answer, _ = pipeline.analyze_single_crop(crop_info, prompt=prompt)  # Discard embedding for Q&A
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"VLM inference failed: {str(e)}")
    
    elapsed = time.perf_counter() - start_time
    
    return {
        "success": True,
        "object_id": object_id,
        "question": request.question,
        "answer": answer,
        "elapsed_seconds": round(elapsed, 2),
    }


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get session metadata."""
    db = await get_db()
    session = await db.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    entities = await db.get_session_entities(session_id)
    
    return {
        "session_id": session_id,
        "num_entities": len(entities),
        "entity_ids": [e.object_id for e in entities],
        "created_at": session.created_at,
    }


@app.post("/analyze/full-stream")
async def analyze_full_stream(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
):
    """
    Analyze image with streaming progress, then return complete results.
    
    Streams SSE events for progress, then final complete event with all data.
    """
    if not pipeline or not pipeline.initialized:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    
    # Load image
    image = await load_upload_image(file)
    analysis_prompt = prompt if prompt else ANALYSIS_PROMPT
    
    async def event_generator():
        start_time = time.perf_counter()
        
        # REFACTOR: Collect results during streaming (single-pass analysis)
        # Previously, we ran analyze_streaming() for progress, then analyze() again
        # for final results. Now we collect results from streaming events to halve runtime.
        collected_results = []
        
        try:
            # Stream analysis with progress events
            if hasattr(pipeline, 'analyze_streaming'):
                try:
                    async for event in pipeline.analyze_streaming(image, prompt=analysis_prompt):
                        # Skip the streaming 'complete' event - we'll send our own with results
                        if event.event_type == "complete":
                            continue
                        
                        # Collect full results from crop_analyzed events (includes crop images)
                        if event.event_type == "crop_analyzed" and event.result is not None:
                            collected_results.append(event.result)
                        
                        elapsed = round(time.perf_counter() - start_time, 2)
                        
                        event_data = {
                            "type": event.event_type,
                            "elapsed": elapsed,
                            "data": event.data,
                        }
                        yield f"data: {json.dumps(event_data)}\n\n"
                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    print(f"Pipeline streaming failed: {error_details}")
                    yield f"data: {json.dumps({'type': 'error', 'message': f'Pipeline streaming failed: {str(e)}'})}\n\n"
                    return
            else:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Streaming not available on pipeline'})}\n\n"
                return
            
            # Get database instance
            db = await get_db()
            
            # Create session ID
            session_id = str(uuid.uuid4())[:12]
            
            # Save full image to disk
            try:
                full_image_path = save_session_image(image, session_id)
            except Exception as e:
                print(f"Failed to save session image: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': f'Failed to save session image: {str(e)}'})}\n\n"
                return
            
            # Save session metadata to database
            try:
                await db.create_session(
                    session_id=session_id,
                    full_image_path=str(full_image_path),
                    image_width=image.width,
                    image_height=image.height,
                )
            except Exception as e:
                print(f"Failed to create session in database: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': f'Failed to create session: {str(e)}'})}\n\n"
                return
            
            # Build final response from collected results (no second analysis needed!)
            final_data = {
                "type": "complete",
                "elapsed": round(time.perf_counter() - start_time, 2),
                "session_id": session_id,
                "results": [],
            }
            
            # Sort results by detection index for consistent ordering
            collected_results.sort(key=lambda r: r.crop_info.detection.index)
            
            for result in collected_results:
                try:
                    object_id = generate_object_id()
                    
                    # Save crop image to disk
                    try:
                        crop_image_path = save_crop_image(result.crop_info.crop_image, object_id)
                    except Exception as e:
                        print(f"Failed to save crop image for {object_id}: {e}")
                        yield f"data: {json.dumps({'type': 'error', 'message': f'Failed to save crop image: {str(e)}'})}\n\n"
                        return
                    
                    # Save entity metadata to database
                    try:
                        await db.create_entity(
                            object_id=object_id,
                            session_id=session_id,
                            class_name=result.crop_info.detection.class_name,
                            confidence=result.crop_info.detection.confidence,
                            box=result.crop_info.detection.box,
                            crop_image_path=str(crop_image_path),
                            initial_analysis=result.analysis_text,
                            stage=result.stage.value,
                        )
                    except Exception as e:
                        print(f"Failed to create entity in database: {e}")
                        yield f"data: {json.dumps({'type': 'error', 'message': f'Failed to save entity to database: {str(e)}'})}\n\n"
                        return
                    
                    # Store embedding from pipeline result (no separate extraction needed)
                    if result.embedding is not None:
                        try:
                            await db.update_entity_embedding(object_id, result.embedding)
                        except Exception as e:
                            print(f"Warning: Failed to store embedding: {e}")
                    
                    # Use URL instead of base64 encoding (much faster)
                    crop_image_url = f"/api/image/crop/{object_id}"
                    
                    final_data["results"].append({
                        "object_id": object_id,
                        "index": result.crop_info.detection.index,
                        "class": result.crop_info.detection.class_name,
                        "confidence": round(result.crop_info.detection.confidence, 3),
                        "box": result.crop_info.detection.box,
                        "stage": result.stage.value,
                        "analysis": result.analysis_text,
                        "reason": result.reason,
                        "crop_image": crop_image_url,
                    })
                except Exception as e:
                    print(f"Error processing result: {e}")
                    # Continue with other results instead of failing completely
                    continue
            
            yield f"data: {json.dumps(final_data)}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# HISTORY & MANAGEMENT ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/sessions")
async def list_sessions(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """
    List all analysis sessions for the history view.
    
    Returns sessions with entity counts and thumbnail paths.
    """
    db = await get_db()
    sessions = await db.list_sessions_with_entity_count(limit=limit, offset=offset)
    total = await db.get_session_count()
    
    return {
        "sessions": sessions,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.get("/api/session/{session_id}")
async def get_session_details(session_id: str):
    """
    Get full session details including all entities with their data.
    """
    db = await get_db()
    session = await db.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    entities = await db.get_session_entities(session_id)
    
    # Build entity data with crop images
    entity_list = []
    for e in entities:
        # Load crop image and convert to base64
        try:
            crop_img = load_image(Path(e.crop_image_path))
            crop_b64 = image_to_base64(crop_img)
        except Exception:
            crop_b64 = None
        
        entity_list.append({
            "object_id": e.object_id,
            "class_name": e.class_name,
            "confidence": e.confidence,
            "box": [e.box_x1, e.box_y1, e.box_x2, e.box_y2],
            "stage": e.stage,
            "analysis": e.initial_analysis,
            "crop_image": f"data:image/jpeg;base64,{crop_b64}" if crop_b64 else None,
            "created_at": e.created_at,
        })
    
    return {
        "session_id": session_id,
        "created_at": session.created_at,
        "image_width": session.image_width,
        "image_height": session.image_height,
        "full_image_path": session.full_image_path,
        "entities": entity_list,
    }


@app.get("/api/image/{image_type}/{image_id}")
async def serve_stored_image(image_type: str, image_id: str):
    """
    Serve stored images (session full images or entity crops).
    
    Args:
        image_type: "session" or "crop"
        image_id: session_id or object_id
    """
    from storage_utils import SESSIONS_DIR, CROPS_DIR
    
    if image_type == "session":
        image_path = SESSIONS_DIR / f"{image_id}.jpg"
    elif image_type == "crop":
        image_path = CROPS_DIR / f"{image_id}.jpg"
    else:
        raise HTTPException(status_code=400, detail="Invalid image type")
    
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(
        image_path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"}
    )


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a session and all its entities.
    
    Also removes associated image files from disk.
    """
    from storage_utils import SESSIONS_DIR, CROPS_DIR
    import os
    
    db = await get_db()
    
    # Get session and entities before deleting
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    entities = await db.get_session_entities(session_id)
    
    # Delete from database
    deleted = await db.delete_session(session_id)
    
    if deleted:
        # Clean up image files
        try:
            session_image = SESSIONS_DIR / f"{session_id}.jpg"
            if session_image.exists():
                os.remove(session_image)
            
            for entity in entities:
                crop_image = CROPS_DIR / f"{entity.object_id}.jpg"
                if crop_image.exists():
                    os.remove(crop_image)
        except Exception as e:
            # Log but don't fail if file cleanup fails
            print(f"Warning: Failed to clean up some files: {e}")
    
    return {
        "success": deleted,
        "session_id": session_id,
        "entities_deleted": len(entities),
    }


@app.delete("/api/entity/{object_id}")
async def delete_entity(object_id: str):
    """
    Delete a single entity.
    
    Also removes the crop image from disk.
    """
    from storage_utils import CROPS_DIR
    import os
    
    db = await get_db()
    
    # Get entity before deleting
    entity = await db.get_entity(object_id)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    # Delete from database
    deleted = await db.delete_entity(object_id)
    
    if deleted:
        # Clean up crop image
        try:
            crop_image = CROPS_DIR / f"{object_id}.jpg"
            if crop_image.exists():
                os.remove(crop_image)
        except Exception as e:
            print(f"Warning: Failed to delete crop image: {e}")
    
    return {
        "success": deleted,
        "object_id": object_id,
        "session_id": entity.session_id,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SEARCH ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

class SearchRequest(BaseModel):
    """Request body for hybrid search."""
    text_query: Optional[str] = None
    text_weight: float = 0.5
    vector_weight: float = 0.5
    limit: int = 20
    min_score: float = 0.1


@app.post("/api/search")
async def hybrid_search(
    request: SearchRequest,
    file: Optional[UploadFile] = File(None),
):
    """
    Hybrid search combining text and image similarity.
    
    This endpoint performs a weighted combination of:
    1. Full-text search on analysis text (PostgreSQL tsvector)
    2. Vector similarity search on image embeddings (cosine similarity)
    
    Search Modes:
    - Text only: Provide text_query, no file
    - Image only: Upload file, no text_query
    - Hybrid: Both text_query and file, scores are combined
    
    Scoring:
        hybrid_score = (text_weight * text_score) + (vector_weight * vector_score)
    
    Args:
        request: SearchRequest with query parameters
        file: Optional image file for similarity search
        
    Returns:
        List of matching entities with scores
    """
    db = await get_db()
    
    # Get text query from request body
    text_query = request.text_query
    
    # Get image embedding if file provided
    image_embedding = None
    if file and embedding_extractor:
        try:
            image = await load_upload_image(file)
            image_embedding = embedding_extractor.generate_embedding(image)
        except Exception as e:
            raise HTTPException(400, f"Failed to process image: {e}")
    
    # Perform hybrid search
    if not text_query and image_embedding is None:
        raise HTTPException(400, "Must provide text_query or image file")
    
    results = await db.hybrid_search(
        text_query=text_query,
        image_embedding=image_embedding,
        text_weight=request.text_weight,
        vector_weight=request.vector_weight,
        limit=request.limit,
        min_score=request.min_score,
    )
    
    # Format results with crop images
    formatted = []
    for sr in results:
        entity = sr.entity
        
        # Load crop image
        try:
            crop_img = load_image(Path(entity.crop_image_path))
            crop_b64 = image_to_base64(crop_img)
        except Exception:
            crop_b64 = None
        
        formatted.append({
            "object_id": entity.object_id,
            "session_id": entity.session_id,
            "class_name": entity.class_name,
            "confidence": entity.confidence,
            "analysis": entity.initial_analysis,
            "crop_image": f"data:image/jpeg;base64,{crop_b64}" if crop_b64 else None,
            "scores": {
                "text": round(sr.text_score, 3),
                "vector": round(sr.vector_score, 3),
                "hybrid": round(sr.hybrid_score, 3),
            }
        })
    
    return {
        "results": formatted,
        "count": len(formatted),
        "query": {
            "text": text_query,
            "has_image": image_embedding is not None,
            "text_weight": request.text_weight,
            "vector_weight": request.vector_weight,
        }
    }


@app.get("/api/search/text")
async def search_by_text(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(50, ge=1, le=200),
):
    """
    Text-only search on entity analysis.
    
    Uses PostgreSQL full-text search with relevance ranking.
    
    Args:
        q: Search query text
        limit: Maximum results
        
    Returns:
        List of matching entities
    """
    db = await get_db()
    results = await db.search_text_with_score(q, limit)
    
    formatted = []
    for entity, score in results:
        try:
            crop_img = load_image(Path(entity.crop_image_path))
            crop_b64 = image_to_base64(crop_img)
        except Exception:
            crop_b64 = None
        
        formatted.append({
            "object_id": entity.object_id,
            "session_id": entity.session_id,
            "class_name": entity.class_name,
            "analysis": entity.initial_analysis,
            "crop_image": f"data:image/jpeg;base64,{crop_b64}" if crop_b64 else None,
            "score": round(score, 3),
        })
    
    return {
        "results": formatted,
        "count": len(formatted),
        "query": q,
    }


@app.post("/api/search/image")
async def search_by_image(
    file: UploadFile = File(...),
    limit: int = Query(20, ge=1, le=100),
    min_similarity: float = Query(0.0, ge=0.0, le=1.0),
):
    """
    Image similarity search using vector embeddings.
    
    Upload an image to find visually similar entities in the database.
    Uses cosine similarity on Qwen2.5-VL vision embeddings.
    
    Args:
        file: Image file to search for
        limit: Maximum results
        min_similarity: Minimum similarity score (0-1)
        
    Returns:
        List of similar entities with similarity scores
    """
    if not embedding_extractor:
        raise HTTPException(503, "Embedding extractor not initialized")
    
    # Generate query embedding
    try:
        image = await load_upload_image(file)
        query_embedding = embedding_extractor.generate_embedding(image)
    except Exception as e:
        raise HTTPException(400, f"Failed to process image: {e}")
    
    # Search database
    db = await get_db()
    results = await db.search_by_vector(
        query_embedding=query_embedding,
        limit=limit,
        min_similarity=min_similarity,
    )
    
    formatted = []
    for entity, similarity in results:
        try:
            crop_img = load_image(Path(entity.crop_image_path))
            crop_b64 = image_to_base64(crop_img)
        except Exception:
            crop_b64 = None
        
        formatted.append({
            "object_id": entity.object_id,
            "session_id": entity.session_id,
            "class_name": entity.class_name,
            "analysis": entity.initial_analysis,
            "crop_image": f"data:image/jpeg;base64,{crop_b64}" if crop_b64 else None,
            "similarity": round(similarity, 3),
        })
    
    return {
        "results": formatted,
        "count": len(formatted),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
