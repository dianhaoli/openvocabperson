#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI Backend for Hierarchical Vision Analysis Pipeline

Endpoints:
    GET  /                      - Serve frontend
    POST /analyze               - Analyze image, return JSON results with object IDs
    POST /analyze/full-stream   - Analyze with SSE streaming progress
    POST /object/{id}/ask       - Ask follow-up question about specific entity
    GET  /health                - Health check

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
from typing import Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from pipeline import (
    ANALYSIS_PROMPT,
    HierarchicalPipeline,
    PipelineConfig,
    create_pipeline,
)


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

# Simple in-memory session storage (for demo purposes)
# In production, use Redis or similar
sessions: Dict[str, SessionData] = {}
MAX_SESSIONS = 10  # Limit memory usage


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
    """Initialize pipeline on startup, cleanup on shutdown."""
    global pipeline
    
    print("Starting Vision Analysis API...")
    
    config = PipelineConfig(
        use_quantization=True,
        use_torch_compile=False,
        max_new_tokens=100,  # Allow longer responses for Q&A
        batch_size=4,
    )
    pipeline = HierarchicalPipeline(config)
    
    print("Loading models (this may take a minute)...")
    await pipeline.initialize()
    print("API ready!")
    
    yield
    
    if pipeline:
        pipeline.cleanup()
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


def cleanup_old_sessions():
    """Remove oldest sessions if over limit."""
    global sessions
    if len(sessions) > MAX_SESSIONS:
        # Sort by creation time, remove oldest
        sorted_sessions = sorted(sessions.items(), key=lambda x: x[1].created_at)
        for session_id, _ in sorted_sessions[:len(sessions) - MAX_SESSIONS]:
            del sessions[session_id]


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
    return {
        "status": "healthy",
        "pipeline_ready": pipeline is not None and pipeline.initialized,
        "active_sessions": len(sessions),
    }


@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
):
    """
    Analyze an uploaded image.
    
    Returns results with stable object_ids for each detection.
    Stores session data for follow-up questions.
    """
    if not pipeline or not pipeline.initialized:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    
    image = await load_upload_image(file)
    analysis_prompt = prompt if prompt else ANALYSIS_PROMPT
    
    start_time = time.perf_counter()
    results = pipeline.analyze(image, prompt=analysis_prompt)
    elapsed = time.perf_counter() - start_time
    
    # Create session
    session_id = str(uuid.uuid4())[:12]
    session = SessionData(
        session_id=session_id,
        full_image=image,
    )
    
    # Build response and store entities
    response_data = {
        "success": True,
        "session_id": session_id,
        "elapsed_seconds": round(elapsed, 2),
        "num_detections": len(results),
        "image_width": image.width,
        "image_height": image.height,
        "results": [],
    }
    
    for result in results:
        object_id = generate_object_id()
        
        # Store entity data
        entity = EntityData(
            object_id=object_id,
            class_name=result.crop_info.detection.class_name,
            confidence=result.crop_info.detection.confidence,
            box=result.crop_info.detection.box,
            crop_image=result.crop_info.crop_image,
            initial_analysis=result.analysis_text,
            stage=result.stage.value,
        )
        session.entities[object_id] = entity
        
        crop_b64 = image_to_base64(result.crop_info.crop_image)
        response_data["results"].append({
            "object_id": object_id,
            "index": result.crop_info.detection.index,
            "class": result.crop_info.detection.class_name,
            "confidence": round(result.crop_info.detection.confidence, 3),
            "box": result.crop_info.detection.box,  # (x1, y1, x2, y2)
            "stage": result.stage.value,
            "analysis": result.analysis_text,
            "reason": result.reason,
            "crop_image": f"data:image/jpeg;base64,{crop_b64}",
        })
    
    # Store session and cleanup old ones
    sessions[session_id] = session
    cleanup_old_sessions()
    
    return response_data


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
    
    # Find the entity across all sessions
    entity: Optional[EntityData] = None
    session: Optional[SessionData] = None
    
    for sess in sessions.values():
        if object_id in sess.entities:
            entity = sess.entities[object_id]
            session = sess
            break
    
    if not entity:
        raise HTTPException(
            status_code=404, 
            detail=f"Object {object_id} not found. Session may have expired."
        )
    
    # Build grounded prompt
    prompt = build_followup_prompt(
        question=request.question,
        initial_analysis=entity.initial_analysis,
        use_full_scene=request.use_full_scene,
    )
    
    # Choose image based on user preference: full scene or crop only
    if request.use_full_scene:
        image_to_analyze = session.full_image
    else:
        image_to_analyze = entity.crop_image
    
    # Run VLM inference
    start_time = time.perf_counter()
    
    try:
        answer = pipeline.analyze_single_crop(
            crop_info=type('CropInfo', (), {
                'crop_image': image_to_analyze,  # Uses chosen image (crop or full scene)
                'detection': type('Detection', (), {
                    'index': 0,
                    'box': entity.box,
                    'confidence': entity.confidence,
                    'class_name': entity.class_name,
                    'class_id': 0,
                })()
            })(),
            prompt=prompt,
        )
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
    """Get session metadata (for debugging)."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    return {
        "session_id": session_id,
        "num_entities": len(session.entities),
        "entity_ids": list(session.entities.keys()),
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
            
            # Create session for follow-up Q&A
            session_id = str(uuid.uuid4())[:12]
            session = SessionData(
                session_id=session_id,
                full_image=image,
            )
            
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
                object_id = generate_object_id()
                
                # Store entity for follow-up Q&A
                entity = EntityData(
                    object_id=object_id,
                    class_name=result.crop_info.detection.class_name,
                    confidence=result.crop_info.detection.confidence,
                    box=result.crop_info.detection.box,
                    crop_image=result.crop_info.crop_image,
                    initial_analysis=result.analysis_text,
                    stage=result.stage.value,
                )
                session.entities[object_id] = entity
                
                crop_b64 = image_to_base64(result.crop_info.crop_image)
                final_data["results"].append({
                    "object_id": object_id,
                    "index": result.crop_info.detection.index,
                    "class": result.crop_info.detection.class_name,
                    "confidence": round(result.crop_info.detection.confidence, 3),
                    "box": result.crop_info.detection.box,
                    "stage": result.stage.value,
                    "analysis": result.analysis_text,
                    "reason": result.reason,
                    "crop_image": f"data:image/jpeg;base64,{crop_b64}",
                })
            
            # Store session and cleanup old ones
            sessions[session_id] = session
            cleanup_old_sessions()
            
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
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
