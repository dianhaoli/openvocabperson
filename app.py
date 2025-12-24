#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI Backend for Hierarchical Vision Analysis Pipeline

Endpoints:
    GET  /           - Serve frontend
    POST /analyze    - Analyze image, return JSON results
    POST /analyze/stream - Analyze image with SSE streaming
    GET  /health     - Health check

Usage:
    uvicorn app:app --reload --port 8000
"""

import asyncio
import base64
import io
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from pipeline import (
    ANALYSIS_PROMPT,
    HierarchicalPipeline,
    PipelineConfig,
    create_pipeline,
)


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL STATE
# ══════════════════════════════════════════════════════════════════════════════

pipeline: Optional[HierarchicalPipeline] = None


# ══════════════════════════════════════════════════════════════════════════════
# LIFESPAN - Initialize/cleanup pipeline
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize pipeline on startup, cleanup on shutdown."""
    global pipeline
    
    print("🚀 Starting Hierarchical Vision Analysis API...")
    
    # Create pipeline with optimized settings
    config = PipelineConfig(
        use_quantization=True,
        use_torch_compile=False,  # Avoid compile issues with quantization
        max_new_tokens=50,
        batch_size=4,
    )
    pipeline = HierarchicalPipeline(config)
    
    # Load models (this takes time)
    print("📦 Loading models (this may take a minute)...")
    await pipeline.initialize()
    print("✅ API ready!")
    
    yield
    
    # Cleanup
    if pipeline:
        pipeline.cleanup()
    print("👋 API shutdown complete.")


# ══════════════════════════════════════════════════════════════════════════════
# APP SETUP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Hierarchical Vision Analysis API",
    description="YOLO detection + VLM analysis pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
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
    return HTMLResponse(content="<h1>API Running</h1><p>Frontend not found. Place index.html in /static/</p>")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "pipeline_ready": pipeline is not None and pipeline.initialized,
    }


@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
):
    """
    Analyze an uploaded image.
    
    Returns complete results as JSON.
    """
    if not pipeline or not pipeline.initialized:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    
    # Load image
    image = await load_upload_image(file)
    
    # Use custom prompt or default
    analysis_prompt = prompt if prompt else ANALYSIS_PROMPT
    
    # Run analysis
    start_time = time.perf_counter()
    results = pipeline.analyze(image, prompt=analysis_prompt)
    elapsed = time.perf_counter() - start_time
    
    # Format response
    response_data = {
        "success": True,
        "elapsed_seconds": round(elapsed, 2),
        "num_detections": len(results),
        "results": [],
    }
    
    for result in results:
        crop_b64 = image_to_base64(result.crop_info.crop_image)
        response_data["results"].append({
            "index": result.crop_info.detection.index,
            "class": result.crop_info.detection.class_name,
            "confidence": round(result.crop_info.detection.confidence, 3),
            "box": result.crop_info.detection.box,
            "stage": result.stage.value,
            "analysis": result.analysis_text,
            "reason": result.reason,
            "crop_image": f"data:image/jpeg;base64,{crop_b64}",
        })
    
    return response_data


@app.post("/analyze/stream")
async def analyze_stream(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
):
    """
    Analyze an uploaded image with streaming results.
    
    Returns Server-Sent Events (SSE) stream.
    """
    if not pipeline or not pipeline.initialized:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    
    # Load image
    image = await load_upload_image(file)
    
    # Use custom prompt or default
    analysis_prompt = prompt if prompt else ANALYSIS_PROMPT
    
    async def event_generator():
        """Generate SSE events as analysis progresses."""
        start_time = time.perf_counter()
        
        try:
            async for event in pipeline.analyze_streaming(image, prompt=analysis_prompt):
                elapsed = round(time.perf_counter() - start_time, 2)
                
                event_data = {
                    "type": event.event_type,
                    "elapsed": elapsed,
                    "data": event.data,
                }
                
                # Add crop image if this is a crop_analyzed event with VLM analysis
                if event.event_type == "crop_analyzed":
                    # Get the crop image from the pipeline's last detection
                    # For streaming, we need to encode crops on-the-fly
                    pass  # Crops will be fetched separately or included in final
                
                yield f"data: {json.dumps(event_data)}\n\n"
            
            # Final event
            total_time = round(time.perf_counter() - start_time, 2)
            yield f"data: {json.dumps({'type': 'done', 'elapsed': total_time})}\n\n"
            
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


@app.post("/analyze/full-stream")
async def analyze_full_stream(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
):
    """
    Analyze image with streaming, including crop images in final results.
    
    Streams progress events, then sends complete results with images.
    """
    if not pipeline or not pipeline.initialized:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    
    # Load image
    image = await load_upload_image(file)
    analysis_prompt = prompt if prompt else ANALYSIS_PROMPT
    
    async def event_generator():
        start_time = time.perf_counter()
        collected_results = []
        
        try:
            async for event in pipeline.analyze_streaming(image, prompt=analysis_prompt):
                elapsed = round(time.perf_counter() - start_time, 2)
                
                # Stream progress events
                if event.event_type in ("detection_complete", "routing_complete"):
                    yield f"data: {json.dumps({'type': event.event_type, 'elapsed': elapsed, 'data': event.data})}\n\n"
                
                elif event.event_type == "crop_analyzed":
                    collected_results.append(event.data)
                    # Stream partial result (without image for speed)
                    yield f"data: {json.dumps({'type': 'crop_analyzed', 'elapsed': elapsed, 'data': event.data})}\n\n"
            
            # Now run sync analysis to get crop images
            # (streaming doesn't preserve crop images, so we re-fetch)
            full_results = pipeline.analyze(image, prompt=analysis_prompt)
            
            # Send final results with images
            final_data = {
                "type": "complete",
                "elapsed": round(time.perf_counter() - start_time, 2),
                "results": [],
            }
            
            for result in full_results:
                crop_b64 = image_to_base64(result.crop_info.crop_image)
                final_data["results"].append({
                    "index": result.crop_info.detection.index,
                    "class": result.crop_info.detection.class_name,
                    "confidence": round(result.crop_info.detection.confidence, 3),
                    "stage": result.stage.value,
                    "analysis": result.analysis_text,
                    "reason": result.reason,
                    "crop_image": f"data:image/jpeg;base64,{crop_b64}",
                })
            
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

