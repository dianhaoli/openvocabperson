# Open-Vocabulary Person Analysis Pipeline

A production-ready hierarchical vision system that combines **YOLO** for fast detection with **Qwen2.5-VL** Vision-Language Model for detailed person analysis. Features natural language search, image similarity matching, and interactive Q&A.

![Demo](https://github.com/user-attachments/assets/81d30065-eaf9-461d-bddb-ee7183e1b14b)

## Demo Video

**[Watch on YouTube](https://youtu.be/xvKNw0Qv5gY)**

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Hierarchical Detection** | YOLO → Class Filter → VLM routing saves compute by only running expensive VLM on priority detections |
| **Natural Language Search** | Full-text search on VLM analysis ("person wearing red jacket") |
| **Image Similarity** | Upload an image to find visually similar people using vector embeddings |
| **Hybrid Search** | Combine text + image queries with configurable weights |
| **Follow-up Q&A** | Ask questions about specific detected people |
| **Streaming Results** | SSE streaming for real-time progress updates |
| **Persistent Storage** | PostgreSQL with pgvector for sessions, entities, and embeddings |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Image                               │
└─────────────────────────────────┬───────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: YOLO Detection (Fast)                                  │
│  • YOLO11-Large for object detection                             │
│  • Returns bounding boxes + class + confidence                   │
└─────────────────────────────────┬───────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2: Hierarchical Routing (Instant)                         │
│  ┌─────────────────┬────────────────────┬─────────────────────┐ │
│  │ Non-Priority    │ Low Confidence     │ Priority + High     │ │
│  │ (car, dog...)   │ (person <0.3)      │ (person ≥0.3)       │ │
│  │       ↓         │        ↓           │        ↓            │ │
│  │   YOLO_ONLY     │  LOW_CONFIDENCE    │     VLM_FULL        │ │
│  │   (skip VLM)    │  (minimal proc)    │   (full analysis)   │ │
│  └─────────────────┴────────────────────┴─────────────────────┘ │
└─────────────────────────────────┬───────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 3: VLM Analysis (Only Priority Detections)                │
│  • Qwen2.5-VL-3B-Instruct                                        │
│  • Batched inference with size-aware batching                    │
│  • Extracts: actions, clothing, objects, visibility              │
│  • Generates embeddings for similarity search                    │
└─────────────────────────────────┬───────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  Storage: PostgreSQL + pgvector                                  │
│  • Sessions & entities with image paths                          │
│  • Full-text search index (GIN)                                  │
│  • Vector similarity index (HNSW)                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
.
├── app.py              # FastAPI server (REST + SSE streaming)
├── pipeline.py         # Hierarchical detection/analysis pipeline
├── database.py         # PostgreSQL + pgvector hybrid search
├── embeddings.py       # Image embedding extraction from VLM
├── storage_utils.py    # Image storage utilities
├── demo.py             # CLI demo for testing
├── model.ipynb         # Development notebook
├── static/
│   └── index.html      # Web frontend (upload, search, Q&A)
├── storage/
│   ├── sessions/       # Full session images
│   └── crops/          # Cropped person images
├── yolo11l.pt          # YOLO model weights
└── requirements.txt    # Python dependencies
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL 15+ with pgvector extension
- CUDA GPU (8GB+ VRAM recommended) or Apple Silicon (MPS)

### Installation

```bash
# Clone and install
git clone https://github.com/yourusername/openvocabperson.git
cd openvocabperson
pip install -r requirements.txt

# Setup PostgreSQL
createdb vision_analysis
psql vision_analysis -c "CREATE EXTENSION vector;"

# Configure database (optional, defaults to localhost)
export DATABASE_URL="postgresql://user:pass@localhost:5432/vision_analysis"
```

### Running

```bash
# Start API server
uvicorn app:app --host 0.0.0.0 --port 8000

# Open browser
open http://localhost:8000
```

Models load on startup (~30-60 seconds). The API will be ready when you see "API ready!"

---

## API Reference

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web frontend |
| `/health` | GET | Health check with session count |
| `/analyze` | POST | Analyze image, return JSON with object IDs |
| `/analyze/full-stream` | POST | SSE streaming with progress events |
| `/object/{id}/ask` | POST | Ask follow-up question about detected person |

### Search Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/search` | POST | Hybrid search (text + image) |
| `/api/search/text` | GET | Text-only search on analysis |
| `/api/search/image` | POST | Image similarity search |

### Session Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/sessions` | GET | List all sessions |
| `/api/session/{id}` | GET | Get session details |
| `/api/session/{id}` | DELETE | Delete session and entities |
| `/api/image/{type}/{id}` | GET | Serve stored images |

### Example: Analyze Image

```bash
curl -X POST -F "file=@image.jpg" http://localhost:8000/analyze
```

**Response:**
```json
{
  "success": true,
  "session_id": "8f71a028-cf5",
  "elapsed_seconds": 2.34,
  "num_detections": 3,
  "results": [
    {
      "object_id": "18a7b3ba",
      "class": "person",
      "confidence": 0.87,
      "stage": "vlm_full",
      "analysis": "A. Standing, walking. B. Holding phone. C. Dark jacket, jeans. D. Face partially obscured. E. Good visibility.",
      "box": [120, 50, 280, 400],
      "crop_image": "data:image/jpeg;base64,..."
    }
  ]
}
```

### Example: Ask Follow-up

```bash
curl -X POST http://localhost:8000/object/18a7b3ba/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What color is their jacket?"}'
```

### Example: Hybrid Search

```bash
curl -X POST http://localhost:8000/api/search \
  -F "file=@reference.jpg" \
  -F 'json={"text_query": "red jacket", "text_weight": 0.3, "vector_weight": 0.7}'
```

---

## Configuration

Edit settings in `pipeline.py`:

```python
PipelineConfig(
    # Models
    yolo_model="yolo11l.pt",
    vlm_model_id="Qwen/Qwen2.5-VL-3B-Instruct",
    
    # Detection
    detection_confidence=0.1,      # Min YOLO confidence
    high_confidence_threshold=0.3, # Route to VLM above this
    priority_classes=("person",),  # Classes for VLM analysis
    
    # Optimization
    use_quantization=True,         # INT8 quantization
    quantization_bits=8,           # 8 for quality, 4 for speed
    use_torch_compile=True,        # PyTorch 2.0 compile
    
    # Batching
    batch_size=4,                  # VLM batch size
    reference_area=512*512,        # Area budget per batch
)
```

---

## Performance Optimizations

| Optimization | Impact | Notes |
|--------------|--------|-------|
| **INT8 Quantization** | 40% less VRAM | Default, best quality/speed |
| **INT4 Quantization** | 60% less VRAM | Lower quality, faster |
| **Hierarchical Routing** | Skip VLM on non-person | Saves 70%+ compute |
| **Size-Aware Batching** | Stable latency | Packs crops by pixel area |
| **SDPA Attention** | 2x faster attention | Auto-enabled |
| **torch.compile** | 10-20% faster | Disabled with quantization |
| **pgvector HNSW** | O(log n) search | Approximate nearest neighbor |

### Memory Requirements

| Configuration | VRAM Usage |
|--------------|------------|
| FP16 (no quantization) | ~10GB |
| INT8 quantization | ~6GB |
| INT4 quantization | ~4GB |

---

## Models Used

| Model | Purpose | Size |
|-------|---------|------|
| [YOLO11-Large](https://docs.ultralytics.com/) | Object detection | 25M params |
| [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | Vision-language analysis | 3B params |

Both models are loaded once at startup and reused for all requests.

---

## Tech Stack

- **Backend**: FastAPI, asyncpg, uvicorn
- **ML**: PyTorch, Transformers, Ultralytics YOLO
- **Database**: PostgreSQL + pgvector
- **Search**: Full-text (tsvector) + Vector similarity (HNSW)

---

## CLI Demo

For testing without the server:

```bash
python demo.py                    # Uses default test image
python demo.py path/to/image.jpg  # Custom image
```

---

## Development

```bash
# Run with hot reload
uvicorn app:app --reload --port 8000

# Run pipeline tests
python pipeline.py test_image.jpg
```

---

## License

MIT

---

## Acknowledgments

- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2-VL) by Alibaba
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [pgvector](https://github.com/pgvector/pgvector)
