# Open-Vocabulary Person Analysis Pipeline

A production-ready hierarchical vision system that combines **YOLO** for fast detection with **Qwen2.5-VL** Vision-Language Model for detailed person analysis. Features natural language search, image similarity matching, and interactive Q&A.

![Demo](https://github.com/user-attachments/assets/81d30065-eaf9-461d-bddb-ee7183e1b14b)

## Demo Image - Video Coming Soon

<img width="1683" height="920" alt="image" src="https://github.com/user-attachments/assets/f7a1219f-46c3-4930-b70b-e4ac124cf136" />

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
| **Modern Frontend** | React + TypeScript frontend with real-time updates |
| **Docker Support** | Full containerization with docker-compose |

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
├── app.py                  # FastAPI server (REST + SSE streaming)
├── pipeline.py             # Hierarchical detection/analysis pipeline
├── database.py             # PostgreSQL + pgvector hybrid search
├── embeddings.py           # Image embedding extraction from VLM
├── storage_utils.py        # Image storage utilities
├── demo.py                 # CLI demo for testing
├── model.ipynb             # Development notebook
├── requirements.txt        # Python dependencies
├── Dockerfile              # Backend Docker image
├── docker-compose.yml      # Full stack orchestration
├── .dockerignore           # Docker build exclusions
├── .gitignore              # Git exclusions
│
├── frontend/               # React + TypeScript frontend
│   ├── src/
│   │   ├── api/            # Typed API client functions
│   │   ├── components/     # React components
│   │   │   ├── canvas/     # Canvas with bounding boxes
│   │   │   ├── entity/     # Entity panel and Q&A
│   │   │   ├── layout/     # Header, Sidebar, MainLayout
│   │   │   ├── results/    # Results grid and cards
│   │   │   ├── sidebar/    # Upload, Search, History tabs
│   │   │   └── ui/         # Reusable UI components
│   │   ├── context/        # React Context for global state
│   │   ├── hooks/          # Custom React hooks
│   │   ├── types/          # TypeScript type definitions
│   │   └── utils/          # Utility functions
│   ├── public/             # Static assets
│   ├── Dockerfile          # Frontend Docker image
│   ├── nginx.conf          # Nginx configuration
│   ├── package.json        # Node dependencies
│   ├── tsconfig.json       # TypeScript configuration
│   └── vite.config.ts      # Vite build configuration
│
├── static/                 # Legacy static frontend (index.html)
│   └── index.html          # Simple HTML/JS frontend
│
└── storage/                # Image storage (gitignored)
    ├── sessions/           # Full session images
    └── crops/              # Cropped person images
```

---

## Quick Start

### Option 1: Docker Compose (Recommended)

The easiest way to get started is using Docker Compose, which sets up the entire stack:

```bash
# Clone repository
git clone https://github.com/dianhaoli/openvocabperson.git
cd openvocabperson

# Start all services (PostgreSQL, backend, frontend)
docker-compose up -d

# View logs
docker-compose logs -f

# Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
```

The services will be available at:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **PostgreSQL**: localhost:5432

Models will download automatically on first startup (~30-60 seconds).

### Option 2: Local Development

#### Prerequisites

- Python 3.10+
- Node.js 18+ (for frontend development)
- PostgreSQL 15+ with pgvector extension
- CUDA GPU (8GB+ VRAM recommended) or Apple Silicon (MPS)

#### Backend Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Setup PostgreSQL
createdb vision_analysis
psql vision_analysis -c "CREATE EXTENSION vector;"

# Configure database (optional, defaults to localhost)
export DATABASE_URL="postgresql://user:pass@localhost:5432/vision_analysis"

# Start API server
uvicorn app:app --host 0.0.0.0 --port 8000
```

Models load on startup (~30-60 seconds). The API will be ready when you see "API ready!"

#### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server (with API proxy to localhost:8000)
npm run dev

# Build for production
npm run build
```

The frontend will be available at http://localhost:5173 (Vite default port) and will proxy API requests to the backend.

---

## Environment Variables

Create a `.env` file in the root directory (optional, defaults provided):

```bash
# Database Configuration
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/vision_analysis

# Optional: Model paths (if using custom models)
# YOLO_MODEL_PATH=yolo11l.pt
# VLM_MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct
```

The application will use sensible defaults if environment variables are not set.

---

## API Reference

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web frontend (legacy static HTML) |
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
| `/api/image/{type}/{id}` | GET | Serve stored images (type: `session` or `crop`) |

### Request/Response Examples

#### Analyze Image

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

#### Streaming Analysis (SSE)

```bash
curl -X POST -F "file=@image.jpg" http://localhost:8000/analyze/full-stream
```

Returns Server-Sent Events with progress updates:
```
event: progress
data: {"stage": "yolo", "message": "Running YOLO detection..."}

event: progress
data: {"stage": "vlm", "message": "Analyzing 3 detections..."}

event: result
data: {"success": true, "session_id": "...", ...}
```

#### Ask Follow-up Question

```bash
curl -X POST http://localhost:8000/object/18a7b3ba/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What color is their jacket?", "use_full_scene": false}'
```

**Response:**
```json
{
  "success": true,
  "answer": "The person is wearing a dark blue jacket.",
  "object_id": "18a7b3ba"
}
```

#### Hybrid Search

```bash
curl -X POST http://localhost:8000/api/search \
  -F "file=@reference.jpg" \
  -F 'json={"text_query": "red jacket", "text_weight": 0.3, "vector_weight": 0.7, "limit": 10}'
```

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "entity": {
        "object_id": "18a7b3ba",
        "session_id": "8f71a028-cf5",
        "class_name": "person",
        "confidence": 0.87,
        "initial_analysis": "...",
        "crop_image_path": "storage/crops/18a7b3ba.jpg"
      },
      "text_score": 0.85,
      "vector_score": 0.92,
      "hybrid_score": 0.90
    }
  ]
}
```

#### Text-Only Search

```bash
curl "http://localhost:8000/api/search/text?q=red+jacket&limit=10"
```

#### Image Similarity Search

```bash
curl -X POST http://localhost:8000/api/search/image \
  -F "file=@reference.jpg" \
  -F "limit=10"
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
    detection_iou=0.5,             # NMS IoU threshold
    expand_ratio=0.1,              # Expand bounding boxes by 10%
    
    # Optimization
    use_quantization=True,         # INT8 quantization
    quantization_bits=8,           # 8 for quality, 4 for speed
    use_torch_compile=True,        # PyTorch 2.0 compile (disabled with quantization)
    
    # Batching
    batch_size=4,                  # VLM batch size
    reference_area=512*512,        # Area budget per batch
    
    # Device
    device="auto",                # "auto", "cuda", "mps", or "cpu"
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
| **SDPA Attention** | 2x faster attention | Auto-enabled in PyTorch 2.0+ |
| **torch.compile** | 10-20% faster | Disabled with quantization |
| **pgvector HNSW** | O(log n) search | Approximate nearest neighbor |

### Memory Requirements

| Configuration | VRAM Usage |
|--------------|------------|
| FP16 (no quantization) | ~10GB |
| INT8 quantization | ~6GB |
| INT4 quantization | ~4GB |

### Performance Tips

1. **GPU Acceleration**: Ensure CUDA is properly installed for NVIDIA GPUs, or use MPS for Apple Silicon
2. **Batch Size**: Increase `batch_size` for better throughput if you have VRAM
3. **Quantization**: Use INT8 for best balance, INT4 for lower-end hardware
4. **Database Indexing**: pgvector HNSW index is created automatically for fast similarity search

---

## Models Used

| Model | Purpose | Size | Source |
|-------|---------|------|--------|
| [YOLO11-Large](https://docs.ultralytics.com/) | Object detection | 25M params | Ultralytics |
| [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | Vision-language analysis | 3B params | Alibaba Qwen |

Both models are loaded once at startup and reused for all requests. Models are automatically downloaded from Hugging Face on first run.

---

## Tech Stack

### Backend
- **FastAPI** - Modern async web framework
- **asyncpg** - Async PostgreSQL driver
- **uvicorn** - ASGI server
- **PyTorch** - Deep learning framework
- **Transformers** - Hugging Face model library
- **Ultralytics YOLO** - Object detection
- **PostgreSQL** - Relational database
- **pgvector** - Vector similarity extension

### Frontend
- **React 19** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **Tailwind CSS v4** - Utility-first styling
- **PostCSS** - CSS processing

### Search
- **Full-text search** - PostgreSQL tsvector (GIN index)
- **Vector similarity** - pgvector HNSW index
- **Hybrid scoring** - Weighted combination of text and vector scores

---

## Development

### Backend Development

```bash
# Run with hot reload
uvicorn app:app --reload --port 8000

# Run pipeline tests
python pipeline.py test_image.jpg

# Run CLI demo
python demo.py                    # Uses default test image
python demo.py path/to/image.jpg  # Custom image
```

### Frontend Development

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Lint code
npm run lint
```

### Database Development

```bash
# Connect to database
psql vision_analysis

# View tables
\dt

# View sessions
SELECT * FROM sessions;

# View entities
SELECT * FROM entities;

# Check pgvector extension
SELECT * FROM pg_extension WHERE extname = 'vector';
```

### Testing

```bash
# Test API health
curl http://localhost:8000/health

# Test image analysis
curl -X POST -F "file=@test_image.jpg" http://localhost:8000/analyze

# Test search
curl "http://localhost:8000/api/search/text?q=person"
```

---

## Docker Deployment

### Building Images

```bash
# Build backend
docker build -t vision-backend .

# Build frontend
cd frontend
docker build -t vision-frontend .

# Or use docker-compose
docker-compose build
```

### Running with Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Stop and remove volumes (clears database)
docker-compose down -v
```

### GPU Support (NVIDIA)

To enable GPU support in Docker, uncomment the GPU section in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

### Production Considerations

1. **Environment Variables**: Use `.env` file or Docker secrets for sensitive data
2. **Storage Volumes**: Persist `storage/` directory for images
3. **Database Backups**: Set up regular PostgreSQL backups
4. **Reverse Proxy**: Use nginx or Traefik for SSL/TLS termination
5. **Resource Limits**: Set appropriate CPU/memory limits in docker-compose

---

## Troubleshooting

### Common Issues

#### Models Not Loading

**Problem**: Models fail to download or load.

**Solutions**:
- Check internet connection (models download from Hugging Face)
- Verify disk space (models are ~6GB total)
- Check CUDA/GPU drivers if using GPU
- Try running with `device="cpu"` to test

#### Database Connection Errors

**Problem**: Cannot connect to PostgreSQL.

**Solutions**:
- Verify PostgreSQL is running: `pg_isready`
- Check `DATABASE_URL` environment variable
- Ensure pgvector extension is installed: `CREATE EXTENSION vector;`
- Check database credentials

#### Out of Memory (OOM)

**Problem**: GPU runs out of memory.

**Solutions**:
- Enable quantization (INT8 or INT4)
- Reduce batch size in `PipelineConfig`
- Use CPU mode: `device="cpu"` (slower but works)
- Close other GPU applications

#### Frontend Not Connecting to Backend

**Problem**: Frontend shows connection errors.

**Solutions**:
- Verify backend is running on port 8000
- Check CORS settings in `app.py`
- Verify API proxy configuration in `vite.config.ts`
- Check browser console for errors

#### Search Not Working

**Problem**: Search returns no results or errors.

**Solutions**:
- Verify pgvector extension is installed
- Check that embeddings are being generated (check `has_embedding` in database)
- Verify full-text search index exists
- Check database logs for errors

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or set environment variable:
```bash
export LOG_LEVEL=DEBUG
```

---

## CLI Demo

For testing without the server:

```bash
# Uses default test image
python demo.py

# Custom image
python demo.py path/to/image.jpg
```

The CLI demo runs the full pipeline locally and prints results to console.

---

## Database Schema

### Sessions Table

```sql
CREATE TABLE sessions (
    session_id VARCHAR PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW(),
    full_image_path VARCHAR NOT NULL,
    image_width INTEGER,
    image_height INTEGER
);
```

### Entities Table

```sql
CREATE TABLE entities (
    object_id VARCHAR PRIMARY KEY,
    session_id VARCHAR REFERENCES sessions(session_id) ON DELETE CASCADE,
    class_name VARCHAR NOT NULL,
    confidence FLOAT,
    box_x1 INTEGER,
    box_y1 INTEGER,
    box_x2 INTEGER,
    box_y2 INTEGER,
    crop_image_path VARCHAR NOT NULL,
    initial_analysis TEXT,
    stage VARCHAR,
    created_at TIMESTAMP DEFAULT NOW(),
    has_embedding BOOLEAN DEFAULT FALSE,
    embedding vector(1536)  -- pgvector column
);

-- Full-text search index
CREATE INDEX idx_entities_analysis ON entities USING GIN (to_tsvector('english', initial_analysis));

-- Vector similarity index
CREATE INDEX idx_entities_embedding ON entities USING hnsw (embedding vector_cosine_ops);
```

---

## License

MIT

---

## Acknowledgments

- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2-VL) by Alibaba
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [pgvector](https://github.com/pgvector/pgvector)
- [FastAPI](https://fastapi.tiangolo.com/)
- [React](https://react.dev/)

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## Roadmap

- [ ] Support for multiple object classes (not just person)
- [ ] Real-time video stream analysis
- [ ] Advanced filtering and sorting in search
- [ ] Export search results to CSV/JSON
- [ ] Batch image upload
- [ ] User authentication and multi-tenancy
- [ ] API rate limiting
- [ ] WebSocket support for real-time updates

---

## Support

For issues, questions, or contributions, please open an issue on GitHub.
