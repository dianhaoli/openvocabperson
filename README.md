# Open-Vocabulary Person Analysis

Upload an image, detect people with YOLO, describe each one with a vision-language model, recognize the same person across uploads, and search everything by text or by example image.

![Demo](https://github.com/user-attachments/assets/81d30065-eaf9-461d-bddb-ee7183e1b14b)

**[Demo video](https://www.youtube.com/watch?v=MsrKKbLeFY0)**

## What it does

- **Detect, then describe.** YOLO11-L finds objects; only confident `person` detections are routed to Qwen2.5-VL-3B, which describes actions, clothing, held objects and visibility. Everything else stays YOLO-only, so the VLM runs only where it is needed.
- **Re-identify across uploads.** Each person crop gets a 512-d OSNet Re-ID embedding, matched by cosine similarity against running-mean cluster centroids stored in pgvector. Results are `matched` (>= 0.75), `new` (< 0.60) or `pending` review in between. Clusters can be named, put on a watchlist, merged or reassigned.
- **Hybrid search.** Full-text search over VLM descriptions (Postgres `tsvector`) combined with image-embedding similarity (pgvector HNSW), using adjustable text/vector weights. You can also look up a person by photo via Re-ID.
- **Grounded follow-up Q&A.** Ask questions about a detected person. By default the VLM sees only the person crop, with a prompt that tells it to answer only from visible evidence and to say when it is unsure. A full-scene option answers from the whole image when the question needs context.

![UI](https://github.com/user-attachments/assets/f7a1219f-46c3-4930-b70b-e4ac124cf136)

## Architecture

```
image ──> YOLO11-L ──> routing ─┬─ non-person / low-conf ──> stored as YOLO-only
                                └─ person (conf >= 0.3) ──┬─> Qwen2.5-VL ──> description + 1536-d embedding
                                                          └─> OSNet ──> 512-d Re-ID ──> cosine vs centroids
                                                                                        (matched / new / pending)
                                        │
                                        v
             PostgreSQL + pgvector: sessions, entities, person clusters
             (GIN full-text index, HNSW vector indexes)
                                        │
                                        v
             FastAPI (REST + SSE streaming)  <──>  React + TypeScript UI
```

| File | Role |
|------|------|
| `pipeline.py` | YOLO detection, routing, batched VLM inference, Re-ID |
| `app.py` | FastAPI server: analyze, stream, Q&A, search, person clusters |
| `database.py` | Postgres/pgvector schema and hybrid search |
| `embeddings.py`, `reid.py`, `person_matcher.py` | VLM embeddings, OSNet extractor, match thresholds |
| `frontend/` | React 19 + Vite + Tailwind UI |

## Quickstart

### Docker Compose

```bash
git clone https://github.com/dianhaoli/openvocabperson.git
cd openvocabperson
docker compose up -d        # postgres (pgvector), backend, frontend
```

- Frontend: http://localhost:3000
- API: http://localhost:8000 (`GET /health`)

Models download on first start. For NVIDIA GPUs, uncomment the `deploy` block in `docker-compose.yml` (requires the NVIDIA Container Toolkit).

### Local

Requires Python 3.10+, Node 18+, PostgreSQL with the pgvector extension, and a CUDA GPU or Apple Silicon (CPU works but is slow).

```bash
# Backend
pip install -r requirements.txt
createdb vision_analysis
psql vision_analysis -c "CREATE EXTENSION vector;"
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/vision_analysis"  # this is the default
uvicorn app:app --host 0.0.0.0 --port 8000

# Frontend (http://localhost:5173, proxies API calls to :8000)
cd frontend
npm install
npm run dev
```

Try it from the command line:

```bash
curl -X POST -F "file=@image.jpg" http://localhost:8000/analyze
python demo.py path/to/image.jpg   # runs the pipeline without the server
```

## Tech stack

- **Models:** Ultralytics YOLO11-L, Qwen2.5-VL-3B-Instruct (Transformers, optional INT8/INT4 via bitsandbytes), OSNet x1.0 (torchreid)
- **Backend:** Python, PyTorch, FastAPI, uvicorn, asyncpg
- **Storage/search:** PostgreSQL 15 + pgvector (HNSW), Postgres full-text search (GIN)
- **Frontend:** React 19, TypeScript, Vite, Tailwind CSS v4
- **Deployment:** Docker Compose (Postgres, backend, nginx-served frontend)

## Limitations

Identity matching is cross-image body Re-ID only. There is no video tracking, face recognition or multi-camera calibration, and clothing changes reduce match reliability.

## More

API reference, configuration options, database schema and troubleshooting are in [docs/DETAILS.md](docs/DETAILS.md).

## License

MIT
