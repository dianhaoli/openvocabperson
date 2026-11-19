# OpenVocab Vision

Find anything in your photos and videos by describing it. Type a query such as "dangerous things", "people holding phones" or "dogs, bicycles". An LLM turns it into detector prompts. An open-vocabulary detector (YOLOE by default) finds and segments the matches, and a vision-language model checks each candidate and describes it. Every person sighting is linked to an identity cluster across uploads. You can then search the library by text or by photo, or ask an analyst agent about it.

It runs on a laptop CPU, an Apple Silicon Mac or a CUDA GPU, with local models, cloud APIs (OpenAI, Anthropic, Gemini, OpenRouter, Ollama, any OpenAI-compatible server) or a mix of the two.

![Demo](https://github.com/user-attachments/assets/81d30065-eaf9-461d-bddb-ee7183e1b14b)

**[Demo video](https://www.youtube.com/watch?v=MsrKKbLeFY0)**

## Quickstart

### Docker (one container for the API and UI, plus PostgreSQL)

```bash
git clone https://github.com/dianhaoli/openvocabperson.git && cd openvocabperson
cp .env.example .env          # optional: add OPENAI_API_KEY (or another provider)
docker compose up -d --build  # → http://localhost:8000
```

- NVIDIA GPU (CUDA torch and local Qwen2.5-VL): `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build`
- Model weights download on first use into the `weights` and `model-cache` volumes. Media and crops go to `storage`.
- Optional nginx front end on :3000: `docker compose --profile nginx up -d`.

### Local

You need Python 3.10+, PostgreSQL with [pgvector](https://github.com/pgvector/pgvector), and Node 20.19+ or 22.12+ (only to build the UI; Vite 7).

```bash
pip install -r requirements.txt          # + requirements-local.txt for local Qwen-VL / HF detectors
createdb vision_analysis && psql vision_analysis -c "CREATE EXTENSION vector"
cp .env.example .env                     # set DATABASE_URL and a provider key
(cd frontend && npm ci && npm run build) # the backend serves frontend/dist at /
python -m ovp serve --port 8000          # same as: uvicorn app:app
```

Open http://localhost:8000. API docs are at `/docs`. For UI development, run `cd frontend && npm run dev` (port 5173, proxies `/api` to :8000).

### Command line

The CLI uses the same database, storage and models as the server, in process and without HTTP:

```bash
python -m ovp analyze photos/ clip.mp4 --query "people with bags" --segment --describe all
python -m ovp analyze bus.jpg --json > result.json   # same JSON shapes as the HTTP API
python -m ovp search "man in a red jacket" --limit 10
python -m ovp search --image suspect.jpg             # CLIP image similarity + person Re-ID
python -m ovp models                                 # providers, availability, defaults (--check loads them)
python -m ovp reembed                                # move person sightings of older Re-ID spaces to the active model
python -m ovp serve --port 8000 --reload
```

Results go to stdout. Progress and logs go to stderr, so `--json` output can be piped. Run `python -m ovp <command> --help` for every flag. Analyzed media show up in the UI library.

![UI](https://github.com/user-attachments/assets/f7a1219f-46c3-4930-b70b-e4ac124cf136)

## What it does

- **Open-vocabulary analysis.** Queries can be literal ("person, dog"), abstract ("dangerous things", "all living things") or attribute-based ("menacing people"). The planner expands abstract concepts into prompts with the LLM, or with a built-in lexicon when no LLM is configured. It also attaches VLM yes/no verification questions to concepts that a detector cannot decide alone. With no query, the detector uses the 80 COCO classes.
- **Detect, segment, verify, describe.** You get boxes and masks (YOLOE-seg masks, or SAM 2.1 / MobileSAM / FastSAM). The VLM sends one JSON reply per crop that both verifies concepts and describes the subject (actions, clothing, held objects for people; colour and details for objects).
- **Images and videos.** Video frames are sampled (2 fps by default) and tracked with ByteTrack, whatever the detector (YOLO, YOLOE, YOLO-World, OWLv2, Grounding DINO or VLM boxes). Each track becomes one entity with a trajectory, a best crop and several Re-ID crops. Track fragments of the same person are stitched back together.
- **Identity across uploads.** Each person sighting gets a Re-ID embedding (OSNet, or CLIP or a colour histogram as fallback) and is matched against cluster centroids in pgvector. The result is `matched`, `new` or `pending` review. You can name clusters, put them on a watchlist, merge them, reassign sightings or re-cluster everything.
- **Hybrid search.** Several signals are fused with reciprocal-rank fusion: Postgres full-text over labels and descriptions, label/lexicon match, text-embedding similarity of descriptions, CLIP text-to-image and image-to-image, and person Re-ID for photo queries.
- **Grounded Q&A.** You can ask about one detection (crop or full scene) or a whole image or video. Answers are stored with the entity.
- **Analyst agent.** A tool-calling LLM can search the library, run open-vocabulary scans over stored media, analyze server-side folders, and manage people. Answers stream over SSE with tool cards.
- **Jobs.** All analysis runs as background jobs with live SSE progress. Closing the browser tab does not stop a job, and jobs can be cancelled.

## Architecture

```
                 query ──► planner (LLM or lexicon) ──► detector prompts + verification questions
                                       │
image ─────────────────────────────────┤
video ─► frame sampler ─► detector + ByteTrack ─► tracks ─► stitching
                                       ▼
                    open-vocab detection (+ masks) ─► cross-prompt NMS ─► routing
                                       │
          ┌──────────────────┬─────────┴────────┬──────────────────┐   (run concurrently)
          ▼                  ▼                  ▼                  ▼
   VLM verify+describe   segmentation      CLIP crop          person Re-ID
   (JSON per crop,       (SAM box prompts  embeddings         embeddings
    batched/concurrent)   or detector masks)
          └──────────────────┴─────────┬────────┴──────────────────┘
                                       ▼
              text embeddings of descriptions ─► one DB transaction ─► identity linking
                                       ▼
   PostgreSQL + pgvector: media · entities (tsvector) · embeddings (one HNSW index per space)
                          persons (centroids) · Q&A
                                       ▼
   FastAPI ─ jobs (SSE progress) ─ search (RRF) ─ persons/review ─ agent (tool calling, SSE)
      ▲                                                                   │
      └──── React UI (Analyze · Library · Search · People · Agent) ───────┘   + python -m ovp CLI
```

Local model inference runs on one dedicated worker thread per device, so the event loop never blocks and GPU work is serialized. Cloud calls run concurrently with semaphores. Default models load in the background at startup, so the library and search are usable immediately.

| Module | Role |
|---|---|
| `ovp/api/app.py` | FastAPI routes ([contract](docs/API.md)) and serving of the built UI |
| `ovp/cli.py` | `python -m ovp serve \| analyze \| search \| reembed \| models` |
| `ovp/context.py` | Wires storage, models and services (one per process) |
| `ovp/config.py` | Settings from environment variables / `.env` |
| `ovp/runtime.py` | Device selection, device thread, CPU pool |
| `ovp/jobs.py` | Background jobs, event replay, cancellation |
| `ovp/pipeline/planner.py`, `analyzer.py` | Query planning; image and video pipeline; Q&A |
| `ovp/models/` | Detectors, segmenters, chat models (VLM/LLM), local Qwen-VL, embedders and Re-ID, and the registry that resolves `provider:model` specs |
| `ovp/services/identity.py`, `search.py` | Identity clustering and review; hybrid search |
| `ovp/storage/db.py`, `files.py` | PostgreSQL schema and queries; originals, thumbnails, crops |
| `ovp/agent/agent.py` | Analyst agent and its tools |
| `frontend/` | React 19, TypeScript, Vite, Tailwind v4 |

More detail is in [docs/DETAILS.md](docs/DETAILS.md): the pipeline, identity, search, the agent, the schema, operations and troubleshooting.

## Models

Every role takes a spec of the form `provider:model`. Set it in `.env` (`OVP_DETECTOR=...`), change it at runtime in the UI's Models dialog (`PATCH /api/config`), or override it per request (`detector`, `segmenter`, `vlm`, `llm` in the analyze options). `auto` picks the best option for the hardware and keys it finds, and `none` disables a role.

| Role | Local | Cloud / server | `auto` picks |
|---|---|---|---|
| Detector | `yoloe:yoloe-11{s,m,l}-seg.pt` (open-vocab and masks), `yolo-world:yolov8l-worldv2.pt`, `yolo:yolo11l.pt` / `yolo:yolo11l-seg.pt` (COCO only), `owlv2:google/owlv2-base-patch16-ensemble`, `grounding-dino:IDEA-Research/grounding-dino-tiny` | `vlm:<vlm spec>` (boxes from a VLM, e.g. `vlm:gemini:gemini-2.5-flash`; use Gemini or Qwen-VL, OpenAI models give coarse boxes) | YOLOE-11L on CUDA, otherwise YOLOE-11M. YOLO11-L without the CLIP package. A VLM without ultralytics |
| Segmenter | `sam:sam2.1_{t,b,l}.pt`, `sam:mobile_sam.pt`, `sam:sam_b.pt`, `fastsam:FastSAM-s.pt`, `detector` (YOLOE-seg masks, free) | – | Detector masks when available, then `sam:sam2.1_b.pt` on CUDA, otherwise `sam:mobile_sam.pt` |
| VLM | `qwen:Qwen/Qwen2.5-VL-3B-Instruct` (or `-7B`; `hf:` works too), `ollama:qwen2.5vl:7b` | `openai:gpt-4.1-mini`, `anthropic:claude-haiku-5-5`, `gemini:gemini-2.5-flash`, `openrouter:<model>`, `compat:<model>` | Local Qwen on a GPU, otherwise the first key found: OpenAI, Anthropic, Gemini, then compat |
| LLM (planner, agent) | `ollama:qwen3:8b` | `openai:gpt-4.1-mini`, `anthropic:claude-sonnet-5-5`, `gemini:gemini-2.5-flash`, `openrouter:<model>`, `compat:<model>` | First key found: OpenAI, Anthropic, Gemini, then compat. The agent needs tool calling |
| Image embedder | `clip:ViT-B-32`, `clip:ViT-B-16`, `clip:ViT-L-14` (open_clip, falling back to the OpenAI CLIP package) | – | `clip:ViT-B-32` |
| Text embedder | `st:all-MiniLM-L6-v2`, `ollama:nomic-embed-text`, `clip` (CLIP text tower) | `openai:text-embedding-3-small`, `gemini:gemini-embedding-001`, `compat:<model>` | OpenAI, then Gemini, then sentence-transformers |
| Re-ID | `osnet:osnet_x1_0` (torchreid), `clip`, `color` (histogram fallback) | – | OSNet. If its weights cannot load: CLIP, then colour |

**Per-request specs** are validated before anything is stored: an unknown provider, a malformed name or a disallowed weights path returns HTTP 400 (the analyze endpoints, `POST /api/scan`, the agent's `llm`). Local weights must lie in the working directory or `OVP_WEIGHTS_DIR`, never in `OVP_STORAGE_DIR` (uploaded media live there), and weight files need one of the suffixes `.pt`, `.pth`, `.ts`, `.onnx`, `.engine`, `.safetensors`, `.bin`; `..` is refused. Models loaded for a request stay cached next to the defaults, at most 4 of them, and the least recently used one is unloaded first.

**Re-ID spaces.** Re-ID vectors are only compared within one embedding space: `reid:osnet:<model>`, `reid:clip:<model>` (for example `reid:clip:ViT-B-32`) or `reid:color:v2`. Switching the Re-ID model, or upgrading from a version that stored the older `reid:clip` / `reid:color` spaces, would leave the known people unmatched. The server therefore re-embeds such sightings from their stored crops in the background after the models are loaded, then moves the persons and recomputes their centroids in the active space. `python -m ovp reembed` does the same on demand. Old vectors are kept, so switching back needs no re-embedding.

Run `python -m ovp models` (or `GET /api/models`) to see what is available on your machine and why the rest is not. On CPU, use `OVP_DETECTOR=yoloe:yoloe-11s-seg.pt` and a cloud VLM. SAM 2.1-B and local Qwen-VL are slow there.

## Configuration

All settings are environment variables. A `.env` file in the working directory is read automatically; see [.env.example](.env.example).

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/vision_analysis` | PostgreSQL with pgvector. The schema is created and migrated on start |
| `OVP_STORAGE_DIR` | `storage` | Originals, thumbnails, crops, upload spool |
| `OVP_WEIGHTS_DIR` | `.` | Where local weight files are looked up, besides the working directory (ultralytics downloads into the working directory). Per-request specs may only point into these two folders |
| `OVP_DEVICE` | `auto` | `auto` (CUDA, then MPS, then CPU), `cpu`, `cuda`, `cuda:1`, `mps` |
| `OVP_DETECTOR` / `OVP_SEGMENTER` / `OVP_VLM` / `OVP_LLM` | `auto` | Model specs (see [Models](#models)) |
| `OVP_EMBEDDER` / `OVP_TEXT_EMBEDDER` / `OVP_REID` | `auto` | CLIP image embedder, text embedder for descriptions, Re-ID model |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL` | –, `https://api.openai.com/v1` | OpenAI (or any proxy that speaks its API) |
| `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL` | –, `https://api.anthropic.com` | Anthropic |
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`), `GEMINI_BASE_URL` | –, `https://generativelanguage.googleapis.com/v1beta/openai` | Google Gemini (OpenAI-compatible endpoint) |
| `OPENROUTER_API_KEY` | – | OpenRouter (`openrouter:<model>`) |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama (`ollama:<model>`) |
| `OVP_COMPAT_BASE_URL`, `OVP_COMPAT_API_KEY` | – | Any OpenAI-compatible server: vLLM, LM Studio, TGI, llama.cpp (`compat:<model>`) |
| `OVP_OPENAI_VLM_MODEL` / `OVP_OPENAI_LLM_MODEL` | `gpt-4.1-mini` | Model used by `auto` and by `openai:` without a model |
| `OVP_ANTHROPIC_VLM_MODEL` / `OVP_ANTHROPIC_LLM_MODEL` | `claude-haiku-5-5` / `claude-sonnet-5-5` | Same for Anthropic |
| `OVP_GEMINI_MODEL` | `gemini-2.5-flash` | Same for Gemini |
| `OVP_LOCAL_VLM_MODEL` | `Qwen/Qwen2.5-VL-3B-Instruct` | Local VLM for `qwen:` / `hf:` |
| `OVP_CONF` | `0.25` | Upper bound of the automatic detector threshold: without `conf` in the options, default-vocabulary runs use min(`OVP_CONF`, 0.2) and text queries min(`OVP_CONF`, 0.15) (VLM verification filters the extra candidates). A `conf` given per request is used exactly |
| `OVP_LOW_CONF` | `0.3` | Persons below this score are kept as `low_confidence`, with no description and no Re-ID |
| `OVP_NMS_IOU` | `0.7` | IoU for cross-prompt NMS |
| `OVP_EXPAND_RATIO` | `0.1` | Crop padding around boxes |
| `OVP_MAX_DETECTIONS` | `100` | Detections per image |
| `OVP_VLM_CONCURRENCY` | `8` | Parallel requests per cloud model |
| `OVP_VLM_BATCH` | `4` | Batch size for the local VLM |
| `OVP_VLM_MAX_TOKENS` | `260` | Max tokens per crop reply |
| `OVP_MAX_VLM_CROPS` | `40` | Max crops per media item sent to the VLM (largest and most confident first) |
| `OVP_VLM_QUANT` | `none` | `8bit` / `4bit` for the local VLM (bitsandbytes, CUDA) |
| `OVP_REQUEST_TIMEOUT` | `90` | Seconds per cloud request |
| `OVP_VIDEO_FPS` | `2.0` | Frames per second sampled from videos |
| `OVP_MAX_FRAMES` | `600` | Max sampled frames per video |
| `OVP_MIN_TRACK_HITS` | `2` | Tracks seen in fewer sampled frames are dropped |
| `OVP_JOB_CONCURRENCY` | `2` | Jobs running at once |
| `OVP_MEDIA_CONCURRENCY` | `3` | Media items processed at once within a job |
| `OVP_REID_MATCH` / `OVP_REID_REVIEW` | per backend | Identity thresholds. Defaults: OSNet 0.75/0.60, CLIP 0.88/0.80, colour 0.87/0.78 |
| `OVP_IMPORT_ROOTS` | – (disabled) | Comma-separated server folders that `POST /api/analyze/paths`, the UI's "Analyze server paths" and the agent may read. Unset → those return 403. The CLI is not restricted |
| `OVP_CORS_ORIGINS` | `http://localhost:5173,http://localhost:3000` | Comma-separated browser origins allowed to call the API cross-origin; `*` allows any. The UI served by the API itself needs no entry |
| `OVP_MAX_UPLOAD_MB` | `2048` | Max request body for uploads (all files of one request together) → 413 |
| `OVP_PRELOAD` | `true` | Load default models in the background at startup |
| `OVP_FRONTEND_DIST` | `frontend/dist` | Built UI served at `/` |

## Limitations

- **Identity is appearance-based body Re-ID.** There is no face recognition. A change of clothes, heavy occlusion or very small people reduce match quality, and the colour-histogram fallback is weak. Uncertain matches go to the review queue instead of being merged automatically.
- **Verification and descriptions are only as good as the VLM.** Small or blurry crops and abstract concepts ("menacing") can be wrong. Each answer comes with the stored reason, so you can audit it.
- **CPU works, but slowly.** On CPU, use YOLOE-11S, MobileSAM and a cloud VLM. Local Qwen-VL and SAM 2.1-L need a GPU. Long videos are capped by `OVP_MAX_FRAMES`.
- **Jobs and agent conversations live in memory.** They are lost on restart. Analysis results themselves are stored in PostgreSQL.
- **There is no authentication.** Anyone who can reach the port can read and change the library, and `/api/analyze/paths` (also used by the agent) reads any media file under `OVP_IMPORT_ROOTS`. Run the server only on a trusted network or behind an authenticating proxy.
- **Weights download on first use.** That includes ultralytics assets from GitHub, OSNet from Google Drive, and CLIP / Qwen / OWLv2 from Hugging Face. Offline machines need the weights placed in `OVP_WEIGHTS_DIR` or the caches in advance. When a download fails, the model is reported as unavailable instead of crashing the server.

## License

MIT
