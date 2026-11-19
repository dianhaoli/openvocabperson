# Details

This file covers how each part works and how to run it. The HTTP contract is in [API.md](API.md). Setup, the model matrix and every environment variable are in the [README](../README.md).

- [Query planning](#query-planning)
- [Image pipeline](#image-pipeline)
- [Video pipeline](#video-pipeline)
- [Identity (Re-ID clusters)](#identity-re-id-clusters)
- [Search](#search)
- [Q&A](#qa)
- [Jobs](#jobs)
- [Analyst agent](#analyst-agent)
- [Model registry and runtime](#model-registry-and-runtime)
- [Database schema](#database-schema)
- [CLI](#cli)
- [Docker](#docker)
- [Development and tests](#development-and-tests)
- [Troubleshooting](#troubleshooting)

## Query planning

`ovp/pipeline/planner.py` turns the free-text query into a `Plan`, a list of concepts. Each concept has detector prompts, and optionally a VLM yes/no question for verification.

| Query | Source | Result |
|---|---|---|
| *(empty)* | `default` | The 80 COCO classes. The detector runs its own vocabulary, with confidence capped at 0.2 |
| `people, dogs` | `literal` / `lexicon` | `person`, `dog` (COCO synonyms are normalised) |
| `dangerous things` | `llm` or `lexicon` | knife, gun, scissors, fire, … plus the check "Is this a potentially dangerous object…?" |
| `menacing people` | `llm` or `lexicon` | Detect `person`, then ask the VLM "Does this person appear menacing?" (a `person_attribute` concept) |

`expand` controls when the LLM is used:

- `"auto"` (the default) asks the LLM only for abstract terms.
- `true` always asks the LLM.
- `false` takes the query literally.

The LLM is the configured LLM, or the VLM when no LLM is set. Plans are cached per (query, model). If the LLM fails or none is configured, the built-in lexicon is used, so planning also works offline. `POST /api/plan` previews a plan without running detection.

## Image pipeline

`Analyzer.analyze_image` in `ovp/pipeline/analyzer.py`:

1. **Store.** The original (EXIF-rotated, saved as JPEG) and a thumbnail are written, and a `media` row is created with status `processing`.
2. **Detect.** The detector runs with the plan's prompts. Closed-set detectors (YOLO11) only receive the prompts that are COCO classes. If the query needs open vocabulary and the default detector is closed-set, the analyzer switches automatically to an open-vocabulary one: an already loaded one first, then YOLOE, YOLO-World, and finally VLM grounding (`vlm:<vlm spec>`: its boxes are only as precise as the VLM; Gemini and Qwen-VL are trained for grounding, OpenAI models give coarse boxes). This does not happen when a detector was requested explicitly. Cross-prompt NMS (`OVP_NMS_IOU`) then removes duplicate boxes from overlapping prompts. Person boxes are protected from being suppressed by object boxes, so a person holding a bag keeps both.
3. **Route.** Each detection becomes a candidate crop, padded by `OVP_EXPAND_RATIO`.
   - Concepts with `verify` get VLM questions.
   - `describe` selects which crops get described: `persons`, `all` or `none`.
   - Persons below `OVP_LOW_CONF` are kept as `low_confidence`, with no description and no Re-ID.
   - Crops smaller than 20×20 px are not sent to the VLM.
   - At most `OVP_MAX_VLM_CROPS` crops go to the VLM, ranked with verification first, then by confidence × √area.
4. **Run concurrently.**
   - VLM: one JSON request per crop that answers the questions (`checks`) and fills the description fields (summary, actions, clothing, held objects, appearance for people; summary, colour, details for objects). If every verification answers "no", the candidate becomes `rejected`. It stays visible, but is excluded from identity and from counts.
   - Segmentation: one SAM call per image with box prompts, or masks taken from YOLOE-seg. This only runs when `segment=true`.
   - CLIP embeddings of all crops.
   - Re-ID embeddings of person crops.
5. **Embed descriptions** with the text embedder, as `label. summary key: value. …`.
6. **Persist** crops, entities and all embeddings in one transaction.
7. **Link identities** (see below), then emit an `entity` event per entity.

Entity `stage` values:

- `detected`: not described
- `described`
- `low_confidence`
- `rejected`: failed verification

## Video pipeline

`Analyzer.analyze_video`:

- **Sampling.** Frames are decoded on a background reader thread at `min(native fps, OVP_VIDEO_FPS)`, up to `OVP_MAX_FRAMES` frames. Longer videos are sampled more sparsely.
- **Tracking.** The detector tracks with ByteTrack across the sampled frames. This needs a YOLO, YOLOE or YOLO-World detector; VLM and HF detectors cannot track. Each track keeps:
  - the majority label
  - a thinned trajectory (`[t, x1, y1, x2, y2]`, at most 240 points)
  - the best-quality crop and its frame index
  - up to 8 Re-ID crops
- **Filtering.** Tracks with fewer than `OVP_MIN_TRACK_HITS` hits are dropped.
- **Stitching.** Person tracks that are never visible at the same time and whose mean Re-ID embeddings are similar are merged. The threshold is the match threshold + 0.03, because this merge is not reviewable. This reconnects a person who leaves the frame and comes back.
- **Processing.** The tracks then go through the same crop processing as images: one entity per track, with `t_start`, `t_end`, `frame_index` and `trajectory`.
- **Cancellation.** It is checked inside the tracking loop.

A library scan (`POST /api/scan`, the agent's `find_in_library`) re-runs detection with a new query over stored media. It only adds entities that do not overlap existing ones (same label, IoU > 0.7, or the same time span for tracks).

## Identity (Re-ID clusters)

`ovp/services/identity.py`. Each person cluster stores a centroid (mean Re-ID embedding) in its Re-ID space, such as `reid:osnet:osnet_x1_0`. pgvector computes it with `AVG` and recomputes it whenever membership changes, including on deletes and merges.

**Linking one media item.** All sightings of one media item are matched together, under a process-wide lock so that concurrent uploads cannot create duplicate clusters:

1. The k = 8 nearest centroids are fetched for every sighting, and all (score, sighting, person) pairs are sorted best first.
2. Pairs are assigned greedily when `score ≥ match`. Two sightings that *conflict* can never go to the same person. In an image, every pair of sightings conflicts; in a video, only tracks that overlap in time do.
3. Any sighting left over whose best free candidate is `≥ review` becomes `pending`, with a `suggested_person`. Otherwise a new cluster is created and the sighting is marked `new`.

**Thresholds (match / review).** Each embedding space has its own:

| Backend | Match | Review |
|---|---|---|
| OSNet | 0.75 | 0.60 |
| CLIP | 0.88 | 0.80 |
| Colour histogram | 0.93 | 0.86 |

`OVP_REID_MATCH` and `OVP_REID_REVIEW` override them.

**Review.** `POST /api/entities/{id}/review` takes `accept` (link to the suggestion), `new` (create a cluster) or `reject` (unlink). `assign` sets the person by hand (`manual`).

**Re-cluster.** `POST /api/persons/recluster` runs centroid-linkage agglomerative clustering over all sightings, with a cannot-link constraint inside the same media. Above 4000 sightings it switches to single-pass leader clustering. Named, watch-listed and manually confirmed clusters are anchors: they keep their sightings and absorb similar clusters. Existing cluster ids are reused where possible.

**Photo lookup.** `POST /api/persons/search` embeds the largest detected person in the photo and returns the nearest clusters.

## Search

`ovp/services/search.py` runs every available signal concurrently and fuses them with weighted reciprocal-rank fusion (k = 60). Signals with incomparable scales therefore combine sensibly:

| Signal | Source | Weight | Min. similarity |
|---|---|---|---|
| `fts` | Postgres full text. Label and concept are weight A, the description is weight B | 1.0 | – |
| `label` | Label/concept match with offline lexicon expansion ("weapons" → knife, gun, …) | 1.2 | – |
| `semantic` | Text embedding of the query vs description embeddings | 1.0 | 0.2 |
| `clip` | CLIP text → crop image embeddings | 0.8 | 0.18 |
| `image` | CLIP image → image (photo query) | 1.0 | 0.5 |
| `reid` | Person Re-ID (photo query) | 1.2 | 0.3 |

Each embedding space has its own HNSW cosine index, a partial index per `(space, dimension)` created on demand. You can filter by `labels`, `media_id`, `person_id` and `kind`. Each result includes the raw per-signal scores (`scores`), so a ranking can be explained.

## Q&A

- `POST /api/entities/{id}/ask` with `scope: "crop"` answers from the crop only, under a grounded prompt: answer from visible evidence and say when unsure.
- `scope: "scene"` uses the full image, or the entity's video frame, with the entity outlined in red.
- `POST /api/media/{id}/ask` asks about a whole image, or about six evenly spaced, timestamped keyframes of a video.

All answers are stored in the `qa` table.

## Jobs

`ovp/jobs.py`:

- **Every analysis is a job**: uploads, server paths, library scans, CLI runs and agent tool calls. Upload handlers spool files to `OVP_STORAGE_DIR/tmp` and return at once, and the job runs on its own. Use `?wait=true` to block until it finishes.
- **Concurrency.** Up to `OVP_JOB_CONCURRENCY` jobs run at once, and up to `OVP_MEDIA_CONCURRENCY` media items per job. Local models are still serialized on the device thread, while cloud VLM calls overlap.
- **Events.** They are kept per job, up to 5 000 (the final `job_done` always). A late SSE subscriber (`GET /api/jobs/{id}/events`) gets a replay first. That is how the UI reattaches to running jobs after a reload. Finished jobs beyond the newest 20 keep only `job_done`, and a subscriber that falls 2 000 events behind is disconnected (it can reconnect and replay).
- **Cancellation** is cooperative: it is checked between stages and between video frames. The media being processed is stored with status `cancelled` and hidden from the library; items that did not finish are not counted in `progress.done`.
- **Summary.** A finished analyze job reports `{media, entities, failed}`; a scan reports `{scanned, media_with_hits, accepted, rejected, updated}` (`accepted`: entities that match the query, newly stored or already stored; `rejected`: candidates the VLM verification denied; `updated`: stored entities whose concept or verification the scan changed).
- **Memory.** Jobs live in memory only and are lost on restart; the newest 100 finished jobs are kept. Their results (media and entities) are in PostgreSQL. On startup, media left in `processing` by a previous run are marked `error`, and the upload spool is emptied. On shutdown, running jobs are cancelled and awaited (up to 10 s) before the database closes.
- **Server paths.** `POST /api/analyze/paths` and the agent's `analyze_paths` only read inside `OVP_IMPORT_ROOTS` (resolved, so symlinks cannot escape; symlinks inside a walked folder are skipped). Without it they return 403. The CLI reads any path.

## Analyst agent

`ovp/agent/agent.py`. A tool-calling LLM (`OVP_LLM`, or `llm` per request) answers in markdown, with up to 12 tool steps per turn.

**Tools:**

- `search`
- `find_in_library`: an open-vocabulary scan of stored media
- `plan_query`
- `list_media`, `get_media`
- `ask_image`
- `list_persons`, `get_person`, `name_person`, `set_watchlist`, `merge_persons`, `recluster_persons`, `find_person_sightings`
- `stats`
- `analyze_paths`: server-side files or folders
- `job_status`

**Results.** Tool results sent to the LLM are compact: no polygons or trajectories, at most 14 000 characters. The SSE `tool_result` event carries the full entities, media and persons for the UI's tool cards.

**Turns** run independently of the HTTP response: a client disconnect stops the stream, not the turn. Turns of one conversation are serialized. Conversations live in memory; the 200 most recent are kept.

## Model registry and runtime

**Specs.** `ovp/models/registry.py` resolves `provider:model` specs into instances. It builds each one lazily, caches it, and warms it up (`ensure`) under a per-model lock.

**Failures.** A model that fails to load, for example because its weights cannot be downloaded, is recorded with a short reason. It then shows up as unavailable in `GET /api/models` instead of failing requests. `PATCH /api/config` clears that error and retries.

**Switching defaults.** `PATCH /api/config` is all-or-nothing: every new default must load, otherwise the previous defaults stay.

**Startup warm-up.** The server loads the detector, image embedder, Re-ID and text embedder in the background, and `GET /api/health` reports `ready`. The CLI does the same warm-up before it runs. If Re-ID fails, it falls back from OSNet to CLIP to the colour histogram.

**Runtime** (`ovp/runtime.py`):

- `OVP_DEVICE=auto` picks CUDA, then MPS, then CPU.
- All local inference runs on a single device thread, which serializes GPU work and never blocks the event loop.
- Image encode/decode runs on a small CPU pool.
- Cloud chat models use `httpx` with `OVP_VLM_CONCURRENCY` parallel requests per model and a timeout of `OVP_REQUEST_TIMEOUT`.

Provider notes:

- **OpenAI, Gemini, OpenRouter, Ollama and `compat`** all use the OpenAI-compatible Chat Completions API. Gemini goes through its `/v1beta/openai` endpoint.
- **Anthropic** uses the native Messages API.
- **Local Qwen-VL** (`qwen:` / `hf:`) uses transformers. It batches `OVP_VLM_BATCH` crops, supports bitsandbytes 8/4-bit (`OVP_VLM_QUANT`), and uses flash-attention when it is installed. It does not support tool calling, so it cannot drive the agent.

## Database schema

`ovp/storage/db.py` creates and migrates the schema on start, under an advisory lock. An old-layout database (`sessions` and `entities.initial_analysis`) is renamed to `legacy_*` and copied into the new schema.

| Table | Contents |
|---|---|
| `media` | `id`, `kind` (image/video), `name`, `path`, `thumb_path`, size, `duration`/`fps`/`frame_count`, `query`, `plan` and `options` (JSONB), `status`, `created_at` |
| `entities` | One detection or track: `label`, `concept`, `score`, box `x1..y2`, `polygon`, `mask_area`, `crop_path`, `description`, `attributes`, `verified`/`verify_reason`, `stage`, video fields (`track_id`, `t_start`, `t_end`, `frame_index`, `trajectory`), identity (`person_id`, `match_status`, `match_score`, `suggested_person_id`), and a generated `tsv` (GIN index) |
| `embeddings` | `(entity_id, space) → vec`. Spaces are model-specific, e.g. `clip:ViT-B-32` or `reid:osnet:osnet_x1_0`. Each (space, dimension) pair has a partial HNSW index |
| `persons` | `label`, `is_watchlist`, `notes`, `space`, `centroid`, `sighting_count`, `representative_entity_id`, timestamps |
| `qa` | Questions and answers per media or entity, with `scope` (`crop`, `scene`, `media`) |
| `ovp_meta` | `schema_version` |

Files live under `OVP_STORAGE_DIR`: `media/` for originals, `thumbs/`, `crops/`, and `tmp/` for the upload spool.

## CLI

`python -m ovp` (`ovp/cli.py`) creates the same `AppContext` as the server: the same database, storage, registry and job manager. Analyses therefore appear in the UI, and searches see everything the UI analyzed.

| Command | What it does |
|---|---|
| `serve [--host --port --reload --log-level]` | Runs uvicorn with `ovp.api.app:app` |
| `analyze PATH... [--query Q] [--describe persons\|all\|none] [--segment] [--expand/--verify auto\|true\|false] [--detector/--segmenter/--vlm/--llm SPEC] [--conf] [--video-fps] [--max-frames] [--no-reid] [--quiet] [--json]` | Analyzes files and folders (folders recursively) as one job. Progress goes to stderr. The result is a table or `{job, media: MediaDetail[]}` |
| `search [TEXT] [--image FILE] [--limit] [--label L]... [--kind] [--media ID] [--person ID] [--json]` | Hybrid search. JSON output has the same shape as `POST /api/search` |
| `models [--check] [--json]` | Shows providers, availability and defaults. `--check` loads every default model. This command needs no database |

Everything that libraries print, including C-level output, is redirected to stderr, so stdout holds only the result. Exit codes:

- 0: success
- 1: the job failed
- 2: usage error, missing path, or the database is unreachable
- 130: interrupted

## Docker

The `Dockerfile` is multi-stage:

1. `node:22-alpine` builds the React UI.
2. `python:3.12-slim` installs CPU torch, then `requirements.txt`, including the ultralytics CLIP package from GitHub, and copies in the UI build. A single container serves the UI and the API on :8000.

Build arguments:

- `TORCH_INDEX_URL`: the torch wheel index. The default is `https://download.pytorch.org/whl/cpu`; use `…/cu124` for CUDA.
- `REQUIREMENTS`: `requirements.txt`, or `requirements-local.txt` to add Qwen-VL, OWLv2, Grounding DINO and sentence-transformers.

Volumes and paths:

- `/data/storage`: media, thumbnails, crops
- `/data/weights`: the working directory. Ultralytics downloads weights there, so they persist
- `/data/cache`: Hugging Face, torch hub (OSNet) and ultralytics config

`docker-compose.yml` services:

- **`postgres`** runs `pgvector/pgvector:pg16` with a health check. It is not published to the host.
- **`backend`** takes model specs and provider keys from the shell or `.env`. Everything in `.env` is also passed in.
- **`frontend`** is an optional nginx service under the `nginx` profile, on :3000. Its config proxies `/api` with SSE buffering disabled.

`docker-compose.gpu.yml` adds the NVIDIA device reservation and builds with CUDA torch and the local extras.

```bash
docker compose up -d --build                     # CPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
docker compose exec backend python -m ovp models # CLI inside the container
docker compose exec backend python -m ovp analyze /data/storage/inbox --query "forklifts"
docker compose down -v                           # also deletes the database and volumes
```

Ollama on the host is reachable as `http://host.docker.internal:11434/v1`, which is the default in compose.

## Development and tests

```bash
pip install -r requirements-dev.txt
createdb ovp_test && psql ovp_test -c "CREATE EXTENSION vector"
pytest                                     # OVP_TEST_DATABASE_URL overrides the test DB
python -m ovp serve --reload               # backend with auto-reload
cd frontend && npm run dev                 # UI on :5173, proxies /api to :8000
cd frontend && npm run lint && npm run build
```

The tests use fake models (`tests/fakes.py`) against a real PostgreSQL. Cloud providers are mocked, so no keys or weights are needed.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| A model shows "model weights could not be downloaded" | The host is offline or blocked. Put the weight file in `OVP_WEIGHTS_DIR` or the working directory, or fill the HF/torch caches, then reselect the model (`PATCH /api/config` retries it) |
| YOLOE / YOLO-World unavailable: "pip install git+https://github.com/ultralytics/CLIP.git" | The CLIP text encoder package is missing. Without it, `auto` falls back to closed-set YOLO11 |
| `vlm: false` in `/api/health` and no descriptions | No VLM is configured. Set a provider key, `OVP_COMPAT_BASE_URL`, or use `qwen:` on a GPU |
| Agent returns 503 | No LLM is configured. Set a key, or `OVP_LLM=compat:<model>` |
| "detector … cannot track video" | Videos need a YOLO, YOLOE or YOLO-World detector |
| Persons never matched across uploads / `reid: color` | OSNet could not load. `torchreid` also needs `scipy` and `tensorboard` to import (both in requirements.txt), and its weights come from Google Drive. Install `torchreid gdown scipy tensorboard` and make sure the download works, or accept the CLIP or colour fallback with its stricter thresholds |
| Too many `pending` reviews, or different people merged | Tune `OVP_REID_MATCH` / `OVP_REID_REVIEW`, then run `POST /api/persons/recluster` |
| Some crops not described | More than `OVP_MAX_VLM_CROPS` candidates in one media item. A `log` warning says how many were skipped |
| Slow on CPU | Use `OVP_DETECTOR=yoloe:yoloe-11s-seg.pt`, MobileSAM, a cloud VLM, and lower `OVP_VIDEO_FPS` |
| Out of GPU memory | Use a smaller detector or SAM, `OVP_VLM_QUANT=4bit`, `OVP_VLM_BATCH=1` |
| "Analyze server paths" or the agent's `analyze_paths` fails with 403 | Server-side imports are off. Set `OVP_IMPORT_ROOTS` to the folders the server may read, or upload the files instead |
| Upload fails with 413 | The request is larger than `OVP_MAX_UPLOAD_MB` (all files together). Raise it, or upload in smaller batches |
| The UI on another origin gets CORS errors | Add its origin to `OVP_CORS_ORIGINS` (the UI served by the API itself and the Vite dev proxy need no entry) |
| Library shows a media item as `processing` forever | The server stopped mid-analysis. The next start marks it `error` (hidden from the library); delete it or analyze the file again |
| UI shows "The React UI is not built" | Run `cd frontend && npm ci && npm run build`, or set `OVP_FRONTEND_DIST` |
| `cannot connect to PostgreSQL` | Check `DATABASE_URL`. The database also needs the `vector` extension, which the server creates if it has the rights to |
