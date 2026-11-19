# API reference

All endpoints are served by the FastAPI backend (`uvicorn app:app`). JSON in, JSON out unless noted.
Times are epoch seconds (float). Boxes are `[x1, y1, x2, y2]` in pixel coordinates of the original
image (or the video frame size for videos).

## Core objects

### Entity
A single detection (image) or a single track (video).

```jsonc
{
  "id": "e_3f9a1c2b7d",
  "media_id": "m_81c2aa90f1",
  "label": "knife",                 // detector label (the prompt that matched)
  "concept": "dangerous things",    // query concept that produced it (null for default vocab)
  "score": 0.83,                    // detector confidence
  "box": [12, 40, 220, 380],
  "polygon": [[[x, y], [x, y], ...]] , // segmentation polygons (null if not segmented)
  "mask_area": 10234,               // pixels, null if no mask
  "crop_url": "/api/entities/e_.../crop",
  "description": "Man in a grey hoodie holding a phone",  // VLM description, null if not described
  "attributes": {"actions": "...", "clothing": "...", "objects": "..."},  // structured VLM output or null
  "verified": true,                 // VLM verification result for abstract concepts (null = not verified)
  "verify_reason": "Blade clearly visible in right hand",
  "stage": "described",             // "described" | "detected" | "low_confidence" | "rejected"
  "track_id": 3,                    // video only
  "t_start": 1.5, "t_end": 7.0,     // video only, seconds
  "frame_index": 45,                // video only: frame of the representative crop
  "trajectory": [[t, x1, y1, x2, y2], ...], // video only: sampled boxes over time
  "person": {"id": "p_...", "label": "Alice", "is_watchlist": false},  // null if not linked
  "match_status": "matched",        // "matched" | "new" | "pending" | "manual" | null
  "match_score": 0.81,
  "suggested_person": {"id": "p_...", "label": null, "is_watchlist": false}, // for pending review
  "created_at": 1731900000.0
}
```

### Media
```jsonc
{
  "id": "m_81c2aa90f1",
  "kind": "image",                  // "image" | "video"
  "name": "IMG_0042.jpg",
  "width": 1920, "height": 1080,
  "duration": null, "fps": null,    // video only
  "url": "/api/media/m_.../file",   // original file (videos support Range requests)
  "thumb_url": "/api/media/m_.../thumb",
  "query": "dangerous things",      // query used at analysis time (null = default)
  "status": "done",                 // "processing" | "done" | "error"
  "entity_count": 7,
  "person_count": 3,
  "created_at": 1731900000.0
}
```
`GET /api/media/{id}` returns `Media` plus `"entities": Entity[]` and `"plan": Plan | null`.

### Plan
How a free-text query is turned into detector prompts.
```jsonc
{
  "query": "dangerous things, dogs",
  "expanded": true,                 // true when an LLM (or the built-in lexicon) expanded it
  "source": "llm",                  // "llm" | "lexicon" | "literal" | "default"
  "concepts": [
    {"name": "dangerous things", "kind": "object",
     "prompts": ["knife", "gun", "scissors", "fire", "baseball bat"],
     "verify": true, "question": "Is this a dangerous object such as a weapon, sharp or hot item?"},
    {"name": "menacing people", "kind": "person_attribute",
     "prompts": ["person"], "verify": true, "question": "Does this person appear threatening or menacing?"},
    {"name": "dogs", "kind": "object", "prompts": ["dog"], "verify": false, "question": null}
  ],
  "prompts": ["knife", "gun", "scissors", "fire", "baseball bat", "person", "dog"]
}
```

### Job
Analysis always runs as a background job.
```jsonc
{
  "id": "j_1a2b3c4d",
  "kind": "analyze",                // "analyze" | "scan"
  "status": "running",              // "queued" | "running" | "done" | "error" | "cancelled"
  "progress": {"done": 3, "total": 10, "stage": "describe", "message": "Describing 4 crops"},
  "media_ids": ["m_..."],
  "error": null,
  "created_at": 1731900000.0, "finished_at": null,
  "options": { /* AnalyzeOptions */ },
  "plan": { /* Plan */ },           // null until planning finished
  "summary": {"media": 1, "entities": 7, "failed": 0}  // scan: {scanned, media_with_hits, entities}
}
```

### AnalyzeOptions
All fields optional.
```jsonc
{
  "query": "dangerous things",      // free text, comma list, or null for default (COCO) vocabulary
  "expand": "auto",                 // true | false | "auto": LLM-expand abstract queries into prompts
  "verify": "auto",                 // true | false | "auto": VLM yes/no check on each candidate
  "segment": false,                 // run segmentation (SAM / detector masks)
  "describe": "persons",            // "persons" | "all" | "none": which detections get a VLM description
  "detector": null,                 // model spec override, e.g. "yoloe:yoloe-11l-seg.pt"
  "segmenter": null,                // e.g. "sam:sam2.1_b.pt"
  "vlm": null,                      // e.g. "openai:gpt-4.1-mini", "anthropic:claude-haiku-5-5", "qwen:Qwen/Qwen2.5-VL-3B-Instruct"
  "llm": null,                      // LLM used to expand the query (falls back to the VLM)
  "conf": 0.25,                     // detector confidence threshold
  "video_fps": 2.0,                 // frames per second sampled from videos
  "max_frames": 600,
  "reid": true                      // link persons to identity clusters
}
```

## Endpoints

### System
| Method | Path | Notes |
|---|---|---|
| GET | `/api/health` | `{status, ready, device, db, capabilities, version}`; `ready` turns true once default models finished loading |
| GET | `/health` | Legacy alias: the same plus `pipeline_ready`, `reid_ready` |
| GET | `/api/models` | Providers per role, availability and current defaults (see below) |
| PATCH | `/api/config` | `{detector?, segmenter?, vlm?, llm?, embedder?, text_embedder?, reid?}` change runtime defaults (`"auto"` re-resolves). All-or-nothing: if any model fails to load nothing changes → 400. Returns the `/api/models` payload |
| GET | `/api/stats` | `{media, images, videos, entities, persons, watchlist, pending, labels: [{label, count}]}` (`pending` = sightings awaiting review) |

`GET /api/models`:
```jsonc
{
  "device": "cuda",
  "defaults": {"detector": "yoloe:yoloe-11l-seg.pt", "segmenter": "sam:sam2.1_b.pt", "vlm": "openai:gpt-4.1-mini",
               "llm": "openai:gpt-4.1-mini", "embedder": "clip:ViT-B-32", "text_embedder": "openai:text-embedding-3-small",
               "reid": "osnet:osnet_x1_0"},
  "providers": {
    "detector":  [{"spec": "yoloe:yoloe-11l-seg.pt", "label": "YOLOE-11L (open-vocab + masks)", "available": true,
                   "local": true, "open_vocab": true, "masks": true, "note": null}, ...],
    "segmenter": [...], "vlm": [...], "llm": [...], "embedder": [...], "reid": [...]
  },
  "capabilities": {"detector": true, "open_vocab": true, "vlm": true, "llm": true, "agent": true,
                   "segmentation": true, "clip": false, "text_search": true, "reid": "osnet", "video": true}
}
```

### Planning & analysis
| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/api/plan` | `{query, expand?, llm?}` | `Plan` (preview, no detection) |
| POST | `/api/analyze` | multipart: `files` (1..n images/videos), `options` (JSON string of AnalyzeOptions), optional `query` form field (used when `options.query` is unset). Query `?wait=true` blocks and returns the finished job with `media` | `{job: Job}` or `{job, media: MediaDetail[]}` |
| POST | `/api/analyze/paths` | `{paths: [server-side files or folders, searched recursively], options}`; `?wait=true` as above | `{job}` |
| POST | `/api/scan` | `{query, media_ids?: string[], limit?: int, options?}` re-run open-vocab detection over already stored media (all if omitted) and add new entities; `describe` defaults to `"none"`; `?wait=true` as above | `{job}` |
| GET | `/api/jobs` | | `{jobs: Job[]}` (most recent first) |
| GET | `/api/jobs/{id}` | | `{job, media: MediaDetail[]}` |
| GET | `/api/jobs/{id}/events` | | **SSE** stream (replays past events first) |
| POST | `/api/jobs/{id}/cancel` | | `{job}` |

Job SSE events (`data: {...}\n\n`, each has `type`, `job_id`, `ts`):
- `job_started` `{job}`
- `plan` `{plan}`
- `media_started` `{index, total, name, kind}`
- `media_created` `{media}` (media row exists, file viewable)
- `stage` `{media_id, stage, message}` stage in `detect|track|segment|verify|describe|embed|identity|save`
- `detections` `{media_id, count, labels: {label: count}, elapsed}`
- `entity` `{entity}` (emitted as each entity is finalized)
- `media_done` `{media: MediaDetail}`
- `progress` `{done, total}`
- `log` `{level: info|warning|error, message}`
- `error` `{message}` the job failed (followed by `job_done`)
- `ping` `{}` keep-alive every 15 s (ignore)
- `job_done` `{job}` — final event (also sent for error/cancel)

### Media & entities
| Method | Path | Notes |
|---|---|---|
| GET | `/api/media?limit=50&offset=0&kind=image\|video` | `{media: Media[], total}` |
| GET | `/api/media/{id}` | `MediaDetail` |
| DELETE | `/api/media/{id}` | Deletes files, entities, and updates person clusters → `{success, media_id, persons_updated}` |
| GET | `/api/media/{id}/file` | Original image / video (Range supported) |
| GET | `/api/media/{id}/thumb` | JPEG thumbnail |
| POST | `/api/media/{id}/ask` | `{question, vlm?}` → `{answer, elapsed}` (VLM over the full image / video keyframes); persisted. 503 without a VLM |
| GET | `/api/media/{id}/qa` | `{qa: [{question, answer, created_at}]}` media-level Q&A history |
| GET | `/api/entities/{id}` | `Entity` + `media` + `qa: [{question, answer, scope, created_at}]` |
| DELETE | `/api/entities/{id}` | |
| GET | `/api/entities/{id}/crop` | JPEG crop |
| POST | `/api/entities/{id}/ask` | `{question, scope: "crop"\|"scene", vlm?}` (legacy `use_full_scene: true` = `"scene"`) → `{answer, elapsed, scope}`; Q&A is persisted. 503 without a VLM |
| POST | `/api/entities/{id}/assign` | `{person_id: string \| null}` (null = create a new person) → `Entity` |
| POST | `/api/entities/{id}/review` | `{action: "accept" \| "new" \| "reject"}` resolve a pending match → `Entity` |

### Search
`POST /api/search` — multipart **or** JSON.
- JSON: `{text, labels?: string[], media_id?, person_id?, kind?: "image"|"video", limit?: 30}`
- multipart: `text` (optional), `file` (optional image), `limit`, `media_id`, `person_id`, `kind`.
- `text_query` is accepted as an alias of `text`; `limit` is clamped to 1..200. Neither text nor image → 400.

Text search fuses (reciprocal-rank fusion) full-text over descriptions/labels, semantic description
embeddings, and CLIP text→image similarity when available. An image query uses CLIP image→image and
person Re-ID.
```jsonc
{"results": [{"entity": Entity, "media": Media, "score": 0.032,
              "scores": {"fts": 0.4, "semantic": 0.71, "clip": 0.28, "label": 1, "image": null, "reid": null}}],
 "count": 12, "query": {"text": "red jacket", "has_image": false}}
```

### Persons (identity clusters)
| Method | Path | Notes |
|---|---|---|
| GET | `/api/persons?watchlist=false&q=&limit=100&offset=0&order=recent\|sightings` | `{persons: Person[], total}` |
| GET | `/api/persons/{id}` | `Person` + `sightings: (Entity & {media: Media})[]` |
| PATCH | `/api/persons/{id}` | `{label?, is_watchlist?, notes?}` → `Person` |
| DELETE | `/api/persons/{id}` | unlinks sightings |
| POST | `/api/persons/{id}/merge` | `{other_id}` → `Person` |
| POST | `/api/persons/search?limit=15&min_similarity=0.2` | multipart `file` → `{results: [{person: Person, similarity}]}` (the largest detected person in the photo is embedded); 503 without Re-ID |
| POST | `/api/persons/recluster` | `{threshold?: float}` → `{persons: int, changed: int, sightings: int}` re-cluster all person sightings; 503 without Re-ID |
| GET | `/api/persons/review?limit=200` | `{pending: Entity[]}` sightings waiting for review |

`Person`:
```jsonc
{"id": "p_...", "label": "Alice", "is_watchlist": false, "notes": null, "sighting_count": 5,
 "media_count": 3, "representative_entity_id": "e_...", "thumb_url": "/api/entities/e_.../crop",
 "first_seen": 1731900000.0, "last_seen": 1731990000.0, "created_at": ..., "updated_at": ...}
```

### Agent
`POST /api/agent/chat` `{message, conversation_id?, llm?}` → **SSE**:
- `conversation` `{conversation_id}`
- `tool_call` `{id, name, arguments}`
- `tool_result` `{id, name, summary, elapsed, entities?: Entity[], media?: Media[], persons?: Person[], job_id?: string}`
- `message` `{text}` assistant text (markdown); may appear multiple times
- `done` `{}`
- `error` `{message}`
- `ping` `{}` keep-alive every 15 s while a tool runs (ignore)

The turn keeps running if the client disconnects, so the conversation stays consistent;
turns of one conversation run one at a time. An unavailable `llm` spec → 400.

`GET /api/agent/conversations` → `{conversations: [{id, title, created_at, updated_at, message_count}]}` (most recent first)

`GET /api/agent/conversations/{id}` → `{id, messages: [{role, text, tool_calls?}]}`

Agent tools: `search`, `find_in_library` (open-vocab scan over stored media), `plan_query`,
`list_media`, `get_media`, `ask_image`, `list_persons`, `get_person`, `name_person`,
`set_watchlist`, `merge_persons`, `recluster_persons`, `find_person_sightings`, `stats`,
`analyze_paths`, `job_status`.
