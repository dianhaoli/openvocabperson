// Types mirroring docs/API.md. Field names are kept in snake_case exactly as in the JSON.

export type Box = [number, number, number, number];
/** One trajectory sample: [t, x1, y1, x2, y2] */
export type TrajectoryPoint = [number, number, number, number, number];
export type Point = [number, number];

export type EntityStage = 'described' | 'detected' | 'low_confidence' | 'rejected';
export type MatchStatus = 'matched' | 'new' | 'pending' | 'manual';
export type MediaKind = 'image' | 'video';

export interface PersonRef {
  id: string;
  label: string | null;
  is_watchlist: boolean;
}

export interface Entity {
  id: string;
  media_id: string;
  label: string;
  concept: string | null;
  score: number;
  box: Box;
  polygon: Point[][] | null;
  mask_area: number | null;
  crop_url: string;
  description: string | null;
  attributes: Record<string, unknown> | null;
  verified: boolean | null;
  verify_reason: string | null;
  stage: EntityStage;
  track_id?: number | null;
  t_start?: number | null;
  t_end?: number | null;
  frame_index?: number | null;
  trajectory?: TrajectoryPoint[] | null;
  person: PersonRef | null;
  match_status: MatchStatus | null;
  match_score: number | null;
  suggested_person?: PersonRef | null;
  created_at: number;
}

export interface Media {
  id: string;
  kind: MediaKind;
  name: string;
  width: number;
  height: number;
  duration: number | null;
  fps: number | null;
  url: string;
  thumb_url: string;
  query: string | null;
  entity_count: number;
  person_count: number;
  created_at: number;
}

export interface MediaDetail extends Media {
  entities: Entity[];
  plan: Plan | null;
}

export interface QAItem {
  question: string;
  answer: string;
  created_at: number;
}

export interface EntityDetail extends Entity {
  media: Media;
  qa: QAItem[];
}

export type ConceptKind = 'object' | 'person_attribute' | string;

export interface PlanConcept {
  name: string;
  kind: ConceptKind;
  prompts: string[];
  verify: boolean;
  question: string | null;
}

export type PlanSource = 'llm' | 'lexicon' | 'literal' | 'default';

export interface Plan {
  query: string | null;
  expanded: boolean;
  source: PlanSource;
  concepts: PlanConcept[];
  prompts: string[];
}

export type JobKind = 'analyze' | 'scan';
export type JobStatus = 'queued' | 'running' | 'done' | 'error' | 'cancelled';

export interface JobProgress {
  done: number;
  total: number;
  stage?: string | null;
  message?: string | null;
}

export type Tristate = boolean | 'auto';
export type DescribeMode = 'persons' | 'all' | 'none';

export interface AnalyzeOptions {
  query?: string | null;
  expand?: Tristate;
  verify?: Tristate;
  segment?: boolean;
  describe?: DescribeMode;
  detector?: string | null;
  segmenter?: string | null;
  vlm?: string | null;
  conf?: number;
  video_fps?: number;
  max_frames?: number;
  reid?: boolean;
}

export interface Job {
  id: string;
  kind: JobKind;
  status: JobStatus;
  progress: JobProgress | null;
  media_ids: string[];
  error: string | null;
  created_at: number;
  finished_at: number | null;
  options: AnalyzeOptions | null;
}

// ---------- Job SSE events ----------
interface JobEventBase {
  job_id: string;
  ts: number;
}
export type PipelineStage = 'detect' | 'track' | 'segment' | 'verify' | 'describe' | 'embed' | 'identity' | 'save';
export type LogLevel = 'info' | 'warning' | 'error';

export type JobEvent =
  | (JobEventBase & { type: 'job_started'; job: Job })
  | (JobEventBase & { type: 'plan'; plan: Plan })
  | (JobEventBase & { type: 'media_started'; index: number; total: number; name: string; kind: MediaKind })
  | (JobEventBase & { type: 'media_created'; media: Media })
  | (JobEventBase & { type: 'stage'; media_id: string; stage: PipelineStage | string; message: string })
  | (JobEventBase & { type: 'detections'; media_id: string; count: number; labels: Record<string, number> })
  | (JobEventBase & { type: 'entity'; entity: Entity })
  | (JobEventBase & { type: 'media_done'; media: MediaDetail })
  | (JobEventBase & { type: 'progress'; done: number; total: number })
  | (JobEventBase & { type: 'log'; level: LogLevel; message: string })
  | (JobEventBase & { type: 'job_done'; job: Job });

// ---------- System ----------
export interface Capabilities {
  vlm: boolean;
  llm: boolean;
  segmentation: boolean;
  clip: boolean;
  text_search: boolean;
  reid: string | boolean | null;
  video: boolean;
  agent: boolean;
}

export interface Health {
  status: string;
  ready: boolean;
  device: string;
  db: string | boolean | null;
  capabilities: Capabilities;
  version: string;
}

export type ModelRole = 'detector' | 'segmenter' | 'vlm' | 'llm' | 'embedder' | 'reid';
export const MODEL_ROLES: ModelRole[] = ['detector', 'segmenter', 'vlm', 'llm', 'embedder', 'reid'];

export interface ModelProvider {
  spec: string;
  label: string;
  available: boolean;
  local?: boolean;
  open_vocab?: boolean;
  masks?: boolean;
  note: string | null;
}

export interface ModelsInfo {
  device: string;
  defaults: Partial<Record<ModelRole | 'text_embedder', string | null>>;
  providers: Partial<Record<ModelRole, ModelProvider[]>>;
  capabilities: Capabilities;
}

export type ConfigPatch = Partial<Record<ModelRole, string>>;

export interface Stats {
  media: number;
  images: number;
  videos: number;
  entities: number;
  persons: number;
  watchlist: number;
  labels: { label: string; count: number }[];
}

// ---------- Search ----------
export interface ScoreBreakdown {
  fts: number | null;
  semantic: number | null;
  clip: number | null;
  label: number | null;
  image: number | null;
  reid: number | null;
}

export interface SearchResult {
  entity: Entity;
  media: Media;
  score: number;
  scores: Partial<ScoreBreakdown>;
}

export interface SearchResponse {
  results: SearchResult[];
  count: number;
  query: { text: string | null; has_image: boolean };
}

export interface SearchRequest {
  text?: string;
  labels?: string[];
  media_id?: string;
  person_id?: string;
  kind?: MediaKind;
  limit?: number;
}

// ---------- Persons ----------
export interface Person {
  id: string;
  label: string | null;
  is_watchlist: boolean;
  notes: string | null;
  sighting_count: number;
  media_count: number;
  representative_entity_id: string | null;
  thumb_url: string | null;
  first_seen: number | null;
  last_seen: number | null;
  created_at: number;
  updated_at: number;
}

export type Sighting = Entity & { media: Media };

export interface PersonDetail extends Person {
  sightings: Sighting[];
}

export interface PersonPatch {
  label?: string | null;
  is_watchlist?: boolean;
  notes?: string | null;
}

export type ReviewAction = 'accept' | 'new' | 'reject';

// ---------- Agent ----------
export interface AgentToolResult {
  id: string;
  name: string;
  summary: string;
  entities?: Entity[];
  media?: Media[];
  persons?: Person[];
  job_id?: string;
}

export type AgentEvent =
  | { type: 'conversation'; conversation_id: string }
  | { type: 'tool_call'; id: string; name: string; arguments: Record<string, unknown> | string | null }
  | ({ type: 'tool_result' } & AgentToolResult)
  | { type: 'message'; text: string }
  | { type: 'done' }
  | { type: 'error'; message: string };

export interface AgentChatRequest {
  message: string;
  conversation_id?: string;
  llm?: string;
}
