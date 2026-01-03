// ══════════════════════════════════════════════════════════════════════════════
// Entity & Detection Types
// ══════════════════════════════════════════════════════════════════════════════

export type AnalysisStage = 'vlm_full' | 'yolo_only' | 'low_confidence';

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface Entity {
  object_id: string;
  index: number;
  class: string;
  confidence: number;
  box: [number, number, number, number]; // [x1, y1, x2, y2]
  stage: AnalysisStage;
  analysis: string | null;
  reason?: string;
  crop_image: string; // base64 data URL
}

export interface EntityFromSession {
  object_id: string;
  class_name: string;
  confidence: number;
  box: [number, number, number, number];
  stage: AnalysisStage;
  analysis: string | null;
  crop_image: string | null;
  created_at: number;
}

// ══════════════════════════════════════════════════════════════════════════════
// Session Types
// ══════════════════════════════════════════════════════════════════════════════

export interface Session {
  session_id: string;
  created_at: number;
  entity_count: number;
}

export interface SessionDetails {
  session_id: string;
  created_at: number;
  image_width: number;
  image_height: number;
  full_image_path: string;
  entities: EntityFromSession[];
}

// ══════════════════════════════════════════════════════════════════════════════
// API Response Types
// ══════════════════════════════════════════════════════════════════════════════

export interface HealthResponse {
  status: string;
  pipeline_ready: boolean;
  total_sessions: number;
}

export interface AnalyzeResponse {
  success: boolean;
  session_id: string;
  elapsed_seconds: number;
  num_detections: number;
  image_width: number;
  image_height: number;
  results: Entity[];
}

export interface SessionsListResponse {
  sessions: Session[];
  total: number;
  limit: number;
  offset: number;
}

export interface AskResponse {
  success: boolean;
  object_id: string;
  question: string;
  answer: string;
  elapsed_seconds: number;
}

export interface DeleteSessionResponse {
  success: boolean;
  session_id: string;
  entities_deleted: number;
}

// ══════════════════════════════════════════════════════════════════════════════
// Search Types
// ══════════════════════════════════════════════════════════════════════════════

export interface SearchScores {
  text: number;
  vector: number;
  hybrid: number;
}

export interface SearchResult {
  object_id: string;
  session_id: string;
  class_name: string;
  confidence?: number;
  analysis: string | null;
  crop_image: string | null;
  scores?: SearchScores;
  similarity?: number;
  score?: number;
}

export interface SearchResponse {
  results: SearchResult[];
  count: number;
  query?: {
    text: string | null;
    has_image: boolean;
    text_weight: number;
    vector_weight: number;
  };
}

export interface TextSearchResponse {
  results: SearchResult[];
  count: number;
  query: string;
}

// ══════════════════════════════════════════════════════════════════════════════
// Streaming Event Types
// ══════════════════════════════════════════════════════════════════════════════

export type StreamEventType = 
  | 'detection_complete'
  | 'routing_complete'
  | 'crop_analyzed'
  | 'complete'
  | 'error';

export interface StreamEvent {
  type: StreamEventType;
  elapsed: number;
  data?: {
    num_detections?: number;
    vlm_full?: number;
    yolo_only?: number;
    low_confidence?: number;
    index?: number;
    analysis?: string;
    reason?: string;
  };
  results?: Entity[];
  message?: string;
  session_id?: string;
}

// ══════════════════════════════════════════════════════════════════════════════
// UI State Types
// ══════════════════════════════════════════════════════════════════════════════

export type SidebarTab = 'upload' | 'search' | 'history';

export interface LogEntry {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  message: string;
  time: string;
}

export interface QAItem {
  id: string;
  question: string;
  answer: string | null;
  loading: boolean;
}

