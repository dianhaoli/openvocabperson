// Single typed client for the OpenVocab Vision backend (see docs/API.md).
import type {
  AgentChatRequest,
  AgentEvent,
  AnalyzeOptions,
  ConfigPatch,
  Entity,
  EntityDetail,
  Health,
  Job,
  JobEvent,
  Media,
  MediaDetail,
  MediaKind,
  ModelsInfo,
  Person,
  PersonDetail,
  PersonPatch,
  Plan,
  ReviewAction,
  SearchRequest,
  SearchResponse,
  Stats,
  Tristate,
} from '../types';

/** Error raised for any non-2xx response; `detail` carries the backend's `detail` field when present. */
export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;

  constructor(status: number, detail: string) {
    super(detail || `Request failed (${status})`);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

export function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === 'AbortError';
}

/** Human readable message for any thrown value. */
export function errorMessage(err: unknown): string {
  if (err instanceof ApiError) return err.detail || err.message;
  if (err instanceof Error) return err.message;
  return String(err);
}

function stringifyDetail(detail: unknown): string {
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    // FastAPI validation errors: [{loc, msg, type}]
    return detail
      .map((d) => {
        if (d && typeof d === 'object' && 'msg' in d) {
          const rec = d as { msg: unknown; loc?: unknown };
          const loc = Array.isArray(rec.loc) ? rec.loc.join('.') : '';
          return loc ? `${loc}: ${String(rec.msg)}` : String(rec.msg);
        }
        return JSON.stringify(d);
      })
      .join('; ');
  }
  if (detail == null) return '';
  return JSON.stringify(detail);
}

async function toApiError(res: Response): Promise<ApiError> {
  let detail = '';
  try {
    const text = await res.text();
    try {
      const body: unknown = JSON.parse(text);
      if (body && typeof body === 'object' && 'detail' in body) {
        detail = stringifyDetail((body as { detail: unknown }).detail);
      } else {
        detail = text;
      }
    } catch {
      detail = text.slice(0, 300);
    }
  } catch {
    /* ignore body read errors */
  }
  return new ApiError(res.status, detail || `${res.status} ${res.statusText}`);
}

interface RequestOptions {
  method?: string;
  json?: unknown;
  form?: FormData;
  signal?: AbortSignal;
  query?: Record<string, string | number | boolean | null | undefined>;
}

function buildUrl(path: string, query?: RequestOptions['query']): string {
  if (!query) return path;
  const params = new URLSearchParams();
  for (const [k, v] of Object.entries(query)) {
    if (v === undefined || v === null || v === '') continue;
    params.set(k, String(v));
  }
  const qs = params.toString();
  return qs ? `${path}?${qs}` : path;
}

function buildInit(opts: RequestOptions): RequestInit {
  const init: RequestInit = { method: opts.method ?? 'GET', signal: opts.signal };
  if (opts.form) {
    init.body = opts.form;
  } else if (opts.json !== undefined) {
    init.body = JSON.stringify(opts.json);
    init.headers = { 'Content-Type': 'application/json' };
  }
  return init;
}

async function request<T>(path: string, opts: RequestOptions = {}): Promise<T> {
  const res = await fetch(buildUrl(path, opts.query), buildInit(opts));
  if (!res.ok) throw await toApiError(res);
  if (res.status === 204) return undefined as T;
  const text = await res.text();
  return (text ? JSON.parse(text) : undefined) as T;
}

// ---------------------------------------------------------------------------
// SSE over fetch
// ---------------------------------------------------------------------------

/**
 * POST/GET a URL and yield each Server-Sent Event's JSON payload.
 * Handles chunk boundaries, CRLF line endings, multi-line `data:` fields and
 * `event:` names (injected as `type` when the payload has none).
 * Aborting `opts.signal` stops the stream (the generator throws AbortError).
 */
export async function* sseStream<T>(path: string, opts: RequestOptions = {}): AsyncGenerator<T, void, undefined> {
  const init = buildInit(opts);
  init.headers = { ...(init.headers as Record<string, string> | undefined), Accept: 'text/event-stream' };
  init.cache = 'no-store';
  const res = await fetch(buildUrl(path, opts.query), init);
  if (!res.ok) throw await toApiError(res);
  if (!res.body) throw new ApiError(res.status, 'Streaming not supported by this browser');

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  const parseBlock = (block: string): T | null => {
    const dataLines: string[] = [];
    let eventName: string | null = null;
    for (const rawLine of block.split('\n')) {
      const line = rawLine.endsWith('\r') ? rawLine.slice(0, -1) : rawLine;
      if (!line || line.startsWith(':')) continue;
      if (line.startsWith('data:')) {
        dataLines.push(line.slice(5).replace(/^ /, ''));
      } else if (line.startsWith('event:')) {
        eventName = line.slice(6).trim();
      }
    }
    if (dataLines.length === 0) return null;
    const data = dataLines.join('\n');
    try {
      const parsed: unknown = JSON.parse(data);
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        const obj = parsed as Record<string, unknown>;
        if (eventName && obj.type === undefined) obj.type = eventName;
        return obj as T;
      }
      return null;
    } catch {
      return null;
    }
  };

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let idx: number;
      while ((idx = buffer.indexOf('\n\n')) !== -1) {
        const block = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        const evt = parseBlock(block);
        if (evt) yield evt;
      }
    }
    buffer += decoder.decode();
    if (buffer.trim()) {
      const evt = parseBlock(buffer);
      if (evt) yield evt;
    }
  } finally {
    try {
      await reader.cancel();
    } catch {
      /* already closed */
    }
  }
}

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------

export const api = {
  // System
  health: (signal?: AbortSignal) => request<Health>('/api/health', { signal }),
  models: (signal?: AbortSignal) => request<ModelsInfo>('/api/models', { signal }),
  patchConfig: (patch: ConfigPatch) => request<ModelsInfo>('/api/config', { method: 'PATCH', json: patch }),
  stats: (signal?: AbortSignal) => request<Stats>('/api/stats', { signal }),

  // Planning & analysis
  plan: (query: string, expand: Tristate, signal?: AbortSignal) =>
    request<Plan>('/api/plan', { method: 'POST', json: { query, expand }, signal }),
  analyze: (files: File[], options: AnalyzeOptions) => {
    const form = new FormData();
    for (const f of files) form.append('files', f, f.name);
    form.append('options', JSON.stringify(options));
    return request<{ job: Job }>('/api/analyze', { method: 'POST', form });
  },
  analyzePaths: (paths: string[], options: AnalyzeOptions) =>
    request<{ job: Job }>('/api/analyze/paths', { method: 'POST', json: { paths, options } }),
  scan: (body: { query: string; media_ids?: string[]; limit?: number; options?: AnalyzeOptions }) =>
    request<{ job: Job }>('/api/scan', { method: 'POST', json: body }),
  jobs: (signal?: AbortSignal) => request<{ jobs: Job[] }>('/api/jobs', { signal }),
  job: (id: string, signal?: AbortSignal) =>
    request<{ job: Job; media: MediaDetail[] }>(`/api/jobs/${encodeURIComponent(id)}`, { signal }),
  jobEvents: (id: string, signal: AbortSignal) =>
    sseStream<JobEvent>(`/api/jobs/${encodeURIComponent(id)}/events`, { signal }),
  cancelJob: (id: string) => request<{ job: Job }>(`/api/jobs/${encodeURIComponent(id)}/cancel`, { method: 'POST' }),

  // Media & entities
  listMedia: (params: { limit?: number; offset?: number; kind?: MediaKind | null }, signal?: AbortSignal) =>
    request<{ media: Media[]; total: number }>('/api/media', { query: params, signal }),
  media: (id: string, signal?: AbortSignal) => request<MediaDetail>(`/api/media/${encodeURIComponent(id)}`, { signal }),
  deleteMedia: (id: string) => request<unknown>(`/api/media/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  askMedia: (id: string, question: string) =>
    request<{ answer: string; elapsed: number }>(`/api/media/${encodeURIComponent(id)}/ask`, {
      method: 'POST',
      json: { question },
    }),
  entity: (id: string, signal?: AbortSignal) =>
    request<EntityDetail>(`/api/entities/${encodeURIComponent(id)}`, { signal }),
  deleteEntity: (id: string) => request<unknown>(`/api/entities/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  askEntity: (id: string, question: string, scope: 'crop' | 'scene') =>
    request<{ answer: string; elapsed: number }>(`/api/entities/${encodeURIComponent(id)}/ask`, {
      method: 'POST',
      json: { question, scope },
    }),
  assignEntity: (id: string, personId: string | null) =>
    request<Entity>(`/api/entities/${encodeURIComponent(id)}/assign`, { method: 'POST', json: { person_id: personId } }),
  reviewEntity: (id: string, action: ReviewAction) =>
    request<Entity>(`/api/entities/${encodeURIComponent(id)}/review`, { method: 'POST', json: { action } }),

  // Search
  search: (req: SearchRequest, image?: File | null, signal?: AbortSignal) => {
    if (image) {
      const form = new FormData();
      if (req.text) form.append('text', req.text);
      form.append('file', image, image.name);
      form.append('limit', String(req.limit ?? 30));
      return request<SearchResponse>('/api/search', { method: 'POST', form, signal });
    }
    return request<SearchResponse>('/api/search', { method: 'POST', json: req, signal });
  },

  // Persons
  persons: (params: { watchlist?: boolean; q?: string; limit?: number; offset?: number }, signal?: AbortSignal) =>
    request<{ persons: Person[]; total: number }>('/api/persons', {
      query: { ...params, watchlist: params.watchlist ? true : undefined },
      signal,
    }),
  person: (id: string, signal?: AbortSignal) =>
    request<PersonDetail>(`/api/persons/${encodeURIComponent(id)}`, { signal }),
  patchPerson: (id: string, patch: PersonPatch) =>
    request<Person>(`/api/persons/${encodeURIComponent(id)}`, { method: 'PATCH', json: patch }),
  deletePerson: (id: string) => request<unknown>(`/api/persons/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  mergePerson: (id: string, otherId: string) =>
    request<Person>(`/api/persons/${encodeURIComponent(id)}/merge`, { method: 'POST', json: { other_id: otherId } }),
  searchPersons: (file: File) => {
    const form = new FormData();
    form.append('file', file, file.name);
    return request<{ results: { person: Person; similarity: number }[] }>('/api/persons/search', {
      method: 'POST',
      form,
    });
  },
  recluster: (threshold?: number) =>
    request<{ persons: number; changed: number }>('/api/persons/recluster', {
      method: 'POST',
      json: threshold === undefined ? {} : { threshold },
    }),
  reviewQueue: (signal?: AbortSignal) => request<{ pending: Entity[] }>('/api/persons/review', { signal }),

  // Agent
  agentChat: (body: AgentChatRequest, signal: AbortSignal) =>
    sseStream<AgentEvent>('/api/agent/chat', { method: 'POST', json: body, signal }),
};
