import type {
  HealthResponse,
  AnalyzeResponse,
  SessionsListResponse,
  SessionDetails,
  AskResponse,
  DeleteSessionResponse,
  SearchResponse,
  TextSearchResponse,
  StreamEvent,
  Person,
  PersonDetail,
  PersonSearchResult,
  AssignEntityResponse,
} from '../types';

// ══════════════════════════════════════════════════════════════════════════════
// Health Check
// ══════════════════════════════════════════════════════════════════════════════

export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch('/health');
  if (!res.ok) throw new Error('Health check failed');
  return res.json();
}

// ══════════════════════════════════════════════════════════════════════════════
// Analysis
// ══════════════════════════════════════════════════════════════════════════════

export async function analyzeImage(file: File): Promise<AnalyzeResponse> {
  const formData = new FormData();
  formData.append('file', file);
  
  const res = await fetch('/analyze', {
    method: 'POST',
    body: formData,
  });
  
  if (!res.ok) {
    let errorMessage = 'Analysis failed';
    try {
      const errorData = await res.json();
      errorMessage = errorData.detail || errorMessage;
    } catch {
      // If JSON parsing fails, use status text
      errorMessage = res.statusText || errorMessage;
    }
    throw new Error(errorMessage);
  }
  return res.json();
}

export async function* analyzeImageStream(
  file: File
): AsyncGenerator<StreamEvent, void, unknown> {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('/analyze/full-stream', {
    method: 'POST',
    body: formData,
  });
  
  if (!response.ok) {
    throw new Error('Streaming analysis failed');
  }
  
  const contentType = response.headers.get('content-type');
  if (!contentType?.includes('text/event-stream')) {
    throw new Error('Streaming not available');
  }
  
  const reader = response.body?.getReader();
  if (!reader) throw new Error('No response body');
  
  const decoder = new TextDecoder();
  let buffer = '';
  
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';
    
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try {
          const event: StreamEvent = JSON.parse(line.slice(6));
          yield event;
        } catch {
          console.error('Failed to parse stream event:', line);
        }
      }
    }
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// Sessions
// ══════════════════════════════════════════════════════════════════════════════

export async function listSessions(
  limit = 50,
  offset = 0
): Promise<SessionsListResponse> {
  const res = await fetch(`/api/sessions?limit=${limit}&offset=${offset}`);
  if (!res.ok) throw new Error('Failed to load sessions');
  return res.json();
}

export async function getSession(sessionId: string): Promise<SessionDetails> {
  const res = await fetch(`/api/session/${sessionId}`);
  if (!res.ok) throw new Error('Failed to load session');
  return res.json();
}

export async function deleteSession(
  sessionId: string
): Promise<DeleteSessionResponse> {
  const res = await fetch(`/api/session/${sessionId}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error('Failed to delete session');
  return res.json();
}

export function getSessionImageUrl(sessionId: string): string {
  return `/api/image/session/${sessionId}`;
}

export function getCropImageUrl(objectId: string): string {
  return `/api/image/crop/${objectId}`;
}

// ══════════════════════════════════════════════════════════════════════════════
// Q&A
// ══════════════════════════════════════════════════════════════════════════════

export async function askQuestion(
  objectId: string,
  question: string,
  useFullScene = false
): Promise<AskResponse> {
  const res = await fetch(`/object/${objectId}/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, use_full_scene: useFullScene }),
  });
  
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Request failed' }));
    throw new Error(err.detail || 'Request failed');
  }
  
  return res.json();
}

// ══════════════════════════════════════════════════════════════════════════════
// Search
// ══════════════════════════════════════════════════════════════════════════════

export async function searchText(
  query: string,
  limit = 50
): Promise<TextSearchResponse> {
  const res = await fetch(
    `/api/search/text?q=${encodeURIComponent(query)}&limit=${limit}`
  );
  if (!res.ok) throw new Error('Text search failed');
  return res.json();
}

export async function searchByImage(
  file: File,
  limit = 20,
  minSimilarity = 0
): Promise<SearchResponse> {
  const formData = new FormData();
  formData.append('file', file);
  
  const res = await fetch(
    `/api/search/image?limit=${limit}&min_similarity=${minSimilarity}`,
    {
      method: 'POST',
      body: formData,
    }
  );
  
  if (!res.ok) throw new Error('Image search failed');
  return res.json();
}

export async function hybridSearch(
  textQuery: string | null,
  imageFile: File | null,
  textWeight = 0.5,
  vectorWeight = 0.5,
  limit = 20
): Promise<SearchResponse> {
  const formData = new FormData();
  if (imageFile) {
    formData.append('file', imageFile);
  }
  
  const params = new URLSearchParams({
    text_weight: textWeight.toString(),
    vector_weight: vectorWeight.toString(),
    limit: limit.toString(),
  });
  
  if (textQuery) {
    params.set('text_query', textQuery);
  }
  
  const res = await fetch(`/api/search?${params}`, {
    method: 'POST',
    body: formData,
  });
  
  if (!res.ok) throw new Error('Hybrid search failed');
  return res.json();
}

// ══════════════════════════════════════════════════════════════════════════════
// Persons / Re-ID
// ══════════════════════════════════════════════════════════════════════════════

export async function listPersons(
  watchlist = false,
  limit = 80,
  offset = 0
): Promise<{ persons: Person[]; count: number }> {
  const params = new URLSearchParams({
    watchlist: String(watchlist),
    limit: String(limit),
    offset: String(offset),
  });
  const res = await fetch(`/api/persons?${params}`);
  if (!res.ok) throw new Error('Failed to load persons');
  return res.json();
}

export async function getPerson(personId: string): Promise<PersonDetail> {
  const res = await fetch(`/api/persons/${encodeURIComponent(personId)}`);
  if (!res.ok) throw new Error('Failed to load person');
  return res.json();
}

export async function patchPerson(
  personId: string,
  body: { label?: string | null; is_watchlist?: boolean; notes?: string | null }
): Promise<{ success: boolean; person: Pick<Person, 'person_id' | 'label' | 'is_watchlist' | 'notes'> }> {
  const res = await fetch(`/api/persons/${encodeURIComponent(personId)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Update failed' }));
    throw new Error(err.detail || 'Update failed');
  }
  return res.json();
}

export async function deletePerson(personId: string): Promise<{ success: boolean; person_id: string }> {
  const res = await fetch(`/api/persons/${encodeURIComponent(personId)}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error('Failed to delete person');
  return res.json();
}

export async function mergePersons(
  keepId: string,
  otherId: string
): Promise<{ success: boolean; kept_person_id: string; removed_person_id: string }> {
  const res = await fetch(`/api/persons/${encodeURIComponent(keepId)}/merge`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ other_id: otherId }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Merge failed' }));
    throw new Error(err.detail || 'Merge failed');
  }
  return res.json();
}

export async function assignEntity(
  objectId: string,
  personId: string
): Promise<AssignEntityResponse> {
  const res = await fetch(`/api/entity/${encodeURIComponent(objectId)}/assign`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ person_id: personId }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Assignment failed' }));
    throw new Error(err.detail || 'Assignment failed');
  }
  return res.json();
}

export async function searchPersonsByPhoto(
  file: File,
  limit = 15,
  minSimilarity = 0.2
): Promise<{ results: PersonSearchResult[]; count: number }> {
  const formData = new FormData();
  formData.append('file', file);
  const params = new URLSearchParams({
    limit: String(limit),
    min_similarity: String(minSimilarity),
  });
  const res = await fetch(`/api/persons/search?${params}`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Person search failed' }));
    throw new Error(err.detail || 'Person search failed');
  }
  return res.json();
}

