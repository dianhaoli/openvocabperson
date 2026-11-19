import { useRef, useState } from 'react';
import { api, isAbortError } from '../../api/client';
import { useActions } from '../../context/hooks';
import { useObjectUrl } from '../../hooks/useObjectUrl';
import type { MediaKind, SearchResponse, SearchResult } from '../../types';
import { labelColor } from '../../utils/colors';
import { personName } from '../../utils/format';
import { IdentityBadge } from '../common/IdentityBadge';
import { Img } from '../common/Img';
import { Button } from '../ui/Button';
import { Dropzone } from '../ui/Dropzone';
import { Badge, EmptyState, Segmented, inputClass } from '../ui/Misc';

const SCORE_KEYS = ['fts', 'semantic', 'clip', 'label', 'image', 'reid'] as const;

function ResultCard({ r, onOpen }: { r: SearchResult; onOpen: () => void }) {
  const e = r.entity;
  return (
    <button
      type="button"
      onClick={onOpen}
      className="flex flex-col text-left rounded-xl overflow-hidden border border-border bg-bg-card hover:border-accent transition-colors min-w-0"
    >
      <div className="relative aspect-[4/3] bg-black/60">
        <Img src={e.crop_url} alt={e.label} className="absolute inset-0 w-full h-full object-contain" />
        <span className="absolute top-1 right-1 font-mono text-[10px] bg-black/75 rounded px-1">{r.score.toFixed(3)}</span>
        {r.media.kind === 'video' && <span className="absolute top-1 left-1 text-[9px] bg-black/70 rounded px-1">▶</span>}
      </div>
      <div className="p-2 flex flex-col gap-1 min-w-0">
        <div className="flex items-center gap-1.5 text-xs min-w-0">
          <span className="w-2 h-2 rounded-full shrink-0" style={{ background: labelColor(e.label) }} />
          <span className="font-medium truncate">{e.label}</span>
          {e.person && <span className="text-text-muted truncate">· {personName(e.person)}</span>}
        </div>
        {e.description && <div className="text-[11px] text-text-muted line-clamp-2">{e.description}</div>}
        <div className="flex flex-wrap gap-1">
          <IdentityBadge entity={e} />
          {SCORE_KEYS.map((k) => {
            const v = r.scores?.[k];
            if (v == null) return null;
            return (
              <Badge key={k} tone="accent" title={`${k} score`}>
                {k} {v.toFixed(2)}
              </Badge>
            );
          })}
        </div>
        <div className="text-[10px] text-text-muted truncate">{r.media.name}</div>
      </div>
    </button>
  );
}

export function SearchTab() {
  const { openMedia, toastError } = useActions();
  const [text, setText] = useState('');
  const [image, setImage] = useState<File | null>(null);
  const [kind, setKind] = useState<'all' | MediaKind>('all');
  const [limit, setLimit] = useState(30);
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);
  const imageUrl = useObjectUrl(image);

  const search = async () => {
    if (!text.trim() && !image) return;
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setLoading(true);
    try {
      const res = await api.search(
        { text: text.trim() || undefined, limit, kind: kind === 'all' ? undefined : kind },
        image,
        controller.signal,
      );
      setResult(res);
    } catch (err) {
      if (!isAbortError(err)) toastError(err, 'Search failed');
    } finally {
      if (controllerRef.current === controller) {
        controllerRef.current = null;
        setLoading(false);
      }
    }
  };

  return (
    <div className="flex flex-col gap-3 p-4">
      <form
        className="flex flex-col gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          void search();
        }}
      >
        <input
          className={inputClass}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder='e.g. "man in a red jacket", "knife", "dog on a sofa"'
        />
        <div className="flex gap-2 items-stretch">
          {image && imageUrl ? (
            <div className="relative w-16 h-16 shrink-0">
              <img src={imageUrl} alt="query" className="w-16 h-16 rounded-lg object-cover border border-accent" />
              <button
                type="button"
                onClick={() => setImage(null)}
                className="absolute -top-1.5 -right-1.5 w-5 h-5 rounded-full bg-bg-tertiary border border-border text-[11px] hover:bg-error"
                aria-label="Remove image"
              >
                ×
              </button>
            </div>
          ) : (
            <Dropzone accept="image/*" onFiles={(f) => setImage(f[0] ?? null)} compact className="w-full">
              <span className="text-xs text-text-muted">📷 Drop / pick a query image (optional)</span>
            </Dropzone>
          )}
          {image && <div className="text-[11px] text-text-muted self-center flex-1">Image query: CLIP image similarity + person Re-ID</div>}
        </div>
        <div className="flex items-center gap-2">
          <Segmented
            value={kind}
            options={[
              { value: 'all', label: 'All' },
              { value: 'image', label: 'Images' },
              { value: 'video', label: 'Videos' },
            ]}
            onChange={setKind}
            size="xs"
          />
          <select
            value={limit}
            onChange={(e) => setLimit(Number(e.target.value))}
            className="bg-bg-primary border border-border rounded-lg px-2 py-1 text-xs"
            title="Max results"
          >
            {[15, 30, 60, 100].map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
          <Button type="submit" size="sm" className="ml-auto" loading={loading} disabled={!text.trim() && !image}>
            Search
          </Button>
        </div>
        {image && kind !== 'all' && <div className="text-[10.5px] text-text-muted">Kind filter applies to text-only searches.</div>}
      </form>

      {!result && !loading && <EmptyState icon="⌕" title="Search detections" hint="Fuses full-text, semantic description embeddings and CLIP similarity." />}
      {result && (
        <>
          <div className="text-[11px] text-text-muted">
            {result.count} result(s)
            {result.query.text ? ` for “${result.query.text}”` : ''}
            {result.query.has_image ? ' + image' : ''}
          </div>
          {result.results.length === 0 ? (
            <EmptyState icon="∅" title="No matches" hint="Try other words, or analyze/scan more media." />
          ) : (
            <div className="grid grid-cols-2 gap-2">
              {result.results.map((r) => (
                <ResultCard key={r.entity.id} r={r} onOpen={() => openMedia(r.media.id, r.entity.id)} />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
