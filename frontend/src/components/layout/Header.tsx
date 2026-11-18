import { useActions, useAppState } from '../../context/hooks';
import type { Capabilities } from '../../types';
import { cn } from '../../utils/cn';
import { Button } from '../ui/Button';

const CAP_LABELS: Record<keyof Capabilities, string> = {
  vlm: 'VLM',
  llm: 'LLM',
  segmentation: 'Masks',
  clip: 'CLIP',
  text_search: 'Text search',
  reid: 'Re-ID',
  video: 'Video',
  agent: 'Agent',
};

export function Header() {
  const { health, healthError } = useAppState();
  const { setModelsOpen } = useActions();

  const status = healthError ? 'offline' : !health ? 'connecting' : health.ready ? 'ready' : 'loading';
  const dotClass = {
    offline: 'bg-error',
    connecting: 'bg-text-muted',
    ready: 'bg-success shadow-[0_0_8px_rgba(34,197,94,0.7)]',
    loading: 'bg-warning animate-pulse',
  }[status];
  const statusText = {
    offline: 'Backend unreachable',
    connecting: 'Connecting…',
    ready: `Ready${health?.device ? ` · ${health.device}` : ''}`,
    loading: `Loading models${health?.device ? ` · ${health.device}` : ''}`,
  }[status];

  const caps = health?.capabilities;

  return (
    <header className="h-14 shrink-0 flex items-center gap-4 px-4 border-b border-border bg-bg-secondary/80 backdrop-blur z-10">
      <div className="flex items-center gap-2.5 shrink-0">
        <div className="w-8 h-8 rounded-[10px] gradient-accent flex items-center justify-center text-white text-base font-bold">◎</div>
        <div className="leading-tight">
          <div className="font-semibold text-[15px]">
            OpenVocab <span className="text-gradient">Vision</span>
          </div>
          {health?.version && <div className="text-[10px] text-text-muted font-mono">v{health.version}</div>}
        </div>
      </div>

      <div className="flex items-center gap-2 text-xs text-text-secondary shrink-0" title={health ? `db: ${String(health.db)}` : undefined}>
        <span className={cn('w-2.5 h-2.5 rounded-full', dotClass)} />
        {statusText}
      </div>

      <div className="flex-1 min-w-0 flex items-center gap-1.5 overflow-x-auto">
        {caps &&
          (Object.keys(CAP_LABELS) as (keyof Capabilities)[]).map((k) => {
            const v = caps[k];
            const on = !!v;
            return (
              <span
                key={k}
                title={typeof v === 'string' ? `${CAP_LABELS[k]}: ${v}` : `${CAP_LABELS[k]} ${on ? 'available' : 'unavailable'}`}
                className={cn(
                  'shrink-0 px-2 py-0.5 rounded-full border text-[10.5px]',
                  on ? 'border-success/40 text-success bg-success-soft' : 'border-border text-text-muted line-through decoration-text-muted/60',
                )}
              >
                {CAP_LABELS[k]}
                {typeof v === 'string' ? ` · ${v}` : ''}
              </span>
            );
          })}
      </div>

      <Button variant="secondary" size="sm" onClick={() => setModelsOpen(true)}>
        ⚙ Models
      </Button>
    </header>
  );
}
