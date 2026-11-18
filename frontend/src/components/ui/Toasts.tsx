import { useActions, useAppState } from '../../context/hooks';
import { cn } from '../../utils/cn';

export function Toasts() {
  const { toasts } = useAppState();
  const { dismissToast } = useActions();
  return (
    <div className="fixed bottom-4 right-4 z-[2000] flex flex-col gap-2 w-[360px] max-w-[calc(100vw-2rem)] pointer-events-none">
      {toasts.map((t) => (
        <div
          key={t.id}
          role={t.kind === 'error' ? 'alert' : 'status'}
          className={cn(
            'pointer-events-auto flex items-start gap-3 px-4 py-3 rounded-xl border shadow-xl backdrop-blur animate-fade-in text-sm',
            t.kind === 'error' && 'bg-[#2a1215]/95 border-error/50 text-[#fecaca]',
            t.kind === 'success' && 'bg-[#0f2418]/95 border-success/50 text-[#bbf7d0]',
            t.kind === 'info' && 'bg-bg-tertiary/95 border-border-highlight text-text-primary',
          )}
        >
          <span className="flex-1 break-words">{t.message}</span>
          <button type="button" className="opacity-60 hover:opacity-100" onClick={() => dismissToast(t.id)} aria-label="Dismiss">
            ×
          </button>
        </div>
      ))}
    </div>
  );
}
