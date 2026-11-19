import type { ReactNode } from 'react';
import { errorMessage } from '../../api/client';
import { cn } from '../../utils/cn';
import { Button } from './Button';
import { Spinner } from './Spinner';

type Tone = 'neutral' | 'accent' | 'success' | 'warning' | 'error';

const tones: Record<Tone, string> = {
  neutral: 'bg-bg-tertiary text-text-secondary border-border',
  accent: 'bg-accent-soft text-[#a5a8ff] border-accent-border',
  success: 'bg-success-soft text-success border-success/40',
  warning: 'bg-warning-soft text-warning border-warning/40',
  error: 'bg-error/15 text-error border-error/40',
};

export function Badge({
  children,
  tone = 'neutral',
  className,
  title,
}: {
  children: ReactNode;
  tone?: Tone;
  className?: string;
  title?: string;
}) {
  return (
    <span
      title={title}
      className={cn(
        'inline-flex items-center gap-1 px-1.5 py-px rounded-md border text-[10.5px] font-medium leading-4 whitespace-nowrap',
        tones[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

export function Chip({
  children,
  active,
  onClick,
  className,
  title,
  color,
}: {
  children: ReactNode;
  active?: boolean;
  onClick?: () => void;
  className?: string;
  title?: string;
  color?: string;
}) {
  const Comp = onClick ? 'button' : 'span';
  return (
    <Comp
      type={onClick ? 'button' : undefined}
      title={title}
      onClick={onClick}
      className={cn(
        'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-xs transition-colors',
        active
          ? 'bg-accent-soft border-accent-border text-text-primary'
          : 'bg-bg-tertiary border-border text-text-secondary',
        onClick && 'hover:border-accent hover:text-text-primary cursor-pointer',
        className,
      )}
    >
      {color && <span className="w-2 h-2 rounded-full shrink-0" style={{ background: color }} />}
      {children}
    </Comp>
  );
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  size = 'sm',
}: {
  value: T;
  options: { value: T; label: string; title?: string }[];
  onChange: (v: T) => void;
  size?: 'xs' | 'sm';
}) {
  return (
    <div className="inline-flex p-0.5 rounded-lg bg-bg-primary border border-border">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          title={o.title}
          onClick={() => onChange(o.value)}
          className={cn(
            'rounded-md transition-colors whitespace-nowrap',
            size === 'xs' ? 'px-2 py-0.5 text-[11px]' : 'px-2.5 py-1 text-xs',
            o.value === value ? 'bg-accent text-white' : 'text-text-secondary hover:text-text-primary',
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

const BAR_FILL = { warning: 'bg-warning/70', error: 'bg-error/70' } as const;

export function ProgressBar({ value, indeterminate, tone }: { value: number; indeterminate?: boolean; tone?: keyof typeof BAR_FILL }) {
  return (
    <div className="h-1.5 w-full rounded-full bg-bg-primary overflow-hidden border border-border">
      {indeterminate ? (
        <div className="h-full w-1/4 gradient-accent animate-indeterminate" />
      ) : (
        <div
          className={cn('h-full transition-[width] duration-300', tone ? BAR_FILL[tone] : 'gradient-accent')}
          style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }}
        />
      )}
    </div>
  );
}

export function EmptyState({ icon, title, hint, children }: { icon?: string; title: string; hint?: ReactNode; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center text-center gap-2 py-8 px-4 text-text-muted">
      {icon && <div className="text-3xl opacity-60">{icon}</div>}
      <div className="text-sm text-text-secondary">{title}</div>
      {hint && <div className="text-xs max-w-[280px]">{hint}</div>}
      {children}
    </div>
  );
}

export function Loading({ label = 'Loading…' }: { label?: string }) {
  return (
    <div className="flex items-center justify-center gap-2 py-8 text-text-muted text-sm">
      <Spinner /> {label}
    </div>
  );
}

export function ErrorState({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  return (
    <div className="flex flex-col items-center gap-2 py-6 px-4 text-center">
      <div className="text-sm text-error">{errorMessage(error)}</div>
      {onRetry && (
        <Button size="xs" variant="secondary" onClick={onRetry}>
          Retry
        </Button>
      )}
    </div>
  );
}

export function Section({ title, right, children, className }: { title: ReactNode; right?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={cn('flex flex-col gap-2', className)}>
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-[11px] uppercase tracking-wider text-text-muted font-semibold">{title}</h3>
        {right}
      </div>
      {children}
    </section>
  );
}

export function Field({ label, children, hint }: { label: ReactNode; children: ReactNode; hint?: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-3 min-h-8">
      <div className="text-xs text-text-secondary">
        {label}
        {hint && <div className="text-[10.5px] text-text-muted">{hint}</div>}
      </div>
      <div className="shrink-0">{children}</div>
    </div>
  );
}

export const inputClass =
  'w-full bg-bg-primary border border-border rounded-[10px] px-3 py-2 text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent transition-colors';
