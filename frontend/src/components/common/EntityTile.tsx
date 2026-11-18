import type { ReactNode } from 'react';
import type { Entity } from '../../types';
import { labelColor } from '../../utils/colors';
import { cn } from '../../utils/cn';
import { personName, pct } from '../../utils/format';
import { Img } from './Img';

/** Compact clickable crop tile used in galleries (agent, search, sightings). */
export function EntityTile({
  entity,
  onClick,
  footer,
  className,
}: {
  entity: Entity;
  onClick?: () => void;
  footer?: ReactNode;
  className?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={entity.description ?? entity.label}
      className={cn(
        'group flex flex-col text-left rounded-xl overflow-hidden border border-border bg-bg-card hover:border-accent transition-colors min-w-0',
        className,
      )}
    >
      <div className="relative aspect-square w-full">
        <Img src={entity.crop_url} alt={entity.label} className="absolute inset-0 w-full h-full object-cover" />
        {entity.person?.is_watchlist && (
          <span className="absolute top-1 right-1 text-[10px] bg-error text-white rounded px-1">★</span>
        )}
      </div>
      <div className="px-1.5 py-1 flex flex-col min-w-0">
        <div className="flex items-center gap-1 text-[11px] min-w-0">
          <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ background: labelColor(entity.label) }} />
          <span className="truncate">{entity.person ? personName(entity.person) : entity.label}</span>
          <span className="ml-auto text-text-muted font-mono text-[10px]">{pct(entity.score)}</span>
        </div>
        {footer}
      </div>
    </button>
  );
}
