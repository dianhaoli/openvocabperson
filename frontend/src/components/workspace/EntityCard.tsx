import type { Entity } from '../../types';
import { labelColor } from '../../utils/colors';
import { cn } from '../../utils/cn';
import { formatTime, personName, pct } from '../../utils/format';
import { IdentityBadge, VerifiedBadge } from '../common/IdentityBadge';
import { Img } from '../common/Img';
import { Badge } from '../ui/Misc';

export function EntityCard({ entity, selected, onClick }: { entity: Entity; selected: boolean; onClick: () => void }) {
  const color = labelColor(entity.label);
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'flex flex-col text-left rounded-xl overflow-hidden border bg-bg-card transition-colors min-w-0',
        selected ? 'border-accent ring-1 ring-accent' : 'border-border hover:border-accent-border',
        entity.stage === 'rejected' && 'opacity-60',
      )}
    >
      <div className="relative aspect-[4/3]">
        <Img src={entity.crop_url} alt={entity.label} className="absolute inset-0 w-full h-full object-cover" />
        <span className="absolute top-0 left-0 h-full w-1" style={{ background: color }} />
        <span className="absolute top-1 right-1 font-mono text-[10px] bg-black/75 rounded px-1">{pct(entity.score)}</span>
        {entity.t_start != null && (
          <span className="absolute bottom-1 left-2 font-mono text-[9.5px] bg-black/75 rounded px-1">
            {formatTime(entity.t_start)}–{formatTime(entity.t_end)}
          </span>
        )}
      </div>
      <div className="p-2 flex flex-col gap-1 min-w-0">
        <div className="flex items-center gap-1.5 min-w-0">
          <span className="w-2 h-2 rounded-full shrink-0" style={{ background: color }} />
          <span className="text-xs font-medium truncate">{entity.person ? personName(entity.person) : entity.label}</span>
        </div>
        <div className="flex flex-wrap gap-1">
          {entity.person && <Badge>{entity.label}</Badge>}
          {entity.concept && entity.concept !== entity.label && <Badge tone="accent" title="Query concept">{entity.concept}</Badge>}
          <VerifiedBadge entity={entity} />
          <IdentityBadge entity={entity} />
          {entity.stage === 'low_confidence' && <Badge tone="warning">low conf.</Badge>}
        </div>
        {entity.description && <div className="text-[11px] text-text-muted line-clamp-2">{entity.description}</div>}
      </div>
    </button>
  );
}
