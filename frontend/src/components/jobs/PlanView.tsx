import type { Plan } from '../../types';
import { labelColor } from '../../utils/colors';
import { Badge, Chip } from '../ui/Misc';

export function PlanView({ plan, compact }: { plan: Plan; compact?: boolean }) {
  const anyVerify = plan.concepts.some((c) => c.verify);
  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-1.5 text-[11px] text-text-muted">
        <Badge tone={plan.expanded ? 'accent' : 'neutral'}>source: {plan.source}</Badge>
        {plan.expanded && <Badge tone="accent">expanded</Badge>}
        <Badge tone={anyVerify ? 'warning' : 'neutral'}>verification {anyVerify ? 'on' : 'off'}</Badge>
        <span>{plan.prompts.length} prompt(s)</span>
      </div>
      {plan.concepts.length === 0 && (
        <div className="text-xs text-text-muted">Default vocabulary (no query) — common objects will be detected.</div>
      )}
      {plan.concepts.map((c) => (
        <div key={c.name} className="rounded-lg border border-border bg-bg-primary/50 p-2 flex flex-col gap-1.5">
          <div className="flex items-center gap-1.5 flex-wrap">
            <span className="text-xs font-medium">{c.name}</span>
            <Badge>{c.kind}</Badge>
            {c.verify && (
              <Badge tone="warning" title={c.question ?? undefined}>
                verify
              </Badge>
            )}
          </div>
          <div className="flex flex-wrap gap-1">
            {c.prompts.map((p) => (
              <Chip key={p} color={labelColor(p)} className="!py-0.5 !text-[11px]">
                {p}
              </Chip>
            ))}
          </div>
          {!compact && c.verify && c.question && <div className="text-[10.5px] text-text-muted italic">“{c.question}”</div>}
        </div>
      ))}
    </div>
  );
}
