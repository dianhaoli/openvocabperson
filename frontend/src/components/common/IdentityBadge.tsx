import type { Entity } from '../../types';
import { personName, pct } from '../../utils/format';
import { Badge } from '../ui/Misc';

export function IdentityBadge({ entity, showName }: { entity: Entity; showName?: boolean }) {
  const badges = [];
  if (entity.person?.is_watchlist) {
    badges.push(
      <Badge key="w" tone="error" title="Person is on the watchlist">
        ★ watchlist
      </Badge>,
    );
  }
  switch (entity.match_status) {
    case 'matched':
      badges.push(
        <Badge key="m" tone="success" title={`Matched (${pct(entity.match_score)})`}>
          {showName && entity.person ? personName(entity.person) : 'matched'}
        </Badge>,
      );
      break;
    case 'manual':
      badges.push(
        <Badge key="m" tone="accent" title="Manually assigned">
          {showName && entity.person ? personName(entity.person) : 'manual'}
        </Badge>,
      );
      break;
    case 'new':
      badges.push(
        <Badge key="m" tone="accent" title="New identity">
          new
        </Badge>,
      );
      break;
    case 'pending':
      badges.push(
        <Badge key="m" tone="warning" title="Waiting for review">
          pending
        </Badge>,
      );
      break;
    default:
      break;
  }
  return <>{badges}</>;
}

export function VerifiedBadge({ entity }: { entity: Entity }) {
  if (entity.verified == null) return null;
  return entity.verified ? (
    <Badge tone="success" title={entity.verify_reason ?? 'Verified by VLM'}>
      ✓ verified
    </Badge>
  ) : (
    <Badge tone="error" title={entity.verify_reason ?? 'Rejected by VLM'}>
      ✗ rejected
    </Badge>
  );
}
