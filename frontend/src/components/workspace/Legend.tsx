import { labelColor } from '../../utils/colors';
import { Chip } from '../ui/Misc';

interface Props {
  counts: [string, number][];
  hidden: Set<string>;
  onToggle: (label: string) => void;
  onShowOnly: (label: string | null) => void;
  rejectedCount: number;
  showRejected: boolean;
  onToggleRejected: () => void;
}

export function Legend({ counts, hidden, onToggle, onShowOnly, rejectedCount, showRejected, onToggleRejected }: Props) {
  if (counts.length === 0 && rejectedCount === 0) return null;
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {counts.map(([label, n]) => (
        <Chip
          key={label}
          color={labelColor(label)}
          active={!hidden.has(label)}
          onClick={() => onToggle(label)}
          title="Click to toggle · double-click to show only this label"
          className={hidden.has(label) ? 'opacity-50 line-through' : ''}
        >
          <span onDoubleClick={() => onShowOnly(label)}>
            {label} <span className="text-text-muted">{n}</span>
          </span>
        </Chip>
      ))}
      {hidden.size > 0 && (
        <button type="button" className="text-[11px] text-accent px-1" onClick={() => onShowOnly(null)}>
          show all
        </button>
      )}
      {rejectedCount > 0 && (
        <label className="ml-auto flex items-center gap-1.5 text-[11px] text-text-secondary cursor-pointer">
          <input type="checkbox" checked={showRejected} onChange={onToggleRejected} className="accent-[#6366f1]" />
          Show rejected ({rejectedCount})
        </label>
      )}
    </div>
  );
}
