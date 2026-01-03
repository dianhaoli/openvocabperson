import { useEffect, useRef } from 'react';
import type { LogEntry } from '../../types';
import { cn } from '../../utils/cn';

interface ProgressLogProps {
  entries: LogEntry[];
  className?: string;
}

export function ProgressLog({ entries, className }: ProgressLogProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new entries
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [entries]);

  return (
    <div
      ref={containerRef}
      className={cn(
        'max-h-[200px] overflow-y-auto font-mono text-xs bg-bg-primary rounded-[12px] p-3',
        className
      )}
    >
      {entries.map((entry) => (
        <LogEntryRow key={entry.id} entry={entry} />
      ))}
    </div>
  );
}

function LogEntryRow({ entry }: { entry: LogEntry }) {
  const typeStyles = {
    info: 'text-accent',
    success: 'text-success',
    warning: 'text-warning',
    error: 'text-error',
  };

  const icons = {
    info: 'i',
    success: '✓',
    warning: '!',
    error: '✗',
  };

  return (
    <div className={cn('flex gap-2 py-0.5', typeStyles[entry.type])}>
      <span className="text-text-muted min-w-[45px]">{entry.time}</span>
      <span className="w-4">{icons[entry.type]}</span>
      <span>{entry.message}</span>
    </div>
  );
}

// Progress bar component
interface ProgressBarProps {
  progress?: number; // 0-100, undefined = indeterminate
  className?: string;
}

export function ProgressBar({ progress, className }: ProgressBarProps) {
  const isIndeterminate = progress === undefined;

  return (
    <div
      className={cn(
        'h-1.5 bg-bg-tertiary rounded-full overflow-hidden',
        className
      )}
    >
      <div
        className={cn(
          'h-full gradient-accent transition-all duration-300',
          isIndeterminate && 'w-[30%] animate-indeterminate'
        )}
        style={!isIndeterminate ? { width: `${progress}%` } : undefined}
      />
    </div>
  );
}

export default ProgressLog;

