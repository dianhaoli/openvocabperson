import { cn } from '../../utils/cn';

export function Spinner({ size = 16, className }: { size?: number; className?: string }) {
  return (
    <span
      role="status"
      aria-label="Loading"
      className={cn('inline-block shrink-0 rounded-full border-2 border-transparent border-t-current animate-spin', className)}
      style={{ width: size, height: size }}
    />
  );
}
