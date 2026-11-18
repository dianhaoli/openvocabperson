import { useRef, useState, type ReactNode } from 'react';
import { cn } from '../../utils/cn';

interface DropzoneProps {
  accept: string;
  multiple?: boolean;
  onFiles: (files: File[]) => void;
  children?: ReactNode;
  className?: string;
  compact?: boolean;
}

function matchesAccept(file: File, accept: string): boolean {
  const parts = accept.split(',').map((p) => p.trim()).filter(Boolean);
  if (parts.length === 0) return true;
  return parts.some((p) => {
    if (p.endsWith('/*')) return file.type.startsWith(p.slice(0, -1));
    if (p.startsWith('.')) return file.name.toLowerCase().endsWith(p.toLowerCase());
    return file.type === p;
  });
}

export function Dropzone({ accept, multiple, onFiles, children, className, compact }: DropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  const handle = (list: FileList | null) => {
    if (!list) return;
    const files = Array.from(list).filter((f) => matchesAccept(f, accept));
    if (files.length) onFiles(multiple ? files : files.slice(0, 1));
  };

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => inputRef.current?.click()}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          inputRef.current?.click();
        }
      }}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        handle(e.dataTransfer.files);
      }}
      className={cn(
        'border-2 border-dashed rounded-[14px] text-center cursor-pointer transition-colors select-none',
        compact ? 'px-3 py-3' : 'px-4 py-6',
        over ? 'border-accent bg-accent-soft' : 'border-border hover:border-accent-border bg-bg-primary/40',
        className,
      )}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        multiple={multiple}
        className="hidden"
        onChange={(e) => {
          handle(e.target.files);
          e.target.value = '';
        }}
      />
      {children}
    </div>
  );
}
