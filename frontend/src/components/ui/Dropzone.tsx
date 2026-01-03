import { useCallback, useState, useRef, type DragEvent, type ChangeEvent } from 'react';
import { cn } from '../../utils/cn';

interface DropzoneProps {
  onFileSelect: (file: File) => void;
  accept?: string;
  maxSize?: number; // in bytes
  preview?: string | null;
  previewAlt?: string;
  compact?: boolean;
  className?: string;
}

export function Dropzone({
  onFileSelect,
  accept = 'image/*',
  maxSize = 10 * 1024 * 1024, // 10MB
  preview,
  previewAlt = 'Preview',
  compact = false,
  className,
}: DropzoneProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = useCallback((e: DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);

      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith('image/')) {
        if (file.size <= maxSize) {
          onFileSelect(file);
        } else {
          alert(`File too large. Max size is ${formatSize(maxSize)}`);
        }
      }
    },
    [onFileSelect, maxSize]
  );

  const handleChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        if (file.size <= maxSize) {
          onFileSelect(file);
        } else {
          alert(`File too large. Max size is ${formatSize(maxSize)}`);
        }
      }
    },
    [onFileSelect, maxSize]
  );

  const handleClick = useCallback(() => {
    inputRef.current?.click();
  }, []);

  if (compact) {
    return (
      <div
        className={cn(
          'border-2 border-dashed rounded-[12px] p-4 text-center cursor-pointer transition-all duration-200',
          isDragOver
            ? 'border-accent bg-accent-soft'
            : preview
            ? 'border-success border-solid'
            : 'border-border hover:border-accent hover:bg-accent-soft',
          className
        )}
        onClick={handleClick}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <input
          ref={inputRef}
          type="file"
          accept={accept}
          onChange={handleChange}
          className="hidden"
        />
        {preview ? (
          <img
            src={preview}
            alt={previewAlt}
            className="max-w-full max-h-20 mx-auto rounded-md"
          />
        ) : (
          <span className="text-xs text-text-secondary">
            Drop image to find similar
          </span>
        )}
      </div>
    );
  }

  return (
    <div
      className={cn(
        'border-2 border-dashed rounded-[12px] p-8 text-center cursor-pointer transition-all duration-200',
        isDragOver
          ? 'border-accent bg-accent-soft'
          : 'border-border hover:border-accent hover:bg-accent-soft',
        className
      )}
      onClick={handleClick}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        onChange={handleChange}
        className="hidden"
      />
      <div className="text-sm text-text-secondary">
        <span className="text-accent font-medium">Click to upload</span> or drag
        and drop
      </div>
      <div className="text-xs text-text-muted mt-2">
        PNG, JPG up to {formatSize(maxSize)}
      </div>
    </div>
  );
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(0) + 'MB';
}

export default Dropzone;

