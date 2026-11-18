import { useState, type ImgHTMLAttributes } from 'react';
import { cn } from '../../utils/cn';

/** <img> with lazy loading and a neutral placeholder on error. */
export function Img({ className, src, alt = '', ...props }: ImgHTMLAttributes<HTMLImageElement>) {
  const [failedSrc, setFailedSrc] = useState<string | null>(null);
  if (!src || failedSrc === src) {
    return (
      <div className={cn('flex items-center justify-center bg-bg-primary text-text-muted text-xs', className)} aria-label={alt}>
        ∅
      </div>
    );
  }
  return (
    <img
      loading="lazy"
      decoding="async"
      src={src}
      alt={alt}
      className={cn('bg-bg-primary', className)}
      onError={() => setFailedSrc(src)}
      {...props}
    />
  );
}
