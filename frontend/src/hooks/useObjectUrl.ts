import { useEffect, useState } from 'react';

/** Object URL for a File/Blob, revoked automatically when it changes or on unmount. */
export function useObjectUrl(blob: Blob | null | undefined): string | null {
  const [entry, setEntry] = useState<{ blob: Blob; url: string } | null>(null);

  useEffect(() => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    // eslint-disable-next-line react-hooks/set-state-in-effect -- URL must be created and revoked as a side effect
    setEntry({ blob, url });
    return () => URL.revokeObjectURL(url);
  }, [blob]);

  return blob && entry && entry.blob === blob ? entry.url : null;
}
