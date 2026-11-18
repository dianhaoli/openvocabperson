import { useEffect, useState, type RefObject } from 'react';

export interface Size {
  width: number;
  height: number;
}

/** Tracks the content-box size of an element with ResizeObserver. */
export function useElementSize(ref: RefObject<HTMLElement | null>): Size {
  const [size, setSize] = useState<Size>({ width: 0, height: 0 });
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const r = entries[0]?.contentRect;
      if (!r) return;
      setSize((prev) =>
        Math.abs(prev.width - r.width) < 0.5 && Math.abs(prev.height - r.height) < 0.5
          ? prev
          : { width: r.width, height: r.height },
      );
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref]);
  return size;
}
