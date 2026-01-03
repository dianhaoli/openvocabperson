/**
 * Simple utility for conditionally joining class names.
 * Tailwind-friendly alternative to clsx/classnames.
 */
export function cn(...classes: (string | undefined | null | false)[]): string {
  return classes.filter(Boolean).join(' ');
}

