import type { ButtonHTMLAttributes } from 'react';
import { cn } from '../../utils/cn';
import { Spinner } from './Spinner';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'danger' | 'ghost';
  size?: 'xs' | 'sm' | 'md' | 'lg';
  loading?: boolean;
}

const variants = {
  primary: 'gradient-accent text-white hover:-translate-y-px hover:shadow-[0_8px_24px_rgba(99,102,241,0.35)]',
  secondary:
    'bg-bg-tertiary border border-border text-text-secondary hover:bg-bg-card hover:text-text-primary hover:border-accent',
  danger: 'bg-error/90 text-white hover:bg-error',
  ghost: 'bg-transparent text-text-secondary hover:text-text-primary hover:bg-bg-tertiary',
};

const sizes = {
  xs: 'px-2 py-1 text-[11px] rounded-lg',
  sm: 'px-3 py-1.5 text-xs rounded-[10px]',
  md: 'px-4 py-2 text-sm rounded-[12px]',
  lg: 'px-6 py-3 text-base rounded-[12px]',
};

export function Button({
  className,
  variant = 'primary',
  size = 'md',
  loading = false,
  disabled,
  children,
  type = 'button',
  ...props
}: ButtonProps) {
  return (
    <button
      type={type}
      className={cn(
        'inline-flex items-center justify-center gap-2 font-medium transition-all duration-150 font-sans whitespace-nowrap',
        'disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:translate-y-0 disabled:hover:shadow-none',
        variants[variant],
        sizes[size],
        className,
      )}
      disabled={disabled || loading}
      {...props}
    >
      {loading && <Spinner size={14} />}
      {children}
    </button>
  );
}
