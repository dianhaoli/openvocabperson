import { forwardRef, type ButtonHTMLAttributes } from 'react';
import { cn } from '../../utils/cn';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'danger' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      className,
      variant = 'primary',
      size = 'md',
      loading = false,
      disabled,
      children,
      ...props
    },
    ref
  ) => {
    const baseStyles =
      'inline-flex items-center justify-center gap-2 font-medium transition-all duration-200 rounded-[12px] font-sans disabled:opacity-50 disabled:cursor-not-allowed';

    const variants = {
      primary:
        'gradient-accent text-white hover:translate-y-[-2px] hover:shadow-[0_8px_24px_rgba(99,102,241,0.4)]',
      secondary:
        'bg-bg-tertiary border border-border text-text-secondary hover:bg-bg-card hover:text-text-primary hover:border-accent',
      danger:
        'bg-error text-white hover:bg-[#dc2626]',
      ghost:
        'bg-transparent text-text-secondary hover:text-text-primary hover:bg-bg-tertiary',
    };

    const sizes = {
      sm: 'px-3 py-1.5 text-xs',
      md: 'px-4 py-2.5 text-sm',
      lg: 'px-6 py-3 text-base',
    };

    return (
      <button
        ref={ref}
        className={cn(baseStyles, variants[variant], sizes[size], className)}
        disabled={disabled || loading}
        {...props}
      >
        {loading && <Spinner />}
        {children}
      </button>
    );
  }
);

Button.displayName = 'Button';

function Spinner() {
  return (
    <div className="w-4 h-4 border-2 border-transparent border-t-current rounded-full animate-spin" />
  );
}

export default Button;

