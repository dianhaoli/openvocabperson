import { forwardRef, type HTMLAttributes } from 'react';
import { cn } from '../../utils/cn';

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  title?: string;
}

export const Card = forwardRef<HTMLDivElement, CardProps>(
  ({ className, title, children, ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={cn(
          'bg-bg-card border border-border rounded-[12px] p-5 m-0.5',
          className
        )}
        {...props}
      >
        {title && (
          <h3 className="text-xs font-medium text-text-muted uppercase tracking-wider mb-3">
            {title}
          </h3>
        )}
        {children}
      </div>
    );
  }
);

Card.displayName = 'Card';

export default Card;

