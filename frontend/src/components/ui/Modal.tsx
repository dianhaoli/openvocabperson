import { useEffect, useRef, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { cn } from '../../utils/cn';
import { Button } from './Button';

// Stack of open modals so Escape only closes the top-most one.
const modalStack: symbol[] = [];

interface ModalProps {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  children: ReactNode;
  actions?: ReactNode;
  className?: string;
}

export function Modal({ open, onClose, title, children, actions, className }: ModalProps) {
  const onCloseRef = useRef(onClose);
  useEffect(() => {
    onCloseRef.current = onClose;
  });

  useEffect(() => {
    if (!open) return;
    const token = Symbol('modal');
    modalStack.push(token);
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && modalStack[modalStack.length - 1] === token) {
        e.stopPropagation();
        onCloseRef.current();
      }
    };
    document.addEventListener('keydown', handler);
    return () => {
      document.removeEventListener('keydown', handler);
      const idx = modalStack.indexOf(token);
      if (idx !== -1) modalStack.splice(idx, 1);
    };
  }, [open]);

  if (!open) return null;

  return createPortal(
    <div
      className="fixed inset-0 z-[1000] flex items-center justify-center bg-black/70 backdrop-blur-sm p-4"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        className={cn(
          'bg-bg-secondary border border-border rounded-[20px] w-full max-w-[420px] max-h-[90vh] flex flex-col animate-fade-in shadow-2xl',
          className,
        )}
      >
        <div className="flex items-center justify-between gap-3 px-5 pt-4 pb-3 border-b border-border">
          <h2 className="text-base font-semibold truncate">{title}</h2>
          <button
            type="button"
            onClick={onClose}
            className="text-text-muted hover:text-text-primary text-xl leading-none px-1"
            aria-label="Close"
          >
            ×
          </button>
        </div>
        <div className="px-5 py-4 overflow-y-auto min-h-0 flex-1 text-sm">{children}</div>
        {actions && <div className="flex gap-2 justify-end px-5 pb-4 pt-2 border-t border-border">{actions}</div>}
      </div>
    </div>,
    document.body,
  );
}

interface ConfirmProps {
  open: boolean;
  title: string;
  message: ReactNode;
  confirmLabel?: string;
  loading?: boolean;
  onConfirm: () => void;
  onClose: () => void;
}

export function ConfirmModal({ open, title, message, confirmLabel = 'Delete', loading, onConfirm, onClose }: ConfirmProps) {
  return (
    <Modal
      open={open}
      onClose={onClose}
      title={title}
      actions={
        <>
          <Button variant="secondary" size="sm" onClick={onClose}>
            Cancel
          </Button>
          <Button variant="danger" size="sm" onClick={onConfirm} loading={loading}>
            {confirmLabel}
          </Button>
        </>
      }
    >
      <div className="text-text-secondary">{message}</div>
    </Modal>
  );
}
