import { useEffect, type ReactNode } from 'react';
import { cn } from '../../utils/cn';
import Button from './Button';

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
  actions?: ReactNode;
}

export function Modal({ isOpen, onClose, title, children, actions }: ModalProps) {
  // Close on escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    
    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
      document.body.style.overflow = 'hidden';
    }
    
    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.body.style.overflow = '';
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div
      className={cn(
        'fixed inset-0 z-[1000] flex items-center justify-center',
        'bg-black/70 backdrop-blur-sm'
      )}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="bg-bg-secondary border border-border rounded-[20px] p-6 max-w-[400px] w-[90%] animate-fade-in">
        <h2 className="text-lg font-semibold mb-3">{title}</h2>
        <div className="text-text-secondary text-sm mb-5">{children}</div>
        {actions && (
          <div className="flex gap-3 justify-end">{actions}</div>
        )}
      </div>
    </div>
  );
}

// Convenience component for delete confirmation
interface DeleteModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  loading?: boolean;
  itemName?: string;
  entityCount?: number;
}

export function DeleteModal({
  isOpen,
  onClose,
  onConfirm,
  loading = false,
  itemName = 'session',
  entityCount = 0,
}: DeleteModalProps) {
  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title={`Delete ${itemName}?`}
      actions={
        <>
          <Button variant="secondary" onClick={onClose}>
            Cancel
          </Button>
          <Button variant="danger" onClick={onConfirm} loading={loading}>
            Delete
          </Button>
        </>
      }
    >
      This will permanently delete this {itemName}
      {entityCount > 0 && ` and ${entityCount} detection${entityCount !== 1 ? 's' : ''}`}.
    </Modal>
  );
}

export default Modal;

