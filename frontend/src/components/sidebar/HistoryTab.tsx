import { useState, useEffect, useCallback } from 'react';
import { useApp } from '../../context/AppContext';
import {
  listSessions,
  getSession,
  deleteSession,
  getSessionImageUrl,
} from '../../api/client';
import { DeleteModal, Button } from '../ui';
import { cn } from '../../utils/cn';
import type { Session, Entity } from '../../types';

export function HistoryTab() {
  const { setEntities, setCurrentImage, setActiveTab } = useApp();

  const [sessions, setSessions] = useState<Session[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(false);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);

  // Delete modal state
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<Session | null>(null);
  const [deleting, setDeleting] = useState(false);

  const limit = 20;
  const hasMore = offset + sessions.length < total;

  const loadSessions = useCallback(async (reset = false) => {
    setLoading(true);
    try {
      const newOffset = reset ? 0 : offset;
      const data = await listSessions(limit, newOffset);

      if (reset) {
        setSessions(data.sessions);
        setOffset(data.sessions.length);
      } else {
        setSessions((prev) => [...prev, ...data.sessions]);
        setOffset((prev) => prev + data.sessions.length);
      }
      setTotal(data.total);
    } catch (error) {
      console.error('Failed to load sessions:', error);
    } finally {
      setLoading(false);
    }
  }, [offset]);

  // Load on mount
  useEffect(() => {
    loadSessions(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleSessionClick = async (session: Session) => {
    try {
      setActiveSessionId(session.session_id);
      const data = await getSession(session.session_id);

      // Load image
      const img = new Image();
      img.onload = () => {
        setCurrentImage(img, getSessionImageUrl(session.session_id));

        // Convert entities
        const entities: Entity[] = data.entities.map((e, idx) => ({
          object_id: e.object_id,
          index: idx,
          class: e.class_name,
          confidence: e.confidence,
          box: e.box,
          stage: e.stage,
          analysis: e.analysis,
          crop_image: e.crop_image || '',
        }));

        setEntities(entities);
        setActiveTab('upload');
      };
      img.src = getSessionImageUrl(session.session_id);
    } catch (error) {
      console.error('Failed to load session:', error);
    }
  };

  const handleDeleteClick = (session: Session, e: React.MouseEvent) => {
    e.stopPropagation();
    setDeleteTarget(session);
    setDeleteModalOpen(true);
  };

  const handleDeleteConfirm = async () => {
    if (!deleteTarget) return;

    setDeleting(true);
    try {
      await deleteSession(deleteTarget.session_id);
      setSessions((prev) =>
        prev.filter((s) => s.session_id !== deleteTarget.session_id)
      );
      setTotal((prev) => prev - 1);
    } catch (error) {
      console.error('Failed to delete session:', error);
    } finally {
      setDeleting(false);
      setDeleteModalOpen(false);
      setDeleteTarget(null);
    }
  };

  return (
    <>
      <div className="flex justify-between items-center mb-2">
        <h3 className="text-xs font-medium text-text-muted uppercase tracking-wider">
          Past Sessions
        </h3>
        <span className="text-xs text-text-muted">
          {total} session{total !== 1 ? 's' : ''}
        </span>
      </div>

      {sessions.length === 0 && !loading ? (
        <div className="text-center py-8 text-text-muted">
          <p>No past sessions yet</p>
        </div>
      ) : (
        <div className="flex flex-col gap-3">
          {sessions.map((session) => (
            <HistoryItem
              key={session.session_id}
              session={session}
              isActive={activeSessionId === session.session_id}
              onClick={() => handleSessionClick(session)}
              onDelete={(e) => handleDeleteClick(session, e)}
            />
          ))}
        </div>
      )}

      {hasMore && (
        <Button
          variant="secondary"
          className="w-full"
          onClick={() => loadSessions(false)}
          loading={loading}
        >
          Load More
        </Button>
      )}

      <DeleteModal
        isOpen={deleteModalOpen}
        onClose={() => setDeleteModalOpen(false)}
        onConfirm={handleDeleteConfirm}
        loading={deleting}
        itemName="session"
        entityCount={deleteTarget?.entity_count || 0}
      />
    </>
  );
}

interface HistoryItemProps {
  session: Session;
  isActive: boolean;
  onClick: () => void;
  onDelete: (e: React.MouseEvent) => void;
}

function HistoryItem({ session, isActive, onClick, onDelete }: HistoryItemProps) {
  const date = new Date(session.created_at * 1000);
  const dateStr = date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });

  return (
    <div
      onClick={onClick}
      className={cn(
        'bg-bg-card border rounded-[12px] overflow-hidden cursor-pointer transition-all duration-200 group m-0.5',
        isActive
          ? 'border-accent'
          : 'border-border hover:border-border-highlight hover:translate-x-0.5'
      )}
    >
      <img
        src={getSessionImageUrl(session.session_id)}
        alt="Session thumbnail"
        className="w-full h-[100px] object-cover bg-bg-tertiary"
        onError={(e) => {
          (e.target as HTMLImageElement).src =
            "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 60'><rect fill='%231a1a24' width='100' height='60'/><text x='50' y='35' text-anchor='middle' fill='%23606070' font-size='10'>No image</text></svg>";
        }}
      />
      <div className="p-3 flex justify-between items-center">
        <div>
          <div className="text-xs text-text-secondary">{dateStr}</div>
          <div className="text-[10px] text-text-muted">
            {session.entity_count} detection
            {session.entity_count !== 1 ? 's' : ''}
          </div>
        </div>
        <button
          onClick={onDelete}
          className={cn(
            'px-2 py-1 text-xs bg-transparent border border-border rounded-md text-text-muted transition-all duration-200',
            'opacity-0 group-hover:opacity-100',
            'hover:bg-error hover:border-error hover:text-white'
          )}
        >
          Delete
        </button>
      </div>
    </div>
  );
}

export default HistoryTab;

