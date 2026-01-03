import { useState, useCallback } from 'react';
import { useApp } from '../context/AppContext';
import { askQuestion } from '../api/client';
import type { QAItem, Entity } from '../types';

/**
 * Hook for managing entity selection and Q&A.
 */
export function useEntity() {
  const { entities, selectedEntityId, selectEntity } = useApp();
  const [qaHistory, setQaHistory] = useState<Map<string, QAItem[]>>(new Map());
  const [isAsking, setIsAsking] = useState(false);

  const selectedEntity = entities.find((e) => e.object_id === selectedEntityId);

  const currentQA = selectedEntityId
    ? qaHistory.get(selectedEntityId) || []
    : [];

  const handleSelectEntity = useCallback(
    (entity: Entity | string | null) => {
      const id = typeof entity === 'string' ? entity : entity?.object_id ?? null;
      selectEntity(id);
    },
    [selectEntity]
  );

  const handleDeselectEntity = useCallback(() => {
    selectEntity(null);
  }, [selectEntity]);

  const handleAskQuestion = useCallback(
    async (question: string, useFullScene = false) => {
      if (!selectedEntityId || !question.trim() || isAsking) return;

      setIsAsking(true);

      const qaItem: QAItem = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
        question,
        answer: null,
        loading: true,
      };

      // Add loading item
      setQaHistory((prev) => {
        const entityHistory = prev.get(selectedEntityId) || [];
        const updated = new Map(prev);
        updated.set(selectedEntityId, [...entityHistory, qaItem]);
        return updated;
      });

      try {
        const response = await askQuestion(selectedEntityId, question, useFullScene);

        // Update with answer
        setQaHistory((prev) => {
          const entityHistory = prev.get(selectedEntityId) || [];
          const updated = new Map(prev);
          updated.set(
            selectedEntityId,
            entityHistory.map((item) =>
              item.id === qaItem.id
                ? { ...item, answer: response.answer, loading: false }
                : item
            )
          );
          return updated;
        });
      } catch (error) {
        // Update with error
        setQaHistory((prev) => {
          const entityHistory = prev.get(selectedEntityId) || [];
          const updated = new Map(prev);
          updated.set(
            selectedEntityId,
            entityHistory.map((item) =>
              item.id === qaItem.id
                ? {
                    ...item,
                    answer: `Error: ${error instanceof Error ? error.message : 'Failed'}`,
                    loading: false,
                  }
                : item
            )
          );
          return updated;
        });
      } finally {
        setIsAsking(false);
      }
    },
    [selectedEntityId, isAsking]
  );

  const clearEntityQA = useCallback(
    (entityId: string) => {
      setQaHistory((prev) => {
        const updated = new Map(prev);
        updated.delete(entityId);
        return updated;
      });
    },
    []
  );

  return {
    selectedEntity,
    selectedEntityId,
    selectEntity: handleSelectEntity,
    deselectEntity: handleDeselectEntity,
    askQuestion: handleAskQuestion,
    isAsking,
    qaHistory: currentQA,
    clearEntityQA,
  };
}

export default useEntity;

