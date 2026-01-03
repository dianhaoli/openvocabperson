import { useApp } from '../../context/AppContext';
import { useEntity } from '../../hooks';
import { ResultCard } from './ResultCard';

export function ResultsGrid() {
  const { entities, analysisTime } = useApp();
  const { selectedEntityId, selectEntity } = useEntity();

  return (
    <section className="flex-1 overflow-y-auto p-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold">Analysis Results</h2>
        
        {entities.length > 0 && (
          <div className="flex gap-3">
            <div className="px-3 py-1.5 bg-bg-tertiary border border-border rounded-full text-sm">
              Detections: <span className="text-accent font-semibold">{entities.length}</span>
            </div>
            {analysisTime !== null && (
              <div className="px-3 py-1.5 bg-bg-tertiary border border-border rounded-full text-sm">
                Time: <span className="text-accent font-semibold">{analysisTime}s</span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Grid */}
      <div className="grid grid-cols-[repeat(auto-fill,minmax(260px,1fr))] gap-4">
        {entities.length === 0 ? (
          <EmptyState />
        ) : (
          entities.map((entity, i) => (
            <ResultCard
              key={entity.object_id}
              entity={entity}
              isSelected={entity.object_id === selectedEntityId}
              onClick={() => selectEntity(entity)}
              animationDelay={i * 0.05}
            />
          ))
        )}
      </div>
    </section>
  );
}

function EmptyState() {
  return (
    <div className="col-span-full text-center py-12 text-text-muted">
      <p>Upload an image to start analysis</p>
    </div>
  );
}

export default ResultsGrid;

