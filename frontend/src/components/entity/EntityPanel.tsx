import { useEntity } from '../../hooks';
import { QASection } from './QASection';
import { cn } from '../../utils/cn';

export function EntityPanel() {
  const { selectedEntity, deselectEntity } = useEntity();

  if (!selectedEntity) return null;

  const hasAnalysis = selectedEntity.analysis && selectedEntity.analysis.trim();

  return (
    <aside className="w-[380px] bg-bg-secondary border-l border-border flex flex-col">
      {/* Header */}
      <div className="px-5 py-4 border-b border-border flex items-center justify-between">
        <span className="text-base font-semibold">
          Person #{selectedEntity.index}
        </span>
        <button
          onClick={deselectEntity}
          className="w-7 h-7 flex items-center justify-center bg-bg-tertiary border border-border rounded-md text-text-secondary text-lg hover:bg-error hover:border-error hover:text-white transition-colors"
        >
          ×
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-5 flex flex-col gap-4">
        {/* Crop image */}
        <img
          src={selectedEntity.crop_image}
          alt="Entity crop"
          className="w-full rounded-[12px] border border-border"
        />

        {/* Metadata grid */}
        <div className="grid grid-cols-2 gap-2">
          <MetaItem label="Class" value={selectedEntity.class} />
          <MetaItem
            label="Confidence"
            value={`${(selectedEntity.confidence * 100).toFixed(0)}%`}
          />
          <MetaItem label="Object ID" value={selectedEntity.object_id} />
          <MetaItem
            label="Stage"
            value={selectedEntity.stage.replace('_', ' ')}
          />
        </div>

        {/* Initial analysis */}
        <div>
          <h4 className="text-xs font-medium text-text-muted uppercase tracking-wider mb-2">
            Initial Analysis
          </h4>
          <div
            className={cn(
              'p-3 bg-bg-tertiary rounded-[12px] text-sm leading-relaxed',
              hasAnalysis ? 'text-text-secondary' : 'text-text-muted italic'
            )}
          >
            {hasAnalysis
              ? selectedEntity.analysis
              : selectedEntity.reason || 'No VLM analysis performed.'}
          </div>
        </div>

        {/* Q&A Section */}
        <QASection />
      </div>
    </aside>
  );
}

interface MetaItemProps {
  label: string;
  value: string;
}

function MetaItem({ label, value }: MetaItemProps) {
  return (
    <div className="p-2 bg-bg-tertiary rounded-md m-0.5">
      <div className="text-[10px] text-text-muted uppercase">{label}</div>
      <div className="text-sm font-medium">{value}</div>
    </div>
  );
}

export default EntityPanel;

