import { cn } from '../../utils/cn';
import type { Entity } from '../../types';

interface ResultCardProps {
  entity: Entity;
  isSelected: boolean;
  onClick: () => void;
  animationDelay?: number;
}

export function ResultCard({
  entity,
  isSelected,
  onClick,
  animationDelay = 0,
}: ResultCardProps) {
  const hasAnalysis = entity.analysis && entity.analysis.trim();

  const stageStyles = {
    vlm_full: 'bg-success-soft text-success border border-success',
    yolo_only: 'bg-warning-soft text-warning border border-warning',
    low_confidence: 'bg-[rgba(100,100,120,0.3)] text-text-muted border border-text-muted',
  };

  return (
    <div
      onClick={onClick}
      className={cn(
        'bg-bg-card border rounded-[20px] overflow-hidden transition-all duration-200 cursor-pointer animate-fade-in m-1',
        isSelected
          ? 'border-accent shadow-[0_0_0_2px_rgba(99,102,241,0.15)]'
          : 'border-border hover:border-border-highlight hover:-translate-y-0.5 hover:shadow-DEFAULT'
      )}
      style={{ animationDelay: `${animationDelay}s` }}
    >
      {/* Image container */}
      <div className="relative aspect-[4/3] bg-bg-secondary">
        <img
          src={entity.crop_image}
          alt={`Crop ${entity.index}`}
          className="w-full h-full object-cover"
        />
        
        {/* Top-left badges */}
        <div className="absolute top-2 left-2 flex gap-1.5">
          <span className="px-2 py-0.5 text-[11px] font-medium rounded-[5px] backdrop-blur-lg bg-accent/85 text-white">
            {entity.class}
          </span>
          <span className="px-2 py-0.5 text-[11px] font-medium rounded-[5px] backdrop-blur-lg bg-black/60 text-white">
            {(entity.confidence * 100).toFixed(0)}%
          </span>
        </div>
        
        {/* Top-right stage badge */}
        <span
          className={cn(
            'absolute top-2 right-2 px-2 py-0.5 text-[10px] font-medium rounded-[5px] uppercase tracking-wide',
            stageStyles[entity.stage]
          )}
        >
          {entity.stage.replace('_', ' ')}
        </span>
      </div>

      {/* Content */}
      <div className="p-3 pt-2">
        <div className="text-xs text-text-muted mb-1">
          Detection #{entity.index} • ID: {entity.object_id}
        </div>
        <div
          className={cn(
            'text-sm leading-relaxed line-clamp-3',
            hasAnalysis ? 'text-text-secondary' : 'text-text-muted italic'
          )}
        >
          {hasAnalysis ? entity.analysis : entity.reason || 'No analysis'}
        </div>
        <div className="mt-2 text-xs text-accent">
          Click to ask questions →
        </div>
      </div>
    </div>
  );
}

export default ResultCard;

