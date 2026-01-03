import { useRef, useEffect, useCallback } from 'react';
import { useApp } from '../../context/AppContext';
import { useEntity } from '../../hooks';
import type { Entity } from '../../types';

export function CanvasSection() {
  const { currentImage, entities } = useApp();
  const { selectedEntityId, selectEntity, deselectEntity } = useEntity();

  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const scaleRef = useRef(1);

  // Draw canvas with image and bounding boxes
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || !currentImage) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Calculate scale to fit container
    const rect = container.getBoundingClientRect();
    const maxW = rect.width - 32;
    const maxH = rect.height - 32;

    const scale = Math.min(maxW / currentImage.width, maxH / currentImage.height, 1);
    scaleRef.current = scale;

    canvas.width = currentImage.width * scale;
    canvas.height = currentImage.height * scale;

    // Draw image
    ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);

    // Draw bounding boxes
    entities.forEach((entity) => {
      const [x1, y1, x2, y2] = entity.box;
      const sx = x1 * scale;
      const sy = y1 * scale;
      const sw = (x2 - x1) * scale;
      const sh = (y2 - y1) * scale;

      const isSelected = entity.object_id === selectedEntityId;

      // Draw box
      ctx.strokeStyle = isSelected ? '#6366f1' : '#22c55e';
      ctx.lineWidth = isSelected ? 3 : 2;
      ctx.strokeRect(sx, sy, sw, sh);

      // Draw label
      const label = `${entity.class} ${(entity.confidence * 100).toFixed(0)}%`;
      ctx.font = '12px Outfit, sans-serif';
      const textWidth = ctx.measureText(label).width;

      ctx.fillStyle = isSelected ? '#6366f1' : '#22c55e';
      ctx.fillRect(sx, sy - 20, textWidth + 8, 18);

      ctx.fillStyle = 'white';
      ctx.fillText(label, sx + 4, sy - 6);
    });
  }, [currentImage, entities, selectedEntityId]);

  // Redraw on dependencies change
  useEffect(() => {
    draw();
  }, [draw]);

  // Redraw on resize
  useEffect(() => {
    const handleResize = () => {
      draw();
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [draw]);

  // Handle click on canvas
  const handleCanvasClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas || entities.length === 0) return;

      const rect = canvas.getBoundingClientRect();
      const clickX = (e.clientX - rect.left) / scaleRef.current;
      const clickY = (e.clientY - rect.top) / scaleRef.current;

      // Find all boxes containing the click point
      const hits = entities.filter((entity) => {
        const [x1, y1, x2, y2] = entity.box;
        return clickX >= x1 && clickX <= x2 && clickY >= y1 && clickY <= y2;
      });

      if (hits.length === 0) {
        deselectEntity();
        return;
      }

      // Select the smallest box (most specific)
      const smallest = hits.reduce((a: Entity, b: Entity) => {
        const areaA = (a.box[2] - a.box[0]) * (a.box[3] - a.box[1]);
        const areaB = (b.box[2] - b.box[0]) * (b.box[3] - b.box[1]);
        return areaA < areaB ? a : b;
      });

      selectEntity(smallest);
    },
    [entities, selectEntity, deselectEntity]
  );

  // Hint text
  const getHint = () => {
    if (!currentImage) return 'Upload an image to begin';
    if (entities.length === 0) return 'Click "Analyze Image" to detect entities';
    return 'Click a bounding box or card to ask questions';
  };

  return (
    <div
      ref={containerRef}
      className="bg-bg-tertiary border-b border-border p-4 flex items-center justify-center min-h-[300px] max-h-[45vh] relative"
    >
      <canvas
        ref={canvasRef}
        onClick={handleCanvasClick}
        className="max-w-full max-h-full cursor-crosshair"
      />
      <div className="absolute bottom-3 left-1/2 -translate-x-1/2 px-3 py-1.5 bg-black/70 rounded-full text-xs text-text-secondary">
        {getHint()}
      </div>
    </div>
  );
}

export default CanvasSection;

