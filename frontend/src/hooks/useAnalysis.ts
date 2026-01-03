import { useCallback } from 'react';
import { useApp } from '../context/AppContext';
import { analyzeImage, analyzeImageStream } from '../api/client';
import type { Entity } from '../types';

/**
 * Hook for managing image analysis with streaming support.
 */
export function useAnalysis() {
  const {
    selectedFile,
    isAnalyzing,
    setIsAnalyzing,
    setEntities,
    setSessionId,
    setAnalysisTime,
    addLog,
    clearLogs,
    selectEntity,
  } = useApp();

  const startAnalysis = useCallback(async () => {
    if (!selectedFile || isAnalyzing) return;

    setIsAnalyzing(true);
    clearLogs();
    setEntities([]);
    selectEntity(null);
    setAnalysisTime(null);

    addLog('info', 'Starting analysis...');

    try {
      // Try streaming first
      let results: Entity[] = [];
      let sessionId: string | null = null;
      let elapsed = 0;

      try {
        for await (const event of analyzeImageStream(selectedFile)) {
          switch (event.type) {
            case 'detection_complete':
              addLog('success', `Detected ${event.data?.num_detections || 0} objects`);
              break;

            case 'routing_complete':
              addLog(
                'info',
                `Routing: ${event.data?.vlm_full || 0} VLM, ${event.data?.yolo_only || 0} skip, ${event.data?.low_confidence || 0} low conf`
              );
              break;

            case 'crop_analyzed':
              if (event.data?.analysis) {
                const preview =
                  event.data.analysis.length > 50
                    ? event.data.analysis.substring(0, 50) + '...'
                    : event.data.analysis;
                addLog('success', `Crop ${event.data.index ?? '?'}: ${preview}`);
              } else {
                addLog(
                  'warning',
                  `Crop ${event.data?.index ?? '?'}: ${event.data?.reason || 'skipped'}`
                );
              }
              break;

            case 'complete':
              addLog('success', `Complete in ${event.elapsed}s`);
              results = event.results || [];
              sessionId = event.session_id || null;
              elapsed = event.elapsed;
              break;

            case 'error':
              addLog('error', event.message || 'Unknown error');
              break;
          }
        }
      } catch {
        // Fallback to sync API
        addLog('warning', 'Streaming unavailable, using sync mode');
        const data = await analyzeImage(selectedFile);
        results = data.results;
        sessionId = data.session_id;
        elapsed = data.elapsed_seconds;
        addLog('success', `Detected ${data.num_detections} entities`);
      }

      setEntities(results);
      setSessionId(sessionId);
      setAnalysisTime(elapsed);
    } catch (error) {
      addLog('error', `Analysis failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsAnalyzing(false);
    }
  }, [
    selectedFile,
    isAnalyzing,
    setIsAnalyzing,
    setEntities,
    setSessionId,
    setAnalysisTime,
    addLog,
    clearLogs,
    selectEntity,
  ]);

  return {
    startAnalysis,
    isAnalyzing,
    canAnalyze: !!selectedFile && !isAnalyzing,
  };
}

export default useAnalysis;

