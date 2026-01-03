import { useCallback } from 'react';
import { useApp } from '../../context/AppContext';
import { useAnalysis } from '../../hooks';
import { Card, Button, Dropzone, ProgressLog, ProgressBar } from '../ui';

export function UploadTab() {
  const {
    selectedFile,
    setSelectedFile,
    setCurrentImage,
    isAnalyzing,
    analysisLogs,
    isPipelineReady,
  } = useApp();
  
  const { startAnalysis, canAnalyze } = useAnalysis();

  const handleFileSelect = useCallback(
    (file: File) => {
      setSelectedFile(file);

      // Load into image element
      const reader = new FileReader();
      reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
          setCurrentImage(img, e.target?.result as string);
        };
        img.src = e.target?.result as string;
      };
      reader.readAsDataURL(file);
    },
    [setSelectedFile, setCurrentImage]
  );

  return (
    <>
      <Card title="Upload Image">
        <div className="mb-1">
          <Dropzone onFileSelect={handleFileSelect} />
        </div>

        {selectedFile && (
          <div className="mt-4">
            <div className="text-text-secondary text-sm flex justify-between">
              <span className="truncate">{selectedFile.name}</span>
              <span>{formatFileSize(selectedFile.size)}</span>
            </div>
          </div>
        )}
      </Card>

      <Button
        variant="primary"
        size="lg"
        className="w-full"
        onClick={startAnalysis}
        disabled={!canAnalyze || !isPipelineReady}
        loading={isAnalyzing}
      >
        {isAnalyzing ? 'Analyzing...' : 'Analyze Image'}
      </Button>

      {(isAnalyzing || analysisLogs.length > 0) && (
        <Card title="Progress">
          <ProgressBar progress={isAnalyzing ? undefined : 100} className="mb-4" />
          <div className="mt-1">
            <ProgressLog entries={analysisLogs} />
          </div>
        </Card>
      )}
    </>
  );
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

export default UploadTab;

