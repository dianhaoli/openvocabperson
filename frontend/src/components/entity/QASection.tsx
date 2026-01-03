import { useState, useRef, useEffect, type KeyboardEvent } from 'react';
import { useEntity } from '../../hooks';
import { Button } from '../ui';
import { cn } from '../../utils/cn';

export function QASection() {
  const [question, setQuestion] = useState('');
  const [useFullScene, setUseFullScene] = useState(false);
  const historyRef = useRef<HTMLDivElement>(null);
  
  const { askQuestion, isAsking, qaHistory } = useEntity();

  const handleSubmit = async () => {
    if (!question.trim() || isAsking) return;
    
    const q = question;
    setQuestion('');
    await askQuestion(q, useFullScene);
  };

  const handleKeyPress = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleSubmit();
    }
  };

  // Auto-scroll to bottom
  useEffect(() => {
    if (historyRef.current) {
      historyRef.current.scrollTop = historyRef.current.scrollHeight;
    }
  }, [qaHistory]);

  return (
    <div className="mt-auto border-t border-border pt-4">
      <h4 className="text-xs font-medium text-text-muted uppercase mb-2">
        Ask Follow-up Question
      </h4>
      
      <div className="flex gap-2">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="What is this person holding?"
          className="flex-1 px-3 py-2.5 bg-bg-tertiary border border-border rounded-lg text-sm text-text-primary outline-none focus:border-accent placeholder:text-text-muted"
        />
        <Button
          onClick={handleSubmit}
          disabled={!question.trim() || isAsking}
          loading={isAsking}
          size="md"
        >
          Ask
        </Button>
      </div>

      <label className="flex items-center gap-2 mt-3 text-xs text-text-secondary cursor-pointer hover:text-text-primary">
        <input
          type="checkbox"
          checked={useFullScene}
          onChange={(e) => setUseFullScene(e.target.checked)}
          className="w-3.5 h-3.5 accent-accent cursor-pointer"
        />
        <span>Use full scene (for context questions)</span>
      </label>

      {qaHistory.length > 0 && (
        <div
          ref={historyRef}
          className="mt-3 max-h-[250px] overflow-y-auto flex flex-col gap-2"
        >
          {qaHistory.map((item) => (
            <div key={item.id} className="p-3 bg-bg-tertiary rounded-lg m-0.5">
              <div className="text-xs text-accent mb-1">Q: {item.question}</div>
              <div
                className={cn(
                  'text-sm leading-relaxed',
                  item.loading
                    ? 'text-text-muted italic'
                    : item.answer?.startsWith('Error:')
                    ? 'text-error'
                    : 'text-text-secondary'
                )}
              >
                {item.loading ? (
                  <span className="flex items-center gap-2">
                    <span className="w-4 h-4 border-2 border-transparent border-t-current rounded-full animate-spin" />
                    Thinking...
                  </span>
                ) : (
                  item.answer
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default QASection;

