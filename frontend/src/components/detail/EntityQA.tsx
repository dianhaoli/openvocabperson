import { useEffect, useRef, useState } from 'react';
import { api } from '../../api/client';
import { useActions } from '../../context/hooks';
import type { QAItem } from '../../types';
import { Button } from '../ui/Button';
import { Markdown } from '../ui/Markdown';
import { Segmented, inputClass } from '../ui/Misc';

type Scope = 'crop' | 'scene';

interface Props {
  entityId: string;
  history: QAItem[];
  onAnswered: (item: QAItem) => void;
}

export function EntityQA({ entityId, history, onAnswered }: Props) {
  const { toastError } = useActions();
  const [scope, setScope] = useState<Scope>('crop');
  const [question, setQuestion] = useState('');
  const [busy, setBusy] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [history.length]);

  const ask = async () => {
    const q = question.trim();
    if (!q) return;
    setBusy(true);
    try {
      const res = await api.askEntity(entityId, q, scope);
      onAnswered({ question: q, answer: res.answer, created_at: Date.now() / 1000 });
      setQuestion('');
    } catch (err) {
      toastError(err, 'Question failed');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-col gap-2">
      {history.length > 0 && (
        <div ref={listRef} className="flex flex-col gap-2 max-h-[320px] overflow-y-auto pr-1 [&>*]:shrink-0">
          {history.map((h, i) => (
            <div key={`${h.created_at}-${i}`} className="flex flex-col gap-1">
              <div className="self-end max-w-[90%] rounded-xl rounded-br-sm bg-accent text-white text-xs px-2.5 py-1.5 whitespace-pre-wrap break-words">
                {h.question}
              </div>
              <div className="self-start max-w-[95%] rounded-xl rounded-bl-sm bg-bg-card border border-border text-xs px-2.5 py-1.5 text-text-secondary">
                <Markdown text={h.answer} />
              </div>
            </div>
          ))}
        </div>
      )}
      <div className="flex items-center gap-2">
        <span className="text-[11px] text-text-muted">Look at</span>
        <Segmented
          value={scope}
          options={[
            { value: 'crop', label: 'Crop', title: 'Ask about the cropped entity only' },
            { value: 'scene', label: 'Scene', title: 'Ask with the full scene as context' },
          ]}
          onChange={setScope}
          size="xs"
        />
      </div>
      <form
        className="flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          void ask();
        }}
      >
        <input className={inputClass} value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="What is this person holding?" />
        <Button type="submit" size="sm" loading={busy} disabled={!question.trim()}>
          Ask
        </Button>
      </form>
    </div>
  );
}
