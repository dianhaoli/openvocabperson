import { useState } from 'react';
import { api } from '../../api/client';
import { useActions } from '../../context/hooks';
import { Markdown } from '../ui/Markdown';
import { Button } from '../ui/Button';
import { inputClass } from '../ui/Misc';

interface QA {
  question: string;
  answer: string;
  elapsed: number;
}

export function SceneQA({ mediaId, kind }: { mediaId: string; kind: 'image' | 'video' }) {
  const { toastError } = useActions();
  const [question, setQuestion] = useState('');
  const [history, setHistory] = useState<QA[]>([]);
  const [busy, setBusy] = useState(false);

  const ask = async () => {
    const q = question.trim();
    if (!q) return;
    setBusy(true);
    try {
      const res = await api.askMedia(mediaId, q);
      setHistory((h) => [...h, { question: q, answer: res.answer, elapsed: res.elapsed }]);
      setQuestion('');
    } catch (err) {
      toastError(err, 'Question failed');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="rounded-xl border border-border bg-bg-card p-3 flex flex-col gap-2">
      <div className="text-[11px] uppercase tracking-wider text-text-muted font-semibold">Ask about this {kind}</div>
      {history.map((h, i) => (
        <div key={i} className="flex flex-col gap-1 text-sm">
          <div className="text-text-primary">Q: {h.question}</div>
          <div className="text-text-secondary pl-3 border-l-2 border-accent-border">
            <Markdown text={h.answer} />
            <div className="text-[10px] text-text-muted mt-0.5">{h.elapsed.toFixed(1)}s</div>
          </div>
        </div>
      ))}
      <form
        className="flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          void ask();
        }}
      >
        <input
          className={inputClass}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder={kind === 'video' ? 'What happens in this video?' : 'What is going on in this scene?'}
        />
        <Button type="submit" size="sm" loading={busy} disabled={!question.trim()}>
          Ask
        </Button>
      </form>
    </div>
  );
}
