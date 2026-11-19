import { useEffect, useReducer, useRef, useState } from 'react';
import { api, errorMessage, isAbortError } from '../../api/client';
import { useActions, useAppState } from '../../context/hooks';
import type { AgentToolResult } from '../../types';
import { cn } from '../../utils/cn';
import { Button } from '../ui/Button';
import { Markdown } from '../ui/Markdown';
import { EmptyState } from '../ui/Misc';
import { Spinner } from '../ui/Spinner';
import { ToolCard } from './ToolCard';

type ChatItem =
  | { kind: 'user'; id: string; text: string }
  | { kind: 'assistant'; id: string; text: string }
  | { kind: 'tool'; id: string; callId: string; name: string; args: unknown; result: AgentToolResult | null }
  | { kind: 'error'; id: string; text: string };

type ChatAction =
  | { type: 'add'; item: ChatItem }
  | { type: 'toolResult'; result: AgentToolResult; id: string }
  | { type: 'reset' };

function chatReducer(items: ChatItem[], action: ChatAction): ChatItem[] {
  switch (action.type) {
    case 'add':
      return [...items, action.item];
    case 'toolResult': {
      for (let i = items.length - 1; i >= 0; i--) {
        const it = items[i];
        if (it.kind === 'tool' && it.callId === action.result.id && !it.result) {
          const next = items.slice();
          next[i] = { ...it, result: action.result };
          return next;
        }
      }
      return [
        ...items,
        { kind: 'tool', id: action.id, callId: action.result.id, name: action.result.name, args: null, result: action.result },
      ];
    }
    case 'reset':
      return [];
    default:
      return items;
  }
}

const EXAMPLES = [
  'Find all dangerous objects in my library',
  'Who appears in the most photos?',
  'Cluster people and list the groups',
  'Find people wearing red',
];

let itemSeq = 0;
const nextId = () => `c${++itemSeq}`;

export function AgentTab() {
  const { health } = useAppState();
  const { toastError, bumpPersons, bumpLibrary } = useActions();
  const [items, dispatch] = useReducer(chatReducer, []);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [input, setInput] = useState('');
  const [streaming, setStreaming] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => () => controllerRef.current?.abort(), []);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [items, streaming]);

  const send = async (text: string) => {
    const message = text.trim();
    if (!message || streaming) return;
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    dispatch({ type: 'add', item: { kind: 'user', id: nextId(), text: message } });
    setInput('');
    setStreaming(true);
    let mutated = false;
    try {
      const stream = api.agentChat({ message, conversation_id: conversationId ?? undefined }, controller.signal);
      for await (const evt of stream) {
        switch (evt.type) {
          case 'conversation':
            setConversationId(evt.conversation_id);
            break;
          case 'tool_call':
            dispatch({
              type: 'add',
              item: { kind: 'tool', id: nextId(), callId: evt.id, name: evt.name, args: evt.arguments, result: null },
            });
            if (/name_person|set_watchlist|merge_persons|recluster/.test(evt.name)) mutated = true;
            break;
          case 'tool_result': {
            const { type: _type, ...result } = evt;
            void _type;
            dispatch({ type: 'toolResult', result, id: nextId() });
            break;
          }
          case 'message':
            if (evt.text) dispatch({ type: 'add', item: { kind: 'assistant', id: nextId(), text: evt.text } });
            break;
          case 'error':
            dispatch({ type: 'add', item: { kind: 'error', id: nextId(), text: evt.message } });
            break;
          case 'done':
            break;
        }
        if (evt.type === 'done') break;
      }
    } catch (err) {
      if (!isAbortError(err)) {
        dispatch({ type: 'add', item: { kind: 'error', id: nextId(), text: errorMessage(err) } });
        toastError(err, 'Agent');
      }
    } finally {
      if (controllerRef.current === controller) controllerRef.current = null;
      setStreaming(false);
      if (mutated) bumpPersons();
      bumpLibrary();
    }
  };

  const stop = () => {
    controllerRef.current?.abort();
    controllerRef.current = null;
    setStreaming(false);
  };

  const newChat = () => {
    stop();
    dispatch({ type: 'reset' });
    setConversationId(null);
  };

  const agentAvailable = health?.capabilities?.agent ?? true;

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="flex items-center justify-between px-4 py-2 border-b border-border">
        <div className="text-[11px] text-text-muted font-mono truncate">
          {conversationId ? `conversation ${conversationId}` : 'new conversation'}
        </div>
        <Button size="xs" variant="secondary" onClick={newChat} disabled={items.length === 0 && !conversationId}>
          + New chat
        </Button>
      </div>

      <div ref={scrollRef} className="flex-1 min-h-0 overflow-y-auto px-3 py-3 flex flex-col gap-2.5 [&>*]:shrink-0">
        {items.length === 0 && (
          <EmptyState icon="✦" title="Ask the vision agent" hint="It can search your library, scan for concepts, manage people and answer questions about images.">
            {!agentAvailable && <div className="text-warning text-xs">The agent capability is unavailable on the backend (no LLM configured).</div>}
            <div className="flex flex-col gap-1.5 mt-2 w-full">
              {EXAMPLES.map((ex) => (
                <button
                  key={ex}
                  type="button"
                  onClick={() => void send(ex)}
                  className="text-left text-xs px-3 py-2 rounded-lg border border-border bg-bg-card hover:border-accent text-text-secondary hover:text-text-primary transition-colors"
                >
                  {ex}
                </button>
              ))}
            </div>
          </EmptyState>
        )}
        {items.map((it) => {
          switch (it.kind) {
            case 'user':
              return (
                <div key={it.id} className="self-end max-w-[88%] rounded-2xl rounded-br-md px-3 py-2 text-sm bg-accent text-white whitespace-pre-wrap break-words">
                  {it.text}
                </div>
              );
            case 'assistant':
              return (
                <div key={it.id} className="self-start max-w-[95%] rounded-2xl rounded-bl-md px-3 py-2 text-sm bg-bg-card border border-border text-text-secondary">
                  <Markdown text={it.text} />
                </div>
              );
            case 'error':
              return (
                <div key={it.id} className="self-start max-w-[95%] rounded-xl px-3 py-2 text-xs bg-error/10 border border-error/40 text-error">
                  {it.text}
                </div>
              );
            case 'tool':
              return <ToolCard key={it.id} name={it.name} args={it.args} result={it.result} />;
            default:
              return null;
          }
        })}
        {streaming && (
          <div className="self-start flex items-center gap-2 text-xs text-text-muted px-1">
            <Spinner size={12} /> thinking…
          </div>
        )}
      </div>

      <form
        className="border-t border-border p-3 flex gap-2 items-end"
        onSubmit={(e) => {
          e.preventDefault();
          void send(input);
        }}
      >
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              void send(input);
            }
          }}
          rows={Math.min(5, Math.max(1, input.split('\n').length))}
          placeholder="Ask about your library…"
          className={cn(
            'flex-1 resize-none bg-bg-primary border border-border rounded-[12px] px-3 py-2 text-sm',
            'placeholder:text-text-muted focus:outline-none focus:border-accent',
          )}
        />
        {streaming ? (
          <Button variant="secondary" size="md" onClick={stop}>
            Stop
          </Button>
        ) : (
          <Button type="submit" size="md" disabled={!input.trim()}>
            Send
          </Button>
        )}
      </form>
    </div>
  );
}
