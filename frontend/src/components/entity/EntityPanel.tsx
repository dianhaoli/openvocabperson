import { useState } from 'react';
import { useEntity } from '../../hooks';
import { QASection } from './QASection';
import { ChangeIdentityModal } from './ChangeIdentityModal';
import { formatPersonDisplayLabel } from '../../utils/personDisplay';
import { Button } from '../ui';
import { patchPerson } from '../../api/client';
import { useApp } from '../../context/AppContext';
import { cn } from '../../utils/cn';

export function EntityPanel() {
  const { selectedEntity, deselectEntity } = useEntity();
  const { updateEntityIdentity } = useApp();
  const [changeIdentityOpen, setChangeIdentityOpen] = useState(false);
  const [promoting, setPromoting] = useState(false);

  if (!selectedEntity) return null;

  const hasAnalysis = selectedEntity.analysis && selectedEntity.analysis.trim();
  const isPersonClass = selectedEntity.class === 'person';
  const personId = selectedEntity.person_id;
  const hasIdentity = isPersonClass && personId;

  const handlePromote = async () => {
    if (!personId) return;
    setPromoting(true);
    try {
      await patchPerson(personId, { is_watchlist: true });
      updateEntityIdentity(selectedEntity.object_id, { is_watchlist: true });
    } catch (e) {
      console.error(e);
    } finally {
      setPromoting(false);
    }
  };

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

        {isPersonClass && (
          <div>
            <h4 className="text-xs font-medium text-text-muted uppercase tracking-wider mb-2">
              Identity
            </h4>
            {hasIdentity ? (
              <div className="p-3 bg-bg-tertiary rounded-[12px] space-y-2">
                <div className="text-sm font-medium text-text-primary">
                  {formatPersonDisplayLabel(
                    personId!,
                    selectedEntity.person_label
                  )}
                  {selectedEntity.is_watchlist && (
                    <span className="ml-2 text-[10px] uppercase text-error">
                      Watchlist
                    </span>
                  )}
                </div>
                <div className="text-xs text-text-muted">
                  Status:{' '}
                  <span className="text-text-secondary">
                    {selectedEntity.match_status ?? '—'}
                  </span>
                  {selectedEntity.match_score != null && (
                    <>
                      {' '}
                      · Match{' '}
                      {(selectedEntity.match_score * 100).toFixed(0)}%
                    </>
                  )}
                </div>
                <div className="flex flex-wrap gap-2 pt-1">
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={() => setChangeIdentityOpen(true)}
                  >
                    Change identity
                  </Button>
                  {!selectedEntity.is_watchlist && (
                    <Button
                      size="sm"
                      onClick={handlePromote}
                      loading={promoting}
                    >
                      Promote to watchlist
                    </Button>
                  )}
                </div>
              </div>
            ) : (
              <div className="p-3 bg-bg-tertiary rounded-[12px] text-xs text-text-muted space-y-2">
                <p>No identity cluster linked (e.g. skipped Re-ID or non-VLM person).</p>
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={() => setChangeIdentityOpen(true)}
                >
                  Assign to person…
                </Button>
              </div>
            )}
          </div>
        )}

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

      <ChangeIdentityModal
        objectId={selectedEntity.object_id}
        isOpen={changeIdentityOpen}
        onClose={() => setChangeIdentityOpen(false)}
      />
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

