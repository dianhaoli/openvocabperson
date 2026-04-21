import type { KeyboardEvent } from 'react';
import { useSearch } from '../../hooks';
import { useApp } from '../../context/AppContext';
import { getSession, getSessionImageUrl } from '../../api/client';
import { Card, Button, Dropzone } from '../ui';
import { cn } from '../../utils/cn';
import type { Entity, SearchResult } from '../../types';

export function SearchTab() {
  const {
    textQuery,
    searchImagePreview,
    textWeight,
    vectorWeight,
    results,
    isSearching,
    hasSearched,
    canSearch,
    showWeights,
    setTextQuery,
    handleImageSelect,
    handleTextWeightChange,
    performSearch,
    clearSearch,
  } = useSearch();

  const { setEntities, setCurrentImage, setActiveTab, selectEntity } = useApp();

  const handleKeyPress = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && canSearch) {
      performSearch();
    }
  };

  const handleResultClick = async (result: SearchResult) => {
    try {
      const session = await getSession(result.session_id);
      
      // Load image
      const img = new Image();
      img.onload = () => {
        setCurrentImage(img, getSessionImageUrl(result.session_id));
        
        // Convert entities
        const entities: Entity[] = session.entities.map((e, idx) => ({
          object_id: e.object_id,
          index: idx,
          class: e.class_name,
          confidence: e.confidence,
          box: e.box,
          stage: e.stage,
          analysis: e.analysis,
          crop_image: e.crop_image || '',
          person_id: e.person_id,
          person_label: e.person_label,
          is_watchlist: e.is_watchlist,
          match_score: e.match_score,
          match_status: e.match_status,
        }));
        
        setEntities(entities);
        setActiveTab('upload');
        
        // Select the specific result
        setTimeout(() => selectEntity(result.object_id), 100);
      };
      img.src = getSessionImageUrl(result.session_id);
    } catch (error) {
      console.error('Failed to load session:', error);
    }
  };

  return (
    <>
      <Card title="Search Entities">
        {/* Text search input */}
        <input
          type="text"
          value={textQuery}
          onChange={(e) => setTextQuery(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Search by description..."
          className="w-full px-4 py-3 bg-bg-tertiary border border-border rounded-[12px] text-text-primary text-sm outline-none focus:border-accent placeholder:text-text-muted mb-1"
        />

        <div className="flex items-center gap-3 my-3 text-text-muted text-xs uppercase">
          <div className="flex-1 h-px bg-border" />
          <span>and/or</span>
          <div className="flex-1 h-px bg-border" />
        </div>

        {/* Image similarity search */}
        <div className="mt-1">
          <Dropzone
            compact
            onFileSelect={handleImageSelect}
            preview={searchImagePreview}
            previewAlt="Search image"
          />
        </div>

        {/* Weight sliders */}
        {showWeights && (
          <div className="flex gap-2 mt-4">
            <div className="flex-1">
              <div className="flex justify-between text-xs text-text-muted mb-1">
                <span>Text</span>
                <span>{textWeight}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                value={textWeight}
                onChange={(e) => handleTextWeightChange(Number(e.target.value))}
                className="w-full accent-accent"
              />
            </div>
            <div className="flex-1">
              <div className="flex justify-between text-xs text-text-muted mb-1">
                <span>Image</span>
                <span>{vectorWeight}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                value={vectorWeight}
                onChange={(e) => handleTextWeightChange(100 - Number(e.target.value))}
                className="w-full accent-accent"
              />
            </div>
          </div>
        )}

        <Button
          className="w-full mt-4"
          onClick={performSearch}
          disabled={!canSearch}
          loading={isSearching}
        >
          Search
        </Button>
      </Card>

      {/* Search Results */}
      {hasSearched && (
        <>
          <div className="flex justify-between items-center py-2 border-b border-border">
            <span className="text-xs text-text-secondary">
              {results.length} result{results.length !== 1 ? 's' : ''}
            </span>
            <button
              onClick={clearSearch}
              className="px-2 py-1 text-xs bg-bg-tertiary border border-border rounded-md text-text-secondary hover:border-error hover:text-error transition-colors"
            >
              Clear
            </button>
          </div>

          <div className="flex flex-col gap-3">
            {results.length === 0 ? (
              <div className="text-center py-8 text-text-muted">
                <p>No matching results found</p>
              </div>
            ) : (
              results.map((result) => (
                <SearchResultItem
                  key={result.object_id}
                  result={result}
                  onClick={() => handleResultClick(result)}
                />
              ))
            )}
          </div>
        </>
      )}
    </>
  );
}

interface SearchResultItemProps {
  result: SearchResult;
  onClick: () => void;
}

function SearchResultItem({ result, onClick }: SearchResultItemProps) {
  return (
    <div
      onClick={onClick}
      className={cn(
        'bg-bg-card border border-border rounded-[12px] p-3 flex gap-3 cursor-pointer transition-all duration-200 m-0.5',
        'hover:border-border-highlight hover:translate-x-0.5'
      )}
    >
      {result.crop_image && (
        <img
          src={result.crop_image}
          alt={result.class_name}
          className="w-[60px] h-[60px] object-cover rounded-md flex-shrink-0"
        />
      )}
      <div className="flex-1 min-w-0">
        <div className="text-xs font-medium text-text-primary">
          {result.class_name}
        </div>
        <div className="text-xs text-text-secondary line-clamp-2 mt-0.5">
          {result.analysis || 'No analysis'}
        </div>
        <div className="flex gap-2 mt-1">
          {result.scores && (
            <span className="px-2 py-0.5 text-[10px] rounded bg-accent-soft text-accent">
              Score: {(result.scores.hybrid * 100).toFixed(0)}%
            </span>
          )}
          {result.similarity !== undefined && (
            <span className="px-2 py-0.5 text-[10px] rounded bg-accent-soft text-accent">
              Similarity: {(result.similarity * 100).toFixed(0)}%
            </span>
          )}
          {result.score !== undefined && (
            <span className="px-2 py-0.5 text-[10px] rounded bg-accent-soft text-accent">
              Relevance: {(result.score * 100).toFixed(0)}%
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

export default SearchTab;

