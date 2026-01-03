import { useState, useCallback } from 'react';
import { searchText, searchByImage, hybridSearch } from '../api/client';
import type { SearchResult } from '../types';

/**
 * Hook for managing search functionality.
 */
export function useSearch() {
  const [textQuery, setTextQuery] = useState('');
  const [searchImageFile, setSearchImageFile] = useState<File | null>(null);
  const [searchImagePreview, setSearchImagePreview] = useState<string | null>(null);
  const [textWeight, setTextWeight] = useState(50);
  const [vectorWeight, setVectorWeight] = useState(50);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);

  const hasText = textQuery.trim().length > 0;
  const hasImage = searchImageFile !== null;
  const canSearch = hasText || hasImage;
  const showWeights = hasText && hasImage;

  const handleImageSelect = useCallback((file: File) => {
    setSearchImageFile(file);
    
    // Create preview
    const reader = new FileReader();
    reader.onload = (e) => {
      setSearchImagePreview(e.target?.result as string);
    };
    reader.readAsDataURL(file);
  }, []);

  const handleTextWeightChange = useCallback((value: number) => {
    setTextWeight(value);
    setVectorWeight(100 - value);
  }, []);

  const handleVectorWeightChange = useCallback((value: number) => {
    setVectorWeight(value);
    setTextWeight(100 - value);
  }, []);

  const performSearch = useCallback(async () => {
    if (!canSearch || isSearching) return;

    setIsSearching(true);
    setHasSearched(true);

    try {
      let searchResults: SearchResult[] = [];

      if (hasText && hasImage) {
        // Hybrid search
        const response = await hybridSearch(
          textQuery,
          searchImageFile,
          textWeight / 100,
          vectorWeight / 100,
          20
        );
        searchResults = response.results;
      } else if (hasImage) {
        // Image-only search
        const response = await searchByImage(searchImageFile!, 20);
        searchResults = response.results;
      } else {
        // Text-only search
        const response = await searchText(textQuery, 50);
        searchResults = response.results;
      }

      setResults(searchResults);
    } catch (error) {
      console.error('Search failed:', error);
      setResults([]);
    } finally {
      setIsSearching(false);
    }
  }, [
    canSearch,
    isSearching,
    hasText,
    hasImage,
    textQuery,
    searchImageFile,
    textWeight,
    vectorWeight,
  ]);

  const clearSearch = useCallback(() => {
    setTextQuery('');
    setSearchImageFile(null);
    setSearchImagePreview(null);
    setTextWeight(50);
    setVectorWeight(50);
    setResults([]);
    setHasSearched(false);
  }, []);

  return {
    // State
    textQuery,
    searchImageFile,
    searchImagePreview,
    textWeight,
    vectorWeight,
    results,
    isSearching,
    hasSearched,
    
    // Computed
    canSearch,
    showWeights,
    
    // Actions
    setTextQuery,
    handleImageSelect,
    handleTextWeightChange,
    handleVectorWeightChange,
    performSearch,
    clearSearch,
  };
}

export default useSearch;

