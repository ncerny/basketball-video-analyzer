/**
 * Annotation List Component
 *
 * Displays list of annotations with:
 * - Filterable by type
 * - Sortable by time
 * - Click to seek to annotation
 * - Edit/delete actions
 * - Visual type indicators
 */

import { useState, useMemo } from 'react';
import { useTimelineStore } from '../store/timelineStore';
import type { Annotation } from '../types/api';

interface AnnotationListProps {
  /** Array of annotations to display */
  annotations: Annotation[];

  /** Callback when annotation is selected for editing */
  onEdit?: (annotation: Annotation) => void;

  /** Callback when annotation is deleted */
  onDelete?: (id: number) => void;

  /** Callback when annotation is verified */
  onVerify?: (id: number) => void;

  /** Show loading state */
  isLoading?: boolean;
}

type SortField = 'time' | 'type' | 'created';
type SortOrder = 'asc' | 'desc';

/**
 * Annotation List Component
 */
export const AnnotationList: React.FC<AnnotationListProps> = ({
  annotations,
  onEdit,
  onDelete,
  onVerify,
  isLoading = false,
}) => {
  const seekToGameTime = useTimelineStore(state => state.seekToGameTime);
  const currentGameTime = useTimelineStore(state => state.currentGameTime);

  // Filter and sort state
  const [filterType, setFilterType] = useState<'all' | 'play' | 'event' | 'note'>('all');
  const [sortField, setSortField] = useState<SortField>('time');
  const [sortOrder, setSortOrder] = useState<SortOrder>('asc');
  const [showVerifiedOnly, setShowVerifiedOnly] = useState(false);

  // Format time as MM:SS
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Filter and sort annotations
  const filteredAnnotations = useMemo(() => {
    let filtered = annotations;

    // Filter by type
    if (filterType !== 'all') {
      filtered = filtered.filter(a => a.annotation_type === filterType);
    }

    // Filter by verified status
    if (showVerifiedOnly) {
      filtered = filtered.filter(a => a.verified);
    }

    // Sort
    filtered = [...filtered].sort((a, b) => {
      let comparison = 0;

      switch (sortField) {
        case 'time':
          comparison = a.game_timestamp_start - b.game_timestamp_start;
          break;
        case 'type':
          comparison = a.annotation_type.localeCompare(b.annotation_type);
          break;
        case 'created':
          comparison = new Date(a.created_at).getTime() - new Date(b.created_at).getTime();
          break;
      }

      return sortOrder === 'asc' ? comparison : -comparison;
    });

    return filtered;
  }, [annotations, filterType, showVerifiedOnly, sortField, sortOrder]);

  // Toggle sort
  const toggleSort = (field: SortField) => {
    if (sortField === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortOrder('asc');
    }
  };

  // Get type color class
  const getTypeColor = (type: string): string => {
    switch (type) {
      case 'play':
        return 'bg-green-500';
      case 'event':
        return 'bg-yellow-500';
      case 'note':
        return 'bg-purple-500';
      default:
        return 'bg-gray-500';
    }
  };

  // Check if annotation is currently active
  const isActive = (annotation: Annotation): boolean => {
    return (
      currentGameTime >= annotation.game_timestamp_start &&
      currentGameTime <= annotation.game_timestamp_end
    );
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <div className="text-gray-400">Loading annotations...</div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Filters and Controls */}
      <div className="flex flex-wrap items-center gap-3 pb-3 border-b border-gray-700">
        {/* Type Filter */}
        <div className="flex gap-1">
          {(['all', 'play', 'event', 'note'] as const).map(type => (
            <button
              key={type}
              onClick={() => setFilterType(type)}
              className={`px-3 py-1 rounded text-sm font-medium transition ${
                filterType === type
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              {type.charAt(0).toUpperCase() + type.slice(1)}
            </button>
          ))}
        </div>

        {/* Verified Filter */}
        <div className="flex items-center">
          <input
            type="checkbox"
            id="verified-only"
            checked={showVerifiedOnly}
            onChange={e => setShowVerifiedOnly(e.target.checked)}
            className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
          />
          <label htmlFor="verified-only" className="ml-2 text-sm text-gray-300">
            Verified only
          </label>
        </div>

        {/* Sort Controls */}
        <div className="ml-auto flex gap-2">
          <button
            onClick={() => toggleSort('time')}
            className={`px-3 py-1 rounded text-sm transition ${
              sortField === 'time'
                ? 'bg-gray-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            Time {sortField === 'time' && (sortOrder === 'asc' ? '↑' : '↓')}
          </button>
          <button
            onClick={() => toggleSort('type')}
            className={`px-3 py-1 rounded text-sm transition ${
              sortField === 'type'
                ? 'bg-gray-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            Type {sortField === 'type' && (sortOrder === 'asc' ? '↑' : '↓')}
          </button>
        </div>
      </div>

      {/* Annotation Count */}
      <div className="text-sm text-gray-400">
        Showing {filteredAnnotations.length} of {annotations.length} annotation
        {annotations.length !== 1 ? 's' : ''}
      </div>

      {/* Annotations List */}
      {filteredAnnotations.length === 0 ? (
        <div className="text-center py-8 text-gray-500">
          No annotations found
        </div>
      ) : (
        <div className="space-y-2">
          {filteredAnnotations.map(annotation => (
            <div
              key={annotation.id}
              className={`p-4 rounded-lg border transition cursor-pointer ${
                isActive(annotation)
                  ? 'bg-gray-700 border-blue-500'
                  : 'bg-gray-800 border-gray-700 hover:border-gray-600'
              }`}
              onClick={() => seekToGameTime(annotation.game_timestamp_start)}
            >
              <div className="flex items-start justify-between gap-4">
                {/* Left: Type indicator and info */}
                <div className="flex items-start gap-3 flex-1">
                  {/* Type Badge */}
                  <div
                    className={`w-2 h-2 rounded-full mt-2 ${getTypeColor(
                      annotation.annotation_type
                    )}`}
                    title={annotation.annotation_type}
                  />

                  {/* Info */}
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-sm text-gray-300">
                        {formatTime(annotation.game_timestamp_start)} -{' '}
                        {formatTime(annotation.game_timestamp_end)}
                      </span>
                      <span className="text-xs px-2 py-0.5 rounded bg-gray-700 text-gray-400">
                        {annotation.annotation_type}
                      </span>
                      {annotation.verified && (
                        <span
                          className="text-green-400"
                          title="Verified"
                        >
                          ✓
                        </span>
                      )}
                      {annotation.confidence_score !== undefined && (
                        <span className="text-xs text-gray-500">
                          {Math.round(annotation.confidence_score * 100)}% confident
                        </span>
                      )}
                    </div>
                    <div className="mt-1 text-sm text-gray-400">
                      Duration: {formatTime(
                        annotation.game_timestamp_end - annotation.game_timestamp_start
                      )}
                    </div>
                    <div className="mt-1 text-xs text-gray-500">
                      Created {annotation.created_by === 'ai' ? 'by AI' : 'manually'}
                    </div>
                  </div>
                </div>

                {/* Right: Actions */}
                <div className="flex items-center gap-2" onClick={e => e.stopPropagation()}>
                  {!annotation.verified && onVerify && (
                    <button
                      onClick={() => onVerify(annotation.id)}
                      className="px-3 py-1 bg-green-600 hover:bg-green-700 text-white text-sm rounded transition"
                      title="Verify annotation"
                    >
                      Verify
                    </button>
                  )}
                  {onEdit && (
                    <button
                      onClick={() => onEdit(annotation)}
                      className="px-3 py-1 bg-gray-600 hover:bg-gray-500 text-white text-sm rounded transition"
                      title="Edit annotation"
                    >
                      Edit
                    </button>
                  )}
                  {onDelete && (
                    <button
                      onClick={() => {
                        if (confirm('Delete this annotation?')) {
                          onDelete(annotation.id);
                        }
                      }}
                      className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-sm rounded transition"
                      title="Delete annotation"
                    >
                      Delete
                    </button>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default AnnotationList;
