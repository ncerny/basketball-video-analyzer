/**
 * Annotation Panel Component
 *
 * Complete annotation management interface with:
 * - List of existing annotations
 * - Create/edit annotation form
 * - API integration for CRUD operations
 * - Integration with timeline state
 */

import { useState, useEffect } from 'react';
import { useTimelineStore } from '../store/timelineStore';
import { AnnotationForm } from './AnnotationForm';
import { AnnotationList } from './AnnotationList';
import { api } from '../services/api';
import type { Annotation, CreateAnnotationDTO } from '../types/api';

interface AnnotationPanelProps {
  /** Current game ID */
  gameId: number;

  /** CSS class name for the container */
  className?: string;
}

type ViewMode = 'list' | 'create' | 'edit';

/**
 * Annotation Panel - Complete annotation management interface
 */
export const AnnotationPanel: React.FC<AnnotationPanelProps> = ({
  gameId,
  className = '',
}) => {
  // State
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [selectedAnnotation, setSelectedAnnotation] = useState<Annotation | undefined>();
  const [viewMode, setViewMode] = useState<ViewMode>('list');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Timeline store
  const loadAnnotations = useTimelineStore(state => state.loadAnnotations);

  // Load annotations on mount
  useEffect(() => {
    fetchAnnotations();
  }, [gameId]);

  // Fetch annotations from API
  const fetchAnnotations = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const data = await api.annotations.listByGame(gameId);
      setAnnotations(data);
      // Also update timeline store for visualization
      loadAnnotations(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load annotations');
      console.error('Error fetching annotations:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // Create annotation
  const handleCreate = async (data: CreateAnnotationDTO) => {
    setIsLoading(true);
    setError(null);

    try {
      const newAnnotation = await api.annotations.create(data);
      setAnnotations(prev => [...prev, newAnnotation]);
      loadAnnotations([...annotations, newAnnotation]);
      setViewMode('list');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create annotation');
      console.error('Error creating annotation:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // Update annotation
  const handleUpdate = async (data: CreateAnnotationDTO) => {
    if (!selectedAnnotation) return;

    setIsLoading(true);
    setError(null);

    try {
      const updated = await api.annotations.update(selectedAnnotation.id, {
        game_timestamp_start: data.game_timestamp_start,
        game_timestamp_end: data.game_timestamp_end,
        annotation_type: data.annotation_type,
        confidence_score: data.confidence_score,
        verified: data.verified,
      });

      setAnnotations(prev =>
        prev.map(a => (a.id === updated.id ? updated : a))
      );
      loadAnnotations(
        annotations.map(a => (a.id === updated.id ? updated : a))
      );
      setSelectedAnnotation(undefined);
      setViewMode('list');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update annotation');
      console.error('Error updating annotation:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // Delete annotation
  const handleDelete = async (id: number) => {
    setIsLoading(true);
    setError(null);

    try {
      await api.annotations.delete(id);
      const updatedAnnotations = annotations.filter(a => a.id !== id);
      setAnnotations(updatedAnnotations);
      loadAnnotations(updatedAnnotations);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete annotation');
      console.error('Error deleting annotation:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // Verify annotation
  const handleVerify = async (id: number) => {
    setIsLoading(true);
    setError(null);

    try {
      const verified = await api.annotations.verify(id);
      setAnnotations(prev =>
        prev.map(a => (a.id === verified.id ? verified : a))
      );
      loadAnnotations(
        annotations.map(a => (a.id === verified.id ? verified : a))
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to verify annotation');
      console.error('Error verifying annotation:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle edit button
  const handleEdit = (annotation: Annotation) => {
    setSelectedAnnotation(annotation);
    setViewMode('edit');
  };

  // Handle cancel
  const handleCancel = () => {
    setSelectedAnnotation(undefined);
    setViewMode('list');
    setError(null);
  };

  return (
    <div className={`bg-gray-900 text-white ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-700">
        <h2 className="text-xl font-semibold">
          {viewMode === 'list'
            ? 'Annotations'
            : viewMode === 'create'
            ? 'Create Annotation'
            : 'Edit Annotation'}
        </h2>
        {viewMode === 'list' && (
          <button
            onClick={() => setViewMode('create')}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition flex items-center gap-2"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 4v16m8-8H4"
              />
            </svg>
            New Annotation
          </button>
        )}
      </div>

      {/* Error Message */}
      {error && (
        <div className="mx-4 mt-4 p-3 bg-red-900 border border-red-700 rounded-lg text-red-200">
          <div className="flex items-center gap-2">
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
              <path
                fillRule="evenodd"
                d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                clipRule="evenodd"
              />
            </svg>
            <span>{error}</span>
          </div>
          <button
            onClick={() => setError(null)}
            className="mt-2 text-sm underline hover:no-underline"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Content */}
      <div className="p-4">
        {viewMode === 'list' ? (
          <AnnotationList
            annotations={annotations}
            onEdit={handleEdit}
            onDelete={handleDelete}
            onVerify={handleVerify}
            isLoading={isLoading}
          />
        ) : (
          <AnnotationForm
            gameId={gameId}
            annotation={selectedAnnotation}
            onSubmit={viewMode === 'create' ? handleCreate : handleUpdate}
            onCancel={handleCancel}
            isLoading={isLoading}
          />
        )}
      </div>
    </div>
  );
};

export default AnnotationPanel;
