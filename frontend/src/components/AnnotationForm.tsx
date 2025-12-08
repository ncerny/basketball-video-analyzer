/**
 * Annotation Form Component
 *
 * Form for creating and editing annotations with:
 * - Start/end time selection
 * - Annotation type selector
 * - Verification status
 * - Integration with timeline for time capture
 */

import { useState, useEffect } from 'react';
import { useTimelineStore } from '../store/timelineStore';
import type { Annotation, CreateAnnotationDTO } from '../types/api';

interface AnnotationFormProps {
  /** Current game ID */
  gameId: number;

  /** Existing annotation to edit (optional) */
  annotation?: Annotation;

  /** Callback when form is submitted */
  onSubmit: (data: CreateAnnotationDTO) => void | Promise<void>;

  /** Callback when form is cancelled */
  onCancel: () => void;

  /** Show loading state */
  isLoading?: boolean;
}

/**
 * Annotation Form Component
 */
export const AnnotationForm: React.FC<AnnotationFormProps> = ({
  gameId,
  annotation,
  onSubmit,
  onCancel,
  isLoading = false,
}) => {
  const currentGameTime = useTimelineStore(state => state.currentGameTime);
  const videos = useTimelineStore(state => state.videos);

  // Form state
  const [startTime, setStartTime] = useState<number>(
    annotation?.game_timestamp_start ?? currentGameTime
  );
  const [endTime, setEndTime] = useState<number>(
    annotation?.game_timestamp_end ?? currentGameTime + 5
  );
  const [annotationType, setAnnotationType] = useState<'play' | 'event' | 'note'>(
    annotation?.annotation_type ?? 'event'
  );
  const [verified, setVerified] = useState<boolean>(
    annotation?.verified ?? false
  );
  const [confidenceScore, setConfidenceScore] = useState<number | undefined>(
    annotation?.confidence_score
  );

  // Validation state
  const [errors, setErrors] = useState<Record<string, string>>({});

  // Update form when annotation changes
  useEffect(() => {
    if (annotation) {
      setStartTime(annotation.game_timestamp_start);
      setEndTime(annotation.game_timestamp_end);
      setAnnotationType(annotation.annotation_type);
      setVerified(annotation.verified);
      setConfidenceScore(annotation.confidence_score);
    }
  }, [annotation]);

  // Format time as MM:SS.ms
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toFixed(2).padStart(5, '0')}`;
  };

  // Parse time from MM:SS.ms format
  const parseTime = (timeStr: string): number => {
    const parts = timeStr.split(':');
    if (parts.length !== 2) return 0;
    const mins = parseInt(parts[0], 10) || 0;
    const secs = parseFloat(parts[1]) || 0;
    return mins * 60 + secs;
  };

  // Capture current timeline position
  const captureStartTime = () => {
    setStartTime(currentGameTime);
    // Auto-set end time if it's before start
    if (endTime <= currentGameTime) {
      setEndTime(currentGameTime + 5);
    }
  };

  const captureEndTime = () => {
    setEndTime(currentGameTime);
  };

  // Validate form
  const validate = (): boolean => {
    const newErrors: Record<string, string> = {};

    if (startTime < 0) {
      newErrors.startTime = 'Start time must be positive';
    }

    if (endTime <= startTime) {
      newErrors.endTime = 'End time must be after start time';
    }

    if (confidenceScore !== undefined && (confidenceScore < 0 || confidenceScore > 1)) {
      newErrors.confidenceScore = 'Confidence must be between 0 and 1';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  // Handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!validate()) {
      return;
    }

    const data: CreateAnnotationDTO = {
      game_id: gameId,
      game_timestamp_start: startTime,
      game_timestamp_end: endTime,
      annotation_type: annotationType,
      confidence_score: confidenceScore,
      verified,
      created_by: 'user',
    };

    await onSubmit(data);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* Annotation Type */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Annotation Type
        </label>
        <div className="flex gap-2">
          {(['play', 'event', 'note'] as const).map(type => (
            <button
              key={type}
              type="button"
              onClick={() => setAnnotationType(type)}
              className={`px-4 py-2 rounded-lg font-medium transition ${
                annotationType === type
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              {type.charAt(0).toUpperCase() + type.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Start Time */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Start Time
        </label>
        <div className="flex gap-2">
          <input
            type="text"
            value={formatTime(startTime)}
            onChange={e => setStartTime(parseTime(e.target.value))}
            className={`flex-1 px-3 py-2 bg-gray-700 border rounded-lg text-white font-mono ${
              errors.startTime
                ? 'border-red-500'
                : 'border-gray-600 focus:border-blue-500 focus:ring-1 focus:ring-blue-500'
            } outline-none`}
            placeholder="MM:SS.ms"
          />
          <button
            type="button"
            onClick={captureStartTime}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition"
            title="Capture current timeline position"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
              />
            </svg>
          </button>
        </div>
        {errors.startTime && (
          <p className="mt-1 text-sm text-red-400">{errors.startTime}</p>
        )}
      </div>

      {/* End Time */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          End Time
        </label>
        <div className="flex gap-2">
          <input
            type="text"
            value={formatTime(endTime)}
            onChange={e => setEndTime(parseTime(e.target.value))}
            className={`flex-1 px-3 py-2 bg-gray-700 border rounded-lg text-white font-mono ${
              errors.endTime
                ? 'border-red-500'
                : 'border-gray-600 focus:border-blue-500 focus:ring-1 focus:ring-blue-500'
            } outline-none`}
            placeholder="MM:SS.ms"
          />
          <button
            type="button"
            onClick={captureEndTime}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition"
            title="Capture current timeline position"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
              />
            </svg>
          </button>
        </div>
        {errors.endTime && (
          <p className="mt-1 text-sm text-red-400">{errors.endTime}</p>
        )}
      </div>

      {/* Duration Display */}
      <div className="text-sm text-gray-400">
        Duration: {formatTime(Math.max(0, endTime - startTime))}
      </div>

      {/* Confidence Score (optional) */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Confidence Score (optional)
        </label>
        <input
          type="number"
          min="0"
          max="1"
          step="0.01"
          value={confidenceScore ?? ''}
          onChange={e =>
            setConfidenceScore(e.target.value ? parseFloat(e.target.value) : undefined)
          }
          className={`w-full px-3 py-2 bg-gray-700 border rounded-lg text-white ${
            errors.confidenceScore
              ? 'border-red-500'
              : 'border-gray-600 focus:border-blue-500 focus:ring-1 focus:ring-blue-500'
          } outline-none`}
          placeholder="0.0 - 1.0"
        />
        {errors.confidenceScore && (
          <p className="mt-1 text-sm text-red-400">{errors.confidenceScore}</p>
        )}
      </div>

      {/* Verified Checkbox */}
      <div className="flex items-center">
        <input
          type="checkbox"
          id="verified"
          checked={verified}
          onChange={e => setVerified(e.target.checked)}
          className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
        />
        <label htmlFor="verified" className="ml-2 text-sm text-gray-300">
          Mark as verified
        </label>
      </div>

      {/* Video Coverage Info */}
      {videos.length > 0 && (
        <div className="p-3 bg-gray-800 rounded-lg text-sm text-gray-400">
          <div className="font-medium text-gray-300 mb-1">Video Coverage</div>
          <div>
            This annotation spans {formatTime(endTime - startTime)} across{' '}
            {videos.filter(v => {
              const vStart = v.game_time_offset ?? 0;
              const vEnd = vStart + v.duration_seconds;
              return !(endTime <= vStart || startTime >= vEnd);
            }).length}{' '}
            video(s)
          </div>
        </div>
      )}

      {/* Form Actions */}
      <div className="flex justify-end gap-3 pt-4 border-t border-gray-700">
        <button
          type="button"
          onClick={onCancel}
          disabled={isLoading}
          className="px-4 py-2 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg transition"
        >
          Cancel
        </button>
        <button
          type="submit"
          disabled={isLoading}
          className="px-6 py-2 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg font-semibold transition"
        >
          {isLoading ? 'Saving...' : annotation ? 'Update' : 'Create'}
        </button>
      </div>
    </form>
  );
};

export default AnnotationForm;
