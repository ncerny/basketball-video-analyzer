/**
 * Annotation Accordion Component
 *
 * Modern accordion-style annotation list with:
 * - Expandable annotation cards
 * - Type filtering tabs
 * - Recording mode for creating new annotations
 * - Inline editing
 * - Click to seek functionality
 */

import { useState, useMemo, useCallback } from 'react';
import {
  Accordion,
  ActionIcon,
  Badge,
  Box,
  Button,
  Group,
  Paper,
  SegmentedControl,
  Stack,
  Text,
  TextInput,
  Textarea,
  Tooltip,
} from '@mantine/core';
import {
  IconCheck,
  IconCircleFilled,
  IconEdit,
  IconPlayerPause,
  IconPlayerRecord,
  IconPlayerStop,
  IconTrash,
  IconX,
} from '@tabler/icons-react';
import { useTimelineStore } from '../store/timelineStore';
import type { Annotation, CreateAnnotationDTO } from '../types/api';

interface AnnotationAccordionProps {
  /** Game ID for creating annotations */
  gameId: number;

  /** Array of annotations to display */
  annotations: Annotation[];

  /** Callback to create a new annotation */
  onCreate: (data: CreateAnnotationDTO) => Promise<void>;

  /** Callback to update an annotation */
  onUpdate: (id: number, data: Partial<CreateAnnotationDTO>) => Promise<void>;

  /** Callback to delete an annotation */
  onDelete: (id: number) => Promise<void>;

  /** Callback to verify an annotation */
  onVerify: (id: number) => Promise<void>;

  /** Whether loading */
  isLoading?: boolean;

  /** CSS class name */
  className?: string;
}

type FilterType = 'all' | 'play' | 'event' | 'note';

interface RecordingState {
  isRecording: boolean;
  startTime: number | null;
  wasPlaying: boolean;
}

/**
 * Format seconds as MM:SS
 */
const formatTime = (seconds: number): string => {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
};

/**
 * Format seconds as MM:SS.ms for editing
 */
const formatTimeDetailed = (seconds: number): string => {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}:${secs.toFixed(1).padStart(4, '0')}`;
};

/**
 * Get color for annotation type
 */
const getTypeColor = (type: string): string => {
  switch (type) {
    case 'play':
      return 'green';
    case 'event':
      return 'yellow';
    case 'note':
      return 'violet';
    default:
      return 'gray';
  }
};

/**
 * Get dot color for annotation type
 */
const getTypeDotColor = (type: string): string => {
  switch (type) {
    case 'play':
      return '#40c057';
    case 'event':
      return '#fab005';
    case 'note':
      return '#be4bdb';
    default:
      return '#868e96';
  }
};

/**
 * Annotation Accordion - Modern expandable annotation list
 */
export const AnnotationAccordion: React.FC<AnnotationAccordionProps> = ({
  gameId,
  annotations,
  onCreate,
  onUpdate,
  onDelete,
  onVerify,
  isLoading = false,
  className = '',
}) => {
  // Timeline store
  const currentGameTime = useTimelineStore(state => state.currentGameTime);
  const isPlaying = useTimelineStore(state => state.isPlaying);
  const play = useTimelineStore(state => state.play);
  const pause = useTimelineStore(state => state.pause);
  const seekToGameTime = useTimelineStore(state => state.seekToGameTime);

  // Local state
  const [filterType, setFilterType] = useState<FilterType>('all');
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editForm, setEditForm] = useState<{
    title: string;
    description: string;
    startTime: number;
    endTime: number;
  }>({ title: '', description: '', startTime: 0, endTime: 0 });
  const [recording, setRecording] = useState<RecordingState>({
    isRecording: false,
    startTime: null,
    wasPlaying: false,
  });
  const [newAnnotation, setNewAnnotation] = useState<{
    title: string;
    description: string;
    type: 'play' | 'event' | 'note';
  }>({ title: '', description: '', type: 'event' });

  // Filter annotations
  const filteredAnnotations = useMemo(() => {
    let filtered = annotations;
    if (filterType !== 'all') {
      filtered = filtered.filter(a => a.annotation_type === filterType);
    }
    return [...filtered].sort((a, b) => a.game_timestamp_start - b.game_timestamp_start);
  }, [annotations, filterType]);

  // Check if annotation is currently active
  const isActive = useCallback(
    (annotation: Annotation): boolean =>
      currentGameTime >= annotation.game_timestamp_start &&
      currentGameTime <= annotation.game_timestamp_end,
    [currentGameTime]
  );

  // Start recording an annotation
  const startRecording = useCallback(() => {
    const wasPlaying = isPlaying;
    if (isPlaying) {
      pause();
    }
    setRecording({
      isRecording: true,
      startTime: currentGameTime,
      wasPlaying,
    });
  }, [currentGameTime, isPlaying, pause]);

  // Stop recording and create annotation
  const stopRecording = useCallback(async () => {
    if (!recording.isRecording || recording.startTime === null) return;

    const endTime = currentGameTime;
    const startTime = recording.startTime;

    // Ensure end > start
    const finalStart = Math.min(startTime, endTime);
    const finalEnd = Math.max(startTime, endTime, startTime + 0.5); // At least 0.5s duration

    try {
      await onCreate({
        game_id: gameId,
        title: newAnnotation.title || undefined,
        description: newAnnotation.description || undefined,
        game_timestamp_start: finalStart,
        game_timestamp_end: finalEnd,
        annotation_type: newAnnotation.type,
        verified: false,
        created_by: 'user',
      });

      // Reset state
      setNewAnnotation({ title: '', description: '', type: 'event' });
    } catch (err) {
      console.error('Failed to create annotation:', err);
    }

    setRecording({
      isRecording: false,
      startTime: null,
      wasPlaying: false,
    });
  }, [recording, currentGameTime, newAnnotation, gameId, onCreate]);

  // Cancel recording
  const cancelRecording = useCallback(() => {
    setRecording({
      isRecording: false,
      startTime: null,
      wasPlaying: false,
    });
    setNewAnnotation({ title: '', description: '', type: 'event' });
  }, []);

  // Continue playback during recording
  const continuePlayback = useCallback(() => {
    if (recording.isRecording) {
      play();
    }
  }, [recording.isRecording, play]);

  // Start editing an annotation
  const startEditing = useCallback((annotation: Annotation) => {
    setEditingId(annotation.id);
    setEditForm({
      title: annotation.title || '',
      description: annotation.description || '',
      startTime: annotation.game_timestamp_start,
      endTime: annotation.game_timestamp_end,
    });
  }, []);

  // Save edit
  const saveEdit = useCallback(async () => {
    if (editingId === null) return;

    // Validate timestamps
    if (editForm.endTime <= editForm.startTime) {
      alert('End time must be after start time');
      return;
    }

    try {
      await onUpdate(editingId, {
        title: editForm.title || undefined,
        description: editForm.description || undefined,
        game_timestamp_start: editForm.startTime,
        game_timestamp_end: editForm.endTime,
      });
      setEditingId(null);
    } catch (err) {
      console.error('Failed to update annotation:', err);
    }
  }, [editingId, editForm, onUpdate]);

  // Cancel edit
  const cancelEdit = useCallback(() => {
    setEditingId(null);
    setEditForm({ title: '', description: '', startTime: 0, endTime: 0 });
  }, []);

  return (
    <Paper withBorder radius="md" p="md" className={className}>
      <Stack gap="md">
        {/* Header */}
        <Group justify="space-between">
          <Group gap="xs">
            <Text fw={600} size="lg">Annotations</Text>
            <Badge variant="light" color="gray" size="sm">
              {filteredAnnotations.length}
            </Badge>
          </Group>

          {/* Add annotation button / Recording controls */}
          {recording.isRecording ? (
            <Group gap="xs">
              <Badge color="red" variant="filled" leftSection={<IconCircleFilled size={8} />}>
                Recording from {formatTime(recording.startTime || 0)}
              </Badge>
              <Tooltip label="Continue playback">
                <ActionIcon
                  variant="light"
                  color="blue"
                  onClick={continuePlayback}
                  disabled={isPlaying}
                >
                  <IconPlayerPause size={16} />
                </ActionIcon>
              </Tooltip>
              <Tooltip label="Stop & Save">
                <ActionIcon variant="filled" color="green" onClick={stopRecording}>
                  <IconPlayerStop size={16} />
                </ActionIcon>
              </Tooltip>
              <Tooltip label="Cancel">
                <ActionIcon variant="light" color="red" onClick={cancelRecording}>
                  <IconX size={16} />
                </ActionIcon>
              </Tooltip>
            </Group>
          ) : (
            <Button
              leftSection={<IconPlayerRecord size={16} />}
              variant="light"
              color="red"
              size="sm"
              onClick={startRecording}
            >
              Add Annotation
            </Button>
          )}
        </Group>

        {/* Recording form (when recording) */}
        {recording.isRecording && (
          <Paper withBorder p="sm" radius="md" bg="dark.7">
            <Stack gap="sm">
              <TextInput
                placeholder="Annotation title (optional)"
                value={newAnnotation.title}
                onChange={e => setNewAnnotation(prev => ({ ...prev, title: e.target.value }))}
                size="sm"
              />
              <Textarea
                placeholder="Description (optional)"
                value={newAnnotation.description}
                onChange={e => setNewAnnotation(prev => ({ ...prev, description: e.target.value }))}
                size="sm"
                rows={2}
              />
              <SegmentedControl
                value={newAnnotation.type}
                onChange={(value) => setNewAnnotation(prev => ({ ...prev, type: value as 'play' | 'event' | 'note' }))}
                data={[
                  { label: 'Play', value: 'play' },
                  { label: 'Event', value: 'event' },
                  { label: 'Note', value: 'note' },
                ]}
                size="xs"
                fullWidth
              />
              <Text size="xs" c="dimmed">
                Press play to continue watching, then click "Stop & Save" when done.
              </Text>
            </Stack>
          </Paper>
        )}

        {/* Filter tabs */}
        <SegmentedControl
          value={filterType}
          onChange={(value) => setFilterType(value as FilterType)}
          data={[
            { label: 'All', value: 'all' },
            { label: 'Plays', value: 'play' },
            { label: 'Events', value: 'event' },
            { label: 'Notes', value: 'note' },
          ]}
          size="xs"
          fullWidth
        />

        {/* Annotations list */}
        {filteredAnnotations.length === 0 ? (
          <Text c="dimmed" ta="center" py="xl" size="sm">
            {annotations.length === 0
              ? 'No annotations yet. Click "Add Annotation" to create one.'
              : 'No annotations match the current filter.'}
          </Text>
        ) : (
          <Accordion
            value={expandedId}
            onChange={setExpandedId}
            chevronPosition="right"
            variant="separated"
            radius="md"
            styles={{
              item: {
                backgroundColor: 'var(--mantine-color-dark-7)',
                border: '1px solid var(--mantine-color-dark-5)',
                '&[data-active]': {
                  backgroundColor: 'var(--mantine-color-dark-6)',
                },
              },
              control: {
                padding: 'var(--mantine-spacing-sm)',
                '&:hover': {
                  backgroundColor: 'transparent',
                },
              },
              content: {
                padding: 'var(--mantine-spacing-sm)',
                paddingTop: 0,
              },
            }}
          >
            {filteredAnnotations.map((annotation) => {
              const active = isActive(annotation);
              const isEditing = editingId === annotation.id;

              return (
                <Accordion.Item key={annotation.id} value={String(annotation.id)}>
                  <Accordion.Control>
                    <Group gap="sm" wrap="nowrap" onClick={(e) => {
                      // Allow clicking to seek without expanding
                      if ((e.target as HTMLElement).closest('[data-accordion-chevron]')) return;
                    }}>
                      {/* Type indicator dot */}
                      <Box
                        style={{
                          width: 8,
                          height: 8,
                          borderRadius: '50%',
                          backgroundColor: getTypeDotColor(annotation.annotation_type),
                          flexShrink: 0,
                        }}
                      />

                      {/* Time range - clickable to seek */}
                      <Text
                        size="sm"
                        ff="monospace"
                        c={active ? 'blue' : 'dimmed'}
                        style={{ cursor: 'pointer', flexShrink: 0 }}
                        onClick={(e) => {
                          e.stopPropagation();
                          seekToGameTime(annotation.game_timestamp_start);
                        }}
                      >
                        {formatTime(annotation.game_timestamp_start)}
                      </Text>

                      {/* Title or type */}
                      <Text size="sm" fw={500} truncate style={{ flex: 1 }}>
                        {annotation.title || annotation.annotation_type}
                      </Text>

                      {/* Badges */}
                      <Group gap={4} wrap="nowrap">
                        <Badge size="xs" variant="light" color={getTypeColor(annotation.annotation_type)}>
                          {annotation.annotation_type}
                        </Badge>
                        {annotation.verified && (
                          <Badge size="xs" variant="light" color="green" leftSection={<IconCheck size={10} />}>
                            Verified
                          </Badge>
                        )}
                        {active && (
                          <Badge size="xs" variant="filled" color="blue">
                            Now
                          </Badge>
                        )}
                      </Group>
                    </Group>
                  </Accordion.Control>

                  <Accordion.Panel>
                    <Stack gap="sm">
                      {/* Time info */}
                      <Group gap="md">
                        <Text size="xs" c="dimmed">
                          Duration: {formatTimeDetailed(annotation.game_timestamp_end - annotation.game_timestamp_start)}
                        </Text>
                        <Text size="xs" c="dimmed">
                          {formatTime(annotation.game_timestamp_start)} → {formatTime(annotation.game_timestamp_end)}
                        </Text>
                        {annotation.confidence_score !== undefined && (
                          <Text size="xs" c="dimmed">
                            {Math.round(annotation.confidence_score * 100)}% confidence
                          </Text>
                        )}
                      </Group>

                      {/* Edit form or description */}
                      {isEditing ? (
                        <Stack gap="xs">
                          <TextInput
                            placeholder="Title"
                            value={editForm.title}
                            onChange={e => setEditForm(prev => ({ ...prev, title: e.target.value }))}
                            size="xs"
                          />
                          <Textarea
                            placeholder="Description"
                            value={editForm.description}
                            onChange={e => setEditForm(prev => ({ ...prev, description: e.target.value }))}
                            size="xs"
                            rows={2}
                          />
                          <Group gap="xs" grow>
                            <TextInput
                              label="Start Time"
                              placeholder="MM:SS"
                              value={formatTime(editForm.startTime)}
                              onChange={e => {
                                const parts = e.target.value.split(':');
                                if (parts.length === 2) {
                                  const mins = parseInt(parts[0], 10) || 0;
                                  const secs = parseInt(parts[1], 10) || 0;
                                  setEditForm(prev => ({ ...prev, startTime: mins * 60 + secs }));
                                }
                              }}
                              size="xs"
                              styles={{ input: { fontFamily: 'monospace' } }}
                            />
                            <TextInput
                              label="End Time"
                              placeholder="MM:SS"
                              value={formatTime(editForm.endTime)}
                              onChange={e => {
                                const parts = e.target.value.split(':');
                                if (parts.length === 2) {
                                  const mins = parseInt(parts[0], 10) || 0;
                                  const secs = parseInt(parts[1], 10) || 0;
                                  setEditForm(prev => ({ ...prev, endTime: mins * 60 + secs }));
                                }
                              }}
                              size="xs"
                              styles={{ input: { fontFamily: 'monospace' } }}
                            />
                          </Group>
                          <Group gap="xs">
                            <Button
                              size="xs"
                              variant="light"
                              onClick={() => setEditForm(prev => ({ ...prev, startTime: currentGameTime }))}
                            >
                              Set Start to Current Time
                            </Button>
                            <Button
                              size="xs"
                              variant="light"
                              onClick={() => setEditForm(prev => ({ ...prev, endTime: currentGameTime }))}
                            >
                              Set End to Current Time
                            </Button>
                          </Group>
                          <Group gap="xs">
                            <Button size="xs" onClick={saveEdit} loading={isLoading}>
                              Save
                            </Button>
                            <Button size="xs" variant="light" onClick={cancelEdit}>
                              Cancel
                            </Button>
                          </Group>
                        </Stack>
                      ) : (
                        annotation.description && (
                          <Text size="sm" c="dimmed">
                            {annotation.description}
                          </Text>
                        )
                      )}

                      {/* Actions */}
                      {!isEditing && (
                        <Group gap="xs">
                          <Button
                            size="xs"
                            variant="light"
                            onClick={() => seekToGameTime(annotation.game_timestamp_start)}
                          >
                            Go to
                          </Button>
                          {!annotation.verified && (
                            <Button
                              size="xs"
                              variant="light"
                              color="green"
                              onClick={() => onVerify(annotation.id)}
                              loading={isLoading}
                            >
                              Verify
                            </Button>
                          )}
                          <Button
                            size="xs"
                            variant="light"
                            leftSection={<IconEdit size={14} />}
                            onClick={() => startEditing(annotation)}
                          >
                            Edit
                          </Button>
                          <Button
                            size="xs"
                            variant="light"
                            color="red"
                            leftSection={<IconTrash size={14} />}
                            onClick={() => {
                              if (confirm('Delete this annotation?')) {
                                onDelete(annotation.id);
                              }
                            }}
                            loading={isLoading}
                          >
                            Delete
                          </Button>
                        </Group>
                      )}
                    </Stack>
                  </Accordion.Panel>
                </Accordion.Item>
              );
            })}
          </Accordion>
        )}
      </Stack>
    </Paper>
  );
};

export default AnnotationAccordion;
