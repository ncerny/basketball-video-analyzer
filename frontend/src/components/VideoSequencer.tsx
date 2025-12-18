/**
 * Video Sequencer Component
 */

import { useEffect, useMemo, useState } from 'react'
import {
  Alert,
  Badge,
  Box,
  Button,
  Card,
  Group,
  NumberInput,
  Paper,
  Stack,
  Text,
} from '@mantine/core'
import {
  IconAlertCircle,
  IconArrowDown,
  IconArrowUp,
  IconCheck,
  IconClock,
  IconDeviceFloppy,
  IconGripVertical,
  IconRefresh,
} from '@tabler/icons-react'
import type { Video } from '../types/api'
import { api } from '../services/api'
import { APIError } from '../types/api'

interface VideoSequencerProps {
  videos: Video[]
  onVideosChange: (videos: Video[]) => void
}

type SequencedVideo = Video & {
  sequence_order: number
  game_time_offset: number
}

const formatSeconds = (seconds: number) => {
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

const baseSequence = (videos: Video[]): SequencedVideo[] => {
  const sorted = [...videos].sort((a, b) => {
    const orderA = a.sequence_order ?? 0
    const orderB = b.sequence_order ?? 0
    if (orderA === orderB) {
      return a.id - b.id
    }
    return orderA - orderB
  })

  let cursor = 0
  return sorted.map((video, index) => {
    const offset = video.game_time_offset ?? cursor
    cursor = offset + video.duration_seconds
    return {
      ...video,
      sequence_order: index + 1,
      game_time_offset: offset,
    }
  })
}

const colors = ['var(--mantine-color-blue-6)', 'var(--mantine-color-green-6)', 'var(--mantine-color-violet-6)', 'var(--mantine-color-teal-6)', 'var(--mantine-color-pink-6)']

export const VideoSequencer: React.FC<VideoSequencerProps> = ({ videos, onVideosChange }) => {
  const [sequence, setSequence] = useState<SequencedVideo[]>(() => baseSequence(videos))
  const [dragIndex, setDragIndex] = useState<number | null>(null)
  const [isDirty, setIsDirty] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)

  useEffect(() => {
    setSequence(baseSequence(videos))
    setIsDirty(false)
  }, [videos])

  const totalTimeline = useMemo(() => {
    if (sequence.length === 0) return 0
    return sequence.reduce((max, video) => Math.max(max, video.game_time_offset + video.duration_seconds), 0)
  }, [sequence])

  const gaps = useMemo(() => {
    const list: { start: number; end: number }[] = []
    for (let i = 0; i < sequence.length - 1; i += 1) {
      const current = sequence[i]
      const next = sequence[i + 1]
      const currentEnd = current.game_time_offset + current.duration_seconds
      if (next.game_time_offset > currentEnd) {
        list.push({ start: currentEnd, end: next.game_time_offset })
      }
    }
    return list
  }, [sequence])

  const overlaps = useMemo(() => {
    const list: { start: number; end: number; left: SequencedVideo; right: SequencedVideo }[] = []
    for (let i = 0; i < sequence.length - 1; i += 1) {
      const current = sequence[i]
      const next = sequence[i + 1]
      const currentEnd = current.game_time_offset + current.duration_seconds
      if (next.game_time_offset < currentEnd) {
        list.push({
          start: next.game_time_offset,
          end: Math.min(currentEnd, next.game_time_offset + next.duration_seconds),
          left: current,
          right: next,
        })
      }
    }
    return list
  }, [sequence])

  const handleDrop = (targetIndex: number) => {
    if (dragIndex === null || dragIndex === targetIndex) return
    setSequence(prev => {
      const updated = [...prev]
      const [moved] = updated.splice(dragIndex, 1)
      updated.splice(targetIndex, 0, moved)
      return updated.map((video, index) => ({
        ...video,
        sequence_order: index + 1,
      }))
    })
    setDragIndex(null)
    setIsDirty(true)
  }

  const handleOffsetChange = (videoId: number, value: string | number) => {
    const parsed = typeof value === 'number' ? value : parseFloat(value)
    if (Number.isNaN(parsed) || parsed < 0) {
      return
    }

    setSequence(prev =>
      prev.map(video =>
        video.id === videoId
          ? {
              ...video,
              game_time_offset: parsed,
            }
          : video
      )
    )
    setIsDirty(true)
  }

  const moveVideo = (index: number, delta: number) => {
    const target = index + delta
    if (target < 0 || target >= sequence.length) return
    setSequence(prev => {
      const updated = [...prev]
      const [item] = updated.splice(index, 1)
      updated.splice(target, 0, item)
      return updated.map((video, idx) => ({
        ...video,
        sequence_order: idx + 1,
      }))
    })
    setIsDirty(true)
  }

  const autoAlignOffsets = () => {
    let cursor = 0
    setSequence(prev =>
      prev.map(video => {
        const aligned = {
          ...video,
          game_time_offset: cursor,
        }
        cursor += video.duration_seconds
        return aligned
      })
    )
    setIsDirty(true)
  }

  const autoSequenceFromTimestamps = async () => {
    if (sequence.length === 0) return
    const gameId = sequence[0].game_id
    setIsSaving(true)
    setMessage(null)

    try {
      // Call the backend sequencing endpoint
      const response = await fetch(`http://localhost:8000/api/games/${gameId}/timeline/sequence`, {
        method: 'POST',
      })
      
      if (!response.ok) {
        throw new Error('Failed to sequence videos')
      }

      // Reload videos to get updated offsets
      const updatedVideos = await api.videos.listByGame(gameId)
      setSequence(baseSequence(updatedVideos))
      onVideosChange(updatedVideos)
      setIsDirty(false)
      setMessage({ type: 'success', text: 'Videos sequenced from recording timestamps (with gaps)' })
    } catch (error) {
      const text = error instanceof APIError ? error.message : 'Failed to sequence from timestamps'
      setMessage({ type: 'error', text })
    } finally {
      setIsSaving(false)
    }
  }

  const saveChanges = async () => {
    if (!isDirty || isSaving) return
    setIsSaving(true)
    setMessage(null)

    try {
      const payload = sequence.map((video, index) => ({
        id: video.id,
        sequence_order: index + 1,
        game_time_offset: video.game_time_offset,
      }))

      const updated = await Promise.all(
        payload.map(item =>
          api.videos.update(item.id, {
            sequence_order: item.sequence_order,
            game_time_offset: item.game_time_offset,
          })
        )
      )

      setSequence(baseSequence(updated))
      onVideosChange(baseSequence(updated))
      setIsDirty(false)
      setMessage({ type: 'success', text: 'Video order saved' })
    } catch (error) {
      const text = error instanceof APIError ? error.message : 'Failed to save sequence'
      setMessage({ type: 'error', text })
    } finally {
      setIsSaving(false)
    }
  }

  const fileName = (path: string) => path.split('/').pop() ?? path

  return (
    <Paper withBorder p="lg" radius="md">
      <Stack gap="md">
        <Group justify="space-between" align="flex-start">
          <div>
            <Text fw={600}>Video Sequencing</Text>
            <Text size="sm" c="dimmed">
              Drag to reorder, adjust start offsets, and visualize coverage gaps or overlaps.
            </Text>
          </div>
          <Group gap="xs">
            <Button
              variant="light"
              leftSection={<IconRefresh size={16} />}
              onClick={autoSequenceFromTimestamps}
              disabled={sequence.length === 0 || isSaving}
              title="Use recorded_at timestamps to calculate offsets with real gaps"
            >
              Sequence from timestamps
            </Button>
            <Button
              variant="subtle"
              leftSection={<IconRefresh size={16} />}
              onClick={autoAlignOffsets}
              disabled={sequence.length === 0}
              title="Remove gaps and align videos back-to-back"
            >
              Remove gaps
            </Button>
            <Button
              leftSection={<IconDeviceFloppy size={16} />}
              onClick={saveChanges}
              loading={isSaving}
              disabled={!isDirty || sequence.length === 0}
            >
              Save order
            </Button>
          </Group>
        </Group>

        {message && (
          <Alert
            icon={message.type === 'success' ? <IconCheck size={18} /> : <IconAlertCircle size={18} />}
            color={message.type === 'success' ? 'green' : 'red'}
            variant="light"
          >
            {message.text}
          </Alert>
        )}

        <Stack gap="sm">
          {sequence.map((video, index) => (
            <Card
              key={video.id}
              withBorder
              p="md"
              radius="md"
              shadow="sm"
              data-testid={`video-row-${video.id}`}
              draggable
              onDragStart={() => setDragIndex(index)}
              onDragOver={event => event.preventDefault()}
              onDrop={() => handleDrop(index)}
              onDragEnd={() => setDragIndex(null)}
            >
              <Group align="flex-start" gap="md" wrap="nowrap">
                <Box style={{ cursor: 'grab', display: 'flex', alignItems: 'center' }}>
                  <IconGripVertical size={20} />
                </Box>

                <Stack gap={4} style={{ flex: 1 }}>
                  <Group justify="space-between" wrap="nowrap">
                    <Group gap="xs">
                      <Badge variant="filled" color="blue">#{video.sequence_order}</Badge>
                      <Text fw={600}>{fileName(video.file_path)}</Text>
                    </Group>
                    <Group gap={4} wrap="nowrap">
                      <Button
                        variant="subtle"
                        color="gray"
                        size="xs"
                        leftSection={<IconArrowUp size={14} />}
                        onClick={() => moveVideo(index, -1)}
                        disabled={index === 0}
                      >
                        Up
                      </Button>
                      <Button
                        variant="subtle"
                        color="gray"
                        size="xs"
                        leftSection={<IconArrowDown size={14} />}
                        onClick={() => moveVideo(index, 1)}
                        disabled={index === sequence.length - 1}
                      >
                        Down
                      </Button>
                    </Group>
                  </Group>

                  <Group gap="xl" wrap="wrap">
                    <Group gap={6}>
                      <IconClock size={16} />
                      <Text size="sm" c="dimmed">Duration</Text>
                      <Text size="sm">{formatSeconds(video.duration_seconds)}</Text>
                    </Group>
                    <Group gap={6}>
                      <Text size="sm" c="dimmed">Start offset (s)</Text>
                      <NumberInput
                        value={video.game_time_offset}
                        min={0}
                        step={0.5}
                        size="sm"
                        w={120}
                        onChange={value => handleOffsetChange(video.id, value)}
                      />
                      <Badge variant="light" color="gray">
                        {formatSeconds(video.game_time_offset)}
                      </Badge>
                    </Group>
                    {video.recorded_at && (
                      <Group gap={6}>
                        <Text size="sm" c="dimmed">Recorded</Text>
                        <Text size="sm">{new Date(video.recorded_at).toLocaleString()}</Text>
                      </Group>
                    )}
                  </Group>
                </Stack>
              </Group>
            </Card>
          ))}

          {sequence.length === 0 && (
            <Card withBorder p="lg" radius="md">
              <Text c="dimmed" ta="center">
                Upload at least one video to configure sequencing.
              </Text>
            </Card>
          )}
        </Stack>

        {sequence.length > 0 && (
          <Stack gap="sm">
            <div>
              <Text fw={500} mb={4}>Timeline coverage</Text>
              <Box
                style={{
                  position: 'relative',
                  height: 32,
                  borderRadius: 8,
                  backgroundColor: 'var(--mantine-color-dark-6)',
                  overflow: 'hidden',
                }}
              >
                {sequence.map((video, index) => {
                  const left = totalTimeline > 0 ? (video.game_time_offset / totalTimeline) * 100 : 0
                  const width = totalTimeline > 0 ? (video.duration_seconds / totalTimeline) * 100 : 0
                  return (
                    <Box
                      key={`segment-${video.id}`}
                      style={{
                        position: 'absolute',
                        top: 0,
                        bottom: 0,
                        left: `${left}%`,
                        width: `${width}%`,
                        backgroundColor: colors[index % colors.length],
                        opacity: 0.55,
                      }}
                      title={`Video ${video.sequence_order}: ${formatSeconds(video.game_time_offset)} - ${formatSeconds(video.game_time_offset + video.duration_seconds)}`}
                    />
                  )
                })}

                {gaps.map((gap, index) => {
                  const left = totalTimeline > 0 ? (gap.start / totalTimeline) * 100 : 0
                  const width = totalTimeline > 0 ? ((gap.end - gap.start) / totalTimeline) * 100 : 0
                  return (
                    <Box
                      key={`gap-${index}`}
                      style={{
                        position: 'absolute',
                        top: 0,
                        bottom: 0,
                        left: `${left}%`,
                        width: `${width}%`,
                        backgroundColor: 'var(--mantine-color-gray-5)',
                        opacity: 0.35,
                      }}
                      title={`Gap ${formatSeconds(gap.start)} - ${formatSeconds(gap.end)}`}
                    />
                  )
                })}

                {overlaps.map((overlap, index) => {
                  const left = totalTimeline > 0 ? (overlap.start / totalTimeline) * 100 : 0
                  const width = totalTimeline > 0 ? ((overlap.end - overlap.start) / totalTimeline) * 100 : 0
                  return (
                    <Box
                      key={`overlap-${index}`}
                      style={{
                        position: 'absolute',
                        top: 0,
                        bottom: 0,
                        left: `${left}%`,
                        width: `${width}%`,
                        backgroundImage: 'repeating-linear-gradient(45deg, rgba(255,165,0,0.3), rgba(255,165,0,0.3) 6px, transparent 6px, transparent 12px)',
                      }}
                      title={`Overlap ${formatSeconds(overlap.start)} - ${formatSeconds(overlap.end)}`}
                    />
                  )
                })}
              </Box>
            </div>

            <Group gap="lg" align="flex-start" wrap="wrap">
              <Stack gap={2}>
                <Text size="sm" c="dimmed">Coverage end</Text>
                <Text fw={600}>{formatSeconds(totalTimeline)}</Text>
              </Stack>
              <Stack gap={2}>
                <Text size="sm" c="dimmed">Gaps</Text>
                <Text fw={600}>{gaps.length === 0 ? 'None' : `${gaps.length} gap${gaps.length === 1 ? '' : 's'}`}</Text>
              </Stack>
              <Stack gap={2}>
                <Text size="sm" c="dimmed">Overlaps</Text>
                <Text fw={600}>{overlaps.length === 0 ? 'None' : `${overlaps.length} overlap${overlaps.length === 1 ? '' : 's'}`}</Text>
              </Stack>
            </Group>
          </Stack>
        )}
      </Stack>
    </Paper>
  )
}

export default VideoSequencer
