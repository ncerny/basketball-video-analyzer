import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, within, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { VideoSequencer } from '../VideoSequencer'
import type { Video } from '../../types/api'
import { MantineProvider } from '@mantine/core'
import type { ReactNode } from 'react'
const renderWithProvider = (ui: ReactNode) =>
  render(<MantineProvider defaultColorScheme="dark">{ui}</MantineProvider>)


const updateMock = vi.hoisted(() => vi.fn())

vi.mock('../../services/api', () => ({
  api: {
    videos: {
      update: updateMock,
    },
  },
}))

const createVideos = (): Video[] => [
  {
    id: 1,
    game_id: 42,
    file_path: '/videos/cam_a.mp4',
    duration_seconds: 90,
    fps: 30,
    resolution: '1920x1080',
    upload_date: '2025-12-10T00:00:00Z',
    processed: true,
    processing_status: 'completed',
    recorded_at: '2025-12-10T00:00:00Z',
    sequence_order: 1,
    game_time_offset: 0,
  },
  {
    id: 2,
    game_id: 42,
    file_path: '/videos/cam_b.mp4',
    duration_seconds: 75,
    fps: 30,
    resolution: '1920x1080',
    upload_date: '2025-12-10T00:05:00Z',
    processed: true,
    processing_status: 'completed',
    recorded_at: '2025-12-10T00:05:00Z',
    sequence_order: 2,
    game_time_offset: 90,
  },
]

describe('VideoSequencer', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders video rows with metadata', () => {
    renderWithProvider(<VideoSequencer videos={createVideos()} onVideosChange={vi.fn()} />)

    expect(screen.getByText('Video Sequencing')).toBeInTheDocument()
    expect(screen.getByText('cam_a.mp4')).toBeInTheDocument()
    expect(screen.getByText('cam_b.mp4')).toBeInTheDocument()
  })

  it('moves videos using reorder controls', async () => {
    const user = userEvent.setup()
    renderWithProvider(<VideoSequencer videos={createVideos()} onVideosChange={vi.fn()} />)

    const rows = screen.getAllByTestId(/video-row-/)
    expect(within(rows[0]).getByText('cam_a.mp4')).toBeInTheDocument()

    const downButton = within(rows[0]).getByRole('button', { name: /down/i })
    await user.click(downButton)

    const reorderedRows = screen.getAllByTestId(/video-row-/)
    expect(within(reorderedRows[0]).getByText('cam_b.mp4')).toBeInTheDocument()
  })

  it('persists order changes and notifies parent', async () => {
    const user = userEvent.setup()
    const onVideosChange = vi.fn()
    const videos = createVideos()
    const baseById = Object.fromEntries(videos.map(video => [video.id, video]))

    updateMock.mockImplementation(async (id: number, payload: Partial<Video>) => ({
      ...baseById[id],
      ...payload,
    }))

    renderWithProvider(<VideoSequencer videos={videos} onVideosChange={onVideosChange} />)

    const downButton = within(screen.getAllByTestId(/video-row-/)[0]).getByRole('button', { name: /down/i })
    await user.click(downButton)

    await user.click(screen.getByRole('button', { name: /save order/i }))

    await waitFor(() => expect(updateMock).toHaveBeenCalledTimes(videos.length))
    expect(onVideosChange).toHaveBeenCalledWith(
      expect.arrayContaining([
        expect.objectContaining({ id: 2, sequence_order: 1 }),
        expect.objectContaining({ id: 1, sequence_order: 2 }),
      ])
    )
    expect(screen.getByText('Video order saved')).toBeInTheDocument()
  })
})
