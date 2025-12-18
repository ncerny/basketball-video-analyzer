import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { GameTimelinePlayer } from '../GameTimelinePlayer'
import { useTimelineStore } from '../../store/timelineStore'

// Mock child components to simplify testing
vi.mock('../MultiVideoPlayer', () => ({
  MultiVideoPlayer: () => (
    <div data-testid="multi-video-player">MultiVideoPlayer Mock</div>
  ),
}))

vi.mock('../TimelineBar', () => ({
  TimelineBar: ({ currentTime, totalDuration, onSeek, annotations }: {
    currentTime: number
    totalDuration: number
    onSeek: (time: number) => void
    annotations: unknown[]
  }) => (
    <div data-testid="timeline-bar">
      <span data-testid="current-time">{currentTime}</span>
      <span data-testid="total-duration">{totalDuration}</span>
      <span data-testid="annotation-count">{annotations?.length ?? 0}</span>
      <button onClick={() => onSeek(50)} data-testid="seek-button">Seek</button>
    </div>
  ),
}))

vi.mock('../../hooks/useKeyboardShortcuts', () => ({
  useKeyboardShortcuts: vi.fn(),
}))

// Mock the timeline store
vi.mock('../../store/timelineStore', () => ({
  useTimelineStore: vi.fn(),
}))

const mockUseTimelineStore = vi.mocked(useTimelineStore)

describe('GameTimelinePlayer', () => {
  const mockPlay = vi.fn()
  const mockPause = vi.fn()
  const mockSeekToGameTime = vi.fn()
  const mockSetPlaybackRate = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
    mockUseTimelineStore.mockImplementation((selector) => {
      const state = {
        currentGameTime: 30,
        isPlaying: false,
        playbackRate: 1.0,
        videos: [
          { id: 1, game_time_offset: 0, duration_seconds: 90 },
          { id: 2, game_time_offset: 90, duration_seconds: 90 },
        ],
        annotations: [
          { id: 1, game_timestamp_start: 10, game_timestamp_end: 20 },
        ],
        play: mockPlay,
        pause: mockPause,
        seekToGameTime: mockSeekToGameTime,
        setPlaybackRate: mockSetPlaybackRate,
      }
      return selector(state as never)
    })
  })

  it('renders video player and timeline', () => {
    render(<GameTimelinePlayer />)

    expect(screen.getByTestId('multi-video-player')).toBeInTheDocument()
    expect(screen.getByTestId('timeline-bar')).toBeInTheDocument()
  })

  it('displays play button when paused', () => {
    render(<GameTimelinePlayer />)

    expect(screen.getByRole('button', { name: /play/i })).toBeInTheDocument()
  })

  it('displays pause button when playing', () => {
    mockUseTimelineStore.mockImplementation((selector) => {
      const state = {
        currentGameTime: 30,
        isPlaying: true,
        playbackRate: 1.0,
        videos: [],
        annotations: [],
        play: mockPlay,
        pause: mockPause,
        seekToGameTime: mockSeekToGameTime,
        setPlaybackRate: mockSetPlaybackRate,
      }
      return selector(state as never)
    })

    render(<GameTimelinePlayer />)

    expect(screen.getByRole('button', { name: /pause/i })).toBeInTheDocument()
  })

  it('calls play when clicking play button', async () => {
    const user = userEvent.setup()
    render(<GameTimelinePlayer />)

    await user.click(screen.getByRole('button', { name: /play/i }))

    expect(mockPlay).toHaveBeenCalledTimes(1)
  })

  it('calls pause when clicking pause button while playing', async () => {
    const user = userEvent.setup()
    mockUseTimelineStore.mockImplementation((selector) => {
      const state = {
        currentGameTime: 30,
        isPlaying: true,
        playbackRate: 1.0,
        videos: [],
        annotations: [],
        play: mockPlay,
        pause: mockPause,
        seekToGameTime: mockSeekToGameTime,
        setPlaybackRate: mockSetPlaybackRate,
      }
      return selector(state as never)
    })

    render(<GameTimelinePlayer />)

    await user.click(screen.getByRole('button', { name: /pause/i }))

    expect(mockPause).toHaveBeenCalledTimes(1)
  })

  it('shows skip forward and backward buttons', () => {
    render(<GameTimelinePlayer />)

    expect(screen.getByTitle(/skip back 10s/i)).toBeInTheDocument()
    expect(screen.getByTitle(/skip forward 10s/i)).toBeInTheDocument()
  })

  it('skips backward 10 seconds when clicking skip back button', async () => {
    const user = userEvent.setup()
    render(<GameTimelinePlayer />)

    await user.click(screen.getByTitle(/skip back 10s/i))

    // currentGameTime is 30, should seek to 20
    expect(mockSeekToGameTime).toHaveBeenCalledWith(20)
  })

  it('skips forward 10 seconds when clicking skip forward button', async () => {
    const user = userEvent.setup()
    render(<GameTimelinePlayer />)

    await user.click(screen.getByTitle(/skip forward 10s/i))

    // currentGameTime is 30, should seek to 40
    expect(mockSeekToGameTime).toHaveBeenCalledWith(40)
  })

  it('shows frame navigation buttons when advanced controls enabled', () => {
    render(<GameTimelinePlayer showAdvancedControls={true} />)

    expect(screen.getByTitle(/previous frame/i)).toBeInTheDocument()
    expect(screen.getByTitle(/next frame/i)).toBeInTheDocument()
  })

  it('hides frame navigation buttons when advanced controls disabled', () => {
    render(<GameTimelinePlayer showAdvancedControls={false} />)

    expect(screen.queryByTitle(/previous frame/i)).not.toBeInTheDocument()
    expect(screen.queryByTitle(/next frame/i)).not.toBeInTheDocument()
  })

  it('shows playback speed button with current rate', () => {
    render(<GameTimelinePlayer />)

    expect(screen.getByText('1x')).toBeInTheDocument()
  })

  it('opens speed menu when clicking speed button', async () => {
    const user = userEvent.setup()
    render(<GameTimelinePlayer />)

    await user.click(screen.getByText('1x'))

    // Speed menu should show all speed options
    expect(screen.getByText('0.25x')).toBeInTheDocument()
    expect(screen.getByText('0.5x')).toBeInTheDocument()
    expect(screen.getByText('2x')).toBeInTheDocument()
  })

  it('changes playback speed when selecting from menu', async () => {
    const user = userEvent.setup()
    render(<GameTimelinePlayer />)

    await user.click(screen.getByText('1x'))
    await user.click(screen.getByText('2x'))

    expect(mockSetPlaybackRate).toHaveBeenCalledWith(2.0)
  })

  it('calculates total duration from videos', () => {
    render(<GameTimelinePlayer />)

    // videos: offset 0 + 90 = 90, offset 90 + 90 = 180
    expect(screen.getByTestId('total-duration')).toHaveTextContent('180')
  })

  it('shows keyboard shortcuts hint when advanced controls enabled', () => {
    render(<GameTimelinePlayer showAdvancedControls={true} />)

    expect(screen.getByText(/space/i)).toBeInTheDocument()
    expect(screen.getByText(/play\/pause/i)).toBeInTheDocument()
  })

  it('hides keyboard shortcuts hint when advanced controls disabled', () => {
    render(<GameTimelinePlayer showAdvancedControls={false} />)

    expect(screen.queryByText(/space.*play\/pause/i)).not.toBeInTheDocument()
  })

  it('passes annotations to timeline bar', () => {
    render(<GameTimelinePlayer showAnnotations={true} />)

    expect(screen.getByTestId('annotation-count')).toHaveTextContent('1')
  })

  it('steps forward one frame when clicking next frame', async () => {
    const user = userEvent.setup()
    render(<GameTimelinePlayer showAdvancedControls={true} />)

    await user.click(screen.getByTitle(/next frame/i))

    // Should seek by 1/30 second (one frame at 30fps)
    expect(mockSeekToGameTime).toHaveBeenCalledWith(expect.closeTo(30.033, 2))
  })

  it('steps backward one frame when clicking previous frame', async () => {
    const user = userEvent.setup()
    render(<GameTimelinePlayer showAdvancedControls={true} />)

    await user.click(screen.getByTitle(/previous frame/i))

    // Should seek by -1/30 second
    expect(mockSeekToGameTime).toHaveBeenCalledWith(expect.closeTo(29.967, 2))
  })

  it('applies custom className', () => {
    const { container } = render(<GameTimelinePlayer className="custom-class" />)

    expect(container.firstChild).toHaveClass('custom-class')
  })

  it('clamps skip backward to not go below 0', async () => {
    const user = userEvent.setup()
    mockUseTimelineStore.mockImplementation((selector) => {
      const state = {
        currentGameTime: 5, // Only 5 seconds in
        isPlaying: false,
        playbackRate: 1.0,
        videos: [{ id: 1, game_time_offset: 0, duration_seconds: 90 }],
        annotations: [],
        play: mockPlay,
        pause: mockPause,
        seekToGameTime: mockSeekToGameTime,
        setPlaybackRate: mockSetPlaybackRate,
      }
      return selector(state as never)
    })

    render(<GameTimelinePlayer />)

    await user.click(screen.getByTitle(/skip back 10s/i))

    // Should clamp to 0
    expect(mockSeekToGameTime).toHaveBeenCalledWith(0)
  })

  it('clamps skip forward to not exceed total duration', async () => {
    const user = userEvent.setup()
    mockUseTimelineStore.mockImplementation((selector) => {
      const state = {
        currentGameTime: 85, // Near end of 90 second video
        isPlaying: false,
        playbackRate: 1.0,
        videos: [{ id: 1, game_time_offset: 0, duration_seconds: 90 }],
        annotations: [],
        play: mockPlay,
        pause: mockPause,
        seekToGameTime: mockSeekToGameTime,
        setPlaybackRate: mockSetPlaybackRate,
      }
      return selector(state as never)
    })

    render(<GameTimelinePlayer />)

    await user.click(screen.getByTitle(/skip forward 10s/i))

    // Should clamp to 90 (total duration)
    expect(mockSeekToGameTime).toHaveBeenCalledWith(90)
  })
})
