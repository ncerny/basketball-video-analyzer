import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { TimelineBar } from '../TimelineBar'
import { useVideoSegments, useVideoGaps, useVideoOverlaps } from '../../store/timelineStore'
import type { Annotation } from '../../types/timeline'

// Mock the timeline store hooks
vi.mock('../../store/timelineStore', () => ({
  useVideoSegments: vi.fn(),
  useVideoGaps: vi.fn(),
  useVideoOverlaps: vi.fn(),
}))

const mockUseVideoSegments = vi.mocked(useVideoSegments)
const mockUseVideoGaps = vi.mocked(useVideoGaps)
const mockUseVideoOverlaps = vi.mocked(useVideoOverlaps)

describe('TimelineBar', () => {
  const defaultProps = {
    currentTime: 30,
    totalDuration: 180,
    onSeek: vi.fn(),
  }

  beforeEach(() => {
    vi.clearAllMocks()
    mockUseVideoSegments.mockReturnValue([
      { videoId: 1, start: 0, end: 90, duration: 90 },
      { videoId: 2, start: 90, end: 180, duration: 90 },
    ])
    mockUseVideoGaps.mockReturnValue([])
    mockUseVideoOverlaps.mockReturnValue([])
  })

  it('renders timeline with current time and total duration', () => {
    render(<TimelineBar {...defaultProps} />)

    expect(screen.getByText('0:30')).toBeInTheDocument()
    expect(screen.getByText('3:00')).toBeInTheDocument()
  })

  it('shows video segment count in stats', () => {
    render(<TimelineBar {...defaultProps} />)

    expect(screen.getByText(/2 videos/)).toBeInTheDocument()
  })

  it('shows gap count when gaps exist', () => {
    mockUseVideoGaps.mockReturnValue([
      { start: 60, end: 70 },
    ])

    render(<TimelineBar {...defaultProps} />)

    expect(screen.getByText(/1 gap/)).toBeInTheDocument()
  })

  it('shows overlap count when overlaps exist', () => {
    mockUseVideoOverlaps.mockReturnValue([
      { start: 85, end: 95, videoIds: [1, 2] },
    ])

    render(<TimelineBar {...defaultProps} />)

    expect(screen.getByText(/1 overlap/)).toBeInTheDocument()
  })

  it('calls onSeek when timeline is clicked', async () => {
    const user = userEvent.setup()
    const onSeek = vi.fn()
    render(<TimelineBar {...defaultProps} onSeek={onSeek} />)

    // Find the timeline scrubber div (not the range input)
    const slider = screen.getByLabelText('Timeline scrubber')
    await user.click(slider)

    expect(onSeek).toHaveBeenCalled()
  })

  it('renders annotation markers', () => {
    const annotations: Annotation[] = [
      {
        id: 1,
        game_id: 1,
        game_timestamp_start: 10,
        game_timestamp_end: 20,
        annotation_type: 'play',
        verified: true,
        created_by: 'user',
      },
      {
        id: 2,
        game_id: 1,
        game_timestamp_start: 50,
        game_timestamp_end: 60,
        annotation_type: 'event',
        verified: false,
        created_by: 'ai',
      },
    ]

    render(<TimelineBar {...defaultProps} annotations={annotations} showAnnotations={true} />)

    expect(screen.getByText(/2 annotations/)).toBeInTheDocument()
  })

  it('hides annotations when showAnnotations is false', () => {
    const annotations: Annotation[] = [
      {
        id: 1,
        game_id: 1,
        game_timestamp_start: 10,
        game_timestamp_end: 20,
        annotation_type: 'play',
        verified: true,
        created_by: 'user',
      },
    ]

    render(<TimelineBar {...defaultProps} annotations={annotations} showAnnotations={false} />)

    expect(screen.queryByText(/annotations/)).not.toBeInTheDocument()
  })

  it('hides time labels when showTimeLabels is false', () => {
    render(<TimelineBar {...defaultProps} showTimeLabels={false} />)

    // Should not show the MM:SS time format
    expect(screen.queryByText('0:30')).not.toBeInTheDocument()
    expect(screen.queryByText('3:00')).not.toBeInTheDocument()
  })

  it('hides stats when showStats is false', () => {
    render(<TimelineBar {...defaultProps} showStats={false} />)

    expect(screen.queryByText(/videos/)).not.toBeInTheDocument()
  })

  it('is not interactive when interactive is false', () => {
    render(<TimelineBar {...defaultProps} interactive={false} />)

    const slider = screen.getByLabelText('Timeline scrubber')
    expect(slider).toHaveAttribute('tabindex', '-1')
  })

  it('renders custom height', () => {
    const { container } = render(<TimelineBar {...defaultProps} height={100} />)

    const track = container.querySelector('[role="slider"]')
    expect(track).toHaveStyle({ height: '100px' })
  })

  it('handles empty video segments gracefully', () => {
    mockUseVideoSegments.mockReturnValue([])

    render(<TimelineBar {...defaultProps} />)

    expect(screen.getByText(/0 videos/)).toBeInTheDocument()
  })

  it('positions current time indicator correctly', () => {
    // currentTime is 30 out of 180 total = 16.67%
    const { container } = render(<TimelineBar {...defaultProps} currentTime={30} totalDuration={180} />)

    const indicator = container.querySelector('.bg-white.shadow-lg')
    expect(indicator).toBeInTheDocument()
    // The indicator should have left style set to approximately 16.67%
    const style = indicator?.getAttribute('style')
    expect(style).toMatch(/left:\s*16\.666/)
  })

  it('clamps seek position to valid range', async () => {
    const user = userEvent.setup()
    const onSeek = vi.fn()
    render(<TimelineBar {...defaultProps} onSeek={onSeek} totalDuration={100} />)

    // Click the timeline scrubber - onSeek should be called
    const slider = screen.getByLabelText('Timeline scrubber')
    await user.click(slider)
    
    // The onSeek should be called
    expect(onSeek).toHaveBeenCalled()
  })

  it('uses correct colors for different annotation types', () => {
    const annotations: Annotation[] = [
      { id: 1, game_id: 1, game_timestamp_start: 10, game_timestamp_end: 15, annotation_type: 'play', verified: true, created_by: 'user' },
      { id: 2, game_id: 1, game_timestamp_start: 20, game_timestamp_end: 25, annotation_type: 'event', verified: true, created_by: 'user' },
      { id: 3, game_id: 1, game_timestamp_start: 30, game_timestamp_end: 35, annotation_type: 'note', verified: true, created_by: 'user' },
    ]

    const { container } = render(<TimelineBar {...defaultProps} annotations={annotations} showAnnotations={true} />)

    // Check for the color classes
    expect(container.querySelector('.bg-green-400')).toBeInTheDocument() // play
    expect(container.querySelector('.bg-yellow-400')).toBeInTheDocument() // event
    expect(container.querySelector('.bg-purple-400')).toBeInTheDocument() // note
  })
})
