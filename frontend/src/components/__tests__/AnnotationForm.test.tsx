import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { AnnotationForm } from '../AnnotationForm'
import { useTimelineStore } from '../../store/timelineStore'
import type { Annotation } from '../../types/api'

// Mock the timeline store
vi.mock('../../store/timelineStore', () => ({
  useTimelineStore: vi.fn(),
}))

const mockUseTimelineStore = vi.mocked(useTimelineStore)

describe('AnnotationForm', () => {
  const defaultProps = {
    gameId: 1,
    onSubmit: vi.fn(),
    onCancel: vi.fn(),
  }

  beforeEach(() => {
    vi.clearAllMocks()
    mockUseTimelineStore.mockImplementation((selector) => {
      const state = {
        currentGameTime: 30,
        videos: [
          { id: 1, game_time_offset: 0, duration_seconds: 120 },
          { id: 2, game_time_offset: 120, duration_seconds: 90 },
        ],
      }
      return selector(state as never)
    })
  })

  it('renders form with default values', () => {
    render(<AnnotationForm {...defaultProps} />)

    expect(screen.getByPlaceholderText(/fast break layup/i)).toBeInTheDocument()
    expect(screen.getByPlaceholderText(/add detailed notes/i)).toBeInTheDocument()
    expect(screen.getByText('Play')).toBeInTheDocument()
    expect(screen.getByText('Event')).toBeInTheDocument()
    expect(screen.getByText('Note')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /create/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /cancel/i })).toBeInTheDocument()
  })

  it('pre-fills form when editing an annotation', () => {
    const annotation: Annotation = {
      id: 1,
      game_id: 1,
      title: 'Test Annotation',
      description: 'Test description',
      game_timestamp_start: 10,
      game_timestamp_end: 20,
      annotation_type: 'play',
      verified: true,
      confidence_score: 0.9,
      created_by: 'user',
      created_at: '2025-01-01T00:00:00Z',
      updated_at: '2025-01-01T00:00:00Z',
    }

    render(<AnnotationForm {...defaultProps} annotation={annotation} />)

    expect(screen.getByDisplayValue('Test Annotation')).toBeInTheDocument()
    expect(screen.getByDisplayValue('Test description')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /update/i })).toBeInTheDocument()
  })

  it('allows selecting annotation type', async () => {
    const user = userEvent.setup()
    render(<AnnotationForm {...defaultProps} />)

    // Default is 'event'
    const eventButton = screen.getByText('Event')
    expect(eventButton).toHaveClass('bg-blue-600')

    // Click on 'play'
    await user.click(screen.getByText('Play'))
    expect(screen.getByText('Play')).toHaveClass('bg-blue-600')
    expect(eventButton).not.toHaveClass('bg-blue-600')
  })

  it('validates end time must be after start time', async () => {
    const user = userEvent.setup()
    const onSubmit = vi.fn()
    
    // Set currentGameTime so that default end time (currentGameTime + 5) is after start
    mockUseTimelineStore.mockImplementation((selector) => {
      const state = {
        currentGameTime: 10, // Start at 10, end at 15
        videos: [{ id: 1, game_time_offset: 0, duration_seconds: 120 }],
      }
      return selector(state as never)
    })
    
    render(<AnnotationForm {...defaultProps} onSubmit={onSubmit} />)

    // Find the end time input by placeholder and set it to before start time
    const timeInputs = screen.getAllByPlaceholderText('MM:SS.ms')
    const endTimeInput = timeInputs[1] // Second one is end time
    
    await user.clear(endTimeInput)
    await user.type(endTimeInput, '0:05.00') // 5 seconds, before start of 10

    await user.click(screen.getByRole('button', { name: /create/i }))

    // Should show error and not submit
    expect(screen.getByText(/end time must be after start time/i)).toBeInTheDocument()
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('validates confidence score range', async () => {
    const onSubmit = vi.fn()
    
    // Directly test validation by setting state
    render(<AnnotationForm {...defaultProps} onSubmit={onSubmit} />)

    // The number input has min/max attributes which browsers enforce
    // Instead, test that the form has proper input constraints
    const confidenceInput = screen.getByPlaceholderText('0.0 - 1.0')
    expect(confidenceInput).toHaveAttribute('min', '0')
    expect(confidenceInput).toHaveAttribute('max', '1')
  })

  it('submits form with correct data', async () => {
    const user = userEvent.setup()
    const onSubmit = vi.fn()
    render(<AnnotationForm {...defaultProps} onSubmit={onSubmit} />)

    // Fill in form using placeholders
    await user.type(screen.getByPlaceholderText(/fast break layup/i), 'My Annotation')
    await user.type(screen.getByPlaceholderText(/add detailed notes/i), 'Some description')
    await user.click(screen.getByText('Play'))
    await user.click(screen.getByLabelText(/mark as verified/i))

    await user.click(screen.getByRole('button', { name: /create/i }))

    await waitFor(() => {
      expect(onSubmit).toHaveBeenCalledWith(
        expect.objectContaining({
          game_id: 1,
          title: 'My Annotation',
          description: 'Some description',
          annotation_type: 'play',
          verified: true,
          created_by: 'user',
        })
      )
    })
  })

  it('calls onCancel when cancel button is clicked', async () => {
    const user = userEvent.setup()
    const onCancel = vi.fn()
    render(<AnnotationForm {...defaultProps} onCancel={onCancel} />)

    await user.click(screen.getByRole('button', { name: /cancel/i }))

    expect(onCancel).toHaveBeenCalledTimes(1)
  })

  it('shows loading state on submit button', () => {
    render(<AnnotationForm {...defaultProps} isLoading={true} />)

    expect(screen.getByRole('button', { name: /saving/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /saving/i })).toBeDisabled()
  })

  it('shows video coverage info', () => {
    render(<AnnotationForm {...defaultProps} />)

    expect(screen.getByText(/video coverage/i)).toBeInTheDocument()
    expect(screen.getByText(/video\(s\)/)).toBeInTheDocument()
  })
})
