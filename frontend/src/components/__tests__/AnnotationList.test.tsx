import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { AnnotationList } from '../AnnotationList'
import { useTimelineStore } from '../../store/timelineStore'
import type { Annotation } from '../../types/api'

// Mock the timeline store
vi.mock('../../store/timelineStore', () => ({
  useTimelineStore: vi.fn(),
}))

const mockUseTimelineStore = vi.mocked(useTimelineStore)
const mockSeekToGameTime = vi.fn()

const createAnnotations = (): Annotation[] => [
  {
    id: 1,
    game_id: 1,
    title: 'Fast break',
    description: 'Quick transition play',
    game_timestamp_start: 10,
    game_timestamp_end: 20,
    annotation_type: 'play',
    verified: true,
    confidence_score: 0.95,
    created_by: 'user',
    created_at: '2025-01-01T00:00:00Z',
    updated_at: '2025-01-01T00:00:00Z',
  },
  {
    id: 2,
    game_id: 1,
    title: 'Timeout',
    description: 'Coach called timeout',
    game_timestamp_start: 30,
    game_timestamp_end: 35,
    annotation_type: 'event',
    verified: false,
    confidence_score: 0.8,
    created_by: 'ai',
    created_at: '2025-01-01T00:01:00Z',
    updated_at: '2025-01-01T00:01:00Z',
  },
  {
    id: 3,
    game_id: 1,
    title: 'Player substitution note',
    description: 'Note about lineup change',
    game_timestamp_start: 50,
    game_timestamp_end: 52,
    annotation_type: 'note',
    verified: true,
    confidence_score: undefined,
    created_by: 'user',
    created_at: '2025-01-01T00:02:00Z',
    updated_at: '2025-01-01T00:02:00Z',
  },
]

describe('AnnotationList', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockUseTimelineStore.mockImplementation((selector) => {
      const state = {
        seekToGameTime: mockSeekToGameTime,
        currentGameTime: 15, // Active during first annotation
      }
      return selector(state as never)
    })
  })

  it('renders list of annotations', () => {
    render(<AnnotationList annotations={createAnnotations()} />)

    expect(screen.getByText('Fast break')).toBeInTheDocument()
    expect(screen.getByText('Timeout')).toBeInTheDocument()
    expect(screen.getByText('Player substitution note')).toBeInTheDocument()
  })

  it('shows empty state when no annotations', () => {
    render(<AnnotationList annotations={[]} />)

    expect(screen.getByText(/no annotations found/i)).toBeInTheDocument()
  })

  it('shows loading state', () => {
    render(<AnnotationList annotations={[]} isLoading={true} />)

    expect(screen.getByText(/loading annotations/i)).toBeInTheDocument()
  })

  it('displays annotation count', () => {
    render(<AnnotationList annotations={createAnnotations()} />)

    expect(screen.getByText(/showing 3 of 3 annotations/i)).toBeInTheDocument()
  })

  it('filters by annotation type', async () => {
    const user = userEvent.setup()
    render(<AnnotationList annotations={createAnnotations()} />)

    // Click on 'Play' filter
    await user.click(screen.getByRole('button', { name: /^play$/i }))

    expect(screen.getByText('Fast break')).toBeInTheDocument()
    expect(screen.queryByText('Timeout')).not.toBeInTheDocument()
    expect(screen.queryByText('Player substitution note')).not.toBeInTheDocument()
    expect(screen.getByText(/showing 1 of 3 annotations/i)).toBeInTheDocument()
  })

  it('filters by verified status', async () => {
    const user = userEvent.setup()
    render(<AnnotationList annotations={createAnnotations()} />)

    // Check verified only checkbox
    await user.click(screen.getByLabelText(/verified only/i))

    expect(screen.getByText('Fast break')).toBeInTheDocument()
    expect(screen.getByText('Player substitution note')).toBeInTheDocument()
    expect(screen.queryByText('Timeout')).not.toBeInTheDocument()
    expect(screen.getByText(/showing 2 of 3 annotations/i)).toBeInTheDocument()
  })

  it('sorts by time ascending by default', () => {
    render(<AnnotationList annotations={createAnnotations()} />)

    const annotationElements = screen.getAllByText(/\d+:\d+ - \d+:\d+/)
    expect(annotationElements[0]).toHaveTextContent('0:10')
    expect(annotationElements[1]).toHaveTextContent('0:30')
    expect(annotationElements[2]).toHaveTextContent('0:50')
  })

  it('toggles sort order when clicking sort button', async () => {
    const user = userEvent.setup()
    render(<AnnotationList annotations={createAnnotations()} />)

    // Click time sort to reverse order
    await user.click(screen.getByRole('button', { name: /time/i }))

    const annotationElements = screen.getAllByText(/\d+:\d+ - \d+:\d+/)
    expect(annotationElements[0]).toHaveTextContent('0:50')
    expect(annotationElements[2]).toHaveTextContent('0:10')
  })

  it('seeks to annotation time on click', async () => {
    const user = userEvent.setup()
    render(<AnnotationList annotations={createAnnotations()} />)

    await user.click(screen.getByText('Timeout'))

    expect(mockSeekToGameTime).toHaveBeenCalledWith(30)
  })

  it('highlights active annotation', () => {
    // currentGameTime is 15, which is within first annotation (10-20)
    render(<AnnotationList annotations={createAnnotations()} />)

    // Find the annotation card containing 'Fast break'
    const fastBreakCard = screen.getByText('Fast break').closest('div[class*="rounded-lg"]')
    expect(fastBreakCard).toHaveClass('border-blue-500')
  })

  it('calls onEdit when edit button is clicked', async () => {
    const user = userEvent.setup()
    const onEdit = vi.fn()
    const annotations = createAnnotations()
    render(<AnnotationList annotations={annotations} onEdit={onEdit} />)

    const editButtons = screen.getAllByRole('button', { name: /edit/i })
    await user.click(editButtons[0])

    expect(onEdit).toHaveBeenCalledWith(annotations[0])
  })

  it('calls onDelete when delete button is clicked and confirmed', async () => {
    const user = userEvent.setup()
    const onDelete = vi.fn()
    const confirmMock = vi.fn(() => true)
    vi.stubGlobal('confirm', confirmMock)

    render(<AnnotationList annotations={createAnnotations()} onDelete={onDelete} />)

    const deleteButtons = screen.getAllByRole('button', { name: /delete/i })
    await user.click(deleteButtons[0])

    expect(onDelete).toHaveBeenCalledWith(1)
    vi.unstubAllGlobals()
  })

  it('does not call onDelete when deletion is cancelled', async () => {
    const user = userEvent.setup()
    const onDelete = vi.fn()
    const confirmMock = vi.fn(() => false)
    vi.stubGlobal('confirm', confirmMock)

    render(<AnnotationList annotations={createAnnotations()} onDelete={onDelete} />)

    const deleteButtons = screen.getAllByRole('button', { name: /delete/i })
    await user.click(deleteButtons[0])

    expect(onDelete).not.toHaveBeenCalled()
    vi.unstubAllGlobals()
  })

  it('calls onVerify for unverified annotations', async () => {
    const user = userEvent.setup()
    const onVerify = vi.fn()
    render(<AnnotationList annotations={createAnnotations()} onVerify={onVerify} />)

    // Only unverified annotation (id: 2) should have verify button
    const verifyButtons = screen.getAllByRole('button', { name: /verify/i })
    expect(verifyButtons).toHaveLength(1)

    await user.click(verifyButtons[0])
    expect(onVerify).toHaveBeenCalledWith(2)
  })

  it('shows confidence score as percentage', () => {
    render(<AnnotationList annotations={createAnnotations()} />)

    expect(screen.getByText('95% confident')).toBeInTheDocument()
    expect(screen.getByText('80% confident')).toBeInTheDocument()
  })

  it('shows creation source (AI vs manual)', () => {
    render(<AnnotationList annotations={createAnnotations()} />)

    // Use getAllBy since there are multiple manually created annotations
    expect(screen.getAllByText(/created manually/i).length).toBeGreaterThan(0)
    expect(screen.getByText(/created by ai/i)).toBeInTheDocument()
  })

  it('shows verification checkmark for verified annotations', () => {
    render(<AnnotationList annotations={createAnnotations()} />)

    // Should have 2 verification checkmarks (annotations 1 and 3)
    const checkmarks = screen.getAllByTitle('Verified')
    expect(checkmarks).toHaveLength(2)
  })
})
