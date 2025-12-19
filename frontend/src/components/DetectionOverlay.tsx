/**
 * Detection Overlay Component
 *
 * Displays ML detection bounding boxes over the video player.
 * Shows player detections with tracking IDs and confidence scores.
 */

import { useEffect, useState, useMemo } from 'react'
import { Box, Text } from '@mantine/core'
import type { Detection } from '../types/api'

interface DetectionOverlayProps {
  /** Video element to overlay detections on */
  videoElement: HTMLVideoElement | null

  /** All detections for the video */
  detections: Detection[]

  /** Current frame number (calculated from video.currentTime * fps) */
  currentFrame: number

  /** Show/hide the overlay */
  visible?: boolean

  /** Minimum confidence threshold to display */
  minConfidence?: number

  /** Show confidence scores on boxes */
  showConfidence?: boolean

  /** Show tracking IDs on boxes */
  showTrackingId?: boolean

  /** Container ref for positioning */
  containerRef: React.RefObject<HTMLDivElement | null>
}

// Generate consistent color for tracking ID
const getTrackingColor = (trackingId: number): string => {
  const colors = [
    '#FF6B6B', // red
    '#4ECDC4', // teal
    '#45B7D1', // blue
    '#FFA07A', // salmon
    '#98D8C8', // mint
    '#F7DC6F', // yellow
    '#BB8FCE', // purple
    '#85C1E2', // light blue
    '#F8B88B', // orange
    '#ABEBC6', // light green
  ]
  return colors[trackingId % colors.length]
}

export const DetectionOverlay: React.FC<DetectionOverlayProps> = ({
  videoElement,
  detections,
  currentFrame,
  visible = true,
  minConfidence = 0.5,
  showConfidence = true,
  showTrackingId = true,
  containerRef,
}) => {
  const [videoDimensions, setVideoDimensions] = useState({ width: 0, height: 0 })
  const [containerDimensions, setContainerDimensions] = useState({ width: 0, height: 0 })

  // Get current frame detections
  const currentDetections = useMemo(() => {
    return detections.filter(
      d => d.frame_number === currentFrame && d.confidence_score >= minConfidence
    )
  }, [detections, currentFrame, minConfidence])

  // Update video dimensions when video loads
  useEffect(() => {
    if (!videoElement) return

    const updateDimensions = () => {
      setVideoDimensions({
        width: videoElement.videoWidth,
        height: videoElement.videoHeight,
      })
    }

    // Set initial dimensions
    updateDimensions()

    // Listen for metadata load
    videoElement.addEventListener('loadedmetadata', updateDimensions)
    videoElement.addEventListener('resize', updateDimensions)

    return () => {
      videoElement.removeEventListener('loadedmetadata', updateDimensions)
      videoElement.removeEventListener('resize', updateDimensions)
    }
  }, [videoElement])

  // Update container dimensions
  useEffect(() => {
    if (!containerRef.current) return

    const updateContainerSize = () => {
      const rect = containerRef.current?.getBoundingClientRect()
      if (rect) {
        setContainerDimensions({
          width: rect.width,
          height: rect.height,
        })
      }
    }

    updateContainerSize()

    const resizeObserver = new ResizeObserver(updateContainerSize)
    resizeObserver.observe(containerRef.current)

    return () => {
      resizeObserver.disconnect()
    }
  }, [containerRef])

  // Don't render if not visible or no video
  if (!visible || !videoElement || videoDimensions.width === 0) {
    return null
  }

  // Calculate scale factors for bounding boxes
  // Video is displayed with object-fit: contain, so we need to calculate the actual display size
  const videoAspect = videoDimensions.width / videoDimensions.height
  const containerAspect = containerDimensions.width / containerDimensions.height

  let displayWidth: number
  let displayHeight: number
  let offsetX = 0
  let offsetY = 0

  if (videoAspect > containerAspect) {
    // Video is wider - fit to width
    displayWidth = containerDimensions.width
    displayHeight = containerDimensions.width / videoAspect
    offsetY = (containerDimensions.height - displayHeight) / 2
  } else {
    // Video is taller - fit to height
    displayHeight = containerDimensions.height
    displayWidth = containerDimensions.height * videoAspect
    offsetX = (containerDimensions.width - displayWidth) / 2
  }

  const scaleX = displayWidth / videoDimensions.width
  const scaleY = displayHeight / videoDimensions.height

  return (
    <Box
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        zIndex: 10,
      }}
    >
      <svg
        width="100%"
        height="100%"
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
        }}
      >
        {currentDetections.map(detection => {
          const x = detection.bbox.x * scaleX + offsetX
          const y = detection.bbox.y * scaleY + offsetY
          const width = detection.bbox.width * scaleX
          const height = detection.bbox.height * scaleY
          const color = getTrackingColor(detection.tracking_id)

          return (
            <g key={detection.id}>
              {/* Bounding box */}
              <rect
                x={x}
                y={y}
                width={width}
                height={height}
                fill="none"
                stroke={color}
                strokeWidth={2}
                opacity={0.8}
              />

              {/* Label background */}
              <rect
                x={x}
                y={y - 20}
                width={showConfidence ? 80 : 40}
                height={18}
                fill={color}
                opacity={0.8}
              />

              {/* Tracking ID and confidence */}
              <text
                x={x + 4}
                y={y - 6}
                fill="white"
                fontSize="12"
                fontWeight="bold"
                fontFamily="monospace"
              >
                {showTrackingId && `#${detection.tracking_id}`}
                {showConfidence &&
                  ` ${Math.round(detection.confidence_score * 100)}%`}
              </text>

              {/* Player ID if assigned */}
              {detection.player_id && (
                <>
                  <rect
                    x={x}
                    y={y + height}
                    width={60}
                    height={18}
                    fill="#1a1b1e"
                    opacity={0.9}
                  />
                  <text
                    x={x + 4}
                    y={y + height + 13}
                    fill={color}
                    fontSize="11"
                    fontWeight="bold"
                    fontFamily="monospace"
                  >
                    P{detection.player_id}
                  </text>
                </>
              )}
            </g>
          )
        })}
      </svg>

      {/* Detection count indicator */}
      {currentDetections.length > 0 && (
        <Box
          style={{
            position: 'absolute',
            top: 10,
            right: 10,
            backgroundColor: 'rgba(0, 0, 0, 0.7)',
            borderRadius: 4,
            padding: '4px 8px',
          }}
        >
          <Text size="xs" c="white" fw={600} ff="monospace">
            {currentDetections.length} detection{currentDetections.length !== 1 ? 's' : ''}
          </Text>
        </Box>
      )}
    </Box>
  )
}

export default DetectionOverlay
