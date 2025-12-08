/**
 * Keyboard shortcuts hook for video player controls
 */

import { useEffect } from 'react';

export interface KeyboardShortcuts {
  /** Space: Play/Pause */
  onPlayPause?: () => void;

  /** Arrow Left: Previous frame */
  onPreviousFrame?: () => void;

  /** Arrow Right: Next frame */
  onNextFrame?: () => void;

  /** J: Skip backward */
  onSkipBackward?: () => void;

  /** L: Skip forward */
  onSkipForward?: () => void;

  /** Arrow Up: Increase speed */
  onIncreaseSpeed?: () => void;

  /** Arrow Down: Decrease speed */
  onDecreaseSpeed?: () => void;

  /** M: Toggle mute (for future use) */
  onToggleMute?: () => void;

  /** F: Toggle fullscreen (for future use) */
  onToggleFullscreen?: () => void;
}

interface UseKeyboardShortcutsOptions {
  /** Enable keyboard shortcuts (default: true) */
  enabled?: boolean;

  /** Prevent default behavior for shortcuts (default: true) */
  preventDefault?: boolean;
}

/**
 * Hook to add keyboard shortcuts for video player
 */
export const useKeyboardShortcuts = (
  shortcuts: KeyboardShortcuts,
  options: UseKeyboardShortcutsOptions = {}
) => {
  const { enabled = true, preventDefault = true } = options;

  useEffect(() => {
    if (!enabled) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      // Don't trigger shortcuts if user is typing in an input
      if (
        event.target instanceof HTMLInputElement ||
        event.target instanceof HTMLTextAreaElement ||
        event.target instanceof HTMLSelectElement
      ) {
        return;
      }

      let handled = false;

      switch (event.key.toLowerCase()) {
        case ' ':
          shortcuts.onPlayPause?.();
          handled = true;
          break;

        case 'arrowleft':
          shortcuts.onPreviousFrame?.();
          handled = true;
          break;

        case 'arrowright':
          shortcuts.onNextFrame?.();
          handled = true;
          break;

        case 'j':
          shortcuts.onSkipBackward?.();
          handled = true;
          break;

        case 'l':
          shortcuts.onSkipForward?.();
          handled = true;
          break;

        case 'arrowup':
          shortcuts.onIncreaseSpeed?.();
          handled = true;
          break;

        case 'arrowdown':
          shortcuts.onDecreaseSpeed?.();
          handled = true;
          break;

        case 'm':
          shortcuts.onToggleMute?.();
          handled = true;
          break;

        case 'f':
          shortcuts.onToggleFullscreen?.();
          handled = true;
          break;
      }

      if (handled && preventDefault) {
        event.preventDefault();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [shortcuts, enabled, preventDefault]);
};

export default useKeyboardShortcuts;
