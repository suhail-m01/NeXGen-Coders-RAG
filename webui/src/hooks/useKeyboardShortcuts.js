import { useCallback, useEffect } from 'react'

/**
 * Global keyboard shortcuts, as a map from a chord to a handler.
 *
 *   useKeyboardShortcuts({
 *     'ctrl+k': () => focusQuestion(),
 *     'alt+1':  () => setTab('ask'),
 *     '?':      () => setHelp(true),
 *     'escape': () => setHelp(false),
 *   })
 *
 * Chords are written lowercase, modifiers first, in the order ctrl (or cmd) /
 * alt / shift, then the key: 'ctrl+shift+p', 'alt+2', '?', 'escape'.
 *
 * One rule decides whether a keystroke is a shortcut or a character, and it is
 * worth being explicit about because the obvious version is wrong. While the user
 * is typing in a field, *bare* keys belong to the field: pressing '?' in the
 * question box has to type a question mark, not open the help overlay. Modifier
 * chords still fire, because Ctrl+K and Alt+1 are not text and a shortcut that
 * stops working the moment a field has focus is a shortcut nobody can rely on --
 * which is exactly where the user is when they want to jump to the question box.
 * Escape always fires, so an overlay can be closed from anywhere.
 */
export default function useKeyboardShortcuts(shortcuts) {
  const onKeyDown = useCallback(
    (event) => {
      const target = event.target
      const typing =
        target &&
        (target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.tagName === 'SELECT' ||
          target.isContentEditable)

      const parts = []
      if (event.ctrlKey || event.metaKey) parts.push('ctrl')
      if (event.altKey) parts.push('alt')
      if (event.shiftKey) parts.push('shift')
      const key = (event.key || '').toLowerCase()
      const bare = parts.length === 0
      if (!['control', 'shift', 'alt', 'meta'].includes(key)) parts.push(key)
      const chord = parts.join('+')

      if (typing && bare && key !== 'escape') return

      const handler = shortcuts[chord] || shortcuts[key]
      if (!handler) return
      event.preventDefault()
      handler(event)
    },
    [shortcuts],
  )

  useEffect(() => {
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [onKeyDown])
}
