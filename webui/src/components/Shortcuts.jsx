import React from 'react'

/**
 * The keyboard-shortcut list, shown as a dialog.
 *
 * It lives in its own component for a practical reason: App cannot be rendered by
 * the smoke test in a signed-in state (effects do not run during a server render,
 * so App always shows its "contacting the service" line), and an overlay buried
 * inside App would therefore never be covered by anything. Passed the tab list it
 * describes, it renders and can be asserted on directly.
 *
 * The list is derived from the tabs it is given, so it cannot claim a shortcut
 * that does not exist. That matters more than it sounds: a help panel naming a
 * dead key is worse than no help panel, because it is read once and remembered.
 */
export default function Shortcuts({ tabs, onClose }) {
  return (
    <div className="overlay" role="dialog" aria-label="Keyboard shortcuts" onClick={onClose}>
      <div className="overlay-card" onClick={(event) => event.stopPropagation()}>
        <h2>Keyboard shortcuts</h2>
        <ul className="shortcut-list">
          {(tabs || []).map((entry, index) => (
            <li key={entry.id}>
              <span className="chord">
                <kbd>Alt</kbd>
                <kbd>{index + 1}</kbd>
              </span>
              <span className="what">{entry.label}</span>
            </li>
          ))}
          <li>
            <span className="chord">
              <kbd>Ctrl</kbd>
              <kbd>⌘</kbd>
              <kbd>K</kbd>
            </span>
            <span className="what">Jump to the question box</span>
          </li>
          <li>
            <span className="chord">
              <kbd>Ctrl</kbd>
              <kbd>⌘</kbd>
              <kbd>Enter</kbd>
            </span>
            <span className="what">Ask the question</span>
          </li>
          <li>
            <span className="chord">
              <kbd>?</kbd>
            </span>
            <span className="what">This list</span>
          </li>
          <li>
            <span className="chord">
              <kbd>Esc</kbd>
            </span>
            <span className="what">Close this list</span>
          </li>
        </ul>
        <p className="muted small">
          Modifier chords still work while you are typing in a field; bare keys
          belong to the field, so a question mark inside a question stays a question
          mark.
        </p>
        <button className="ghost" onClick={onClose}>
          Close
        </button>
      </div>
    </div>
  )
}
