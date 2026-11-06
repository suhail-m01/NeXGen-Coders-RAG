import React, { useCallback, useRef, useState } from 'react'

/**
 * Drag-and-drop target for ingestion.
 *
 * Two decisions here are deliberate.
 *
 * The accepted extensions come from the server's run manifest rather than being
 * written into this file. A hard-coded list drifts the first time the router
 * learns a new format, and a drop zone that accepts a file the backend will
 * refuse wastes the user's time twice: once on the drop, once on the error.
 *
 * Rejected files are reported by name with the reason, never dropped silently.
 * A drop of twelve files where two quietly disappear reads as "all twelve are
 * queued", and the user finds out otherwise minutes later when the index does not
 * contain them.
 */
export default function DropZone({ accepted, maxBytes, onFiles, disabled }) {
  const [over, setOver] = useState(false)
  const inputRef = useRef(null)

  const extensions = React.useMemo(() => {
    if (!accepted) return []
    if (Array.isArray(accepted)) return accepted.map((ext) => ext.toLowerCase())
    return [...(accepted.documents || []), ...(accepted.images || []), ...(accepted.audio || [])]
      .map((ext) => String(ext).toLowerCase())
      .sort()
  }, [accepted])

  const partition = useCallback(
    (files) => {
      const kept = []
      const refused = []
      for (const file of Array.from(files || [])) {
        const name = file.name || ''
        const dot = name.lastIndexOf('.')
        const ext = dot > 0 ? name.slice(dot).toLowerCase() : ''
        if (extensions.length && !extensions.includes(ext)) {
          refused.push({
            name,
            reason: ext
              ? `.${ext.slice(1)} is not a format this index accepts`
              : 'the filename has no extension',
          })
        } else if (maxBytes && file.size > maxBytes) {
          refused.push({
            name,
            reason: `${(file.size / (1024 * 1024)).toFixed(1)} MB exceeds the ` +
              `${Math.round(maxBytes / (1024 * 1024))} MB limit`,
          })
        } else if (!file.size) {
          // A dropped folder arrives as a zero-byte entry with no usable content.
          refused.push({ name, reason: 'empty — folders cannot be dropped, only files' })
        } else {
          kept.push(file)
        }
      }
      return { kept, refused }
    },
    [extensions, maxBytes],
  )

  const take = useCallback(
    (files) => {
      const { kept, refused } = partition(files)
      if (kept.length || refused.length) onFiles(kept, refused)
    },
    [onFiles, partition],
  )

  function onDrop(event) {
    event.preventDefault()
    event.stopPropagation()
    setOver(false)
    if (disabled) return
    take(event.dataTransfer?.files)
  }

  function onSelect(event) {
    take(event.target.files)
    // Reset so choosing the same file again still fires a change event.
    event.target.value = ''
  }

  return (
    <div
      className={`dropzone${over ? ' over' : ''}${disabled ? ' disabled' : ''}`}
      onDragOver={(event) => {
        event.preventDefault()
        event.stopPropagation()
        if (!disabled) setOver(true)
      }}
      onDragEnter={(event) => {
        event.preventDefault()
        if (!disabled) setOver(true)
      }}
      onDragLeave={(event) => {
        event.preventDefault()
        setOver(false)
      }}
      onDrop={onDrop}
    >
      <p className="dropzone-title">
        {over ? 'Drop to queue these files' : 'Drop files here, or choose them'}
      </p>
      <p className="muted small dropzone-hint">
        {extensions.length
          ? `Accepted: ${extensions.map((ext) => ext.slice(1)).join(', ')}`
          : 'Reading the accepted formats from the server…'}
        {maxBytes ? ` · up to ${Math.round(maxBytes / (1024 * 1024))} MB each` : ''}
      </p>
      <button
        type="button"
        className="ghost"
        disabled={disabled}
        onClick={() => inputRef.current?.click()}
      >
        Choose files
      </button>
      <input
        ref={inputRef}
        type="file"
        multiple
        className="dropzone-input"
        disabled={disabled}
        accept={extensions.length ? extensions.join(',') : undefined}
        onChange={onSelect}
      />
    </div>
  )
}
