import React, { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../api.js'

function bytes(size) {
  if (size == null) return '—'
  if (size < 1024) return `${size} B`
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(0)} KB`
  return `${(size / (1024 * 1024)).toFixed(2)} MB`
}

/**
 * Corpus panel: what is indexed, which files have more than one version in the
 * index, and (for roles that may) adding or removing sources.
 *
 * The stale-version list matters because re-uploading a changed file adds a second
 * version beside the first: both answer questions, and the older one can rank
 * higher. Every other view in this project shows a current snapshot; this is the
 * one place that says "two versions of this file are both live".
 */
export default function Sources({ can }) {
  const [data, setData] = useState(null)
  const [stale, setStale] = useState({})
  const [error, setError] = useState(null)
  const [notice, setNotice] = useState(null)
  const [busy, setBusy] = useState(false)
  const [collection, setCollection] = useState('general')
  const [replace, setReplace] = useState(true)
  const fileRef = useRef(null)

  const load = useCallback(() => {
    setError(null)
    Promise.all([api.sources(), api.staleSources().catch(() => ({ stale_versions: {} }))])
      .then(([sources, staleData]) => {
        setData(sources)
        setStale(staleData.stale_versions || {})
      })
      .catch((err) => setError(err))
  }, [])

  useEffect(() => {
    load()
  }, [load])

  async function upload(event) {
    event.preventDefault()
    const files = fileRef.current?.files
    if (!files || !files.length) return
    setBusy(true)
    setError(null)
    setNotice(null)
    try {
      const result = await api.ingest(files, collection, replace)
      const lines = (result.results || []).map((row) =>
        row.success
          ? `${row.filename}: ${row.items_indexed} item(s) indexed` +
            (row.warnings?.length ? ` · ${row.warnings.length} warning(s)` : '')
          : `${row.filename}: failed — ${row.error || 'unknown error'}`,
      )
      setNotice(lines.join('\n'))
      if (fileRef.current) fileRef.current.value = ''
      load()
    } catch (err) {
      setError(err)
    } finally {
      setBusy(false)
    }
  }

  async function remove(source, reason) {
    const message = reason
      ? `${reason}\n\nRemove this version from the index? The file on disk and any saved chat history are left untouched.`
      : `Remove ${source.filename} from the index? The file on disk and any saved chat history are left untouched.`
    if (!window.confirm(message)) return
    setBusy(true)
    setError(null)
    setNotice(null)
    try {
      const result = await api.deleteSource(source.source_id)
      setNotice(`Removed ${result.items_removed} indexed item(s) from ${source.filename}.`)
      load()
    } catch (err) {
      setError(err)
    } finally {
      setBusy(false)
    }
  }

  const sources = data?.sources || []
  const staleNames = Object.keys(stale)
  const collections = (data?.collections || []).map((row) =>
    typeof row === 'string' ? row : row.collection,
  )

  return (
    <>
      {staleNames.length > 0 && (
        <div className="banner warn">
          <strong>{staleNames.length} file(s) have more than one version indexed</strong>
          Both versions answer questions and either can be cited. Re-ingest with
          "replace existing versions" checked, or remove the older one below.
          <ul className="small" style={{ marginBottom: 0 }}>
            {staleNames.map((name) => {
              const versions = stale[name] || []
              return (
                <li key={name}>
                  <strong>{name}</strong> — {versions.length} versions; newest is{' '}
                  <code>{versions[versions.length - 1]?.source_id?.slice(0, 8)}</code>
                  {can('delete') &&
                    versions.slice(0, -1).map((version) => (
                      <button
                        key={version.source_id}
                        className="danger"
                        style={{ marginLeft: 8 }}
                        disabled={busy}
                        onClick={() =>
                          remove(
                            version,
                            `Older version of ${name} (${version.source_id.slice(0, 8)}).`,
                          )
                        }
                      >
                        Remove older {version.source_id.slice(0, 8)}
                      </button>
                    ))}
                </li>
              )
            })}
          </ul>
        </div>
      )}

      {can('ingest') && (
        <div className="card">
          <h2>Add documents</h2>
          <form onSubmit={upload}>
            <label className="field">
              <span>Files (PDF, DOCX, CSV, TXT, MD, PNG, JPG, WAV, MP3)</span>
              <input type="file" multiple ref={fileRef} />
            </label>
            <div className="row">
              <label className="field" style={{ marginBottom: 0, minWidth: 200 }}>
                <span>Collection</span>
                <input
                  type="text"
                  list="collections"
                  value={collection}
                  onChange={(e) => setCollection(e.target.value)}
                />
                <datalist id="collections">
                  {collections.map((name) => (
                    <option key={name} value={name} />
                  ))}
                </datalist>
              </label>
              <label className="row small" style={{ gap: 6, marginBottom: 0 }}>
                <input
                  type="checkbox"
                  checked={replace}
                  disabled={!can('delete')}
                  onChange={(e) => setReplace(e.target.checked)}
                />
                Replace existing versions of the same filename
              </label>
              <button className="primary" type="submit" disabled={busy}>
                {busy ? 'Indexing…' : 'Ingest'}
              </button>
            </div>
            {!can('delete') && (
              <p className="muted small" style={{ marginBottom: 0 }}>
                Replacing a version deletes the older one, so it needs the{' '}
                <code>delete</code> permission (admin). Your role may still add new
                files.
              </p>
            )}
          </form>
          {notice && (
            <pre className="banner ok small" style={{ whiteSpace: 'pre-wrap' }}>
              {notice}
            </pre>
          )}
        </div>
      )}

      {error && (
        <div className="banner bad" role="alert">
          <strong>Request failed ({error.status})</strong>
          {error.message}
        </div>
      )}

      <div className="card">
        <h2>Indexed sources</h2>
        {sources.length === 0 ? (
          <p className="muted">
            Nothing is indexed yet. Run{' '}
            <code>python scripts/ingest_demo.py</code> for the demo corpus, or add
            files above.
          </p>
        ) : (
          <table className="grid">
            <thead>
              <tr>
                <th>File</th>
                <th>Type</th>
                <th>Collection</th>
                <th>Size</th>
                <th>Source ID</th>
                {can('delete') && <th />}
              </tr>
            </thead>
            <tbody>
              {sources.map((source) => (
                <tr key={source.source_id}>
                  <td>{source.filename}</td>
                  <td>
                    <span className="pill">{source.source_type}</span>
                  </td>
                  <td>{source.collection}</td>
                  <td className="mono">{bytes(source.file_size)}</td>
                  <td className="mono">{source.source_id.slice(0, 8)}</td>
                  {can('delete') && (
                    <td>
                      <button
                        className="danger"
                        disabled={busy}
                        onClick={() => remove(source)}
                      >
                        Remove
                      </button>
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <p className="muted small">
          Removing a source deletes its indexed items and rebuilds the vector index;
          the original file on disk is left alone.
        </p>
      </div>
    </>
  )
}
