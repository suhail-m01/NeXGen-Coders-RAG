import React, { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../api.js'
import Collections from './Collections.jsx'
import DropZone from './DropZone.jsx'
import FileQueue from './FileQueue.jsx'

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
 *
 * Ingestion goes through a queue, one request per file, so each file reports its
 * own outcome: items indexed, warnings, or the reason it failed. A batch request
 * answers "the upload finished" and leaves the user scanning a paragraph to find
 * the one PDF that was password protected.
 */
export default function Sources({ can }) {
  const [data, setData] = useState(null)
  // Declared before the panels that render it: the collections manager needs the
  // source rows to show which files sit in which collection.
  const sources = data?.sources || []
  const [stale, setStale] = useState({})
  const [error, setError] = useState(null)
  const [notice, setNotice] = useState(null)
  const [busy, setBusy] = useState(false)
  const [collection, setCollection] = useState('general')
  const [replace, setReplace] = useState(true)
  // Ingestion queue. `formats` is the server's own list of accepted extensions and
  // the per-file size cap, read from the run manifest so the drop zone cannot drift
  // from the router.
  const [queue, setQueue] = useState([])
  const [uploading, setUploading] = useState(false)
  const [formats, setFormats] = useState(null)
  const queueId = useRef(0)

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
    api
      .manifest()
      .then((manifest) => setFormats(manifest.ingestion || null))
      // An older server has no ingestion block in its manifest. The drop zone then
      // accepts anything and lets the server do the refusing, which is still honest
      // -- it just reports the reason a beat later.
      .catch(() => setFormats(null))
  }, [load])

  function nextId() {
    queueId.current += 1
    return queueId.current
  }

  const patchRow = useCallback((id, changes) => {
    setQueue((rows) => rows.map((row) => (row.id === id ? { ...row, ...changes } : row)))
  }, [])

  // Files that cannot be queued are shown as rows with a reason rather than dropped
  // on the floor: a drop of twelve where two quietly vanish reads as "twelve queued"
  // and the user finds out otherwise minutes later, from an index that lacks them.
  function enqueue(files, refused) {
    setError(null)
    setNotice(null)
    const seen = new Set(queue.map((row) => `${row.name}::${row.size}`))
    const rejected = [...(refused || [])]
    const added = []
    for (const file of files || []) {
      const key = `${file.name}::${file.size}`
      if (seen.has(key)) {
        rejected.push({ name: file.name, reason: 'already in this queue' })
        continue
      }
      seen.add(key)
      added.push({
        id: nextId(),
        file,
        name: file.name,
        size: file.size,
        status: 'queued',
      })
    }
    const refusedRows = rejected.map((entry) => ({
      id: nextId(),
      name: entry.name,
      size: null,
      status: 'refused',
      error: entry.reason,
    }))
    if (!added.length && !refusedRows.length) return
    setQueue([...queue, ...added, ...refusedRows])
  }

  function removeFromQueue(id) {
    setQueue((rows) => rows.filter((row) => row.id !== id))
  }

  function clearFinished() {
    setQueue((rows) =>
      rows.filter((row) => ['queued', 'failed', 'uploading', 'waiting'].includes(row.status)),
    )
  }

  async function ingestOne(entry, attempt) {
    try {
      return await api.ingest([entry.file], collection, replace)
    } catch (err) {
      // A 429 is the rate limiter doing its job, not a failure of this file: wait
      // the Retry-After it hands back and try once more. The wait is capped so a
      // misconfigured limit cannot park the queue.
      const wait = Number(err?.retryAfter || 0)
      if (err?.status === 429 && attempt === 0 && wait > 0 && wait <= 60) {
        patchRow(entry.id, { status: 'waiting', waiting: wait })
        await new Promise((resolve) => setTimeout(resolve, wait * 1000 + 200))
        return ingestOne(entry, attempt + 1)
      }
      throw err
    }
  }

  async function runIngest() {
    const pending = queue.filter((row) => row.status === 'queued' || row.status === 'failed')
    if (!pending.length) return
    setUploading(true)
    setError(null)
    setNotice(null)
    let indexed = 0
    let failed = 0
    let warned = 0
    let items = 0
    for (const entry of pending) {
      patchRow(entry.id, { status: 'uploading', error: null, waiting: null })
      try {
        const result = await ingestOne(entry, 0)
        const row = (result.results || [])[0] || {}
        if (row.success) {
          indexed += 1
          items += row.items_indexed || 0
          warned += (row.warnings || []).length
          patchRow(entry.id, {
            status: (row.warnings || []).length ? 'warning' : 'indexed',
            items: row.items_indexed ?? 0,
            warnings: row.warnings || [],
            waiting: null,
          })
        } else {
          failed += 1
          patchRow(entry.id, {
            status: 'failed',
            waiting: null,
            error: row.error || 'the server refused this file',
          })
        }
      } catch (err) {
        failed += 1
        patchRow(entry.id, {
          status: 'failed',
          waiting: null,
          error:
            err?.status === 429
              ? 'The ingest rate limit is still in force after waiting. Try again in a minute, or index from disk with python scripts/reindex.py.'
              : `${err?.message || 'request failed'}${err?.status ? ` (${err.status})` : ''}`,
        })
      }
    }
    setUploading(false)
    load()
    setNotice(
      `${indexed} of ${pending.length} file(s) indexed, ${items} item(s) added` +
        (warned ? `, ${warned} warning(s) listed on their rows` : '') +
        (failed ? `, ${failed} failed` : '') +
        '.',
    )
  }

  async function move(source, collection) {
    if (!collection || collection === source.collection) return
    setBusy(true)
    setError(null)
    setNotice(null)
    try {
      const result = await api.assignSources(collection, [source.source_id])
      setNotice(
        `${source.filename} filed under "${collection}". Nothing was re-indexed: a ` +
          `collection is a label on the source row.`,
      )
      if (result.not_found?.length) {
        setError({ status: 404, message: `Not found: ${result.not_found.join(', ')}` })
      }
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

      <Collections can={can} sources={sources} onChanged={load} />

      {can('ingest') && (
        <div className="card">
          <h2>Add documents</h2>
          <DropZone
            accepted={formats}
            maxBytes={formats?.max_file_bytes}
            disabled={uploading}
            onFiles={enqueue}
          />
          <div className="row" style={{ gap: 14, marginTop: 12, flexWrap: 'wrap' }}>
            <label className="field" style={{ marginBottom: 0, minWidth: 200 }}>
              <span>Collection</span>
              <input
                type="text"
                list="collections"
                value={collection}
                disabled={uploading}
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
                disabled={!can('delete') || uploading}
                onChange={(e) => setReplace(e.target.checked)}
              />
              Replace existing versions of the same filename
            </label>
          </div>
          <FileQueue
            queue={queue}
            uploading={uploading}
            collection={collection}
            onIngest={runIngest}
            onRemove={removeFromQueue}
            onClear={clearFinished}
          />
          {notice && (
            <p className="banner ok small" style={{ marginBottom: 0 }}>
              {notice}
            </p>
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
                  <td>
                    {can('manage_collections') ? (
                      <select
                        value={source.collection}
                        disabled={busy}
                        onChange={(event) => move(source, event.target.value)}
                      >
                        {collections.includes(source.collection) ? null : (
                          <option value={source.collection}>{source.collection}</option>
                        )}
                        {collections.map((name) => (
                          <option key={name} value={name}>
                            {name}
                          </option>
                        ))}
                      </select>
                    ) : (
                      source.collection
                    )}
                  </td>
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
          the original file on disk is left alone. Changing a source's collection
          deletes nothing and rebuilds nothing.
        </p>
      </div>
    </>
  )
}
