import React, { useCallback, useEffect, useState } from 'react'
import { api } from '../api.js'

/**
 * Collections manager.
 *
 * A collection is a label on an indexed source, and a query can be scoped to one.
 * Two things about it are worth putting in front of the operator rather than in a
 * docstring, because both are the opposite of what the words suggest:
 *
 *   - deleting a collection does not delete anything in it. Its sources move to
 *     `general`, keep their items and vectors, and stay citable.
 *   - renaming or regrouping never re-embeds. A collection lives in one column of
 *     the sources table, so this cannot damage the index.
 *
 * `general` cannot be renamed or deleted: ingestion defaults to it and a deleted
 * collection moves its sources there, so it has to keep its name.
 */
export default function Collections({ can, sources, onChanged }) {
  const [rows, setRows] = useState([])
  const [defaultName, setDefaultName] = useState('general')
  const [draft, setDraft] = useState('')
  const [open, setOpen] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [notice, setNotice] = useState(null)

  const manage = can('manage_collections')

  const load = useCallback(() => {
    api
      .collections()
      .then((payload) => {
        setRows(payload.collections || [])
        setDefaultName(payload.default || 'general')
        setError(null)
      })
      .catch((err) => setError(err))
  }, [])

  useEffect(() => {
    load()
  }, [load])

  async function act(label, run) {
    setBusy(true)
    setError(null)
    setNotice(null)
    try {
      const message = await run()
      setNotice(message || `${label} done.`)
      load()
      if (onChanged) onChanged()
    } catch (err) {
      setError(err)
    } finally {
      setBusy(false)
    }
  }

  function members(name) {
    return (sources || []).filter((source) => source.collection === name)
  }

  async function onCreate(event) {
    event.preventDefault()
    const name = draft.trim()
    if (!name) return
    await act('Collection created', async () => {
      await api.createCollection(name)
      setDraft('')
      return `"${name}" created. It is empty until sources are filed under it.`
    })
  }

  async function onRename(row) {
    const answer = window.prompt(
      `Rename "${row.collection}". Its ${row.sources} source(s) move with it; nothing is re-indexed.`,
      row.collection,
    )
    if (answer === null) return
    const name = answer.trim()
    if (!name || name === row.collection) return
    await act('Collection renamed', async () => {
      const result = await api.renameCollection(row.collection, name)
      return `"${row.collection}" is now "${result.collection}" (${result.sources_moved} source(s) moved).`
    })
  }

  async function onDelete(row) {
    const count = row.sources || 0
    if (
      !window.confirm(
        `Delete the collection "${row.collection}"?\n\n` +
          `Its ${count} source(s) move to "${defaultName}" and stay indexed and citable. ` +
          `Nothing is deleted except the grouping, and a question scoped to ` +
          `"${row.collection}" will simply stop matching.`,
      )
    ) {
      return
    }
    await act('Collection deleted', async () => {
      const result = await api.deleteCollection(row.collection)
      return `"${row.collection}" removed: ${result.sources_moved} source(s) moved to ` +
        `"${result.moved_to}", ${result.sources_deleted} deleted.`
    })
  }

  async function onAssign(name, source) {
    await act('Source moved', async () => {
      const result = await api.assignSources(name, [source.source_id])
      return `${source.filename} filed under "${name}" (${result.moved} moved).`
    })
  }

  return (
    <div className="card">
      <h2>Collections</h2>
      <p className="muted small" style={{ marginTop: 0 }}>
        A collection groups indexed sources so a question can be scoped to it. It is a
        label on a source row: creating, renaming or deleting one never re-embeds
        anything, and deleting one moves its sources to <code>{defaultName}</code> rather
        than removing them.
      </p>

      {error && (
        <div className="banner bad" role="alert">
          <strong>Request failed{error.status ? ` (${error.status})` : ''}</strong>
          {error.message}
        </div>
      )}
      {notice && (
        <div className="banner ok small" role="status">
          {notice}
        </div>
      )}

      <table className="grid">
        <thead>
          <tr>
            <th>Collection</th>
            <th>Sources</th>
            <th>Indexed items</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.collection}>
              <td>
                <button
                  type="button"
                  className="link"
                  onClick={() => setOpen(open === row.collection ? null : row.collection)}
                >
                  {open === row.collection ? '▾' : '▸'} {row.collection}
                </button>{' '}
                {row.collection === defaultName && <span className="pill">default</span>}
              </td>
              <td className="mono">{row.sources}</td>
              <td className="mono">{row.items}</td>
              <td className="row" style={{ gap: 6, justifyContent: 'flex-end' }}>
                {manage && row.collection !== defaultName && (
                  <>
                    <button
                      type="button"
                      className="ghost small"
                      disabled={busy}
                      onClick={() => onRename(row)}
                    >
                      rename
                    </button>
                    <button
                      type="button"
                      className="danger small"
                      disabled={busy}
                      onClick={() => onDelete(row)}
                    >
                      delete
                    </button>
                  </>
                )}
              </td>
            </tr>
          ))}
          {rows.length === 0 && (
            <tr>
              <td colSpan={4} className="muted">
                No collections yet. Everything indexed lands in <code>{defaultName}</code>.
              </td>
            </tr>
          )}
        </tbody>
      </table>

      {open && (
        <div className="collection-members">
          {members(open).length === 0 ? (
            <p className="muted small">
              Nothing is filed under <code>{open}</code> yet. Use the collection column of
              the source table below, or ingest with this name.
            </p>
          ) : (
            <ul className="small collection-member-list">
              {members(open).map((source) => (
                <li key={source.source_id}>
                  <code>{source.source_id.slice(0, 8)}</code> {source.filename}
                  <span className="muted"> · {source.source_type}</span>
                  {manage && source.collection !== defaultName && (
                    <button
                      type="button"
                      className="ghost small"
                      style={{ marginLeft: 8 }}
                      disabled={busy}
                      onClick={() => onAssign(defaultName, source)}
                    >
                      move to {defaultName}
                    </button>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {manage ? (
        <form className="row" style={{ marginTop: 12 }} onSubmit={onCreate}>
          <label className="field" style={{ marginBottom: 0, minWidth: 220 }}>
            <span>New collection</span>
            <input
              type="text"
              value={draft}
              placeholder="board-packs-2026"
              onChange={(event) => setDraft(event.target.value)}
            />
          </label>
          <button type="submit" className="primary" disabled={busy || !draft.trim()}>
            {busy ? 'Working…' : 'Create'}
          </button>
        </form>
      ) : (
        <p className="muted small" style={{ marginBottom: 0 }}>
          Your role can scope a question to a collection but not create, rename or delete
          one — that needs the <code>manage_collections</code> permission (admin).
        </p>
      )}
    </div>
  )
}
