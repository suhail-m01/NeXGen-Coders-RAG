import React, { useEffect, useState } from 'react'
import { api } from '../api.js'

/* Landing view: what this service is holding, and whether it is healthy, before
 * anybody asks it anything.
 *
 * Every figure on this page comes from an endpoint - /health, /models,
 * /collections, /history/sessions - and anything an endpoint did not supply is
 * shown as an em dash rather than as a zero or a plausible constant. That is a
 * deliberate rule: an overview that hardcodes "Vector DB: ChromaDB" or
 * "Mode: Offline" reports the string it was written with, not the service it is
 * describing, and it will keep reporting it after the service changes.
 */

function Stat({ label, value, note }) {
  return (
    <div className="metric">
      <h3 style={{ marginBottom: 4 }}>{label}</h3>
      <p className="value">{value ?? '—'}</p>
      {note ? <p className="muted small">{note}</p> : null}
    </div>
  )
}

export default function Overview({ onNavigate }) {
  const [health, setHealth] = useState(null)
  const [models, setModels] = useState(null)
  const [collections, setCollections] = useState(null)
  const [sessions, setSessions] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let alive = true
    Promise.all([
      api.health(),
      api.models().catch(() => null),
      api.collections().catch(() => null),
      api.historySessions(50).catch(() => null),
    ])
      .then(([healthData, modelsData, collectionsData, sessionsData]) => {
        if (!alive) return
        setHealth(healthData)
        setModels(modelsData)
        setCollections(collectionsData)
        setSessions(sessionsData)
      })
      .catch((err) => alive && setError(err))
      .finally(() => alive && setLoading(false))
    return () => {
      alive = false
    }
  }, [])

  if (loading) return <div className="card"><p className="muted">Reading service state…</p></div>
  if (error) {
    return (
      <div className="banner bad">
        <strong>Overview could not read the service ({error.status})</strong>
        {error.message}
      </div>
    )
  }
  if (!health) return null

  const runtime = health.runtime || {}
  const index = health.index || {}
  const rows = (collections && collections.collections) || []
  const sessionRows = (sessions && sessions.sessions) || (Array.isArray(sessions) ? sessions : [])
  const model = models || {}
  const actions = [
    { id: 'ask', label: 'Ask a question', note: 'Answer with citations, scores and the pipeline trace' },
    { id: 'sources', label: 'Browse the corpus', note: 'Every indexed source, its chunks and its warnings' },
    { id: 'history', label: 'Read the audit history', note: 'Recorded turns, replayed with the trace they produced' },
    { id: 'health', label: 'Service and build', note: 'Manifest, guards, rate limits, installed models' },
  ]

  return (
    <>
      <div className="card">
        <div className="row spread">
          <h2 style={{ margin: 0 }}>Evidence AI</h2>
          <span className={`pill ${health.status === 'ok' ? 'ok' : 'warn'}`}>
            {health.status === 'ok' ? 'service ok' : `service ${health.status || 'degraded'}`}
          </span>
        </div>
        <div className="grid">
          <Stat label="Sources" value={index.sources} note="files in the index" />
          <Stat label="Indexed items" value={index.items} note="chunks, cells and segments" />
          <Stat label="Relationships" value={index.relationships} note="links used to expand evidence" />
          <Stat
            label="Collections"
            value={rows.length || '—'}
            note={rows.length ? rows.map((row) => row.collection).join(', ') : 'only the default'}
          />
          <Stat
            label="Language model"
            value={model.configured || runtime.llm_model || '—'}
            note={
              model.reachable === false
                ? 'inference server not answering'
                : model.configured_present
                  ? `installed here · ${model.names.length} model(s) on the server`
                  : model.names && model.names.length
                    ? `not installed here · available: ${model.names.join(', ')}`
                    : undefined
            }
          />
          <Stat
            label="Recorded sessions"
            value={sessions ? sessionRows.length : '—'}
            note={
              sessions
                ? 'of the 50 most recent'
                : 'not readable by this role or not served'
            }
          />
        </div>
        {model.hint && (
          <div className="banner warn" style={{ marginTop: 10 }}>
            <strong>Model mismatch</strong>
            {model.hint}
          </div>
        )}
        <p className="muted small" style={{ marginTop: 8 }}>
          Chunking mode <code>{runtime.chunking_mode || '—'}</code> · commit{' '}
          <code>{runtime.git_commit || '—'}</code>
          {runtime.git_dirty ? ' (dirty tree)' : ''} · Python{' '}
          <code>{runtime.python || '—'}</code>. Nothing on this page is a constant: it is
          read from the service each time the tab is opened.
        </p>
      </div>

      <div className="card">
        <h2>Where to go</h2>
        <div className="grid">
          {actions.map((action, position) => (
            <button
              key={action.id}
              className="ghost"
              style={{ textAlign: 'left', padding: 12 }}
              onClick={() => onNavigate && onNavigate(action.id)}
            >
              <strong>{action.label}</strong>
              <span className="muted small" style={{ display: 'block' }}>{action.note}</span>
              <span className="muted small">Alt + {position + 2}</span>
            </button>
          ))}
        </div>
      </div>
    </>
  )
}
