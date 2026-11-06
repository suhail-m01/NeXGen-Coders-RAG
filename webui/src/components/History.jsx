import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { api } from '../api.js'
import { splitAnswer } from './Answer.jsx'
import StageTrace from './StageTrace.jsx'

// Chat history in the browser.
//
// The Streamlit UI has kept this audit trail since batch 2. What it could not do
// was let someone who is not sitting at that terminal read it, so an answer's
// provenance lived on one machine. Every message stored here carries the pipeline
// trace, the citations and the figures the numeric guard could not verify, as they
// were at record time -- which is what makes an answer reviewable, and challengeable,
// later. Filenames were resolved into the trace when it was written rather than now,
// so a citation still names its source after that source has been deleted.
//
// Two things this view does not pretend to do. Sessions carry no owner column, so
// every signed-in account sees every session -- the same as the Streamlit UI, not a
// new gap opened here. And a stored trace is a record, not a re-run: the index may
// have changed since, and nothing in this component re-scores the answer.

function stamp(value) {
  if (!value) return ''
  const parsed = new Date(String(value).replace(' ', 'T'))
  return Number.isNaN(parsed.getTime()) ? String(value) : parsed.toLocaleString()
}

function download(filename, text) {
  // The export endpoint needs the bearer token, which a plain <a href> cannot
  // send, so the document is fetched and handed to the browser as a blob.
  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

function Blocks({ text }) {
  // Same splitting the live answer uses, so a guard caveat stays visually separate
  // from the claim it qualifies instead of reading as part of it.
  const blocks = splitAnswer(text)
  return (
    <div className="answer">
      {blocks.map((block, index) => (
        <div className={block.kind === 'claim' ? 'claim' : block.kind} key={index}>
          {block.text}
        </div>
      ))}
    </div>
  )
}

function Message({ message }) {
  const [openTrace, setOpenTrace] = useState(false)

  if (message.role === 'user') {
    return (
      <div className="history-turn">
        <span className="history-role">Q</span>
        <div className="history-body">
          <Blocks text={message.content} />
          <p className="muted small">{stamp(message.created_at)}</p>
        </div>
      </div>
    )
  }

  const citations = Array.isArray(message.citations) ? message.citations : []
  const unverified = Array.isArray(message.unverified) ? message.unverified : []
  const trace = message.trace || {}
  const stages = Array.isArray(trace.stages) ? trace.stages : []

  return (
    <div className="history-turn">
      <span className="history-role">A</span>
      <div className="history-body">
        {message.abstained ? (
          <div className="banner warn">
            <strong>Refused</strong>
            {trace.decision ||
              'The evidence did not support an answer to this question. The refusal was recorded as the result.'}
          </div>
        ) : null}

        <Blocks text={message.content} />

        <div className="history-meta muted small">
          {typeof message.confidence === 'number'
            ? `confidence ${message.confidence.toFixed(3)}`
            : 'no confidence recorded'}
          {trace.latency_ms != null ? ` · ${Math.round(trace.latency_ms)} ms` : ''}
          {` · ${citations.length} citation${citations.length === 1 ? '' : 's'}`}
          {` · ${stamp(message.created_at)}`}
        </div>

        {unverified.length > 0 ? (
          <div className="banner warn">
            <strong>Figures the guard could not verify</strong>
            {unverified.join(', ')} — stated in the answer, not found in the evidence.
          </div>
        ) : null}

        {citations.length > 0 ? (
          <div className="history-citations">
            {citations.map((citation, index) => (
              <div className="evidence-item" key={citation.citation_id || index}>
                <div className="rank">
                  <span className="pill">{citation.citation_id || `C${index + 1}`}</span>{' '}
                  {citation.filename || 'unknown source'}
                </div>
                <div className="where">
                  {citation.location ? <span>{citation.location}</span> : null}
                  {citation.modality ? <span> · {String(citation.modality).toLowerCase()}</span> : null}
                </div>
                {citation.excerpt ? <p className="excerpt">{citation.excerpt}</p> : null}
              </div>
            ))}
          </div>
        ) : null}

        {stages.length > 0 ? (
          <div>
            <button
              type="button"
              className="ghost small"
              onClick={() => setOpenTrace((value) => !value)}
            >
              {openTrace ? '▾ Hide' : '▸ Show'} the trace as recorded ({stages.length} stages)
            </button>
            {openTrace ? <StageTrace result={{ stages }} /> : null}
          </div>
        ) : (
          <p className="muted small history-note">
            No trace was stored with this answer. Sessions recorded before trace capture keep the
            question, the answer and the citations only.
          </p>
        )}
      </div>
    </div>
  )
}

export default function History({ can }) {
  const [sessions, setSessions] = useState([])
  const [stats, setStats] = useState(null)
  const [selected, setSelected] = useState(null)
  const [messages, setMessages] = useState([])
  const [status, setStatus] = useState('loading')
  const [error, setError] = useState(null)

  const canExport = can('export_history')

  const load = useCallback(async () => {
    try {
      const payload = await api.historySessions(100)
      const rows = payload.sessions || []
      setSessions(rows)
      setStats(payload.stats || null)
      setStatus(rows.length === 0 ? 'empty' : 'ready')
      setError(null)
    } catch (err) {
      setError(err)
      setStatus('error')
    }
  }, [])

  useEffect(() => {
    load()
  }, [load])

  useEffect(() => {
    if (!selected) {
      setMessages([])
      return undefined
    }
    let alive = true
    api
      .historySession(selected)
      .then((payload) => alive && setMessages(payload.messages || []))
      .catch((err) => alive && setError(err))
    return () => {
      alive = false
    }
  }, [selected])

  const active = useMemo(
    () => sessions.find((session) => session.session_id === selected) || null,
    [sessions, selected],
  )

  async function onNew() {
    try {
      const created = await api.createSession('Untitled session')
      await load()
      setSelected(created.session_id)
      setError(null)
    } catch (err) {
      setError(err)
    }
  }

  async function onRename(session) {
    const answer = window.prompt('Rename this session', session.title || '')
    if (answer === null) return
    const title = answer.trim()
    if (!title) return
    try {
      await api.renameSession(session.session_id, title)
      await load()
      setError(null)
    } catch (err) {
      setError(err)
    }
  }

  async function onDelete(session) {
    const count = session.message_count || 0
    if (
      !window.confirm(
        `Delete "${session.title}" and its ${count} message${count === 1 ? '' : 's'}? ` +
          'Indexed sources are untouched.',
      )
    ) {
      return
    }
    try {
      await api.deleteSession(session.session_id)
      if (selected === session.session_id) setSelected(null)
      await load()
      setError(null)
    } catch (err) {
      setError(err)
    }
  }

  async function onExport(format) {
    if (!active) return
    try {
      const text = await api.exportSession(active.session_id, format)
      download(
        `evidence_session_${active.session_id.slice(0, 8)}.${format === 'csv' ? 'csv' : 'md'}`,
        text,
      )
      setError(null)
    } catch (err) {
      setError(err)
    }
  }

  return (
    <div className="card">
      <h2>Conversation history</h2>
      <div className="row spread" style={{ marginBottom: 12 }}>
        <div className="row">
          <button type="button" className="ghost small" onClick={onNew}>
            New session
          </button>
          <button type="button" className="ghost small" onClick={load}>
            Refresh
          </button>
        </div>
        {stats ? (
          <span className="muted small">
            {stats.sessions} session{stats.sessions === 1 ? '' : 's'} · {stats.messages} message
            {stats.messages === 1 ? '' : 's'}
          </span>
        ) : null}
      </div>

      {error ? (
        <div className="banner bad" role="alert">
          <strong>Request failed{error.status ? ` (${error.status})` : ''}</strong>
          {error.message}
        </div>
      ) : null}

      <div className="history-layout">
        <div className="history-list">
          {status === 'loading' ? <p className="muted">Loading sessions…</p> : null}
          {status === 'error' ? <p className="muted">Could not read the history.</p> : null}
          {status === 'empty' ? (
            <p className="muted small">
              No conversations yet. Ask a question on the Ask tab with recording on — a turn is
              stored only when the request carries a session, so probes, health checks and
              evaluation runs stay out of the trail.
            </p>
          ) : null}
          {sessions.map((session) => (
            <div
              className={`history-item${session.session_id === selected ? ' selected' : ''}`}
              key={session.session_id}
            >
              <button
                type="button"
                className="link"
                onClick={() => setSelected(session.session_id)}
              >
                {session.title || 'Untitled'}
              </button>
              <div className="history-item-meta muted small">
                <span>
                  {session.message_count || 0} message{(session.message_count || 0) === 1 ? '' : 's'}
                </span>
                <span>{stamp(session.updated_at)}</span>
                {session.backend ? <span className="pill">{session.backend}</span> : null}
              </div>
              <div className="row history-item-actions">
                <button type="button" className="ghost small" onClick={() => onRename(session)}>
                  rename
                </button>
                <button type="button" className="ghost small" onClick={() => onDelete(session)}>
                  delete
                </button>
              </div>
            </div>
          ))}
        </div>

        <div className="history-detail">
          {!active ? (
            <p className="muted">Select a session to read it back.</p>
          ) : (
            <>
              <div className="row spread" style={{ marginBottom: 6 }}>
                <h3 className="history-title">{active.title || 'Untitled'}</h3>
                {canExport ? (
                  <div className="row">
                    <button
                      type="button"
                      className="ghost small"
                      onClick={() => onExport('markdown')}
                    >
                      export Markdown
                    </button>
                    <button type="button" className="ghost small" onClick={() => onExport('csv')}>
                      export CSV
                    </button>
                  </div>
                ) : (
                  <span className="muted small">Exporting a session needs the admin role.</span>
                )}
              </div>
              <p className="muted small history-meta">
                session {active.session_id.slice(0, 8)} · created {stamp(active.created_at)} ·
                updated {stamp(active.updated_at)}
              </p>
              {messages.length === 0 ? (
                <p className="muted">This session has no messages yet.</p>
              ) : (
                messages.map((message) => <Message key={message.message_id} message={message} />)
              )}
              <p className="muted small history-note">
                Shown exactly as recorded. Nothing here re-runs the pipeline: if the index has
                changed since, the trace is the only account of what was retrieved then.
              </p>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
