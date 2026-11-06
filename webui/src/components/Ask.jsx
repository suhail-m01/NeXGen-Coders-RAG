import React, { useEffect, useRef, useState } from 'react'
import { api, ApiError } from '../api.js'
import Answer from './Answer.jsx'
import Evidence from './Evidence.jsx'
import StageTrace from './StageTrace.jsx'

// Questions the demo corpus can answer, so a first-time user is not staring at an
// empty box. They are ordinary questions, not tuned to the index.
const SAMPLES = [
  'How many records were processed by the pipeline?',
  'What percentage of the budget was utilised?',
  'What is the top risk carried into 2025?',
  'How many milestones were completed and by how many engineers?',
]

export default function Ask({ can, focusSignal }) {
  const [question, setQuestion] = useState('')
  const [collection, setCollection] = useState('general')
  const [collections, setCollections] = useState(['general'])
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [active, setActive] = useState(null)
  // The answer as it streams in, and the scope it was asked under. The scope is
  // captured at ask time rather than read from the dropdown, so changing the
  // dropdown afterwards does not rewrite what the answer on screen means.
  const [streamed, setStreamed] = useState('')
  const [scope, setScope] = useState(null)
  const [asked, setAsked] = useState('')
  // A conversation id. null means "nothing is being recorded". Once the first
  // recorded answer comes back the API reports which session it wrote to, and
  // every later question is appended to that one, so a browser can hold a
  // multi-turn conversation that is readable on the History tab afterwards.
  const [sessionId, setSessionId] = useState(null)
  const [record, setRecord] = useState(true)
  // Files attached to the next question, ChatGPT-style. They stay File objects
  // until send: nothing is uploaded before the question is asked, and the chips
  // can be removed again for free. The server extracts them into ephemeral
  // evidence for that one turn - they never join the index.
  const [attached, setAttached] = useState([])
  // What the answer on screen was asked with, captured at ask time like the
  // scope, so removing a chip afterwards does not rewrite the answer's context.
  const [askedWith, setAskedWith] = useState([])
  const answerRef = useRef(null)
  const questionRef = useRef(null)
  const fileRef = useRef(null)

  const MAX_ATTACHED = 5

  function addFiles(incoming) {
    setAttached((current) => {
      const next = [...current]
      const rejected = []
      for (const file of incoming) {
        if (next.length >= MAX_ATTACHED) {
          rejected.push(file.name)
          continue
        }
        if (next.some((f) => f.name === file.name && f.size === file.size)) continue
        next.push(file)
      }
      if (rejected.length) {
        setError(
          new ApiError(
            `At most ${MAX_ATTACHED} attachments may accompany one question; ` +
              `not added: ${rejected.join(', ')}.`,
            400,
          ),
        )
      }
      return next
    })
  }

  // Ctrl/Cmd+K anywhere in the app lands the cursor in the question box. The
  // signal is a counter rather than a ref handle because the box only exists
  // while this tab is mounted, and App has no way to hold a ref into it.
  useEffect(() => {
    if (focusSignal) questionRef.current?.focus()
  }, [focusSignal])

  useEffect(() => {
    api
      .collections()
      .then((data) => {
        // /collections also lists a collection that was declared but is still
        // empty, so a scope picked here is never a name the server does not know.
        const names = (data.collections || []).map((row) =>
          typeof row === 'string' ? row : row.collection,
        )
        const fallback = data.default || 'general'
        setCollections(names.length ? names : [fallback])
        if (!names.includes(collection)) setCollection(names[0] || fallback)
      })
      .catch(() => {
        // Older servers without /collections: fall back to what /sources reports.
        api
          .sources()
          .then((data) => {
            const names = (data.collections || []).map((row) =>
              typeof row === 'string' ? row : row.collection,
            )
            if (names.length) {
              setCollections(names.includes('general') ? names : ['general', ...names])
            }
          })
          .catch(() => {})
      })
  }, [])

  async function ask(text) {
    const trimmed = (text ?? question).trim()
    if (!trimmed || busy) return
    setBusy(true)
    setError(null)
    setActive(null)
    setStreamed('')
    setScope(collection)
    setAsked(trimmed)
    setAskedWith(attached.map((f) => f.name))
    const sending = attached
    try {
      // With recording on and no session yet, a placeholder id goes out: the API
      // starts a real session and returns its id, so no extra round trip is needed
      // before the first question. With attachments the multipart route carries
      // the same contract back, so streaming, citations and history behave
      // identically either way.
      const payload = sending.length
        ? await api.queryAttachStream(
            trimmed,
            sending,
            collection,
            record ? sessionId || 'new' : null,
            (token) => setStreamed((soFar) => soFar + token),
          )
        : await api.queryStream(
            trimmed,
            collection,
            record ? sessionId || 'new' : null,
            (token) => setStreamed((soFar) => soFar + token),
          )
      setStreamed('')
      setResult(payload)
      if (payload.session_id) setSessionId(payload.session_id)
      setQuestion(trimmed)
      setAttached([])
      if (answerRef.current) answerRef.current.scrollIntoView({ block: 'start' })
    } catch (err) {
      setResult(null)
      setError(err)
    } finally {
      setBusy(false)
    }
  }

  function submit(event) {
    event.preventDefault()
    ask()
  }

  function newConversation() {
    setSessionId(null)
    setResult(null)
    setQuestion('')
    setActive(null)
    setError(null)
    setAttached([])
    setAskedWith([])
  }

  return (
    <>
      <div className="card">
        <h2>Ask the corpus</h2>
        <form onSubmit={submit}>
          <label className="field">
            <span>Question</span>
            <textarea
              ref={questionRef}
              rows={1}
              value={question}
              placeholder="Ask something the indexed documents, images or audio should answer…"
              onChange={(e) => {
                setQuestion(e.target.value)
                // Grow with the question instead of scrolling inside one row: a
                // long question should be readable while it is being typed.
                const box = e.target
                box.style.height = 'auto'
                box.style.height = `${Math.min(box.scrollHeight, 200)}px`
              }}
              onKeyDown={(e) => {
                // Enter asks, Shift+Enter makes a new line. Ctrl/Cmd+Enter still
                // works, because the shortcuts overlay has always advertised it.
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  ask()
                } else if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
                  ask()
                }
              }}
            />
          </label>

          <div className="row" style={{ gap: 6, marginTop: 8, flexWrap: 'wrap' }}>
            <button
              type="button"
              className="ghost small"
              disabled={busy}
              onClick={() => fileRef.current?.click()}
              title="Attach documents or images to the next question"
            >
              📎 Attach files
            </button>
            <input
              ref={fileRef}
              type="file"
              multiple
              hidden
              accept=".pdf,.doc,.docx,.txt,.md,.csv,.png,.jpg,.jpeg,.webp,.bmp,.tiff,.tif"
              onChange={(e) => {
                addFiles(Array.from(e.target.files || []))
                e.target.value = ''
              }}
            />
            {attached.map((file, index) => (
              <span key={`${file.name}-${file.size}-${index}`} className="pill">
                {file.name}
                <button
                  type="button"
                  className="ghost small"
                  disabled={busy}
                  aria-label={`Remove ${file.name}`}
                  onClick={() =>
                    setAttached((current) => current.filter((_, j) => j !== index))
                  }
                >
                  ×
                </button>
              </span>
            ))}
            <span className="muted small">
              {attached.length
                ? 'answered from these files and the corpus; attachments are not indexed'
                : 'documents & images, up to 5 per question'}
            </span>
          </div>

          <div className="row spread">
            <div className="row">
              <label className="field" style={{ marginBottom: 0, minWidth: 190 }}>
                <span>Collection</span>
                <select value={collection} onChange={(e) => setCollection(e.target.value)}>
                  {collections.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </label>
              <button className="primary" type="submit" disabled={busy || !question.trim()}>
                {busy ? 'Retrieving and answering…' : 'Ask'}
              </button>
            </div>
            <span className="muted small">
              Enter to ask · Shift + Enter for a new line · first query loads the models
            </span>
          </div>
        </form>

        <div className="row spread" style={{ marginTop: 12 }}>
          <label className="row" style={{ gap: 6 }}>
            <input
              type="checkbox"
              checked={record}
              onChange={(e) => setRecord(e.target.checked)}
            />
            <span className="muted small">record this conversation in the audit history</span>
          </label>
          {sessionId ? (
            <div className="row" style={{ gap: 6 }}>
              <span className="pill">session {sessionId.slice(0, 8)}</span>
              <button type="button" className="ghost small" onClick={newConversation}>
                New conversation
              </button>
            </div>
          ) : null}
        </div>
        <p className="muted small" style={{ marginTop: 6 }}>
          {record
            ? 'Each turn is stored with its citations, unverified figures and pipeline trace, and can be read back or exported on the History tab.'
            : 'Recording is off: the answer is shown once and not stored. Useful for a probe you do not want in the trail.'}
        </p>

        <div className="row" style={{ marginTop: 12 }}>
          <span className="muted small">Try:</span>
          {SAMPLES.map((s) => (
            <button key={s} className="ghost small" onClick={() => ask(s)} disabled={busy}>
              {s}
            </button>
          ))}
        </div>
      </div>

      {busy && streamed && (
        <div className="card">
          <h2>Answer</h2>
          <p className="streaming">
            {streamed}
            <span className="stream-cursor" aria-hidden="true">▌</span>
          </p>
          <p className="muted small">
            Streaming — citations, scores and the pipeline trace arrive with the final
            result.
          </p>
        </div>
      )}

      {busy && (
        <div className="banner info">
          <strong>Working</strong>
          Retrieving candidates, reranking, checking the evidence against the
          question, then generating. A refusal is a real result here: if the corpus
          does not support an answer, that is what comes back. The first question
          after the service starts also loads the embedding, reranker and language
          models, so it can take half a minute; later questions are seconds.
        </div>
      )}

      {error && (
        <div className="banner bad" role="alert">
          <strong>Request failed ({error.status})</strong>
          {error.message}
          {error.status === 403 && (
            <div className="small" style={{ marginTop: 4 }}>
              Your role may not run this query. Ask an administrator for a role with
              the <code>ask</code> permission.
            </div>
          )}
        </div>
      )}

      {result && (
        <div ref={answerRef}>
          <div className="row" style={{ gap: 6, marginBottom: 8 }}>
            <span className={scope && scope !== 'general' ? 'pill ok' : 'pill'}>
              {scope && scope !== 'general'
                ? `scoped to ${scope}`
                : 'searching every collection'}
            </span>
            {askedWith.length ? (
              <span className="pill ok" title={askedWith.join(', ')}>
                with {askedWith.length} attachment{askedWith.length > 1 ? 's' : ''}
              </span>
            ) : null}
            <span className="pill">{asked}</span>
          </div>
          <Answer result={result} active={active} onSelect={setActive} can={can} />
          <Evidence result={result} active={active} onSelect={setActive} />
          <StageTrace result={result} />
        </div>
      )}
    </>
  )
}
