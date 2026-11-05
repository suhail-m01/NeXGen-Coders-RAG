import React, { useEffect, useRef, useState } from 'react'
import { api } from '../api.js'
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

export default function Ask({ can }) {
  const [question, setQuestion] = useState('')
  const [collection, setCollection] = useState('general')
  const [collections, setCollections] = useState(['general'])
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [active, setActive] = useState(null)
  const answerRef = useRef(null)

  useEffect(() => {
    api
      .sources()
      .then((data) => {
        // /sources reports collections as {collection, sources} rows.
        const names = (data.collections || []).map((row) =>
          typeof row === 'string' ? row : row.collection,
        )
        if (names.length) {
          setCollections(names.includes('general') ? names : ['general', ...names])
        }
      })
      .catch(() => {})
  }, [])

  async function ask(text) {
    const trimmed = (text ?? question).trim()
    if (!trimmed || busy) return
    setBusy(true)
    setError(null)
    setActive(null)
    try {
      setResult(await api.query(trimmed, collection))
      setQuestion(trimmed)
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

  return (
    <>
      <div className="card">
        <h2>Ask the corpus</h2>
        <form onSubmit={submit}>
          <label className="field">
            <span>Question</span>
            <textarea
              value={question}
              placeholder="Ask something the indexed documents, images or audio should answer…"
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') ask()
              }}
            />
          </label>

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
            <span className="muted small">Ctrl / ⌘ + Enter · first query loads the models</span>
          </div>
        </form>

        <div className="row" style={{ marginTop: 12 }}>
          <span className="muted small">Try:</span>
          {SAMPLES.map((s) => (
            <button key={s} className="ghost small" onClick={() => ask(s)} disabled={busy}>
              {s}
            </button>
          ))}
        </div>
      </div>

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
          <Answer result={result} active={active} onSelect={setActive} can={can} />
          <Evidence result={result} active={active} onSelect={setActive} />
          <StageTrace result={result} />
        </div>
      )}
    </>
  )
}
