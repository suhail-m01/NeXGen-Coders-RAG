import React from 'react'

/* Sentences the guards append after generation. They are shown as distinct
 * blocks rather than left buried in the paragraph: a caveat that reads like part
 * of the answer is a caveat nobody notices. These are the only three strings the
 * pipeline appends (numeric_guard, constraint_guard/synthesis, streaming
 * correction), so the list is checked in one place. */
const GUARD_PREFIXES = [
  { prefix: 'Evidence warning:', kind: 'caveat' },
  { prefix: 'Correction:', kind: 'caveat' },
  { prefix: 'Also stated in the evidence:', kind: 'added' },
]

const GUARD_RE = new RegExp(`(${GUARD_PREFIXES.map((g) => g.prefix).join('|')})`, 'g')

function classify(marker) {
  const found = GUARD_PREFIXES.find((g) => g.prefix === marker)
  return found ? found.kind : 'claim'
}

/** Split an answer into claim / caveat / added blocks, keeping the text intact. */
export function splitAnswer(text) {
  const blocks = []
  let kind = 'claim'
  let buffer = ''

  const flush = () => {
    if (buffer.trim()) blocks.push({ kind, text: buffer.trim() })
    buffer = ''
  }

  for (const part of String(text || '').split(GUARD_RE)) {
    if (GUARD_PREFIXES.some((g) => g.prefix === part)) {
      flush()
      kind = classify(part)
      buffer = part
    } else {
      buffer += part
    }
  }
  flush()
  return blocks
}

/** Render one block, turning [n] markers into buttons that select that citation. */
function Block({ text, citations, active, onSelect, kind }) {
  const known = new Set(citations.map((c) => c.citation_id))
  const pieces = text.split(/\[(\d+)\]/)

  return (
    <div className={kind === 'claim' ? 'claim' : kind}>
      {pieces.map((piece, index) => {
        if (index % 2 === 1) {
          const number = Number(piece)
          if (!known.has(number)) return `[${piece}]`
          return (
            <button
              key={`c${index}`}
              className="cite"
              aria-pressed={active === number}
              aria-label={`Show evidence ${number}`}
              title={`Evidence ${number}`}
              onClick={() => onSelect(active === number ? null : number)}
            >
              {number}
            </button>
          )
        }
        return <React.Fragment key={`t${index}`}>{piece}</React.Fragment>
      })}
    </div>
  )
}

export default function Answer({ result, active, onSelect }) {
  const citations = result.citations || []
  const items = result.retrieved_items || []
  const total = result.latency_ms?.total_ms
  const blocks = splitAnswer(result.answer)

  return (
    <div className="card">
      <h2>Answer</h2>

      {result.abstained && (
        <div className="banner bad" role="status">
          <strong>Refused to answer{result.error_code ? ` — ${result.error_code}` : ''}</strong>
          The evidence in the corpus does not support an answer to this question, so
          none was generated. This is the intended behaviour for a question the
          indexed material cannot answer.
        </div>
      )}

      <div className="answer">
        {blocks.length ? (
          blocks.map((block, index) => (
            <Block
              key={index}
              text={block.text}
              kind={block.kind}
              citations={citations}
              active={active}
              onSelect={onSelect}
            />
          ))
        ) : (
          <p className="muted">No answer text was produced.</p>
        )}
      </div>

      {result.technical_detail && (
        <details style={{ marginTop: 10 }}>
          <summary className="muted small">Technical detail</summary>
          <pre className="mono small" style={{ whiteSpace: 'pre-wrap' }}>
            {result.technical_detail}
          </pre>
        </details>
      )}

      {result.query_representation && (
        <details style={{ marginTop: 6 }}>
          <summary className="muted small">Search representation used for retrieval</summary>
          <p className="mono small" style={{ whiteSpace: 'pre-wrap' }}>
            {result.query_representation}
          </p>
        </details>
      )}

      {citations.length > 0 && (
        <>
          <h3>Cited evidence</h3>
          <div className="row">
            {citations.map((citation) => (
              <button
                key={citation.citation_id}
                className="ghost small"
                aria-pressed={active === citation.citation_id}
                onClick={() =>
                  onSelect(active === citation.citation_id ? null : citation.citation_id)
                }
                title={citation.excerpt || ''}
              >
                [{citation.citation_id}] {citation.filename}
                {citation.location?.page_number != null && ` · p.${citation.location.page_number}`}
                {citation.location?.section ? ` · ${citation.location.section}` : ''}
              </button>
            ))}
          </div>
        </>
      )}

      <div className="metrics">
        <div className="metric">
          <div className="value">
            {result.confidence == null ? '—' : result.confidence.toFixed(3)}
          </div>
          <div className="label">Confidence</div>
        </div>
        <div className="metric">
          <div className="value">{citations.length}</div>
          <div className="label">Citations</div>
        </div>
        <div className="metric">
          <div className="value">{items.length}</div>
          <div className="label">Evidence shown</div>
        </div>
        <div className="metric">
          <div className="value">{total ? (total / 1000).toFixed(2) : '—'}s</div>
          <div className="label">End to end</div>
        </div>
      </div>
      <p className="muted small" style={{ marginTop: 8 }}>
        Confidence is the best cross-encoder score normalised to 0–1. It orders
        answers; it is not a calibrated probability that this one is right.
      </p>
    </div>
  )
}
