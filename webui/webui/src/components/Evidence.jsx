import React, { useEffect, useState } from 'react'
import { api } from '../api.js'

/** Mirror of SourceLocation.human() on the server, so a citation reads the same
 *  way in the UI as it does in the evidence block the model was given. */
function clock(seconds) {
  const total = Math.max(0, Math.round(Number(seconds) || 0))
  const mm = String(Math.floor(total / 60)).padStart(2, '0')
  const ss = String(total % 60).padStart(2, '0')
  return `${mm}:${ss}`
}

function humanLocation(location) {
  if (!location) return ''
  const parts = []
  if (location.page_number != null) parts.push(`Page ${location.page_number}`)
  if (location.timestamp_start != null && location.timestamp_end != null) {
    parts.push(`${clock(location.timestamp_start)}-${clock(location.timestamp_end)}`)
  }
  if (location.section) parts.push(location.section)
  return parts.join(' - ')
}

function Excerpt({ text }) {
  const [open, setOpen] = useState(false)
  const body = String(text || '')
  const limit = 420
  if (body.length <= limit) return <div className="excerpt">{body}</div>
  return (
    <div className="excerpt">
      {open ? body : `${body.slice(0, limit)}… `}
      <button className="ghost small" onClick={() => setOpen(!open)}>
        {open ? 'Show less' : 'Show all'}
      </button>
    </div>
  )
}

export default function Evidence({ result, active, onSelect }) {
  const [sources, setSources] = useState({})
  const items = result.retrieved_items || []
  const citations = result.citations || []

  // ContentItem carries a source_id, not a filename; resolve names from /sources
  // so the panel can say which file each block came from.
  useEffect(() => {
    let alive = true
    api
      .sources()
      .then((data) => {
        if (!alive) return
        const map = {}
        for (const source of data.sources || []) map[source.source_id] = source
        setSources(map)
      })
      .catch(() => {})
    return () => {
      alive = false
    }
  }, [])

  // Selecting a citation in the answer scrolls its block into view, so the link
  // between a claim and the evidence for it is something you can follow, not
  // something you have to hunt for.
  useEffect(() => {
    if (active == null) return
    const citation = citations.find((entry) => entry.citation_id === active)
    if (!citation) return
    const node = document.getElementById(`evidence-${citation.item_id}`)
    if (node) node.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }, [active, citations])

  const citedBy = {}
  for (const citation of citations) {
    if (!citedBy[citation.item_id]) citedBy[citation.item_id] = []
    citedBy[citation.item_id].push(citation.citation_id)
  }

  const labelOf = {}
  for (const entry of items) {
    const item = entry.item
    const source = sources[item.source_id]
    labelOf[item.item_id] = `${source ? source.filename : item.source_id} · ${
      humanLocation(item.location) || item.modality
    }`
  }

  const relationships = (result.relationships || []).filter(
    (rel) => labelOf[rel.source_item_id] && labelOf[rel.target_item_id],
  )

  if (!items.length && !result.abstained) {
    return (
      <div className="card">
        <h2>Evidence</h2>
        <p className="muted">No items were retrieved for this question.</p>
      </div>
    )
  }

  return (
    <div className="card">
      <h2>Evidence behind the answer</h2>
      <p className="muted small" style={{ marginTop: 0 }}>
        These blocks, in this order, are what the model was allowed to use.
        Selecting a citation in the answer highlights the block it came from.
      </p>

      {items.map((entry) => {
        const item = entry.item
        const meta = item.metadata || {}
        const source = sources[item.source_id]
        const citationIds = citedBy[item.item_id] || []
        const isActive = citationIds.includes(active)
        const passages = Array.isArray(meta.retrieved_texts) ? meta.retrieved_texts : []

        return (
          <div
            key={item.item_id}
            id={`evidence-${item.item_id}`}
            className="evidence-item"
            data-active={isActive}
          >
            <div className="head">
              <span className="rank">#{entry.rank}</span>
              <span className="file">{source ? source.filename : item.source_id}</span>
              <span className="pill">{item.modality}</span>
              {source?.collection && source.collection !== 'general' && (
                <span className="pill">{source.collection}</span>
              )}
              <span className="where">{humanLocation(item.location)}</span>
              {citationIds.map((id) => (
                <button
                  key={id}
                  className="cite"
                  aria-pressed={active === id}
                  onClick={() => onSelect(active === id ? null : id)}
                >
                  {id}
                </button>
              ))}
              {meta.expanded_from && <span className="pill">expanded parent</span>}
            </div>

            <div className="scores">
              <span>retrieval {Number(entry.retrieval_score).toFixed(3)}</span>
              <span>rerank {Number(entry.rerank_score).toFixed(3)}</span>
              {meta.ocr_confidence != null && (
                <span>
                  OCR confidence {Number(meta.ocr_confidence).toFixed(1)}
                  {meta.ocr_words_kept != null &&
                    ` · ${meta.ocr_words_kept}/${meta.ocr_words_total} words kept`}
                  {meta.ocr_upscale > 1 && ` · upscaled ${meta.ocr_upscale}×`}
                </span>
              )}
            </div>

            {meta.ocr_unreliable && (
              <div className="banner warn small" style={{ margin: '8px 0 0' }}>
                OCR on this image fell below the digit-confidence floor: figures read
                from it were dropped rather than quoted.
                {meta.ocr_digits_dropped > 0 &&
                  ` ${meta.ocr_digits_dropped}/${meta.ocr_digits_total} digits dropped.`}
              </div>
            )}
            {meta.vision_ok === false && (
              <div className="banner info small" style={{ margin: '8px 0 0' }}>
                The configured model cannot see images, so this block is OCR text only.
              </div>
            )}

            {passages.length > 0 && (
              <div className="passage">
                <strong className="small">Matched passage</strong>
                {meta.retrieved_child && (
                  <span className="muted small"> · retrieved at {meta.retrieved_child}</span>
                )}
                <div className="excerpt" style={{ marginTop: 4 }}>
                  {passages.join('\n\n')}
                </div>
                <div className="muted small" style={{ marginTop: 4 }}>
                  The block below is the wider section this passage sits in; the guards
                  judged the passage, so a number appearing only elsewhere in the
                  section is not treated as supported.
                </div>
              </div>
            )}

            <Excerpt text={item.content} />
            {meta.merged_sections && (
              <div className="muted small" style={{ marginTop: 6 }}>
                Merged sections: {String(meta.merged_sections)}
              </div>
            )}
          </div>
        )
      })}

      {!items.length && result.abstained && (
        <p className="muted">
          Nothing was retrieved that could answer this, so the pipeline stopped before
          generation. The trace below shows which check refused it.
        </p>
      )}

      {relationships.length > 0 && (
        <>
          <h3>Links the index found between these blocks</h3>
          <ul className="small" style={{ marginTop: 0 }}>
            {relationships.slice(0, 8).map((rel) => (
              <li key={rel.relationship_id}>
                <code>{rel.relationship_type}</code>
                {rel.metadata?.cross_modal && <span className="pill">cross-modal</span>}{' '}
                {labelOf[rel.source_item_id]} → {labelOf[rel.target_item_id]}
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  )
}
