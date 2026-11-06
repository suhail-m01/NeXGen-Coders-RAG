import React from 'react'

/**
 * The ingestion queue, one row per file.
 *
 * Files are uploaded one request each rather than as a batch, so a row can say
 * what happened to *that* file: how many items it produced, which warnings came
 * back with it, or why it failed. A single multi-file request answers "the batch
 * is done" and leaves the user reading a wall of text to find the one PDF that
 * was password protected.
 *
 * Uploads run one at a time on purpose. Embedding is CPU-bound and this service
 * is meant to run on a laptop that is also doing the demoing; four parallel
 * uploads do not finish sooner, they finish later and make the machine unusable
 * while they do it.
 */
function bytes(size) {
  if (size == null) return '—'
  if (size < 1024) return `${size} B`
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(0)} KB`
  return `${(size / (1024 * 1024)).toFixed(1)} MB`
}

const STATUS = {
  queued: { label: 'queued', className: 'pill' },
  uploading: { label: 'indexing…', className: 'pill busy' },
  waiting: { label: 'waiting on the rate limit', className: 'pill busy' },
  indexed: { label: 'indexed', className: 'pill ok' },
  warning: { label: 'indexed with warnings', className: 'pill warn' },
  failed: { label: 'failed', className: 'pill bad' },
  refused: { label: 'not accepted', className: 'pill bad' },
}

export default function FileQueue({ queue, uploading, onIngest, onRemove, onClear, collection }) {
  if (!queue || queue.length === 0) return null

  const pending = queue.filter((entry) => entry.status === 'queued' || entry.status === 'failed')
  const done = queue.filter((entry) => entry.status === 'indexed' || entry.status === 'warning')
  const failed = queue.filter((entry) => entry.status === 'failed' || entry.status === 'refused')

  return (
    <div className="queue">
      <div className="row spread queue-head">
        <strong>
          {queue.length} file{queue.length === 1 ? '' : 's'} queued
          {done.length ? ` · ${done.length} indexed` : ''}
          {failed.length ? ` · ${failed.length} failed` : ''}
          {collection ? ` · into “${collection}”` : ''}
        </strong>
        <div className="row" style={{ gap: 6 }}>
          <button
            type="button"
            className="primary"
            disabled={uploading || pending.length === 0}
            onClick={onIngest}
          >
            {uploading
              ? 'Indexing…'
              : pending.length
                ? `Ingest ${pending.length} file${pending.length === 1 ? '' : 's'}`
                : 'Nothing left to ingest'}
          </button>
          {(done.length > 0 || failed.length > 0) && !uploading ? (
            <button type="button" className="ghost small" onClick={onClear}>
              Clear finished
            </button>
          ) : null}
        </div>
      </div>

      <ul className="queue-list">
        {queue.map((entry) => {
          const status = STATUS[entry.status] || STATUS.queued
          return (
            <li className={`queue-row ${entry.status}`} key={entry.id}>
              <div className="queue-row-main">
                <span className={`queue-status ${status.className}`}>{status.label}</span>
                <span className="queue-name" title={entry.name}>
                  {entry.name}
                </span>
                <span className="muted small queue-size">{bytes(entry.size)}</span>
                {entry.items != null ? (
                  <span className="muted small">
                    {entry.items} item{entry.items === 1 ? '' : 's'} indexed
                  </span>
                ) : null}
                {!uploading && entry.status !== 'uploading' ? (
                  <button
                    type="button"
                    className="ghost small queue-remove"
                    onClick={() => onRemove(entry.id)}
                    aria-label={`Remove ${entry.name} from the queue`}
                  >
                    remove
                  </button>
                ) : null}
              </div>
              {entry.error ? <p className="queue-detail bad">{entry.error}</p> : null}
              {entry.waiting ? (
                <p className="queue-detail muted">
                  The rate limiter asked us to wait {entry.waiting}s. Ingestion is limited
                  to protect the machine, because every file costs seconds of embedding
                  time; the queue resumes on its own.
                </p>
              ) : null}
              {entry.warnings && entry.warnings.length > 0 ? (
                <ul className="queue-detail queue-warnings">
                  {entry.warnings.map((warning, index) => (
                    <li key={index}>{warning}</li>
                  ))}
                </ul>
              ) : null}
            </li>
          )
        })}
      </ul>

      {queue.length > 20 ? (
        <p className="muted small" style={{ marginBottom: 0 }}>
          More than twenty files: this will take a while, one at a time, and the ingest
          rate limit will pace it. For a large corpus it is faster to put the files in{' '}
          <code>data\documents</code> and run <code>python scripts\reindex.py</code>, which
          indexes from disk and never goes through HTTP.
        </p>
      ) : null}
    </div>
  )
}
