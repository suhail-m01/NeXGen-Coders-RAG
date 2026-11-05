import React, { useEffect, useState } from 'react'
import { api } from '../api.js'

function Flag({ label, on }) {
  return (
    <span className="pill" style={on ? { borderColor: '#1f7a4d', color: '#1f7a4d' } : undefined}>
      {label} {on ? 'on' : 'off'}
    </span>
  )
}

function Stat({ label, value }) {
  return (
    <div className="metric">
      <div className="value">{value}</div>
      <div className="label">{label}</div>
    </div>
  )
}

/** Render a manifest section (nested plain data) as a definition list. */
function Section({ title, value }) {
  if (value == null) return null
  if (typeof value !== 'object') {
    return (
      <div className="row small">
        <strong>{title}</strong>
        <span className="mono">{String(value)}</span>
      </div>
    )
  }
  return (
    <div style={{ marginBottom: 10 }}>
      <h3 style={{ marginBottom: 4 }}>{title}</h3>
      <table className="grid">
        <tbody>
          {Object.entries(value).map(([key, entry]) => (
            <tr key={key}>
              <td className="mono" style={{ width: '34%' }}>
                {key}
              </td>
              <td className="mono">
                {entry != null && typeof entry === 'object' ? JSON.stringify(entry) : String(entry)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/**
 * Service health and the build manifest.
 *
 * Two different questions live here. "Is the service usable right now?" is the
 * health block: backend readiness, OCR and audio dependencies, index size, host
 * headroom. "Which exact build produced this result?" is the manifest: commit,
 * models, every retrieval and guard setting, dataset version. An evaluation score
 * or a screenshot of an answer is only reproducible if the second one is recorded
 * too, so both are one click away and both are also in the API (/health,
 * /manifest) and in every evaluation report.
 */
export default function Health() {
  const [health, setHealth] = useState(null)
  const [manifest, setManifest] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let alive = true
    Promise.all([api.health(), api.manifest().catch(() => null)])
      .then(([healthData, manifestData]) => {
        if (!alive) return
        setHealth(healthData)
        setManifest(manifestData)
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
        <strong>Health check failed ({error.status})</strong>
        {error.message}
      </div>
    )
  }
  if (!health) return null

  const runtime = health.runtime || {}
  const index = health.index || {}
  const guards = runtime.guards || {}
  const generation = runtime.generation || {}
  const ram = runtime.ram_mb || {}

  return (
    <>
      <div className="card">
        <div className="row spread">
          <h2 style={{ margin: 0 }}>Service status</h2>
          <span
            className={`role-badge ${health.status === 'ok' ? 'admin' : 'viewer'}`}
            style={health.status === 'ok' ? { background: '#1f7a4d', color: '#fff' } : { background: '#a15c00', color: '#fff' }}
          >
            {health.status}
          </span>
        </div>

        {(runtime.notes || []).map((note) => (
          <div className="banner warn small" key={note}>
            {note}
          </div>
        ))}

        <div className="metrics">
          <Stat label="Backend" value={health.backend || '—'} />
          <Stat label="LLM model" value={health.models?.llm || '—'} />
          <Stat label="Embedding" value={health.models?.embedding || '—'} />
          <Stat label="Reranker" value={health.models?.reranker || '—'} />
        </div>

        <p className="small" style={{ marginTop: 12 }}>
          {health.inference?.ready
            ? `Inference ready — ${health.inference?.label || health.backend}.`
            : `Inference not ready — ${health.inference?.message || 'see detail'}.`}{' '}
          OCR (Tesseract): {health.tesseract?.available ? 'available' : 'missing'}. Audio
          transcription model: <code>{health.models?.whisper || '—'}</code>.
        </p>
        {health.inference?.technical && !health.inference?.ready && (
          <pre className="mono small" style={{ whiteSpace: 'pre-wrap' }}>
            {health.inference.technical}
          </pre>
        )}
      </div>

      <div className="card">
        <h2>Index</h2>
        <div className="metrics">
          <Stat label="Sources" value={index.sources ?? '—'} />
          <Stat label="Items" value={index.items ?? '—'} />
          <Stat label="Relationships" value={index.relationships ?? '—'} />
          <Stat label="Parent sections" value={index.parents ?? 0} />
        </div>
        <p className="muted small">
          Chunking mode <code>{runtime.chunking_mode || '—'}</code>. Parent sections
          exist only when the index was built in <code>structure</code> mode; in{' '}
          <code>fixed</code> mode every retrieved block is quoted as-is.
        </p>
      </div>

      <div className="card">
        <h2>Guards and generation</h2>
        <div className="row">
          {Object.entries(guards).map(([name, on]) => (
            <Flag key={name} label={name.replace(/_/g, ' ')} on={Boolean(on)} />
          ))}
        </div>
        <p className="small" style={{ marginTop: 10 }}>
          Generation: temperature <code>{String(generation.temperature ?? '—')}</code>, seed{' '}
          <code>{String(generation.seed ?? '—')}</code>, context{' '}
          <code>{String(generation.num_ctx ?? '—')}</code> tokens, output capped at{' '}
          <code>{String(generation.num_predict ?? '—')}</code> tokens. The output cap is
          what stops a small local model looping until the request times out.
        </p>
      </div>

      <div className="card">
        <h2>Host and build</h2>
        <div className="metrics">
          <Stat label="Commit" value={(runtime.git_commit || '—').slice(0, 7)} />
          <Stat label="Working tree" value={runtime.git_dirty ? 'dirty' : 'clean'} />
          <Stat label="CPUs" value={runtime.cpu_count ?? '—'} />
          <Stat
            label="RAM available"
            value={ram.available != null ? `${Math.round(ram.available)} MB` : '—'}
          />
          <Stat
            label="Disk free"
            value={runtime.disk_free_mb != null ? `${Math.round(runtime.disk_free_mb)} MB` : '—'}
          />
          <Stat
            label="Uptime"
            value={runtime.uptime_s != null ? `${Math.round(runtime.uptime_s)} s` : '—'}
          />
        </div>
        <p className="small" style={{ marginTop: 10 }}>
          Python <code>{runtime.python || '—'}</code> on{' '}
          <code>{runtime.platform || '—'}</code>. API authentication{' '}
          {runtime.auth_enabled ? 'enabled' : 'disabled'}. Evaluation dataset version{' '}
          <code>{String(runtime.dataset_version ?? '—')}</code>.
        </p>
        {runtime.git_dirty && (
          <div className="banner warn small">
            The working tree has uncommitted changes, so this build is not exactly the
            commit shown. Commit before quoting a result.
          </div>
        )}
      </div>

      <div className="card">
        <h2>Full run manifest</h2>
        <p className="muted small" style={{ marginTop: 0 }}>
          The same manifest is embedded in every evaluation report and available at{' '}
          <code>GET /manifest</code> or <code>python scripts/manifest.py</code>.
        </p>
        {manifest ? (
          <details>
            <summary className="small">Show all {Object.keys(manifest).length} sections</summary>
            <div style={{ marginTop: 12 }}>
              {Object.entries(manifest).map(([key, value]) => (
                <Section key={key} title={key.replace(/_/g, ' ')} value={value} />
              ))}
            </div>
          </details>
        ) : (
          <p className="muted small">
            The manifest endpoint refused this request (it needs a signed-in role with
            the <code>ask</code> permission).
          </p>
        )}
      </div>
    </>
  )
}
