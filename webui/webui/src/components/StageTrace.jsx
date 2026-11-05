import React from 'react'

/**
 * The per-query pipeline trace (observability.stage_trace).
 *
 * Every stage is listed whether or not it ran: a stage that never executed says
 * "not reached" and one switched off in config says which flag disabled it,
 * because "the guard did not fire" and "the guard is off" are different facts and
 * an operator has to be able to tell them apart.
 */
export default function StageTrace({ result }) {
  const stages = result.stages || []
  if (!stages.length) {
    return (
      <div className="card">
        <h2>Pipeline trace</h2>
        <p className="muted">
          This response carries no trace (it was produced before tracing was wired,
          or replayed from older chat history).
        </p>
      </div>
    )
  }

  const timed = stages.filter((s) => s.stage !== 'total' && typeof s.ms === 'number')
  const slowest = timed.reduce((max, s) => Math.max(max, s.ms), 0) || 1
  const total = stages.find((s) => s.stage === 'total')
  const decision = stages.find((s) => s.stage === 'decision')

  return (
    <div className="card">
      <h2>Pipeline trace</h2>
      <p className="muted small" style={{ marginTop: 0 }}>
        What ran for this question, how long each stage took, and what it moved.
      </p>

      {decision && (
        <div className="banner info">
          <strong>Decisive check</strong>
          {decision.note}
        </div>
      )}

      <table className="trace">
        <thead>
          <tr>
            <th style={{ width: '32%' }}>Stage</th>
            <th style={{ width: '18%' }}>Time</th>
            <th style={{ width: '16%' }} />
            <th>What it did</th>
          </tr>
        </thead>
        <tbody>
          {stages.map((stage) => (
            <tr
              key={stage.stage}
              data-ran={stage.ran}
              data-stage={stage.stage}
            >
              <td>
                {stage.label}
                {!stage.enabled && <span className="pill">off</span>}
                {stage.enabled && !stage.ran && <span className="pill">not reached</span>}
              </td>
              <td className="ms">
                {typeof stage.ms === 'number' ? `${stage.ms.toFixed(1)} ms` : '—'}
              </td>
              <td>
                <div className="bar" aria-hidden="true">
                  <span
                    style={{
                      width: `${
                        typeof stage.ms === 'number' && stage.stage !== 'total'
                          ? Math.max(2, (stage.ms / slowest) * 100)
                          : stage.stage === 'total'
                            ? 100
                            : 0
                      }%`,
                      opacity: stage.stage === 'total' ? 0.35 : 1,
                    }}
                  />
                </div>
              </td>
              <td className="small">{stage.note || ''}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {total && (
        <p className="muted small" style={{ marginTop: 10 }}>
          End to end {(total.ms / 1000).toFixed(2)} s
          {total.note ? ` · ${total.note}` : ''}. Generation dominates on a local
          model; retrieval and the guards are milliseconds.
        </p>
      )}
    </div>
  )
}
