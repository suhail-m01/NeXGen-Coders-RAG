import React, { useEffect, useState } from 'react'
import { api } from '../api.js'

/**
 * Sign-in, and the one-shot first-run door beside it.
 *
 * A fresh install has no accounts, so nobody can sign in and nobody can be given
 * the permission that creates an account. The API exposes that state at
 * GET /bootstrap and closes POST /bootstrap the moment any account exists, so the
 * form below can only ever create the *first* administrator — the same rule the
 * Streamlit UI applies. After that, accounts are created by someone signed in
 * with manage_users, or headlessly with scripts/create_user.py.
 *
 * Accounts are local (auth/users.db), passwords are PBKDF2-HMAC-SHA256 with a
 * per-user salt, and five failed attempts lock an account for five minutes —
 * which the server reports as 423 with the unlock time, so a mistyping user is
 * told to stop instead of extending their own lock.
 */
export default function Login({ onLogin }) {
  const [mode, setMode] = useState('loading') // 'loading' | 'signin' | 'bootstrap'
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    let alive = true
    api
      .bootstrap()
      .then((status) => {
        if (alive) setMode(status.needs_bootstrap ? 'bootstrap' : 'signin')
      })
      .catch(() => alive && setMode('signin'))
    return () => {
      alive = false
    }
  }, [])

  async function submit(event) {
    event.preventDefault()
    if (mode === 'bootstrap' && password !== confirm) {
      setError('The two passwords do not match.')
      return
    }
    setBusy(true)
    setError(null)
    try {
      const result =
        mode === 'bootstrap'
          ? await api.createFirstAccount(username.trim(), password)
          : await api.login(username.trim(), password)
      onLogin(result)
    } catch (err) {
      if (err.status === 409) {
        // Somebody created the first account first. That is the door closing,
        // not a failure: fall back to signing in.
        setMode('signin')
        setConfirm('')
        setError('An account now exists, so the first-run form has closed. Sign in instead.')
      } else {
        setError(err.message)
      }
    } finally {
      setBusy(false)
    }
  }

  if (mode === 'loading') {
    return (
      <div className="login-wrap">
        <div className="card">
          <p className="muted">Checking whether this service has an administrator yet…</p>
        </div>
      </div>
    )
  }

  const bootstrapping = mode === 'bootstrap'

  return (
    <div className="login-wrap">
      <div className="card">
        <h1>Evidence AI</h1>
        <p className="muted" style={{ marginTop: 0 }}>
          {bootstrapping
            ? 'No account exists yet. Create the first administrator — this form closes as soon as one does.'
            : 'Sign in to ask questions against the indexed corpus.'}
        </p>

        {error && (
          <div className="banner bad" role="alert">
            <strong>{errorTitle(error)}</strong>
            {error}
          </div>
        )}

        <form onSubmit={submit}>
          <label className="field">
            <span>Username</span>
            <input
              type="text"
              value={username}
              autoComplete="username"
              onChange={(e) => setUsername(e.target.value)}
              required
            />
          </label>
          <label className="field">
            <span>Password</span>
            <input
              type="password"
              value={password}
              autoComplete={bootstrapping ? 'new-password' : 'current-password'}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </label>
          {bootstrapping && (
            <label className="field">
              <span>Confirm password</span>
              <input
                type="password"
                value={confirm}
                autoComplete="new-password"
                onChange={(e) => setConfirm(e.target.value)}
                required
              />
            </label>
          )}
          <button
            className="primary"
            type="submit"
            disabled={busy || !username || !password || (bootstrapping && !confirm)}
          >
            {busy ? 'Working…' : bootstrapping ? 'Create administrator' : 'Sign in'}
          </button>
        </form>

        <p className="muted" style={{ marginTop: 14 }}>
          {bootstrapping ? (
            <>
              The first account is an <code>admin</code>. You can add{' '}
              <code>analyst</code> (may ingest) and <code>viewer</code> (may only
              ask) accounts afterwards from the Streamlit operator UI or with{' '}
              <code>python scripts/create_user.py</code>. Equivalent on the command
              line:{' '}
              <code>python scripts/create_user.py demo demo-pass --role admin</code>.
            </>
          ) : (
            <>
              Roles: <code>viewer</code> asks questions, <code>analyst</code> may
              also ingest documents, <code>admin</code> may delete sources and
              manage accounts. Accounts are created with{' '}
              <code>python scripts/create_user.py demo demo-pass --role admin</code>.
            </>
          )}
        </p>
      </div>
    </div>
  )
}

function errorTitle(message) {
  const text = String(message || '').toLowerCase()
  if (text.includes('lock')) return 'Account locked'
  if (text.includes('password')) return 'Password refused'
  return 'Not signed in'
}
