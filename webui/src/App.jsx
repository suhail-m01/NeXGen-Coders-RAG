import React, { useCallback, useEffect, useState } from 'react'
import { api, clearSession, readSession, saveSession } from './api.js'
import Login from './components/Login.jsx'
import Ask from './components/Ask.jsx'
import Sources from './components/Sources.jsx'
import Health from './components/Health.jsx'
import History from './components/History.jsx'
import Shortcuts from './components/Shortcuts.jsx'
import Overview from './components/Overview.jsx'
import useKeyboardShortcuts from './hooks/useKeyboardShortcuts.js'

const TABS = [
  { id: 'ask', label: 'Ask' },
  { id: 'history', label: 'History' },
  { id: 'sources', label: 'Sources' },
  { id: 'health', label: 'Health & build' },
  // Appended rather than put first: Ask stays Alt+1 and the default tab, so the
  // shortcuts the documentation and the overlay have always advertised do not move.
  { id: 'overview', label: 'Overview' },
]

export default function App() {
  const [session, setSession] = useState(() => readSession())
  const [ready, setReady] = useState(false)
  const [tab, setTab] = useState('ask')
  const [shortcutsOpen, setShortcutsOpen] = useState(false)
  const [focusSignal, setFocusSignal] = useState(0)

  // Hooks must run in the same order on every render, so the shortcut map is
  // registered even before login and simply stays empty until the shell exists.
  // Alt+1..4 follow the tab order, which keeps the shortcut list honest when a
  // tab is added or reordered: it is derived from TABS, not written beside it.
  const signedIn = ready && !!session.user
  const shortcuts = React.useMemo(() => {
    if (!signedIn) return {}
    const map = {
      'ctrl+k': () => {
        setTab('ask')
        setFocusSignal((count) => count + 1)
      },
      '?': () => setShortcutsOpen(true),
      escape: () => setShortcutsOpen(false),
    }
    TABS.forEach((entry, index) => {
      map[`alt+${index + 1}`] = () => setTab(entry.id)
    })
    return map
  }, [signedIn])
  useKeyboardShortcuts(shortcuts)

  // Resolve the credential against the server on load rather than trusting what
  // the browser stored: a revoked token, a deleted account or a changed role must
  // show up here. With api_auth_enabled=false /me answers anonymously as admin,
  // so the same call also covers the no-auth mode.
  useEffect(() => {
    let alive = true
    const stored = readSession()
    api
      .me()
      .then((me) => {
        if (!alive) return
        saveSession(stored.token, me)
        setSession({ token: stored.token, user: me })
      })
      .catch(() => {
        if (!alive) return
        clearSession()
        setSession({ token: null, user: null })
      })
      .finally(() => alive && setReady(true))
    return () => {
      alive = false
    }
  }, [])

  const onLogin = useCallback((result) => {
    saveSession(result.token, result)
    setSession({ token: result.token, user: result })
    setTab('ask')
  }, [])

  const onLogout = useCallback(async () => {
    await api.logout()
    clearSession()
    setSession({ token: null, user: null })
  }, [])

  if (!ready) {
    return (
      <div className="shell">
        <p className="muted">Contacting the Evidence AI service…</p>
      </div>
    )
  }

  const user = session.user
  if (!user) {
    return (
      <div className="shell">
        <Login onLogin={onLogin} />
      </div>
    )
  }

  const can = (permission) => (user.permissions || []).includes(permission)

  return (
    <div className="shell">
      <header className="top">
        <div className="brand">
          <span className="mark" aria-hidden="true" />
          <h1>Evidence AI</h1>
          <span className="tag">cited answers from documents, images and audio — offline</span>
        </div>
        <div className="session">
          <span className="muted">{user.username}</span>
          <span className={`role-badge ${user.role}`}>{user.role}</span>
          <button className="ghost" onClick={onLogout}>
            Sign out
          </button>
        </div>
      </header>

      <nav className="tabs" role="tablist">
        {TABS.map((t) => (
          <button
            key={t.id}
            role="tab"
            aria-selected={tab === t.id}
            title={`Alt+${TABS.indexOf(t) + 1}`}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      {tab === 'ask' && <Ask can={can} role={user.role} focusSignal={focusSignal} />}
      {tab === 'history' && <History can={can} />}
      {tab === 'sources' && <Sources can={can} />}
      {tab === 'health' && <Health />}
      {tab === 'overview' && <Overview onNavigate={setTab} />}

      {shortcutsOpen && <Shortcuts tabs={TABS} onClose={() => setShortcutsOpen(false)} />}

      <p className="muted" style={{ marginTop: 24 }}>
        Every answer here carries the pipeline trace and the exact evidence it was
        built from. A refusal is reported with the check that produced it — the
        system never answers from a model's memory of the world.{' '}
        <button className="link" onClick={() => setShortcutsOpen(true)}>
          Keyboard shortcuts
        </button>
      </p>
    </div>
  )
}
