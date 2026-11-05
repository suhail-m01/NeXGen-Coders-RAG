/* Thin API client. Relative URLs only: the built bundle is served by the same
 * FastAPI process as the API, so there is no host to configure and no CORS
 * preflight. A token, when present, rides on every request. */

const TOKEN_KEY = 'evidenceai.token'
const USER_KEY = 'evidenceai.user'

export class ApiError extends Error {
  constructor(message, status, detail) {
    super(message)
    this.status = status
    this.detail = detail
  }
}

export function readSession() {
  try {
    const raw = localStorage.getItem(USER_KEY)
    return {
      token: localStorage.getItem(TOKEN_KEY),
      user: raw ? JSON.parse(raw) : null,
    }
  } catch {
    return { token: null, user: null }
  }
}

export function saveSession(token, user) {
  if (token) localStorage.setItem(TOKEN_KEY, token)
  else localStorage.removeItem(TOKEN_KEY)
  if (user) localStorage.setItem(USER_KEY, JSON.stringify(user))
  else localStorage.removeItem(USER_KEY)
}

export function clearSession() {
  localStorage.removeItem(TOKEN_KEY)
  localStorage.removeItem(USER_KEY)
}

async function request(path, { method = 'GET', body, formData } = {}) {
  const { token } = readSession()
  const headers = {}
  if (token) headers.Authorization = `Bearer ${token}`
  if (body !== undefined) headers['Content-Type'] = 'application/json'

  const response = await fetch(path, {
    method,
    headers,
    body: formData || (body !== undefined ? JSON.stringify(body) : undefined),
  })

  let payload = null
  const text = await response.text()
  if (text) {
    try { payload = JSON.parse(text) } catch { payload = { detail: text } }
  }
  if (!response.ok) {
    const detail = payload && payload.detail ? payload.detail : payload
    const message =
      (detail && (detail.error || detail.message)) ||
      (typeof detail === 'string' ? detail : '') ||
      `${response.status} ${response.statusText}`
    throw new ApiError(message, response.status, detail)
  }
  return payload
}

export const api = {
  login: (username, password) =>
    request('/login', { method: 'POST', body: { username, password } }),
  // First-run door: GET says whether the service still has no accounts, POST
  // creates the first administrator and closes behind them.
  bootstrap: () => request('/bootstrap'),
  createFirstAccount: (username, password) =>
    request('/bootstrap', { method: 'POST', body: { username, password } }),
  logout: () => request('/logout', { method: 'POST' }).catch(() => ({ revoked: false })),
  me: () => request('/me'),
  query: (text, collection) =>
    request('/query', {
      method: 'POST',
      body: {
        query: text,
        query_modality: 'text',
        ...(collection && collection !== 'general' ? { collection } : {}),
      },
    }),
  health: () => request('/health'),
  manifest: () => request('/manifest'),
  sources: () => request('/sources'),
  staleSources: () => request('/sources/stale'),
  sourceDetail: (id) => request(`/source/${encodeURIComponent(id)}`),
  deleteSource: (id) => request(`/source/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  ingest: (files, collection, replace) => {
    const formData = new FormData()
    for (const file of files) formData.append('files', file)
    formData.append('collection', collection || 'general')
    formData.append('replace', replace ? 'true' : 'false')
    return request('/ingest', { method: 'POST', formData })
  },
}
