/* Thin API client. Relative URLs only: the built bundle is served by the same
 * FastAPI process as the API, so there is no host to configure and no CORS
 * preflight. A token, when present, rides on every request. */

const TOKEN_KEY = 'evidenceai.token'
const USER_KEY = 'evidenceai.user'

export class ApiError extends Error {
  constructor(message, status, detail, retryAfter) {
    super(message)
    this.status = status
    this.detail = detail
    // A 429 carries Retry-After. The ingestion queue waits it out instead of
    // reporting a failure the user did nothing wrong to cause.
    this.retryAfter = retryAfter || null
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

async function request(path, { method = 'GET', body, formData, accept } = {}) {
  const { token } = readSession()
  const headers = {}
  if (token) headers.Authorization = `Bearer ${token}`
  if (body !== undefined) headers['Content-Type'] = 'application/json'

  const response = await fetch(path, {
    method,
    headers,
    body: formData || (body !== undefined ? JSON.stringify(body) : undefined),
  })

  // Exports are documents, not JSON: hand back the raw text and let the caller
  // turn it into a download.
  if (accept === 'text') {
    if (!response.ok) throw new ApiError(`${response.status} ${response.statusText}`, response.status)
    return response.text()
  }

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
    const retryAfter = Number(response.headers.get('Retry-After') || 0) || null
    throw new ApiError(message, response.status, detail, retryAfter)
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
  // `sessionId` opts the turn into chat history. Without it the question is
  // answered and forgotten, which is what a probe or a monitor should do.
  query: (text, collection, sessionId) =>
    request(
      `/query${sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : ''}`,
      {
        method: 'POST',
        body: {
          query: text,
          query_modality: 'text',
          ...(collection && collection !== 'general' ? { collection } : {}),
        },
      },
    ),
  health: () => request('/health'),
  // What the inference server actually has installed, next to what this build
  // configured. Diagnostic only: nothing here can change the configuration,
  // because the run manifest is the record of what produced a result.
  models: () => request('/models'),
  manifest: () => request('/manifest'),
  sources: () => request('/sources'),
  staleSources: () => request('/sources/stale'),
  sourceDetail: (id) => request(`/source/${encodeURIComponent(id)}`),
  deleteSource: (id) => request(`/source/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  collections: () => request('/collections'),
  createCollection: (name) =>
    request('/collections', { method: 'POST', body: { name } }),
  renameCollection: (from, to) =>
    request(`/collections/${encodeURIComponent(from)}`, { method: 'PATCH', body: { name: to } }),
  deleteCollection: (name) =>
    request(`/collections/${encodeURIComponent(name)}`, { method: 'DELETE' }),
  // One call moves any number of sources; the API reports which ids it did not
  // find instead of failing the batch, so a stale list still moves what is there.
  assignSources: (name, sourceIds) =>
    request(`/collections/${encodeURIComponent(name)}/sources`, {
      method: 'POST',
      body: { source_ids: sourceIds },
    }),

  historySessions: (limit = 50) => request(`/history/sessions?limit=${limit}`),
  createSession: (title) =>
    request('/history/sessions', { method: 'POST', body: { title: title || null } }),
  historySession: (id) => request(`/history/sessions/${encodeURIComponent(id)}`),
  renameSession: (id, title) =>
    request(`/history/sessions/${encodeURIComponent(id)}`, { method: 'PATCH', body: { title } }),
  deleteSession: (id) =>
    request(`/history/sessions/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  exportSession: (id, format) =>
    request(`/history/sessions/${encodeURIComponent(id)}/export?format=${format}`, {
      accept: 'text',
    }),

  /* The same answer as query(), arriving as it is produced.

   * Reads newline-delimited JSON from the response body: {type:'token'} lines
   * are handed to onToken as they land, and the final {type:'result'} line
   * carries exactly what query() would have returned, so the caller renders one
   * shape either way. A refusal sends no token lines at all - abstention happens
   * before generation on the server, and streaming must not make a half-second
   * refusal slow.
   *
   * Falls back to query() when the server has no streaming route (404/405) or
   * the browser exposes no readable body, so an older deployment still works.
   */
  queryStream: async (text, collection, sessionId, onToken) => {
    const { token } = readSession()
    const headers = { 'Content-Type': 'application/json' }
    if (token) headers.Authorization = `Bearer ${token}`
    const path = `/query_stream${sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : ''}`
    const body = JSON.stringify({
      query: text,
      query_modality: 'text',
      ...(collection && collection !== 'general' ? { collection } : {}),
    })

    let response
    try {
      response = await fetch(path, { method: 'POST', headers, body })
    } catch (err) {
      throw new ApiError(err.message || 'network error', 0)
    }
    if (response.status === 404 || response.status === 405 || !response.body) {
      return api.query(text, collection, sessionId)
    }
    if (!response.ok) {
      let detail = null
      try { detail = await response.json() } catch { detail = null }
      const inner = detail && detail.detail ? detail.detail : detail
      const message =
        (inner && (inner.error || inner.message)) ||
        `${response.status} ${response.statusText}`
      const retryAfter = Number(response.headers.get('Retry-After') || 0) || null
      throw new ApiError(message, response.status, inner, retryAfter)
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    let result = null
    for (;;) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      let cut
      while ((cut = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, cut).trim()
        buffer = buffer.slice(cut + 1)
        if (!line) continue
        let event
        try { event = JSON.parse(line) } catch { continue }
        if (event.type === 'token') onToken?.(event.content || '')
        else if (event.type === 'result') result = event.content
      }
    }
    if (!result) throw new ApiError('the stream ended without a result', 0)
    return result
  },

  ingest: (files, collection, replace) => {
    const formData = new FormData()
    for (const file of files) formData.append('files', file)
    formData.append('collection', collection || 'general')
    formData.append('replace', replace ? 'true' : 'false')
    return request('/ingest', { method: 'POST', formData })
  },
}
