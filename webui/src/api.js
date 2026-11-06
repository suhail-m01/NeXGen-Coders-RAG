/* Thin API client. Relative URLs only: the built bundle is served by the same
 * FastAPI process as the API, so there is no host to configure and no CORS
 * preflight. A token, when present, rides on every request. */

const TOKEN_KEY = 'evidenceai.token'
const USER_KEY = 'evidenceai.user'

export class ApiError extends Error {
  constructor(message, status, detail, retryAfter, missing) {
    super(message)
    this.status = status
    this.detail = detail
    // A 429 carries Retry-After. The ingestion queue waits it out instead of
    // reporting a failure the user did nothing wrong to cause.
    this.retryAfter = retryAfter || null
    // True when the server said "Missing bearer token" although this client
    // believes it holds one: something between the two dropped the header.
    this.missing = !!missing
  }
}

/* The session lives in memory first and in localStorage second.

 * A sandboxed embed viewer - an iframe without allow-same-origin, which is
 * exactly how preview panes ship a running app - makes every localStorage
 * call throw. The old code read the token back out of storage on each
 * request, so in that context the shell rendered "signed in" from React
 * state while every data call went out unauthenticated and came back
 * 401 Missing bearer token. Memory is now the source of truth while the page
 * lives; storage is only a best-effort bridge across reloads. */
let memoryToken = null
let memoryUser = null
// Flipped once a request had to retry with the token as a query parameter,
// i.e. something between the browser and the server ate the Authorization
// header. From then on every request carries the token in the URL as well.
let queryTransport = false

function storageGet(key) {
  try {
    return window.localStorage.getItem(key)
  } catch {
    return null
  }
}

function storageSet(key, value) {
  try {
    if (value === null) window.localStorage.removeItem(key)
    else window.localStorage.setItem(key, value)
  } catch {
    /* Blocked storage is not fatal: memory still carries the session. */
  }
}

function safeParse(raw) {
  try {
    return JSON.parse(raw)
  } catch {
    return null
  }
}

export function readSession() {
  if (memoryToken || memoryUser) return { token: memoryToken, user: memoryUser }
  const raw = storageGet(USER_KEY)
  memoryToken = storageGet(TOKEN_KEY)
  memoryUser = raw ? safeParse(raw) : null
  return { token: memoryToken, user: memoryUser }
}

export function saveSession(token, user) {
  memoryToken = token || null
  memoryUser = user || null
  storageSet(TOKEN_KEY, memoryToken)
  storageSet(USER_KEY, memoryUser ? JSON.stringify(memoryUser) : null)
}

export function clearSession() {
  memoryToken = null
  memoryUser = null
  storageSet(TOKEN_KEY, null)
  storageSet(USER_KEY, null)
}

function authHeaders() {
  const { token } = readSession()
  const headers = {}
  if (token) {
    headers.Authorization = `Bearer ${token}`
    // Second transport on purpose: a proxy or embed viewer that strips
    // Authorization still forwards an unknown header, so the session
    // survives where the standard one does not.
    headers['X-EAI-Token'] = token
  }
  return headers
}

function withQueryToken(path) {
  const { token } = readSession()
  if (!token) return path
  return path + (path.includes('?') ? '&' : '?') + 'access_token=' + encodeURIComponent(token)
}

async function request(path, options = {}) {
  try {
    return await sendRequest(path, options, queryTransport)
  } catch (err) {
    // "Missing bearer token" while holding a token means the header never
    // arrived. Retry once with the token in the URL, and remember that this
    // deployment needs that transport instead of guessing again per call.
    if (err instanceof ApiError && err.status === 401 && err.missing && !options._retried) {
      queryTransport = true
      return sendRequest(path, { ...options, _retried: true }, true)
    }
    throw err
  }
}

async function sendRequest(
  path,
  { method = 'GET', body, formData, accept, _retried } = {},
  useQueryToken = false,
) {
  const headers = authHeaders()
  if (body !== undefined) headers['Content-Type'] = 'application/json'

  const response = await fetch(useQueryToken ? withQueryToken(path) : path, {
    method,
    headers,
    body: formData || (body !== undefined ? JSON.stringify(body) : undefined),
  })

  // Exports are documents, not JSON: hand back the raw text and let the caller
  // turn it into a download.
  if (accept === 'text') {
    if (!response.ok) {
      const probe = await response.text()
      const missing = probe.includes('Missing bearer token')
      throw new ApiError(`${response.status} ${response.statusText}`, response.status, probe, null, missing)
    }
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
    const missing = String(message).includes('Missing bearer token')
    throw new ApiError(message, response.status, detail, retryAfter, missing)
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
    const headers = { 'Content-Type': 'application/json', ...authHeaders() }
    const path = `/query_stream${sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : ''}`
    const body = JSON.stringify({
      query: text,
      query_modality: 'text',
      ...(collection && collection !== 'general' ? { collection } : {}),
    })

    let response
    try {
      response = await fetch(queryTransport ? withQueryToken(path) : path, {
        method: 'POST',
        headers,
        body,
      })
    } catch (err) {
      throw new ApiError(err.message || 'network error', 0)
    }
    if (response.status === 401 && !queryTransport) {
      // Same fallback as request(): if the header was stripped in transit,
      // replay the question once with the token in the URL. A streamed answer
      // cannot be un-said, so this happens before a single token is read.
      let missing = false
      try {
        const probe = await response.clone().json()
        const inner = probe && probe.detail ? probe.detail : probe
        missing = String((inner && inner.error) || '').includes('Missing bearer token')
      } catch {
        missing = false
      }
      if (missing) {
        queryTransport = true
        response = await fetch(withQueryToken(path), { method: 'POST', headers, body })
      }
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

  /* A question with documents or images attached, answered from them.

   * Multipart out, the same NDJSON contract back, including the 401-missing
   * fallback: an embed viewer that dropped the Authorization header gets the
   * same one-retry-with-the-token-in-the-URL treatment as every other call.
   * The browser sets the multipart boundary itself, so the only headers sent
   * are the auth ones. */
  queryAttachStream: async (text, attachedFiles, collection, sessionId, onToken) => {
    const headers = authHeaders()
    const form = new FormData()
    form.append('query', text)
    if (sessionId) form.append('session_id', sessionId)
    if (collection && collection !== 'general') form.append('collection', collection)
    for (const file of attachedFiles) form.append('files', file)
    const path = '/query_attachment'

    let response
    try {
      response = await fetch(queryTransport ? withQueryToken(path) : path, {
        method: 'POST',
        headers,
        body: form,
      })
    } catch (err) {
      throw new ApiError(err.message || 'network error', 0)
    }
    if (response.status === 401 && !queryTransport) {
      let missing = false
      try {
        const probe = await response.clone().json()
        const inner = probe && probe.detail ? probe.detail : probe
        missing = String((inner && inner.error) || '').includes('Missing bearer token')
      } catch {
        missing = false
      }
      if (missing) {
        queryTransport = true
        response = await fetch(withQueryToken(path), { method: 'POST', headers, body: form })
      }
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
    if (!response.body) throw new ApiError('the server sent no stream', 0)

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
