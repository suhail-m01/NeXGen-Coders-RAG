import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Every API path the UI calls. In production the bundle is served by FastAPI
// itself, so these are same-origin relative URLs and no proxy is involved; the
// proxy exists only for `npm run dev`, where Vite serves on 5173 and the API on
// 8000. Nothing in src/ ever names a host, which is what keeps the built bundle
// portable and offline.
const API_PATHS = [
  '/query', '/query_file', '/ingest', '/sources', '/source', '/health',
  '/manifest', '/login', '/logout', '/me', '/bootstrap',
]

export default defineConfig({
  // Relative asset URLs, so the bundle works mounted at /ui/ or opened from any
  // prefix without a rebuild.
  base: './',
  plugins: [react()],
  build: { outDir: 'dist', emptyOutDir: true },
  server: {
    port: 5173,
    proxy: API_PATHS.reduce((proxy, path) => {
      proxy[path] = { target: 'http://127.0.0.1:8000', changeOrigin: true }
      return proxy
    }, {}),
  },
})
