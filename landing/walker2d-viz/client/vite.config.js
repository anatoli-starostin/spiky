import { defineConfig } from 'vite'

// Dev server on :5173; the WS server URL is configurable in the UI (defaults to ws://<host>:8765).
export default defineConfig({
  server: { host: true, port: 5173 },
})
