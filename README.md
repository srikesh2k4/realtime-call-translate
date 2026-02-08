# Live Call Translation

Frontend runs on HTTPS by default (required for microphone access). The Vite dev server uses a self-signed certificate via `@vitejs/plugin-basic-ssl`.

## Run frontend over HTTPS

1) Install deps: `cd frontend && npm install`
2) Start dev server: `npm run dev`
3) Open `https://localhost:5173` (the browser will warn about the self-signed cert—accept/trust it once to enable audio input).

If you still see a microphone permission error, ensure you're using HTTPS (note the lock icon) or the exact `localhost` host.
