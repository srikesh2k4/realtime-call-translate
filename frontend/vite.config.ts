import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],

  // 🔧 Dev server (used by Vite + ngrok + IP access)
  server: {
    host: "0.0.0.0",
    port: 5173,

    // ✅ Allow ALL hosts (ngrok, IP, localhost)
    allowedHosts: true,

    // 🔌 WebSocket proxy to Go backend (DEV / local)
    proxy: {
      "/ws": {
        target: "ws://127.0.0.1:8000",
        ws: true,
        changeOrigin: true,
        secure: false,
      },
    },
  },

  // 🔍 Preview mode (used inside Docker)
  preview: {
    host: "0.0.0.0",
    port: 5173,

    // 🔌 WebSocket proxy to backend service in Docker network
    proxy: {
      "/ws": {
        target: process.env.BACKEND_URL || "ws://backend:8000",
        ws: true,
        changeOrigin: true,
        secure: false,
      },
    },
  },

  // 🧠 Prevent Vite from breaking ffmpeg / websocket deps
  optimizeDeps: {
    exclude: ["@ffmpeg/ffmpeg"],
  },

  // ⚙️ Build settings
  build: {
    target: "esnext",
  },
});