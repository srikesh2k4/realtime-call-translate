import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],

  // 🔧 Dev server (used by Vite + ngrok + Cloudflare)
  server: {
    host: "0.0.0.0",
    port: 5173,

    // ✅ Allow ALL hosts (Cloudflare, ngrok, IP, localhost)
    allowedHosts: true,

    // ✅ CORS - Allow all origins (Cloudflare tunnels)
    cors: true,

    // ✅ HMR through Cloudflare tunnel
    hmr: {
      clientPort: 443,  // Cloudflare uses HTTPS
      protocol: "wss",  // Secure WebSocket
    },

    // ✅ Headers for Cloudflare compatibility
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
      "Access-Control-Allow-Headers": "*",
    },

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

  // 🔍 Preview mode (used inside Docker or production preview)
  preview: {
    host: "0.0.0.0",
    port: 5173,
    
    // ✅ Allow ALL hosts for preview too
    cors: true,
    
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
      "Access-Control-Allow-Headers": "*",
    },

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