import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import basicSsl from "@vitejs/plugin-basic-ssl";

export default defineConfig({
  plugins: [basicSsl(), react()],

  server: {
    host: "0.0.0.0",
    port: 5173,

    // 🔐 REQUIRED for Cloudflare tunnels
    allowedHosts: [
      "localhost",
      "127.0.0.1",
      ".trycloudflare.com",
    ],

    // 🔁 WebSocket proxy ONLY for Go backend
    proxy: {
      "/ws": {
        target: "ws://127.0.0.1:8000",
        ws: true,
        changeOrigin: true,
        secure: false,
      },
    },
  },

  build: {
    target: "esnext",
  },
});
