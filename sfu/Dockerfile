# =====================================================
# Mediasoup SFU Dockerfile (Production)
# =====================================================

FROM node:20-bookworm-slim

# ---------------- SYSTEM DEPS ----------------
RUN apt-get update && apt-get install -y \
    python3 \
    make \
    g++ \
    gcc \
    libc6 \
    libstdc++6 \
    libssl-dev \
    ca-certificates \
    iproute2 \
    net-tools \
    dumb-init \
 && rm -rf /var/lib/apt/lists/*

# ---------------- WORKDIR ----------------
WORKDIR /app

# ---------------- NODE DEPENDENCIES ----------------
COPY package.json package-lock.json ./

# Mediasoup needs unsafe-perm inside Docker
RUN npm install --unsafe-perm --omit=dev

# ---------------- APP FILES ----------------
COPY server.js .
COPY mediasoup.js .

# ---------------- ENV ----------------
ENV NODE_ENV=production
ENV MEDIASOUP_MIN_PORT=40000
ENV MEDIASOUP_MAX_PORT=49999

# ---------------- NETWORK ----------------
EXPOSE 3000
EXPOSE 40000-49999/udp

# ---------------- RUN ----------------
ENTRYPOINT ["dumb-init", "--"]
CMD ["node", "server.js"]
