import express from "express";
import http from "http";
import { Server } from "socket.io";
import { initMediasoup, getRouter } from "./mediasoup.js";

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: { origin: "*" }
});

const rooms = new Map();

await initMediasoup();

io.on("connection", socket => {
  console.log("🔌 Client connected:", socket.id);

  socket.on("joinRoom", async ({ roomId }, cb) => {
    if (!rooms.has(roomId)) {
      rooms.set(roomId, {
        router: getRouter(),
        peers: new Map()
      });
    }

    rooms.get(roomId).peers.set(socket.id, {
      socket,
      transports: [],
      producers: [],
      consumers: []
    });

    socket.roomId = roomId;

    cb(rooms.get(roomId).router.rtpCapabilities);
  });

  socket.on("createTransport", async (_, cb) => {
    const router = rooms.get(socket.roomId).router;

    const transport = await router.createWebRtcTransport({
      listenIps: [{
        ip: "0.0.0.0",
        announcedIp: process.env.MEDIASOUP_ANNOUNCED_IP || null
      }],
      enableUdp: true,
      enableTcp: true,
      preferUdp: true
    });

    rooms.get(socket.roomId).peers
      .get(socket.id)
      .transports.push(transport);

    cb({
      id: transport.id,
      iceParameters: transport.iceParameters,
      iceCandidates: transport.iceCandidates,
      dtlsParameters: transport.dtlsParameters
    });
  });

  socket.on("connectTransport", async ({ transportId, dtlsParameters }) => {
    const peer = rooms.get(socket.roomId).peers.get(socket.id);
    const transport = peer.transports.find(t => t.id === transportId);
    await transport.connect({ dtlsParameters });
  });

  socket.on("produce", async ({ transportId, kind, rtpParameters }, cb) => {
    const peer = rooms.get(socket.roomId).peers.get(socket.id);
    const transport = peer.transports.find(t => t.id === transportId);

    const producer = await transport.produce({
      kind,
      rtpParameters
    });

    peer.producers.push(producer);

    cb({ id: producer.id });
  });

  socket.on("disconnect", () => {
    const room = rooms.get(socket.roomId);
    if (!room) return;

    const peer = room.peers.get(socket.id);
    if (!peer) return;

    peer.transports.forEach(t => t.close());
    peer.producers.forEach(p => p.close());
    peer.consumers.forEach(c => c.close());

    room.peers.delete(socket.id);
    console.log("❌ Client disconnected:", socket.id);
  });
});

server.listen(3001, () => {
  console.log("🚀 SFU running on port 3001");
});
