import mediasoup from "mediasoup";

let worker;
let router;

const mediaCodecs = [
  {
    kind: "audio",
    mimeType: "audio/opus",
    clockRate: 48000,
    channels: 2
  }
];

export async function initMediasoup() {
  worker = await mediasoup.createWorker({
    rtcMinPort: 40000,
    rtcMaxPort: 49999
  });

  worker.on("died", () => {
    console.error("💥 mediasoup worker died");
    process.exit(1);
  });

  router = await worker.createRouter({ mediaCodecs });

  console.log("✅ Mediasoup initialized");
}

export function getRouter() {
  return router;
}
