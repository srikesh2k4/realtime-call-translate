import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";

/* ================= TYPES ================= */

type Language = "en" | "hi";

type WSMessage = {
  type?: "final";
  text?: string;
  translated?: string;
  audio?: string;
};

/* ================= CONFIG ================= */

// Use same-origin WS (Vite proxy handles LAN + HTTPS)
const WS_URL =
  `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;

/* ================= APP ================= */

export default function App() {
  const wsRef = useRef<WebSocket | null>(null);

  /* ================= AUDIO ================= */

  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);

  const audioQueueRef = useRef<string[]>([]);
  const isPlayingRef = useRef(false);

  const [audioUnlocked, setAudioUnlocked] = useState(false);
  const [micLevel, setMicLevel] = useState(0);

  /* ================= UI STATE ================= */

  const [room, setRoom] = useState("demo");
  const [lang, setLang] = useState<Language>("en");
  const [inCall, setInCall] = useState(false);

  const [recognized, setRecognized] = useState("");
  const [translated, setTranslated] = useState("");

  const [status, setStatus] = useState<
    "idle" | "listening" | "processing" | "speaking"
  >("idle");

  /* ================= TIMER ================= */

  const [seconds, setSeconds] = useState(0);

  useEffect(() => {
    if (!inCall) return;
    const t = setInterval(() => setSeconds(s => s + 1), 1000);
    return () => clearInterval(t);
  }, [inCall]);

  const formatTime = () =>
    `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(
      seconds % 60
    ).padStart(2, "0")}`;

  /* ================= AUDIO CONTEXT ================= */

  const ensureAudioContext = async () => {
    if (!audioCtxRef.current || audioCtxRef.current.state === "closed") {
      audioCtxRef.current = new AudioContext({ sampleRate: 16000 });
    }
    await audioCtxRef.current.resume();
    setAudioUnlocked(true);
  };

  /* ================= TTS PLAYBACK ================= */

  const playNext = async () => {
    if (!audioUnlocked || isPlayingRef.current) return;

    const base64 = audioQueueRef.current.shift();
    if (!base64) return;

    isPlayingRef.current = true;
    setStatus("speaking");

    try {
      const ctx = audioCtxRef.current!;
      const bytes = Uint8Array.from(atob(base64), c => c.charCodeAt(0));
      const buffer = await ctx.decodeAudioData(bytes.buffer);

      const src = ctx.createBufferSource();
      src.buffer = buffer;
      src.connect(ctx.destination);

      src.onended = () => {
        isPlayingRef.current = false;
        setStatus("idle");
        playNext();
      };

      src.start();
    } catch {
      isPlayingRef.current = false;
    }
  };

  const enqueueAudio = (b64: string) => {
    audioQueueRef.current.push(b64);
    playNext();
  };

  /* ================= START CALL ================= */

  const startCall = async () => {
    if (inCall || !audioUnlocked) return;

    setSeconds(0);
    setRecognized("");
    setTranslated("");
    setStatus("listening");

    await ensureAudioContext();

    const ws = new WebSocket(WS_URL);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    ws.onopen = async () => {
      ws.send(JSON.stringify({ room, lang }));

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const ctx = audioCtxRef.current!;

      analyserRef.current = ctx.createAnalyser();
      analyserRef.current.fftSize = 256;

      const mic = ctx.createMediaStreamSource(stream);
      mic.connect(analyserRef.current);

      /* ---- mic level animation ---- */
      const data = new Uint8Array(analyserRef.current.frequencyBinCount);
      const loop = () => {
        analyserRef.current?.getByteFrequencyData(data);
        setMicLevel(data.reduce((a, b) => a + b, 0) / data.length);
        requestAnimationFrame(loop);
      };
      loop();

      /* ---- audio worklet ---- */
      await ctx.audioWorklet.addModule("/audio-worklet.js");
      const worklet = new AudioWorkletNode(ctx, "pcm-processor");

      let buffers: Float32Array[] = [];
      let size = 0;
      const TARGET = 5120;

      worklet.port.onmessage = e => {
        if (!(e.data instanceof Float32Array)) return;
        if (ws.readyState !== WebSocket.OPEN) return;

        buffers.push(e.data);
        size += e.data.length;

        if (size >= TARGET) {
          const merged = new Float32Array(size);
          let offset = 0;
          for (const b of buffers) {
            merged.set(b, offset);
            offset += b.length;
          }
          ws.send(merged.buffer);
          buffers = [];
          size = 0;
        }
      };

      mic.connect(worklet);
      setInCall(true);
    };

    /* ================= RECEIVE ================= */

    ws.onmessage = async e => {
      const ctx = audioCtxRef.current!;

      // 🔵 SAME LANGUAGE → RAW PCM AUDIO
      if (e.data instanceof ArrayBuffer) {
        const pcm = new Float32Array(e.data);
        const buffer = ctx.createBuffer(1, pcm.length, 16000);
        buffer.copyToChannel(pcm, 0);

        const src = ctx.createBufferSource();
        src.buffer = buffer;
        src.connect(ctx.destination);
        src.start();
        return;
      }

      // 🟢 DIFFERENT LANGUAGE → TRANSLATED JSON
      const msg: WSMessage = JSON.parse(e.data);
      if (msg.type !== "final") return;

      setStatus("processing");
      msg.text && setRecognized(msg.text);
      msg.translated && setTranslated(msg.translated);
      msg.audio && enqueueAudio(msg.audio);
    };

    ws.onerror = stopCall;
    ws.onclose = stopCall;
  };

  /* ================= STOP ================= */

  const stopCall = () => {
    wsRef.current?.close();
    wsRef.current = null;

    audioQueueRef.current = [];
    isPlayingRef.current = false;

    audioCtxRef.current?.close();
    audioCtxRef.current = null;

    setInCall(false);
    setStatus("idle");
    setSeconds(0);
  };

  /* ================= UI ================= */

  return (
    <div style={styles.page}>
      <AnimatePresence mode="wait">
        <motion.div
          key={inCall ? "call" : "home"}
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -30 }}
          transition={{ duration: 0.3 }}
          style={styles.card}
        >
          <h2 style={styles.title}>Two-Way Live Translator</h2>

          {!inCall && (
            <>
              <input
                value={room}
                onChange={e => setRoom(e.target.value)}
                placeholder="Room ID"
                style={styles.input}
              />

              <div style={styles.row}>
                <button style={btn(lang === "en")} onClick={() => setLang("en")}>
                  Hear English
                </button>
                <button style={btn(lang === "hi")} onClick={() => setLang("hi")}>
                  Hear Hindi
                </button>
              </div>

              <p style={styles.hint}>
                You will hear translations in{" "}
                <b>{lang === "en" ? "English" : "Hindi"}</b>
              </p>

              <button style={styles.secondary} onClick={ensureAudioContext}>
                Enable Audio
              </button>

              <button
                style={styles.primary}
                onClick={startCall}
                disabled={!audioUnlocked}
              >
                Start Call
              </button>
            </>
          )}

          {inCall && (
            <>
              <div style={styles.timer}>{formatTime()}</div>

              <motion.div
                style={styles.mic}
                animate={{ scale: 1 + micLevel / 300 }}
              />

              <p style={styles.status}>
                {status === "listening" && "Listening…"}
                {status === "processing" && "Translating…"}
                {status === "speaking" && "Speaking…"}
              </p>

              <div style={styles.transcript}>
                <b>Recognized</b>
                <p>{recognized}</p>
                <hr />
                <b>Translated (You hear)</b>
                <p>{translated}</p>
              </div>

              <button style={styles.end} onClick={stopCall}>
                End Call
              </button>
            </>
          )}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}

/* ================= STYLES ================= */

const btn = (active: boolean) => ({
  flex: 1,
  padding: 12,
  borderRadius: 12,
  border: "none",
  background: active ? "#22c55e" : "#1e293b",
  color: "white",
});

const styles: any = {
  page: {
    minHeight: "100vh",
    background: "#020617",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    padding: 16,
  },
  card: {
    width: "100%",
    maxWidth: 420,
    background: "#020617",
    borderRadius: 24,
    padding: 24,
    color: "white",
    boxShadow: "0 20px 40px rgba(0,0,0,.6)",
  },
  title: { textAlign: "center", marginBottom: 12 },
  input: {
    width: "100%",
    padding: 12,
    borderRadius: 12,
    border: "none",
    marginBottom: 12,
  },
  row: { display: "flex", gap: 8, marginBottom: 8 },
  hint: {
    textAlign: "center",
    opacity: 0.7,
    marginBottom: 12,
    fontSize: 13,
  },
  secondary: {
    width: "100%",
    padding: 12,
    background: "#0ea5e9",
    borderRadius: 12,
    border: "none",
    marginBottom: 8,
  },
  primary: {
    width: "100%",
    padding: 14,
    background: "#22c55e",
    borderRadius: 14,
    border: "none",
    fontWeight: 700,
  },
  timer: { textAlign: "center", fontSize: 18, marginBottom: 8 },
  mic: {
    width: 60,
    height: 60,
    borderRadius: "50%",
    background: "#22c55e",
    margin: "12px auto",
  },
  status: { textAlign: "center", opacity: 0.8 },
  transcript: {
    background: "#020617",
    padding: 12,
    borderRadius: 12,
    marginBottom: 12,
  },
  end: {
    width: "100%",
    padding: 14,
    background: "#ef4444",
    borderRadius: 14,
    border: "none",
    fontWeight: 700,
  },
};
