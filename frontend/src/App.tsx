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
  const [lang, setLang] = useState<Language>("en"); // language I want to hear
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

  /* ================= AUDIO UNLOCK ================= */

  const unlockAudio = async () => {
    if (!audioCtxRef.current) {
      audioCtxRef.current = new AudioContext();
    }
    if (audioCtxRef.current.state === "suspended") {
      await audioCtxRef.current.resume();
    }
    setAudioUnlocked(true);
    playNext();
  };

  /* ================= AUDIO PLAYBACK ================= */

  const playNext = async () => {
    if (!audioUnlocked) return;
    if (isPlayingRef.current) return;
    if (audioQueueRef.current.length === 0) return;

    const base64 = audioQueueRef.current.shift()!;
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
      playNext();
    }
  };

  const enqueueAudio = (b64: string) => {
    audioQueueRef.current.push(b64);
    playNext();
  };

  /* ================= START CALL ================= */

  const startCall = async () => {
    if (inCall) return;

    setSeconds(0);
    setRecognized("");
    setTranslated("");
    setStatus("listening");

    const ws = new WebSocket("ws://localhost:8000/ws");
    wsRef.current = ws;

    ws.onopen = async () => {
      ws.send(JSON.stringify({ room, lang }));

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      const ctx = new AudioContext({ sampleRate: 16000 });
      audioCtxRef.current = ctx;
      await ctx.resume();

      analyserRef.current = ctx.createAnalyser();
      analyserRef.current.fftSize = 256;

      const mic = ctx.createMediaStreamSource(stream);
      mic.connect(analyserRef.current);

      /* ---- mic level animation ---- */
      const data = new Uint8Array(analyserRef.current.frequencyBinCount);
      const loop = () => {
        analyserRef.current?.getByteFrequencyData(data);
        const avg = data.reduce((a, b) => a + b, 0) / data.length;
        setMicLevel(avg);
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

    ws.onmessage = e => {
      if (typeof e.data !== "string") return;

      const msg: WSMessage = JSON.parse(e.data);
      if (msg.type !== "final") return;

      setStatus("processing");
      if (msg.text) setRecognized(msg.text);
      if (msg.translated) setTranslated(msg.translated);
      if (msg.audio) enqueueAudio(msg.audio);
    };

    ws.onerror = stopCall;
    ws.onclose = stopCall;
  };

  /* ================= STOP ================= */

  const stopCall = () => {
    wsRef.current?.close();
    wsRef.current = null;
    audioQueueRef.current = [];
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
                You will hear translations in <b>{lang === "en" ? "English" : "Hindi"}</b>
              </p>

              <button style={styles.secondary} onClick={unlockAudio}>
                Enable Audio
              </button>

              <button style={styles.primary} onClick={startCall}>
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
  title: {
    textAlign: "center",
    marginBottom: 12,
  },
  input: {
    width: "100%",
    padding: 12,
    borderRadius: 12,
    border: "none",
    marginBottom: 12,
  },
  row: {
    display: "flex",
    gap: 8,
    marginBottom: 8,
  },
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
  timer: {
    textAlign: "center",
    fontSize: 18,
    marginBottom: 8,
  },
  mic: {
    width: 60,
    height: 60,
    borderRadius: "50%",
    background: "#22c55e",
    margin: "12px auto",
  },
  status: {
    textAlign: "center",
    opacity: 0.8,
  },
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
