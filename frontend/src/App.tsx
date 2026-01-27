import { useEffect, useRef, useState, useCallback } from "react";
import { AnimatePresence, motion } from "framer-motion";

type Language = "en" | "hi" | "te";

type WSMessage = {
  type?: "final";
  sourceText?: string;
  translatedText?: string;
  audio?: string;
  sourceLang?: string;
  targetLang?: string;
  confidence?: number;
  processingTime?: number;
};

const LANGUAGES: Record<Language, { name: string; flag: string; script: string }> = {
  en: { name: "English", flag: "��", script: "Latin" },
  hi: { name: "Hindi", flag: "🇮🇳", script: "हिन्दी" },
  te: { name: "Telugu", flag: "🇮🇳", script: "తెలుగు" },
};

const WS_URL = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;
const SAMPLE_RATE = 16000;
const CHUNK_SIZE = 3200; // ~200ms chunks
const MAX_AUDIO_QUEUE = 10; // Prevent queue overflow
const RECONNECT_DELAY = 3000; // 3 seconds

export default function App() {
  const wsRef = useRef<WebSocket | null>(null);

  /* ================= AUDIO ================= */
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const workletRef = useRef<AudioWorkletNode | null>(null);

  const audioQueueRef = useRef<string[]>([]);
  const isPlayingRef = useRef(false);

  const [audioUnlocked, setAudioUnlocked] = useState(false);
  const [micLevel, setMicLevel] = useState(0);

  /* ================= UI STATE ================= */
  const [room, setRoom] = useState("demo");
  const [speakLang, setSpeakLang] = useState<Language>("en");
  const [inCall, setInCall] = useState(false);

  const [recognized, setRecognized] = useState("");
  const [translated, setTranslated] = useState("");
  const [targetLang, setTargetLang] = useState<Language | "">("");
  const [processingTime, setProcessingTime] = useState(0);
  const [confidence, setConfidence] = useState(0);

  const [status, setStatus] = useState<
    "idle" | "connecting" | "listening" | "processing" | "speaking"
  >("idle");

  const [error, setError] = useState("");

  const [history, setHistory] = useState<
    Array<{ source: string; translated: string; time: string; sourceLang: string; targetLang: string }>
  >([]);

  const [isReconnecting, setIsReconnecting] = useState(false);
  const reconnectTimeoutRef = useRef<number | null>(null);

  /* ================= TIMER ================= */
  const [seconds, setSeconds] = useState(0);

  useEffect(() => {
    if (!inCall) return;
    const t = setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => clearInterval(t);
  }, [inCall]);

  const formatTime = () =>
    `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;

  /* ================= AUDIO CONTEXT ================= */
  const ensureAudioContext = useCallback(async () => {
    try {
      if (!audioCtxRef.current || audioCtxRef.current.state === "closed") {
        audioCtxRef.current = new AudioContext({ sampleRate: SAMPLE_RATE });
      }
      await audioCtxRef.current.resume();
      setAudioUnlocked(true);
      setError("");
    } catch (e) {
      setError("Failed to initialize audio. Please try again.");
      console.error("Audio context error:", e);
    }
  }, []);

  /* ================= TTS PLAYBACK ================= */
  const playNext = useCallback(async function playNextInner() {
    if (!audioUnlocked || isPlayingRef.current) return;
    const base64 = audioQueueRef.current.shift();
    if (!base64) return;

    isPlayingRef.current = true;
    setStatus("speaking");

    try {
      const ctx = audioCtxRef.current!;
      const bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
      const buffer = await ctx.decodeAudioData(bytes.buffer.slice(0));
      const src = ctx.createBufferSource();
      src.buffer = buffer;
      src.connect(ctx.destination);
      src.onended = () => {
        isPlayingRef.current = false;
        setStatus("listening");
        playNextInner();
      };
      src.start();
    } catch (e) {
      console.error("Audio playback error:", e);
      isPlayingRef.current = false;
      setStatus("listening");
    }
  }, [audioUnlocked]);

  const enqueueAudio = useCallback(
    (b64: string) => {
      // Prevent queue overflow
      if (audioQueueRef.current.length >= MAX_AUDIO_QUEUE) {
        console.warn("Audio queue full, dropping oldest");
        audioQueueRef.current.shift();
      }
      audioQueueRef.current.push(b64);
      playNext();
    },
    [playNext]
  );

  /* ================= STOP CALL ================= */
  const stopCall = useCallback(() => {
    // Clear reconnection timeout
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    wsRef.current?.close();
    wsRef.current = null;

    mediaStreamRef.current?.getTracks().forEach((t) => t.stop());
    mediaStreamRef.current = null;

    workletRef.current?.disconnect();
    workletRef.current = null;

    audioQueueRef.current = [];
    isPlayingRef.current = false;

    if (audioCtxRef.current && audioCtxRef.current.state !== "closed") {
      audioCtxRef.current.close().catch(() => { });
      audioCtxRef.current = null;
    }

    setInCall(false);
    setStatus("idle");
    setMicLevel(0);
    setIsReconnecting(false);
  }, []);

  /* ================= START CALL ================= */
  const startCall = async () => {
    if (inCall || !audioUnlocked) return;

    setError("");
    setSeconds(0);
    setRecognized("");
    setTranslated("");
    setHistory([]);
    setStatus("connecting");

    try {
      await ensureAudioContext();
      const ws = new WebSocket(WS_URL);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      ws.onopen = async () => {
        ws.send(JSON.stringify({ room, lang: speakLang }));

        try {
          const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
              echoCancellation: true,
              noiseSuppression: true,
              autoGainControl: true,
              sampleRate: SAMPLE_RATE,
            },
          });

          mediaStreamRef.current = stream;
          const ctx = audioCtxRef.current!;

          analyserRef.current = ctx.createAnalyser();
          analyserRef.current.fftSize = 256;

          const mic = ctx.createMediaStreamSource(stream);
          mic.connect(analyserRef.current);

          const data = new Uint8Array(analyserRef.current.frequencyBinCount);
          const loop = () => {
            analyserRef.current?.getByteFrequencyData(data);
            setMicLevel(data.reduce((a, b) => a + b, 0) / data.length);
            requestAnimationFrame(loop);
          };
          loop();

          await ctx.audioWorklet.addModule("/audio-worklet.js");
          const worklet = new AudioWorkletNode(ctx, "pcm-processor");
          workletRef.current = worklet;

          let buffers: Float32Array[] = [];
          let size = 0;

          worklet.port.onmessage = (e) => {
            if (!(e.data instanceof Float32Array)) return;
            if (ws.readyState !== WebSocket.OPEN) return;

            buffers.push(e.data);
            size += e.data.length;

            if (size >= CHUNK_SIZE) {
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
          setStatus("listening");
        } catch (e) {
          console.error("Mic error:", e);
          setError("Failed to access microphone. Please allow microphone access.");
          ws.close();
        }
      };

      ws.onmessage = async (e) => {
        const ctx = audioCtxRef.current;
        if (!ctx) return;

        if (e.data instanceof ArrayBuffer) {
          try {
            const pcm = new Float32Array(e.data);
            const buffer = ctx.createBuffer(1, pcm.length, SAMPLE_RATE);
            buffer.copyToChannel(pcm, 0);
            const src = ctx.createBufferSource();
            src.buffer = buffer;
            src.connect(ctx.destination);
            src.start();
          } catch (err) {
            console.error("PCM playback error:", err);
          }
          return;
        }

        try {
          const msg: WSMessage = JSON.parse(e.data);
          if (msg.type !== "final") return;

          setStatus("processing");
          if (msg.sourceText) setRecognized(msg.sourceText);
          if (msg.translatedText) setTranslated(msg.translatedText);

          setHistory((prev) => [
            ...prev.slice(-9),
            {
              source: msg.sourceText || "",
              translated: msg.translatedText || "",
              time: new Date().toLocaleTimeString(),
              sourceLang: msg.sourceLang || "",
              targetLang: msg.targetLang || "",
            },
          ]);

          if (msg.targetLang) setTargetLang(msg.targetLang as Language);
          if (msg.confidence) setConfidence(Math.round(msg.confidence * 100));
          if (msg.processingTime) setProcessingTime(Math.round(msg.processingTime * 1000));
          if (msg.audio) enqueueAudio(msg.audio);
          else setStatus("listening");
        } catch (err) {
          console.error("Message parse error:", err);
        }
      };

      ws.onerror = () => {
        if (!isReconnecting) {
          setError("Connection lost. Reconnecting...");
          setIsReconnecting(true);

          // Attempt reconnection
          reconnectTimeoutRef.current = window.setTimeout(() => {
            console.log("Attempting to reconnect...");
            startCall();
          }, RECONNECT_DELAY);
        }
      };

      ws.onclose = () => {
        if (inCall && !isReconnecting) {
          setError("Connection closed. Click 'Start Call' to reconnect.");
          stopCall();
        }
      };
    } catch (e) {
      console.error("Start call error:", e);
      setError("Failed to start call. Please try again.");
      setStatus("idle");
    }
  };

  useEffect(() => () => stopCall(), [stopCall]);

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
          <h2 style={styles.title}>🌐 Live Call Translator</h2>
          <p style={styles.subtitle}>English • Hindi • Telugu</p>

          {error && (
            <div style={styles.error}>
              ⚠️ {error}
              {isReconnecting && <div style={styles.reconnecting}>Reconnecting...</div>}
            </div>
          )}

          {!inCall && (
            <>
              <label style={styles.label}>Room ID</label>
              <input
                value={room}
                onChange={(e) => setRoom(e.target.value)}
                placeholder="Enter room name"
                style={styles.input}
              />

              <label style={styles.label}>You Speak</label>
              <div style={styles.row}>
                {(["en", "hi", "te"] as Language[]).map((l) => (
                  <button key={l} style={btn(speakLang === l)} onClick={() => setSpeakLang(l)}>
                    {LANGUAGES[l].flag} {LANGUAGES[l].name}
                  </button>
                ))}
              </div>
              <p style={styles.hint}>
                Each participant sets the language they speak. The system auto-translates to the other languages (all 6
                pairs).
              </p>

              <button style={styles.secondary} onClick={ensureAudioContext}>
                🔊 Enable Audio
              </button>
              <button
                style={{ ...styles.primary, opacity: audioUnlocked ? 1 : 0.5 }}
                onClick={startCall}
                disabled={!audioUnlocked}
              >
                📞 Start Call
              </button>
            </>
          )}

          {inCall && (
            <>
              <div style={styles.timer}>{formatTime()}</div>
              <motion.div
                style={{
                  ...styles.mic,
                  background:
                    status === "speaking" ? "#f59e0b" : status === "processing" ? "#3b82f6" : "#22c55e",
                }}
                animate={{ scale: 1 + micLevel / 300 }}
              />
              <p style={styles.status}>
                {status === "listening" && "🎤 Listening…"}
                {status === "processing" && "⚙️ Translating…"}
                {status === "speaking" && "🔊 Speaking…"}
              </p>

              {(confidence > 0 || processingTime > 0) && (
                <div style={styles.metrics}>
                  {confidence > 0 && <span>Confidence: {confidence}%</span>}
                  {processingTime > 0 && <span>Latency: {processingTime}ms</span>}
                </div>
              )}

              <div style={styles.transcript}>
                <div style={styles.transcriptSection}>
                  <b>📝 Recognized</b>
                  <p style={styles.transcriptText}>{recognized || "..."}</p>
                </div>
                <hr style={styles.divider} />
                <div style={styles.transcriptSection}>
                  <b>🔊 Translated</b>
                  <p style={styles.transcriptText}>{translated || "..."}</p>
                </div>
                {targetLang && (
                  <p style={styles.hint}>Target: {LANGUAGES[targetLang as Language]?.name || targetLang}</p>
                )}
              </div>

              {history.length > 0 && (
                <div style={styles.historyContainer}>
                  <b style={styles.historyTitle}>📜 History</b>
                  <div style={styles.historyScroll}>
                    {history.map((item, i) => (
                      <div key={i} style={styles.historyItem}>
                        <span style={styles.historyTime}>{item.time}</span>
                        <span style={styles.historyLang}>
                          {item.sourceLang.toUpperCase()} → {item.targetLang.toUpperCase()}
                        </span>
                        <p style={styles.historySource}>{item.source}</p>
                        <p style={styles.historyTranslated}>→ {item.translated}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <button style={styles.danger} onClick={stopCall}>
                🛑 End Call
              </button>
            </>
          )}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  page: {
    minHeight: "100vh",
    background:
      "radial-gradient(circle at 20% 20%, #0ea5e9 0, transparent 25%), radial-gradient(circle at 80% 0%, #22c55e 0, transparent 25%), #0b1224",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    padding: "24px",
    color: "#e5e7eb",
    fontFamily: "Inter, system-ui, -apple-system, sans-serif",
  },
  card: {
    width: "min(960px, 100%)",
    background: "rgba(15, 23, 42, 0.8)",
    border: "1px solid rgba(148, 163, 184, 0.2)",
    borderRadius: "24px",
    padding: "24px",
    boxShadow: "0 25px 50px -12px rgba(15, 23, 42, 0.5)",
    backdropFilter: "blur(12px)",
  },
  title: { fontSize: "28px", margin: "0 0 8px", fontWeight: 800 },
  subtitle: { marginTop: 0, color: "#94a3b8" },
  label: { display: "block", marginTop: 12, marginBottom: 6, fontWeight: 600 },
  input: {
    width: "100%",
    padding: "12px",
    borderRadius: "12px",
    border: "1px solid rgba(148, 163, 184, 0.3)",
    background: "rgba(15, 23, 42, 0.6)",
    color: "white",
    outline: "none",
  },
  row: { display: "flex", gap: 12, margin: "12px 0" },
  primary: {
    width: "100%",
    background: "linear-gradient(90deg, #2563eb, #22c55e)",
    border: "none",
    color: "white",
    padding: "14px",
    borderRadius: "12px",
    fontWeight: 700,
    cursor: "pointer",
    marginTop: 12,
  },
  secondary: {
    width: "100%",
    background: "rgba(255,255,255,0.08)",
    border: "1px solid rgba(148, 163, 184, 0.2)",
    color: "white",
    padding: "12px",
    borderRadius: "12px",
    fontWeight: 600,
    cursor: "pointer",
    marginTop: 8,
  },
  danger: {
    width: "100%",
    background: "#ef4444",
    border: "none",
    color: "white",
    padding: "12px",
    borderRadius: "12px",
    fontWeight: 700,
    cursor: "pointer",
    marginTop: 16,
  },
  hint: { color: "#94a3b8", fontSize: "14px", marginTop: 4 },
  error: {
    background: "rgba(239, 68, 68, 0.1)",
    color: "#fecdd3",
    padding: "12px",
    borderRadius: "12px",
    border: "1px solid rgba(239, 68, 68, 0.4)",
    marginBottom: 12,
  },
  reconnecting: {
    marginTop: 8,
    fontSize: 14,
    color: "#fbbf24",
    fontWeight: 600,
    animation: "pulse 2s infinite",
  },
  mic: {
    width: 96,
    height: 96,
    borderRadius: "50%",
    margin: "16px auto 8px",
    boxShadow: "0 10px 40px rgba(37, 99, 235, 0.4)",
  },
  status: { textAlign: "center", fontWeight: 700 },
  metrics: {
    display: "flex",
    gap: 12,
    justifyContent: "center",
    color: "#cbd5e1",
    fontSize: 14,
  },
  transcript: {
    marginTop: 16,
    padding: 12,
    border: "1px solid rgba(148, 163, 184, 0.2)",
    borderRadius: 12,
    background: "rgba(15, 23, 42, 0.6)",
  },
  transcriptSection: { marginBottom: 8 },
  transcriptText: { margin: "6px 0", lineHeight: 1.6 },
  divider: { border: "none", borderTop: "1px solid rgba(148, 163, 184, 0.2)" },
  historyContainer: {
    marginTop: 16,
    border: "1px solid rgba(148, 163, 184, 0.2)",
    borderRadius: 12,
    padding: 12,
    background: "rgba(15, 23, 42, 0.6)",
  },
  historyTitle: { margin: "0 0 8px", display: "block" },
  historyScroll: { maxHeight: 220, overflowY: "auto", display: "grid", gap: 8 },
  historyItem: {
    padding: 10,
    borderRadius: 10,
    background: "rgba(255,255,255,0.03)",
    border: "1px solid rgba(148, 163, 184, 0.15)",
  },
  historyTime: { color: "#94a3b8", fontSize: 12 },
  historyLang: { color: "#cbd5e1", fontWeight: 700, marginLeft: 6 },
  historySource: { margin: "4px 0", color: "#e2e8f0" },
  historyTranslated: { margin: 0, color: "#a5b4fc" },
  timer: { textAlign: "center", fontWeight: 700, color: "#cbd5e1" },
};

const btn = (active: boolean): React.CSSProperties => ({
  flex: 1,
  padding: "12px 10px",
  borderRadius: 12,
  border: active ? "1px solid #22c55e" : "1px solid rgba(148, 163, 184, 0.3)",
  background: active ? "rgba(34,197,94,0.15)" : "rgba(255,255,255,0.04)",
  color: "white",
  cursor: "pointer",
  fontWeight: 700,
});
