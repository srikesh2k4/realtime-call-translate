import { useEffect, useRef, useState, useCallback } from "react";
import { AnimatePresence, motion } from "framer-motion";

/* ================= TYPES ================= */

type Language = "en" | "hi";

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

/* ================= CONFIG ================= */

// Use same-origin WS (Vite proxy handles LAN + HTTPS)
const WS_URL =
  `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;

// Audio configuration
const SAMPLE_RATE = 16000;
const CHUNK_SIZE = 4096; // Smaller chunks for lower latency

/* ================= APP ================= */

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
  const [lang, setLang] = useState<Language>("en");
  const [inCall, setInCall] = useState(false);

  const [recognized, setRecognized] = useState("");
  const [translated, setTranslated] = useState("");
  const [processingTime, setProcessingTime] = useState(0);
  const [confidence, setConfidence] = useState(0);

  const [status, setStatus] = useState<
    "idle" | "listening" | "processing" | "speaking"
  >("idle");

  /* ================= CONVERSATION HISTORY ================= */
  
  const [history, setHistory] = useState<Array<{
    source: string;
    translated: string;
    time: string;
    lang: string;
  }>>([]);

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

  const ensureAudioContext = useCallback(async () => {
    if (!audioCtxRef.current || audioCtxRef.current.state === "closed") {
      audioCtxRef.current = new AudioContext({ sampleRate: SAMPLE_RATE });
    }
    await audioCtxRef.current.resume();
    setAudioUnlocked(true);
  }, []);

  /* ================= TTS PLAYBACK ================= */

  const playNext = useCallback(async () => {
    if (!audioUnlocked || isPlayingRef.current) return;

    const base64 = audioQueueRef.current.shift();
    if (!base64) return;

    isPlayingRef.current = true;
    setStatus("speaking");

    try {
      const ctx = audioCtxRef.current!;
      const bytes = Uint8Array.from(atob(base64), c => c.charCodeAt(0));
      const buffer = await ctx.decodeAudioData(bytes.buffer.slice(0));

      const src = ctx.createBufferSource();
      src.buffer = buffer;
      src.connect(ctx.destination);

      src.onended = () => {
        isPlayingRef.current = false;
        setStatus("listening");
        playNext();
      };

      src.start();
    } catch (e) {
      console.error("Audio playback error:", e);
      isPlayingRef.current = false;
      setStatus("listening");
    }
  }, [audioUnlocked]);

  const enqueueAudio = useCallback((b64: string) => {
    audioQueueRef.current.push(b64);
    playNext();
  }, [playNext]);

  /* ================= START CALL ================= */

  const startCall = async () => {
    if (inCall || !audioUnlocked) return;

    setSeconds(0);
    setRecognized("");
    setTranslated("");
    setHistory([]);
    setStatus("listening");

    await ensureAudioContext();

    const ws = new WebSocket(WS_URL);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    ws.onopen = async () => {
      ws.send(JSON.stringify({ room, lang }));

      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: SAMPLE_RATE,
        } 
      });
      mediaStreamRef.current = stream;
      
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

      /* ---- audio worklet with noise suppression ---- */
      await ctx.audioWorklet.addModule("/audio-worklet.js");
      const worklet = new AudioWorkletNode(ctx, "pcm-processor");
      workletRef.current = worklet;

      let buffers: Float32Array[] = [];
      let size = 0;

      worklet.port.onmessage = e => {
        if (!(e.data instanceof Float32Array)) return;
        if (ws.readyState !== WebSocket.OPEN) return;

        buffers.push(e.data);
        size += e.data.length;

        // Send smaller chunks more frequently for lower latency
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
    };

    /* ================= RECEIVE ================= */

    ws.onmessage = async e => {
      const ctx = audioCtxRef.current!;

      // 🔵 SAME LANGUAGE → RAW PCM AUDIO
      if (e.data instanceof ArrayBuffer) {
        const pcm = new Float32Array(e.data);
        const buffer = ctx.createBuffer(1, pcm.length, SAMPLE_RATE);
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
      
      if (msg.sourceText) {
        setRecognized(msg.sourceText);
      }
      if (msg.translatedText) {
        setTranslated(msg.translatedText);
        
        // Add to history
        setHistory(prev => [...prev.slice(-9), {
          source: msg.sourceText || "",
          translated: msg.translatedText || "",
          time: new Date().toLocaleTimeString(),
          lang: msg.sourceLang || "",
        }]);
      }
      if (msg.confidence) {
        setConfidence(Math.round(msg.confidence * 100));
      }
      if (msg.processingTime) {
        setProcessingTime(Math.round(msg.processingTime * 1000));
      }
      if (msg.audio) {
        enqueueAudio(msg.audio);
      }
    };

    ws.onerror = stopCall;
    ws.onclose = stopCall;
  };

  /* ================= STOP ================= */

  const stopCall = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;

    // Stop media stream
    mediaStreamRef.current?.getTracks().forEach(track => track.stop());
    mediaStreamRef.current = null;

    // Disconnect worklet
    workletRef.current?.disconnect();
    workletRef.current = null;

    audioQueueRef.current = [];
    isPlayingRef.current = false;

    audioCtxRef.current?.close();
    audioCtxRef.current = null;

    setInCall(false);
    setStatus("idle");
    setSeconds(0);
  }, []);

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
          <h2 style={styles.title}>🌐 Two-Way Live Translator</h2>

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
                  🇬🇧 Hear English
                </button>
                <button style={btn(lang === "hi")} onClick={() => setLang("hi")}>
                  🇮🇳 Hear Hindi
                </button>
              </div>

              <p style={styles.hint}>
                You will hear translations in{" "}
                <b>{lang === "en" ? "English" : "Hindi"}</b>
              </p>

              <button style={styles.secondary} onClick={ensureAudioContext}>
                🔊 Enable Audio
              </button>

              <button
                style={{...styles.primary, opacity: audioUnlocked ? 1 : 0.5}}
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
                  background: status === "speaking" ? "#f59e0b" : 
                              status === "processing" ? "#3b82f6" : "#22c55e"
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
                  <b>🔊 Translated (You hear)</b>
                  <p style={styles.transcriptText}>{translated || "..."}</p>
                </div>
              </div>

              {history.length > 0 && (
                <div style={styles.historyContainer}>
                  <b style={styles.historyTitle}>📜 History</b>
                  <div style={styles.historyScroll}>
                    {history.map((item, i) => (
                      <div key={i} style={styles.historyItem}>
                        <span style={styles.historyTime}>{item.time}</span>
                        <span style={styles.historyLang}>{item.lang.toUpperCase()}</span>
                        <p style={styles.historySource}>{item.source}</p>
                        <p style={styles.historyTranslated}>→ {item.translated}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <button style={styles.end} onClick={stopCall}>
                📵 End Call
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
  cursor: "pointer",
  transition: "all 0.2s",
});

const styles: Record<string, React.CSSProperties> = {
  page: {
    minHeight: "100vh",
    background: "linear-gradient(135deg, #020617 0%, #0f172a 100%)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    padding: 16,
  },
  card: {
    width: "100%",
    maxWidth: 440,
    background: "rgba(2, 6, 23, 0.95)",
    borderRadius: 24,
    padding: 24,
    color: "white",
    boxShadow: "0 25px 50px rgba(0,0,0,.7)",
    border: "1px solid rgba(255,255,255,0.1)",
  },
  title: { 
    textAlign: "center", 
    marginBottom: 16,
    fontSize: 22,
    fontWeight: 700,
  },
  input: {
    width: "100%",
    padding: 14,
    borderRadius: 12,
    border: "1px solid rgba(255,255,255,0.1)",
    marginBottom: 12,
    background: "#0f172a",
    color: "white",
    fontSize: 16,
  },
  row: { display: "flex", gap: 8, marginBottom: 12 },
  hint: {
    textAlign: "center",
    opacity: 0.7,
    marginBottom: 16,
    fontSize: 14,
  },
  secondary: {
    width: "100%",
    padding: 14,
    background: "#0ea5e9",
    borderRadius: 12,
    border: "none",
    marginBottom: 10,
    cursor: "pointer",
    fontWeight: 600,
    color: "white",
  },
  primary: {
    width: "100%",
    padding: 16,
    background: "linear-gradient(135deg, #22c55e 0%, #16a34a 100%)",
    borderRadius: 14,
    border: "none",
    fontWeight: 700,
    cursor: "pointer",
    fontSize: 16,
    color: "white",
  },
  timer: { 
    textAlign: "center", 
    fontSize: 20, 
    marginBottom: 8,
    fontFamily: "monospace",
    fontWeight: 600,
  },
  mic: {
    width: 70,
    height: 70,
    borderRadius: "50%",
    background: "#22c55e",
    margin: "16px auto",
    boxShadow: "0 0 30px rgba(34, 197, 94, 0.4)",
  },
  status: { 
    textAlign: "center", 
    opacity: 0.9,
    fontSize: 15,
    marginBottom: 12,
  },
  metrics: {
    display: "flex",
    justifyContent: "center",
    gap: 20,
    fontSize: 12,
    opacity: 0.7,
    marginBottom: 12,
  },
  transcript: {
    background: "rgba(15, 23, 42, 0.8)",
    padding: 16,
    borderRadius: 16,
    marginBottom: 16,
    border: "1px solid rgba(255,255,255,0.1)",
  },
  transcriptSection: {
    marginBottom: 8,
  },
  transcriptText: {
    margin: "8px 0 0 0",
    fontSize: 15,
    lineHeight: 1.5,
    minHeight: 24,
  },
  divider: {
    border: "none",
    borderTop: "1px solid rgba(255,255,255,0.1)",
    margin: "12px 0",
  },
  historyContainer: {
    background: "rgba(15, 23, 42, 0.6)",
    padding: 12,
    borderRadius: 12,
    marginBottom: 16,
    maxHeight: 200,
    overflow: "hidden",
  },
  historyTitle: {
    fontSize: 13,
    opacity: 0.8,
  },
  historyScroll: {
    maxHeight: 160,
    overflowY: "auto" as const,
    marginTop: 8,
  },
  historyItem: {
    padding: "8px 0",
    borderBottom: "1px solid rgba(255,255,255,0.05)",
    fontSize: 12,
  },
  historyTime: {
    opacity: 0.5,
    marginRight: 8,
  },
  historyLang: {
    background: "rgba(255,255,255,0.1)",
    padding: "2px 6px",
    borderRadius: 4,
    fontSize: 10,
  },
  historySource: {
    margin: "4px 0 2px 0",
    opacity: 0.7,
  },
  historyTranslated: {
    margin: 0,
    color: "#22c55e",
  },
  end: {
    width: "100%",
    padding: 16,
    background: "linear-gradient(135deg, #ef4444 0%, #dc2626 100%)",
    borderRadius: 14,
    border: "none",
    fontWeight: 700,
    cursor: "pointer",
    fontSize: 16,
    color: "white",
  },
};
