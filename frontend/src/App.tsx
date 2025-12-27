import { useRef, useState } from "react";

type WSMessage = {
  type?: "final";
  text?: string;
  translated?: string;
  audio?: string;
};

export default function App() {
  const wsRef = useRef<WebSocket | null>(null);

  /* ================= AUDIO STATE ================= */

  const audioQueueRef = useRef<string[]>([]);
  const isPlayingRef = useRef(false);
  const audioCtxRef = useRef<AudioContext | null>(null);

  const [audioUnlocked, setAudioUnlocked] = useState(false);

  /* ================= UI STATE ================= */

  const [room, setRoom] = useState("demo");
  const [mode, setMode] = useState<0 | 1>(0);
  const [isSpeaker, setIsSpeaker] = useState(true);
  const [inCall, setInCall] = useState(false);

  const [text, setText] = useState("");
  const [translated, setTranslated] = useState("");

  /* ================= AUDIO UNLOCK ================= */

  const unlockAudio = async () => {
    if (!audioCtxRef.current) {
      audioCtxRef.current = new AudioContext();
    }

    if (audioCtxRef.current.state === "suspended") {
      await audioCtxRef.current.resume();
    }

    setAudioUnlocked(true);
    console.log("🔓 Audio unlocked");

    // 🔥 If audio already queued, start playing
    playNext();
  };

  /* ================= AUDIO QUEUE ================= */

  const playNext = async () => {
    if (!audioUnlocked) return;
    if (isPlayingRef.current) return;
    if (audioQueueRef.current.length === 0) return;

    const base64 = audioQueueRef.current.shift()!;
    isPlayingRef.current = true;

    try {
      const audioCtx = audioCtxRef.current!;
      const binary = Uint8Array.from(atob(base64), c => c.charCodeAt(0));
      const buffer = await audioCtx.decodeAudioData(binary.buffer);

      const source = audioCtx.createBufferSource();
      source.buffer = buffer;
      source.connect(audioCtx.destination);

      source.onended = () => {
        isPlayingRef.current = false;
        playNext();
      };

      source.start(0);
    } catch (e) {
      console.error("❌ WebAudio decode/play failed", e);
      isPlayingRef.current = false;
      playNext();
    }
  };

  const enqueueAudio = (base64: string) => {
    console.log("🔊 Audio enqueued", base64.length);
    audioQueueRef.current.push(base64);
    playNext();
  };

  /* ================= START CALL ================= */

  const startCall = async () => {
    if (inCall) return;

    const ws = new WebSocket("ws://localhost:8000/ws");
    wsRef.current = ws;

    ws.onopen = async () => {
      ws.send(JSON.stringify({ room, mode, isSpeaker }));

      if (!isSpeaker) {
        setInCall(true);
        return;
      }

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      const audioCtx = new AudioContext({ sampleRate: 16000 });
      audioCtxRef.current = audioCtx;
      await audioCtx.resume();

      await audioCtx.audioWorklet.addModule("/audio-worklet.js");

      const source = audioCtx.createMediaStreamSource(stream);
      const worklet = new AudioWorkletNode(audioCtx, "pcm-processor");

      const TARGET = 5120;
      let buffers: Float32Array[] = [];
      let size = 0;

      worklet.port.onmessage = (e) => {
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

      source.connect(worklet);
      setInCall(true);
    };

    ws.onmessage = (e) => {
      if (typeof e.data !== "string") return;

      const msg: WSMessage = JSON.parse(e.data);
      if (msg.type !== "final") return;

      if (msg.text) setText(msg.text);
      if (msg.translated) setTranslated(msg.translated);

      if (msg.audio) enqueueAudio(msg.audio);
    };

    ws.onerror = stopCall;
    ws.onclose = stopCall;
  };

  /* ================= STOP CALL ================= */

  const stopCall = () => {
    wsRef.current?.close();
    wsRef.current = null;

    audioQueueRef.current = [];
    isPlayingRef.current = false;

    audioCtxRef.current?.close();
    audioCtxRef.current = null;

    setInCall(false);
    setText("");
    setTranslated("");
  };

  /* ================= UI ================= */

  return (
    <div style={styles.page}>
      <div style={styles.card}>
        <h2>📞 Live Call Translation</h2>

        {!inCall && (
          <>
            <input
              value={room}
              onChange={(e) => setRoom(e.target.value)}
              placeholder="Room ID"
              style={styles.input}
            />

            <div style={{ display: "flex", gap: 8 }}>
              <button style={btn(mode === 0)} onClick={() => setMode(0)}>
                EN → HI
              </button>
              <button style={btn(mode === 1)} onClick={() => setMode(1)}>
                HI → EN
              </button>
            </div>

            <button
              style={styles.modeBtn}
              onClick={() => setIsSpeaker(!isSpeaker)}
            >
              {isSpeaker ? "🎤 Speaker" : "🔊 Listener"}
            </button>

            <button
              style={{ ...styles.callBtn, background: "#0ea5e9" }}
              onClick={unlockAudio}
            >
              🔓 Enable Audio
            </button>

            <button style={styles.callBtn} onClick={startCall}>
              📞 Start Call
            </button>
          </>
        )}

        {inCall && (
          <>
            <div style={styles.transcript}>
              <b>Recognized</b>
              <p>{text}</p>
              <hr />
              <b>Translated</b>
              <p>{translated}</p>
            </div>

            <button style={styles.endBtn} onClick={stopCall}>
              ⛔ End Call
            </button>
          </>
        )}
      </div>
    </div>
  );
}

/* ================= STYLES ================= */

const btn = (active: boolean) => ({
  flex: 1,
  padding: 10,
  borderRadius: 8,
  border: "none",
  background: active ? "#22c55e" : "#334155",
  color: "white",
});

const styles: any = {
  page: {
    height: "100vh",
    background: "#0f172a",
    color: "white",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
  },
  card: {
    width: 360,
    background: "#020617",
    borderRadius: 20,
    padding: 24,
  },
  input: {
    width: "100%",
    padding: 10,
    borderRadius: 8,
    marginBottom: 12,
    border: "none",
  },
  transcript: {
    background: "#020617",
    padding: 12,
    borderRadius: 8,
    marginBottom: 12,
  },
  callBtn: {
    width: "100%",
    padding: 14,
    borderRadius: 12,
    background: "#22c55e",
    border: "none",
    marginTop: 10,
  },
  endBtn: {
    width: "100%",
    padding: 14,
    borderRadius: 12,
    background: "#ef4444",
    border: "none",
  },
  modeBtn: {
    width: "100%",
    padding: 10,
    borderRadius: 8,
    background: "#334155",
    border: "none",
    marginTop: 8,
  },
};
