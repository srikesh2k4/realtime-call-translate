package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"io"
	"log"
	"math"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
)

/* ================= TYPES ================= */

type Client struct {
	conn   *websocket.Conn
	room   string
	lang   string
	send   chan []byte
	mlBusy int32
}

type JoinMessage struct {
	Room string `json:"room"`
	Lang string `json:"lang"`
}

type MLResponse struct {
	Type           string `json:"type"`
	SourceText     string `json:"sourceText"`
	TranslatedText string `json:"translatedText"`
	Audio          string `json:"audio"`
	SourceLang     string `json:"sourceLang"`
	TargetLang     string `json:"targetLang"`
}

/* ================= AUDIO ================= */

type AudioState struct {
	samples       []float32
	silenceFrames int       // consecutive low-energy frames
	lastSendTime  time.Time // when we last dispatched to ML
}

/* ================= GLOBALS ================= */

const (
	SAMPLE_RATE = 16000

	// Minimum audio before we'll even consider sending (2s — avoids short hallucination-prone clips)
	MIN_SAMPLES = SAMPLE_RATE * 2

	// Maximum audio buffer before forced send (8s — keeps utterances manageable)
	MAX_SAMPLES = SAMPLE_RATE * 8

	// How much silence (in frames) triggers end-of-utterance flush
	// Each frame from the frontend ≈ 6400 samples = 400ms
	// 4 frames ≈ 1.6s of silence → confident sentence boundary
	SILENCE_FRAMES_THRESHOLD = 4

	// RMS threshold for "silence" detection
	SILENCE_RMS = float64(0.008)

	// Minimum RMS energy for the entire buffer to be worth sending
	// Prevents sending buffers that are mostly silence with a tiny bit of speech
	MIN_BUFFER_RMS = float64(0.005)

	// Max time before forced flush even if still speaking (10s)
	MAX_BUFFER_DURATION = 10 * time.Second
)

var (
	upgrader = websocket.Upgrader{
		ReadBufferSize:  16384,
		WriteBufferSize: 16384,
		CheckOrigin:     func(r *http.Request) bool { return true },
	}

	rooms        = map[string]map[*Client]bool{}
	audioBuffers = map[*Client]*AudioState{}
	mu           sync.RWMutex

	httpClient = &http.Client{Timeout: 45 * time.Second}
)

/* ================= WS HANDLER ================= */

func wsHandler(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		return
	}

	client := &Client{
		conn: conn,
		send: make(chan []byte, 64),
	}

	mu.Lock()
	audioBuffers[client] = &AudioState{lastSendTime: time.Now()}
	mu.Unlock()

	go writePump(client)
	readPump(client)
}

/* ================= READ LOOP ================= */

func readPump(c *Client) {
	defer cleanup(c)

	for {
		msgType, data, err := c.conn.ReadMessage()
		if err != nil {
			return
		}

		if msgType == websocket.TextMessage {
			var join JoinMessage
			if json.Unmarshal(data, &join) == nil {
				joinRoom(c, join.Room, join.Lang)
			}
			continue
		}

		if msgType == websocket.BinaryMessage {
			handleAudio(c, data)
		}
	}
}

/* ================= AUDIO ================= */

func handleAudio(c *Client, data []byte) {
	if len(data)%4 != 0 {
		return
	}

	samples := make([]float32, len(data)/4)
	if err := binary.Read(bytes.NewReader(data), binary.LittleEndian, &samples); err != nil {
		return
	}

	// Compute RMS energy of this incoming chunk
	var sumSq float64
	for _, s := range samples {
		sumSq += float64(s) * float64(s)
	}
	rms := math.Sqrt(sumSq / float64(len(samples)))

	mu.Lock()
	state := audioBuffers[c]
	state.samples = append(state.samples, samples...)

	totalSamples := len(state.samples)
	isSilent := rms < SILENCE_RMS

	if isSilent {
		state.silenceFrames++
	} else {
		state.silenceFrames = 0
	}

	// Decide whether to flush the buffer to ML:
	// 1. Enough audio + silence detected (sentence boundary)
	// 2. Buffer hit max size (forced flush)
	// 3. Timeout since last send (forced flush)
	shouldFlush := false
	reason := ""

	if totalSamples >= MIN_SAMPLES && state.silenceFrames >= SILENCE_FRAMES_THRESHOLD {
		shouldFlush = true
		reason = "silence-boundary"
	} else if totalSamples >= MAX_SAMPLES {
		shouldFlush = true
		reason = "max-buffer"
	} else if totalSamples >= MIN_SAMPLES && !state.lastSendTime.IsZero() &&
		time.Since(state.lastSendTime) > MAX_BUFFER_DURATION {
		shouldFlush = true
		reason = "timeout"
	}

	if !shouldFlush || atomic.LoadInt32(&c.mlBusy) == 1 {
		mu.Unlock()
		return
	}

	// Take all buffered audio
	audio := make([]float32, totalSamples)
	copy(audio, state.samples)
	state.samples = state.samples[:0]
	state.silenceFrames = 0
	state.lastSendTime = time.Now()

	// Check overall buffer energy — reject if mostly silence
	var bufferSumSq float64
	for _, s := range audio {
		bufferSumSq += float64(s) * float64(s)
	}
	bufferRMS := math.Sqrt(bufferSumSq / float64(len(audio)))
	if bufferRMS < MIN_BUFFER_RMS {
		log.Printf("Skipping %d samples (%.1fs) — buffer too quiet (RMS=%.5f)", len(audio), float64(len(audio))/float64(SAMPLE_RATE), bufferRMS)
		mu.Unlock()
		return
	}

	// Trim trailing silence from the buffer to give ML cleaner audio
	trimIdx := len(audio)
	windowSize := 1600 // 100ms windows
	for trimIdx > MIN_SAMPLES {
		end := trimIdx
		start := end - windowSize
		if start < 0 {
			break
		}
		var winSumSq float64
		for _, s := range audio[start:end] {
			winSumSq += float64(s) * float64(s)
		}
		winRMS := math.Sqrt(winSumSq / float64(windowSize))
		if winRMS >= SILENCE_RMS {
			break
		}
		trimIdx -= windowSize
	}
	if trimIdx < len(audio) && trimIdx >= MIN_SAMPLES {
		audio = audio[:trimIdx]
	}

	atomic.StoreInt32(&c.mlBusy, 1)
	mu.Unlock()

	log.Printf("Flushing %d samples (%.1fs) reason=%s", len(audio), float64(len(audio))/float64(SAMPLE_RATE), reason)
	go forward(c, audio)
}

/* ================= WRITE ================= */

func writePump(c *Client) {
	defer c.conn.Close()

	for msg := range c.send {
		if err := c.conn.WriteMessage(websocket.TextMessage, msg); err != nil {
			return
		}
	}
}

/* ================= ROOMS ================= */

func joinRoom(c *Client, room, lang string) {
	mu.Lock()
	defer mu.Unlock()

	if rooms[room] == nil {
		rooms[room] = map[*Client]bool{}
	}

	c.room = room
	c.lang = lang
	rooms[room][c] = true

	log.Println("Joined:", room, "lang:", lang)
}

/* ================= CLEANUP ================= */

func cleanup(c *Client) {
	mu.Lock()
	defer mu.Unlock()

	delete(audioBuffers, c)
	if c.room != "" {
		delete(rooms[c.room], c)
	}
	close(c.send)
	_ = c.conn.Close()
}

/* ================= FORWARD ================= */

func forward(sender *Client, pcm []float32) {
	defer atomic.StoreInt32(&sender.mlBusy, 0)

	/* ---------- RAW AUDIO (SAME LANG) ---------- */

	mu.RLock()
	for peer := range rooms[sender.room] {
		if peer != sender && peer.lang == sender.lang {
			buf := new(bytes.Buffer)
			_ = binary.Write(buf, binary.LittleEndian, pcm)
			peer.conn.WriteMessage(websocket.BinaryMessage, buf.Bytes())
		}
	}
	mu.RUnlock()

	/* ---------- CHECK IF ML IS NEEDED ---------- */

	needML := false
	mu.RLock()
	for peer := range rooms[sender.room] {
		if peer.lang != sender.lang {
			needML = true
			break
		}
	}
	mu.RUnlock()

	if !needML {
		return
	}

	/* ---------- ML REQUEST ---------- */

	buf := new(bytes.Buffer)
	_ = binary.Write(buf, binary.LittleEndian, pcm)

	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()

	req, _ := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		"http://127.0.0.1:9001/process",
		buf,
	)
	req.Header.Set("X-Room", sender.room)
	req.Header.Set("X-Speaker-Lang", sender.lang)

	resp, err := httpClient.Do(req)
	if err != nil {
		log.Println("ML error:", err)
		return
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil || !json.Valid(body) {
		return
	}

	var results []MLResponse
	if json.Unmarshal(body, &results) != nil {
		return
	}

	/* ---------- DISPATCH ---------- */

	mu.RLock()
	defer mu.RUnlock()

	for _, res := range results {
		for peer := range rooms[sender.room] {
			if peer.lang == res.TargetLang {
				b, _ := json.Marshal(res)
				select {
				case peer.send <- b:
				default:
				}
			}
		}
	}
}

/* ================= MAIN ================= */

func main() {
	log.Println("🚀 Go WebSocket server running on :8000")
	http.HandleFunc("/ws", wsHandler)
	log.Fatal(http.ListenAndServe("0.0.0.0:8000", nil))
}
