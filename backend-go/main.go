package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
)

/*
🚀 OPTIMIZED GO WEBSOCKET SERVER FOR LIVE TRANSLATION
=====================================================
Key Optimizations:
1. Per-speaker tracking with unique IDs
2. Smart VAD-based audio segmentation
3. Lower latency with smaller processing windows
4. Connection health monitoring with ping/pong
5. Graceful degradation under load
*/

/* ================= TYPES ================= */

type Client struct {
	id       string
	conn     *websocket.Conn
	room     string
	lang     string
	send     chan []byte
	sendBin  chan []byte // Binary channel for raw audio
	mlBusy   int32
	lastPing time.Time
}

type JoinMessage struct {
	Room string `json:"room"`
	Lang string `json:"lang"`
}

type MLResponse struct {
	Type           string  `json:"type"`
	SourceText     string  `json:"sourceText"`
	TranslatedText string  `json:"translatedText"`
	Audio          string  `json:"audio"`
	SourceLang     string  `json:"sourceLang"`
	TargetLang     string  `json:"targetLang"`
	Confidence     float64 `json:"confidence,omitempty"`
	ProcessingTime float64 `json:"processingTime,omitempty"`
}

/* ================= AUDIO STATE ================= */

type AudioState struct {
	samples         []float32
	lastActivity    time.Time
	isSpeaking      bool
	speechStartTime time.Time
	silenceSamples  int
}

/* ================= CONFIGURATION ================= */

const (
	// Audio settings - optimized for low latency
	SAMPLE_RATE = 16000

	// Smaller window for lower latency (changed from 2.5s)
	MIN_SAMPLES = SAMPLE_RATE * 1 / 2 // 500ms minimum
	MAX_SAMPLES = SAMPLE_RATE * 15    // 15s maximum for long speech

	// Silence detection - end of utterance
	SILENCE_THRESHOLD_SAMPLES = SAMPLE_RATE * 6 / 10 // 600ms silence = end of utterance
	RMS_THRESHOLD             = 0.003

	// WebSocket settings
	WRITE_WAIT       = 10 * time.Second
	PONG_WAIT        = 60 * time.Second
	PING_PERIOD      = 30 * time.Second
	MAX_MESSAGE_SIZE = 1024 * 1024 // 1MB

	// ML settings
	ML_TIMEOUT = 60 * time.Second
)

/* ================= GLOBALS ================= */

var (
	upgrader = websocket.Upgrader{
		ReadBufferSize:  65536,  // 64KB for better throughput
		WriteBufferSize: 65536,  // 64KB for better throughput
		CheckOrigin:     func(r *http.Request) bool { return true },
	}

	rooms        = map[string]map[*Client]bool{}
	audioBuffers = map[*Client]*AudioState{}
	mu           sync.RWMutex

	httpClient = &http.Client{
		Timeout: ML_TIMEOUT,
		Transport: &http.Transport{
			MaxIdleConns:        200,
			MaxIdleConnsPerHost: 100,
			MaxConnsPerHost:     100,
			IdleConnTimeout:     90 * time.Second,
			DisableKeepAlives:   false,
			ForceAttemptHTTP2:   true,
		},
	}

	// Metrics
	activeConnections int64
	totalProcessed    int64
	clientCounter     int64
)

/* ================= HELPERS ================= */

func calculateRMS(samples []float32) float64 {
	if len(samples) == 0 {
		return 0
	}
	var sum float64
	for _, s := range samples {
		sum += float64(s) * float64(s)
	}
	return math.Sqrt(sum / float64(len(samples)))
}

func hasVoiceActivity(samples []float32) bool {
	return calculateRMS(samples) > RMS_THRESHOLD
}

func generateClientID() string {
	id := atomic.AddInt64(&clientCounter, 1)
	return fmt.Sprintf("c%d", id)
}

// Supported languages - only en, hi, te allowed
var supportedLanguages = map[string]bool{
	"en": true, // English
	"hi": true, // Hindi
	"te": true, // Telugu
}

func isValidLanguage(lang string) bool {
	return supportedLanguages[lang]
}

func getMLEndpoint() string {
	// Check for Docker environment (ML_WORKER_HOST env var)
	if host := os.Getenv("ML_WORKER_HOST"); host != "" {
		return "http://" + host + ":9001"
	}
	// Default to localhost for local development
	return "http://127.0.0.1:9001"
}

// Retry ML request with exponential backoff
func retryMLRequest(req *http.Request, maxRetries int) (*http.Response, error) {
	var resp *http.Response
	var err error
	
	for attempt := 0; attempt <= maxRetries; attempt++ {
		if attempt > 0 {
			// Exponential backoff: 100ms, 200ms, 400ms
			backoff := time.Duration(100 * (1 << (attempt - 1))) * time.Millisecond
			time.Sleep(backoff)
			log.Printf("🔄 Retry attempt %d/%d after %v", attempt, maxRetries, backoff)
		}
		
		resp, err = httpClient.Do(req)
		if err == nil && resp.StatusCode == http.StatusOK {
			return resp, nil
		}
		
		if resp != nil {
			resp.Body.Close()
		}
	}
	
	return nil, fmt.Errorf("failed after %d retries: %w", maxRetries, err)
}

/* ================= WS HANDLER ================= */

func wsHandler(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("Upgrade error: %v", err)
		return
	}

	// Configure connection
	conn.SetReadLimit(MAX_MESSAGE_SIZE)
	conn.SetReadDeadline(time.Now().Add(PONG_WAIT))
	conn.SetPongHandler(func(string) error {
		conn.SetReadDeadline(time.Now().Add(PONG_WAIT))
		return nil
	})

	client := &Client{
		id:       generateClientID(),
		conn:     conn,
		send:     make(chan []byte, 128),
		sendBin:  make(chan []byte, 64),
		lastPing: time.Now(),
	}

	mu.Lock()
	audioBuffers[client] = &AudioState{
		lastActivity: time.Now(),
	}
	atomic.AddInt64(&activeConnections, 1)
	mu.Unlock()

	go writePump(client)
	go pingPump(client)
	readPump(client)
}

/* ================= PING PUMP ================= */

func pingPump(c *Client) {
	ticker := time.NewTicker(PING_PERIOD)
	defer ticker.Stop()

	for range ticker.C {
		c.conn.SetWriteDeadline(time.Now().Add(WRITE_WAIT))
		if err := c.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
			return
		}
	}
}

/* ================= READ LOOP ================= */

func readPump(c *Client) {
	defer cleanup(c)

	for {
		msgType, data, err := c.conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				log.Printf("Read error: %v", err)
			}
			return
		}

		if msgType == websocket.TextMessage {
			var join JoinMessage
			if json.Unmarshal(data, &join) == nil && join.Room != "" && join.Lang != "" {
				joinRoom(c, join.Room, join.Lang)
			}
			continue
		}

		if msgType == websocket.BinaryMessage {
			handleAudio(c, data)
		}
	}
}

/* ================= AUDIO HANDLING ================= */

func handleAudio(c *Client, data []byte) {
	if len(data)%4 != 0 || c.room == "" {
		return
	}

	samples := make([]float32, len(data)/4)
	if err := binary.Read(bytes.NewReader(data), binary.LittleEndian, &samples); err != nil {
		return
	}

	mu.Lock()
	state := audioBuffers[c]
	state.samples = append(state.samples, samples...)
	state.lastActivity = time.Now()

	// Voice activity detection
	hasVoice := hasVoiceActivity(samples)

	if hasVoice {
		if !state.isSpeaking {
			state.isSpeaking = true
			state.speechStartTime = time.Now()
		}
		state.silenceSamples = 0
	} else {
		state.silenceSamples += len(samples)
	}

	// Decide when to process
	shouldProcess := false
	totalSamples := len(state.samples)

	// End of utterance: silence after speech
	if state.isSpeaking && state.silenceSamples > SILENCE_THRESHOLD_SAMPLES && totalSamples >= MIN_SAMPLES {
		shouldProcess = true
		state.isSpeaking = false
	}

	// Force process on max samples (long speech)
	if totalSamples >= MAX_SAMPLES {
		shouldProcess = true
	}

	// Skip if already processing
	if atomic.LoadInt32(&c.mlBusy) == 1 {
		mu.Unlock()
		return
	}

	if !shouldProcess {
		mu.Unlock()
		return
	}

	// Extract audio for processing
	var audio []float32
	if state.silenceSamples > SILENCE_THRESHOLD_SAMPLES {
		// End of utterance - take all
		audio = make([]float32, len(state.samples))
		copy(audio, state.samples)
		state.samples = state.samples[:0]
	} else {
		// Long speech - take chunk but keep overlap
		audio = make([]float32, MAX_SAMPLES)
		copy(audio, state.samples[:MAX_SAMPLES])
		// Keep last 0.5s for continuity
		keepSamples := SAMPLE_RATE / 2
		if len(state.samples) > keepSamples {
			state.samples = state.samples[len(state.samples)-keepSamples:]
		}
	}

	atomic.StoreInt32(&c.mlBusy, 1)
	mu.Unlock()

	go forward(c, audio)
}

/* ================= WRITE PUMP ================= */

func writePump(c *Client) {
	defer c.conn.Close()

	for {
		select {
		case msg, ok := <-c.send:
			if !ok {
				c.conn.WriteMessage(websocket.CloseMessage, []byte{})
				return
			}

			c.conn.SetWriteDeadline(time.Now().Add(WRITE_WAIT))
			if err := c.conn.WriteMessage(websocket.TextMessage, msg); err != nil {
				return
			}

		case bin, ok := <-c.sendBin:
			if !ok {
				return
			}

			c.conn.SetWriteDeadline(time.Now().Add(WRITE_WAIT))
			if err := c.conn.WriteMessage(websocket.BinaryMessage, bin); err != nil {
				return
			}
		}
	}
}

/* ================= ROOMS ================= */

func joinRoom(c *Client, room, lang string) {
	// Validate language - only en/hi/te allowed; if invalid, reject the join and close
	if !isValidLanguage(lang) {
		log.Printf("⚠️ [%s] Invalid language '%s', closing connection", c.id, lang)
		_ = c.conn.WriteMessage(websocket.TextMessage, []byte(`{"error":"unsupported_language"}`))
		_ = c.conn.Close()
		return
	}

	mu.Lock()
	defer mu.Unlock()

	// Leave old room
	if c.room != "" && rooms[c.room] != nil {
		delete(rooms[c.room], c)
	}

	// Join new room
	if rooms[room] == nil {
		rooms[room] = map[*Client]bool{}
	}

	c.room = room
	c.lang = lang
	rooms[room][c] = true

	log.Printf("✅ [%s] Joined room '%s' with lang '%s'", c.id, room, lang)
}

/* ================= CLEANUP ================= */

func cleanup(c *Client) {
	mu.Lock()
	defer mu.Unlock()

	delete(audioBuffers, c)
	if c.room != "" {
		delete(rooms[c.room], c)
		if len(rooms[c.room]) == 0 {
			delete(rooms, c.room)
		}
	}

	close(c.send)
	close(c.sendBin)
	c.conn.Close()

	atomic.AddInt64(&activeConnections, -1)
	log.Printf("👋 [%s] Disconnected", c.id)
}

/* ================= FORWARD TO ML ================= */

func forward(sender *Client, pcm []float32) {
	defer atomic.StoreInt32(&sender.mlBusy, 0)

	startTime := time.Now()

	/* ---------- RELAY RAW AUDIO TO SAME LANGUAGE PEERS ---------- */
	mu.RLock()
	peers := make([]*Client, 0)
	mlNeeded := false

	for peer := range rooms[sender.room] {
		if peer == sender {
			continue
		}
		if peer.lang == sender.lang {
			peers = append(peers, peer)
		} else {
			mlNeeded = true
		}
	}
	mu.RUnlock()

	// Send raw audio to same-language peers
	if len(peers) > 0 {
		buf := new(bytes.Buffer)
		binary.Write(buf, binary.LittleEndian, pcm)
		audioBytes := buf.Bytes()

		for _, peer := range peers {
			select {
			case peer.sendBin <- audioBytes:
			default:
				// Channel full, skip
			}
		}
	}

	/* ---------- ML PROCESSING FOR DIFFERENT LANGUAGES ---------- */
	if !mlNeeded {
		return
	}

	// Prepare audio buffer
	buf := new(bytes.Buffer)
	binary.Write(buf, binary.LittleEndian, pcm)

	ctx, cancel := context.WithTimeout(context.Background(), ML_TIMEOUT)
	defer cancel()

	req, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		getMLEndpoint()+"/process",
		buf,
	)
	if err != nil {
		log.Printf("Request creation error: %v", err)
		return
	}

	req.Header.Set("Content-Type", "application/octet-stream")
	req.Header.Set("X-Room", sender.room)
	req.Header.Set("X-Speaker-Lang", sender.lang)
	req.Header.Set("X-Speaker-Id", sender.id)

	resp, err := retryMLRequest(req, 2)  // Retry up to 2 times
	if err != nil {
		log.Printf("❌ ML request error after retries: %v", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		log.Printf("⚠️ ML response status: %d", resp.StatusCode)
		return
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil || len(body) == 0 {
		return
	}

	var results []MLResponse
	if err := json.Unmarshal(body, &results); err != nil {
		return
	}

	if len(results) == 0 {
		return
	}

	atomic.AddInt64(&totalProcessed, 1)
	elapsed := time.Since(startTime)
	log.Printf("📨 [%s] Processed %d results in %v", sender.id, len(results), elapsed)

	/* ---------- DISPATCH TO TARGET LANGUAGE PEERS ---------- */
	mu.RLock()
	defer mu.RUnlock()

	for _, res := range results {
		for peer := range rooms[sender.room] {
			if peer.lang == res.TargetLang && peer != sender {
				b, _ := json.Marshal(res)
				select {
				case peer.send <- b:
				default:
					log.Printf("⚠️ [%s] Send buffer full", peer.id)
				}
			}
		}
	}
}

/* ================= HEALTH ENDPOINT ================= */

func healthHandler(w http.ResponseWriter, r *http.Request) {
	mu.RLock()
	roomCount := len(rooms)
	mu.RUnlock()

	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":      "healthy",
		"connections": atomic.LoadInt64(&activeConnections),
		"rooms":       roomCount,
		"processed":   atomic.LoadInt64(&totalProcessed),
	})
}

/* ================= MAIN ================= */

func main() {
	log.Println("🚀 Optimized Go WebSocket server starting on :8000")
	log.Printf("📊 Config: MinSamples=%d, MaxSamples=%d, SilenceThreshold=%d",
		MIN_SAMPLES, MAX_SAMPLES, SILENCE_THRESHOLD_SAMPLES)

	http.HandleFunc("/ws", wsHandler)
	http.HandleFunc("/health", healthHandler)

	log.Fatal(http.ListenAndServe("0.0.0.0:8000", nil))
}
