package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"io"
	"log"
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
	samples []float32
}

/* ================= GLOBALS ================= */

var (
	upgrader = websocket.Upgrader{
		ReadBufferSize:  8192,
		WriteBufferSize: 8192,
		CheckOrigin:     func(r *http.Request) bool { return true },
	}

	rooms        = map[string]map[*Client]bool{}
	audioBuffers = map[*Client]*AudioState{}
	mu           sync.RWMutex

	httpClient = &http.Client{Timeout: 120 * time.Second}

	// MUST MATCH BACKEND (2.5s @ 16k)
	WINDOW_SAMPLES = 16000 * 25 / 10
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
	audioBuffers[client] = &AudioState{}
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

	mu.Lock()
	state := audioBuffers[c]
	state.samples = append(state.samples, samples...)

	if len(state.samples) < WINDOW_SAMPLES || atomic.LoadInt32(&c.mlBusy) == 1 {
		mu.Unlock()
		return
	}

	audio := make([]float32, WINDOW_SAMPLES)
	copy(audio, state.samples[:WINDOW_SAMPLES])
	state.samples = state.samples[WINDOW_SAMPLES:]
	atomic.StoreInt32(&c.mlBusy, 1)
	mu.Unlock()

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

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
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
