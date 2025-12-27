package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

/* ================= TYPES ================= */

type Client struct {
	conn *websocket.Conn
	room string
	lang string // language client wants to hear
	send chan []byte
	id   string
}

type JoinMessage struct {
	Room string `json:"room"`
	Lang string `json:"lang"` // "en" or "hi"
}

type MLResponse struct {
	Type       string `json:"type"`
	Text       string `json:"text"`
	Translated string `json:"translated"`
	Audio      string `json:"audio"`
	TargetLang string `json:"targetLang"`
}

/* ================= AUDIO BUFFER ================= */

type AudioState struct {
	buffer []byte
}

/* ================= GLOBALS ================= */

var (
	upgrader = websocket.Upgrader{
		ReadBufferSize:  8192,
		WriteBufferSize: 8192,
		CheckOrigin:     func(r *http.Request) bool { return true },
	}

	rooms        = make(map[string]map[*Client]bool)
	audioBuffers = make(map[*Client]*AudioState)

	mu sync.RWMutex

	httpClient = &http.Client{Timeout: 90 * time.Second}
)

/* ================= WS HANDLER ================= */

func wsHandler(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		return
	}

	client := &Client{
		conn: conn,
		send: make(chan []byte, 32),
		id:   conn.RemoteAddr().String(),
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

		// JOIN MESSAGE
		if msgType == websocket.TextMessage {
			var msg JoinMessage
			if json.Unmarshal(data, &msg) != nil {
				continue
			}
			joinRoom(c, msg.Room, msg.Lang)
			continue
		}

		// AUDIO PCM
		if msgType == websocket.BinaryMessage {
			handleAudio(c, data)
		}
	}
}

/* ================= AUDIO HANDLING ================= */

func handleAudio(c *Client, data []byte) {
	mu.Lock()
	state := audioBuffers[c]
	state.buffer = append(state.buffer, data...)
	size := len(state.buffer)
	mu.Unlock()

	// 0.32s frame = 20480 bytes
	// 6 seconds ≈ 19 frames
	if size >= 20480*19 {
		mu.Lock()
		audio := make([]byte, size)
		copy(audio, state.buffer)
		state.buffer = nil
		mu.Unlock()

		go forwardToML(c, audio)
	}
}

/* ================= WRITE LOOP ================= */

func writePump(c *Client) {
	for msg := range c.send {
		if err := c.conn.WriteMessage(websocket.TextMessage, msg); err != nil {
			return
		}
	}
}

/* ================= ROOM JOIN ================= */

func joinRoom(c *Client, room, lang string) {
	mu.Lock()
	defer mu.Unlock()

	// remove from old room
	if c.room != "" && rooms[c.room] != nil {
		delete(rooms[c.room], c)
	}

	if rooms[room] == nil {
		rooms[room] = make(map[*Client]bool)
	}

	c.room = room
	c.lang = lang
	rooms[room][c] = true

	log.Println("Joined:", c.id, "room:", room, "lang:", lang)
}

/* ================= CLEANUP ================= */

func cleanup(c *Client) {
	mu.Lock()
	defer mu.Unlock()

	delete(audioBuffers, c)

	if c.room != "" && rooms[c.room] != nil {
		delete(rooms[c.room], c)
		if len(rooms[c.room]) == 0 {
			delete(rooms, c.room)
		}
	}

	close(c.send)
	_ = c.conn.Close()
}

/* ================= ML FORWARD ================= */

func forwardToML(sender *Client, pcm []byte) {
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		"http://127.0.0.1:9001/process",
		bytes.NewReader(pcm),
	)
	if err != nil {
		return
	}

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

	mu.RLock()
	defer mu.RUnlock()

	for _, res := range results {
		for client := range rooms[sender.room] {
			if client.lang == res.TargetLang {
				b, _ := json.Marshal(res)
				select {
				case client.send <- b:
				default:
					// drop if slow client
				}
			}
		}
	}
}

/* ================= MAIN ================= */

func main() {
	log.Println("🚀 Go WS server on :8000")
	http.HandleFunc("/ws", wsHandler)
	log.Fatal(http.ListenAndServe("0.0.0.0:8000", nil))
}
