package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"strconv"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

/* ================= CONSTANTS ================= */

const (
	frameBytes   = 20480          // 0.32s of float32 @16kHz
	windowFrames = 19             // 19 * 0.32 ≈ 6s
	windowBytes  = frameBytes * windowFrames
)

/* ================= CLIENT ================= */

type Client struct {
	conn      *websocket.Conn
	room      string
	mode      int
	isSpeaker bool
	send      chan []byte
	id        string
}

/* ================= CONTROL ================= */

type ControlMessage struct {
	Room      string `json:"room"`
	Mode      int    `json:"mode"`
	IsSpeaker bool   `json:"isSpeaker"`
}

/* ================= AUDIO STATE ================= */

type AudioState struct {
	buffer []byte
	busy   bool
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

	httpClient = &http.Client{
		Timeout: 90 * time.Second,
	}
)

/* ================= WS HANDLER ================= */

func wsHandler(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Println("❌ Upgrade error:", err)
		return
	}

	client := &Client{
		conn: conn,
		send: make(chan []byte, 64),
		id:   conn.RemoteAddr().String(),
	}

	mu.Lock()
	audioBuffers[client] = &AudioState{}
	mu.Unlock()

	log.Println("🔌 Client connected:", client.id)

	go writePump(client)
	readPump(client)
}

/* ================= READ LOOP ================= */

func readPump(c *Client) {
	defer cleanup(c)

	for {
		msgType, data, err := c.conn.ReadMessage()
		if err != nil {
			log.Println("❌ Read error:", c.id, err)
			return
		}

		/* ---------- CONTROL ---------- */
		if msgType == websocket.TextMessage {
			var msg ControlMessage
			if err := json.Unmarshal(data, &msg); err != nil {
				continue
			}

			if msg.Room != "" {
				joinRoom(c, msg.Room)
			}

			c.mode = msg.Mode
			c.isSpeaker = msg.IsSpeaker
			continue
		}

		/* ---------- AUDIO ---------- */
		if msgType == websocket.BinaryMessage {
			if !c.isSpeaker || c.room == "" {
				continue
			}

			// Must be EXACT frame size
			if len(data) != frameBytes {
				continue
			}

			mu.Lock()
			state := audioBuffers[c]

			if state.busy {
				mu.Unlock()
				continue
			}

			state.buffer = append(state.buffer, data...)

			if len(state.buffer) >= windowBytes {
				audio := make([]byte, windowBytes)
				copy(audio, state.buffer[:windowBytes])
				state.buffer = state.buffer[windowBytes:]
				state.busy = true
				mu.Unlock()

				go forwardToML(c, audio)
			} else {
				mu.Unlock()
			}
		}
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

func joinRoom(c *Client, room string) {
	mu.Lock()
	defer mu.Unlock()

	if c.room != "" && rooms[c.room] != nil {
		delete(rooms[c.room], c)
	}

	if rooms[room] == nil {
		rooms[room] = make(map[*Client]bool)
	}

	c.room = room
	rooms[room][c] = true
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

func forwardToML(c *Client, audio []byte) {
	defer func() {
		mu.Lock()
		if state := audioBuffers[c]; state != nil {
			state.busy = false
		}
		mu.Unlock()
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		"http://127.0.0.1:9000/process",
		bytes.NewReader(audio),
	)
	if err != nil {
		return
	}

	req.Header.Set("Content-Type", "application/octet-stream")
	req.Header.Set("X-Mode", strconv.Itoa(c.mode))
	req.Header.Set("X-Room", c.room)

	resp, err := httpClient.Do(req)
	if err != nil {
		log.Println("❌ ML request failed:", err)
		return
	}
	defer resp.Body.Close()

	out, err := io.ReadAll(resp.Body)
	if err != nil || !json.Valid(out) {
		return
	}

	var msg map[string]any
	if err := json.Unmarshal(out, &msg); err != nil {
		return
	}

	// 🔥 ONLY FORWARD FINAL
	if msg["type"] != "final" {
		return
	}

	mu.RLock()
	defer mu.RUnlock()

	for client := range rooms[c.room] {
		if client.isSpeaker {
			continue
		}
		select {
		case client.send <- out:
		default:
		}
	}
}

/* ================= MAIN ================= */

func main() {
	log.Println("🚀 WebSocket server running on :8000")
	http.HandleFunc("/ws", wsHandler)
	log.Fatal(http.ListenAndServe(":8000", nil))
}
