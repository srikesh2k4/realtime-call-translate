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

/* ================= GLOBALS ================= */

var (
	upgrader = websocket.Upgrader{
		CheckOrigin: func(r *http.Request) bool { return true },
	}

	rooms = make(map[string]map[*Client]bool)
	mu    sync.RWMutex

	httpClient = &http.Client{Timeout: 90 * time.Second}
)

/* ================= WS HANDLER ================= */

func wsHandler(w http.ResponseWriter, r *http.Request) {
	conn, _ := upgrader.Upgrade(w, r, nil)

	client := &Client{
		conn: conn,
		send: make(chan []byte, 32),
		id:   conn.RemoteAddr().String(),
	}

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
			forwardToML(c, data)
		}
	}
}

/* ================= WRITE LOOP ================= */

func writePump(c *Client) {
	for msg := range c.send {
		c.conn.WriteMessage(websocket.TextMessage, msg)
	}
}

/* ================= ROOM JOIN ================= */

func joinRoom(c *Client, room, lang string) {
	mu.Lock()
	defer mu.Unlock()

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

	if c.room != "" {
		delete(rooms[c.room], c)
	}
	close(c.send)
	c.conn.Close()
}

/* ================= ML FORWARD ================= */

func forwardToML(sender *Client, pcm []byte) {
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	req, _ := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		"http://127.0.0.1:9000/process",
		bytes.NewReader(pcm),
	)

	req.Header.Set("X-Room", sender.room)
	req.Header.Set("X-Speaker-Lang", sender.lang)

	resp, err := httpClient.Do(req)
	if err != nil {
		log.Println("ML error:", err)
		return
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

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
				client.send <- b
			}
		}
	}
}

/* ================= MAIN ================= */

func main() {
	log.Println("Go WS server on :8000")
	http.HandleFunc("/ws", wsHandler)
	http.ListenAndServe(":8000", nil)
}
