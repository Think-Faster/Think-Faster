package main

// Окно «Логи»: GET /log — история объекта из архива, GET /stream — WebSocket с живыми показаниями и
// сменами молчания каналов. Пользователь — по токену think-auth (cookie access_token, которую браузер
// шлёт сам, или Authorization: Bearer); какие датчики ему видны — по ответу BFF (scope.go).
//
// Сообщения /stream (JSON, по одному в кадре):
//   {"type":"ready","objectIds":[...],"sensors":N}             — подписка принята
//   {"type":"reading", ...Reading}                              — показание
//   {"type":"status","sensorId":N,"status":"silent|ok","at":..,"since":..} — канал замолчал или ожил
//   {"type":"dropped","count":N}                                — клиент не успевал, N показаний пропущено
// Закрытие 4401 — срок токена истёк: обновить токен (любой запрос к BFF) и подключиться снова.

import (
	"context"
	"errors"
	"log/slog"
	"math"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coder/websocket"
)

// ----- раздача живых показаний -----------------------------------------------------------------

type subscriber struct {
	sensors map[int64]bool
	out     chan []byte
	dropped atomic.Int64
}

type Hub struct {
	Buffer int
	mu     sync.RWMutex
	subs   map[*subscriber]struct{}
}

func NewHub() *Hub { return &Hub{Buffer: 1024, subs: map[*subscriber]struct{}{}} }

func (h *Hub) add(sensors []int64) *subscriber {
	s := &subscriber{sensors: make(map[int64]bool, len(sensors)), out: make(chan []byte, h.Buffer)}
	for _, id := range sensors {
		s.sensors[id] = true
	}
	h.mu.Lock()
	h.subs[s] = struct{}{}
	h.mu.Unlock()
	return s
}

func (h *Hub) remove(s *subscriber) {
	h.mu.Lock()
	delete(h.subs, s)
	h.mu.Unlock()
}

// Subscribers — сколько окон «Логи» подключено (для /status).
func (h *Hub) Subscribers() int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return len(h.subs)
}

// deliver — не ждёт медленного клиента: переполненный буфер — пропуск и счётчик.
func (h *Hub) deliver(channel int64, msg func() []byte) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	var data []byte
	for s := range h.subs {
		if !s.sensors[channel] {
			continue
		}
		if data == nil {
			data = msg()
		}
		select {
		case s.out <- data:
		default:
			s.dropped.Add(1)
		}
	}
}

type readingMsg struct {
	Type string `json:"type"`
	Reading
}

type statusMsg struct {
	Type     string `json:"type"`
	SensorID int64  `json:"sensorId"`
	Status   string `json:"status"`
	At       string `json:"at"`
	Since    string `json:"since"`
}

func (h *Hub) Publish(rows []Row) {
	for _, r := range rows {
		r := r
		h.deliver(r.Channel, func() []byte { return marshal(readingMsg{"reading", r.Reading()}) })
	}
}

func (h *Hub) PublishStatus(channel int64, state, at, since string) {
	h.deliver(channel, func() []byte { return marshal(statusMsg{"status", channel, state, at, since}) })
}

// ----- пользователь и его объекты ----------------------------------------------------------------

// userToken — токен из Authorization: Bearer или cookie (браузер сам шлёт её и в WebSocket).
func userToken(r *http.Request, cookie string) string {
	if auth := r.Header.Get("Authorization"); strings.HasPrefix(strings.ToLower(auth), "bearer ") {
		return strings.TrimSpace(auth[7:])
	}
	if c, err := r.Cookie(cookie); err == nil {
		return c.Value
	}
	return ""
}

// objectIDs — ?objectId=1&objectId=2 (или через запятую).
func objectIDs(r *http.Request) ([]int64, error) {
	var ids []int64
	for _, v := range r.URL.Query()["objectId"] {
		for _, p := range splitList(v) {
			id, err := strconv.ParseInt(p, 10, 64)
			if err != nil || id <= 0 {
				return nil, errors.New("objectId — положительное целое")
			}
			ids = append(ids, id)
		}
	}
	return ids, nil
}

// viewer — кто смотрит и какие датчики ему видны. ok=false — ответ уже отправлен.
func (s *server) viewer(w http.ResponseWriter, r *http.Request) (Claims, ViewScope, bool) {
	if s.scoper == nil {
		writeJSON(w, 503, detail{"логи выключены: нет адреса BFF (TF_BFF_URL)"})
		return nil, ViewScope{}, false
	}
	token := userToken(r, s.cookie)
	var claims Claims
	if s.verifier != nil {
		if token == "" {
			w.Header().Set("WWW-Authenticate", "Bearer")
			writeJSON(w, 401, detail{"нужен вход: нет токена"})
			return nil, ViewScope{}, false
		}
		c, err := s.verifier.Verify(token, "")
		if err != nil {
			var te *TokenError
			if !errors.As(err, &te) {
				writeJSON(w, 503, detail{"проверка токена не удалась"})
			} else if te.Audit {
				s.refuse(w, r, "token.refused", te.Status, te.Reason, nil, te.JTI)
			} else {
				writeJSON(w, te.Status, detail{te.Reason})
			}
			return nil, ViewScope{}, false
		}
		claims = c
	}
	objs, err := objectIDs(r)
	if err != nil {
		writeJSON(w, 422, detail{err.Error()})
		return nil, ViewScope{}, false
	}
	sc, err := s.scoper.Scope(r.Context(), token, objs)
	if err != nil {
		var te *TokenError
		if errors.As(err, &te) {
			if te.Status == 403 {
				s.refuse(w, r, "access.denied", 403, te.Reason, claims, "")
			} else {
				writeJSON(w, te.Status, detail{te.Reason})
			}
		} else {
			writeJSON(w, 503, detail{"права на показания не проверить"})
		}
		return nil, ViewScope{}, false
	}
	if len(sc.SensorIDs) == 0 {
		reason := "у этих объектов нет датчиков, доступных вам"
		if len(objs) == 0 {
			reason = "укажите objectId"
			if !sc.All {
				reason = "нет заявок в работе — показания смотреть не по чему"
			}
		}
		writeJSON(w, 403, detail{reason})
		return nil, ViewScope{}, false
	}
	return claims, sc, true
}

// ----- GET /log ----------------------------------------------------------------------------------

type logPage struct {
	Items []Reading `json:"items"`
	More  bool      `json:"more"`
}

// parseTime — ISO 8601 с поясом или секунды эпохи; пусто — def.
func parseTime(v string, def float64) (float64, error) {
	if v == "" {
		return def, nil
	}
	if f, err := strconv.ParseFloat(v, 64); err == nil {
		return f, nil
	}
	t, err := time.Parse(time.RFC3339Nano, v)
	if err != nil {
		return 0, errors.New("время — ISO 8601 с поясом, например 2026-09-27T10:00:00+03:00")
	}
	return float64(t.UnixNano()) / 1e9, nil
}

func (s *server) log(w http.ResponseWriter, r *http.Request) {
	if s.f.Archive == nil {
		writeJSON(w, 503, detail{"архив показаний выключен"})
		return
	}
	_, sc, ok := s.viewer(w, r)
	if !ok {
		return
	}
	q := r.URL.Query()
	// архив хранит время приёма до секунды: без to берём и то, что пришло в эту секунду
	to, err := parseTime(q.Get("to"), math.Floor(s.f.Now())+1)
	if err != nil {
		writeJSON(w, 422, detail{err.Error()})
		return
	}
	from, err := parseTime(q.Get("from"), to-24*3600)
	if err != nil {
		writeJSON(w, 422, detail{err.Error()})
		return
	}
	limit := 200
	if v := q.Get("limit"); v != "" {
		if limit, err = strconv.Atoi(v); err != nil || limit < 1 || limit > 1000 {
			writeJSON(w, 422, detail{"limit — от 1 до 1000"})
			return
		}
	}
	if from >= to {
		writeJSON(w, 422, detail{"from должно быть раньше to"})
		return
	}
	channels := make(map[int64]bool, len(sc.SensorIDs))
	for _, id := range sc.SensorIDs {
		channels[id] = true
	}
	rows, more, err := s.f.Archive.Read(channels, from, to, limit, s.maxHours)
	if err != nil {
		slog.Error("логи: " + err.Error())
		writeJSON(w, 500, detail{"архив не читается"})
		return
	}
	page := logPage{Items: make([]Reading, len(rows)), More: more}
	for i, row := range rows {
		page.Items[i] = row.Reading()
	}
	writeJSON(w, 200, page)
}

// ----- GET /stream -------------------------------------------------------------------------------

func (s *server) stream(w http.ResponseWriter, r *http.Request) {
	if s.f.Hub == nil {
		writeJSON(w, 503, detail{"живой поток выключен"})
		return
	}
	claims, sc, ok := s.viewer(w, r)
	if !ok {
		return
	}
	c, err := websocket.Accept(w, r, &websocket.AcceptOptions{OriginPatterns: s.origins})
	if err != nil {
		return // Accept уже ответил: не WebSocket или чужой Origin
	}
	defer c.CloseNow()
	sub := s.f.Hub.add(sc.SensorIDs)
	defer s.f.Hub.remove(sub)
	ctx := c.CloseRead(r.Context())

	write := func(data []byte) bool {
		wctx, cancel := context.WithTimeout(ctx, 10*time.Second)
		defer cancel()
		return c.Write(wctx, websocket.MessageText, data) == nil
	}
	if !write(marshal(struct {
		Type      string  `json:"type"`
		ObjectIDs []int64 `json:"objectIds"`
		Sensors   int     `json:"sensors"`
	}{"ready", sc.ObjectIDs, len(sc.SensorIDs)})) {
		return
	}

	var expire <-chan time.Time
	if exp, ok := claims["exp"]; ok {
		if f, err := strconv.ParseFloat(strings.TrimSpace(toString(exp)), 64); err == nil {
			d := time.Until(time.Unix(int64(f), 0))
			if d < 0 {
				d = 0
			}
			timer := time.NewTimer(d)
			defer timer.Stop()
			expire = timer.C
		}
	}
	ping := time.NewTicker(30 * time.Second)
	defer ping.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-s.done:
			c.Close(websocket.StatusGoingAway, "воронка перезапускается")
			return
		case <-expire:
			c.Close(4401, "срок токена истёк")
			return
		case <-ping.C:
			pctx, cancel := context.WithTimeout(ctx, 10*time.Second)
			err := c.Ping(pctx)
			cancel()
			if err != nil {
				return
			}
		case data := <-sub.out:
			if !write(data) {
				return
			}
			if n := sub.dropped.Swap(0); n > 0 {
				if !write(marshal(struct {
					Type  string `json:"type"`
					Count int64  `json:"count"`
				}{"dropped", n})) {
					return
				}
			}
		}
	}
}

// toString — exp из токена: json.Number или float64.
func toString(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case float64:
		return strconv.FormatFloat(x, 'f', -1, 64)
	}
	if s, ok := v.(interface{ String() string }); ok {
		return s.String()
	}
	return ""
}
