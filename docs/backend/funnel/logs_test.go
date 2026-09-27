package main

// Окно «Логи»: архив со сжатием, /log, /stream и права из BFF — без сети наружу (BFF подменён).

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
)

func mskTime(s string) float64 {
	tm, err := time.ParseInLocation("2006-01-02 15:04:05", s, msk)
	if err != nil {
		panic(err)
	}
	return float64(tm.Unix())
}

func evt(ch int64, v string) Event {
	return Event{Channel: ch, Date: "2026-09-27", Time: "10:00:00", Value: v}
}

func values(rows []Row) []string {
	out := []string{}
	for _, r := range rows {
		out = append(out, r.Value)
	}
	return out
}

func TestRowKey(t *testing.T) {
	line := marshal(Row{"2026-09-27T10:15:00+03:00", evt(1234, "0.4")})
	at, ch, ok := rowKey(line)
	if !ok || at != "2026-09-27T10:15:00+03:00" || ch != 1234 {
		t.Fatal(string(line), at, ch, ok)
	}
	if _, _, ok := rowKey([]byte(`{"x":1}`)); ok {
		t.Fatal("чужая строка принята")
	}
}

func TestArchiveNewestFirstByChannel(t *testing.T) {
	a := NewArchive(t.TempDir(), 0)
	defer a.Close()
	a.Put([]Event{evt(1, "a"), evt(2, "b"), evt(1, "c")}, mskTime("2026-09-27 10:15:00"))
	a.Put([]Event{evt(1, "d")}, mskTime("2026-09-27 11:05:00"))
	ch1 := map[int64]bool{1: true}
	day, noon := mskTime("2026-09-27 00:00:00"), mskTime("2026-09-27 12:00:00")

	rows, more, err := a.Read(ch1, day, noon, 10, 72)
	if err != nil || more || !reflect.DeepEqual(values(rows), []string{"d", "c", "a"}) {
		t.Fatal(values(rows), more, err)
	}
	if rows[0].At != "2026-09-27T11:05:00+03:00" || rows[0].Channel != 1 {
		t.Fatal(rows[0])
	}
	if rows, more, _ := a.Read(ch1, day, noon, 2, 72); !more || !reflect.DeepEqual(values(rows), []string{"d", "c"}) {
		t.Fatal("limit", values(rows), more)
	}
	if rows, _, _ := a.Read(ch1, mskTime("2026-09-27 10:20:00"), noon, 10, 72); !reflect.DeepEqual(values(rows), []string{"d"}) {
		t.Fatal("from", values(rows))
	}
	if rows, more, _ := a.Read(ch1, day, mskTime("2026-09-27 11:30:00"), 10, 1); !more || !reflect.DeepEqual(values(rows), []string{"d"}) {
		t.Fatal("maxHours", values(rows), more)
	}
}

func TestArchiveCompactShrinksAndStaysReadable(t *testing.T) {
	dir := t.TempDir()
	a := NewArchive(dir, 0)
	defer a.Close()
	batch := make([]Event, 5000)
	for i := range batch {
		batch[i] = evt(int64(1+i%50), fmt.Sprintf("%.2f", float64(i%100)/10))
	}
	a.Put(batch, mskTime("2026-09-27 10:15:00"))
	raw := filepath.Join(dir, "2026-09-27", "10.jsonl")
	st, err := os.Stat(raw)
	if err != nil {
		t.Fatal(err)
	}
	if n, _ := a.Compact(mskTime("2026-09-27 11:03:00")); n != 0 { // час ещё может дописываться
		t.Fatal("сжат раньше времени")
	}
	if n, err := a.Compact(mskTime("2026-09-27 11:10:00")); n != 1 || err != nil {
		t.Fatal(n, err)
	}
	if _, err := os.Stat(raw); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("несжатый час остался")
	}
	zst, err := os.Stat(raw + ".zst")
	if err != nil || zst.Size()*5 > st.Size() {
		t.Fatalf("сжатие %d → %v (%v)", st.Size(), zst, err)
	}

	// запоздалая запись в сжатый час — второй кадр в тот же файл
	a.Put([]Event{evt(1, "поздно")}, mskTime("2026-09-27 10:59:59"))
	if n, _ := a.Compact(mskTime("2026-09-27 11:20:00")); n != 1 {
		t.Fatal("поздняя запись не сжата")
	}
	rows, _, err := a.Read(map[int64]bool{1: true}, mskTime("2026-09-27 10:00:00"), mskTime("2026-09-27 11:00:00"), 1000, 72)
	if err != nil || len(rows) != 101 || rows[0].Value != "поздно" || rows[100].Value != "0.00" {
		t.Fatal(len(rows), err)
	}
}

func TestArchiveDropsOldDays(t *testing.T) {
	dir := t.TempDir()
	a := NewArchive(dir, 2)
	a.Put([]Event{evt(1, "старое")}, mskTime("2026-09-20 10:00:00"))
	a.Put([]Event{evt(1, "свежее")}, mskTime("2026-09-26 10:00:00"))
	a.Compact(mskTime("2026-09-27 12:00:00"))
	a.Close()
	if _, err := os.Stat(filepath.Join(dir, "2026-09-20")); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("старый день не удалён")
	}
	if _, err := os.Stat(filepath.Join(dir, "2026-09-26", "10.jsonl.zst")); err != nil {
		t.Fatal("свежий день пропал:", err)
	}
}

func TestHubDropsForSlowViewer(t *testing.T) {
	h := NewHub()
	h.Buffer = 1
	s := h.add([]int64{1})
	h.Publish([]Row{{"t", evt(1, "a")}, {"t", evt(1, "b")}, {"t", evt(2, "c")}})
	if len(s.out) != 1 || s.dropped.Load() != 1 {
		t.Fatal(len(s.out), s.dropped.Load())
	}
	h.remove(s)
	if h.Subscribers() != 0 {
		t.Fatal("подписчик остался")
	}
}

// ----- HTTP и WebSocket --------------------------------------------------------------------------

type fakeScoper struct {
	mu     sync.Mutex
	s      ViewScope
	err    error
	tokens []string
	objs   [][]int64
}

func (f *fakeScoper) Scope(_ context.Context, token string, objs []int64) (ViewScope, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.tokens = append(f.tokens, token)
	f.objs = append(f.objs, objs)
	return f.s, f.err
}

func logsFixture(t *testing.T, v *Verifier, sc Scoper, done <-chan struct{}) (http.Handler, fixture) {
	x := newFixture()
	x.f.Archive = NewArchive(t.TempDir(), 0)
	t.Cleanup(x.f.Archive.Close)
	x.f.Hub = NewHub()
	return NewHandler(x.f, v, x.f.Audit, nil, &LogsConf{Scoper: sc, Done: done}), x
}

func userToken1(over map[string]any) string {
	c := map[string]any{"sub": "u-1", "scope": nil}
	for k, v := range over {
		c[k] = v
	}
	return token(nil, c)
}

func TestLogShowsOnlyScopedSensorsNewestFirst(t *testing.T) {
	sc := &fakeScoper{s: ViewScope{All: true, ObjectIDs: []int64{5}, SensorIDs: []int64{1, 2}}}
	h, x := logsFixture(t, nil, sc, nil)
	x.f.Now = func() float64 { return mskTime("2026-09-27 10:15:00") }
	if _, err := x.f.Take([]any{ev1(1), ev(3, "0.5", false, nil), ev(2, "Норма", true, nil)}, "http", nil, ""); err != nil {
		t.Fatal(err)
	}
	x.f.Now = func() float64 { return mskTime("2026-09-27 10:15:00") + 0.4 } // та же секунда
	w := call(h, "GET", "/api/funnel/log?objectId=5", "", nil)
	if w.Code != 200 {
		t.Fatal(w.Code, w.Body.String())
	}
	var page logPage
	json.Unmarshal(w.Body.Bytes(), &page)
	if len(page.Items) != 2 || page.More || page.Items[0].SensorID != 2 || !page.Items[0].Alarm ||
		page.Items[0].Value != "Норма" || page.Items[1].SensorID != 1 ||
		page.Items[1].ReceivedAt != "2026-09-27T10:15:00+03:00" || page.Items[1].Date != "2026-09-26" {
		t.Fatal(w.Body.String())
	}
	if !reflect.DeepEqual(sc.objs[0], []int64{5}) || sc.tokens[0] != "" {
		t.Fatal(sc.objs, sc.tokens)
	}
	if w := call(h, "GET", "/log?objectId=5&from=2026-09-27T10:15:30%2B03:00&to=2026-09-27T10:20:00%2B03:00", "", nil); !strings.Contains(w.Body.String(), `"items":[]`) {
		t.Fatal("from", w.Body.String())
	}
	for _, bad := range []string{"/log?objectId=x", "/log?objectId=5&limit=0", "/log?objectId=5&from=вчера",
		"/log?objectId=5&from=2026-09-28T00:00:00Z"} {
		if w := call(h, "GET", bad, "", nil); w.Code != 422 {
			t.Fatal(bad, w.Code)
		}
	}
}

func TestLogDenials(t *testing.T) {
	x := newFixture()
	x.f.Archive = NewArchive(t.TempDir(), 0)
	if w := call(NewHandler(x.f, nil, x.f.Audit, nil, nil), "GET", "/log?objectId=5", "", nil); w.Code != 503 {
		t.Fatal("без BFF", w.Code)
	}
	h, _ := logsFixture(t, nil, &fakeScoper{s: ViewScope{All: true}}, nil)
	if w := call(h, "GET", "/log", "", nil); w.Code != 403 || answer(w)["detail"] != "укажите objectId" {
		t.Fatal(w.Code, w.Body.String())
	}
	h, _ = logsFixture(t, nil, &fakeScoper{s: ViewScope{}}, nil)
	if w := call(h, "GET", "/log", "", nil); w.Code != 403 || !strings.Contains(answer(w)["detail"].(string), "нет заявок") {
		t.Fatal(w.Code, w.Body.String())
	}
	h, y := logsFixture(t, nil, &fakeScoper{err: &TokenError{Status: 403, Reason: "нет доступа к показаниям (user_inactive)"}}, nil)
	if w := call(h, "GET", "/log?objectId=5", "", nil); w.Code != 403 {
		t.Fatal(w.Code)
	}
	if ev := y.events(); len(ev) != 1 || ev[0]["event_type"] != "access.denied" {
		t.Fatal(ev)
	}
	h, _ = logsFixture(t, nil, &fakeScoper{err: &TokenError{Status: 503, Reason: "BFF не отвечает"}}, nil)
	if w := call(h, "GET", "/log?objectId=5", "", nil); w.Code != 503 {
		t.Fatal(w.Code)
	}
}

func TestLogUserTokenFromCookie(t *testing.T) {
	_, p := testKey()
	sc := &fakeScoper{s: ViewScope{SensorIDs: []int64{1}}}
	h, _ := logsFixture(t, NewVerifier(p, "", nil), sc, nil)
	if w := call(h, "GET", "/log?objectId=5", "", nil); w.Code != 401 {
		t.Fatal("без токена", w.Code)
	}
	user := userToken1(nil)
	if w := call(h, "GET", "/log?objectId=5", "", map[string]string{"Cookie": "access_token=" + user}); w.Code != 200 {
		t.Fatal(w.Code, w.Body.String())
	}
	if w := call(h, "GET", "/log?objectId=5", "", map[string]string{"Authorization": "Bearer " + user}); w.Code != 200 {
		t.Fatal(w.Code)
	}
	if len(sc.tokens) != 2 || sc.tokens[0] != user || sc.tokens[1] != user {
		t.Fatal("BFF не получил токен пользователя")
	}
	old := userToken1(map[string]any{"exp": time.Now().Unix() - 10})
	if w := call(h, "GET", "/log?objectId=5", "", map[string]string{"Cookie": "access_token=" + old}); w.Code != 401 ||
		answer(w)["detail"] != "срок токена истёк" {
		t.Fatal(w.Code, w.Body.String())
	}
}

func wsURL(srv *httptest.Server, path string) string {
	return "ws" + strings.TrimPrefix(srv.URL, "http") + path
}

func readMsg(t *testing.T, c *websocket.Conn) map[string]any {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, data, err := c.Read(ctx)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	json.Unmarshal(data, &m)
	return m
}

func waitFor(t *testing.T, cond func() bool) {
	t.Helper()
	for i := 0; i < 200 && !cond(); i++ {
		time.Sleep(10 * time.Millisecond)
	}
	if !cond() {
		t.Fatal("не дождались")
	}
}

func TestStreamLiveReadingsAndStatus(t *testing.T) {
	sc := &fakeScoper{s: ViewScope{All: true, ObjectIDs: []int64{5}, SensorIDs: []int64{1, 2}}}
	h, x := logsFixture(t, nil, sc, nil)
	srv := httptest.NewServer(h)
	defer srv.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	c, _, err := websocket.Dial(ctx, wsURL(srv, "/api/funnel/stream?objectId=5"), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer c.CloseNow()
	if m := readMsg(t, c); m["type"] != "ready" || m["sensors"] != 2.0 || !reflect.DeepEqual(m["objectIds"], []any{5.0}) {
		t.Fatal(m)
	}
	x.f.Take([]any{ev(3, "1", false, nil), ev1(1)}, "http", nil, "")
	if m := readMsg(t, c); m["type"] != "reading" || m["sensorId"] != 1.0 || m["value"] != "0.4" || m["alarm"] != false {
		t.Fatal(m)
	}
	x.f.Status([]int64{2}, "silent", now())
	if m := readMsg(t, c); m["type"] != "status" || m["sensorId"] != 2.0 || m["status"] != "silent" {
		t.Fatal(m)
	}
	c.Close(websocket.StatusNormalClosure, "")
	waitFor(t, func() bool { return x.f.Hub.Subscribers() == 0 })
}

func TestStreamClosesWhenTokenExpiresAndOnShutdown(t *testing.T) {
	_, p := testKey()
	done := make(chan struct{})
	h, _ := logsFixture(t, NewVerifier(p, "", nil), &fakeScoper{s: ViewScope{SensorIDs: []int64{1}}}, done)
	srv := httptest.NewServer(h)
	defer srv.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if _, _, err := websocket.Dial(ctx, wsURL(srv, "/stream?objectId=5"), nil); err == nil {
		t.Fatal("пустили без токена")
	}
	cookie := func(tok string) *websocket.DialOptions {
		return &websocket.DialOptions{HTTPHeader: http.Header{"Cookie": {"access_token=" + tok}}}
	}
	c, _, err := websocket.Dial(ctx, wsURL(srv, "/stream?objectId=5"), cookie(userToken1(map[string]any{"exp": time.Now().Unix() + 1})))
	if err != nil {
		t.Fatal(err)
	}
	readMsg(t, c)
	if _, _, err := c.Read(ctx); websocket.CloseStatus(err) != 4401 {
		t.Fatal("ожидали 4401:", err)
	}

	c, _, err = websocket.Dial(ctx, wsURL(srv, "/stream?objectId=5"), cookie(userToken1(nil)))
	if err != nil {
		t.Fatal(err)
	}
	readMsg(t, c)
	close(done)
	if _, _, err := c.Read(ctx); websocket.CloseStatus(err) != websocket.StatusGoingAway {
		t.Fatal("ожидали 1001:", err)
	}
}

func TestStreamThroughRequestLog(t *testing.T) {
	x := newFixture()
	x.f.Hub = NewHub()
	audit := x.f.Audit.(*Audit)
	h := NewHandler(x.f, nil, audit, NewRequestLog(audit, nil, 100),
		&LogsConf{Scoper: &fakeScoper{s: ViewScope{SensorIDs: []int64{1}}}})
	srv := httptest.NewServer(h)
	defer srv.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	c, _, err := websocket.Dial(ctx, wsURL(srv, "/api/funnel/stream?objectId=5"), nil)
	if err != nil {
		t.Fatal("WebSocket не прошёл через журнал запросов:", err)
	}
	readMsg(t, c)
	c.Close(websocket.StatusNormalClosure, "")
	waitFor(t, func() bool { return len(x.audit.events(RequestsStream)) == 1 })
	if row := x.audit.events(RequestsStream)[0]; row["route"] != "/api/funnel/stream" || row["status"] != 101.0 {
		t.Fatal(row)
	}
}

func TestBffScoper(t *testing.T) {
	calls := 0
	var mu sync.Mutex
	bff := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		calls++
		mu.Unlock()
		if r.URL.Path != "/readings/scope" {
			w.WriteHeader(404)
			return
		}
		switch r.Header.Get("Authorization") {
		case "Bearer ok":
			if !reflect.DeepEqual(r.URL.Query()["objectId"], []string{"6", "5"}) && !reflect.DeepEqual(r.URL.Query()["objectId"], []string{"5", "6"}) {
				w.WriteHeader(400)
				return
			}
			w.Write([]byte(`{"all":false,"objectIds":[5],"sensorIds":[10,11]}`))
		case "Bearer gone":
			w.WriteHeader(403)
			w.Write([]byte(`{"code":"user_inactive","message":"User is inactive."}`))
		case "Bearer boom":
			w.WriteHeader(500)
		default:
			w.WriteHeader(401)
		}
	}))
	s := NewBffScoper(bff.URL + "/")
	ctx := context.Background()
	sc, err := s.Scope(ctx, "ok", []int64{6, 5})
	if err != nil || sc.All || !reflect.DeepEqual(sc.SensorIDs, []int64{10, 11}) || !reflect.DeepEqual(sc.ObjectIDs, []int64{5}) {
		t.Fatal(sc, err)
	}
	if _, err := s.Scope(ctx, "ok", []int64{5, 6}); err != nil || calls != 1 {
		t.Fatal("ответ не взят из памяти", calls, err)
	}
	status := func(tok string) (int, string) {
		_, err := s.Scope(ctx, tok, nil)
		var te *TokenError
		if !errors.As(err, &te) {
			return 0, fmt.Sprint(err)
		}
		return te.Status, te.Reason
	}
	if st, reason := status("gone"); st != 403 || !strings.Contains(reason, "user_inactive") {
		t.Fatal(st, reason)
	}
	if st, _ := status("bad"); st != 401 {
		t.Fatal(st)
	}
	if st, _ := status("boom"); st != 503 {
		t.Fatal(st)
	}
	bff.Close()
	if st, _ := status("down"); st != 503 {
		t.Fatal(st)
	}
}
