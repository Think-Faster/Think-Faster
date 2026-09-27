package main

// Воронка без сети: запись в Kafka и эмулятор подменены, аудит пишет в список.

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
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

	"github.com/golang-jwt/jwt/v5"
	"github.com/twmb/franz-go/pkg/kgo"
	"github.com/twmb/franz-go/pkg/sasl"
)

func TestMain(m *testing.M) {
	for _, v := range []string{"VAULT_ADDR", "VAULT_TOKEN", "VAULT_TOKEN_FILE", "VAULT_ROLE_ID", "VAULT_SECRET_ID", "TF_REDIS_URL", "TF_AUTH_PUBLIC_KEY",
		"TF_AUTH_JWKS", "TF_FUNNEL_SERVICE_SUBS", "TF_KAFKA_FUNNEL_PASSWORD", "TF_FUNNEL_PULL", "TF_KAFKA_BOOTSTRAP"} {
		os.Unsetenv(v)
	}
	os.Setenv("TF_ENV", "dev")
	os.Exit(m.Run())
}

// jsonValue — значение так, как его отдаёт decode: числа json.Number.
func jsonValue(v any) any {
	var out any
	if err := decode(marshal(v), &out); err != nil {
		panic(err)
	}
	return out
}

// ev — событие шины; over дополняет или заменяет поля, del — убирает.
func ev(channel, value any, alarm any, over map[string]any, del ...string) any {
	e := map[string]any{"ид_события": 1, "ид_канала_данных": channel, "дата": "2026-09-26", "время": "03:09:27",
		"тревожное": alarm, "значение_датчика": value}
	for k, v := range over {
		e[k] = v
	}
	for _, k := range del {
		delete(e, k)
	}
	return jsonValue(e)
}

func ev1(channel any) any { return ev(channel, "0.4", false, nil) }

type memSink struct {
	mu   sync.Mutex
	sent []Message
	down bool
}

func (s *memSink) Send(msgs []Message) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.down {
		return &Unavailable{"Kafka не подтвердила 1 сообщений"}
	}
	s.sent = append(s.sent, msgs...)
	return nil
}

func (s *memSink) last() Message { return s.sent[len(s.sent)-1] }

// body — тело сообщения как объект JSON.
func body(m Message) map[string]any { return jsonValue(m.Value).(map[string]any) }

type memStream struct {
	mu   sync.Mutex
	rows map[string][]string
}

func (s *memStream) XAdd(_ context.Context, stream, payload string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.rows[stream] = append(s.rows[stream], payload)
	return nil
}

func (s *memStream) events(stream string) []map[string]any {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := []map[string]any{}
	for _, r := range s.rows[stream] {
		var m map[string]any
		json.Unmarshal([]byte(r), &m)
		out = append(out, m)
	}
	return out
}

type fixture struct {
	f     *Funnel
	sink  *memSink
	audit *memStream
}

func newFixture() fixture {
	s := &memStream{rows: map[string][]string{}}
	sink := &memSink{}
	return fixture{NewFunnel(sink, NewAudit("funnel", s, ""), NewChannels(3600)), sink, s}
}

func (x fixture) events() []map[string]any { return x.audit.events("audit") }

// ----- разбор ----------------------------------------------------------------------------------

func TestNormalizeKeepsJournalFieldsOnly(t *testing.T) {
	raw := ev(196771, 12, "True", map[string]any{"курсор": 5, "название_объекта": "Объект 1", "сбой": "stuck"})
	got, err := Normalize(raw)
	id := int64(1)
	want := Event{EventID: &id, Channel: 196771, Date: "2026-09-26", Time: "03:09:27", Alarm: true, Value: "12"}
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("%+v %v", got, err)
	}
	if s := string(marshal(got)); s != `{"ид_события":1,"ид_канала_данных":196771,"дата":"2026-09-26","время":"03:09:27","тревожное":true,"значение_датчика":"12"}` {
		t.Fatal(s)
	}
}

func TestNormalizeRejects(t *testing.T) {
	cases := []struct {
		raw    any
		reason string
	}{
		{"строка", "не объект"},
		{jsonValue(map[string]any{"дата": "2026-09-26"}), "нет ид_канала"},
		{ev1("abc"), "не число"},
		{ev1(true), "не число"},
		{ev1(-3), "не число"},
		{ev(1, "0.4", false, map[string]any{"дата": "26.09.2026"}), "не в формате"},
		{ev(1, "0.4", false, map[string]any{"время": nil}), "не в формате"},
		{ev(1, "0.4", false, nil, "время"), "нет даты"},
		{ev(1, "0.4", "может быть", nil), "тревожное"},
		{ev(1, nil, false, nil), "нет значения"},
		{ev(1, false, false, nil), "нет значения"},
		{ev(1, "  ", false, nil), "пустое"},
		{ev(1, strings.Repeat("x", 256), false, nil), "длиннее"},
		{ev(1, "0.4", false, map[string]any{"ид_события": "x"}), "ид_события"},
	}
	for i, c := range cases {
		if _, err := Normalize(c.raw); err == nil || !strings.Contains(err.Error(), c.reason) {
			t.Errorf("случай %d: ждали «%s», получили %v", i, c.reason, err)
		}
	}
}

func TestNumberValuePrintedLikePython(t *testing.T) {
	for in, want := range map[string]string{`12`: "12", `1.50`: "1.5", `1e-3`: "0.001", `1e20`: "1e+20", `-0.0`: "-0.0"} {
		e, err := Normalize(jsonValue(map[string]any{"ид_канала_данных": 1, "дата": "2026-09-26", "время": "03:09:27",
			"значение_датчика": json.Number(in)}))
		if err != nil || e.Value != want {
			t.Errorf("%s → %q (%v), ждали %q", in, e.Value, err, want)
		}
	}
}

func TestTopic(t *testing.T) {
	cases := []struct {
		value string
		alarm bool
		topic string
	}{
		{"0.4", false, TopicReadings},
		{"-12", false, TopicReadings},
		{"1e-3", false, TopicReadings},
		{"0.4", true, TopicJournal}, // тревога — в ленту BFF, даже числом
		{"Норма", false, TopicJournal},
		{"##.##.2026 ##:##", false, TopicJournal},
		{"nan", false, TopicJournal},
		{"0x1A", false, TopicJournal}, // float() в Python шестнадцатеричное не читает
		{"1_000", false, TopicReadings},
		{"1__0", false, TopicJournal},
	}
	for _, c := range cases {
		e, err := Normalize(ev(1, c.value, c.alarm, nil))
		if err != nil || TopicOf(e) != c.topic {
			t.Errorf("%q тревожное=%v → %s (%v)", c.value, c.alarm, TopicOf(e), err)
		}
	}
}

func TestPacketShapes(t *testing.T) {
	for _, c := range []struct {
		in   string
		want int
	}{{`{"events": [1]}`, 1}, {`[1, 2]`, 2}, {`{"курсор": 9, "событий": 1, "события": [3]}`, 1}} {
		var b any
		decode([]byte(c.in), &b)
		if got, err := EventsOf(b); err != nil || len(got) != c.want {
			t.Errorf("%s → %v %v", c.in, got, err)
		}
	}
	for _, bad := range []string{`{"x": 1}`, `"текст"`, `null`, `{"events": {}}`} {
		var b any
		decode([]byte(bad), &b)
		if _, err := EventsOf(b); err == nil || !strings.Contains(err.Error(), "список") {
			t.Errorf("%s → %v", bad, err)
		}
	}
	if _, err := EventsOf(make([]any, MaxEvents+1)); err == nil || !strings.Contains(err.Error(), "больше") {
		t.Error(err)
	}
}

// ----- приём -----------------------------------------------------------------------------------

func TestTakePartialRoutesByChannelAndAuditsWithoutValues(t *testing.T) {
	x := newFixture()
	out, err := x.f.Take([]any{ev(1, "0.4", false, nil), ev(2, "Норма", false, nil), ev("x", "секрет", false, nil),
		ev(3, "секрет", false, map[string]any{"дата": "вчера"})}, "http", &Actor{"bus-1", "service"}, "r1")
	if err != nil || out.Accepted != 2 || len(out.Rejected) != 2 || out.Rejected[0].Index != 2 || out.Rejected[1].Index != 3 {
		t.Fatalf("%+v %v", out, err)
	}
	if len(x.sink.sent) != 2 || x.sink.sent[0].Topic != TopicReadings || x.sink.sent[0].Key != "1" ||
		x.sink.sent[1].Topic != TopicJournal || x.sink.sent[1].Key != "2" {
		t.Fatalf("%+v", x.sink.sent)
	}
	rows := x.events()
	if len(rows) != 1 {
		t.Fatal(rows)
	}
	row := rows[0]
	if row["event_type"] != "telemetry.rejected" || row["outcome"] != "denied" || row["actor_id"] != "bus-1" ||
		row["request_id"] != "r1" {
		t.Fatal(row)
	}
	d := row["details"].(map[string]any)
	if d["events"] != 4.0 || d["rejected"] != 2.0 || len(d["reasons"].([]any)) != 2 {
		t.Fatal(d)
	}
	if strings.Contains(string(marshal(row)), "секрет") {
		t.Fatal("значение датчика попало в аудит")
	}
	if s := x.f.Stats(); s.Accepted != 2 || s.Rejected != 2 || s.Packets != 1 {
		t.Fatal(s)
	}
}

func TestTakeKafkaDownNothingMarked(t *testing.T) {
	x := newFixture()
	x.sink.down = true
	var u *Unavailable
	if _, err := x.f.Take([]any{ev1(1)}, "http", nil, ""); !errors.As(err, &u) {
		t.Fatal(err)
	}
	if len(x.f.Channels.last) != 0 || x.f.Stats().Accepted != 0 || x.f.Stats().Unavailable != 1 {
		t.Fatal(x.f.Stats())
	}
}

type panicAudit struct{}

func (panicAudit) Event(AuditEvent) error { panic("redis") }
func (panicAudit) Flush() int             { return 0 }

func TestAuditFailureDoesNotLosePacket(t *testing.T) {
	x := newFixture()
	x.f.Audit = panicAudit{}
	if out, err := x.f.Take([]any{ev1(1), "мусор"}, "http", nil, ""); err != nil || out.Accepted != 1 {
		t.Fatal(out, err)
	}
}

// ----- молчание --------------------------------------------------------------------------------

func TestChannelGoesSilentAndBack(t *testing.T) {
	x := newFixture()
	for _, at := range []float64{0, 10, 20} {
		x.f.Channels.Seen(1, at)
	}
	x.f.Channels.Seen(2, 20) // слышали один раз — обычного интервала нет, не судим
	if got := x.f.Sweep(20 + 3599); len(got) != 0 {
		t.Fatal(got)
	}
	if got := x.f.Sweep(20 + 3601); !reflect.DeepEqual(got, []int64{1}) {
		t.Fatal(got)
	}
	if got := x.f.Sweep(20 + 7200); len(got) != 0 { // уже помечен
		t.Fatal(got)
	}
	m := x.sink.last()
	b := body(m)
	if m.Topic != TopicReference || m.Key != "channel-status:1" || b["status"] != "silent" || b["kind"] != "channel.status" ||
		b["since"] != iso(20) {
		t.Fatal(m.Topic, m.Key, b)
	}
	if s := x.f.Channels.Snapshot(200); s.SilentChannels[0].Channel != 1 || s.Silent != 1 {
		t.Fatal(s)
	}
	x.f.Take([]any{ev1(1)}, "http", nil, "")
	if m := x.sink.last(); m.Key != "channel-status:1" || body(m)["status"] != "ok" {
		t.Fatal(m)
	}
	if len(x.f.Channels.silent) != 0 || x.f.Channels.gap[1] != 10 { // молчание не раздуло интервал
		t.Fatal(x.f.Channels.silent, x.f.Channels.gap)
	}
}

func TestRareChannelJudgedByItsOwnInterval(t *testing.T) {
	ch := NewChannels(3600)
	for _, at := range []float64{0, 7200, 14400} { // раз в два часа
		ch.Seen(5, at)
	}
	if got, _ := ch.Sweep(14400 + 4*3600); len(got) != 0 {
		t.Fatal(got)
	}
	if got, _ := ch.Sweep(14400 + 4*7200 + 1); !reflect.DeepEqual(got, []int64{5}) {
		t.Fatal(got)
	}
}

func TestBusSilence(t *testing.T) {
	x := newFixture()
	x.f.Take([]any{ev1(1)}, "http", nil, "")
	at := x.f.Channels.sourceLast
	for _, c := range []struct {
		dt   float64
		want bool
	}{{3599, false}, {3601, true}, {7200, false}} { // один раз на переход
		if _, s := x.f.Channels.Sweep(at + c.dt); s != c.want {
			t.Fatal(c.dt, s)
		}
	}
	if !x.f.Channels.Packet(at+7300) || x.f.Channels.SourceSilent() {
		t.Fatal("шина не вернулась")
	}
}

func TestStatusWriteFailureIsNotFatal(t *testing.T) {
	x := newFixture()
	for _, at := range []float64{0, 10, 20} {
		x.f.Channels.Seen(1, at)
	}
	x.sink.down = true
	if got := x.f.Sweep(10_000); !reflect.DeepEqual(got, []int64{1}) {
		t.Fatal(got)
	}
}

// ----- HTTP ------------------------------------------------------------------------------------

var (
	keyOnce sync.Once
	key     *rsa.PrivateKey
	keyPEM  []byte
)

func testKey() (*rsa.PrivateKey, []byte) {
	keyOnce.Do(func() {
		key, _ = rsa.GenerateKey(rand.Reader, 2048)
		der, _ := x509.MarshalPKIXPublicKey(&key.PublicKey)
		keyPEM = pem.EncodeToMemory(&pem.Block{Type: "PUBLIC KEY", Bytes: der})
	})
	return key, keyPEM
}

// token — токен think-auth; значение nil в over убирает поле.
func token(k *rsa.PrivateKey, over map[string]any) string {
	if k == nil {
		k, _ = testKey()
	}
	c := jwt.MapClaims{"sub": "bus-1", "jti": "j1", "iss": "auth-service", "aud": "api", "token_type": "access",
		"scope": "telemetry.push", "exp": time.Now().Unix() + 600}
	for kk, v := range over {
		if v == nil {
			delete(c, kk)
		} else {
			c[kk] = v
		}
	}
	s, err := jwt.NewWithClaims(jwt.SigningMethodRS256, c).SignedString(k)
	if err != nil {
		panic(err)
	}
	return s
}

func bearer(over map[string]any) map[string]string {
	return map[string]string{"Authorization": "Bearer " + token(nil, over), "X-Request-Id": "rq"}
}

func call(h http.Handler, method, path string, payload any, headers map[string]string) *httptest.ResponseRecorder {
	var rd *bytes.Reader
	if s, ok := payload.(string); ok {
		rd = bytes.NewReader([]byte(s))
	} else {
		rd = bytes.NewReader(marshal(payload))
	}
	r := httptest.NewRequest(method, path, rd)
	for k, v := range headers {
		r.Header.Set(k, v)
	}
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	return w
}

func answer(w *httptest.ResponseRecorder) map[string]any {
	var m map[string]any
	json.Unmarshal(w.Body.Bytes(), &m)
	return m
}

func httpFixture() (http.Handler, fixture) {
	x := newFixture()
	_, p := testKey()
	return NewHandler(x.f, NewVerifier(p, "", []string{"think-bus"}), x.f.Audit, nil, nil), x
}

func TestHTTPAcceptsPacket(t *testing.T) {
	h, x := httpFixture()
	w := call(h, "POST", "/events", map[string]any{"events": []any{ev1(1), ev(2, "Норма", false, nil)}}, bearer(nil))
	if w.Code != 202 || w.Body.String() != `{"accepted":2,"rejected":[]}` {
		t.Fatal(w.Code, w.Body.String())
	}
	if len(x.sink.sent) != 2 || len(x.events()) != 0 {
		t.Fatal(x.sink.sent, x.events())
	}
	w = call(h, "POST", "/api/funnel/events", []any{ev1(3)}, bearer(nil))
	if w.Code != 202 || answer(w)["accepted"] != 1.0 {
		t.Fatal(w.Code, w.Body.String())
	}
}

func TestHTTPServiceSubWithoutScope(t *testing.T) {
	h, _ := httpFixture()
	if w := call(h, "POST", "/events", []any{ev1(1)}, bearer(map[string]any{"sub": "think-bus", "scope": nil})); w.Code != 202 {
		t.Fatal(w.Code, w.Body.String())
	}
}

func TestHTTPNoToken(t *testing.T) {
	h, x := httpFixture()
	w := call(h, "POST", "/events", []any{ev1(1)}, nil)
	if w.Code != 401 || w.Header().Get("WWW-Authenticate") != "Bearer" {
		t.Fatal(w.Code, w.Header())
	}
	rows := x.events()
	if len(rows) != 1 || rows[0]["event_type"] != "token.refused" || rows[0]["actor_kind"] != "anonymous" ||
		rows[0]["object_id"] != "/events" || len(x.sink.sent) != 0 {
		t.Fatal(rows, x.sink.sent)
	}
}

func TestHTTPForeignToken(t *testing.T) {
	h, x := httpFixture()
	other, _ := rsa.GenerateKey(rand.Reader, 2048)
	w := call(h, "POST", "/events", []any{ev1(1)},
		map[string]string{"Authorization": "Bearer " + token(other, map[string]any{"jti": "чужой"})})
	if w.Code != 401 {
		t.Fatal(w.Code)
	}
	rows := x.events()
	if len(rows) != 1 || rows[0]["event_type"] != "token.refused" || !strings.Contains(string(marshal(rows[0])), "чужой") ||
		len(x.sink.sent) != 0 {
		t.Fatal(rows)
	}
}

func TestHTTPExpiredTokenNotAudited(t *testing.T) {
	h, x := httpFixture()
	w := call(h, "POST", "/events", []any{ev1(1)}, bearer(map[string]any{"exp": time.Now().Unix() - 60}))
	if w.Code != 401 || len(x.events()) != 0 {
		t.Fatal(w.Code, x.events())
	}
}

func TestHTTPUserWithoutPushRight(t *testing.T) {
	h, x := httpFixture()
	w := call(h, "POST", "/events", []any{ev1(1)}, bearer(map[string]any{"sub": "ivanov", "scope": "forecast.read", "kind": "user"}))
	if w.Code != 403 {
		t.Fatal(w.Code)
	}
	rows := x.events()
	if len(rows) != 1 || rows[0]["event_type"] != "access.denied" || rows[0]["actor_kind"] != "user" || rows[0]["actor_id"] != "ivanov" {
		t.Fatal(rows)
	}
}

func TestHTTPAllBadIs422(t *testing.T) {
	h, x := httpFixture()
	w := call(h, "POST", "/events", []any{ev1("x")}, bearer(nil))
	rej, _ := answer(w)["rejected"].([]any)
	if w.Code != 422 || len(rej) != 1 || rej[0].(map[string]any)["index"] != 0.0 {
		t.Fatal(w.Code, w.Body.String())
	}
	if rows := x.events(); rows[0]["event_type"] != "telemetry.rejected" {
		t.Fatal(rows)
	}
}

func TestHTTPNotAPacket(t *testing.T) {
	h, x := httpFixture()
	if w := call(h, "POST", "/events", map[string]any{"x": 1}, bearer(nil)); w.Code != 422 {
		t.Fatal(w.Code)
	}
	if rows := x.events(); rows[0]["event_type"] != "telemetry.rejected" {
		t.Fatal(rows)
	}
	if w := call(h, "POST", "/events", "{не json", bearer(nil)); w.Code != 422 {
		t.Fatal(w.Code)
	}
}

func TestHTTPKafkaDown503(t *testing.T) {
	h, x := httpFixture()
	x.sink.down = true
	w := call(h, "POST", "/events", []any{ev1(1)}, bearer(nil))
	if w.Code != 503 || w.Header().Get("Retry-After") != "5" || !strings.Contains(answer(w)["detail"].(string), "повторите") {
		t.Fatal(w.Code, w.Header(), w.Body.String())
	}
}

func TestHTTPHealthOpenStatusClosed(t *testing.T) {
	h, _ := httpFixture()
	if w := call(h, "GET", "/health", "", nil); w.Code != 200 || answer(w)["ok"] != true {
		t.Fatal(w.Code, w.Body.String())
	}
	if w := call(h, "GET", "/api/funnel/status", "", nil); w.Code != 401 {
		t.Fatal(w.Code)
	}
	s := answer(call(h, "GET", "/status", "", bearer(nil)))
	if s["channels"] != 0.0 || s["source_silent"] != false || s["packets"] != 0.0 {
		t.Fatal(s)
	}
}

func TestHTTPDevWithoutKey(t *testing.T) {
	x := newFixture()
	if w := call(NewHandler(x.f, nil, x.f.Audit, nil, nil), "POST", "/events", []any{ev1(1)}, nil); w.Code != 202 {
		t.Fatal(w.Code)
	}
}

func TestHTTPRequestLogRouteTemplate(t *testing.T) {
	x := newFixture()
	_, p := testKey()
	v := NewVerifier(p, "", nil)
	h := NewHandler(x.f, v, x.f.Audit, NewRequestLog(x.f.Audit.(*Audit), v, 100), nil)
	call(h, "GET", "/api/funnel/health", "", nil) // пробы живости не пишутся
	call(h, "POST", "/api/funnel/events", []any{ev1(1)}, bearer(nil))
	call(h, "GET", "/nope", "", nil)
	var rows []map[string]any
	for dl := time.Now().Add(2 * time.Second); time.Now().Before(dl); time.Sleep(10 * time.Millisecond) {
		if rows = x.audit.events(RequestsStream); len(rows) == 2 {
			break
		}
	}
	if len(rows) != 2 || rows[0]["route"] != "/api/funnel/events" || rows[0]["status"] != 202.0 ||
		rows[0]["actor_id"] != "bus-1" || rows[0]["actor_kind"] != "service" || rows[1]["route"] != "(нет маршрута)" ||
		rows[1]["status"] != 404.0 || rows[1]["actor_kind"] != "anonymous" {
		t.Fatal(rows)
	}
}

// ----- стенд: забор у эмулятора ----------------------------------------------------------------

// stopAfter не спит: считает паузы и останавливает цикл после n.
type stopAfter struct {
	n     int
	waits []float64
}

func (s *stopAfter) IsSet() bool { return len(s.waits) >= s.n }
func (s *stopAfter) Wait(d float64) bool {
	s.waits = append(s.waits, d)
	return s.IsSet()
}

func page(cursors ...int) map[string]any {
	rows := []any{}
	for _, c := range cursors {
		rows = append(rows, ev(c%3+1, "0.4", false, map[string]any{"курсор": c}))
	}
	last := 0
	if len(cursors) > 0 {
		last = cursors[len(cursors)-1]
	}
	return jsonValue(map[string]any{"курсор": last, "событий": len(cursors), "события": rows}).(map[string]any)
}

func TestPullFollowsCursorAndSkipsPauseOnFullPage(t *testing.T) {
	x := newFixture()
	var asked []string
	answers := []map[string]any{page(1, 2), page(3), page(), page()}
	get := func(url string) (map[string]any, error) {
		asked = append(asked, url)
		a := answers[0]
		answers = answers[1:]
		return a, nil
	}
	stop := &stopAfter{n: 3}
	Pull(x.f, "http://emu/", stop, 1, 2, get)
	want := []string{"http://emu/events?cursor=0&limit=2", "http://emu/events?cursor=2&limit=2", "http://emu/events?cursor=3&limit=2"}
	if !reflect.DeepEqual(asked[:3], want) || stop.waits[0] != 1 || x.f.Stats().Accepted != 3 {
		t.Fatal(asked, stop.waits, x.f.Stats())
	}
}

func TestPullRestartedEmulatorReadFromStart(t *testing.T) {
	x := newFixture()
	var asked []string
	get := func(url string) (map[string]any, error) {
		asked = append(asked, url)
		if strings.Contains(url, "/health") {
			return jsonValue(map[string]any{"ok": true, "курсор": 4}).(map[string]any), nil
		}
		if len(asked) == 1 {
			return page(100), nil
		}
		return page(), nil
	}
	Pull(x.f, "http://emu", &stopAfter{n: 4}, 15, 5000, get)
	if !contains(asked, "http://emu/health") || !strings.HasPrefix(asked[len(asked)-1], "http://emu/events?cursor=0&") {
		t.Fatal(asked)
	}
}

func TestPullSurvivesEmulatorDownWithBackoff(t *testing.T) {
	x := newFixture()
	stop := &stopAfter{n: 5}
	Pull(x.f, "http://emu", stop, 1, 5000, func(string) (map[string]any, error) {
		return nil, errors.New("connection refused")
	})
	if !reflect.DeepEqual(stop.waits, []float64{2, 4, 8, 16, 32}) {
		t.Fatal(stop.waits)
	}
}

func TestPullKafkaDownKeepsCursor(t *testing.T) {
	x := newFixture()
	x.sink.down = true
	var asked []string
	Pull(x.f, "http://emu", &stopAfter{n: 2}, 1, 5000, func(url string) (map[string]any, error) {
		asked = append(asked, url)
		return page(1), nil
	})
	want := []string{"http://emu/events?cursor=0&limit=5000", "http://emu/events?cursor=0&limit=5000"}
	if !reflect.DeepEqual(asked, want) {
		t.Fatal(asked)
	}
}

// ----- запись ----------------------------------------------------------------------------------

func TestFileSink(t *testing.T) {
	path := filepath.Join(t.TempDir(), "out", "f.jsonl")
	if err := NewFileSink(path).Send([]Message{{TopicReadings, "1", map[string]any{"значение_датчика": "0.4"}}}); err != nil {
		t.Fatal(err)
	}
	b, _ := os.ReadFile(path)
	if string(b) != `{"topic":"tf.ingest.readings","key":"1","value":{"значение_датчика":"0.4"}}`+"\n" {
		t.Fatal(string(b))
	}
}

type fakeProducer struct {
	recs []*kgo.Record
	err  error
}

func (p *fakeProducer) ProduceSync(_ context.Context, rs ...*kgo.Record) kgo.ProduceResults {
	out := kgo.ProduceResults{}
	for _, r := range rs {
		p.recs = append(p.recs, r)
		out = append(out, kgo.ProduceResult{Record: r, Err: p.err})
	}
	return out
}

func TestKafkaSink(t *testing.T) {
	p := &fakeProducer{}
	s := &KafkaSink{P: p, Timeout: time.Second}
	if err := s.Send([]Message{{TopicJournal, "7", map[string]any{"значение_датчика": "Норма"}}}); err != nil {
		t.Fatal(err)
	}
	r := p.recs[0]
	if r.Topic != TopicJournal || string(r.Key) != "7" || string(r.Value) != `{"значение_датчика":"Норма"}` {
		t.Fatal(r.Topic, string(r.Key), string(r.Value))
	}
}

func TestKafkaSinkUnconfirmed(t *testing.T) {
	for _, e := range []error{errors.New("NOT_ENOUGH_REPLICAS"), context.DeadlineExceeded} {
		s := &KafkaSink{P: &fakeProducer{err: e}, Timeout: time.Second}
		var u *Unavailable
		if err := s.Send([]Message{{TopicReadings, "1", map[string]any{}}}); !errors.As(err, &u) {
			t.Fatal(e, err)
		}
	}
}

func TestKafkaOpts(t *testing.T) {
	cl, err := kgo.NewClient(KafkaOpts(KafkaConf{Bootstrap: "k1:9092, k2:9092", User: "tf-funnel", Password: "dev"})...)
	if err != nil {
		t.Fatal(err)
	}
	defer cl.Close()
	if cl.OptValue(kgo.RequiredAcks) != kgo.AllISRAcks() || cl.OptValue(kgo.DisableIdempotentWrite) != false {
		t.Fatal("нужны acks=all и идемпотентность")
	}
	if m := cl.OptValue(kgo.SASL).([]sasl.Mechanism); len(m) != 1 || m[0].Name() != "PLAIN" {
		t.Fatal(m)
	}
	if seeds := cl.OptValue(kgo.SeedBrokers).([]string); len(seeds) != 2 {
		t.Fatal(seeds)
	}
}

// ----- запуск ----------------------------------------------------------------------------------

func TestMakeSinkOff(t *testing.T) {
	t.Setenv("TF_KAFKA_BOOTSTRAP", "off")
	t.Setenv("TF_FUNNEL_OUT", filepath.Join(t.TempDir(), "x.jsonl"))
	s, _, err := makeSink()
	if _, ok := s.(*FileSink); !ok || err != nil {
		t.Fatal(s, err)
	}
}

func TestMakeSinkProdNeedsVault(t *testing.T) {
	t.Setenv("TF_ENV", "prod")
	t.Setenv("TF_KAFKA_BOOTSTRAP", "k:9092")
	t.Setenv("TF_KAFKA_FUNNEL_PASSWORD", "не-берётся-в-prod")
	var se *SecretError
	if _, _, err := makeSink(); !errors.As(err, &se) {
		t.Fatal(err)
	}
}

// fakeVault — Vault с входом AppRole (r1/s1 → токены a0, a1, …) и двумя секретами; good — действующие токены.
type fakeVault struct {
	mu   sync.Mutex
	seen [][2]string
	good map[string]bool
	deny bool // политика не пускает к секретам ни с каким токеном
	n    int
}

func startVault(t *testing.T) *fakeVault {
	fv := &fakeVault{good: map[string]bool{}}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fv.mu.Lock()
		defer fv.mu.Unlock()
		if r.Method == "POST" {
			var b map[string]string
			json.NewDecoder(r.Body).Decode(&b)
			fv.seen = append(fv.seen, [2]string{r.URL.Path, ""})
			if r.URL.Path != "/v1/auth/approle/login" || b["role_id"] != "r1" || b["secret_id"] != "s1" {
				w.WriteHeader(400)
				return
			}
			tok := fmt.Sprintf("a%d", fv.n)
			fv.n++
			fv.good[tok] = true
			json.NewEncoder(w).Encode(map[string]any{"auth": map[string]string{"client_token": tok}})
			return
		}
		tok := r.Header.Get("X-Vault-Token")
		fv.seen = append(fv.seen, [2]string{r.URL.Path, tok})
		data := map[string]map[string]string{"/v1/secret/data/tf/kafka": {"funnel_password": "p1"},
			"/v1/secret/data/tf/redis": {"TF_REDIS_PASSWORD": "r1"}}[r.URL.Path]
		if data == nil {
			w.WriteHeader(404)
			return
		}
		if !fv.good[tok] || fv.deny {
			w.WriteHeader(403)
			return
		}
		json.NewEncoder(w).Encode(map[string]any{"data": map[string]any{"data": data, "metadata": map[string]any{}}})
	}))
	t.Cleanup(srv.Close)
	t.Setenv("VAULT_ADDR", srv.URL)
	t.Setenv("VAULT_ROLE_ID", "r1")
	t.Setenv("VAULT_SECRET_ID", "s1")
	resetVault := func() {
		vaultMu.Lock()
		vaultCache, vaultLogin = map[string]map[string]any{}, ""
		vaultMu.Unlock()
	}
	resetVault()
	t.Cleanup(resetVault)
	return fv
}

func TestVaultAppRoleLogin(t *testing.T) {
	// как docs/vault-entrypoint.sh think-infra: роль и секрет → токен, дальше чтение по токену
	fv := startVault(t)
	for _, c := range [][3]string{{"kafka", "funnel_password", "p1"}, {"redis", "TF_REDIS_PASSWORD", "r1"}} {
		if v, err := Secret(c[0], c[1], "", true); v != c[2] || err != nil {
			t.Fatal(c, v, err)
		}
	}
	want := [][2]string{{"/v1/auth/approle/login", ""}, {"/v1/secret/data/tf/kafka", "a0"}, {"/v1/secret/data/tf/redis", "a0"}}
	if !reflect.DeepEqual(fv.seen, want) { // вход один раз на процесс
		t.Fatal(fv.seen)
	}
}

func TestVaultAppRoleReloginOnExpiredToken(t *testing.T) {
	fv := startVault(t)
	if _, err := VaultRead("kafka"); err != nil {
		t.Fatal(err)
	}
	fv.mu.Lock()
	delete(fv.good, "a0") // токен истёк
	fv.mu.Unlock()
	vaultMu.Lock()
	vaultCache = map[string]map[string]any{}
	vaultMu.Unlock()
	if v, err := Secret("kafka", "funnel_password", "", true); v != "p1" || err != nil {
		t.Fatal(v, err)
	}
	logins := 0
	for _, s := range fv.seen {
		if s[0] == "/v1/auth/approle/login" {
			logins++
		}
	}
	if logins != 2 {
		t.Fatal(fv.seen)
	}
}

func TestVaultAppRoleWrongSecretID(t *testing.T) {
	startVault(t)
	t.Setenv("VAULT_SECRET_ID", "чужой")
	_, err := VaultRead("kafka")
	var se *SecretError
	if !errors.As(err, &se) || !strings.Contains(err.Error(), "400 на вход AppRole") || strings.Contains(err.Error(), "чужой") {
		t.Fatal(err)
	}
}

func TestVaultReadForbiddenWithoutRelogin(t *testing.T) {
	// 403 и после нового входа (путь не в политике роли) — ошибка сразу, без ожидания TF_VAULT_WAIT
	fv := startVault(t)
	t.Setenv("TF_VAULT_WAIT", "30")
	fv.deny = true
	start := time.Now()
	_, err := VaultRead("kafka")
	if err == nil || !strings.Contains(err.Error(), "403") || time.Since(start) > 3*time.Second || len(fv.seen) != 4 {
		t.Fatal(err, time.Since(start), fv.seen) // вход, 403, повторный вход, 403
	}
}

func TestKafkaConfDevPassword(t *testing.T) {
	t.Setenv("TF_KAFKA_BOOTSTRAP", "k:9092")
	t.Setenv("TF_KAFKA_FUNNEL_PASSWORD", "dev")
	c, err := kafkaConf()
	if err != nil || c.User != "tf-funnel" || c.Password != "dev" || c.Bootstrap != "k:9092" {
		t.Fatal(c, err)
	}
}

func TestMakeVerifier(t *testing.T) {
	if v, err := makeVerifier(); v != nil || err != nil { // dev без ключа — открыто
		t.Fatal(v, err)
	}
	t.Setenv("TF_ENV", "prod")
	if v, _ := makeVerifier(); v == nil || v.KeyURL != "http://tf-auth:8080/.well-known/jwks" {
		t.Fatal(v)
	}
	_, p := testKey()
	t.Setenv("TF_ENV", "dev")
	t.Setenv("TF_AUTH_PUBLIC_KEY", string(p))
	t.Setenv("TF_FUNNEL_SERVICE_SUBS", "think-bus, emu")
	v, err := makeVerifier()
	if err != nil || v.KeyURL != "" || !reflect.DeepEqual(v.ServiceSubs, map[string]bool{"think-bus": true, "emu": true}) {
		t.Fatal(v, err)
	}
	if _, err := v.Key(); err != nil {
		t.Fatal(err)
	}
}

func TestSplitList(t *testing.T) {
	if got := splitList(" a, b,,c "); fmt.Sprint(got) != "[a b c]" {
		t.Fatal(got)
	}
}
