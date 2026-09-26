package main

import (
	"errors"
	"io"
	"log/slog"
	"net"
	"net/http"
	"strings"
	"time"
)

const maxBody = 64 << 20 // 10 000 событий по ~200 байт с большим запасом

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	w.Write(marshal(v))
}

type detail struct {
	Detail string `json:"detail"`
}

func clientIP(r *http.Request) string {
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return host
}

// statusWriter запоминает код ответа для журнала запросов.
type statusWriter struct {
	http.ResponseWriter
	status int
}

func (s *statusWriter) WriteHeader(code int) {
	s.status = code
	s.ResponseWriter.WriteHeader(code)
}

type server struct {
	f        *Funnel
	verifier *Verifier
	audit    Auditor
}

// refuse — отказ по токену или праву: событие в аудит и ответ.
func (s *server) refuse(w http.ResponseWriter, r *http.Request, event string, status int, reason string,
	claims Claims, jti string) {
	kind, sub := "anonymous", ""
	if claims != nil {
		kind, sub = s.verifier.Kind(claims), claims.str("sub")
		if j := claims.str("jti"); j != "" {
			jti = j
		}
	}
	if err := safeAudit(s.audit, AuditEvent{Type: event, Outcome: "denied", ActorKind: kind, ActorID: sub,
		RequestID: r.Header.Get("X-Request-Id"), IP: clientIP(r), ObjectType: "route", ObjectID: r.URL.Path,
		Details: map[string]any{"reason": reason, "jti": nilIfEmpty(jti)}}); err != nil {
		slog.Error("аудит " + event + " не записан")
	}
	if status == 401 {
		w.Header().Set("WWW-Authenticate", "Bearer")
	}
	writeJSON(w, status, detail{reason})
}

// caller — кто зовёт. ok=false — ответ уже отправлен. Без проверяющего (dev без ключа) — открыто.
func (s *server) caller(w http.ResponseWriter, r *http.Request, scope string) (*Actor, bool) {
	if s.verifier == nil {
		return nil, true
	}
	auth := r.Header.Get("Authorization")
	if !strings.HasPrefix(strings.ToLower(auth), "bearer ") {
		s.refuse(w, r, "token.refused", 401, "нужен токен шины: Authorization: Bearer <токен>", nil, "")
		return nil, false
	}
	claims, err := s.verifier.Verify(strings.TrimSpace(auth[7:]), "")
	if err != nil {
		var te *TokenError
		if !errors.As(err, &te) {
			writeJSON(w, 503, detail{"проверка токена не удалась"})
			return nil, false
		}
		if !te.Audit {
			writeJSON(w, te.Status, detail{te.Reason})
			return nil, false
		}
		s.refuse(w, r, "token.refused", te.Status, te.Reason, nil, te.JTI)
		return nil, false
	}
	if scope != "" && !s.verifier.HasScope(claims, scope) {
		s.refuse(w, r, "access.denied", 403, "нужно право "+scope, claims, "")
		return nil, false
	}
	return &Actor{Sub: claims.str("sub"), Kind: s.verifier.Kind(claims)}, true
}

type health struct {
	OK           bool    `json:"ok"`
	Accepted     int     `json:"accepted"`
	LastAccepted *string `json:"last_accepted"`
	SourceSilent bool    `json:"source_silent"`
}

func (s *server) health(w http.ResponseWriter, _ *http.Request) {
	st := s.f.Stats()
	writeJSON(w, 200, health{true, st.Accepted, st.LastAccepted, s.f.Channels.SourceSilent()})
}

func (s *server) status(w http.ResponseWriter, r *http.Request) {
	if _, ok := s.caller(w, r, ""); !ok {
		return
	}
	writeJSON(w, 200, struct {
		Stats
		channelsSnapshot
	}{s.f.Stats(), s.f.Channels.Snapshot(200)})
}

func (s *server) events(w http.ResponseWriter, r *http.Request) {
	who, ok := s.caller(w, r, Scope)
	if !ok {
		return
	}
	rid := r.Header.Get("X-Request-Id")
	data, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxBody))
	if err != nil {
		writeJSON(w, 413, detail{"пакет больше 64 МБ"})
		return
	}
	var body any
	if err := decode(data, &body); err != nil {
		writeJSON(w, 422, detail{"тело — не JSON"})
		return
	}
	raw, err := EventsOf(body)
	if err != nil {
		s.f.Reject(nil, 0, "http", who, rid, err.Error())
		writeJSON(w, 422, detail{err.Error()})
		return
	}
	res, err := s.f.Take(raw, "http", who, rid)
	if err != nil {
		w.Header().Set("Retry-After", "5")
		writeJSON(w, 503, detail{err.Error() + "; повторите пакет"})
		return
	}
	if len(raw) > 0 && res.Accepted == 0 {
		writeJSON(w, 422, res)
		return
	}
	writeJSON(w, 202, res)
}

// NewHandler — ручки воронки; за nginx те же пути с префиксом /api/funnel. rl — журнал запросов
// (права-и-аудит §6.1), nil — без него.
func NewHandler(f *Funnel, v *Verifier, audit Auditor, rl *RequestLog) http.Handler {
	s := &server{f, v, audit}
	mux := http.NewServeMux()
	for _, p := range []string{"", "/api/funnel"} {
		mux.HandleFunc("GET "+p+"/health", s.health)
		mux.HandleFunc("GET "+p+"/status", s.status)
		mux.HandleFunc("POST "+p+"/events", s.events)
	}
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) { writeJSON(w, 404, detail{"Not Found"}) })
	if rl == nil {
		return mux
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t0 := time.Now()
		sw := &statusWriter{ResponseWriter: w, status: 200}
		defer func() {
			route := "(нет маршрута)"
			if _, pattern := mux.Handler(r); pattern != "" && pattern != "/" {
				route = pattern[strings.Index(pattern, " ")+1:]
			}
			rl.Put(r.Method, route, sw.status, time.Since(t0), r.Header.Get("Authorization"),
				r.Header.Get("X-Request-Id"), clientIP(r))
		}()
		mux.ServeHTTP(sw, r)
	})
}
