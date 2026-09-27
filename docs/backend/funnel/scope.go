package main

// Кому какие показания видны, решает BFF (GET /readings/scope): диспетчер, главный диспетчер и админ —
// любой объект (право readings:read), инженер — объекты заявок, где он назначен и работа не закрыта.
// Воронка спрашивает с токеном пользователя и держит ответ минуту.

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ViewScope — ответ BFF: доступные объекты из запрошенных (с потомками — их датчики) и сами датчики.
// All — пользователь видит любой объект; без запрошенных объектов инженеру приходят объекты его
// заявок, остальным — пустые списки.
type ViewScope struct {
	All       bool    `json:"all"`
	ObjectIDs []int64 `json:"objectIds"`
	SensorIDs []int64 `json:"sensorIds"`
}

type Scoper interface {
	Scope(ctx context.Context, token string, objects []int64) (ViewScope, error)
}

type cachedScope struct {
	s  ViewScope
	at time.Time
}

type BffScoper struct {
	Base   string
	Client *http.Client
	TTL    time.Duration

	mu    sync.Mutex
	cache map[string]cachedScope
}

func NewBffScoper(base string) *BffScoper {
	return &BffScoper{Base: strings.TrimRight(base, "/"), Client: &http.Client{Timeout: 10 * time.Second},
		TTL: time.Minute, cache: map[string]cachedScope{}}
}

func scopeKey(token string, objects []int64) string {
	h := sha256.Sum256([]byte(token))
	ids := append([]int64(nil), objects...)
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	var b strings.Builder
	b.WriteString(hex.EncodeToString(h[:16]))
	for _, id := range ids {
		b.WriteByte(',')
		b.WriteString(strconv.FormatInt(id, 10))
	}
	return b.String()
}

type bffError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

func (b *BffScoper) Scope(ctx context.Context, token string, objects []int64) (ViewScope, error) {
	key := scopeKey(token, objects)
	b.mu.Lock()
	if c, ok := b.cache[key]; ok && time.Since(c.at) < b.TTL {
		b.mu.Unlock()
		return c.s, nil
	}
	b.mu.Unlock()

	q := url.Values{}
	for _, id := range objects {
		q.Add("objectId", strconv.FormatInt(id, 10))
	}
	u := b.Base + "/readings/scope"
	if len(q) > 0 {
		u += "?" + q.Encode()
	}
	req, err := http.NewRequestWithContext(ctx, "GET", u, nil)
	if err != nil {
		return ViewScope{}, err
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := b.Client.Do(req)
	if err != nil {
		return ViewScope{}, &TokenError{Status: 503, Reason: "BFF не отвечает — права на показания не проверить"}
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(io.LimitReader(resp.Body, 16<<20))
	switch {
	case resp.StatusCode == 401:
		return ViewScope{}, &TokenError{Status: 401, Reason: "BFF не принял токен"}
	case resp.StatusCode == 403:
		var e bffError
		_ = decode(data, &e)
		reason := "нет доступа к показаниям"
		if e.Code != "" {
			reason += " (" + e.Code + ")"
		}
		return ViewScope{}, &TokenError{Status: 403, Reason: reason}
	case resp.StatusCode != 200:
		return ViewScope{}, &TokenError{Status: 503, Reason: fmt.Sprintf("BFF ответил %d на запрос прав", resp.StatusCode)}
	}
	var s ViewScope
	if err := decode(data, &s); err != nil {
		return ViewScope{}, &TokenError{Status: 503, Reason: "BFF вернул права не в том виде"}
	}
	b.mu.Lock()
	if len(b.cache) > 10_000 {
		b.cache = map[string]cachedScope{}
	}
	b.cache[key] = cachedScope{s, time.Now()}
	b.mu.Unlock()
	return s, nil
}
