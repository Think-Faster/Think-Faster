package main

// То, что воронка берёт из общего пакета сервисов контура (docs/backend/tfkit/tfkit.py), на Go:
// секреты из Vault, проверка токенов think-auth, события аудита и журнал запросов
// (INTEGRATION §13.2, §13.4, §13.5). Поведение то же, что у tfkit: при правке одного — поправить и другое.

import (
	"bufio"
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/redis/go-redis/v9"
)

// Env — режим контура: prod (по умолчанию) или dev. В dev секрет, которого нет в Vault, берётся из
// переменной окружения; в prod — только из Vault.
func Env() string {
	if v := os.Getenv("TF_ENV"); v != "" {
		return v
	}
	return "prod"
}

func IsDev() bool { return Env() == "dev" }

// ----- секреты ---------------------------------------------------------------------------------

// SecretError — секрет не прочитан. В тексте только путь и поле, значения никогда.
type SecretError struct{ msg string }

func (e *SecretError) Error() string { return e.msg }

var (
	vaultMu    sync.Mutex
	vaultCache = map[string]map[string]any{}
	vaultLogin string            // токен, выданный по AppRole, на время жизни процесса
	vaultRetry = 5 * time.Second // пауза между попытками, пока Vault запечатан или не поднялся
)

func approle() (string, string) {
	return strings.TrimSpace(os.Getenv("VAULT_ROLE_ID")), strings.TrimSpace(os.Getenv("VAULT_SECRET_ID"))
}

func vaultConfigured() bool {
	role, sid := approle()
	return os.Getenv("VAULT_TOKEN") != "" || os.Getenv("VAULT_TOKEN_FILE") != "" || (role != "" && sid != "")
}

// vaultToken — готовый токен (VAULT_TOKEN, VAULT_TOKEN_FILE) или вход ролью AppRole, как
// docs/vault-entrypoint.sh think-infra. status — код ответа Vault на вход (0 — не входили или сеть).
func vaultToken(client *http.Client, addr string) (tok string, status int, err error) {
	if t := strings.TrimSpace(os.Getenv("VAULT_TOKEN")); t != "" {
		return t, 0, nil
	}
	if p := os.Getenv("VAULT_TOKEN_FILE"); p != "" {
		if b, err := os.ReadFile(p); err == nil {
			return strings.TrimSpace(string(b)), 0, nil
		}
	}
	role, sid := approle()
	if role == "" || sid == "" {
		return "", 0, nil
	}
	vaultMu.Lock()
	cached := vaultLogin
	vaultMu.Unlock()
	if cached != "" {
		return cached, 0, nil
	}
	body, _ := json.Marshal(map[string]string{"role_id": role, "secret_id": sid})
	resp, err := client.Post(strings.TrimRight(addr, "/")+"/v1/auth/approle/login", "application/json",
		strings.NewReader(string(body)))
	if err != nil {
		return "", 0, err
	}
	data, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 200 {
		return "", resp.StatusCode, nil
	}
	var v struct {
		Auth struct {
			ClientToken string `json:"client_token"`
		} `json:"auth"`
	}
	if json.Unmarshal(data, &v) != nil || v.Auth.ClientToken == "" {
		return "", 200, nil
	}
	vaultMu.Lock()
	vaultLogin = v.Auth.ClientToken
	vaultMu.Unlock()
	return v.Auth.ClientToken, 0, nil
}

// VaultRead — все поля secret/tf/<path> (KV v2), кэш на время жизни процесса. Vault Гриши после
// перезапуска запечатан и отвечает 503: TF_VAULT_WAIT секунд воронка ждёт, а не падает; 403 и 404 —
// сразу ошибка. Истёкший токен AppRole (403) — один повторный вход.
func VaultRead(path string) (map[string]any, error) {
	vaultMu.Lock()
	if d, ok := vaultCache[path]; ok {
		vaultMu.Unlock()
		return d, nil
	}
	vaultMu.Unlock()
	addr := os.Getenv("VAULT_ADDR")
	if addr == "" || !vaultConfigured() {
		return nil, &SecretError{fmt.Sprintf(
			"Vault не настроен (VAULT_ADDR и VAULT_ROLE_ID/VAULT_SECRET_ID или VAULT_TOKEN), нужен secret/tf/%s", path)}
	}
	var wait time.Duration
	if s := os.Getenv("TF_VAULT_WAIT"); s != "" {
		var sec float64
		fmt.Sscan(s, &sec)
		wait = time.Duration(sec * float64(time.Second))
	}
	deadline := time.Now().Add(wait)
	client := &http.Client{Timeout: 5 * time.Second}
	var said time.Time
	relogged := false
	for {
		var problem string
		again := false
		tok, status, err := vaultToken(client, addr)
		if err != nil {
			problem, again = fmt.Sprintf("Vault недоступен для входа AppRole: %T", errors.Unwrap(err)), true
		} else if status != 0 {
			problem = fmt.Sprintf("Vault ответил %d на вход AppRole", status)
			again = status == 502 || status == 503 || status == 504
		} else if tok == "" {
			return nil, &SecretError{fmt.Sprintf("нет токена Vault (файл VAULT_TOKEN_FILE не найден), нужен secret/tf/%s", path)}
		}
		if problem != "" {
			if !again || !time.Now().Before(deadline) {
				return nil, &SecretError{problem}
			}
			if said.IsZero() || time.Since(said) >= time.Minute {
				slog.Warn(problem + "; жду распечатывания Vault")
				said = time.Now()
			}
			time.Sleep(vaultRetry)
			continue
		}
		req, _ := http.NewRequest("GET", strings.TrimRight(addr, "/")+"/v1/secret/data/tf/"+path, nil)
		req.Header.Set("X-Vault-Token", tok)
		resp, err := client.Do(req)
		if err != nil {
			problem, again = fmt.Sprintf("Vault недоступен для secret/tf/%s: %T", path, errors.Unwrap(err)), true
		} else {
			body, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			if resp.StatusCode == 200 {
				var v struct {
					Data struct {
						Data map[string]any `json:"data"`
					} `json:"data"`
				}
				if err := json.Unmarshal(body, &v); err != nil || v.Data.Data == nil {
					return nil, &SecretError{fmt.Sprintf("ответ Vault не KV v2 для secret/tf/%s", path)}
				}
				vaultMu.Lock()
				vaultCache[path] = v.Data.Data
				vaultMu.Unlock()
				return v.Data.Data, nil
			}
			problem = fmt.Sprintf("Vault ответил %d на secret/tf/%s", resp.StatusCode, path)
			again = resp.StatusCode == 502 || resp.StatusCode == 503 || resp.StatusCode == 504
			vaultMu.Lock()
			expired := resp.StatusCode == 403 && vaultLogin != "" && !relogged
			if expired {
				vaultLogin = "" // токен AppRole истёк: один раз войти заново, сразу
			}
			vaultMu.Unlock()
			if expired {
				relogged = true
				continue
			}
		}
		if !again || !time.Now().Before(deadline) {
			return nil, &SecretError{problem}
		}
		if said.IsZero() || time.Since(said) >= time.Minute {
			slog.Warn(problem + "; жду распечатывания Vault")
			said = time.Now()
		}
		time.Sleep(vaultRetry)
	}
}

// Secret — поле секрета. Сначала Vault; в dev при неудаче — переменная окружения envVar.
// Пусто и nil — секрета нет, а required=false.
func Secret(path, field, envVar string, required bool) (string, error) {
	var problem string
	data, err := VaultRead(path)
	if err == nil {
		if v, ok := data[field].(string); ok && v != "" {
			return v, nil
		}
		problem = fmt.Sprintf("в secret/tf/%s нет поля %s", path, field)
	} else {
		problem = err.Error()
	}
	if envVar != "" && IsDev() && os.Getenv(envVar) != "" {
		// только в dev (локальный стенд без Vault); в логе имя переменной, не значение
		slog.Warn(problem + "; беру из переменной " + envVar)
		return os.Getenv(envVar), nil
	}
	if required {
		return "", &SecretError{problem}
	}
	return "", nil
}

// ----- токены ----------------------------------------------------------------------------------

// TokenError — отказ по токену. Audit — писать ли отказ в журнал действий: протухший токен — штатная
// работа фронта, в журнал действий не идёт (права-и-аудит §6.2); отказ по нашей вине (нет ключа) тоже.
type TokenError struct {
	Status   int
	Reason   string
	JTI, Sub string
	Audit    bool
}

func (e *TokenError) Error() string { return e.Reason }

func tokenError(status int, reason, jti, sub string) *TokenError {
	return &TokenError{Status: status, Reason: reason, JTI: jti, Sub: sub, Audit: status < 500}
}

// Claims — поля токена как есть.
type Claims map[string]any

func (c Claims) str(k string) string {
	s, _ := c[k].(string)
	return s
}

// Verifier проверяет RS256-токены think-auth (INTEGRATION §13.2). Поля в концепте и в think-auth пока
// расходятся, поэтому принимаются оба варианта: typ или token_type, издатель auth-service или tf-auth.
// Техучётка — по scope, а пока think-auth его не кладёт, по sub из списка ServiceSubs.
type Verifier struct {
	PEM         []byte
	KeyURL      string // think-auth отдаёт ключ PEM-ом (сертификат X.509 или открытый ключ)
	Audience    string
	Issuers     []string
	ServiceSubs map[string]bool
	KeyTTL      time.Duration

	mu    sync.Mutex
	key   *rsa.PublicKey
	keyAt time.Time
}

func NewVerifier(pemKey []byte, keyURL string, subs []string) *Verifier {
	m := map[string]bool{}
	for _, s := range subs {
		m[s] = true
	}
	return &Verifier{PEM: pemKey, KeyURL: keyURL, Audience: "api", Issuers: []string{"auth-service", "tf-auth"},
		ServiceSubs: m, KeyTTL: time.Hour}
}

func parseKey(data []byte) (*rsa.PublicKey, error) {
	block, _ := pem.Decode(data)
	if block == nil {
		return nil, errors.New("не PEM")
	}
	var pub any
	var err error
	switch block.Type {
	case "CERTIFICATE":
		var cert *x509.Certificate
		if cert, err = x509.ParseCertificate(block.Bytes); err == nil {
			pub = cert.PublicKey
		}
	case "RSA PUBLIC KEY":
		pub, err = x509.ParsePKCS1PublicKey(block.Bytes)
	default:
		pub, err = x509.ParsePKIXPublicKey(block.Bytes)
	}
	if err != nil {
		return nil, err
	}
	k, ok := pub.(*rsa.PublicKey)
	if !ok {
		return nil, errors.New("ключ не RSA")
	}
	return k, nil
}

func (v *Verifier) Key() (*rsa.PublicKey, error) {
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.key != nil && (v.PEM != nil || time.Since(v.keyAt) <= v.KeyTTL) {
		return v.key, nil
	}
	data := v.PEM
	if data == nil {
		if v.KeyURL == "" {
			return nil, tokenError(503, "нет публичного ключа аутентификации", "", "")
		}
		resp, err := (&http.Client{Timeout: 5 * time.Second}).Get(v.KeyURL)
		if err != nil {
			return nil, tokenError(503, "аутентификация не отдала публичный ключ", "", "")
		}
		data, _ = io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode != 200 {
			return nil, tokenError(503, "аутентификация не отдала публичный ключ", "", "")
		}
	}
	k, err := parseKey(data)
	if err != nil {
		return nil, tokenError(503, "публичный ключ аутентификации не читается", "", "")
	}
	v.key, v.keyAt = k, time.Now()
	return k, nil
}

// Verify — поля проверенного токена доступа; иначе *TokenError с кодом 401, 403 или 503.
func (v *Verifier) Verify(token, scope string) (Claims, error) {
	key, err := v.Key()
	if err != nil {
		return nil, err
	}
	opts := []jwt.ParserOption{jwt.WithValidMethods([]string{"RS256"}), jwt.WithExpirationRequired()}
	if v.Audience != "" {
		opts = append(opts, jwt.WithAudience(v.Audience))
	}
	claims := jwt.MapClaims{}
	_, err = jwt.ParseWithClaims(token, claims, func(*jwt.Token) (any, error) { return key, nil }, opts...)
	if err == nil {
		if _, ok := claims["sub"].(string); !ok {
			err = jwt.ErrTokenRequiredClaimMissing
		}
	}
	if err != nil {
		if errors.Is(err, jwt.ErrTokenExpired) && !errors.Is(err, jwt.ErrTokenSignatureInvalid) {
			return nil, &TokenError{Status: 401, Reason: "срок токена истёк"}
		}
		jti := ""
		if t, _, e := jwt.NewParser().ParseUnverified(token, jwt.MapClaims{}); e == nil {
			if m, ok := t.Claims.(jwt.MapClaims); ok {
				jti, _ = m["jti"].(string)
			}
		}
		return nil, tokenError(401, "токен не прошёл проверку: "+jwtKind(err), jti, "")
	}
	c := Claims(claims)
	jti, sub := c.str("jti"), c.str("sub")
	if len(v.Issuers) > 0 && !contains(v.Issuers, c.str("iss")) {
		return nil, tokenError(401, "чужой издатель", jti, sub)
	}
	typ := c.str("typ")
	if typ == "" {
		typ = c.str("token_type")
	}
	if typ != "access" {
		return nil, tokenError(401, "это не токен доступа", jti, sub)
	}
	if scope != "" && !v.HasScope(c, scope) {
		return nil, tokenError(403, "нужно право "+scope, jti, sub)
	}
	return c, nil
}

// jwtKind — вид ошибки для причины отказа, без содержимого токена.
func jwtKind(err error) string {
	for _, k := range []struct {
		e    error
		name string
	}{{jwt.ErrTokenMalformed, "DecodeError"}, {jwt.ErrTokenSignatureInvalid, "InvalidSignatureError"},
		{jwt.ErrTokenUnverifiable, "InvalidAlgorithmError"}, {jwt.ErrTokenInvalidAudience, "InvalidAudienceError"},
		{jwt.ErrTokenRequiredClaimMissing, "MissingRequiredClaimError"}, {jwt.ErrTokenNotValidYet, "ImmatureSignatureError"},
		{jwt.ErrTokenUsedBeforeIssued, "ImmatureSignatureError"}} {
		if errors.Is(err, k.e) {
			return k.name
		}
	}
	return "InvalidTokenError"
}

func contains(list []string, s string) bool {
	for _, x := range list {
		if x == s {
			return true
		}
	}
	return false
}

func (v *Verifier) HasScope(c Claims, scope string) bool {
	if raw, has := c["scope"]; has {
		return contains(strings.Fields(fmt.Sprint(raw)), scope)
	}
	return v.ServiceSubs[c.str("sub")]
}

// Kind — техучётка по scope, а без него по списку ServiceSubs (как в HasScope).
func (v *Verifier) Kind(c Claims) string {
	if k := c.str("kind"); k == "user" || k == "service" {
		return k
	}
	if _, has := c["scope"]; has || v.ServiceSubs[c.str("sub")] {
		return "service"
	}
	return "user"
}

// ----- аудит -----------------------------------------------------------------------------------
// Строка — audit.events из docs/common/права-и-аудит.md §6.4; транспорт — вариант Б (§6.5): XADD в
// поток Redis `audit`. Лёг Redis — событие дописывается в файл и досылается позже.

var forbidden = map[string]bool{"password": true, "token": true, "authorization": true, "cookie": true,
	"text": true, "body": true}

const RequestsStream = "audit:requests"

// Stream — куда уходит строка аудита (Redis XADD; в тестах — подмена).
type Stream interface {
	XAdd(ctx context.Context, stream, payload string) error
}

type redisStream struct {
	c      *redis.Client
	maxLen int64
}

func (r redisStream) XAdd(ctx context.Context, stream, payload string) error {
	return r.c.XAdd(ctx, &redis.XAddArgs{Stream: stream, MaxLen: r.maxLen, Approx: true,
		Values: map[string]any{"event": payload}}).Err()
}

// NewRedisStream — поток аудита по TF_REDIS_URL; пусто — без транспорта. Пароль, если его нет в
// адресе, — из Vault secret/tf/redis (TF_REDIS_PASSWORD), как tfkit.redis_url.
func NewRedisStream(url string) Stream {
	if url == "" {
		return nil
	}
	opt, err := redis.ParseURL(url)
	if err != nil {
		slog.Warn("TF_REDIS_URL не читается: " + fmt.Sprintf("%T", err))
		return nil
	}
	if opt.Password == "" {
		if pw, _ := Secret("redis", "TF_REDIS_PASSWORD", "TF_REDIS_PASSWORD", false); pw != "" {
			opt.Password = pw
		}
	}
	opt.DialTimeout, opt.ReadTimeout, opt.WriteTimeout = 2*time.Second, 2*time.Second, 2*time.Second
	return redisStream{redis.NewClient(opt), 1_000_000}
}

// AuditEvent — событие журнала действий.
type AuditEvent struct {
	Type, Outcome, ActorKind string
	ActorID, RequestID, IP   string
	ObjectType, ObjectID     string
	Details                  map[string]any
}

// Auditor — то, что воронке нужно от аудита.
type Auditor interface {
	Event(e AuditEvent) error
	Flush() int
}

type Audit struct {
	Service, Stream string
	Out             Stream
	Spool           string // файл, куда копятся события, пока Redis лежит
	mu              sync.Mutex
}

func NewAudit(service string, out Stream, spool string) *Audit {
	return &Audit{Service: service, Stream: "audit", Out: out, Spool: spool}
}

func nilIfEmpty(s string) any {
	if s == "" {
		return nil
	}
	return s
}

func uuid4() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	b[6], b[8] = b[6]&0x0f|0x40, b[8]&0x3f|0x80
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:])
}

func (a *Audit) Event(e AuditEvent) error {
	for k := range e.Details { // §6.3: пароли, токены, куки, тела и тексты сообщений не пишутся никогда
		if forbidden[strings.ToLower(k)] {
			return fmt.Errorf("в аудит нельзя писать поле %s", k)
		}
	}
	if e.Outcome == "" {
		e.Outcome = "success"
	}
	if e.ActorKind == "" {
		e.ActorKind = "service"
	}
	details := e.Details
	if details == nil {
		details = map[string]any{}
	}
	row := orderedRow{
		{"event_id", uuid4()}, {"occurred_at", time.Now().UTC().Format("2006-01-02T15:04:05.000000-07:00")},
		{"service", a.Service}, {"event_type", e.Type}, {"outcome", e.Outcome},
		{"actor_kind", e.ActorKind}, {"actor_id", nilIfEmpty(e.ActorID)}, {"actor_login", nil},
		{"request_id", nilIfEmpty(e.RequestID)}, {"ip", nilIfEmpty(e.IP)},
		{"object_type", nilIfEmpty(e.ObjectType)}, {"object_id", nilIfEmpty(e.ObjectID)},
		{"area_id", nil}, {"details", details},
	}
	a.Send(string(marshal(row)))
	return nil
}

// orderedRow — объект JSON с полями в заданном порядке (как строка аудита у tfkit).
type orderedRow []struct {
	k string
	v any
}

func (r orderedRow) MarshalJSON() ([]byte, error) {
	var b strings.Builder
	b.WriteByte('{')
	for i, kv := range r {
		if i > 0 {
			b.WriteByte(',')
		}
		b.Write(marshal(kv.k))
		b.WriteByte(':')
		b.Write(marshal(kv.v))
	}
	b.WriteByte('}')
	return []byte(b.String()), nil
}

// Send — в поток; не вышло — в файл. Аудит не роняет сервис.
func (a *Audit) Send(payload string) bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.Out != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if _, err := a.flush(ctx); err == nil {
			if err = a.Out.XAdd(ctx, a.Stream, payload); err == nil {
				return true
			}
		}
		slog.Warn("поток аудита недоступен")
	}
	if a.Spool != "" {
		_ = os.MkdirAll(filepath.Dir(a.Spool), 0o755)
		if f, err := os.OpenFile(a.Spool, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600); err == nil {
			f.WriteString(payload + "\n")
			f.Close()
		}
	} else {
		slog.Info("аудит без транспорта: " + payload)
	}
	return false
}

// Flush досылает то, что копилось в файле, пока Redis лежал. Оборвётся посреди — часть строк уйдёт
// второй раз; сервис аудита отбрасывает повтор по event_id.
func (a *Audit) Flush() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.Out == nil {
		return 0
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	n, err := a.flush(ctx)
	if err != nil {
		slog.Warn("поток аудита недоступен")
	}
	return n
}

func (a *Audit) flush(ctx context.Context) (int, error) {
	if a.Spool == "" {
		return 0, nil
	}
	f, err := os.Open(a.Spool)
	if err != nil {
		return 0, nil
	}
	var lines []string
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 16<<20)
	for sc.Scan() {
		if sc.Text() != "" {
			lines = append(lines, sc.Text())
		}
	}
	f.Close()
	for _, l := range lines {
		if err := a.Out.XAdd(ctx, a.Stream, l); err != nil {
			return 0, err
		}
	}
	os.Remove(a.Spool)
	return len(lines), nil
}

// ----- журнал запросов -------------------------------------------------------------------------
// Строка на каждый HTTP-запрос — audit.requests (права-и-аудит §6.1, §6.4): шаблон маршрута, а не путь;
// кто спросил — из токена, если он проходит проверку, иначе anonymous. Поток — audit:requests того же
// Redis. Запись не держит ответ: строка кладётся в очередь, отправляет её фоновая горутина.

type RequestLog struct {
	out      *Audit
	verifier *Verifier
	q        chan string
	mu       sync.Mutex
	Dropped  int
}

func NewRequestLog(a *Audit, v *Verifier, limit int) *RequestLog {
	spool := ""
	if a.Spool != "" {
		ext := filepath.Ext(a.Spool)
		spool = strings.TrimSuffix(a.Spool, ext) + "-requests" + ext
	}
	out := &Audit{Service: a.Service, Stream: RequestsStream, Out: a.Out, Spool: spool}
	l := &RequestLog{out: out, verifier: v, q: make(chan string, limit)}
	go func() {
		for row := range l.q {
			l.out.Send(row)
		}
	}()
	return l
}

func (l *RequestLog) actor(auth string) (string, string) {
	if l.verifier == nil || !strings.HasPrefix(strings.ToLower(auth), "bearer ") {
		return "anonymous", ""
	}
	c, err := l.verifier.Verify(strings.TrimSpace(auth[7:]), "")
	if err != nil { // чужой или протухший токен — отказ уже в журнале событий
		return "anonymous", ""
	}
	return l.verifier.Kind(c), c.str("sub")
}

// Put — строка запроса. Пробы живости (…/health) не пишутся; очередь полна — строка теряется.
func (l *RequestLog) Put(method, route string, status int, took time.Duration, auth, requestID, ip string) {
	if strings.HasSuffix(route, "/health") {
		return
	}
	kind, sub := l.actor(auth)
	row := orderedRow{
		{"occurred_at", time.Now().UTC().Format("2006-01-02T15:04:05.000000-07:00")}, {"service", l.out.Service},
		{"method", strings.ToUpper(method)}, {"route", strings.SplitN(route, "?", 2)[0]}, {"status", status},
		{"duration_ms", took.Round(time.Millisecond).Milliseconds()}, {"actor_kind", kind},
		{"actor_id", nilIfEmpty(sub)}, {"request_id", nilIfEmpty(requestID)}, {"ip", nilIfEmpty(ip)},
	}
	select {
	case l.q <- string(marshal(row)):
	default:
		l.mu.Lock()
		l.Dropped++
		l.mu.Unlock()
	}
}
