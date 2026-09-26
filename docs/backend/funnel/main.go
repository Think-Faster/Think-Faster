// Воронка показаний Think Faster (INTEGRATION §13.9): шина объектов (Вариант Б) присылает события
// пакетами POST /events, воронка проверяет каждое, пишет хорошие в Kafka (tf.ingest.journal — тревожные
// и нечисловые, tf.ingest.readings — числа) и следит, какие каналы замолчали (channel.status в
// tf.ingest.reference). На стенде TF_FUNNEL_PULL — забор у эмулятора вместо приёма.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/twmb/franz-go/pkg/kgo"
)

func getenv(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

// splitList — "a, b,,c" → [a b c].
func splitList(s string) []string {
	out := []string{}
	for _, p := range strings.Split(s, ",") {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

// kafkaConf — куда писать: nil — в файл (TF_KAFKA_BOOTSTRAP=off). Пароль tf-funnel из Vault; в prod
// обязателен, в dev без него — без SASL.
func kafkaConf() (*KafkaConf, error) {
	bootstrap := getenv("TF_KAFKA_BOOTSTRAP", "tf-kafka:9092")
	if bootstrap == "off" {
		return nil, nil
	}
	password, err := Secret("kafka/funnel", "TF_KAFKA_FUNNEL_PASSWORD", "TF_KAFKA_FUNNEL_PASSWORD", !IsDev())
	if err != nil {
		return nil, err
	}
	return &KafkaConf{Bootstrap: bootstrap, User: getenv("TF_KAFKA_USER", "tf-funnel"), Password: password}, nil
}

func makeSink() (Sink, func(), error) {
	conf, err := kafkaConf()
	if err != nil {
		return nil, nil, err
	}
	if conf == nil {
		return NewFileSink(getenv("TF_FUNNEL_OUT", "/tmp/tf-funnel.jsonl")), func() {}, nil
	}
	client, err := kgo.NewClient(KafkaOpts(*conf)...)
	if err != nil {
		return nil, nil, err
	}
	return &KafkaSink{P: client, Timeout: 30 * time.Second}, client.Close, nil
}

// makeVerifier — ключ из TF_AUTH_PUBLIC_KEY (открытый, в Vault его нет), иначе по адресу TF_AUTH_JWKS
// (think-auth). nil — dev без ключа: пакеты принимаются без токена.
func makeVerifier() (*Verifier, error) {
	pemKey := os.Getenv("TF_AUTH_PUBLIC_KEY")
	keyURL := os.Getenv("TF_AUTH_JWKS")
	if pemKey == "" && keyURL == "" {
		if IsDev() {
			slog.Warn("dev: ключа проверки токенов нет — пакеты принимаются без токена")
			return nil, nil
		}
		keyURL = "http://tf-auth:8080/.well-known/jwks"
	}
	subs, err := Secret("app/tf-funnel", "TF_FUNNEL_SERVICE_SUBS", "TF_FUNNEL_SERVICE_SUBS", false)
	if err != nil {
		return nil, err
	}
	var pemBytes []byte
	if pemKey != "" {
		pemBytes, keyURL = []byte(pemKey), ""
	}
	return NewVerifier(pemBytes, keyURL, splitList(subs)), nil
}

// probe — проверка живости для HEALTHCHECK образа (в образе нет curl): 0 — воронка отвечает.
func probe(port int) int {
	resp, err := (&http.Client{Timeout: 3 * time.Second}).Get(fmt.Sprintf("http://127.0.0.1:%d/health", port))
	if err != nil {
		return 1
	}
	resp.Body.Close()
	if resp.StatusCode != 200 {
		return 1
	}
	return 0
}

func main() {
	port, _ := strconv.Atoi(getenv("TF_FUNNEL_PORT", "8000"))
	flag.IntVar(&port, "port", port, "порт HTTP")
	health := flag.Bool("health", false, "проверить живость запущенной воронки и выйти")
	flag.Parse()
	if *health {
		os.Exit(probe(port))
	}
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, nil)))
	fail := func(what string, err error) {
		slog.Error(what + ": " + err.Error())
		os.Exit(1)
	}

	var spool string
	if s := os.Getenv("TF_AUDIT_SPOOL"); s != "" {
		spool = s
	}
	audit := NewAudit("funnel", NewRedisStream(os.Getenv("TF_REDIS_URL")), spool)
	sink, closeSink, err := makeSink()
	if err != nil {
		fail("Kafka", err)
	}
	defer closeSink()
	verifier, err := makeVerifier()
	if err != nil {
		fail("ключ проверки токенов", err)
	}
	silentMin, err := strconv.ParseFloat(getenv("TF_FUNNEL_SILENT_MIN", "60"), 64)
	if err != nil {
		fail("TF_FUNNEL_SILENT_MIN", err)
	}
	funnel := NewFunnel(sink, audit, NewChannels(60*silentMin))

	stop := make(stopChan)
	go func() { // обход молчания и досылка аудита
		for !stop.Wait(30) {
			func() {
				defer func() {
					if r := recover(); r != nil {
						slog.Error(fmt.Sprintf("обход молчания: %v", r))
					}
				}()
				funnel.Sweep(now())
				audit.Flush()
			}()
		}
	}()
	if base := os.Getenv("TF_FUNNEL_PULL"); base != "" {
		go Pull(funnel, base, stop, 1, 5000, nil)
		slog.Info("стенд: забираю поток у " + base)
	}

	srv := &http.Server{Addr: fmt.Sprintf(":%d", port), Handler: NewHandler(funnel, verifier, audit,
		NewRequestLog(audit, verifier, 10_000)), ReadHeaderTimeout: 10 * time.Second}
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	go func() {
		<-ctx.Done()
		close(stop)
		shut, done := context.WithTimeout(context.Background(), 20*time.Second)
		defer done()
		srv.Shutdown(shut)
	}()
	slog.Info(fmt.Sprintf("воронка слушает :%d", port))
	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		fail("HTTP", err)
	}
}
