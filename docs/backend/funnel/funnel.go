package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"sync"
	"time"

	"github.com/twmb/franz-go/pkg/kgo"
	"github.com/twmb/franz-go/pkg/sasl/plain"
)

// Unavailable — Kafka не подтвердила запись: пакет не принят, шина повторит.
type Unavailable struct{ msg string }

func (e *Unavailable) Error() string { return e.msg }

// Message — одно сообщение Kafka: топик, ключ (номер канала или channel-status:<канал>), тело JSON.
type Message struct {
	Topic, Key string
	Value      any
}

// Sink — куда пишутся события: Kafka или файл. Все сообщения или *Unavailable.
type Sink interface {
	Send(msgs []Message) error
}

// ----- запись ----------------------------------------------------------------------------------

// Producer — то, что нужно от клиента Kafka (kgo.Client; в тестах — подмена).
type Producer interface {
	ProduceSync(ctx context.Context, rs ...*kgo.Record) kgo.ProduceResults
}

type KafkaSink struct {
	P       Producer
	Timeout time.Duration
}

// KafkaConf — подключение к Kafka: адреса и пользователь SASL/PLAIN (пароль из Vault).
type KafkaConf struct {
	Bootstrap, User, Password string
}

// KafkaOpts — acks=all, идемпотентность (у franz-go по умолчанию), lz4, пачка до 1 МБ.
func KafkaOpts(c KafkaConf) []kgo.Opt {
	opts := []kgo.Opt{
		kgo.SeedBrokers(splitList(c.Bootstrap)...),
		kgo.RequiredAcks(kgo.AllISRAcks()),
		kgo.ProducerBatchCompression(kgo.Lz4Compression()),
		kgo.ProducerLinger(20 * time.Millisecond),
		kgo.ProducerBatchMaxBytes(1_048_576),
		kgo.MaxBufferedRecords(100_000),
		kgo.RecordDeliveryTimeout(30 * time.Second),
	}
	if c.Password != "" {
		opts = append(opts, kgo.SASL(plain.Auth{User: c.User, Pass: c.Password}.AsMechanism()))
	}
	return opts
}

func (s *KafkaSink) Send(msgs []Message) error {
	if len(msgs) == 0 {
		return nil
	}
	rs := make([]*kgo.Record, len(msgs))
	for i, m := range msgs {
		rs[i] = &kgo.Record{Topic: m.Topic, Key: []byte(m.Key), Value: marshal(m.Value)}
	}
	ctx, cancel := context.WithTimeout(context.Background(), s.Timeout)
	defer cancel()
	failed := 0
	for _, r := range s.P.ProduceSync(ctx, rs...) {
		if r.Err != nil {
			failed++
		}
	}
	if failed > 0 {
		return &Unavailable{fmt.Sprintf("Kafka не подтвердила %d сообщений", failed)}
	}
	return nil
}

// FileSink — без Kafka (стенд, отладка): строка JSON на сообщение.
type FileSink struct {
	Path string
	mu   sync.Mutex
}

func NewFileSink(path string) *FileSink {
	_ = os.MkdirAll(filepath.Dir(path), 0o755)
	return &FileSink{Path: path}
}

func (s *FileSink) Send(msgs []Message) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	f, err := os.OpenFile(s.Path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return &Unavailable{"файл вывода не открывается"}
	}
	defer f.Close()
	for _, m := range msgs {
		row := orderedRow{{"topic", m.Topic}, {"key", m.Key}, {"value", m.Value}}
		if _, err := f.Write(append(marshal(row), '\n')); err != nil {
			return &Unavailable{"файл вывода не пишется"}
		}
	}
	return nil
}

// ----- воронка ---------------------------------------------------------------------------------

type Rejected struct {
	Index  int    `json:"index"`
	Reason string `json:"reason"`
}

type TakeResult struct {
	Accepted int        `json:"accepted"`
	Rejected []Rejected `json:"rejected"`
}

// Actor — кто прислал пакет (из токена).
type Actor struct{ Sub, Kind string }

type Stats struct {
	Packets       int     `json:"packets"`
	Accepted      int     `json:"accepted"`
	Rejected      int     `json:"rejected"`
	Unavailable   int     `json:"unavailable"`
	ArchiveErrors int     `json:"archive_errors"`
	LastAccepted  *string `json:"last_accepted"`
}

// Funnel — приём. Archive и Hub — окно «Логи» (archive.go, stream.go); nil — выключены.
type Funnel struct {
	Sink     Sink
	Audit    Auditor
	Channels *Channels
	Archive  *Archive
	Hub      *Hub
	Now      func() float64

	mu    sync.Mutex
	stats Stats
}

func NewFunnel(sink Sink, audit Auditor, ch *Channels) *Funnel {
	if ch == nil {
		ch = NewChannels(3600)
	}
	return &Funnel{Sink: sink, Audit: audit, Channels: ch, Now: now}
}

func (f *Funnel) Stats() Stats {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.stats
}

// Take — разобрать пакет, записать хорошие события, отметить каналы услышанными.
func (f *Funnel) Take(raw []any, via string, actor *Actor, requestID string) (TakeResult, error) {
	t := f.Now()
	good := make([]Event, 0, len(raw))
	bad := []Rejected{}
	for i, e := range raw {
		ev, err := Normalize(e)
		if err != nil {
			bad = append(bad, Rejected{i, err.Error()})
			continue
		}
		good = append(good, ev)
	}
	if len(bad) > 0 {
		f.Reject(bad, len(raw), via, actor, requestID, "")
	}
	if len(good) > 0 {
		msgs := make([]Message, len(good))
		for i, ev := range good {
			msgs[i] = Message{TopicOf(ev), strconv.FormatInt(ev.Channel, 10), ev}
		}
		if err := f.Sink.Send(msgs); err != nil {
			f.mu.Lock()
			f.stats.Unavailable++
			f.mu.Unlock()
			return TakeResult{}, err
		}
		f.keep(good, t)
	}
	var back []int64
	for _, ev := range good {
		if f.Channels.Seen(ev.Channel, t) {
			back = append(back, ev.Channel)
		}
	}
	if f.Channels.Packet(t) {
		slog.Info("шина снова на связи")
	}
	if len(back) > 0 {
		f.Status(back, "ok", t)
	}
	f.mu.Lock()
	f.stats.Packets++
	f.stats.Accepted += len(good)
	f.stats.Rejected += len(bad)
	if len(good) > 0 {
		s := iso(t)
		f.stats.LastAccepted = &s
	}
	f.mu.Unlock()
	if len(bad) > 100 {
		bad = bad[:100]
	}
	return TakeResult{len(good), bad}, nil
}

// keep — принятое в архив и в открытые окна «Логи». Ошибка архива приём не отменяет: Kafka уже
// подтвердила запись.
func (f *Funnel) keep(good []Event, t float64) {
	if f.Archive == nil && f.Hub == nil {
		return
	}
	var rows []Row
	if f.Archive != nil {
		var err error
		if rows, err = f.Archive.Put(good, t); err != nil {
			slog.Error("архив показаний: " + err.Error())
			f.mu.Lock()
			f.stats.ArchiveErrors++
			f.mu.Unlock()
		}
	} else {
		at := iso(t)
		rows = make([]Row, len(good))
		for i, ev := range good {
			rows[i] = Row{at, ev}
		}
	}
	if f.Hub != nil {
		f.Hub.Publish(rows)
	}
}

// Reject — telemetry.rejected: почему и сколько; значений датчиков в журнале нет (§6.3).
func (f *Funnel) Reject(bad []Rejected, total int, via string, actor *Actor, requestID, reason string) {
	reasons := []string{reason}
	rejected := total
	if len(bad) > 0 {
		set := map[string]bool{}
		for _, b := range bad {
			set[b.Reason] = true
		}
		reasons = reasons[:0]
		for r := range set {
			reasons = append(reasons, r)
		}
		sort.Strings(reasons)
		if len(reasons) > 5 {
			reasons = reasons[:5]
		}
		rejected = len(bad)
	}
	a := Actor{Kind: "service"}
	if actor != nil {
		a = *actor
	}
	err := safeAudit(f.Audit, AuditEvent{Type: "telemetry.rejected", Outcome: "denied", ActorKind: a.Kind,
		ActorID: a.Sub, RequestID: requestID, ObjectType: "packet",
		Details: map[string]any{"via": via, "events": total, "rejected": rejected, "reasons": reasons}})
	if err != nil {
		slog.Error("аудит telemetry.rejected не записан")
	}
}

// safeAudit — аудит не роняет приём: ни ошибка, ни паника в нём.
func safeAudit(a Auditor, e AuditEvent) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("аудит: %v", r)
		}
	}()
	return a.Event(e)
}

// Status — переход каналов в «молчит» или обратно: channel.status в компактный tf.ingest.reference.
func (f *Funnel) Status(channels []int64, state string, t float64) {
	msgs := make([]Message, 0, len(channels))
	for _, c := range channels {
		since := t
		if state == "silent" {
			if at, ok := f.Channels.SilentSince(c); ok {
				since = at
			}
		}
		body := orderedRow{{"kind", "channel.status"}, {"ид_канала_данных", c}, {"status", state},
			{"at", iso(t)}, {"since", iso(since)}}
		msgs = append(msgs, Message{TopicReference, fmt.Sprintf("channel-status:%d", c), body})
		if f.Hub != nil {
			f.Hub.PublishStatus(c, state, iso(t), iso(since))
		}
	}
	if err := f.Sink.Send(msgs); err != nil {
		var u *Unavailable
		if errors.As(err, &u) {
			slog.Warn(fmt.Sprintf("статус %d каналов не записан: Kafka недоступна", len(msgs)))
		}
	}
}

// Sweep — обход молчания: кто замолчал, тем channel.status silent.
func (f *Funnel) Sweep(t float64) []int64 {
	went, source := f.Channels.Sweep(t)
	if source {
		slog.Warn(fmt.Sprintf("шина молчит дольше %.0f мин", f.Channels.SilentAfter/60))
	}
	if len(went) > 0 {
		slog.Info(fmt.Sprintf("замолчали каналы: %d", len(went)))
		f.Status(went, "silent", t)
	}
	return went
}
