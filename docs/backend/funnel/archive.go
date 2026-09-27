package main

// Архив показаний (окно «Логи»): всё, что воронка приняла, копится по часу в файл
// <каталог>/<ГГГГ-ММ-ДД>/<ЧЧ>.jsonl по времени приёма (МСК); закрытые часы сжимаются zstd в
// <ЧЧ>.jsonl.zst, дни старше TF_FUNNEL_ARCHIVE_DAYS удаляются. Показания в BFF не пишутся: словарь
// датчиков там, значения — здесь и в Kafka (модель читает Kafka, архив — для людей).

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/klauspost/compress/zstd"
)

// Row — строка архива: когда воронка приняла событие и само событие в формате журнала.
type Row struct {
	At string `json:"получено"`
	Event
}

// Reading — показание для фронта: sensorId — ид_канала_данных (он же id датчика в BFF).
type Reading struct {
	SensorID   int64  `json:"sensorId"`
	EventID    *int64 `json:"eventId"`
	Date       string `json:"date"`
	Time       string `json:"time"`
	Alarm      bool   `json:"alarm"`
	Value      string `json:"value"`
	ReceivedAt string `json:"receivedAt"`
}

func (r Row) Reading() Reading {
	return Reading{r.Channel, r.EventID, r.Date, r.Time, r.Alarm, r.Value, r.At}
}

var (
	rowAtPrefix   = []byte(`{"получено":"`)
	rowChanMarker = []byte(`"ид_канала_данных":`)
)

// rowKey — время приёма и канал строки архива без разбора JSON: строки пишет только Put, порядок
// полей известен. ok=false — строка чужая или битая.
func rowKey(line []byte) (at string, channel int64, ok bool) {
	if !bytes.HasPrefix(line, rowAtPrefix) || len(line) < len(rowAtPrefix)+25 {
		return "", 0, false
	}
	at = string(line[len(rowAtPrefix) : len(rowAtPrefix)+25])
	i := bytes.Index(line, rowChanMarker)
	if i < 0 {
		return "", 0, false
	}
	rest := line[i+len(rowChanMarker):]
	j := 0
	for j < len(rest) && rest[j] >= '0' && rest[j] <= '9' {
		j++
	}
	n, err := strconv.ParseInt(string(rest[:j]), 10, 64)
	return at, n, err == nil
}

type Archive struct {
	Dir      string
	KeepDays int     // 0 — хранить всё
	Grace    float64 // сколько секунд после конца часа файл ещё открыт для записи

	mu   sync.Mutex
	hour string // "2026-09-27/14" — открытый файл
	f    *os.File
}

func NewArchive(dir string, keepDays int) *Archive {
	return &Archive{Dir: dir, KeepDays: keepDays, Grace: 300}
}

// hourKey — "ГГГГ-ММ-ДД/ЧЧ" часа приёма по МСК.
func hourKey(t float64) string {
	sec, _ := math.Modf(t)
	return time.Unix(int64(sec), 0).In(msk).Format("2006-01-02/15")
}

// Put дописывает события в файл часа t. Ошибка записи не отменяет приём: Kafka уже подтвердила.
func (a *Archive) Put(evs []Event, t float64) ([]Row, error) {
	rows := make([]Row, len(evs))
	var buf bytes.Buffer
	at := iso(t)
	for i, ev := range evs {
		rows[i] = Row{at, ev}
		buf.Write(marshal(rows[i]))
		buf.WriteByte('\n')
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	key := hourKey(t)
	if a.f == nil || a.hour != key {
		if a.f != nil {
			a.f.Close()
			a.f = nil
		}
		path := filepath.Join(a.Dir, filepath.FromSlash(key)+".jsonl")
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			return rows, err
		}
		f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
		if err != nil {
			return rows, err
		}
		a.f, a.hour = f, key
	}
	_, err := a.f.Write(buf.Bytes())
	return rows, err
}

// Compact сжимает закрытые часы и удаляет старые дни; зовётся из обхода молчания.
func (a *Archive) Compact(t float64) (compressed int, err error) {
	open := hourKey(t - a.Grace)
	current := hourKey(t)
	a.mu.Lock()
	if a.f != nil && a.hour != open && a.hour != current {
		a.f.Close()
		a.f, a.hour = nil, ""
	}
	a.mu.Unlock()

	days, err := os.ReadDir(a.Dir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return 0, nil
		}
		return 0, err
	}
	sec, _ := math.Modf(t)
	oldest := ""
	if a.KeepDays > 0 {
		oldest = time.Unix(int64(sec), 0).In(msk).AddDate(0, 0, -a.KeepDays).Format("2006-01-02")
	}
	for _, d := range days {
		if !d.IsDir() || len(d.Name()) != 10 {
			continue
		}
		dir := filepath.Join(a.Dir, d.Name())
		if oldest != "" && d.Name() < oldest {
			if err := os.RemoveAll(dir); err != nil {
				slog.Error("архив: старый день не удалён: " + err.Error())
			}
			continue
		}
		files, _ := os.ReadDir(dir)
		for _, f := range files {
			name := f.Name()
			if !strings.HasSuffix(name, ".jsonl") {
				continue
			}
			key := d.Name() + "/" + strings.TrimSuffix(name, ".jsonl")
			if key == open || key == current {
				continue
			}
			if err := compressHour(filepath.Join(dir, name)); err != nil {
				slog.Error("архив: час " + key + " не сжат: " + err.Error())
				continue
			}
			compressed++
		}
	}
	return compressed, nil
}

// compressHour дописывает час кадром zstd в <час>.jsonl.zst (кадры склеиваются: если час уже сжимали,
// а потом в него дописали, новый кадр идёт следом) и удаляет несжатый файл.
func compressHour(raw string) error {
	in, err := os.Open(raw)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.OpenFile(raw+".zst", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	enc, err := zstd.NewWriter(out, zstd.WithEncoderLevel(zstd.SpeedBetterCompression))
	if err != nil {
		out.Close()
		return err
	}
	if _, err := io.Copy(enc, in); err != nil {
		enc.Close()
		out.Close()
		return err
	}
	if err := enc.Close(); err != nil {
		out.Close()
		return err
	}
	if err := out.Sync(); err != nil {
		out.Close()
		return err
	}
	if err := out.Close(); err != nil {
		return err
	}
	in.Close()
	return os.Remove(raw)
}

// scanHour — строки часа по порядку записи: сначала сжатые кадры, потом то, что дописано после.
func scanHour(base string, fn func(line []byte)) error {
	for _, path := range []string{base + ".jsonl.zst", base + ".jsonl"} {
		f, err := os.Open(path)
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				continue
			}
			return err
		}
		var r io.Reader = f
		var dec *zstd.Decoder
		if strings.HasSuffix(path, ".zst") {
			if dec, err = zstd.NewReader(f, zstd.WithDecoderConcurrency(1)); err != nil {
				f.Close()
				return err
			}
			r = dec
		}
		sc := bufio.NewScanner(r)
		sc.Buffer(make([]byte, 64<<10), 1<<20)
		for sc.Scan() {
			fn(sc.Bytes())
		}
		err = sc.Err()
		if dec != nil {
			dec.Close()
		}
		f.Close()
		if err != nil && !errors.Is(err, io.ErrUnexpectedEOF) {
			return err
		}
	}
	return nil
}

// Read — показания каналов за [from, to), новые сверху, не больше limit; просматривает не больше
// maxHours часов от to назад. more — упёрлись в limit или maxHours, раньше from ещё могут быть записи.
func (a *Archive) Read(channels map[int64]bool, from, to float64, limit, maxHours int) (out []Row, more bool, err error) {
	fromS, toS := iso(from), iso(to)
	firstHour := hourKey(from)
	sec, _ := math.Modf(to)
	h := time.Unix(int64(sec), 0).In(msk).Truncate(time.Hour)
	for n := 0; ; n++ {
		key := h.Format("2006-01-02/15")
		if key < firstHour {
			return out, false, nil
		}
		if n >= maxHours {
			return out, true, nil
		}
		var hour [][]byte
		err := scanHour(filepath.Join(a.Dir, filepath.FromSlash(key)), func(line []byte) {
			at, ch, ok := rowKey(line)
			if ok && channels[ch] && at >= fromS && at < toS {
				hour = append(hour, append([]byte(nil), line...))
			}
		})
		if err != nil {
			return out, false, fmt.Errorf("архив %s: %w", key, err)
		}
		// Внутри часа строки идут по приёму — для «новые сверху» берём с конца.
		for i := len(hour) - 1; i >= 0; i-- {
			var row Row
			if err := decode(hour[i], &row); err != nil {
				continue
			}
			out = append(out, row)
			if len(out) >= limit {
				return out, true, nil
			}
		}
		h = h.Add(-time.Hour)
	}
}

// Close закрывает открытый файл часа (остановка воронки).
func (a *Archive) Close() {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.f != nil {
		a.f.Close()
		a.f, a.hour = nil, ""
	}
}
