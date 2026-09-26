package main

import (
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"strings"
	"time"
)

// Stopper — остановка цикла забора: Wait спит до d или до остановки и говорит, остановлены ли.
type Stopper interface {
	IsSet() bool
	Wait(d float64) bool
}

// stopChan — Stopper на закрываемом канале.
type stopChan chan struct{}

func (s stopChan) IsSet() bool {
	select {
	case <-s:
		return true
	default:
		return false
	}
}

func (s stopChan) Wait(d float64) bool {
	select {
	case <-s:
		return true
	case <-time.After(time.Duration(d * float64(time.Second))):
		return false
	}
}

// Getter — GET и разбор JSON (в тестах — подмена).
type Getter func(url string) (map[string]any, error)

var pullClient = &http.Client{Timeout: 10 * time.Second}

func fetch(url string) (map[string]any, error) {
	resp, err := pullClient.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 256<<20))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("эмулятор ответил %d", resp.StatusCode)
	}
	var out map[string]any
	if err := decode(body, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// Pull забирает поток эмулятора по курсору. Перезапуск эмулятора (его курсор меньше нашего) — читать
// сначала: повтор модель отбросит по первичному ключу. Kafka лежит — курсор стоит.
func Pull(f *Funnel, base string, stop Stopper, interval float64, limit int, get Getter) {
	if get == nil {
		get = fetch
	}
	var cursor int64
	idle, wait := 0.0, interval
	base = strings.TrimRight(base, "/")
	for !stop.IsSet() {
		full := false
		err := func() error {
			data, err := get(fmt.Sprintf("%s/events?cursor=%d&limit=%d", base, cursor, limit))
			if err != nil {
				return err
			}
			rows, _ := data["события"].([]any)
			if len(rows) > 0 {
				if _, err := f.Take(rows, "pull", nil, ""); err != nil {
					return err
				}
				if c, ok := toInt(data["курсор"]); ok {
					cursor = c
				}
				idle = 0
				full = len(rows) >= limit // полная страница — сразу за следующей
			} else {
				idle += wait
				if idle >= 30 {
					h, err := get(base + "/health")
					if err != nil {
						return err
					}
					if c, ok := toInt(h["курсор"]); ok && c < cursor {
						slog.Warn("эмулятор перезапущен — читаю поток сначала")
						cursor = 0
					}
					idle = 0
				}
			}
			return nil
		}()
		var u *Unavailable
		switch {
		case err == nil:
			wait = interval
		case errors.As(err, &u):
			wait = math.Min(wait*2, 60) // Kafka лежит — курсор не двигаем
		default:
			slog.Warn(fmt.Sprintf("эмулятор недоступен (%T) — снова через %.0f с", err, wait))
			wait = math.Min(wait*2, 60)
		}
		if !full {
			stop.Wait(wait)
		}
	}
}
