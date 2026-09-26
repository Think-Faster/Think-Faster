package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"
	"unicode"
)

const (
	TopicReadings  = "tf.ingest.readings"
	TopicJournal   = "tf.ingest.journal"
	TopicReference = "tf.ingest.reference"
	Scope          = "telemetry.push"
	MaxEvents      = 10_000 // как limit у эмулятора; сообщение Kafka — одно событие
	MaxValue       = 255
)

// Event — событие в формате журнала. Порядок полей — как в журнале и в сообщении Kafka.
type Event struct {
	EventID *int64 `json:"ид_события"`
	Channel int64  `json:"ид_канала_данных"`
	Date    string `json:"дата"`
	Time    string `json:"время"`
	Alarm   bool   `json:"тревожное"`
	Value   string `json:"значение_датчика"`
}

// decode — JSON с числами как json.Number: номер канала и значение датчика не теряют знаков.
func decode(data []byte, v any) error {
	d := json.NewDecoder(bytes.NewReader(data))
	d.UseNumber()
	if err := d.Decode(v); err != nil {
		return err
	}
	if d.More() {
		return errors.New("после JSON лишние данные")
	}
	return nil
}

// marshal — JSON без экранирования <, > и & (как json.dumps с ensure_ascii=False).
func marshal(v any) []byte {
	var b bytes.Buffer
	e := json.NewEncoder(&b)
	e.SetEscapeHTML(false)
	_ = e.Encode(v)
	return bytes.TrimRight(b.Bytes(), "\n")
}

// toInt — целое, как int() в Python: число (дробное отбрасывается) или строка из цифр.
func toInt(v any) (int64, bool) {
	switch x := v.(type) {
	case json.Number:
		if n, err := x.Int64(); err == nil {
			return n, true
		}
		f, err := x.Float64()
		if err != nil || math.IsInf(f, 0) || math.IsNaN(f) || math.Abs(f) > math.MaxInt64 {
			return 0, false
		}
		return int64(f), true
	case string:
		n, err := strconv.ParseInt(strings.TrimSpace(x), 10, 64)
		return n, err == nil
	}
	return 0, false
}

// Normalize — событие в формате журнала или ошибка с причиной (без значения датчика).
func Normalize(raw any) (Event, error) {
	e, ok := raw.(map[string]any)
	if !ok {
		return Event{}, errors.New("событие не объект")
	}
	rc, has := e["ид_канала_данных"]
	if !has {
		return Event{}, errors.New("нет ид_канала_данных")
	}
	channel, ok := toInt(rc)
	if !ok || channel <= 0 {
		return Event{}, errors.New("ид_канала_данных не число")
	}
	rd, hasD := e["дата"]
	rt, hasT := e["время"]
	if !hasD || !hasT {
		return Event{}, errors.New("нет даты или времени")
	}
	date, okD := rd.(string)
	clock, okT := rt.(string)
	if !okD || !okT {
		return Event{}, errors.New("дата или время не в формате ГГГГ-ММ-ДД ЧЧ:ММ:СС")
	}
	// Как strptime('%Y-%m-%d %H:%M:%S'): месяц, день и время — одна или две цифры.
	if _, err := time.Parse("2006-1-2 15:4:5", date+" "+clock); err != nil {
		return Event{}, errors.New("дата или время не в формате ГГГГ-ММ-ДД ЧЧ:ММ:СС")
	}
	alarm := false
	if ra, has := e["тревожное"]; has {
		switch a := ra.(type) {
		case bool:
			alarm = a
		case string:
			switch strings.ToLower(strings.TrimSpace(a)) {
			case "true", "t", "1":
				alarm = true
			case "false", "f", "0":
			default:
				return Event{}, errors.New("тревожное не true/false")
			}
		default:
			return Event{}, errors.New("тревожное не true/false")
		}
	}
	var value string
	switch v := e["значение_датчика"].(type) {
	case string:
		value = v
	case json.Number:
		value = pyNumber(v)
	default:
		return Event{}, errors.New("нет значения датчика")
	}
	if strings.TrimSpace(value) == "" || len([]rune(value)) > MaxValue {
		return Event{}, fmt.Errorf("значение датчика пустое или длиннее %d", MaxValue)
	}
	out := Event{Channel: channel, Date: date, Time: clock, Alarm: alarm, Value: value}
	if rid, has := e["ид_события"]; has && rid != nil {
		id, ok := toInt(rid)
		if !ok {
			return Event{}, errors.New("ид_события не число")
		}
		out.EventID = &id
	}
	return out, nil
}

// pyNumber — число JSON так, как его напечатал бы str() в Python: 12 → "12", 1.50 → "1.5".
func pyNumber(n json.Number) string {
	s := n.String()
	if !strings.ContainsAny(s, ".eE") {
		if i, err := strconv.ParseInt(s, 10, 64); err == nil {
			return strconv.FormatInt(i, 10)
		}
		return s
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil && !math.IsInf(f, 0) {
		return s
	}
	return pyRepr(f)
}

// pyRepr — repr(float) в Python: кратчайшая запись, экспонента при порядке < -4 или >= 16.
func pyRepr(f float64) string {
	switch {
	case math.IsInf(f, 1):
		return "inf"
	case math.IsInf(f, -1):
		return "-inf"
	case math.IsNaN(f):
		return "nan"
	}
	e := strconv.FormatFloat(f, 'e', -1, 64) // -1.2345e+06
	mant, exps, _ := strings.Cut(e, "e")
	exp, _ := strconv.Atoi(exps)
	if exp < -4 || exp >= 16 {
		sign := "+"
		if exp < 0 {
			sign, exp = "-", -exp
		}
		return fmt.Sprintf("%se%s%02d", mant, sign, exp)
	}
	s := strconv.FormatFloat(f, 'f', -1, 64)
	if !strings.Contains(s, ".") {
		s += ".0"
	}
	return s
}

// Numeric — как модель (storage.clean_event): число — то, что читает float() и что конечно.
func Numeric(value string) bool {
	s := strings.TrimFunc(value, unicode.IsSpace)
	if s == "" || strings.ContainsAny(s, "xXpP") { // у Go шестнадцатеричные числа, у float() их нет
		return false
	}
	if strings.Contains(s, "_") { // float() пропускает «_» только между цифрами
		r := []rune(s)
		for i, c := range r {
			if c == '_' && (i == 0 || i == len(r)-1 || !unicode.IsDigit(r[i-1]) || !unicode.IsDigit(r[i+1])) {
				return false
			}
		}
		s = strings.ReplaceAll(s, "_", "")
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil && !errors.Is(err, strconv.ErrRange) {
		return false
	}
	return !math.IsInf(f, 0) && !math.IsNaN(f)
}

// TopicOf — тревожное или текстовое состояние идёт в журнал, число — в показания.
func TopicOf(ev Event) string {
	if ev.Alarm || !Numeric(ev.Value) {
		return TopicJournal
	}
	return TopicReadings
}

// EventsOf — пакет: {"events": [...]}, голый список или ответ эмулятора {"события": [...]}.
func EventsOf(body any) ([]any, error) {
	if m, ok := body.(map[string]any); ok {
		if v, has := m["events"]; has {
			body = v
		} else {
			body = m["события"]
		}
	}
	list, ok := body.([]any)
	if !ok {
		return nil, errors.New("пакет — список событий в поле events")
	}
	if len(list) > MaxEvents {
		return nil, fmt.Errorf("в пакете %d событий, больше %d нельзя", len(list), MaxEvents)
	}
	return list, nil
}
