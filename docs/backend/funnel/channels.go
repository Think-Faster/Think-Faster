package main

import (
	"math"
	"sort"
	"sync"
	"time"
)

var msk = time.FixedZone("MSK", 3*3600)

// iso — момент в секундах эпохи как 2026-09-26T03:09:27+03:00.
func iso(ts float64) string {
	sec, frac := math.Modf(ts)
	return time.Unix(int64(sec), int64(frac*1e9)).In(msk).Format("2006-01-02T15:04:05-07:00")
}

func now() float64 { return float64(time.Now().UnixNano()) / 1e9 }

// Channels — молчание каналов и шины. Время — секунды эпохи.
type Channels struct {
	SilentAfter  float64
	mu           sync.Mutex
	last         map[int64]float64
	gap          map[int64]float64 // обычный интервал канала, сглаженный
	count        map[int64]int
	silent       map[int64]float64 // канал → с какого времени молчит
	sourceLast   float64
	sourceSilent bool
}

func NewChannels(silentAfter float64) *Channels {
	return &Channels{SilentAfter: silentAfter, last: map[int64]float64{}, gap: map[int64]float64{},
		count: map[int64]int{}, silent: map[int64]float64{}}
}

// Seen отмечает событие. true — канал молчал и снова заговорил.
func (c *Channels) Seen(channel int64, at float64) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	prev, had := c.last[channel]
	_, back := c.silent[channel]
	delete(c.silent, channel)
	if had && at > prev && !back { // само молчание в интервал не входит
		d := at - prev
		if g, ok := c.gap[channel]; ok {
			c.gap[channel] = 0.8*g + 0.2*d
		} else {
			c.gap[channel] = d
		}
	}
	c.last[channel] = math.Max(at, prev)
	c.count[channel]++
	return back
}

// Packet отмечает пакет. true — шина молчала и снова заговорила.
func (c *Channels) Packet(at float64) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	back := c.sourceSilent
	c.sourceLast, c.sourceSilent = at, false
	return back
}

func (c *Channels) limit(channel int64) float64 {
	return math.Max(c.SilentAfter, 4*c.gap[channel])
}

// Sweep — кто замолчал с прошлого обхода; второе — замолчала ли шина целиком.
func (c *Channels) Sweep(now float64) ([]int64, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	went := []int64{}
	for ch, at := range c.last {
		if _, s := c.silent[ch]; !s && c.count[ch] >= 3 && now-at > c.limit(ch) {
			c.silent[ch] = at
			went = append(went, ch)
		}
	}
	sort.Slice(went, func(i, j int) bool { return went[i] < went[j] })
	sourceWent := c.sourceLast != 0 && !c.sourceSilent && now-c.sourceLast > c.SilentAfter
	if sourceWent {
		c.sourceSilent = true
	}
	return went, sourceWent
}

// SilentSince — с какого времени молчит канал (для channel.status).
func (c *Channels) SilentSince(channel int64) (float64, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	at, ok := c.silent[channel]
	return at, ok
}

func (c *Channels) SourceSilent() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.sourceSilent
}

type silentChannel struct {
	Channel int64  `json:"channel"`
	Since   string `json:"since"`
}

type channelsSnapshot struct {
	Channels       int             `json:"channels"`
	Silent         int             `json:"silent"`
	SilentChannels []silentChannel `json:"silent_channels"`
	SourceSilent   bool            `json:"source_silent"`
	SourceLast     *string         `json:"source_last"`
}

func (c *Channels) Snapshot(limit int) channelsSnapshot {
	c.mu.Lock()
	defer c.mu.Unlock()
	list := make([]silentChannel, 0, len(c.silent))
	type pair struct {
		ch int64
		at float64
	}
	ps := make([]pair, 0, len(c.silent))
	for ch, at := range c.silent {
		ps = append(ps, pair{ch, at})
	}
	sort.Slice(ps, func(i, j int) bool {
		if ps[i].at != ps[j].at {
			return ps[i].at < ps[j].at
		}
		return ps[i].ch < ps[j].ch
	})
	for i, p := range ps {
		if i >= limit {
			break
		}
		list = append(list, silentChannel{p.ch, iso(p.at)})
	}
	s := channelsSnapshot{Channels: len(c.last), Silent: len(ps), SilentChannels: list, SourceSilent: c.sourceSilent}
	if c.sourceLast != 0 {
		v := iso(c.sourceLast)
		s.SourceLast = &v
	}
	return s
}
