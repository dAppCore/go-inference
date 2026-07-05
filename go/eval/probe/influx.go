// SPDX-Licence-Identifier: EUPL-1.2

// The v0-schema metrics sink (#97): training events out, InfluxDB line
// protocol in the v0 LEM instrument's exact shape — the schema the lab
// dashboards already read (lthn/LEM pkg/lem, training_loss + content_score
// measurements). The native loop emits through this sink and the existing
// hot store works unchanged: train and val loss on one iteration clock,
// the lem-scorer's quality readings beside them, so loss-amplitude
// patterns and any corresponding quality patterns sit on one screen.
//
// Two destinations, both optional, both best-effort beyond the format:
//   - FilePath: every line appended as it happens — the durable cold copy,
//     tail-able mid-run, shippable later (`lem ingest` style).
//   - Post: batches of lines handed to a poster (NewInfluxPoster builds
//     the InfluxDB HTTP one). A post failure NEVER interrupts training —
//     the dashboard is an observer, not a dependency; failures count in
//     Dropped().

package probe

import (
	"net/http"
	"sort"
	"sync"
	"time"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

// LineProtocolConfig configures a LineProtocolSink.
type LineProtocolConfig struct {
	Model string // tag: the model being trained (required for useful lines)
	RunID string // tag: this run's identity on the dashboard

	FilePath string // append every line here; "" disables the file copy
	// Post receives newline-joined batches of lines. Use NewInfluxPoster
	// for the standard InfluxDB write endpoint; "" Post + "" FilePath
	// makes the sink format-and-drop (still countable via Lines()).
	Post       func(body string) error
	BatchLines int // lines per Post flush (default 20)

	// now is the clock for iterations_per_sec / tokens_per_sec; tests
	// inject a fixed-step clock. Defaults to time.Now.
	now func() time.Time
}

// LineProtocolSink converts probe events to v0-schema InfluxDB line
// protocol. Implements Sink; safe for concurrent emitters.
type LineProtocolSink struct {
	cfg LineProtocolConfig

	mu          sync.Mutex
	pending     []string
	lines       int
	dropped     int
	lastTrainAt time.Time
}

// NewLineProtocolSink builds the sink. Always usable — destinations are
// optional and a nil receiver no-ops, mirroring the probe-sink contract.
func NewLineProtocolSink(cfg LineProtocolConfig) *LineProtocolSink {
	if cfg.BatchLines <= 0 {
		cfg.BatchLines = 20
	}
	if cfg.now == nil {
		cfg.now = time.Now
	}
	return &LineProtocolSink{cfg: cfg}
}

// NewInfluxPoster returns a Post func for the InfluxDB write endpoint —
// url is the full write URL (org/bucket/precision in the query string),
// token the API token ("" sends no auth header).
//
//	sink := probe.NewLineProtocolSink(probe.LineProtocolConfig{
//		Model: "LEM-gemma3-1b", RunID: "gold-1",
//		FilePath: "~/Lethean/lem/sft/gold-1/metrics.lp",
//		Post: probe.NewInfluxPoster("http://localhost:8086/api/v2/write?org=lem&bucket=training", token),
//	})
func NewInfluxPoster(url, token string) func(body string) error {
	client := &http.Client{Timeout: 10 * time.Second}
	return func(body string) error {
		res := core.NewHTTPRequest("POST", url, core.NewReader(body))
		if !res.OK {
			return res.Value.(error)
		}
		req := res.Value.(*core.Request)
		req.Header.Set("Content-Type", "text/plain; charset=utf-8")
		if token != "" {
			req.Header.Set("Authorization", "Token "+token)
		}
		resp, err := client.Do(req)
		if err != nil {
			return err
		}
		defer resp.Body.Close()
		if resp.StatusCode >= 300 {
			return core.NewError(core.Sprintf("influx write: HTTP %d", resp.StatusCode))
		}
		return nil
	}
}

// EmitProbe converts one event. Training events become training_loss
// lines; score events become content_score lines; everything else is
// ignored — this sink is the training-run instrument, not a firehose.
func (s *LineProtocolSink) EmitProbe(event Event) {
	if s == nil {
		return
	}
	switch event.Kind {
	case KindTraining:
		if event.Training == nil {
			return
		}
		s.emitTraining(event.Training)
	case KindScore:
		if event.Score == nil {
			return
		}
		s.emitScore(event.Step, event.Score)
	}
}

func (s *LineProtocolSink) emitTraining(t *Training) {
	s.mu.Lock()
	defer s.mu.Unlock()
	switch t.LossType {
	case LossTypeVal:
		// v0: training_loss,...,loss_type=val loss= iteration=
		s.add(core.Sprintf("training_loss,model=%s,run_id=%s,loss_type=val loss=%f,iteration=%di",
			escapeLp(s.cfg.Model), escapeLp(s.cfg.RunID), t.Loss, t.Step))
	default:
		// v0: the train line carries the run's throughput. Rates derive
		// from the wall time between train events — one event per
		// optimizer step — so the loop never clocks itself. The first
		// step has no interval and reads 0.
		now := s.cfg.now()
		itersPerSec := 0.0
		tokensPerSec := 0.0
		if !s.lastTrainAt.IsZero() {
			if dt := now.Sub(s.lastTrainAt).Seconds(); dt > 0 {
				itersPerSec = 1.0 / dt
				tokensPerSec = float64(t.Tokens) / dt
			}
		}
		s.lastTrainAt = now
		s.add(core.Sprintf("training_loss,model=%s,run_id=%s,loss_type=train loss=%f,learning_rate=%f,iterations_per_sec=%f,tokens_per_sec=%f,iteration=%di",
			escapeLp(s.cfg.Model), escapeLp(s.cfg.RunID), t.Loss, t.LearningRate, itersPerSec, tokensPerSec, t.Step))
	}
}

func (s *LineProtocolSink) emitScore(step int, score *Score) {
	if len(score.Values) == 0 {
		return
	}
	label := score.Label
	if label == "" {
		label = "unknown"
	}
	hasKernel := core.Contains(core.Lower(label), "kernel")
	dims := make([]string, 0, len(score.Values))
	for dim := range score.Values {
		dims = append(dims, dim)
	}
	sort.Strings(dims)

	s.mu.Lock()
	defer s.mu.Unlock()
	for _, dim := range dims {
		s.add(core.Sprintf("content_score,model=%s,run_id=%s,label=%s,dimension=%s,has_kernel=%t score=%f,iteration=%di",
			escapeLp(s.cfg.Model), escapeLp(s.cfg.RunID), escapeLp(label), escapeLp(dim), hasKernel, score.Values[dim], step))
	}
}

// add records one finished line: appended to the file immediately
// (durable, tail-able), queued for the poster, flushed on batch size.
// Caller holds s.mu.
func (s *LineProtocolSink) add(line string) {
	s.lines++
	if s.cfg.FilePath != "" {
		if w, err := coreio.Local.Append(s.cfg.FilePath); err == nil {
			_, writeErr := w.Write([]byte(line + "\n"))
			if closeErr := w.Close(); writeErr != nil || closeErr != nil {
				// Append opened but the line did not land durably (write or
				// clean close failed) — count it dropped, same as an open failure.
				s.dropped++
			}
		} else {
			s.dropped++
		}
	}
	if s.cfg.Post == nil {
		return
	}
	s.pending = append(s.pending, line)
	if len(s.pending) >= s.cfg.BatchLines {
		s.flushLocked()
	}
}

func (s *LineProtocolSink) flushLocked() {
	if len(s.pending) == 0 || s.cfg.Post == nil {
		return
	}
	body := core.Join("\n", s.pending...)
	count := len(s.pending)
	s.pending = s.pending[:0]
	if err := s.cfg.Post(body); err != nil {
		s.dropped += count
	}
}

// Flush posts any pending lines now.
func (s *LineProtocolSink) Flush() {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.flushLocked()
}

// Close flushes; the sink stays usable after Close (it holds no
// descriptors open between emits).
func (s *LineProtocolSink) Close() { s.Flush() }

// Lines reports how many lines the sink has formatted.
func (s *LineProtocolSink) Lines() int {
	if s == nil {
		return 0
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lines
}

// Dropped reports lines that failed a destination (file append or post) —
// the honesty counter: the run never stops for the dashboard, but the gap
// is never silent either.
func (s *LineProtocolSink) Dropped() int {
	if s == nil {
		return 0
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.dropped
}

// escapeLp escapes tag values per InfluxDB line protocol — the v0
// instrument's escaping, verbatim.
func escapeLp(s string) string {
	s = core.Replace(s, `,`, `\,`)
	s = core.Replace(s, `=`, `\=`)
	s = core.Replace(s, ` `, `\ `)
	return s
}
