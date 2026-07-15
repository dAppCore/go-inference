// SPDX-Licence-Identifier: EUPL-1.2

package scheduler

import (
	"context"
	"iter"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// serialProbeModel is a single-session base model double: it declares
// inference.SerialModel and PANICS the instant a second generation enters
// concurrently — standing in for the MTP speculative pair, whose shared GPU
// scratch a concurrent lane SIGSEGVs (loadKV nil drafter-KV, #1842). A brief
// sleep widens the entry window so an unserialised scheduler reliably races it.
// It carries the full inference.TextModel surface plus CancelRequest so the
// wrapper-transparency test has a second optional capability to reach through
// inference.As.
type serialProbeModel struct {
	tokens      int
	inFlight    int32
	maxInFlight int32
}

func (m *serialProbeModel) SerialGeneration() bool { return true }

func (m *serialProbeModel) run() iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		n := atomic.AddInt32(&m.inFlight, 1)
		defer atomic.AddInt32(&m.inFlight, -1)
		for {
			hi := atomic.LoadInt32(&m.maxInFlight)
			if n <= hi || atomic.CompareAndSwapInt32(&m.maxInFlight, hi, n) {
				break
			}
		}
		if n > 1 {
			// Mimic the production failure: two concurrent generations raced the
			// single-session pair and the process died. An unserialised scheduler
			// crashes here; a serialised one never sees n > 1.
			panic("serialProbeModel: concurrent generation entered a single-session model (the #1842 SIGSEGV)")
		}
		time.Sleep(2 * time.Millisecond)
		for i := 0; i < m.tokens; i++ {
			if !yield(inference.Token{Text: "t" + strconv.Itoa(i)}) {
				return
			}
		}
	}
}

func (m *serialProbeModel) Generate(_ context.Context, _ string, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.run()
}

func (m *serialProbeModel) Chat(_ context.Context, _ []inference.Message, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.run()
}

func (m *serialProbeModel) Classify(_ context.Context, _ []string, _ ...inference.GenerateOption) core.Result {
	return core.Ok([]inference.ClassifyResult(nil))
}

func (m *serialProbeModel) BatchGenerate(_ context.Context, _ []string, _ ...inference.GenerateOption) core.Result {
	return core.Ok([]inference.BatchResult(nil))
}

func (m *serialProbeModel) ModelType() string { return "serial-probe" }
func (m *serialProbeModel) Info() inference.ModelInfo {
	return inference.ModelInfo{Architecture: "serial-probe"}
}
func (m *serialProbeModel) Metrics() inference.GenerateMetrics { return inference.GenerateMetrics{} }
func (m *serialProbeModel) Err() core.Result                   { return core.Ok(nil) }
func (m *serialProbeModel) Close() core.Result                 { return core.Ok(nil) }

func (m *serialProbeModel) CancelRequest(_ context.Context, id string) (inference.RequestCancelResult, error) {
	return inference.RequestCancelResult{ID: id, Cancelled: id != ""}, nil
}

var (
	_ inference.TextModel        = (*serialProbeModel)(nil)
	_ inference.SerialModel      = (*serialProbeModel)(nil)
	_ inference.CancellableModel = (*serialProbeModel)(nil)
)

// TestSerialModel_ConcurrentSchedule_Serialises is the #1842 regression: an
// interleave scheduler with 8-way concurrency, driving a single-session base
// model, must complete every request WITHOUT ever running two generations at
// once. Against a scheduler that does not serialise SerialModels the probe
// panics (the process-death the ticket describes); with the wrap in New every
// lane completes and the observed peak concurrency is exactly one.
func TestSerialModel_ConcurrentSchedule_Serialises(t *testing.T) {
	const lanes, tokensPerLane = 8, 3
	base := &serialProbeModel{tokens: tokensPerLane}
	scheduled, err := New(base, Config{Mode: ModeInterleave, MaxConcurrent: lanes, MaxQueue: lanes, StreamBuffer: tokensPerLane + 1})
	if err != nil {
		t.Fatalf("New(interleave) error = %v", err)
	}
	defer scheduled.CloseEngine()

	var wg sync.WaitGroup
	counts := make([]int, lanes)
	for lane := range lanes {
		_, tokens, schedErr := scheduled.Schedule(context.Background(), inference.ScheduledRequest{
			ID:     "lane-" + strconv.Itoa(lane),
			Prompt: "go",
		})
		if schedErr != nil {
			t.Fatalf("Schedule(lane %d) error = %v", lane, schedErr)
		}
		wg.Add(1)
		go func(lane int, ch <-chan inference.ScheduledToken) {
			defer wg.Done()
			counts[lane] = len(drainScheduled(t, ch))
		}(lane, tokens)
	}
	wg.Wait()

	for lane, got := range counts {
		if got != tokensPerLane {
			t.Errorf("lane %d produced %d tokens, want %d — a lane did not complete", lane, got, tokensPerLane)
		}
	}
	if peak := atomic.LoadInt32(&base.maxInFlight); peak != 1 {
		t.Fatalf("peak concurrent generations = %d, want 1 — the single-session lane was not serialised", peak)
	}
}

// TestSerialModel_Unwrap_ReachesBaseCapabilities guards the capability-stripping
// bug class: the serialising wrapper must stay transparent to inference.As, so
// the base model's SerialModel declaration AND its other optional capabilities
// (here CancellableModel) remain reachable through the wrapper's Unwrap.
func TestSerialModel_Unwrap_ReachesBaseCapabilities(t *testing.T) {
	base := &serialProbeModel{tokens: 1}
	wrapped := newSerialModel(base)

	if sm, ok := inference.As[inference.SerialModel](wrapped); !ok || !sm.SerialGeneration() {
		t.Fatalf("inference.As[SerialModel](wrapped) = (%v, %v), want a true SerialGeneration through the wrapper", sm, ok)
	}
	if _, ok := inference.As[inference.CancellableModel](wrapped); !ok {
		t.Fatalf("inference.As[CancellableModel](wrapped) = false — the wrapper stripped a base capability")
	}
	if got := inference.BaseTextModel(wrapped); got != base {
		t.Fatalf("BaseTextModel(wrapped) = %T, want the underlying *serialProbeModel", got)
	}
}
