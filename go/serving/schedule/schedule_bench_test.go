// SPDX-Licence-Identifier: EUPL-1.2

// Allocation contract for the continuous-batching scheduler (AX-11). Run owns
// the working buffers (a defensive queue copy, the running set, the results) and
// accumulates each sequence's generated tokens. The fake Stepper reuses its
// result maps (cleared each step) so each bench line isolates the scheduler's
// OWN buffering from the decode step's map churn.
//
// Run: go test -bench=. -benchmem -run='^$' ./schedule/
package schedule

import (
	"context"
	"testing"

	core "dappco.re/go"
)

// benchStepper emits one token per running sequence per step and finishes each
// at stop tokens. It reuses its result maps so the stepper allocates nothing
// per step — the scheduler's own allocations are what the bench measures.
type benchStepper struct {
	stop     int
	tokens   map[string]int
	finished map[string]bool
}

func newBenchStepper(stop, capHint int) *benchStepper {
	return &benchStepper{
		stop:     stop,
		tokens:   make(map[string]int, capHint),
		finished: make(map[string]bool, capHint),
	}
}

func (s *benchStepper) Step(_ context.Context, running []*Seq) (StepResult, error) {
	clear(s.tokens)
	clear(s.finished)
	for _, sq := range running {
		s.tokens[sq.Request.ID] = 1
		if sq.Generated+1 >= s.stop {
			s.finished[sq.Request.ID] = true
		}
	}
	return StepResult{Tokens: s.tokens, Finished: s.finished}, nil
}

var (
	benchResults []Result
	benchRunErr  error
)

func benchRequests(n, promptTokens, maxNew int) []Request {
	reqs := make([]Request, n)
	for i := range reqs {
		reqs[i] = Request{ID: core.Sprintf("req-%d", i), PromptTokens: promptTokens, MaxNewTokens: maxNew}
	}
	return reqs
}

func BenchmarkEngine_Run(b *testing.B) {
	const (
		n      = 32   // requests
		prompt = 64   // prompt tokens each
		maxNew = 64   // generated tokens each
		capN   = 8    // concurrency cap
		budget = 8192 // running token budget
	)
	ctx := context.Background()
	reqs := benchRequests(n, prompt, maxNew)
	stepper := newBenchStepper(maxNew, capN)
	e := New(Scheduler{MaxConcurrency: capN, MaxBatchTokens: budget})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r, err := e.Run(ctx, reqs, stepper, nil)
		if err != nil {
			b.Fatal(err)
		}
		benchResults = r
	}
}
