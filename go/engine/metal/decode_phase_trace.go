// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"time"

	"dappco.re/go/inference"
)

// decode_phase_trace.go is the production-path exposure of the pieceTiming
// diagnostic (piece_timing.go): it flips the process-global counters on for one
// generation and folds the accumulated GPU-piece / GPU-span totals into the
// engine-neutral inference.DecodePhaseBudget the shared engine.TextModel reads.
// Before this, the split was reachable only from test code poking the globals;
// `lem generate -trace` now prints it through inference.GenerateMetrics.

// BeginDecodePhaseTrace turns per-token phase timing on for the next generation
// on this session and returns a stop function producing the aggregate budget
// (engine.DecodePhaseTracer). It resets the pieceTiming counters, records the
// starting cache position + wall clock, and the returned closure reads the
// counters back, turns tracing off, and averages over the tokens generated
// between begin and stop. Tracing is a single-flight diagnostic — the counters
// it drives are process-global (see piece_timing.go), so a concurrent traced
// generation would corrupt the split; `generate -trace` runs one at a time.
func (s *ArchSession) BeginDecodePhaseTrace() func() inference.DecodePhaseBudget {
	pieceNs = [3]int64{}
	icbGPUNs = 0
	chainedGPUSpanNs = 0
	pieceTimingOn = true
	startPos := 0
	if s != nil {
		startPos = s.pos
	}
	start := time.Now()
	return func() inference.DecodePhaseBudget {
		wall := time.Since(start)
		pieceTimingOn = false
		tokens := 0
		if s != nil {
			tokens = s.pos - startPos
		}
		return buildDecodePhaseBudget(wall, tokens, pieceNs, chainedGPUSpanNs)
	}
}

// buildDecodePhaseBudget averages the raw pieceTiming totals over the tokens
// decoded during the trace. Which counters are populated depends on the decode
// path the model + config took: the chained / pipelined GPU tail (the e2b greedy
// default) accumulates one whole-token GPU execution span (chainedSpanNs); the
// step-body path accumulates the three GPU pieces (PLE projection, ICB layer
// stack, head). GPU-busy time is taken from whichever is present; each non-zero
// piece is reported as its own phase so the caller can see the split when it
// exists. A zero token count yields an empty budget (nothing was measured).
func buildDecodePhaseBudget(wall time.Duration, tokens int, pieces [3]int64, chainedSpanNs int64) inference.DecodePhaseBudget {
	if tokens <= 0 {
		return inference.DecodePhaseBudget{}
	}
	perToken := func(ns int64) time.Duration { return time.Duration(ns / int64(tokens)) }
	gpuNs := chainedSpanNs
	if gpuNs == 0 {
		gpuNs = pieces[0] + pieces[1] + pieces[2]
	}
	budget := inference.DecodePhaseBudget{
		Tokens:        tokens,
		TotalPerToken: wall / time.Duration(tokens),
		GPUPerToken:   perToken(gpuNs),
	}
	add := func(name string, ns int64) {
		if ns <= 0 {
			return
		}
		budget.Phases = append(budget.Phases, inference.DecodePhaseShare{Name: name, PerToken: perToken(ns), GPU: true})
	}
	add("PLE projection", pieces[0])
	add("layer stack (ICB)", pieces[1])
	add("head — norm + lm_head", pieces[2])
	add("chained GPU span", chainedSpanNs)
	return budget
}
