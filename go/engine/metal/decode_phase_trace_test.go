// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"
	"time"
)

// TestBuildDecodePhaseBudget_ChainedSpan_Good pins the default e2b path shape:
// only the chained-decode GPU span is populated (the pieces are zero), so GPU-busy
// time comes from the span and the single reported phase is the chained span.
func TestBuildDecodePhaseBudget_ChainedSpan_Good(t *testing.T) {
	wall := 20 * time.Millisecond // 10 tokens → 2ms/token wall
	budget := buildDecodePhaseBudget(wall, 10, [3]int64{}, int64(10*time.Millisecond))
	if budget.Tokens != 10 {
		t.Fatalf("Tokens = %d, want 10", budget.Tokens)
	}
	if budget.TotalPerToken != 2*time.Millisecond {
		t.Fatalf("TotalPerToken = %v, want 2ms", budget.TotalPerToken)
	}
	if budget.GPUPerToken != time.Millisecond {
		t.Fatalf("GPUPerToken = %v, want 1ms (span/tokens)", budget.GPUPerToken)
	}
	if len(budget.Phases) != 1 || budget.Phases[0].Name != "chained GPU span" || !budget.Phases[0].GPU {
		t.Fatalf("phases = %+v, want one GPU 'chained GPU span'", budget.Phases)
	}
}

// TestBuildDecodePhaseBudget_Pieces_Bad pins the step-path shape: no chained span,
// so GPU-busy time is the sum of the three GPU pieces and each non-zero piece is a
// reported phase (PLE, ICB layer stack, head).
func TestBuildDecodePhaseBudget_Pieces_Bad(t *testing.T) {
	// 4 tokens; pieces total 8ms → GPU 2ms/token.
	pieces := [3]int64{int64(2 * time.Millisecond), int64(4 * time.Millisecond), int64(2 * time.Millisecond)}
	budget := buildDecodePhaseBudget(12*time.Millisecond, 4, pieces, 0)
	if budget.GPUPerToken != 2*time.Millisecond {
		t.Fatalf("GPUPerToken = %v, want 2ms (piece sum/tokens)", budget.GPUPerToken)
	}
	if len(budget.Phases) != 3 {
		t.Fatalf("phases = %d, want 3 (PLE, ICB, head)", len(budget.Phases))
	}
	names := []string{"PLE projection", "layer stack (ICB)", "head — norm + lm_head"}
	for i, want := range names {
		if budget.Phases[i].Name != want {
			t.Fatalf("phase %d = %q, want %q", i, budget.Phases[i].Name, want)
		}
	}
}

// TestBuildDecodePhaseBudget_ZeroTokens_Ugly pins the guard: nothing decoded
// yields an empty budget (no divide-by-zero, no phantom phases).
func TestBuildDecodePhaseBudget_ZeroTokens_Ugly(t *testing.T) {
	budget := buildDecodePhaseBudget(time.Second, 0, [3]int64{1, 2, 3}, 4)
	if budget.Tokens != 0 || budget.TotalPerToken != 0 || len(budget.Phases) != 0 {
		t.Fatalf("zero-token budget = %+v, want empty", budget)
	}
}

// TestBeginDecodePhaseTrace_MeasuresPosDelta pins the production-path exposure:
// begin resets + arms the counters, the stop closure reads back the pos delta and
// the GPU span accumulated during the trace, and tracing is off afterwards.
func TestBeginDecodePhaseTrace_MeasuresPosDelta(t *testing.T) {
	oldOn, oldPieces, oldSpan := pieceTimingOn, pieceNs, chainedGPUSpanNs
	t.Cleanup(func() { pieceTimingOn, pieceNs, chainedGPUSpanNs = oldOn, oldPieces, oldSpan })

	s := &ArchSession{pos: 100}
	stop := s.BeginDecodePhaseTrace()
	if !pieceTimingOn {
		t.Fatal("BeginDecodePhaseTrace did not arm pieceTimingOn")
	}
	// Simulate a 6-token decode that accumulated a GPU span.
	s.pos = 106
	chainedGPUSpanNs = int64(6 * time.Millisecond)
	budget := stop()
	if pieceTimingOn {
		t.Fatal("stop did not disarm pieceTimingOn")
	}
	if budget.Tokens != 6 {
		t.Fatalf("Tokens = %d, want 6 (pos delta)", budget.Tokens)
	}
	if budget.GPUPerToken != time.Millisecond {
		t.Fatalf("GPUPerToken = %v, want 1ms", budget.GPUPerToken)
	}
}

// TestBeginDecodePhaseTrace_NilSessionSafe pins the nil-guard: a nil session
// still returns a working stop closure that yields an empty budget.
func TestBeginDecodePhaseTrace_NilSessionSafe(t *testing.T) {
	oldOn := pieceTimingOn
	t.Cleanup(func() { pieceTimingOn = oldOn })
	var s *ArchSession
	budget := s.BeginDecodePhaseTrace()()
	if budget.Tokens != 0 {
		t.Fatalf("nil-session budget Tokens = %d, want 0", budget.Tokens)
	}
}

// TestSupportedCacheModes pins the capability seam's honest answer: the no-cgo
// engine reports its single native cache mode (no go-mlx-era selector).
func TestSupportedCacheModes(t *testing.T) {
	m := &NativeTokenModel{}
	modes := m.SupportedCacheModes()
	if len(modes) != 1 || modes[0] != "native" {
		t.Fatalf("SupportedCacheModes = %v, want [native]", modes)
	}
}
