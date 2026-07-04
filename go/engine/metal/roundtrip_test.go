// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"
	"time"
)

// TestGPURoundTripLatency measures the per-op Commit()+WaitUntilCompleted() round-trip on a TRIVIAL op
// (256-elem AddBF16 — negligible GPU compute), isolating the synchronous dispatch latency. The native
// decode does one such round-trip per op (~11/token outside the ICB: embed + ~6 PLE ops + head), with the
// GPU idle between. If round-trip × 11 ≈ the per-token wall, the 95→169 gap is host synchronisation, not
// GPU compute — and batching ops into one command buffer (async, like mlx) is the lever.
func TestGPURoundTripLatency(t *testing.T) {
	if err := ensureInit(); err != nil {
		t.Fatal(err)
	}
	a := make([]byte, 256*bf16Size)
	b := make([]byte, 256*bf16Size)
	for i := 0; i < 50; i++ { // warmup (shader compile, queue warm)
		if _, err := AddBF16(a, b); err != nil {
			t.Fatal(err)
		}
	}
	const N = 2000
	start := time.Now()
	for i := 0; i < N; i++ {
		if _, err := AddBF16(a, b); err != nil {
			t.Fatal(err)
		}
	}
	per := time.Since(start) / N
	t.Logf("per-op Commit+WaitUntilCompleted round-trip: %v", per)
	t.Logf("  ×11 ops/token ≈ %v/token  (95.4 tok/s = 10.5ms/token; cgo 169 = 5.9ms/token)", per*11)
}
