// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"
	"time"
)

// TestNativeICBNoBarrierCeiling measures the ICB replay's GPU span with EVERY barrier removed — a
// timing-only floor probe (the output is racy garbage, never verified). The gap between the real
// barriered span (~6.26ms/token) and this no-barrier span is the barrier-serialisation cost — i.e. the
// prize FUSION chases (fewer ops/layer → fewer barriers). If the floor is well under cgo's 5.9ms whole
// token, native beating cgo is on the table.
func TestNativeICBNoBarrierCeiling(t *testing.T) {
	dir := os.Getenv("NATIVE_BENCH_DIR")
	if dir == "" {
		t.Skip("set NATIVE_BENCH_DIR to a real gemma4 checkpoint dir")
	}
	allBarriersOffForTest = true // record the ICB with NO barriers (timing only; output garbage)
	defer func() { allBarriersOffForTest = false }()
	sess, err := LoadDir(dir, 1024)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	defer func() { _ = sess.Close() }()
	prompt := []int32{2, 1841, 689, 573, 6182, 576}
	if _, err := sess.Generate(prompt, 24, -1); err != nil {
		t.Fatalf("warmup: %v", err)
	}
	icbGPUNs = 0
	pieceTimingOn = true
	const N = 512
	start := time.Now()
	_, err = sess.Generate(prompt, N, -1)
	wall := time.Since(start)
	pieceTimingOn = false
	if err != nil {
		t.Fatalf("measure: %v", err)
	}
	t.Logf("NO-BARRIER ICB (timing only, output garbage): GPU span %v/token, %.1f tok/s",
		time.Duration(icbGPUNs/N), float64(N)/wall.Seconds())
	t.Logf("  vs barriered span 6.26ms/token — the difference is the barrier idle FUSION recovers (fewer ops → fewer barriers); cgo whole token = 5.9ms")
}
