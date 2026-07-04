// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"
	"time"
)

// TestNativeDecodePieceSplit measures where the per-token decode wall goes across the three GPU pieces — the
// PLE projection, the ICB layer stack, and the head (final norm + lm_head) — to settle (by measurement, not
// arithmetic) whether the remaining gap to cgo is the layer kernels or a specific op. Env-guarded.
func TestNativeDecodePieceSplit(t *testing.T) {
	dir := os.Getenv("NATIVE_BENCH_DIR")
	if dir == "" {
		t.Skip("set NATIVE_BENCH_DIR to a real gemma4 checkpoint dir")
	}
	sess, err := LoadDir(dir, 1024)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	defer func() { _ = sess.Close() }()
	prompt := []int32{2, 1841, 689, 573, 6182, 576}
	if _, err := sess.Generate(prompt, 24, -1); err != nil { // warmup (untimed)
		t.Fatalf("warmup: %v", err)
	}
	pieceNs = [3]int64{}
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
	tot := pieceNs[0] + pieceNs[1] + pieceNs[2]
	wns := float64(wall.Nanoseconds())
	pct := func(ns int64) float64 { return 100 * float64(ns) / wns }
	t.Logf("decode %d tokens in %v = %.1f tok/s", N, wall.Round(time.Millisecond), float64(N)/wall.Seconds())
	t.Logf("  PLE   %7v  %4.1f%% wall", time.Duration(pieceNs[0]).Round(time.Millisecond), pct(pieceNs[0]))
	t.Logf("  ICB   %7v  %4.1f%% wall   (the layer stack)", time.Duration(pieceNs[1]).Round(time.Millisecond), pct(pieceNs[1]))
	t.Logf("  head  %7v  %4.1f%% wall   (final norm + lm_head)", time.Duration(pieceNs[2]).Round(time.Millisecond), pct(pieceNs[2]))
	t.Logf("  GPU pieces %4.1f%% of wall; rest (embed dequant + host encode + sample) %4.1f%%", pct(tot), 100-pct(tot))
	t.Logf("  ICB GPU span %v/token (ICB wall %v/token; host submit/wait %v/token)",
		time.Duration(icbGPUNs/N), time.Duration(pieceNs[1]/N), time.Duration((pieceNs[1]-icbGPUNs)/N))
	t.Logf("  ⇒ ICB GPU %v/token vs cgo's WHOLE 5.9ms token — excess over kernel compute is barrier-serialisation idle",
		time.Duration(icbGPUNs/N))
}

func TestPieceTimingDisabledIsNoop(t *testing.T) {
	oldOn, oldNs := pieceTimingOn, pieceNs
	pieceTimingOn = false
	pieceNs = [3]int64{}
	t.Cleanup(func() {
		pieceTimingOn = oldOn
		pieceNs = oldNs
	})

	if got := ptStart(); !got.IsZero() {
		t.Fatalf("ptStart disabled = %v, want zero time", got)
	}
	ptEnd(0, time.Now().Add(-time.Millisecond))
	if pieceNs[0] != 0 {
		t.Fatalf("ptEnd disabled changed pieceNs[0] to %d", pieceNs[0])
	}
}

func TestPieceTimingEnabledAccumulates(t *testing.T) {
	oldOn, oldNs := pieceTimingOn, pieceNs
	pieceTimingOn = true
	pieceNs = [3]int64{}
	t.Cleanup(func() {
		pieceTimingOn = oldOn
		pieceNs = oldNs
	})

	if got := ptStart(); got.IsZero() {
		t.Fatal("ptStart enabled returned zero time")
	}
	ptEnd(1, time.Now().Add(-time.Millisecond))
	if pieceNs[1] <= 0 {
		t.Fatalf("ptEnd enabled pieceNs[1] = %d, want > 0", pieceNs[1])
	}
}
