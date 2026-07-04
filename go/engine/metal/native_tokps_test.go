// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"
	"time"
)

// TestNativeDecodeTokPerSec measures the pkg/native (no-cgo, ICB-replay) decode throughput on a REAL
// gemma4 checkpoint, to compare against the cgo pkg/metal path's tg512 baseline (~169 tok/s on e2b). This
// is the instrument the host-vs-GPU thesis actually needs — the 1-layer micro-bench is too small to show
// the ICB encode-bypass win. Env-guarded (NATIVE_BENCH_DIR); a functional perf run on a real model.
func TestNativeDecodeTokPerSec(t *testing.T) {
	dir := os.Getenv("NATIVE_BENCH_DIR")
	if dir == "" {
		t.Skip("set NATIVE_BENCH_DIR to a real gemma4 checkpoint dir")
	}
	sess, err := LoadDir(dir, 1024)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	defer func() { _ = sess.Close() }()

	prompt := []int32{2, 1841, 689, 573, 6182, 576} // short prompt; greedy decode is timing-only
	// Warmup: records the per-session ICB + primes the GPU/shaders (excluded from the measurement).
	if _, err := sess.Generate(prompt, 24, -1); err != nil {
		t.Fatalf("warmup: %v", err)
	}
	const N = 512
	start := time.Now()
	gen, err := sess.Generate(prompt, N, -1)
	wall := time.Since(start)
	if err != nil {
		t.Fatalf("measure: %v", err)
	}
	tps := float64(len(gen)) / wall.Seconds()
	t.Logf("native decode: %d tokens in %v = %.1f tok/s  (ICB eligible=%v) — cgo pkg/metal baseline ~169 tok/s",
		len(gen), wall.Round(time.Millisecond), tps, sess.icbEligible())
}

// TestNativeResidentPLEByteIdentity proves the resident no-copy PLE projection (MatVecBF16Buf, weight bound
// at its shard offset) decodes BIT-EXACT to the host-bytes path (MatVecBF16, weight re-uploaded per token).
// CRITICAL: the native decode is reproducible WITHIN a load but diverges ACROSS loads (per-load alignment /
// a per-load global — see TestNativeDecodeReproducibilityOneLoad), so a multi-session resident-vs-host
// compare is confounded. This holds ONE load fixed and toggles the PLE path at CALL time (pleResidentDisabled
// flips bufView per token, not at build), with a position reset between the two decodes — no cross-load drift.
func TestNativeResidentPLEByteIdentity(t *testing.T) {
	dir := os.Getenv("NATIVE_BENCH_DIR")
	if dir == "" {
		t.Skip("set NATIVE_BENCH_DIR to a real gemma4 checkpoint dir")
	}
	s, err := LoadDir(dir, 256)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	defer func() { _ = s.Close() }()
	prompt := []int32{2, 1841, 689, 573, 6182, 576}

	pleResidentDisabled = false // resident PLE projection on this load
	resident, err := s.Generate(prompt, 48, -1)
	if err != nil {
		t.Fatalf("resident gen: %v", err)
	}
	s.pos = 0                  // fresh prefill on the SAME load
	pleResidentDisabled = true // host PLE projection (call-time toggle)
	host, err := s.Generate(prompt, 48, -1)
	pleResidentDisabled = false
	if err != nil {
		t.Fatalf("host gen: %v", err)
	}
	if len(resident) != len(host) {
		t.Fatalf("length mismatch: resident %d, host %d", len(resident), len(host))
	}
	for i := range resident {
		if resident[i] != host[i] {
			t.Fatalf("token %d diverged: resident %d != host %d (same load) → the resident PLE projection CHANGES the decode", i, resident[i], host[i])
		}
	}
	t.Logf("✓ resident PLE projection == host PLE on the same load — %d tokens bit-exact", len(resident))
}
