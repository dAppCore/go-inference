// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"
)

// TestNativeDecodeReproducibilityOneLoad discriminates the reproducible token-9 divergence seen across
// session loads: is it an order-dependent STATE bug (decode #1 leaves global state corrupting decode #2),
// or load-time (alignment / a per-load global)? Both host path, ONE model load, two prefills at a reset
// position. Differ → within-load state corruption. Match → the divergence is per-LOAD, not within-load.
func TestNativeDecodeReproducibilityOneLoad(t *testing.T) {
	dir := os.Getenv("NATIVE_BENCH_DIR")
	if dir == "" {
		t.Skip("set NATIVE_BENCH_DIR to a real gemma4 checkpoint dir")
	}
	pleResidentDisabled = true // host path for both decodes
	defer func() { pleResidentDisabled = false }()
	s, err := LoadDir(dir, 256)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	defer func() { _ = s.Close() }()
	prompt := []int32{2, 1841, 689, 573, 6182, 576}
	a, err := s.Generate(prompt, 48, -1)
	if err != nil {
		t.Fatalf("gen a: %v", err)
	}
	s.pos = 0 // reset position: a second fresh prefill on the SAME load (overwrites the KV cache)
	b, err := s.Generate(prompt, 48, -1)
	if err != nil {
		t.Fatalf("gen b: %v", err)
	}
	for i := range a {
		if i < len(b) && a[i] != b[i] {
			t.Fatalf("ONE LOAD, two prefills diverge at token %d (%d != %d) → order-dependent STATE within a load", i, a[i], b[i])
		}
	}
	t.Logf("one-load two-prefill reproducible over %d tokens → the cross-LOAD divergence is per-load (alignment / per-load global), not within-load state", len(a))
}

// TestNativeDecodeCrossLoadICBAB answers whether the cross-load decode divergence is the Generate→ICB change
// (6eddab66) or inherent alignment/fp. It decodes the same prompt on two FRESH loads with the ICB replay ON
// (the default path) and again with it OFF (per-op stepToken, the pre-6eddab66 path), and reports the first
// cross-load divergence token for each. ICB drifts but per-op is stable ⇒ the ICB introduced it (regression);
// both drift ⇒ inherent. Diagnostic (always passes); reads the verdict from the logs.
func TestNativeDecodeCrossLoadICBAB(t *testing.T) {
	dir := os.Getenv("NATIVE_BENCH_DIR")
	if dir == "" {
		t.Skip("set NATIVE_BENCH_DIR to a real gemma4 checkpoint dir")
	}
	prompt := []int32{2, 1841, 689, 573, 6182, 576}
	decode := func() []int32 {
		resetResidentBufsForTest() // drop the prior load's address-keyed cache — rule it out of the A/B
		s, err := LoadDir(dir, 256)
		if err != nil {
			t.Fatalf("LoadDir: %v", err)
		}
		defer func() { _ = s.Close() }()
		g, err := s.Generate(prompt, 48, -1)
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		return g
	}
	firstDiff := func(a, b []int32) int {
		for i := range a {
			if i < len(b) && a[i] != b[i] {
				return i
			}
		}
		return -1 // reproducible
	}
	pleResidentDisabled = false

	icbDisabledForTest = false // A: ICB replay ON (default)
	onDiff := firstDiff(decode(), decode())
	icbDisabledForTest = true // B: per-op stepToken (pre-6eddab66)
	offDiff := firstDiff(decode(), decode())
	icbDisabledForTest = false

	t.Logf("ICB ON  cross-load first-divergence token = %d  (-1 = reproducible)", onDiff)
	t.Logf("ICB OFF cross-load first-divergence token = %d  (-1 = reproducible)", offDiff)
	switch {
	case onDiff >= 0 && offDiff < 0:
		t.Logf("⇒ VERDICT: the ICB replay introduces cross-load non-reproducibility (per-op is load-stable) — REGRESSION from 6eddab66")
	case onDiff < 0 && offDiff < 0:
		t.Logf("⇒ VERDICT: both paths load-stable — the cross-load drift did not reproduce this run")
	case onDiff >= 0 && offDiff >= 0:
		t.Logf("⇒ VERDICT: both paths drift across loads — INHERENT (per-load alignment/fp), not the ICB")
	default:
		t.Logf("⇒ VERDICT: ICB stable but per-op drifts — unexpected, investigate")
	}
}
