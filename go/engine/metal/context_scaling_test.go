// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"
	"time"
)

// TestRealE2BContextScaling measures how native decode tok/s degrades as the KV context grows — the curve
// behind "improving the KV improves toks, more so as context grows". Native dispatches MLX's SINGLE-PASS
// sdpa_vector (one threadgroup per head reducing the whole cache); past ~1024 it can't parallelise the
// cache reduction, so the global-attention layers' SDPA degrades. The 2-pass kernels (sdpa_vector_2pass_*,
// already in the metallib for bfloat16_t) split the cache into blocks across threadgroups — the un-wired
// native follow-up. This sizes the gap: decode tok/s after prefilling progressively longer contexts.
func TestRealE2BContextScaling(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if os.Getenv("LEM_REAL_E2B") == "" {
		t.Skip("set LEM_REAL_E2B=1 to run the context-scaling measurement (loads ~2.7GB)")
	}
	dir := resolveE2B4bitDir(t)
	const maxLen, decodeN = 4096, 24
	lm, dm, err := loadRegistered(dir)
	if err != nil {
		t.Fatalf("loadRegistered: %v", err)
	}
	defer func() { _ = dm.Close() }()
	sb, err := buildShardBuffers(dm)
	if err != nil {
		t.Fatalf("buildShardBuffers: %v", err)
	}
	defer func() { _ = sb.Close() }()
	qm, err := loadedToQuant(lm, lm.Embed.GroupSize, lm.Embed.Bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}

	measure := func(promptLen int) float64 {
		prompt := make([]int32, promptLen)
		for i := range prompt {
			prompt[i] = int32(2 + (i*131+7)%32000) // synthetic in-vocab ids
		}
		s, serr := newArchQuantSessionShards(qm, lm.Arch, maxLen, sb)
		if serr != nil {
			t.Fatalf("session: %v", serr)
		}
		if perr := s.PrefillTokens(prompt); perr != nil {
			t.Fatalf("prefill %d: %v", promptLen, perr)
		}
		if _, werr := s.GenerateFromCache(4, -1); werr != nil { // warmup (untimed)
			t.Fatalf("warmup: %v", werr)
		}
		t0 := time.Now()
		if _, gerr := s.GenerateFromCache(decodeN, -1); gerr != nil {
			t.Fatalf("decode: %v", gerr)
		}
		return float64(decodeN) / time.Since(t0).Seconds()
	}

	for _, n := range []int{128, 512, 1024, 2048, 3072} {
		tps := measure(n)
		t.Logf("context %4d tokens: decode %.1f tok/s", n, tps)
	}
}

// TestRealE2BLivePath2PassDelta measures the 2-pass SDPA effect on the LIVE (re-encode)
// decode path — the path the big ICB-ineligible models (12B / 26B-MoE / 31B) run in
// production, exercised here on e2b by forcing icbDisabledForTest. At a long prefill the
// global-attention layers attend the full cache (> the single-pass knee), so the router
// engages the 2-pass kernels there; A/B'ing sdpa2PassDisabledForTest on the SAME path
// isolates their decode-tok/s effect from the re-encode overhead. This is the receipt
// for "improving the KV improves toks, more so as context grows" on the wired path.
func TestRealE2BLivePath2PassDelta(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if os.Getenv("LEM_REAL_E2B") == "" {
		t.Skip("set LEM_REAL_E2B=1 to run the live-path 2-pass delta (loads ~2.7GB)")
	}
	dir := resolveE2B4bitDir(t)
	const maxLen, decodeN = 8192, 64 // longer cache + decode window to push past the knee and damp timing noise
	lm, dm, err := loadRegistered(dir)
	if err != nil {
		t.Fatalf("loadRegistered: %v", err)
	}
	defer func() { _ = dm.Close() }()
	sb, err := buildShardBuffers(dm)
	if err != nil {
		t.Fatalf("buildShardBuffers: %v", err)
	}
	defer func() { _ = sb.Close() }()
	qm, err := loadedToQuant(lm, lm.Embed.GroupSize, lm.Embed.Bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}

	// force the live re-encode path (the encAttnHalfKV ▸ encSDPADecode route), the one
	// the wiring lands on; ICB replay + chained GPU inputs OFF.
	chainedGPUInputsDisabled = true
	icbDisabledForTest = true
	defer func() { chainedGPUInputsDisabled = false; icbDisabledForTest = false }()

	measure := func(promptLen int, twoPass bool) float64 {
		sdpa2PassDisabledForTest = !twoPass
		defer func() { sdpa2PassDisabledForTest = false }()
		prompt := make([]int32, promptLen)
		for i := range prompt {
			prompt[i] = int32(2 + (i*131+7)%32000)
		}
		s, serr := newArchQuantSessionShards(qm, lm.Arch, maxLen, sb)
		if serr != nil {
			t.Fatalf("session: %v", serr)
		}
		if perr := s.PrefillTokens(prompt); perr != nil {
			t.Fatalf("prefill %d: %v", promptLen, perr)
		}
		if _, werr := s.GenerateFromCache(4, -1); werr != nil {
			t.Fatalf("warmup: %v", werr)
		}
		t0 := time.Now()
		if _, gerr := s.GenerateFromCache(decodeN, -1); gerr != nil {
			t.Fatalf("decode: %v", gerr)
		}
		return float64(decodeN) / time.Since(t0).Seconds()
	}

	for _, n := range []int{2048, 4096, 7168} {
		// two repeats each, take the best (least-contended) to damp scheduler noise.
		best := func(twoPass bool) float64 {
			a, b := measure(n, twoPass), measure(n, twoPass)
			if a > b {
				return a
			}
			return b
		}
		off := best(false) // single-pass (2-pass disabled)
		on := best(true)   // 2-pass routed for global layers past the knee
		delta := (on/off - 1) * 100
		t.Logf("live path · context %4d: single-pass %.1f tok/s  2-pass %.1f tok/s  Δ %+.1f%%", n, off, on, delta)
	}
}
