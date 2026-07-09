// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"math"
	"os"
	"sort"
	"testing"
)

// TestDecodeGPUProfileVsDepthRealE2B localises the #365 deep-context tax. It arms the
// per-op GPU counter profiler at a SHALLOW (256) and a DEEP (16K) context and diffs the
// per-label ms/token — the label that GROWS with depth is the scaling cost. The KV scan
// (attn.sdpa) moves only a few tens of MB/token even at 16K (~0.1ms of bandwidth), so if
// attn.sdpa grows by ~1ms the cost is OVERHEAD (dispatch / occupancy / access pattern),
// not bytes — which quant would NOT fix. If a non-attn label grows, the tax isn't even
// the scan. This is the "check the code" instrument: it says WHERE to read.
//
//	LEM_REAL_E2B=1 MLX_METALLIB_PATH=... go test -run TestDecodeGPUProfileVsDepthRealE2B -v ./engine/metal/
func TestDecodeGPUProfileVsDepthRealE2B(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if os.Getenv("LEM_REAL_E2B") == "" {
		t.Skip("set LEM_REAL_E2B=1 to run the real e2b-4bit deep-context GPU profile (loads ~2.7GB)")
	}
	dir := resolveE2B4bitDir(t)
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

	// the per-op profiler seams live in the stepToken path; a dense model like E2B would
	// otherwise replay via ICB (no seams -> empty spans). Force stepToken.
	prevICB := icbDisabledForTest
	icbDisabledForTest = true
	defer func() { icbDisabledForTest = prevICB }()

	const maxLen = 16384 + 256
	const N = 2 // few tokens: the per-token decode emits ~8 sampled encoders/layer and the
	// device caps the counter sample buffer, so a big N overflows it (buffer-alloc error).
	nLayers := len(lm.Arch.Layer)

	profileAt := func(depth int) map[string]uint64 {
		sess, serr := newArchQuantSessionShards(qm, lm.Arch, maxLen, sb)
		if serr != nil {
			t.Fatalf("session %d: %v", depth, serr)
		}
		prompt := make([]int32, depth)
		prompt[0] = 2
		for i := 1; i < depth; i++ {
			prompt[i] = int32(100 + (i*131)%3000)
		}
		if perr := sess.PrefillTokens(prompt); perr != nil {
			t.Fatalf("prefill %d: %v", depth, perr)
		}
		if _, gerr := sess.GenerateFromCache(4, -1); gerr != nil {
			t.Fatalf("warmup %d: %v", depth, gerr)
		}
		prof, perr := newGPUCounterProfiler((8*nLayers + 4) * N)
		if perr != nil {
			t.Skipf("timestamp counters unavailable: %v", perr)
		}
		sess.state.gpuProf = prof
		_, gerr := sess.GenerateFromCache(N, -1)
		sess.state.gpuProf = nil
		if gerr != nil {
			t.Fatalf("profiled gen %d: %v", depth, gerr)
		}
		spans, serr2 := prof.spans()
		if serr2 != nil {
			t.Fatalf("spans %d: %v", depth, serr2)
		}
		return spans
	}

	shallow := profileAt(256)
	deep := profileAt(16384)

	labs := map[string]bool{}
	for l := range shallow {
		labs[l] = true
	}
	for l := range deep {
		labs[l] = true
	}
	order := make([]string, 0, len(labs))
	for l := range labs {
		order = append(order, l)
	}
	sort.Slice(order, func(i, j int) bool { return (deep[order[i]] - shallow[order[i]]) > (deep[order[j]] - shallow[order[j]]) })

	msPer := func(ns uint64) float64 { return float64(ns) / 1e6 / float64(N) }
	t.Logf("=== #365 per-op GPU ms/token: depth 256 vs 16384 (real e2b-4bit), sorted by Δ ===")
	for _, l := range order {
		s, d := msPer(shallow[l]), msPer(deep[l])
		t.Logf("  %-12s  256:%6.3f   16K:%6.3f   Δ%+6.3f ms  (%+5.0f%%)", l, s, d, d-s, 100*(d-s)/math.Max(s, 1e-6))
	}
	t.Logf("  => the big-Δ label is the scaling cost. attn.sdpa Δ >> 0.1ms = overhead not bytes (quant won't fix); a non-attn Δ = the tax isn't the scan at all")
}
