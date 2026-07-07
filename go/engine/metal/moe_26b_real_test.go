// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"sort"
	"testing"
	"time"

	core "dappco.re/go"
)

// TestRealMoE26BHostProfile decodes the real gemma-4-26B-A4B MoE checkpoint so a CPU profile can show where
// its per-token time goes. Dense e2b decode is GPU-bound (cgocall ≈ GPU wait, 99% gpu-busy); the MoE arch
// can't use the recorded-ICB path (the router top-k forces a host readback), so MoEBlockQuant orchestrates
// ~a dozen separately-host-synced Metal calls per layer per token. If this path is HOST-bound (cgocall a
// small fraction, native orchestration large), that's the reclaimable headroom the dense path doesn't have.
// Gated behind LEM_REAL_MOE (loads ~15 GB). Run with -cpuprofile to read the split.
func TestRealMoE26BHostProfile(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if os.Getenv("LEM_REAL_MOE") == "" {
		t.Skip("set LEM_REAL_MOE=1 to run the real 26B-A4B MoE decode profile (loads ~15GB)")
	}
	dir := resolveMoE26BDir(t)
	const maxLen, warmup, N = 320, 4, 24

	lm, dm, err := loadRegistered(dir)
	if err != nil {
		t.Fatalf("loadRegistered: %v", err)
	}
	defer func() { _ = dm.Close() }()
	if !quantised(lm) {
		t.Fatalf("expected a quantised 26B checkpoint")
	}
	sb, err := buildShardBuffers(dm)
	if err != nil {
		t.Fatalf("buildShardBuffers: %v", err)
	}
	defer func() { _ = sb.Close() }()
	qm, err := loadedToQuant(lm, lm.Embed.GroupSize, lm.Embed.Bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	sess, err := newArchQuantSessionShards(qm, lm.Arch, maxLen, sb)
	if err != nil {
		t.Fatalf("newArchQuantSessionShards: %v", err)
	}

	prompt := []int32{2, 1000, 2500, 4000, 8000, 16000}
	if err := sess.PrefillTokens(prompt); err != nil {
		t.Fatalf("prefill: %v", err)
	}
	if _, err := sess.GenerateFromCache(warmup, -1); err != nil {
		t.Fatalf("warmup: %v", err)
	}
	pieceTimingOn, chainedGPUSpanNs = true, 0
	t0 := time.Now()
	if _, err := sess.GenerateFromCache(N, -1); err != nil {
		pieceTimingOn = false
		t.Fatalf("generate: %v", err)
	}
	wall := time.Since(t0)
	pieceTimingOn = false
	t.Logf("real 26B-A4B MoE decode (tg%d): %.1f tok/s (%.2f ms/token) — run under -cpuprofile; low cgocall ⇒ host-bound",
		N, float64(N)/wall.Seconds(), wall.Seconds()*1000/float64(N))
	t.Logf("chained live links: %d (0 = the chained live lane did not engage)", chainedLiveLinks.Load())
	// cb GPU span vs wall: the residual host/sync gap a submit-ahead pipeline could overlap;
	// span minus the profiler's kernel-sum is the intra-cb bubble tax a concurrent pass chases.
	t.Logf("cb GPU span %.2f ms/token vs wall %.2f — host/sync gap %.2f ms/token",
		float64(chainedGPUSpanNs)/1e6/float64(N), wall.Seconds()*1000/float64(N),
		wall.Seconds()*1000/float64(N)-float64(chainedGPUSpanNs)/1e6/float64(N))
}

// TestRealMoE26BFamilyGPUProfile splits the decode's per-token encoder at the attn/moe family
// seams with timestamp counter sampling and prints where the GPU milliseconds go — the ranked
// table that picks the next cut. (The b8-mega experiment proved this lane is not
// dispatch-count-bound: cuts must chase kernel time, and this is the instrument for that.)
func TestRealMoE26BFamilyGPUProfile(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if os.Getenv("LEM_REAL_MOE") == "" {
		t.Skip("set LEM_REAL_MOE=1 to run the real 26B-A4B GPU family profile (loads ~15GB)")
	}
	dir := resolveMoE26BDir(t)
	const warmup, N = 4, 8
	// LEM_MOE_PROFILE_CTX prefills a synthetic prompt of that length first, so the table
	// shows the DEEP-context split (the long-context droop's anatomy — #339). Default: the
	// short-context profile.
	ctx := 0
	if v := os.Getenv("LEM_MOE_PROFILE_CTX"); v != "" {
		if r := core.ParseInt(v, 10, 32); r.OK {
			if n := int(r.Value.(int64)); n > 0 {
				ctx = n
			}
		}
	}
	maxLen := 320
	if ctx > 0 {
		maxLen = ctx + 64
	}

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
	sess, err := newArchQuantSessionShards(qm, lm.Arch, maxLen, sb)
	if err != nil {
		t.Fatalf("newArchQuantSessionShards: %v", err)
	}
	prompt := []int32{2, 1000, 2500, 4000, 8000, 16000}
	if ctx > 0 {
		prompt = make([]int32, ctx)
		for i := range prompt {
			prompt[i] = int32(2 + (i*97)%16000)
		}
	}
	if err := sess.PrefillTokens(prompt); err != nil {
		t.Fatalf("prefill: %v", err)
	}
	if _, err := sess.GenerateFromCache(warmup, -1); err != nil {
		t.Fatalf("warmup: %v", err)
	}
	t.Logf("profiling at position ~%d (maxLen %d)", len(prompt)+warmup, maxLen)

	// 5 sampled encoders per layer per token (attn, moe.router, moe.local, moe.expert,
	// moe.tail) + slack for the first encoder.
	prof, err := newGPUCounterProfiler((5*len(lm.Arch.Layer) + 4) * N)
	if err != nil {
		t.Skipf("timestamp counter sampling unavailable: %v", err)
	}
	sess.state.gpuProf = prof
	t0 := time.Now()
	_, genErr := sess.GenerateFromCache(N, -1)
	wall := time.Since(t0)
	sess.state.gpuProf = nil
	if genErr != nil {
		t.Fatalf("profiled generate: %v", genErr)
	}
	spans, err := prof.spans()
	if err != nil {
		t.Fatalf("spans: %v", err)
	}
	var total uint64
	labels := make([]string, 0, len(spans))
	for label, ns := range spans {
		total += ns
		labels = append(labels, label)
	}
	if total == 0 {
		t.Fatal("counter sampling resolved zero GPU time")
	}
	sort.Slice(labels, func(i, j int) bool { return spans[labels[i]] > spans[labels[j]] })
	for _, label := range labels {
		ns := spans[label]
		t.Logf("%-11s %7.2f ms/token  %5.1f%% of sampled GPU", label,
			float64(ns)/1e6/float64(N), 100*float64(ns)/float64(total))
	}
	t.Logf("sampled %d encoders over tg%d; sampled GPU %.2f ms/token; wall %.2f ms/token",
		prof.sampled(), N, float64(total)/1e6/float64(N), wall.Seconds()*1000/float64(N))
}

func resolveMoE26BDir(t *testing.T) string {
	home := os.Getenv("HOME")
	base := home + "/.cache/huggingface/hub/models--mlx-community--gemma-4-26B-A4B-it-qat-4bit/snapshots"
	entries, err := os.ReadDir(base)
	if err != nil {
		t.Skipf("26B-A4B snapshot dir not found (%v)", err)
	}
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		dir := base + "/" + e.Name()
		if _, serr := os.Stat(dir + "/config.json"); serr == nil {
			return dir
		}
	}
	t.Skip("no 26B-A4B snapshot with config.json")
	return ""
}
