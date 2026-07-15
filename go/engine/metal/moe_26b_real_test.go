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
	const warmup, N = 4, 24
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
	t.Logf("lean gather dispatches: %d (0 = the fc-specialised gather lane did not engage — #280)", leanGatherDispatches.Load())
	t.Logf("fused router dispatches: %d (0 = the single-dispatch router did not engage — #340)", routerFusedDispatches.Load())
	t.Logf("sdpa single-cell dispatches: %d (0 = the P1-final fast path did not engage — #340)", sdpaSingleCellDispatches.Load())
	t.Logf("concurrent-encoder carries: %d (0 = the encoder-carry lane did not engage — #341)", concEncoderCarries.Load())
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
	// N is small (the token count only sets the averaging window) because the device caps the
	// counter sample buffer — full per-token seam sampling (~330 encoders/token) over more tokens
	// overruns MTLCounterSampleBuffer's maximum sample count.
	const warmup, N = 4, 4
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

	// Up to ~11 sampled encoders per layer per token (attn + the attn.proj/sdpa/tail re-seams,
	// moe.router + the router.rms/qmv/topk re-seams, moe.local, moe.expert, moe.tail) + head/PLE
	// slack. Sized with headroom over the measured ~330 encoders/token so the buffer never
	// saturates — a saturated buffer degrades the tail encoders to unsampled, which clips BOTH
	// the spans and the per-label dispatch counts — while staying under the device's sample cap.
	prof, err := newGPUCounterProfiler((14*len(lm.Arch.Layer) + 8) * N)
	if err != nil {
		t.Skipf("timestamp counter sampling unavailable: %v", err)
	}
	sess.state.gpuProf = prof
	// pieceTimingOn arms dispatchCountForTest at the encSink funnel, so each sampled encoder's
	// live-dispatch count is recoverable per label (dispatchCountsByLabel) — the structural
	// companion to the GPU-time spans (#392 STEP 1: the round's op-level split). It changes
	// only measurement, not the decode path. Reset the counter so warmup dispatches don't leak.
	pieceTimingOn, dispatchCountForTest = true, 0
	t0 := time.Now()
	_, genErr := sess.GenerateFromCache(N, -1)
	wall := time.Since(t0)
	pieceTimingOn = false
	sess.state.gpuProf = nil
	if genErr != nil {
		t.Fatalf("profiled generate: %v", genErr)
	}
	spans, err := prof.spans()
	if err != nil {
		t.Fatalf("spans: %v", err)
	}
	counts := prof.dispatchCountsByLabel(dispatchCountForTest)
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
		t.Logf("%-11s %7.2f ms/token  %5.1f%% of sampled GPU  %6.1f dispatch/token", label,
			float64(ns)/1e6/float64(N), 100*float64(ns)/float64(total), float64(counts[label])/float64(N))
	}
	t.Logf("sampled %d encoders over tg%d; sampled GPU %.2f ms/token; wall %.2f ms/token",
		prof.sampled(), N, float64(total)/1e6/float64(N), wall.Seconds()*1000/float64(N))

	// STEP-1 round-split VERDICT (#392): aggregate the labels into the op classes the campaign
	// tracks. The matvec/gather classes (attn.proj, moe.local, moe.expert, router.qmv) are the
	// bandwidth-bound single-row weight streams; everything else (norms, top-k, sdpa, residual/
	// combine tail) is the thin-op + attention-core remainder. MoE-block dispatch counts are
	// exact (the whole block runs through encSink); attn's is an encSink-only lower bound (it
	// also dispatches through the Object funnel, which the shared test counter leaves untouched).
	attnLabels := map[string]bool{"attn": true, "attn.proj": true, "attn.sdpa": true, "attn.tail": true}
	gemvLabels := map[string]bool{"attn.proj": true, "moe.local": true, "moe.expert": true, "router.qmv": true}
	var moeDisp, attnDisp int64
	var gemvNs, thinNs uint64
	for label, ns := range spans {
		if attnLabels[label] {
			attnDisp += counts[label]
		} else {
			moeDisp += counts[label] // moe.* and router.*
		}
		if gemvLabels[label] {
			gemvNs += ns
		} else {
			thinNs += ns
		}
	}
	t.Logf("VERDICT round-split: MoE-block %.0f dispatch/token, attn %.0f dispatch/token (attn = encSink lower bound); "+
		"matvec/gather GPU %.1f%% vs thin/sdpa %.1f%% — the K=1 round's GPU time concentrates in the bandwidth-bound "+
		"GEMV/gather classes; the host/sync gap is closed (see TestRealMoE26BHostProfile: ~0.18 ms/token) and the "+
		"expert stream is already all-routes amortised (see the lean gather receipts). Relative shares only under "+
		"sibling GPU contention; absolute tok/s waits for a quiescent box.",
		float64(moeDisp)/float64(N), float64(attnDisp)/float64(N),
		100*float64(gemvNs)/float64(total), 100*float64(thinNs)/float64(total))
}

func resolveMoE26BDir(t *testing.T) string {
	// LEM_MOE_DIR points the profile at any MoE snapshot (e.g. the non-QAT
	// 26B) — checkpoint A/Bs reuse the same instrument.
	if dir := os.Getenv("LEM_MOE_DIR"); dir != "" {
		return dir
	}
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
