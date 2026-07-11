// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"math"
	"os"
	"testing"

	"dappco.re/go/inference/model"
	g4 "dappco.re/go/inference/model/gemma4"
)

// newKVQ8ICBFixture builds a GLOBAL-attention quant session at a q8-qualifying
// geometry (hd=256, kvd 64-aligned) with maxLen past sdpa2PassMinKV, so the
// recorded ICB exercises BOTH the q8 store rebinds and the 2-pass q8 SDPA.
func newKVQ8ICBFixture(t testing.TB) *ArchSession {
	return newKVQ8ICBFixtureLen(t, 2048)
}

func newKVQ8ICBFixtureLen(t testing.TB, maxLen int) *ArchSession {
	t.Helper()
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 256, 256, 64
	const numLayers, gs, bits = 3, 64, 4
	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: numLayers, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim, VocabSize: vocab, RMSNormEps: 1e-6,
		Quantization: &model.QuantConfig{GroupSize: gs, Bits: bits},
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	lm, err := model.Assemble(quantGemma4Tensors(t, arch, gs, bits), arch, model.StandardWeightNames())
	if err != nil {
		t.Fatalf("model.Assemble: %v", err)
	}
	g, err := loadedToQuant(lm, gs, bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	sess, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession: %v", err)
	}
	t.Cleanup(func() { sess.Close() })
	return sess
}

// TestKVQ8ICBDecodeTracksBF16 pins the #367 slice-B contract: with
// LTHN_KV_Q8_ICB armed, the recorded ICB stores GLOBAL K/V as int8 + group
// scales and reads them through the q8 SDPA — the per-step hiddens must stay
// within quantisation distance of the bf16-ICB session (a structural bug —
// wrong offset, missing rebind, stale staging — produces garbage far outside
// it). Also pins the v1 coherence gates: the batched dense lane and KV
// snapshots decline on q8 sessions.
func TestKVQ8ICBDecodeTracksBF16(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	// bf16 reference session, pinned against the q8 default (the flag is
	// read at session build).
	kvQ8ICBOffForTest = true
	ref := newKVQ8ICBFixture(t)
	kvQ8ICBOffForTest = false
	if ref.state.icb == nil {
		t.Fatal("fixture must record an ICB")
	}
	if ref.state.icb.hasKVQ8() {
		t.Fatal("reference session must be bf16 (flag leaked)")
	}

	kvQ8ICBForTest = true
	t.Cleanup(func() { kvQ8ICBForTest = false })
	q8 := newKVQ8ICBFixture(t)
	if q8.state.icb == nil {
		t.Fatal("q8 fixture must record an ICB")
	}
	if !q8.state.icb.hasKVQ8() {
		t.Fatal("q8 session did not arm q8 KV — no global layer qualified; the lane was never exercised")
	}
	armed := 0
	for li := range q8.state.specs {
		if q8.state.icb.kvQ8.on(li) {
			if q8.state.specs[li].Attention != model.GlobalAttention {
				t.Fatalf("layer %d: q8 armed on a non-global layer", li)
			}
			armed++
		}
	}
	t.Logf("q8 armed on %d global owner layers", armed)

	// step the same token sequence through both; hiddens must track within
	// quantisation distance from the FIRST step (structural failures are
	// orders of magnitude outside this).
	ids := []int32{1, 5, 3, 9, 7, 2, 4, 8, 6, 1, 5, 3}
	worstAll := 0.0
	for i, id := range ids {
		hq, err := q8.stepID(id)
		if err != nil {
			t.Fatalf("q8 stepID(%d): %v", i, err)
		}
		hqCopy := append([]byte(nil), hq...)
		hr, err := ref.stepID(id)
		if err != nil {
			t.Fatalf("ref stepID(%d): %v", i, err)
		}
		worst, scale := 0.0, 0.0
		for j := 0; j+1 < len(hr); j += 2 {
			a := float64(bf16ToF32(hqCopy[j], hqCopy[j+1]))
			b := float64(bf16ToF32(hr[j], hr[j+1]))
			if d := math.Abs(a - b); d > worst {
				worst = d
			}
			if m := math.Abs(b); m > scale {
				scale = m
			}
		}
		if worst > worstAll {
			worstAll = worst
		}
		if worst > 0.05*math.Max(scale, 1) {
			t.Fatalf("step %d: q8 hidden diverges structurally: worst |Δ|=%g (hidden scale %g)", i, worst, scale)
		}
	}
	t.Logf("q8 tracked bf16 over %d steps: worst |Δ| = %.5g", len(ids), worstAll)

	// slice C: the batched verify ENGAGES on q8 (multiQ q8 read + staged
	// rows-store landing) while shallow — the fold is the supported shape.
	q8.state.verifyFoldSmallK = true
	_, batched, err := q8.verifyBatchedHiddens([]int32{2, 4, 6})
	q8.state.verifyFoldSmallK = false
	if err != nil {
		t.Fatalf("verifyBatchedHiddens: %v", err)
	}
	if !batched {
		t.Fatal("batched verify must ENGAGE on q8 ICB caches at shallow depth (slice C)")
	}
	// …and KV snapshots serve the dequantised bf16 MIRROR (slice D): the view
	// is bf16-sized so every codec and cross-session restore stays compatible.
	q8li := -1
	for li := range q8.state.specs {
		if q8.state.icb.kvQ8.on(li) {
			q8li = li
			break
		}
	}
	kBuf, _, kPtr, vPtr, serr := q8.snapshotCacheViews(q8li)
	if serr != nil {
		t.Fatalf("snapshotCacheViews on a q8 layer: %v", serr)
	}
	if kPtr == nil || vPtr == nil {
		t.Fatal("q8 snapshot mirror returned nil pointers")
	}
	kvd := q8.state.icb.rowBytes[q8li] / bf16Size
	if got, want := int(bufferLengthFast(kBuf)), q8.state.icb.cacheRows[q8li]*kvd*bf16Size; got != want {
		t.Fatalf("q8 mirror is not bf16-sized: %d != %d", got, want)
	}
}

// TestKVQ8ICBBatchedPrefillMatchesSequential pins slice C: a prompt-scale
// chunk ENGAGES the batched dense fold on a q8 session (staged rows-store
// landing + the multi-query causal q8 read) and its per-row hiddens track a
// second q8 session stepping the same ids through the per-token replay — the
// landed q8 bytes are the same store math either way; only the attention
// summation order differs (the fold's token-identity tier).
func TestKVQ8ICBBatchedPrefillMatchesSequential(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	kvQ8ICBForTest = true
	t.Cleanup(func() { kvQ8ICBForTest = false })
	batchedSess := newKVQ8ICBFixture(t)
	seqSess := newKVQ8ICBFixture(t)
	if !batchedSess.state.icb.hasKVQ8() {
		t.Fatal("fixture did not arm q8")
	}

	ids := make([]int32, 40) // > batchedDenseICBMaxRows: the fold lane
	for i := range ids {
		ids[i] = int32((i*7 + 3) % 60)
	}
	embs := make([][]byte, len(ids))
	for i, id := range ids {
		emb, err := batchedSess.embedID(id)
		if err != nil {
			t.Fatalf("embedID: %v", err)
		}
		embs[i] = append([]byte(nil), emb...)
	}
	var out [][]byte
	var ok bool
	var err error
	withAutoreleasePool(func() {
		out, ok, err = batchedSess.state.stepTokensBatchedDense(embs, 0)
	})
	if err != nil {
		t.Fatalf("stepTokensBatchedDense: %v", err)
	}
	if !ok {
		t.Fatal("prompt-scale batch must ENGAGE the fold on a q8 session (slice C)")
	}
	batchedRows := make([][]byte, len(out))
	for i := range out {
		batchedRows[i] = append([]byte(nil), out[i]...)
	}

	worstAll := 0.0
	for i, id := range ids {
		hs, serr := seqSess.stepID(id)
		if serr != nil {
			t.Fatalf("sequential stepID(%d): %v", i, serr)
		}
		worst, hscale := 0.0, 0.0
		for j := 0; j+1 < len(hs); j += 2 {
			a := float64(bf16ToF32(batchedRows[i][j], batchedRows[i][j+1]))
			b := float64(bf16ToF32(hs[j], hs[j+1]))
			if d := math.Abs(a - b); d > worst {
				worst = d
			}
			if m := math.Abs(b); m > hscale {
				hscale = m
			}
		}
		if worst > worstAll {
			worstAll = worst
		}
		if worst > 0.05*math.Max(hscale, 1) {
			t.Fatalf("row %d: batched q8 prefill diverges structurally: worst |Δ|=%g (scale %g)", i, worst, hscale)
		}
	}
	t.Logf("batched q8 prefill tracked sequential over %d rows: worst |Δ| = %.5g", len(ids), worstAll)

	// the verify block over the just-prefilled q8 caches engages too and tracks
	// the sequential continuation.
	batchedSess.pos = len(ids)
	draft := []int32{2, 4, 6, 8, 1}
	batchedSess.state.verifyFoldSmallK = true
	vh, vBatched, verr := batchedSess.verifyBatchedHiddens(draft)
	batchedSess.state.verifyFoldSmallK = false
	if verr != nil {
		t.Fatalf("verifyBatchedHiddens: %v", verr)
	}
	if !vBatched {
		t.Fatal("verify must ENGAGE on the q8 session at shallow depth")
	}
	for i, id := range draft {
		hs, serr := seqSess.stepID(id)
		if serr != nil {
			t.Fatalf("verify sequential stepID(%d): %v", i, serr)
		}
		worst, hscale := 0.0, 0.0
		for j := 0; j+1 < len(hs); j += 2 {
			a := float64(bf16ToF32(vh[i][j], vh[i][j+1]))
			b := float64(bf16ToF32(hs[j], hs[j+1]))
			if d := math.Abs(a - b); d > worst {
				worst = d
			}
			if m := math.Abs(b); m > hscale {
				hscale = m
			}
		}
		if worst > 0.05*math.Max(hscale, 1) {
			t.Fatalf("verify row %d: diverges structurally: worst |Δ|=%g (scale %g)", i, worst, hscale)
		}
	}
	t.Logf("q8 batched verify tracked sequential over %d rows", len(draft))
}

// TestKVQ8ICBStateSleepWakeRoundTrip pins the -state contract under q8 (the
// star feature): a q8 session sleeps (SerializeState reads the dequantised
// mirror), a fresh q8 session wakes from those bytes (restore requantises —
// the symmetric codec makes the round trip an exact identity), and the woken
// session's continuation is BYTE-IDENTICAL to the uninterrupted one. The same
// snapshot also wakes a bf16 session (portability both ways), tracking within
// quantisation distance.
func TestKVQ8ICBStateSleepWakeRoundTrip(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	// bf16 portability target, pinned against the q8 default.
	kvQ8ICBOffForTest = true
	bf16Sess := newKVQ8ICBFixture(t)
	kvQ8ICBOffForTest = false

	kvQ8ICBForTest = true
	t.Cleanup(func() { kvQ8ICBForTest = false })
	live := newKVQ8ICBFixture(t)
	woken := newKVQ8ICBFixture(t)
	if !live.state.icb.hasKVQ8() {
		t.Fatal("fixture did not arm q8")
	}

	warm := []int32{1, 5, 3, 9, 7, 2, 4, 8}
	for _, id := range warm {
		if _, err := live.stepID(id); err != nil {
			t.Fatalf("warm stepID: %v", err)
		}
	}
	snap, err := live.SerializeState()
	if err != nil {
		t.Fatalf("SerializeState: %v", err)
	}

	cont := []int32{6, 1, 5, 3, 2, 9}
	liveHiddens := make([][]byte, len(cont))
	for i, id := range cont {
		h, err := live.stepID(id)
		if err != nil {
			t.Fatalf("live cont stepID: %v", err)
		}
		liveHiddens[i] = append([]byte(nil), h...)
	}

	if err := woken.RestoreState(snap); err != nil {
		t.Fatalf("RestoreState (q8→q8): %v", err)
	}
	// save→restore→save identity FIRST (before the continuation moves pos):
	// a woken session re-sleeps to the same bytes.
	resnap, err := woken.SerializeState()
	if err != nil {
		t.Fatalf("re-SerializeState: %v", err)
	}
	if !bytesEqualForTest(resnap, snap) {
		if len(resnap) != len(snap) {
			t.Fatalf("save→restore→save: LENGTH differs %d vs %d", len(resnap), len(snap))
		}
		first, diffs := -1, 0
		for i := range snap {
			if snap[i] != resnap[i] {
				if first < 0 {
					first = i
				}
				diffs++
			}
		}
		t.Fatalf("save→restore→save differs: %d/%d bytes, first at offset %d", diffs, len(snap), first)
	}
	for i, id := range cont {
		h, err := woken.stepID(id)
		if err != nil {
			t.Fatalf("woken stepID: %v", err)
		}
		if !bytesEqualForTest(h, liveHiddens[i]) {
			t.Fatalf("step %d: woken q8 session diverges from the uninterrupted one — the sleep/wake round trip is not the identity", i)
		}
	}
	t.Logf("q8 sleep/wake continuation byte-identical over %d steps", len(cont))

	// portability: the q8 snapshot wakes a bf16 session within quantisation distance.
	if err := bf16Sess.RestoreState(snap); err != nil {
		t.Fatalf("RestoreState (q8→bf16): %v", err)
	}
	for i, id := range cont {
		h, err := bf16Sess.stepID(id)
		if err != nil {
			t.Fatalf("bf16 woken stepID: %v", err)
		}
		worst, hscale := 0.0, 0.0
		for j := 0; j+1 < len(h); j += 2 {
			a := float64(bf16ToF32(h[j], h[j+1]))
			b := float64(bf16ToF32(liveHiddens[i][j], liveHiddens[i][j+1]))
			if d := math.Abs(a - b); d > worst {
				worst = d
			}
			if m := math.Abs(b); m > hscale {
				hscale = m
			}
		}
		if worst > 0.05*math.Max(hscale, 1) {
			t.Fatalf("q8→bf16 wake step %d diverges structurally: worst |Δ|=%g (scale %g)", i, worst, hscale)
		}
	}
	t.Logf("q8 snapshot woke a bf16 session within quantisation distance over %d steps", len(cont))
}

// TestKVQ8ICBDeepVerifyEngages pins the deep-verify q8 branch (#367, the
// MTP+q8 default stance): a verify block past sdpa2PassMinKV with small K —
// the corner that used to decline to the sequential replay — now ENGAGES the
// batched fold, routing each row through the per-row 2-pass q8 read, and
// tracks a sequential q8 twin within quantisation distance.
func TestKVQ8ICBDeepVerifyEngages(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	kvQ8ICBForTest = true
	t.Cleanup(func() { kvQ8ICBForTest = false })
	deep := newKVQ8ICBFixture(t)
	seq := newKVQ8ICBFixture(t)
	if !deep.state.icb.hasKVQ8() {
		t.Fatal("fixture did not arm q8")
	}

	// prefill past the 2-pass knee via the batched fold (one call)
	warm := make([]int32, sdpa2PassMinKV+6)
	for i := range warm {
		warm[i] = int32((i*11 + 2) % 60)
	}
	embs := make([][]byte, len(warm))
	for i, id := range warm {
		emb, err := deep.embedID(id)
		if err != nil {
			t.Fatalf("embedID: %v", err)
		}
		embs[i] = append([]byte(nil), emb...)
	}
	var ok bool
	var err error
	withAutoreleasePool(func() {
		_, ok, err = deep.state.stepTokensBatchedDense(embs, 0)
	})
	if err != nil || !ok {
		t.Fatalf("prefill fold: ok=%v err=%v", ok, err)
	}
	deep.pos = len(warm)
	for _, id := range warm {
		if _, serr := seq.stepID(id); serr != nil {
			t.Fatalf("seq warm: %v", serr)
		}
	}

	// the corner: basePos+K >= sdpa2PassMinKV with K < steelGEMMMinRows
	draft := []int32{3, 7, 1, 9, 5}
	deep.state.verifyFoldSmallK = true
	vh, batched, verr := deep.verifyBatchedHiddens(draft)
	deep.state.verifyFoldSmallK = false
	if verr != nil {
		t.Fatalf("verifyBatchedHiddens: %v", verr)
	}
	if !batched {
		t.Fatal("deep verify must ENGAGE under q8 (the per-row 2-pass q8 branch)")
	}
	for i, id := range draft {
		hs, serr := seq.stepID(id)
		if serr != nil {
			t.Fatalf("seq draft stepID(%d): %v", i, serr)
		}
		worst, hscale := 0.0, 0.0
		for j := 0; j+1 < len(hs); j += 2 {
			a := float64(bf16ToF32(vh[i][j], vh[i][j+1]))
			b := float64(bf16ToF32(hs[j], hs[j+1]))
			if d := math.Abs(a - b); d > worst {
				worst = d
			}
			if m := math.Abs(b); m > hscale {
				hscale = m
			}
		}
		if worst > 0.05*math.Max(hscale, 1) {
			t.Fatalf("deep verify row %d diverges structurally: worst |Δ|=%g (scale %g)", i, worst, hscale)
		}
	}
	t.Logf("deep q8 verify engaged at basePos=%d and tracked sequential over %d rows", len(warm), len(draft))
}

// TestKVQ8ICBBatchedGEMMPrefixMatchesSequential pins the q8 GEMM prefix
// (#367): past sdpaPromptGEMMMinKV the batched fold dequantises the attended
// prefix into the layer mirrors and runs the steel GEMM composition against
// them — the probe fold's hiddens must track a sequential twin within
// quantisation distance. The warm fold stays BELOW the GEMM knee so the
// caches feeding the probe are built by the already-pinned multiQ q8 lane.
func TestKVQ8ICBBatchedGEMMPrefixMatchesSequential(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	kvQ8ICBForTest = true
	t.Cleanup(func() { kvQ8ICBForTest = false })
	batched := newKVQ8ICBFixtureLen(t, 8192)
	seq := newKVQ8ICBFixtureLen(t, 8192)
	if !batched.state.icb.hasKVQ8() {
		t.Fatal("fixture did not arm q8")
	}
	if !gpuHasKVQ8DequantRows() {
		t.Fatal("q8 dequant-rows kernel unavailable")
	}

	// warm to just below the GEMM knee on BOTH sessions via the fold
	warm := make([]int32, sdpaPromptGEMMMinKV-96)
	for i := range warm {
		warm[i] = int32((i*13 + 5) % 60)
	}
	wembs := make([][]byte, len(warm))
	for i, id := range warm {
		emb, err := batched.embedID(id)
		if err != nil {
			t.Fatalf("embedID: %v", err)
		}
		wembs[i] = append([]byte(nil), emb...)
	}
	for _, sess := range []*ArchSession{batched, seq} {
		var ok bool
		var err error
		withAutoreleasePool(func() {
			_, ok, err = sess.state.stepTokensBatchedDense(wembs, 0)
		})
		if err != nil || !ok {
			t.Fatalf("warm fold: ok=%v err=%v", ok, err)
		}
		sess.pos = len(warm)
	}

	// the probe: a fold whose basePos+K crosses the GEMM knee
	probe := make([]int32, 512)
	for i := range probe {
		probe[i] = int32((i*17 + 11) % 60)
	}
	pembs := make([][]byte, len(probe))
	for i, id := range probe {
		emb, err := batched.embedID(id)
		if err != nil {
			t.Fatalf("embedID: %v", err)
		}
		pembs[i] = append([]byte(nil), emb...)
	}
	if base := batched.pos; base+len(probe) < sdpaPromptGEMMMinKV {
		t.Fatalf("probe does not reach the GEMM knee: %d+%d < %d", base, len(probe), sdpaPromptGEMMMinKV)
	}
	var out [][]byte
	var ok bool
	var err error
	withAutoreleasePool(func() {
		out, ok, err = batched.state.stepTokensBatchedDense(pembs, batched.pos)
	})
	if err != nil || !ok {
		t.Fatalf("probe fold: ok=%v err=%v", ok, err)
	}

	worstAll := 0.0
	for i, id := range probe {
		hs, serr := seq.stepID(id)
		if serr != nil {
			t.Fatalf("sequential stepID(%d): %v", i, serr)
		}
		worst, hscale := 0.0, 0.0
		for j := 0; j+1 < len(hs); j += 2 {
			a := float64(bf16ToF32(out[i][j], out[i][j+1]))
			b := float64(bf16ToF32(hs[j], hs[j+1]))
			if d := math.Abs(a - b); d > worst {
				worst = d
			}
			if m := math.Abs(b); m > hscale {
				hscale = m
			}
		}
		if worst > worstAll {
			worstAll = worst
		}
		if hscale > 0 && worst > 0.05*hscale {
			t.Fatalf("probe row %d: batched GEMM-q8 diverges from sequential: worst |Δ| = %v (scale %v)", i, worst, hscale)
		}
	}
	t.Logf("q8 GEMM prefix tracked sequential over %d probe rows at base %d (worst |Δ| = %v)", len(probe), len(warm), worstAll)
}

// TestKVQ8PrefillReleasesGEMMMirrors pins the prefill→decode seam: a deep
// prompt materialises the q8 GEMM prefix mirrors (full-cacheRows bf16 per
// global owner — 31B@256K paid ~17GB and swapped a 96GB box), and
// PrefillTokens must hand the memory back before decode. -state saves still
// work afterwards by re-materialising only the layers they read.
func TestKVQ8PrefillReleasesGEMMMirrors(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	kvQ8ICBForTest = true
	t.Cleanup(func() { kvQ8ICBForTest = false })
	sess := newKVQ8ICBFixtureLen(t, 8192)
	if !sess.state.icb.hasKVQ8() {
		t.Fatal("fixture did not arm q8")
	}

	ids := make([]int32, sdpaPromptGEMMMinKV+512)
	for i := range ids {
		ids[i] = int32((i*13 + 5) % 60)
	}
	if err := sess.PrefillTokens(ids); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	for li := range sess.state.icb.kvQ8.enabled {
		if !sess.state.icb.kvQ8.on(li) {
			continue
		}
		if sess.state.icb.kvQ8.kMirrors != nil && sess.state.icb.kvQ8.kMirrors[li] != nil {
			t.Fatalf("layer %d: GEMM prefix mirror survived the prefill seam", li)
		}
	}

	// -state still works: the sleep re-materialises what it reads.
	if _, err := sess.SerializeState(); err != nil {
		t.Fatalf("SerializeState after mirror release: %v", err)
	}
	if _, err := sess.stepID(7); err != nil {
		t.Fatalf("decode step after mirror release: %v", err)
	}
	t.Logf("mirrors released at the prefill seam; -state and decode intact")
}

// TestArchSessionCloseReleasesICBKVCaches pins the session KV lifecycle:
// model.Generate opens and closes a session per call, and Close used to drop
// the archICBReplay without releasing its per-owner KV set — every generate
// leaked a complete cache set (10.7GB q8 per close on 31B@256K; the 2026-07-11
// receipt run carried a full dead set from the CLI's first session). Three
// open+step+close cycles must return the device allocation to baseline.
func TestArchSessionCloseReleasesICBKVCaches(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	// one throwaway build first so PSO/library/pool one-time costs are excluded
	warm := newKVQ8ICBFixtureLen(t, 65536)
	if _, err := warm.stepID(1); err != nil {
		t.Fatalf("warm stepID: %v", err)
	}
	warm.Close()
	baseline := device.CurrentAllocatedSize()
	for range 3 {
		sess := newKVQ8ICBFixtureLen(t, 65536) // 3 global owners @65536 rows ≈ 100MB+/session
		if _, err := sess.stepID(1); err != nil {
			t.Fatalf("stepID: %v", err)
		}
		sess.Close()
	}
	end := device.CurrentAllocatedSize()
	var grew uint
	if end > baseline {
		grew = end - baseline
	}
	if grew > 64<<20 { // one leaked set alone exceeds this
		t.Fatalf("session close leaked ICB KV caches: device allocation grew %dMB over 3 open+close cycles", grew>>20)
	}
}

// TestFlashQ8PromptMatchesMirrorLane pins the #375 phase-3 contract: the
// q8-READING flash (codes+scales dequantised inside the tile load, no
// mirrors) must track the mirror-dequant lane it replaces within
// quantisation distance across a prompt-scale prefill + decode. Fresh
// session per lane; both are token-identity tiers against the sequential
// oracle, so the bar is closeness, not bytes.
func TestFlashQ8PromptMatchesMirrorLane(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if !flashQ8Usable(256, 1<<20) {
		t.Skip("q8 flash kernel unavailable")
	}
	const maxLen = 8192
	prompt := make([]int32, 4600) // past sdpaPromptGEMMMinKV with the fold engaged
	for i := range prompt {
		prompt[i] = int32(1 + i%50)
	}
	run := func(mirrorLane bool) [][]byte {
		flashQ8OffForTest = mirrorLane
		defer func() { flashQ8OffForTest = false }()
		sess := newKVQ8ICBFixtureLen(t, maxLen)
		if err := sess.PrefillTokens(prompt); err != nil {
			t.Fatalf("PrefillTokens (mirror=%v): %v", mirrorLane, err)
		}
		var outs [][]byte
		for _, id := range []int32{7, 3, 9, 5} {
			h, err := sess.stepID(id)
			if err != nil {
				t.Fatalf("stepID (mirror=%v): %v", mirrorLane, err)
			}
			outs = append(outs, append([]byte(nil), h...))
		}
		return outs
	}
	flash := run(false)
	mirror := run(true)
	for i := range flash {
		if len(flash[i]) != len(mirror[i]) {
			t.Fatalf("step %d: hidden sizes differ", i)
		}
		worst := 0.0
		for j := 0; j+1 < len(flash[i]); j += 2 {
			a := float64(bf16ToF32(flash[i][j], flash[i][j+1]))
			b := float64(bf16ToF32(mirror[i][j], mirror[i][j+1]))
			if d := math.Abs(a - b); d > worst {
				worst = d
			}
		}
		// quantisation-distance bar: both lanes read the SAME q8 codes; the only
		// difference is dequant placement + accumulation order.
		if worst > 0.5 {
			t.Fatalf("step %d: q8 flash diverges from the mirror lane: worst |Δ| = %.4f", i, worst)
		}
		t.Logf("step %d worst |Δ| = %.4f", i, worst)
	}
}
