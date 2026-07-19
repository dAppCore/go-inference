// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"os"
	"testing"

	"dappco.re/go/inference/model"
	g4 "dappco.re/go/inference/model/gemma4"
)

// tqTestQuantModelHD is tqTestQuantModel with a caller-chosen (TQ-instantiated)
// head dim — the real gemma4 GLOBAL layers run head dim 512, twice the 128 the
// default fixture uses, and the store/dequant reductions span the full
// LTHN_TQ_KV_CAP there.
func tqTestQuantModelHD(t *testing.T, headDim int) (*QuantModel, model.Arch) {
	t.Helper()
	const gs, bits = 32, 4
	cfg := g4.Config{
		HiddenSize: headDim, NumHiddenLayers: 2, IntermediateSize: 256,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: headDim, VocabSize: 32, RMSNormEps: 1e-6,
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
	return g, arch
}

// decode_batched_tq_test.go — the #48 batched TurboQuant prefill parity gate.
//
// A prompt-scale chunk ENGAGES the batched dense TQ landing (chunk-wide code
// store + the per-layer dequant scratch the SDPA reads) and BOTH its per-row
// hiddens AND the next-token decode logits off the boundary hidden track a
// second TQ session stepping the SAME ids through the per-token replay (the
// codes-reading decode SDPA — the existing sequential path). The codes are the
// same bytes either way; the batched read scores the reconstructed bf16 scratch
// (Πᵀ·γ·c rounded once to bf16) while the sequential decode keeps c/γ split and
// applies γ after the score sum, so the two agree within the CODEC band, not
// byte-tight — the honest accumulation-order gap the kernel's numeric note
// predicts (q·k̃ = q·γΠᵀc = γ·(Πq)·c is the same value, differently rounded).

// TestArchQuantSessionTurboQuantBatchedPrefillMatchesSequential_Good is the #48
// parity gate: the batched TQ prefill vs the per-token replay, per-row hiddens
// and the next-token logits.
func TestArchQuantSessionTurboQuantBatchedPrefillMatchesSequential_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := tqTestQuantModel(t)
	const maxLen = 128

	batched, err := newArchQuantSessionShardsWithHeadConfig(g, arch, maxLen, nil, nil, archSessionConfig{kvCacheMode: "turboquant:4"})
	if err != nil {
		t.Fatalf("batched session: %v", err)
	}
	seq, err := newArchQuantSessionShardsWithHeadConfig(g, arch, maxLen, nil, nil, archSessionConfig{kvCacheMode: "turboquant:4"})
	if err != nil {
		t.Fatalf("sequential session: %v", err)
	}
	if batched.state.icb == nil || !batched.state.icb.hasKVTQ() {
		t.Fatal("turboquant session did not arm the TQ carrier")
	}

	ids := make([]int32, 40) // > batchedDenseICBMaxRows: the batched fold lane
	for i := range ids {
		ids[i] = int32((i*7 + 3) % arch.Vocab)
	}
	embs := make([][]byte, len(ids))
	for i, id := range ids {
		emb, eerr := batched.embedID(id)
		if eerr != nil {
			t.Fatalf("embedID: %v", eerr)
		}
		embs[i] = append([]byte(nil), emb...)
	}

	var (
		out [][]byte
		ok  bool
	)
	withAutoreleasePool(func() { out, ok, err = batched.state.stepTokensBatchedDense(embs, 0) })
	if err != nil {
		t.Fatalf("stepTokensBatchedDense: %v", err)
	}
	if !ok {
		t.Fatal("prompt-scale TQ batch must ENGAGE the batched fold (#48) — a decline means the per-token fallback is still the only lane")
	}
	if batched.state.tqPrefill == nil {
		t.Fatal("batched TQ prefill scratch not allocated — the chunk did not take the TQ lane")
	}
	batchedRows := make([][]byte, len(out))
	for i := range out {
		batchedRows[i] = append([]byte(nil), out[i]...)
	}

	// Per-row hiddens: the batched chunk vs the per-token replay, id by id.
	worstAll, scaleAll := 0.0, 0.0
	var seqBoundary []byte
	for i, id := range ids {
		hs, serr := seq.StepWithID(id, embs[i])
		if serr != nil {
			t.Fatalf("sequential StepWithID(%d): %v", i, serr)
		}
		seqBoundary = hs
		a, b := bf16ToF32Slice(batchedRows[i]), bf16ToF32Slice(hs)
		worst, hscale := 0.0, 0.0
		for j := range a {
			if d := math.Abs(float64(a[j]) - float64(b[j])); d > worst {
				worst = d
			}
			if m := math.Abs(float64(b[j])); m > hscale {
				hscale = m
			}
		}
		if worst > worstAll {
			worstAll = worst
		}
		if hscale > scaleAll {
			scaleAll = hscale
		}
		if worst > 0.05*math.Max(hscale, 1) {
			t.Fatalf("row %d: batched TQ prefill diverges structurally from the replay: worst |Δ|=%g (hidden scale %g)", i, worst, hscale)
		}
	}
	t.Logf("batched TQ prefill tracked the per-token replay over %d rows: worst |Δ| = %.5g (hidden scale %.4g)", len(ids), worstAll, scaleAll)

	// The gate's headline: the next-token decode logits off the boundary hidden.
	gs, bits := g.GroupSize, g.Bits
	head := func(hidden []byte) []float32 {
		logits, herr := LMHeadQuant(hidden, g.FinalNorm, g.LMHead, g.LMHeadScales, g.LMHeadBiases, arch.Hidden, arch.Vocab, gs, bits, arch.Eps, arch.SoftCap)
		if herr != nil {
			t.Fatalf("LMHeadQuant: %v", herr)
		}
		return bf16ToF32Slice(logits)
	}
	argmax := func(l []float32) int {
		best := 0
		for i, v := range l {
			if v > l[best] {
				best = i
			}
		}
		return best
	}
	lB, lS := head(batchedRows[len(ids)-1]), head(seqBoundary)
	maxLogitDelta := 0.0
	for i := range lB {
		if d := math.Abs(float64(lB[i]) - float64(lS[i])); d > maxLogitDelta {
			maxLogitDelta = d
		}
	}
	aB, aS := argmax(lB), argmax(lS)
	// Measured band (this synthetic, k4v4, 2026-07-19): per-row worst |Δ| 0.625
	// at hidden scale 14.25 (~4.4%, the same tier the q8 batched-vs-sequential
	// gate trades at) and next-token argmax AGREES with max |logit Δ| 0.125 —
	// NOT byte-tight: the batched multiQ online-softmax over the bf16 scratch and
	// the per-token decode over the codes accumulate in different orders (and the
	// scratch rounds Πᵀ·γ·c to bf16 once). The asserted floors (argmax equal,
	// |Δ| ≤ 1.0) carry ~8× margin while still collapsing on a structural break
	// (stale codes, wrong γ addressing, a dropped bind — those blow the logits
	// apart by orders of magnitude, not by 0.125).
	t.Logf("next-token decode logits: batched argmax %d, sequential argmax %d, max |logit Δ| %.5g", aB, aS, maxLogitDelta)
	if aB != aS {
		t.Fatalf("next-token argmax differs between the batched prefill and the sequential replay: %d vs %d", aB, aS)
	}
	if maxLogitDelta > 1.0 || math.IsNaN(maxLogitDelta) {
		t.Fatalf("next-token max |logit Δ| %.5g outside the codec band (≤ 1.0)", maxLogitDelta)
	}
}

// TestArchQuantSessionTurboQuantBatchedPrefillMultiChunk_Good pins the
// INCREMENTAL scratch across chunks: prefilling the same ids as TWO batched
// chunks (basePos 0 then basePos 24) must track the per-token replay exactly as
// one chunk does — chunk 2 reads the history chunk 1 reconstructed into the
// scratch, never re-dequantising it (the high-water-mark path, #48).
func TestArchQuantSessionTurboQuantBatchedPrefillMultiChunk_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := tqTestQuantModel(t)
	const maxLen, split = 128, 20 // both chunks > batchedDenseICBMaxRows (16) to engage the fold

	batched, err := newArchQuantSessionShardsWithHeadConfig(g, arch, maxLen, nil, nil, archSessionConfig{kvCacheMode: "turboquant:4"})
	if err != nil {
		t.Fatalf("batched session: %v", err)
	}
	seq, err := newArchQuantSessionShardsWithHeadConfig(g, arch, maxLen, nil, nil, archSessionConfig{kvCacheMode: "turboquant:4"})
	if err != nil {
		t.Fatalf("sequential session: %v", err)
	}

	ids := make([]int32, 40)
	embs := make([][]byte, len(ids))
	for i := range ids {
		ids[i] = int32((i*7 + 3) % arch.Vocab)
		emb, eerr := batched.embedID(ids[i])
		if eerr != nil {
			t.Fatalf("embedID: %v", eerr)
		}
		embs[i] = append([]byte(nil), emb...)
	}

	// Two batched chunks: [0,split) then [split,40) — basePos > 0 exercises the
	// incremental append.
	var out1, out2 [][]byte
	var ok1, ok2 bool
	withAutoreleasePool(func() { out1, ok1, err = batched.state.stepTokensBatchedDense(embs[:split], 0) })
	if err != nil || !ok1 {
		t.Fatalf("chunk 1 stepTokensBatchedDense: ok=%v err=%v", ok1, err)
	}
	withAutoreleasePool(func() { out2, ok2, err = batched.state.stepTokensBatchedDense(embs[split:], split) })
	if err != nil || !ok2 {
		t.Fatalf("chunk 2 stepTokensBatchedDense: ok=%v err=%v", ok2, err)
	}
	rows := make([][]byte, 0, len(ids))
	for _, r := range out1 {
		rows = append(rows, append([]byte(nil), r...))
	}
	for _, r := range out2 {
		rows = append(rows, append([]byte(nil), r...))
	}

	worstAll, worstChunk2 := 0.0, 0.0
	for i, id := range ids {
		hs, serr := seq.StepWithID(id, embs[i])
		if serr != nil {
			t.Fatalf("sequential StepWithID(%d): %v", i, serr)
		}
		a, b := bf16ToF32Slice(rows[i]), bf16ToF32Slice(hs)
		worst, hscale := 0.0, 0.0
		for j := range a {
			if d := math.Abs(float64(a[j]) - float64(b[j])); d > worst {
				worst = d
			}
			if m := math.Abs(float64(b[j])); m > hscale {
				hscale = m
			}
		}
		if worst > worstAll {
			worstAll = worst
		}
		if i >= split && worst > worstChunk2 {
			worstChunk2 = worst
		}
		if worst > 0.05*math.Max(hscale, 1) {
			t.Fatalf("row %d (chunk %d): two-chunk batched TQ diverges: worst |Δ|=%g (scale %g)", i, 1+boolToInt(i >= split), worst, hscale)
		}
	}
	t.Logf("two-chunk batched TQ tracked the per-token replay: worst |Δ| all=%.5g, chunk-2 (incremental read)=%.5g", worstAll, worstChunk2)
}

func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

// TestTurboQuantKVDequantRoundTrip_Good isolates the dequant kernel: store random
// rows then dequant them; the reconstruction error must be the codec band at
// EVERY instantiated head dim (128/256/512) — a head-dim-specific blow-up here
// is a kernel bug, not the attention accumulation.
func TestTurboQuantKVDequantRoundTrip_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	// 605 rows (>512, many simd-groups interleaving) is the regression guard for
	// the store's γ-read/reuse race — it only surfaced past head dim 256 under a
	// batched dispatch (#48).
	for _, headDim := range []int{128, 256, 512} {
		const heads, bits = 605, 4
		rows := make([]float32, heads*headDim)
		for i := range rows {
			rows[i] = float32(math.Sin(float64(i)*0.017)) * 1.5
		}
		rowsBf := toBF16Bytes(rows)
		codes, gammas, err := TurboQuantKVStoreDevice(rowsBf, heads, headDim, bits)
		if err != nil {
			t.Fatalf("hd %d store: %v", headDim, err)
		}
		recon, err := TurboQuantKVDequantDevice(codes, gammas, heads, headDim, bits)
		if err != nil {
			t.Fatalf("hd %d dequant: %v", headDim, err)
		}
		rec := bf16ToF32Slice(recon)
		var num, den float64
		for i := range rows {
			d := float64(rec[i]) - float64(rows[i])
			num += d * d
			den += float64(rows[i]) * float64(rows[i])
		}
		rel := math.Sqrt(num / den)
		t.Logf("hd %d: store→dequant relative L2 error %.4f", headDim, rel)
		// 4-bit Lloyd-Max on a rotated unit vector: ~10-20% relative is the codec
		// floor. A transpose/stride bug blows this to ~1.0 (uncorrelated).
		if rel > 0.35 {
			t.Fatalf("hd %d: store→dequant relative error %.4f — kernel bug (codec floor is ~0.2)", headDim, rel)
		}
	}
}

// TestArchQuantSessionTurboQuantBatchedPrefillDeep_Good pins a DEEP single chunk
// (past the multiQ knee) tracking the per-token replay — the depth the real-model
// retrieval exercises, on the fast synthetic.
func TestArchQuantSessionTurboQuantBatchedPrefillDeep_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := tqTestQuantModelHD(t, 512) // the real gemma4 GLOBAL head dim
	const maxLen, n = 1024, 700

	batched, err := newArchQuantSessionShardsWithHeadConfig(g, arch, maxLen, nil, nil, archSessionConfig{kvCacheMode: "turboquant:4"})
	if err != nil {
		t.Fatalf("batched session: %v", err)
	}
	seq, err := newArchQuantSessionShardsWithHeadConfig(g, arch, maxLen, nil, nil, archSessionConfig{kvCacheMode: "turboquant:4"})
	if err != nil {
		t.Fatalf("sequential session: %v", err)
	}
	ids := make([]int32, n)
	embs := make([][]byte, n)
	for i := range ids {
		ids[i] = int32((i*13 + 5) % arch.Vocab)
		emb, eerr := batched.embedID(ids[i])
		if eerr != nil {
			t.Fatalf("embedID: %v", eerr)
		}
		embs[i] = append([]byte(nil), emb...)
	}
	var out [][]byte
	var ok bool
	withAutoreleasePool(func() { out, ok, err = batched.state.stepTokensBatchedDense(embs, 0) })
	if err != nil || !ok {
		t.Fatalf("deep stepTokensBatchedDense: ok=%v err=%v", ok, err)
	}
	worstAll := 0.0
	for i, id := range ids {
		hs, serr := seq.StepWithID(id, embs[i])
		if serr != nil {
			t.Fatalf("sequential StepWithID(%d): %v", i, serr)
		}
		if i%50 != 0 && i != n-1 {
			continue // sample rows — the whole 700-step replay is the slow half
		}
		a, b := bf16ToF32Slice(out[i]), bf16ToF32Slice(hs)
		worst, hscale := 0.0, 0.0
		for j := range a {
			if d := math.Abs(float64(a[j]) - float64(b[j])); d > worst {
				worst = d
			}
			if m := math.Abs(float64(b[j])); m > hscale {
				hscale = m
			}
		}
		if worst > worstAll {
			worstAll = worst
		}
		// Head dim 512 + 700-deep + 4-bit is the widest batched-vs-sequential
		// codec band (accumulation order + reconstruction rounding over 512 dims);
		// 10% still collapses on a structural break (a race, stale codes) — the
		// pre-fix bug read >100% here.
		if worst > 0.10*math.Max(hscale, 1) {
			t.Fatalf("deep row %d: batched TQ diverges: worst |Δ|=%g (scale %g)", i, worst, hscale)
		}
	}
	t.Logf("deep (%d-token single chunk) batched TQ tracked the replay: worst |Δ| = %.5g", n, worstAll)
}

// TestArchQuantSessionTurboQuantBatchedPrefillDeclines_Bad proves the lever the
// parity test relies on: with the batched TQ prefill disabled, a prompt-scale
// TQ chunk DECLINES the batched dense pass (ok=false, no scratch) so the session
// falls through to the per-token replay — the honest sequential fallback.
func TestArchQuantSessionTurboQuantBatchedPrefillDeclines_Bad(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := tqTestQuantModel(t)
	sess, err := newArchQuantSessionShardsWithHeadConfig(g, arch, 128, nil, nil, archSessionConfig{kvCacheMode: "turboquant:4"})
	if err != nil {
		t.Fatalf("turboquant session: %v", err)
	}
	ids := make([]int32, 40)
	embs := make([][]byte, len(ids))
	for i := range ids {
		ids[i] = int32((i*7 + 3) % arch.Vocab)
		emb, eerr := sess.embedID(ids[i])
		if eerr != nil {
			t.Fatalf("embedID: %v", eerr)
		}
		embs[i] = append([]byte(nil), emb...)
	}

	batchedTQPrefillDisabledForTest = true
	t.Cleanup(func() { batchedTQPrefillDisabledForTest = false })
	var ok bool
	withAutoreleasePool(func() { _, ok, err = sess.state.stepTokensBatchedDense(embs, 0) })
	if err != nil {
		t.Fatalf("stepTokensBatchedDense: %v", err)
	}
	if ok {
		t.Fatal("with the lever set the batched TQ prefill must DECLINE (ok=false) to the per-token replay")
	}
	if sess.state.tqPrefill != nil {
		t.Fatal("a declined batched TQ prefill must NOT allocate the read scratch")
	}
}
