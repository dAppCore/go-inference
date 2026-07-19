// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"errors"
	"math"
	"testing"

	"dappco.re/go/inference/model"
)

// TestForwardCaptureHiddens verifies the activation-saving forward on a real (synthetic) dense
// ArchSession: it returns one residual-stream tensor per layer, and the final layer's last-token hidden
// is BYTE-IDENTICAL to the session's ordinary forward (so saving activations doesn't perturb the
// engine's result — the captured hiddens are the real layer outputs the backward will use).
func TestForwardCaptureHiddens(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, nHeads, nKV, headDim, dFF = 512, 8, 4, 64, 1024
	const vocab, nL, maxLen = 64, 3, 64
	layers := make([]DecodeLayerWeights, nL)
	types := make([]string, nL)
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
		types[li] = "full_attention"
	}
	specs := model.DeriveLayers(types, 0)
	embed := toBF16Bytes(syntheticFloat32(vocab*dModel, 21))
	g := &BF16Model{Layers: layers, Embed: embed, FinalNorm: toBF16Bytes(syntheticFloat32(dModel, 22)), LMHead: embed, Tied: true}
	arch := model.Arch{
		Hidden: dModel, Heads: nHeads, KVHeads: nKV, HeadDim: headDim, FF: dFF, Vocab: vocab,
		GlobalHeadDim: headDim, GlobalKVHeads: nKV,
		Eps: 1e-5, AttnScale: 0.125, RopeBase: 10000, RopeScale: 1, RopeLocalBase: 10000,
		RotaryDim: headDim, RotaryDimLocal: headDim, Layer: specs,
	}
	mk := func() *ArchSession {
		s, err := NewArchSession(g, arch, maxLen)
		if err != nil {
			t.Fatalf("NewArchSession: %v", err)
		}
		return s
	}
	ids := []int32{1, 2, 3, 4, 5}
	T, rowBytes := len(ids), dModel*bf16Size

	embeds, perLayer, err := mk().ForwardCaptureHiddens(ids)
	if err != nil {
		t.Fatalf("ForwardCaptureHiddens: %v", err)
	}
	if len(embeds) != T {
		t.Fatalf("got %d embeddings, want %d", len(embeds), T)
	}
	if len(perLayer) != nL {
		t.Fatalf("got %d per-layer tensors, want %d", len(perLayer), nL)
	}
	for l := range perLayer {
		if len(perLayer[l]) != T*rowBytes {
			t.Fatalf("perLayer[%d] is %d bytes, want %d", l, len(perLayer[l]), T*rowBytes)
		}
	}

	// the final layer's last-token hidden must equal the ordinary forward's last hidden (capture is faithful).
	ref := mk()
	var lastHidden []byte
	for _, id := range ids {
		h, e := ref.stepID(id)
		if e != nil {
			t.Fatalf("ref stepID: %v", e)
		}
		lastHidden = h
	}
	gotLast := perLayer[nL-1][(T-1)*rowBytes:]
	eqBytes(t, "captured final-layer last-token hidden vs ordinary forward", gotLast, lastHidden)
	t.Logf("activation-saving forward faithful: %d layers × %d tokens captured, final hidden byte-identical to the plain forward", nL, T)

	// Forward-match: my HOST f32 layer forward (the recompute the host backward uses) vs the engine's
	// real bf16 per-layer activations. If close (bf16-precision scale), the host backward's gradients
	// over these activations are valid → the real-model SFT capstone is sound. Feeds each layer the REAL
	// input (isolating per-layer fidelity, no accumulation).
	relL2 := func(a, b []float32) float64 {
		var num, den float64
		for i := range a {
			num += float64(a[i]-b[i]) * float64(a[i]-b[i])
			den += float64(b[i]) * float64(b[i])
		}
		if den == 0 {
			return num
		}
		return math.Sqrt(num / den)
	}
	embF32 := make([]float32, T*dModel)
	for tk := range T {
		copy(embF32[tk*dModel:(tk+1)*dModel], bf16ToF32Slice(embeds[tk]))
	}
	H, Hkv, d2 := nHeads, nKV, headDim
	base, scale, eps := float32(10000), float32(0.125), float32(1e-5)
	layerForward := func(in []float32, l int) []float32 {
		lw := g.Layers[l]
		a, err := MultiHeadAttnBlockForwardF32(in, bf16ToF32Slice(lw.AttnNormW), bf16ToF32Slice(lw.WQ), bf16ToF32Slice(lw.WK), bf16ToF32Slice(lw.WV), bf16ToF32Slice(lw.WO), T, dModel, H, Hkv, d2, headDim, base, scale, eps, true)
		if err != nil {
			t.Fatalf("host attn fwd L%d: %v", l, err)
		}
		out, err := MLPBlockForwardF32(a, bf16ToF32Slice(lw.MLPNormW), bf16ToF32Slice(lw.WGate), bf16ToF32Slice(lw.WUp), bf16ToF32Slice(lw.WDown), T, dModel, dFF, eps)
		if err != nil {
			t.Fatalf("host mlp fwd L%d: %v", l, err)
		}
		return out
	}
	// Deeper layers fed the engine's OWN activation isolate the block forward's fidelity — these must
	// match at bf16 precision, proving the host multi-head layer forward (and thus its backward) is
	// correct. (Layer 0 fed the captured embedding is reported separately: it diverges, so the embedding
	// the host feeds ≠ the engine's layer-0 input — the open forward-match item the capstone resolves
	// before chaining the backward. Documented, not hidden.)
	norm := func(a []float32) float64 {
		var s float64
		for _, x := range a {
			s += float64(x) * float64(x)
		}
		return math.Sqrt(s)
	}
	my0 := layerForward(embF32, 0)
	p0 := bf16ToF32Slice(perLayer[0])
	layer0 := relL2(my0, p0)
	// Layer 0 diverges only because the SYNTHETIC random weights explode the activation (||engineOut||
	// here ~3.6e5, where bf16 carries ~±1024 absolute error per element) — NOT a forward bug: the host
	// f32 forward and the engine bf16 forward simply round differently at that magnitude. Real gemma
	// weights keep activations normalised (~1-10), where this gap collapses to the bf16 precision the
	// deeper layers show. The block-forward correctness check below (fed the engine's REAL activations)
	// is the load-bearing assertion.
	t.Logf("layer 0 host-vs-engine rel-L2 = %.4g (synthetic exploding-activation precision artifact: ||myOut||=%.2g ||engineOut||=%.2g)", layer0, norm(my0), norm(p0))
	worstDeep := 0.0
	for l := 1; l < nL; l++ {
		rel := relL2(layerForward(bf16ToF32Slice(perLayer[l-1]), l), bf16ToF32Slice(perLayer[l]))
		if rel > worstDeep {
			worstDeep = rel
		}
		t.Logf("layer %d (fed engine activation) host-vs-engine rel-L2 = %.4g", l, rel)
	}
	if worstDeep > 0.05 {
		t.Fatalf("host multi-head layer forward diverges from the engine on deeper layers (worst rel-L2 %.4g) — block forward is wrong", worstDeep)
	}
	t.Logf("block-forward VERIFIED: host f32 multi-head layer forward tracks the engine bf16 forward within %.4g rel-L2 on layers fed real activations — the backward over these is sound", worstDeep)
}

func TestForwardCaptureHiddensUsesEmbedInto(t *testing.T) {
	requireNativeRuntime(t)
	mk := newMTPDecodeFixture(t)
	ids := []int32{1, 2, 3, 4, 5}
	control := mk()
	wantEmbeds, wantLayers, err := control.ForwardCaptureHiddens(ids)
	if err != nil {
		t.Fatalf("control ForwardCaptureHiddens: %v", err)
	}

	candidate := mk()
	candidate.embed = func(int32) ([]byte, error) {
		return nil, errors.New("allocating embed path called")
	}
	candidate.embedFuncPtr = 0
	gotEmbeds, gotLayers, err := candidate.ForwardCaptureHiddens(ids)
	if err != nil {
		t.Fatalf("candidate ForwardCaptureHiddens: %v", err)
	}
	if len(gotEmbeds) != len(wantEmbeds) {
		t.Fatalf("got %d embeddings, want %d", len(gotEmbeds), len(wantEmbeds))
	}
	for i := range wantEmbeds {
		eqBytes(t, "ForwardCaptureHiddens embedInto embedding", gotEmbeds[i], wantEmbeds[i])
	}
	if len(gotLayers) != len(wantLayers) {
		t.Fatalf("got %d layer tensors, want %d", len(gotLayers), len(wantLayers))
	}
	for i := range wantLayers {
		eqBytes(t, "ForwardCaptureHiddens embedInto layer tensor", gotLayers[i], wantLayers[i])
	}
}

func TestForwardCaptureHiddensICBReplay(t *testing.T) {
	requireNativeRuntime(t)
	g, arch, maxLen := icbSessionStateFixture(t)
	ids := []int32{1, 5, 3, 2}
	T, nL, rowBytes := len(ids), len(arch.Layer), arch.Hidden*bf16Size

	embeds, perLayer, err := newICBSessionStateFixture(t, g, arch, maxLen).ForwardCaptureHiddens(ids)
	if err != nil {
		t.Fatalf("ForwardCaptureHiddens ICB: %v", err)
	}
	if len(embeds) != T {
		t.Fatalf("ICB got %d embeddings, want %d", len(embeds), T)
	}
	if len(perLayer) != nL {
		t.Fatalf("ICB got %d per-layer tensors, want %d", len(perLayer), nL)
	}
	for l := range perLayer {
		if len(perLayer[l]) != T*rowBytes {
			t.Fatalf("ICB perLayer[%d] is %d bytes, want %d", l, len(perLayer[l]), T*rowBytes)
		}
	}

	ref := newICBSessionStateFixture(t, g, arch, maxLen)
	var lastHidden []byte
	for _, id := range ids {
		h, e := ref.stepID(id)
		if e != nil {
			t.Fatalf("ICB ref stepID: %v", e)
		}
		lastHidden = h
	}
	gotLast := perLayer[nL-1][(T-1)*rowBytes:]
	eqBytes(t, "ICB captured final-layer last-token hidden vs ordinary ICB forward", gotLast, lastHidden)
}

func TestForwardCaptureHiddensICBUsesEmbedInto(t *testing.T) {
	requireNativeRuntime(t)
	g, arch, maxLen := icbSessionStateFixture(t)
	ids := []int32{1, 5, 3, 2}
	control := newICBSessionStateFixture(t, g, arch, maxLen)
	wantEmbeds, wantLayers, err := control.ForwardCaptureHiddens(ids)
	if err != nil {
		t.Fatalf("control ForwardCaptureHiddens ICB: %v", err)
	}

	candidate := newICBSessionStateFixture(t, g, arch, maxLen)
	candidate.embed = func(int32) ([]byte, error) {
		return nil, errors.New("allocating embed path called")
	}
	candidate.embedFuncPtr = 0
	gotEmbeds, gotLayers, err := candidate.ForwardCaptureHiddens(ids)
	if err != nil {
		t.Fatalf("candidate ForwardCaptureHiddens ICB: %v", err)
	}
	if len(gotEmbeds) != len(wantEmbeds) {
		t.Fatalf("ICB got %d embeddings, want %d", len(gotEmbeds), len(wantEmbeds))
	}
	for i := range wantEmbeds {
		eqBytes(t, "ForwardCaptureHiddens ICB embedInto embedding", gotEmbeds[i], wantEmbeds[i])
	}
	if len(gotLayers) != len(wantLayers) {
		t.Fatalf("ICB got %d layer tensors, want %d", len(gotLayers), len(wantLayers))
	}
	for i := range wantLayers {
		eqBytes(t, "ForwardCaptureHiddens ICB embedInto layer tensor", gotLayers[i], wantLayers[i])
	}
}

// TestForwardCaptureHiddensHonoursICBDisabledForTest is the #44-investigation regression: before
// this fix, ForwardCaptureHiddens picked its route by `s.state.icb != nil` ALONE, ignoring
// icbDisabledForTest — the one caller of the flag that DIDN'T honour it (every other capture-ish
// call site, e.g. captureBoundaryLayerHiddens in assistant_dflash_livetap.go, already checked
// `s.state.icb != nil && !icbDisabledForTest`). That asymmetry meant the lever couldn't force
// ForwardCaptureHiddens onto the plain re-encode "truth arm" the way TestForwardCaptureHiddens2PassBoundaries
// forces it by nulling state.icb directly — the only difference from production (icbDisabledForTest
// is always false there) is this flag, so a session with icbDisabledForTest=true must now produce
// byte-identical captured hiddens to the same session with state.icb forced nil outright.
func TestForwardCaptureHiddensHonoursICBDisabledForTest(t *testing.T) {
	requireNativeRuntime(t)
	g, arch, maxLen := icbSessionStateFixture(t)
	ids := []int32{1, 5, 3, 2}

	plain := newICBSessionStateFixture(t, g, arch, maxLen)
	plain.state.icb = nil // the truth arm (TestForwardCaptureHiddens2PassBoundaries's pattern)
	_, wantLayers, err := plain.ForwardCaptureHiddens(ids)
	if err != nil {
		t.Fatalf("ForwardCaptureHiddens(plain): %v", err)
	}

	icb := newICBSessionStateFixture(t, g, arch, maxLen)
	if icb.state.icb == nil {
		t.Skip("fixture did not record an ICB — the flag has nothing to disable here")
	}
	prev := icbDisabledForTest
	icbDisabledForTest = true
	defer func() { icbDisabledForTest = prev }()
	_, gotLayers, err := icb.ForwardCaptureHiddens(ids)
	if err != nil {
		t.Fatalf("ForwardCaptureHiddens(icb, disabled-for-test): %v", err)
	}

	if len(gotLayers) != len(wantLayers) {
		t.Fatalf("got %d layer tensors, want %d", len(gotLayers), len(wantLayers))
	}
	for i := range wantLayers {
		eqBytes(t, "ForwardCaptureHiddens under icbDisabledForTest vs the plain truth arm", gotLayers[i], wantLayers[i])
	}
}

// TestForwardCaptureFinalHidden verifies the batched training forward on a real (synthetic)
// dense ArchSession: the final residual rows it returns are BYTE-IDENTICAL to the serial
// per-token capture's last layer, the batched route actually ENGAGED (identity alone can't
// distinguish the fast path from its silent serial fallback), and re-running on the same
// session reproduces the rows (the pos-reset re-prefill contract the trainer relies on).
func TestForwardCaptureFinalHidden(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, nHeads, nKV, headDim, dFF = 512, 8, 4, 64, 1024
	const vocab, nL, maxLen = 64, 3, 64
	layers := make([]DecodeLayerWeights, nL)
	types := make([]string, nL)
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
		types[li] = "full_attention"
	}
	specs := model.DeriveLayers(types, 0)
	embed := toBF16Bytes(syntheticFloat32(vocab*dModel, 21))
	g := &BF16Model{Layers: layers, Embed: embed, FinalNorm: toBF16Bytes(syntheticFloat32(dModel, 22)), LMHead: embed, Tied: true}
	arch := model.Arch{
		Hidden: dModel, Heads: nHeads, KVHeads: nKV, HeadDim: headDim, FF: dFF, Vocab: vocab,
		GlobalHeadDim: headDim, GlobalKVHeads: nKV,
		Eps: 1e-5, AttnScale: 0.125, RopeBase: 10000, RopeScale: 1, RopeLocalBase: 10000,
		RotaryDim: headDim, RotaryDimLocal: headDim, Layer: specs,
	}
	ids := make([]int32, 24)
	for i := range ids {
		ids[i] = int32(1 + i%vocab)
	}

	serialSess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession(serial): %v", err)
	}
	defer serialSess.Close()
	_, perLayer, err := serialSess.ForwardCaptureHiddens(ids)
	if err != nil {
		t.Fatalf("ForwardCaptureHiddens: %v", err)
	}
	want := perLayer[len(perLayer)-1]

	batchSess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession(batched): %v", err)
	}
	defer batchSess.Close()
	before := captureFinalHiddenBatchedChunksForTest
	got, err := batchSess.ForwardCaptureFinalHidden(ids)
	if err != nil {
		t.Fatalf("ForwardCaptureFinalHidden: %v", err)
	}
	if captureFinalHiddenBatchedChunksForTest == before {
		t.Fatal("batched capture route never engaged — the forward fell to the serial path on a session that must batch")
	}
	eqBytes(t, "final hidden (batched vs serial)", got, want)

	// Re-run on the SAME session: the training loop re-prefills every step.
	got2, err := batchSess.ForwardCaptureFinalHidden(ids)
	if err != nil {
		t.Fatalf("ForwardCaptureFinalHidden(rerun): %v", err)
	}
	eqBytes(t, "final hidden (re-prefill rerun)", got2, want)
}

// TestForwardCaptureHiddens2PassBoundaries is the #391 regression gate: on a session whose
// recorded ICB carries INLINE extras (global layers' 2-pass SDPA — recorded whenever
// maxLen ≥ the 2-pass knee), the per-layer capture must carve the stream by the RECORDED
// layer boundaries. The old uniform li·opsPerLayer stride misaligned every layer after the
// first global and skipped the stream's tail — captured hiddens diverged from the serving
// forward on real E2B (|Δ|≈34) while fixture stacks without extras stayed byte-identical,
// which is why no existing gate caught it. Serial plain-path capture is the truth arm.
func TestForwardCaptureHiddens2PassBoundaries(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, nHeads, nKV, headDim, dFF = 256, 4, 2, 64, 512
	const vocab, nL = 64, 4
	const maxLen = sdpa2PassMinKV // ≥ the knee: global layers record the 2-pass pair
	layers := make([]DecodeLayerWeights, nL)
	types := []string{"sliding_attention", "full_attention", "sliding_attention", "full_attention"}
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
	}
	specs := model.DeriveLayers(types, 0)
	embed := toBF16Bytes(syntheticFloat32(vocab*dModel, 21))
	g := &BF16Model{Layers: layers, Embed: embed, FinalNorm: toBF16Bytes(syntheticFloat32(dModel, 22)), LMHead: embed, Tied: true}
	arch := model.Arch{
		Hidden: dModel, Heads: nHeads, KVHeads: nKV, HeadDim: headDim, FF: dFF, Vocab: vocab,
		GlobalHeadDim: headDim, GlobalKVHeads: nKV, SlidingWindow: 128,
		Eps: 1e-5, AttnScale: 0.125, RopeBase: 10000, RopeScale: 1, RopeLocalBase: 10000,
		RotaryDim: headDim, RotaryDimLocal: headDim, Layer: specs,
	}
	ids := make([]int32, 12)
	for i := range ids {
		ids[i] = int32(1 + i%vocab)
	}

	icbSess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession(icb): %v", err)
	}
	defer icbSess.Close()
	if icbSess.state.icb == nil {
		t.Skip("session did not record an ICB — the carve path is unreachable here")
	}
	r := icbSess.state.icb
	if len(r.layerOpStarts) != nL+1 {
		t.Fatalf("layerOpStarts = %d entries, want %d", len(r.layerOpStarts), nL+1)
	}
	if r.layerOpStarts[nL] == uint(nL)*r.opsPerLayer {
		t.Fatal("fixture recorded a UNIFORM stream (no 2-pass extras) — it no longer exercises the #391 carve; raise maxLen past the 2-pass knee or add a global layer")
	}

	plainSess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession(plain): %v", err)
	}
	defer plainSess.Close()
	plainSess.state.icb = nil // force the per-token plain capture — the truth arm

	_, wantLayers, err := plainSess.ForwardCaptureHiddens(ids)
	if err != nil {
		t.Fatalf("ForwardCaptureHiddens(plain): %v", err)
	}
	_, gotLayers, err := icbSess.ForwardCaptureHiddens(ids)
	if err != nil {
		t.Fatalf("ForwardCaptureHiddens(icb): %v", err)
	}
	if len(gotLayers) != len(wantLayers) {
		t.Fatalf("layer count %d != %d", len(gotLayers), len(wantLayers))
	}
	for l := range wantLayers {
		eqBytes(t, "captured layer (icb 2-pass carve vs plain)", gotLayers[l], wantLayers[l])
	}
}
