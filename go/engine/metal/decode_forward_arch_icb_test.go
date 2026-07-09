// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"os"
	"testing"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"github.com/tmc/apple/metal"
)

func TestArchICBReplayCachesLastOutContentsPointer(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("native init unavailable: %v", err)
	}
	buf := scratchBF16(4)
	first := toBF16Bytes([]float32{1, 2, 3, 4})
	copy(unsafe.Slice((*byte)(buf.Contents()), len(first)), first)

	r := &archICBReplay{lastOut: buf, dModel: 4}
	r.cacheLastOutContents()
	if r.lastOutPtr == nil {
		t.Fatal("lastOut contents pointer was not cached")
	}
	got := make([]byte, len(first))
	r.copyLastOutInto(got)
	if !bytes.Equal(got, first) {
		t.Fatalf("first cached lastOut copy = %v, want %v", got, first)
	}

	second := toBF16Bytes([]float32{5, 6, 7, 8})
	copy(unsafe.Slice((*byte)(buf.Contents()), len(second)), second)
	r.copyLastOutInto(got)
	if !bytes.Equal(got, second) {
		t.Fatalf("second cached lastOut copy = %v, want %v", got, second)
	}
}

func TestArchICBReplayCachesStepContentsPointers(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("native init unavailable: %v", err)
	}
	input := toBF16Bytes([]float32{1, 2, 3, 4})
	pli := toBF16Bytes([]float32{5, 6, 7, 8})
	r := &archICBReplay{
		offBuf:        device.NewBufferWithLengthOptions(4, metal.MTLResourceStorageModeShared),
		nGlobalBuf:    device.NewBufferWithLengthOptions(4, metal.MTLResourceStorageModeShared),
		nSlidingBuf:   device.NewBufferWithLengthOptions(4, metal.MTLResourceStorageModeShared),
		ping0:         scratchBF16(4),
		pleInput:      scratchBF16(4),
		specs:         []model.LayerSpec{{CacheIndex: -1}},
		hasPLE:        true,
		nLayers:       1,
		plePliDim:     4,
		slidingWindow: 3,
		dModel:        4,
	}
	r.cacheStepContents()
	if r.offPtr == nil || r.nGlobalPtr == nil || r.nSlidingPtr == nil || r.ping0Ptr == nil || r.pleInputPtr == nil {
		t.Fatal("step contents pointers were not cached")
	}
	r.prepareStep(input, 5, pli)
	if got := *(*int32)(r.offBuf.Contents()); got != 5 {
		t.Fatalf("offBuf = %d, want 5", got)
	}
	if got := *(*int32)(r.nGlobalBuf.Contents()); got != 6 {
		t.Fatalf("nGlobalBuf = %d, want 6", got)
	}
	if got := *(*int32)(r.nSlidingBuf.Contents()); got != 3 {
		t.Fatalf("nSlidingBuf = %d, want 3", got)
	}
	gotInput := unsafe.Slice((*byte)(r.ping0.Contents()), len(input))
	if !bytes.Equal(gotInput, input) {
		t.Fatalf("ping0 input = %v, want %v", gotInput, input)
	}
	gotPLE := unsafe.Slice((*byte)(r.pleInput.Contents()), len(pli))
	if !bytes.Equal(gotPLE, pli) {
		t.Fatalf("ple input = %v, want %v", gotPLE, pli)
	}
}

func TestDecodeForwardArchICBAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers, maxLen = 64, 1, 1, 64, 128, 32, 1, 4
	arch := archFixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	inputs := decodeInputsFixture(2, dModel)
	layers := []DecodeLayerWeights{decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)}
	if _, err := DecodeForwardArchICB(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm); err != nil {
		t.Fatalf("DecodeForwardArchICB warmup: %v", err)
	}

	var forwardErr error
	allocs := testing.AllocsPerRun(5, func() {
		_, forwardErr = DecodeForwardArchICB(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm)
	})
	if forwardErr != nil {
		t.Fatalf("DecodeForwardArchICB: %v", forwardErr)
	}
	if allocs > 240 {
		t.Fatalf("DecodeForwardArchICB allocations = %.0f, want <= 240", allocs)
	}
}

func TestDecodeForwardArchICBIntoReusesOutputBacking(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers, maxLen = 64, 1, 1, 64, 128, 32, 1, 4
	arch := archFixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	inputs := decodeInputsFixture(2, dModel)
	layers := []DecodeLayerWeights{decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)}
	want, err := DecodeForwardArchICB(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm)
	if err != nil {
		t.Fatalf("DecodeForwardArchICB reference: %v", err)
	}
	out := [][]byte{
		bytes.Repeat([]byte{0xa5}, dModel*bf16Size),
		bytes.Repeat([]byte{0x5a}, dModel*bf16Size),
	}
	ptrs := []unsafe.Pointer{unsafe.Pointer(&out[0][0]), unsafe.Pointer(&out[1][0])}

	got, err := DecodeForwardArchICBInto(out, inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm)
	if err != nil {
		t.Fatalf("DecodeForwardArchICBInto: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("DecodeForwardArchICBInto returned %d outputs, want %d", len(got), len(want))
	}
	for tok := range want {
		if len(got[tok]) != dModel*bf16Size || unsafe.Pointer(&got[tok][0]) != ptrs[tok] {
			t.Fatalf("DecodeForwardArchICBInto token %d did not reuse caller-owned output backing", tok)
		}
		eqBytes(t, "DecodeForwardArchICBInto token", got[tok], want[tok])
	}
}

func TestArchICBReplayScratchOutputViewsUseCallerBacking(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, nLayers = 64, 1, 1, 64, 128, 1
	sc := newArchICBReplayScratch(dModel, nHeads*headDim, nKV*headDim, dFF, dFF, nLayers, 0, 0, true, false)
	t.Cleanup(sc.closeOutputViews)

	out := [][]byte{
		bytes.Repeat([]byte{0xa5}, dModel*bf16Size),
		bytes.Repeat([]byte{0x5a}, dModel*bf16Size),
	}
	views, ok := sc.outputViews(out, dModel*bf16Size)
	if !ok {
		t.Fatal("arch outputViews did not create no-copy views for caller-owned outputs")
	}
	for i := range out {
		if views[i] == nil || views[i].Contents() != unsafe.Pointer(&out[i][0]) {
			t.Fatalf("arch output view %d not backed by caller output slice", i)
		}
	}
	firstID := views[0].GetID()
	reused, ok := sc.outputViews(out, dModel*bf16Size)
	if !ok {
		t.Fatal("arch outputViews did not reuse no-copy views for unchanged caller outputs")
	}
	if reused[0].GetID() != firstID {
		t.Fatal("arch outputViews rebuilt an unchanged caller output view")
	}
}

func TestArchICBReplayScratchOutputViewsReusePinnedOwnerBuffers(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, nLayers = 64, 1, 1, 64, 128, 1
	pinned := make([]*pinnedNoCopyBytes, 2)
	t.Cleanup(func() {
		for _, p := range pinned {
			if p != nil {
				p.Close()
			}
		}
	})
	sc := newArchICBReplayScratch(dModel, nHeads*headDim, nKV*headDim, dFF, dFF, nLayers, 0, 0, true, false)
	t.Cleanup(sc.closeOutputViews)

	outputs := make([][]byte, len(pinned))
	for i := range pinned {
		var err error
		pinned[i], err = newPinnedNoCopyBytes(dModel * bf16Size)
		if err != nil {
			t.Fatalf("newPinnedNoCopyBytes(%d): %v", i, err)
		}
		outputs[i] = pinned[i].bytes
	}
	views, ok := sc.outputViews(outputs, dModel*bf16Size)
	if !ok {
		t.Fatal("arch outputViews did not create no-copy views for pinned-owner outputs")
	}
	for i := range pinned {
		requirePinnedOwnerBuffer(t, core.Sprintf("arch output view %d", i), views[i], pinned[i])
	}
}

func TestArchICBReplayDirectOutputResourcesIncludeCallerBacking(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, nLayers = 64, 1, 1, 64, 128, 1
	sc := newArchICBReplayScratch(dModel, nHeads*headDim, nKV*headDim, dFF, dFF, nLayers, 0, 0, true, false)
	t.Cleanup(sc.closeOutputViews)
	base := []metal.MTLResource{scratchBF16(1)}
	r := &archICBReplay{scratch: sc, residentRes: base, hasFinalOut: true}
	out := [][]byte{
		bytes.Repeat([]byte{0xa5}, dModel*bf16Size),
		bytes.Repeat([]byte{0x5a}, dModel*bf16Size),
	}

	views, resources, ids, ok := r.directOutputResources(out, dModel*bf16Size)
	if !ok {
		t.Fatal("directOutputResources did not create caller-backed output resources")
	}
	if len(views) != len(out) || len(resources) != len(base)+len(out) || len(ids) != len(resources) {
		t.Fatalf("directOutputResources sizes views=%d resources=%d ids=%d", len(views), len(resources), len(ids))
	}
	for i := range out {
		if views[i] == nil || views[i].Contents() != unsafe.Pointer(&out[i][0]) {
			t.Fatalf("direct output view %d not backed by caller output slice", i)
		}
	}
}

// TestDecodeForwardArchICB gates the arch-driven cache-grow ICB (the encode-bypass
// replay) against the proven re-encode arch forward DecodeForwardArch — byte-for-byte
// across every arch axis: all-owner/global, KV-share, sliding-window, and KV-share +
// sliding combined. Same weights + inputs + arch → the ICB replay must equal the
// re-encode path exactly. MoE layers route through the MoE-capable re-encode
// path instead of rejecting the direct ICB API.
func TestDecodeForwardArchICB(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF = 512, 8, 4, 64, 1024
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	const maxLen = 8

	mkInputs := func(n int) [][]byte {
		in := make([][]byte, n)
		for i := range in {
			f := make([]float32, dModel)
			for j := range f {
				f[j] = float32((j*(i+3)+5)%97-48) * 0.02
			}
			in[i] = toBF16Bytes(f)
		}
		return in
	}
	buildLayers := func(n int) []DecodeLayerWeights {
		ls := make([]DecodeLayerWeights, n)
		for li := range ls {
			ls[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
		}
		return ls
	}

	// check: DecodeForwardArchICB ≡ DecodeForwardArch byte-for-byte on the given arch.
	check := func(name string, layers []DecodeLayerWeights, specs []model.LayerSpec, T, slidingWindow int) {
		inputs := mkInputs(T)
		got, err := DecodeForwardArchICB(inputs, layers, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, slidingWindow, base, scale, eps, false)
		if err != nil {
			t.Fatalf("%s: DecodeForwardArchICB: %v", name, err)
		}
		want, err := DecodeForwardArch(inputs, layers, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, slidingWindow, base, scale, eps, false)
		if err != nil {
			t.Fatalf("%s: DecodeForwardArch: %v", name, err)
		}
		for tok := range T {
			eqBytes(t, core.Sprintf("%s tok%d", name, tok), got[tok], want[tok])
		}
	}

	// (a) all-owner, all-global.
	full3 := []string{"full_attention", "full_attention", "full_attention"}
	check("all-owner/global", buildLayers(3), model.DeriveLayers(full3, 0), 4, 0)

	// (b) KV-share: layer 1 shares layer 0's cache.
	check("kv-share", buildLayers(2), model.DeriveLayers([]string{"full_attention", "full_attention"}, 1), 4, 0)

	// (c) sliding-window: all sliding, W=3 over 6 tokens (toks 3..5 clip).
	slide3 := []string{"sliding_attention", "sliding_attention", "sliding_attention"}
	check("sliding-W3", buildLayers(3), model.DeriveLayers(slide3, 0), 6, 3)

	// (d) KV-share + sliding combined: 4 layers, mixed types, 2 shared → the last
	// sliding/full layers share the matching owner's cache, sliding layers windowed.
	mixed := []string{"sliding_attention", "full_attention", "sliding_attention", "full_attention"}
	check("kv-share+sliding", buildLayers(4), model.DeriveLayers(mixed, 2), 6, 3)

	// (e) MoE falls back to the re-encode arch path instead of rejecting the API.
	moeLayers := buildLayers(2)
	moeSpecs := model.DeriveLayers([]string{"full_attention", "full_attention"}, 0)
	moeSpecs[1].MoE = true
	moeLayers[1].MoE = buildMoEWeights(4, 2, dModel, dFF, 768, 700)
	moeInputs := mkInputs(3)
	gotMoE, err := DecodeForwardArchICB(moeInputs, moeLayers, moeSpecs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArchICB MoE fallback: %v", err)
	}
	wantMoE, err := DecodeForwardArch(moeInputs, moeLayers, moeSpecs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArch MoE: %v", err)
	}
	for tok := range moeInputs {
		eqBytes(t, core.Sprintf("moe fallback tok%d", tok), gotMoE[tok], wantMoE[tok])
	}

	t.Logf("arch ICB: replay ≡ DecodeForwardArch byte-for-byte across all-owner/global, KV-share, sliding(W=3), and KV-share+sliding; direct MoE ICB API falls back to re-encode parity")
}

// TestDecodeForwardArchICBNorms gates the gemma4 norms on the ICB path: with all four
// gemma4 norms set (QK-norm + post-attn + post-FF), the cache-grow ICB replay equals the
// now-norm-complete re-encode arch forward byte-for-byte — across a mixed sliding +
// KV-share arch, for both bf16 and 4-bit — and differs from the same arch with the norms
// dropped (the recorded norm ops are genuinely live).
func TestDecodeForwardArchICBNorms(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, gs, bits = 512, 8, 4, 64, 1024, 64, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	const maxLen, T, W = 8, 6, 3
	mixed := []string{"sliding_attention", "full_attention", "sliding_attention", "full_attention"}
	specs := model.DeriveLayers(mixed, 2)
	nL := len(specs)

	inputs := make([][]byte, T)
	for i := range inputs {
		f := make([]float32, dModel)
		for j := range f {
			f[j] = float32((j*(i+3)+5)%97-48) * 0.02
		}
		inputs[i] = toBF16Bytes(f)
	}
	dnorm := func(salt int) []byte {
		f := make([]float32, dModel)
		for j := range f {
			f[j] = float32((j*salt+3)%29-14) * 0.03
		}
		return toBF16Bytes(f)
	}
	hnorm := func(salt int) []byte {
		f := make([]float32, headDim)
		for j := range f {
			f[j] = float32((j*salt+5)%23-11) * 0.04
		}
		return toBF16Bytes(f)
	}

	// bf16: ICB ≡ re-encode, with the four norms.
	layers := make([]DecodeLayerWeights, nL)
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
		layers[li].QNormW, layers[li].KNormW = hnorm(li*4+1), hnorm(li*4+2)
		layers[li].PostAttnNormW, layers[li].PostFFNormW = dnorm(li*4+3), dnorm(li*4+4)
	}
	gotICB, err := DecodeForwardArchICB(inputs, layers, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, W, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArchICB norms: %v", err)
	}
	want, err := DecodeForwardArch(inputs, layers, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, W, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArch norms: %v", err)
	}
	for tok := range T {
		eqBytes(t, core.Sprintf("bf16 ICB-norms vs re-encode tok%d", tok), gotICB[tok], want[tok])
	}

	// non-vacuous: dropping the norms changes the ICB output.
	bare := make([]DecodeLayerWeights, nL)
	copy(bare, layers)
	for li := range bare {
		bare[li].QNormW, bare[li].KNormW, bare[li].PostAttnNormW, bare[li].PostFFNormW = nil, nil, nil, nil
	}
	gotBare, err := DecodeForwardArchICB(inputs, bare, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, W, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArchICB bare: %v", err)
	}
	if !lastTokenDiffers(gotICB, gotBare) {
		t.Fatal("ICB norms made no difference — the recorded norm ops were not live")
	}

	// 4-bit: ICB ≡ re-encode, with the four norms.
	ql := make([]QuantizedLayerWeights, nL)
	for li := range ql {
		ql[li] = buildQuantLayer(t, dModel, nHeads, nKV, headDim, dFF, gs, bits, (li+1)*100)
		ql[li].QNormW, ql[li].KNormW = hnorm(li*4+1), hnorm(li*4+2)
		ql[li].PostAttnNormW, ql[li].PostFFNormW = dnorm(li*4+3), dnorm(li*4+4)
	}
	gotQICB, err := DecodeForwardArchICBQuant(inputs, ql, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, W, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArchICBQuant norms: %v", err)
	}
	wantQ, err := DecodeForwardArchQuant(inputs, ql, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, W, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArchQuant norms: %v", err)
	}
	for tok := range T {
		eqBytes(t, core.Sprintf("quant ICB-norms vs re-encode tok%d", tok), gotQICB[tok], wantQ[tok])
	}

	t.Logf("arch ICB norms: replay ≡ norm-complete re-encode byte-for-byte (bf16 + 4-bit) across sliding+KV-share with QK-norm + post-attn + post-FF, and differs from without — the ICB fast path is now gemma4-norm-complete")
}

// TestDecodeForwardArchICBMixedHeadDimFallback gates the mixed-head-dim fallback in BOTH whole-sequence
// ICB forwards (bf16 DecodeForwardArchICB + 4-bit DecodeForwardArchICBQuant — both production paths via
// backend.go). These record a single uniform projection shape + base-rope spectrum and take no
// proportional-rope params, so they cannot represent gemma4's wider global head dim (head_dim 512 vs
// sliding 256, on proportional partial rope). On a mixed-head-dim arch — a sliding layer (head_dim 64)
// + a global layer (head_dim 128), gemma4 E2B's 256/512 in miniature — they MUST fall back to the
// per-layer-correct re-encode forward and return its output BYTE-for-byte, never the broken ICB
// recording (which diverged at the first global layer — see q4_icb_localize_test for the session path,
// where the fast per-hd ICB IS correct). Drop the fallback and the broken ICB output makes this fail.
func TestDecodeForwardArchICBMixedHeadDimFallback(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, globalHeadDim, dFF, gs, bits = 256, 2, 1, 64, 128, 512, 64, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	const maxLen, T, W = 8, 4, 3
	specs := []model.LayerSpec{
		{Attention: model.SlidingAttention, KVShareFrom: 0, CacheIndex: 0, HeadDim: headDim, KVHeads: nKV},
		{Attention: model.GlobalAttention, KVShareFrom: 1, CacheIndex: 1, HeadDim: globalHeadDim, KVHeads: nKV},
	}
	inputs := make([][]byte, T)
	for i := range inputs {
		inputs[i] = toBF16Bytes(syntheticFloat32(dModel, i+3))
	}

	// bf16 whole-seq ICB (the backend.go production path) — wider global layer + value-norm ON, so the
	// uniform-shape recorder must hand off to the per-layer-correct re-encode forward.
	layers := []DecodeLayerWeights{
		forwardLayer(dModel, nHeads, nKV, headDim, dFF, 100),
		forwardLayer(dModel, nHeads, nKV, globalHeadDim, dFF, 200),
	}
	gotICB, err := DecodeForwardArchICB(inputs, layers, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, W, base, scale, eps, true)
	if err != nil {
		t.Fatalf("DecodeForwardArchICB: %v", err)
	}
	want, err := DecodeForwardArch(inputs, layers, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, W, base, scale, eps, true)
	if err != nil {
		t.Fatalf("DecodeForwardArch: %v", err)
	}
	for tok := range T {
		eqBytes(t, core.Sprintf("bf16 mixed-head-dim ICB (fallback) vs re-encode tok%d", tok), gotICB[tok], want[tok])
	}

	// 4-bit quant whole-seq: DecodeForwardArchICBQuant must likewise fall back to the re-encode forward
	// DecodeForwardArchQuant (now per-layer-head-dim correct) and match it byte-for-byte — its own ICB
	// recorder ropes the wider global layer wrong past pos 0 (simpleICBRope).
	ql := []QuantizedLayerWeights{
		buildQuantLayer(t, dModel, nHeads, nKV, headDim, dFF, gs, bits, 100),
		buildQuantLayer(t, dModel, nHeads, nKV, globalHeadDim, dFF, gs, bits, 200),
	}
	gotQ, err := DecodeForwardArchICBQuant(inputs, ql, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, W, base, scale, eps, true)
	if err != nil {
		t.Fatalf("DecodeForwardArchICBQuant: %v", err)
	}
	wantQ, err := DecodeForwardArchQuant(inputs, ql, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, W, base, scale, eps, true)
	if err != nil {
		t.Fatalf("DecodeForwardArchQuant: %v", err)
	}
	for tok := range T {
		eqBytes(t, core.Sprintf("quant mixed-head-dim ICB (fallback) vs re-encode tok%d", tok), gotQ[tok], wantQ[tok])
	}
}

// TestDecodeForwardArchICBHeteroDFF gates the HETEROGENEOUS-shape ICB recorder: a two-layer
// stack whose layers have DIFFERENT FFN widths (gemma4 E2B/E4B MatFormer varies dFF per
// layer). The arch is the simplest possible — all-owner full_attention, no sliding — so the
// ONLY thing varying across the two recorded layers is dFF. It proves the cache-grow ICB
// recorder + replay handles per-layer-varying FFN width byte-for-byte against the non-ICB
// re-encode path, for both bf16 and 4-bit:
//
//   - bf16: DecodeForwardArchICB ≡ DecodeForwardArch — the bf16 oracle has NO weight-size
//     validation, so this is the UNMODIFIED-reference anchor (the core's maxDFF scratch,
//     per-dFF count buffers, and per-layer dispatch widths are all exercised here).
//   - 4-bit: DecodeForwardArchICBQuant ≡ DecodeForwardArchQuant — exercises the per-distinct-dFF
//     qmv PSO + dim-scalar keying in the quant wrapper.
//
// The uniform dFF parameter is set to the WIDER width; layer 0 carries the narrower width via
// its per-layer DFF field. This is "the recorder handles per-layer-varying shapes" (step 1) —
// no PLE, no per-layer head_dim, no real E2B yet.
func TestDecodeForwardArchICBHeteroDFF(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, gs, bits = 512, 8, 4, 64, 64, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	const maxLen, T = 8, 4
	const dffNarrow, dffWide = 768, 1024 // both ÷ gs (down's inDim = lff must be a GroupSize multiple)

	mkInputs := func(n int) [][]byte {
		in := make([][]byte, n)
		for i := range in {
			f := make([]float32, dModel)
			for j := range f {
				f[j] = float32((j*(i+3)+5)%97-48) * 0.02
			}
			in[i] = toBF16Bytes(f)
		}
		return in
	}
	inputs := mkInputs(T)
	// all-owner, all-global, no sliding: only dFF varies between the two layers.
	specs := model.DeriveLayers([]string{"full_attention", "full_attention"}, 0)
	dffs := []int{dffNarrow, dffWide}

	// --- bf16 anchor: ICB ≡ DecodeForwardArch (unmodified oracle), heterogeneous dFF.
	bf16Layers := make([]DecodeLayerWeights, 2)
	for li := range bf16Layers {
		bf16Layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dffs[li], (li+1)*100)
		bf16Layers[li].DFF = dffs[li] // each layer declares its own FFN width
	}
	gotBF, err := DecodeForwardArchICB(inputs, bf16Layers, specs, dModel, nHeads, nKV, headDim, maxLen, dffWide, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("bf16 hetero ICB: %v", err)
	}
	wantBF, err := DecodeForwardArch(inputs, bf16Layers, specs, dModel, nHeads, nKV, headDim, maxLen, dffWide, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("bf16 hetero re-encode: %v", err)
	}
	for tok := range T {
		eqBytes(t, core.Sprintf("bf16 hetero-dFF tok%d", tok), gotBF[tok], wantBF[tok])
	}

	// --- 4-bit: ICB ≡ DecodeForwardArchQuant, heterogeneous dFF.
	qLayers := make([]QuantizedLayerWeights, 2)
	for li := range qLayers {
		qLayers[li] = buildQuantLayer(t, dModel, nHeads, nKV, headDim, dffs[li], gs, bits, (li+1)*100)
		qLayers[li].DFF = dffs[li]
	}
	gotQ, err := DecodeForwardArchICBQuant(inputs, qLayers, specs, dModel, nHeads, nKV, headDim, maxLen, dffWide, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("quant hetero ICB: %v", err)
	}
	wantQ, err := DecodeForwardArchQuant(inputs, qLayers, specs, dModel, nHeads, nKV, headDim, maxLen, dffWide, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("quant hetero re-encode: %v", err)
	}
	for tok := range T {
		eqBytes(t, core.Sprintf("quant hetero-dFF tok%d", tok), gotQ[tok], wantQ[tok])
	}

	t.Logf("hetero-dFF ICB: replay ≡ re-encode byte-for-byte (bf16 + 4-bit) with two layers at dFF=%d and dFF=%d — the cache-grow ICB recorder handles per-layer-varying FFN width", dffNarrow, dffWide)
}

// TestDecodeForwardArchICBMixedKEqV gates the PER-LAYER K==V projection choice — the 12B-unified
// shape: sliding layers carry their own V weight, global layers don't (V rides the k-proj). The
// layer-0-derived choice this replaced picked ONE projection index for every layer, so a mixed
// model (layer 0 sliding WITH V) projected the global layers' V from an EMPTY weight slot —
// garbage V rows, the #254 real-12B garbage decode. Uniform head dim keeps the whole-seq ICB on
// the recorded path (no mixed-hd fallback), so the recorded per-layer choice is exactly what runs,
// gated byte-for-byte against the per-layer-correct re-encode oracle.
func TestDecodeForwardArchICBMixedKEqV(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF = 512, 8, 4, 64, 1024
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	const maxLen = 8
	const T, slidingWindow = 5, 3

	inputs := make([][]byte, T)
	for i := range inputs {
		f := make([]float32, dModel)
		for j := range f {
			f[j] = float32((j*(i+3)+5)%97-48) * 0.02
		}
		inputs[i] = toBF16Bytes(f)
	}
	layers := []DecodeLayerWeights{
		forwardLayer(dModel, nHeads, nKV, headDim, dFF, 100),
		forwardLayer(dModel, nHeads, nKV, headDim, dFF, 200),
		forwardLayer(dModel, nHeads, nKV, headDim, dFF, 300),
	}
	layers[1].WV = nil // the global K==V layer: V must ride the k-proj, NOT an empty V slot
	specs := model.DeriveLayers([]string{"sliding_attention", "full_attention", "sliding_attention"}, 0)

	got, err := DecodeForwardArchICB(inputs, layers, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, slidingWindow, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArchICB mixed K==V: %v", err)
	}
	want, err := DecodeForwardArch(inputs, layers, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, slidingWindow, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArch mixed K==V: %v", err)
	}
	for tok := range T {
		eqBytes(t, core.Sprintf("mixed-keqv tok%d", tok), got[tok], want[tok])
	}
	t.Logf("arch ICB mixed K==V (sliding-with-V + global-without-V) ≡ re-encode oracle byte-for-byte over %d tokens", T)
}
