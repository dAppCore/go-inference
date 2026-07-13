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
)

// archShareRef is the arch-aware oracle for DecodeForwardArch, composed from the
// parity-proven standalone ops: owner layers project+append+attend their own
// seq-major cache; sharer layers project only Q and attend the OWNER's cache (read
// head-major for the proven SDPA). Mirrors DecodeForwardArch op-for-op.
func archShareRef(t *testing.T, layers []DecodeLayerWeights, specs []model.LayerSpec, inputs [][]byte, dModel, nHeads, nKV, headDim, dFF, maxLen, slidingWindow int, base, scale, eps float32) [][]byte {
	t.Helper()
	qDim, kvDim := nHeads*headDim, nKV*headDim
	rowBytes := kvDim * bf16Size
	nLayers, T := len(layers), len(inputs)
	must := func(b []byte, err error) []byte {
		if err != nil {
			t.Fatalf("archShareRef op: %v", err)
		}
		return b
	}
	kC := make([][]byte, nLayers)
	vC := make([][]byte, nLayers)
	for li := range specs {
		if specs[li].OwnsCache() {
			kC[li] = make([]byte, maxLen*rowBytes)
			vC[li] = make([]byte, maxLen*rowBytes)
		}
	}
	out := make([][]byte, T)
	for tok := range T {
		x := inputs[tok]
		for li := range nLayers {
			w := layers[li]
			normed := must(RMSNormBF16(x, w.AttnNormW, 1, dModel, eps))
			qr := must(RoPEBF16(must(MatVecBF16(w.WQ, normed, qDim, dModel)), 1, nHeads, headDim, base, scale, tok, false))
			var aK, aV []byte
			if specs[li].OwnsCache() {
				knew := must(RoPEBF16(must(MatVecBF16(w.WK, normed, kvDim, dModel)), 1, nKV, headDim, base, scale, tok, false))
				vnew := must(MatVecBF16(w.WV, normed, kvDim, dModel))
				copy(kC[li][tok*rowBytes:(tok+1)*rowBytes], knew)
				copy(vC[li][tok*rowBytes:(tok+1)*rowBytes], vnew)
				aK, aV = kC[li], vC[li]
			} else {
				own := specs[li].KVShareFrom
				aK, aV = kC[own], vC[own] // owner wrote row tok earlier this token
			}
			slideW := 0
			if specs[li].Attention == model.SlidingAttention {
				slideW = slidingWindow
			}
			start, n := slideWindow(tok, slideW)
			off := start * rowBytes
			attn := must(SDPA(qr, seqToHeadMajor(aK[off:], nKV, headDim, n), seqToHeadMajor(aV[off:], nKV, headDim, n), 1, nHeads, nKV, headDim, n, scale))
			h := must(AddBF16(x, must(MatVecBF16(w.WO, attn, dModel, qDim))))
			if w.MoE != nil {
				x = moeBlockRef(t, h, *w.MoE, dModel, dFF, eps) // dual-branch MoE FFN
			} else {
				x = must(MLPBlockBF16(h, w.MLPNormW, w.WGate, w.WUp, w.WDown, dModel, dFF, eps))
			}
		}
		out[tok] = x
	}
	return out
}

// TestDecodeForwardArch gates the executor's first slice — the arch-driven forward
// honouring KV-cache-sharing. (a) an all-owner arch is byte-for-byte the proven
// DecodeForward (the arch consumes the spec but routes nothing → identical), and
// equals the composed reference. (b) a 2-layer arch where layer 1 SHARES layer 0's
// cache equals the reference where layer 1 attends layer 0's KV — proving the
// sharer skips its own K/V and reads the owner's, the cache-topology made live.
func TestDecodeForwardArch(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF = 512, 8, 4, 64, 1024
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	const T, maxLen = 4, 8
	inputs := make([][]byte, T)
	for i := range inputs {
		f := make([]float32, dModel)
		for j := range f {
			f[j] = float32((j*(i+3)+5)%97-48) * 0.02
		}
		inputs[i] = toBF16Bytes(f)
	}

	// (a) all-owner ≡ DecodeForward AND ≡ the reference
	const nL = 3
	layers := make([]DecodeLayerWeights, nL)
	ownTypes := make([]string, nL)
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
		ownTypes[li] = "full_attention"
	}
	specsOwn := model.DeriveLayers(ownTypes, 0)
	ref0, err := DecodeForward(inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	if err != nil {
		t.Fatalf("DecodeForward: %v", err)
	}
	gotOwn, err := DecodeForwardArch(inputs, layers, specsOwn, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArch all-owner: %v", err)
	}
	refOwn := archShareRef(t, layers, specsOwn, inputs, dModel, nHeads, nKV, headDim, dFF, maxLen, 0, base, scale, eps)
	for tok := range T {
		eqBytes(t, core.Sprintf("all-owner vs DecodeForward tok%d", tok), gotOwn[tok], ref0[tok])
		eqBytes(t, core.Sprintf("all-owner vs ref tok%d", tok), gotOwn[tok], refOwn[tok])
	}

	// (b) KV-share: 2 layers, layer 1 shares layer 0's cache
	layers2 := []DecodeLayerWeights{
		forwardLayer(dModel, nHeads, nKV, headDim, dFF, 100),
		forwardLayer(dModel, nHeads, nKV, headDim, dFF, 200),
	}
	specsShare := model.DeriveLayers([]string{"full_attention", "full_attention"}, 1)
	if specsShare[1].OwnsCache() || specsShare[1].KVShareFrom != 0 {
		t.Fatalf("expected layer 1 to share layer 0: %+v", specsShare[1])
	}
	gotShare, err := DecodeForwardArch(inputs, layers2, specsShare, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArch share: %v", err)
	}
	refShare := archShareRef(t, layers2, specsShare, inputs, dModel, nHeads, nKV, headDim, dFF, maxLen, 0, base, scale, eps)
	for tok := range T {
		eqBytes(t, core.Sprintf("KV-share vs ref tok%d", tok), gotShare[tok], refShare[tok])
	}

	// (c) sliding-window: W=3 with T2=6 tokens (so toks 3..5 clip to the last 3),
	// a sliding arch all-owner. Gated vs the windowed reference — proving sliding
	// layers attend only the last W cache rows. Also assert it DIFFERS from the
	// global forward on the same weights (the window genuinely clips, not vacuous).
	const W, T2, maxLen2 = 3, 6, 8
	in2 := make([][]byte, T2)
	for i := range in2 {
		f := make([]float32, dModel)
		for j := range f {
			f[j] = float32((j*(i+2)+3)%89-44) * 0.02
		}
		in2[i] = toBF16Bytes(f)
	}
	slideTypes := make([]string, nL)
	for li := range slideTypes {
		slideTypes[li] = "sliding_attention"
	}
	specsSlide := model.DeriveLayers(slideTypes, 0) // all sliding, all own
	gotSlide, err := DecodeForwardArch(in2, layers, specsSlide, dModel, nHeads, nKV, headDim, maxLen2, dFF, W, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArch sliding: %v", err)
	}
	refSlide := archShareRef(t, layers, specsSlide, in2, dModel, nHeads, nKV, headDim, dFF, maxLen2, W, base, scale, eps)
	for tok := range T2 {
		eqBytes(t, core.Sprintf("sliding vs windowed ref tok%d", tok), gotSlide[tok], refSlide[tok])
	}
	// the window must actually clip: full-attention on the same weights differs at a
	// token past the window (tok 5 sees all 6 vs only the last 3).
	gotFull := archShareRef(t, layers, model.DeriveLayers(slideTypes, 0), in2, dModel, nHeads, nKV, headDim, dFF, maxLen2, 0, base, scale, eps)
	same := true
	for i := range gotSlide[T2-1] {
		if gotSlide[T2-1][i] != gotFull[T2-1][i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("sliding (W=3) produced the same last-token output as full attention over 6 tokens — window did not clip")
	}
	t.Logf("executor: DecodeForwardArch honours the arch — all-owner ≡ DecodeForward; KV-share ≡ ref; sliding-window (W=%d, %d tokens) ≡ windowed ref and clips vs full attention", W, T2)
}

func TestDecodeForwardArchIntoReusesOutputBacking(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers, maxLen = 64, 1, 1, 64, 128, 32, 1, 4
	arch := archFixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	inputs := decodeInputsFixture(2, dModel)
	layers := []DecodeLayerWeights{decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)}
	want, err := DecodeForwardArch(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm)
	if err != nil {
		t.Fatalf("DecodeForwardArch reference: %v", err)
	}
	out := [][]byte{
		bytes.Repeat([]byte{0xa5}, dModel*bf16Size),
		bytes.Repeat([]byte{0x5a}, dModel*bf16Size),
	}
	ptrs := []unsafe.Pointer{unsafe.Pointer(&out[0][0]), unsafe.Pointer(&out[1][0])}

	got, err := DecodeForwardArchInto(out, inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm)
	if err != nil {
		t.Fatalf("DecodeForwardArchInto: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("DecodeForwardArchInto returned %d outputs, want %d", len(got), len(want))
	}
	for tok := range want {
		if len(got[tok]) != dModel*bf16Size || unsafe.Pointer(&got[tok][0]) != ptrs[tok] {
			t.Fatalf("DecodeForwardArchInto token %d did not reuse caller-owned output backing", tok)
		}
		eqBytes(t, "DecodeForwardArchInto token", got[tok], want[tok])
	}
}

func TestDecodeForwardArchAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers, maxLen = 64, 1, 1, 64, 128, 32, 1, 4
	arch := archFixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	inputs := decodeInputsFixture(2, dModel)
	layers := []DecodeLayerWeights{decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)}
	if _, err := DecodeForwardArch(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm); err != nil {
		t.Fatalf("DecodeForwardArch warmup: %v", err)
	}

	var forwardErr error
	allocs := testing.AllocsPerRun(5, func() {
		_, forwardErr = DecodeForwardArch(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm)
	})
	if forwardErr != nil {
		t.Fatalf("DecodeForwardArch: %v", forwardErr)
	}
	if allocs > 20 {
		t.Fatalf("DecodeForwardArch allocations = %.0f, want <= 20", allocs)
	}
}

func TestDecodeForwardArchMoEAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers, maxLen = 64, 1, 1, 64, 128, 32, 1, 4
	const numExperts, topK, expertDFF = 4, 2, 96
	arch := archFixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	arch.Layer[0].MoE = true
	inputs := decodeInputsFixture(2, dModel)
	layers := []DecodeLayerWeights{decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)}
	layers[0].MoE = buildMoEWeights(numExperts, topK, dModel, dFF, expertDFF, 9)
	if _, err := DecodeForwardArch(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm); err != nil {
		t.Fatalf("DecodeForwardArch MoE warmup: %v", err)
	}

	var forwardErr error
	allocs := testing.AllocsPerRun(3, func() {
		_, forwardErr = DecodeForwardArch(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm)
	})
	if forwardErr != nil {
		t.Fatalf("DecodeForwardArch MoE: %v", forwardErr)
	}
	if allocs > 25 {
		t.Fatalf("DecodeForwardArch MoE allocations = %.0f, want <= 25", allocs)
	}
}

func TestArchDecodeStateSetupAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, maxLen = 64, 1, 1, 64, 128, 4
	specs := []model.LayerSpec{{CacheIndex: -1}}
	layers := []archLayerBufs{{dFF: dFF}}

	withAutoreleasePool(func() {
		warm := newArchDecodeState(specs, layers, nil, dModel, nHeads, nKV, headDim, dFF, 0, headDim, headDim, 10000, 10000, 0.125, 1e-5, false, maxLen)
		warm.Close()

		allocs := testing.AllocsPerRun(10, func() {
			st := newArchDecodeState(specs, layers, nil, dModel, nHeads, nKV, headDim, dFF, 0, headDim, headDim, 10000, 10000, 0.125, 1e-5, false, maxLen)
			st.Close()
		})
		if allocs > 1 {
			t.Fatalf("arch decode state setup allocations = %.0f, want <= 1", allocs)
		}
	})
}

// TestArchDecodeStateStepTokenNoResult_CommitsCache pins the prefill form of
// stepToken: it deliberately declines a host readback, but it must still write
// token zero's KV rows. Token one therefore has to equal a state that read token
// zero normally. This is a stateful invariant, not an output-only smoke test.
func TestArchDecodeStateStepTokenNoResult_CommitsCache(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, maxLen = 64, 1, 1, 64, 128, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	specs := []model.LayerSpec{{Attention: model.GlobalAttention, KVShareFrom: 0, CacheIndex: 0, HeadDim: headDim, KVHeads: nKV}}
	layers := []DecodeLayerWeights{decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 31)}
	inputs := decodeInputsFixture(2, dModel)

	var testErr error
	withAutoreleasePool(func() {
		newState := func() (*archDecodeState, error) {
			lb, moe, err := buildBF16ArchLayerBufs(layers, specs, dModel, nHeads, nKV, headDim, dFF, maxLen, 0, nil)
			if err != nil {
				return nil, err
			}
			state := newArchDecodeState(specs, lb, moe, dModel, nHeads, nKV, headDim, dFF, 0, headDim, headDim, base, base, scale, eps, false, maxLen)
			return &state, nil
		}
		prefill, err := newState()
		if err != nil {
			testErr = err
			return
		}
		defer prefill.Close()
		if err := prefill.stepTokenNoResult(inputs[0], 0); err != nil {
			testErr = err
			return
		}
		got, err := prefill.stepToken(inputs[1], 1)
		if err != nil {
			testErr = err
			return
		}

		ref, err := newState()
		if err != nil {
			testErr = err
			return
		}
		defer ref.Close()
		if _, err := ref.stepToken(inputs[0], 0); err != nil {
			testErr = err
			return
		}
		want, err := ref.stepToken(inputs[1], 1)
		if err != nil {
			testErr = err
			return
		}
		eqBytes(t, "stepTokenNoResult cache continuation", got, want)
	})
	if testErr != nil {
		t.Fatal(testErr)
	}
}

// TestArchDecodeStateStepTokenResult_NoReadback pins readResult=false: the
// wrapper returns no transient host slice, while its GPU work remains a normal
// cache-growing step for the next token.
func TestArchDecodeStateStepTokenResult_NoReadback(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, maxLen = 64, 1, 1, 64, 128, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	specs := []model.LayerSpec{{Attention: model.GlobalAttention, KVShareFrom: 0, CacheIndex: 0, HeadDim: headDim, KVHeads: nKV}}
	layers := []DecodeLayerWeights{decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 37)}
	inputs := decodeInputsFixture(2, dModel)

	var testErr error
	withAutoreleasePool(func() {
		newState := func() (*archDecodeState, error) {
			lb, moe, err := buildBF16ArchLayerBufs(layers, specs, dModel, nHeads, nKV, headDim, dFF, maxLen, 0, nil)
			if err != nil {
				return nil, err
			}
			state := newArchDecodeState(specs, lb, moe, dModel, nHeads, nKV, headDim, dFF, 0, headDim, headDim, base, base, scale, eps, false, maxLen)
			return &state, nil
		}
		noRead, err := newState()
		if err != nil {
			testErr = err
			return
		}
		defer noRead.Close()
		if got, err := noRead.stepTokenResult(inputs[0], 0, false); err != nil {
			testErr = err
			return
		} else if got != nil {
			testErr = core.NewError("stepTokenResult(readResult=false) returned a host result")
			return
		}
		got, err := noRead.stepToken(inputs[1], 1)
		if err != nil {
			testErr = err
			return
		}

		ref, err := newState()
		if err != nil {
			testErr = err
			return
		}
		defer ref.Close()
		if _, err := ref.stepToken(inputs[0], 0); err != nil {
			testErr = err
			return
		}
		want, err := ref.stepToken(inputs[1], 1)
		if err != nil {
			testErr = err
			return
		}
		eqBytes(t, "stepTokenResult no-read cache continuation", got, want)
	})
	if testErr != nil {
		t.Fatal(testErr)
	}
}

// TestArchDecodeStateStepTokenLoaded_UsesPreloadedInput pins the transient
// prefill hand-off: stepTokenLoaded must consume xA as already populated, not
// copy its argument into xA again. The deliberately unrelated argument makes
// the source of the decoded embedding observable.
func TestArchDecodeStateStepTokenLoaded_UsesPreloadedInput(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, maxLen = 64, 1, 1, 64, 128, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	specs := []model.LayerSpec{{Attention: model.GlobalAttention, KVShareFrom: 0, CacheIndex: 0, HeadDim: headDim, KVHeads: nKV}}
	layers := []DecodeLayerWeights{decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 41)}
	input := decodeInputsFixture(1, dModel)[0]
	ignoredArgument := bytes.Repeat([]byte{0xa5}, len(input))

	var testErr error
	withAutoreleasePool(func() {
		newState := func() (*archDecodeState, error) {
			lb, moe, err := buildBF16ArchLayerBufs(layers, specs, dModel, nHeads, nKV, headDim, dFF, maxLen, 0, nil)
			if err != nil {
				return nil, err
			}
			state := newArchDecodeState(specs, lb, moe, dModel, nHeads, nKV, headDim, dFF, 0, headDim, headDim, base, base, scale, eps, false, maxLen)
			return &state, nil
		}
		loaded, err := newState()
		if err != nil {
			testErr = err
			return
		}
		defer loaded.Close()
		copy(loaded.bufferBytes(loaded.xA, dModel*bf16Size), input)
		got, err := loaded.stepTokenLoaded(ignoredArgument, 0)
		if err != nil {
			testErr = err
			return
		}

		ref, err := newState()
		if err != nil {
			testErr = err
			return
		}
		defer ref.Close()
		want, err := ref.stepToken(input, 0)
		if err != nil {
			testErr = err
			return
		}
		eqBytes(t, "stepTokenLoaded preloaded input", got, want)
	})
	if testErr != nil {
		t.Fatal(testErr)
	}
}

// TestArchPLEBF16Runtime_ResidentProjection exercises both transient forms the
// runtime installs for a model projection: compute returns host-readable PLE
// rows, while computeBuffer returns their resident buffer for the preloaded
// decode path. Both forms represent the same nLayers*pliDim bf16 tensor.
func TestArchPLEBF16Runtime_ResidentProjection(t *testing.T) {
	requireNativeRuntime(t)

	const nLayers, tokens, dModel, vocabPLI, pliDim = 2, 2, 64, 8, 4
	payload := &ArchPLEBF16{
		TokenIDs:           []int32{2, 5},
		VocabPLI:           vocabPLI,
		PliDim:             pliDim,
		EmbedPerLayer:      toBF16Bytes(syntheticFloat32(vocabPLI*nLayers*pliDim, 101)),
		PerLayerModelProjW: toBF16Bytes(syntheticFloat32(nLayers*pliDim*dModel, 103)),
		PerLayerProjNormW:  toBF16Bytes(fillConst(pliDim, 1)),
	}
	runtime, dim, err := archPLEBF16Runtime("TestArchPLEBF16Runtime", payload, nLayers, tokens, dModel, 1e-5)
	if err != nil {
		t.Fatalf("archPLEBF16Runtime: %v", err)
	}
	if runtime == nil || dim != pliDim {
		t.Fatalf("archPLEBF16Runtime = (%v, %d), want runtime and dim %d", runtime, dim, pliDim)
	}
	emb := toBF16Bytes(syntheticFloat32(dModel, 107))
	host, err := runtime.compute(payload.TokenIDs[0], emb)
	if err != nil {
		t.Fatalf("archPLEBF16Runtime.compute: %v", err)
	}
	wantBytes := nLayers * pliDim * bf16Size
	if len(host) != wantBytes || runtime.buffer == nil {
		t.Fatalf("archPLEBF16Runtime.compute = %d bytes, buffer=%v; want %d resident bytes", len(host), runtime.buffer, wantBytes)
	}
	n, buf, transientHost, err := runtime.computeBuffer(payload.TokenIDs[1], emb, nil)
	if err != nil {
		t.Fatalf("archPLEBF16Runtime.computeBuffer: %v", err)
	}
	if n != wantBytes || buf == nil || transientHost != nil {
		t.Fatalf("archPLEBF16Runtime.computeBuffer = (n=%d, buf=%v, host=%d), want (%d, resident buffer, nil host)", n, buf, len(transientHost), wantBytes)
	}
}

// TestArchDecodeStatePLESlabBuffer_ReusesAndResizes pins the transient
// batched-prefill PLE upload: equal-shaped slabs retain their no-copy backing,
// while a new shape replaces it and copies the new tensor bytes.
func TestArchDecodeStatePLESlabBuffer_ReusesAndResizes(t *testing.T) {
	requireNativeRuntime(t)

	state := &archDecodeState{}
	t.Cleanup(state.Close)
	first := toBF16Bytes(syntheticFloat32(8, 113))
	buf, err := state.pleSlabBuffer(first)
	if err != nil {
		t.Fatalf("pleSlabBuffer(first): %v", err)
	}
	if state.pleSlabScratch == nil || buf == nil || buf.Contents() != unsafe.Pointer(&state.pleSlabScratch.bytes[0]) {
		t.Fatal("pleSlabBuffer(first) did not return the pinned slab backing")
	}
	if got := unsafe.Slice((*byte)(buf.Contents()), len(first)); !bytes.Equal(got, first) {
		t.Fatal("pleSlabBuffer(first) did not copy the PLE slab")
	}
	firstID := buf.GetID()
	buf, err = state.pleSlabBuffer(bytes.Repeat([]byte{0x5a}, len(first)))
	if err != nil {
		t.Fatalf("pleSlabBuffer(reuse): %v", err)
	}
	if buf.GetID() != firstID {
		t.Fatal("pleSlabBuffer rebuilt an equal-sized pinned slab")
	}
	larger := toBF16Bytes(syntheticFloat32(12, 127))
	buf, err = state.pleSlabBuffer(larger)
	if err != nil {
		t.Fatalf("pleSlabBuffer(larger): %v", err)
	}
	if len(state.pleSlabScratch.bytes) != len(larger) || buf.Contents() != unsafe.Pointer(&state.pleSlabScratch.bytes[0]) {
		t.Fatal("pleSlabBuffer(larger) did not replace the pinned backing for the new shape")
	}
	if got := unsafe.Slice((*byte)(buf.Contents()), len(larger)); !bytes.Equal(got, larger) {
		t.Fatal("pleSlabBuffer(larger) did not copy the resized PLE slab")
	}
}

func TestArchDecodeStateDevicePagedKVOwnerShareMatchesLinearState(t *testing.T) {
	requireSDPAPagedKernel(t)

	const dModel, nHeads, nKV, headDim, dFF, maxLen = 128, 2, 1, 64, 256, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	specs := []model.LayerSpec{
		{Attention: model.GlobalAttention, KVShareFrom: 0, CacheIndex: 0, HeadDim: headDim, KVHeads: nKV},
		{Attention: model.GlobalAttention, KVShareFrom: 0, CacheIndex: -1, HeadDim: headDim, KVHeads: nKV},
	}
	layers := []DecodeLayerWeights{
		decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 31),
		decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 37),
	}
	inputs := [][]byte{
		toBF16Bytes(syntheticFloat32(dModel, 401)),
		toBF16Bytes(syntheticFloat32(dModel, 409)),
		toBF16Bytes(syntheticFloat32(dModel, 419)),
	}

	var testErr error
	withAutoreleasePool(func() {
		linearLB, linearMoE, err := buildBF16ArchLayerBufs(layers, specs, dModel, nHeads, nKV, headDim, dFF, maxLen, 0, nil)
		if err != nil {
			testErr = err
			return
		}
		pagedLB, pagedMoE, err := buildBF16ArchLayerBufs(layers, specs, dModel, nHeads, nKV, headDim, dFF, maxLen, 0, nil)
		if err != nil {
			testErr = err
			return
		}
		linear := newArchDecodeState(specs, linearLB, linearMoE, dModel, nHeads, nKV, headDim, dFF, 0, headDim, headDim, base, base, scale, eps, false, maxLen)
		defer linear.Close()
		paged := newArchDecodeState(specs, pagedLB, pagedMoE, dModel, nHeads, nKV, headDim, dFF, 0, headDim, headDim, base, base, scale, eps, false, maxLen)
		defer paged.Close()
		if err := paged.initDevicePagedKV(2); err != nil {
			testErr = err
			return
		}
		for pos, input := range inputs {
			want, err := linear.stepToken(input, pos)
			if err != nil {
				testErr = err
				return
			}
			got, err := paged.stepToken(input, pos)
			if err != nil {
				testErr = err
				return
			}
			if !bytes.Equal(got, want) {
				if cos := cosineBF16(got, want); cos < 0.999 {
					testErr = core.NewError(core.Sprintf("paged arch state pos %d cosine = %.6f", pos, cos))
					return
				}
			}
		}
		if len(paged.pagedKV) != len(specs) || paged.pagedKV[0] == nil || paged.pagedKV[1] != nil {
			testErr = core.NewError("paged arch state did not initialise owner-only device pages")
			return
		}
		if got := paged.pagedKV[0].length; got != len(inputs) {
			testErr = core.NewError(core.Sprintf("paged arch state length = %d, want %d", got, len(inputs)))
			return
		}
		if got := len(paged.pagedKV[0].kPages); got != 2 {
			testErr = core.NewError(core.Sprintf("paged arch state pages = %d, want 2", got))
			return
		}
	})
	if testErr != nil {
		t.Fatal(testErr)
	}
}

func TestArchDecodeStateDevicePagedKVSerializesAndRestores(t *testing.T) {
	requireSDPAPagedKernel(t)

	const dModel, nHeads, nKV, headDim, dFF, maxLen = 64, 1, 1, 64, 128, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	specs := []model.LayerSpec{{Attention: model.GlobalAttention, KVShareFrom: 0, CacheIndex: 0, HeadDim: headDim, KVHeads: nKV}}
	layers := []DecodeLayerWeights{decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 43)}
	inputs := [][]byte{
		toBF16Bytes(syntheticFloat32(dModel, 701)),
		toBF16Bytes(syntheticFloat32(dModel, 709)),
		toBF16Bytes(syntheticFloat32(dModel, 719)),
	}

	var testErr error
	withAutoreleasePool(func() {
		lb, moe, err := buildBF16ArchLayerBufs(layers, specs, dModel, nHeads, nKV, headDim, dFF, maxLen, 0, nil)
		if err != nil {
			testErr = err
			return
		}
		state := newArchDecodeState(specs, lb, moe, dModel, nHeads, nKV, headDim, dFF, 0, headDim, headDim, base, base, scale, eps, false, maxLen)
		defer state.Close()
		if err := state.initDevicePagedKV(2); err != nil {
			testErr = err
			return
		}
		for pos, input := range inputs {
			if _, err := state.stepToken(input, pos); err != nil {
				testErr = err
				return
			}
		}
		arch := model.Arch{Hidden: dModel, Heads: nHeads, KVHeads: nKV, HeadDim: headDim, FF: dFF, Layer: specs}
		sess := &ArchSession{arch: arch, state: state, maxLen: maxLen, pos: len(inputs), cachedIDs: []int32{1, 2, 3}}
		data, err := sess.SerializeState()
		if err != nil {
			testErr = err
			return
		}
		_, _, kWant, vWant, err := sess.snapshotCacheViews(0)
		if err != nil {
			testErr = err
			return
		}

		restoredLB, restoredMoE, err := buildBF16ArchLayerBufs(layers, specs, dModel, nHeads, nKV, headDim, dFF, maxLen, 0, nil)
		if err != nil {
			testErr = err
			return
		}
		restoredState := newArchDecodeState(specs, restoredLB, restoredMoE, dModel, nHeads, nKV, headDim, dFF, 0, headDim, headDim, base, base, scale, eps, false, maxLen)
		defer restoredState.Close()
		if err := restoredState.initDevicePagedKV(2); err != nil {
			testErr = err
			return
		}
		restored := &ArchSession{arch: arch, state: restoredState, maxLen: maxLen}
		if err := restored.RestoreState(data); err != nil {
			testErr = err
			return
		}
		if restored.pos != len(inputs) || restored.state.pagedKV[0].length != len(inputs) {
			testErr = core.NewError("restored paged state did not retain position and page length")
			return
		}
		_, _, kGot, vGot, err := restored.snapshotCacheViews(0)
		if err != nil {
			testErr = err
			return
		}
		n := maxLen * nKV * headDim * bf16Size
		eqBytes(t, "restored paged K cache", unsafe.Slice(kGot, n), unsafe.Slice(kWant, n))
		eqBytes(t, "restored paged V cache", unsafe.Slice(vGot, n), unsafe.Slice(vWant, n))
	})
	if testErr != nil {
		t.Fatal(testErr)
	}
}

func TestBuildBF16ArchLayerBufsScratchReusesKVCaches(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers, maxLen = 64, 1, 1, 64, 128, 32, 1, 4
	arch := archFixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	layers := []DecodeLayerWeights{decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)}
	setup := getArchBF16LayerBufScratch(nLayers)
	defer putArchBF16LayerBufScratch(setup)

	withAutoreleasePool(func() {
		lb, _, err := buildBF16ArchLayerBufsIntoScratch(setup, layers, arch.Layer, dModel, nHeads, nKV, headDim, dFF, maxLen, arch.SlidingWindow, nil)
		if err != nil {
			t.Fatalf("first buildBF16ArchLayerBufsIntoScratch: %v", err)
		}
		firstK, firstV := uint64(lb[0].kCache.GetID()), uint64(lb[0].vCache.GetID())
		firstKPtr, firstVPtr := lb[0].kCachePtr, lb[0].vCachePtr
		if firstK == 0 || firstV == 0 || firstKPtr == nil || firstVPtr == nil {
			t.Fatal("first BF16 arch layer build did not initialise KV cache buffers and pointers")
		}

		lb, _, err = buildBF16ArchLayerBufsIntoScratch(setup, layers, arch.Layer, dModel, nHeads, nKV, headDim, dFF, maxLen, arch.SlidingWindow, nil)
		if err != nil {
			t.Fatalf("second buildBF16ArchLayerBufsIntoScratch: %v", err)
		}
		if got := uint64(lb[0].kCache.GetID()); got != firstK {
			t.Fatalf("K cache buffer was not reused: first=%d second=%d", firstK, got)
		}
		if got := uint64(lb[0].vCache.GetID()); got != firstV {
			t.Fatalf("V cache buffer was not reused: first=%d second=%d", firstV, got)
		}
		if lb[0].kCachePtr != firstKPtr || lb[0].vCachePtr != firstVPtr {
			t.Fatal("KV cache contents pointers were not reused")
		}
	})
}

// TestDecodeForwardArchMoE gates the MoE wiring into the executor: a multi-layer arch
// where one layer is MoE (spec.MoE + layer.MoE weights) decodes byte-for-byte the
// arch reference (which routes that layer through moeBlockRef instead of the dense
// MLP). A non-vacuous check confirms the MoE layer genuinely changes the output: the
// same arch with that layer forced dense differs at the final token.
func TestDecodeForwardArchMoE(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	// headDim 64: the metallib ships sdpa_vector specializations for {64,96,128,256},
	// not 32 (real gemma4 E2B uses 256) — match the proven attention dims here.
	const dModel, nHeads, nKV, headDim, dFF = 512, 8, 4, 64, 1024
	const numExperts, topK, expertDFF = 8, 2, 768
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	const T, maxLen, nL, moeIdx = 3, 8, 3, 1

	inputs := make([][]byte, T)
	for i := range inputs {
		f := make([]float32, dModel)
		for j := range f {
			f[j] = float32((j*(i+3)+5)%97-48) * 0.02
		}
		inputs[i] = toBF16Bytes(f)
	}
	layers := make([]DecodeLayerWeights, nL)
	types := make([]string, nL)
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
		types[li] = "full_attention"
	}
	specs := model.DeriveLayers(types, 0)
	specs[moeIdx].MoE = true
	layers[moeIdx].MoE = buildMoEWeights(numExperts, topK, dModel, dFF, expertDFF, 200)

	got, err := DecodeForwardArch(inputs, layers, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArch MoE: %v", err)
	}
	ref := archShareRef(t, layers, specs, inputs, dModel, nHeads, nKV, headDim, dFF, maxLen, 0, base, scale, eps)
	for tok := range T {
		eqBytes(t, core.Sprintf("MoE-layer arch vs ref tok%d", tok), got[tok], ref[tok])
	}

	// non-vacuous: forcing that one layer dense changes the output (the MoE FFN is
	// genuinely live, not a no-op that happens to match the dense path).
	denseLayers := make([]DecodeLayerWeights, nL)
	copy(denseLayers, layers)
	denseLayers[moeIdx].MoE = nil
	denseSpecs := model.DeriveLayers(types, 0) // all MoE=false
	gotDense, err := DecodeForwardArch(inputs, denseLayers, denseSpecs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArch dense: %v", err)
	}
	same := true
	for i := range got[T-1] {
		if got[T-1][i] != gotDense[T-1][i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("the MoE layer produced the same final output as forcing it dense — the MoE FFN did not engage")
	}
	t.Logf("executor MoE wiring: layer %d MoE decodes ≡ arch ref over %d tokens and differs from the all-dense arch", moeIdx, T)
}

func TestArchDecodeStateHostScratchReusesBacking(t *testing.T) {
	var s archDecodeState
	first := s.hostHiddenScratch(64)
	if len(first) != 64*bf16Size {
		t.Fatalf("first scratch length = %d, want %d", len(first), 64*bf16Size)
	}
	second := s.hostHiddenScratch(64)
	if len(second) != len(first) {
		t.Fatalf("second scratch length = %d, want %d", len(second), len(first))
	}
	if &second[0] != &first[0] {
		t.Fatal("host scratch did not reuse backing for the same hidden size")
	}
	smaller := s.hostHiddenScratch(32)
	if len(smaller) != 32*bf16Size {
		t.Fatalf("smaller scratch length = %d, want %d", len(smaller), 32*bf16Size)
	}
	if &smaller[0] != &first[0] {
		t.Fatal("host scratch did not reuse backing for a smaller hidden size")
	}
	larger := s.hostHiddenScratch(128)
	if len(larger) != 128*bf16Size {
		t.Fatalf("larger scratch length = %d, want %d", len(larger), 128*bf16Size)
	}
	if &larger[0] == &first[0] {
		t.Fatal("host scratch reused undersized backing for a larger hidden size")
	}
}

func TestArchDecodeStateHostPinnedScratchReusesBacking(t *testing.T) {
	requireNativeRuntime(t)

	var s archDecodeState
	first, firstBuf, err := s.hostHiddenPinnedScratch(64)
	if err != nil {
		t.Fatalf("hostHiddenPinnedScratch first: %v", err)
	}
	if len(first) != 64*bf16Size || firstBuf == nil {
		t.Fatalf("first pinned scratch length/buffer = %d/%v", len(first), firstBuf)
	}
	second, secondBuf, err := s.hostHiddenPinnedScratch(64)
	if err != nil {
		t.Fatalf("hostHiddenPinnedScratch second: %v", err)
	}
	if &second[0] != &first[0] || secondBuf != firstBuf {
		t.Fatal("pinned host scratch did not reuse backing for the same hidden size")
	}
	larger, largerBuf, err := s.hostHiddenPinnedScratch(128)
	if err != nil {
		t.Fatalf("hostHiddenPinnedScratch larger: %v", err)
	}
	if len(larger) != 128*bf16Size || &larger[0] == &first[0] || largerBuf == firstBuf {
		t.Fatal("pinned host scratch did not reallocate for a larger hidden size")
	}
	s.Close()
	if s.hostPinnedScratch != nil {
		t.Fatal("Close did not clear pinned host scratch")
	}
}

func TestArchDecodeStateHostPLEInputBufferUsesCallerBacking(t *testing.T) {
	requireNativeRuntime(t)

	const nLayers, pliDim = 3, 16
	pli := toBF16Bytes(syntheticFloat32(nLayers*pliDim, 17))
	s := archDecodeState{specs: make([]model.LayerSpec, nLayers), pliDim: pliDim, perLayerInput: pli}
	buf, err := s.hostPLEInputBuffer(len(pli))
	if err != nil {
		t.Fatalf("hostPLEInputBuffer: %v", err)
	}
	if got, want := uintptr(buf.Contents()), uintptr(unsafe.Pointer(&pli[0])); got != want {
		t.Fatalf("PLE input buffer pointer = %#x, want caller backing %#x", got, want)
	}
	reused, err := s.hostPLEInputBuffer(len(pli))
	if err != nil {
		t.Fatalf("hostPLEInputBuffer reused: %v", err)
	}
	if reused.GetID() != buf.GetID() {
		t.Fatal("hostPLEInputBuffer did not reuse the pinned no-copy view")
	}
	s.Close()
	if s.pleInputScratch != nil {
		t.Fatal("Close did not clear PLE input buffer")
	}
}

func TestArchDecodeStateInputEmbBufferUsesCallerBacking(t *testing.T) {
	requireNativeRuntime(t)

	const dModel = 64
	emb := toBF16Bytes(syntheticFloat32(dModel, 19))
	var s archDecodeState
	buf, ok := s.inputEmbBuffer(emb, dModel)
	if !ok {
		t.Fatal("inputEmbBuffer ok = false")
	}
	if got, want := uintptr(buf.Contents()), uintptr(unsafe.Pointer(&emb[0])); got != want {
		t.Fatalf("input embedding buffer pointer = %#x, want caller backing %#x", got, want)
	}
	reused, ok := s.inputEmbBuffer(emb, dModel)
	if !ok {
		t.Fatal("reused inputEmbBuffer ok = false")
	}
	if reused.GetID() != buf.GetID() {
		t.Fatal("inputEmbBuffer did not reuse the pinned no-copy view")
	}
	s.Close()
	if s.inputEmbScratch != nil {
		t.Fatal("Close did not clear input embedding buffer")
	}
}

func TestArchDecodeStateCachesStepContentsPointers(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF = 8, 1, 1, 8, 16
	s := newArchDecodeState([]model.LayerSpec{{CacheIndex: -1}}, []archLayerBufs{{}}, nil, dModel, nHeads, nKV, headDim, dFF, 0, headDim, headDim, 10000, 10000, 0.125, 1e-5, false, 4)
	if s.offPtr == nil || s.xAPtr == nil || s.xBPtr == nil || s.hBufPtr == nil {
		t.Fatal("arch decode state did not cache step buffer contents pointers")
	}

	*s.offPtr = 3
	if got := *(*int32)(s.offBuf.Contents()); got != 3 {
		t.Fatalf("cached offset write = %d, want 3", got)
	}

	input := toBF16Bytes([]float32{1, 2, 3, 4, 5, 6, 7, 8})
	copy(unsafe.Slice(s.xAPtr, len(input)), input)
	if got := unsafe.Slice((*byte)(s.xA.Contents()), len(input)); !bytes.Equal(got, input) {
		t.Fatalf("cached xA write = %v, want %v", got, input)
	}

	output := toBF16Bytes([]float32{8, 7, 6, 5, 4, 3, 2, 1})
	copy(unsafe.Slice(s.xBPtr, len(output)), output)
	if got := unsafe.Slice(s.bufferPtr(s.xB), len(output)); !bytes.Equal(got, output) {
		t.Fatalf("cached xB read = %v, want %v", got, output)
	}
}

func TestArchDecodeStateCachesGlobalProportionalRopePeriodsBuffer(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, maxLen = 64, 1, 1, 64, 128, 4
	specs := []model.LayerSpec{{Attention: model.GlobalAttention, KVShareFrom: 0, CacheIndex: 0, HeadDim: headDim, KVHeads: nKV}}
	layers := []archLayerBufs{{dFF: dFF}}

	states := make([]archDecodeState, 0, 2)
	withAutoreleasePool(func() {
		st := newArchDecodeState(specs, layers, nil, dModel, nHeads, nKV, headDim, dFF, 0, 32, headDim, 10000, 10000, 0.125, 1e-5, false, maxLen)
		if st.globalRopeFreqs == nil || st.globalRopeFreqs.GetID() == 0 {
			t.Fatal("first arch decode state did not build global proportional rope periods")
		}
		states = append(states, st)

		st = newArchDecodeState(specs, layers, nil, dModel, nHeads, nKV, headDim, dFF, 0, 32, headDim, 10000, 10000, 0.125, 1e-5, false, maxLen)
		if st.globalRopeFreqs == nil || st.globalRopeFreqs.GetID() == 0 {
			t.Fatal("second arch decode state did not build global proportional rope periods")
		}
		states = append(states, st)
	})
	first := uint64(states[0].globalRopeFreqs.GetID())
	second := uint64(states[1].globalRopeFreqs.GetID())
	if first != second {
		t.Fatalf("global proportional rope periods buffer was not reused: first=%d second=%d", first, second)
	}
}

func TestSharedLayerSuffixStart(t *testing.T) {
	owner := model.LayerSpec{Attention: model.SlidingAttention, CacheIndex: 0}
	sharer := model.LayerSpec{Attention: model.SlidingAttention, CacheIndex: -1, KVShareFrom: 0}
	cases := []struct {
		name  string
		specs []model.LayerSpec
		want  int
	}{
		{"owners then shared suffix (gemma4 E-family shape)", []model.LayerSpec{owner, owner, sharer, sharer}, 2},
		{"single trailing sharer", []model.LayerSpec{owner, sharer}, 1},
		{"all owners", []model.LayerSpec{owner, owner}, -1},
		{"sharer below an owner is not a clean suffix", []model.LayerSpec{owner, sharer, owner}, -1},
		{"all sharers (0 stays inert: activation requires > 0)", []model.LayerSpec{sharer, sharer}, 0},
		{"empty", nil, -1},
	}
	for _, tc := range cases {
		if got := sharedLayerSuffixStart(tc.specs); got != tc.want {
			t.Errorf("%s: sharedLayerSuffixStart = %d, want %d", tc.name, got, tc.want)
		}
	}
}
