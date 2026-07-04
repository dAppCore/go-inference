// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"math"
	"os"
	"testing"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"github.com/tmc/apple/metal"
)

// lastTokenDiffers reports whether two forwards' final-token outputs differ.
func lastTokenDiffers(a, b [][]byte) bool {
	la, lb := a[len(a)-1], b[len(b)-1]
	if len(la) != len(lb) {
		return true
	}
	for i := range la {
		if la[i] != lb[i] {
			return true
		}
	}
	return false
}

// TestDecodeForwardArchQuant gates the 4-bit arch-driven forward. (a) an all-owner,
// all-global, dense quant arch is byte-for-byte the proven DecodeForwardQuant (the arch
// executor + qmv projector ≡ the standalone quant forward when the arch routes nothing)
// — the correctness anchor. (b) a KV-share quant arch differs from the all-owner one
// (sharing genuinely reroutes layer 1's attention on the quant path). (c) a sliding
// quant arch (W=3) differs from full attention over 6 tokens (the window clips on the
// quant path).
func TestDecodeForwardArchQuant(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, gs, bits = 512, 8, 4, 64, 1024, 64, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)

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

	// (a) all-owner all-global ≡ DecodeForwardQuant byte-for-byte.
	const nL, T, maxLen = 3, 4, 8
	ql := make([]QuantizedLayerWeights, nL)
	types := make([]string, nL)
	for l := range ql {
		ql[l] = buildQuantLayer(t, dModel, nHeads, nKV, headDim, dFF, gs, bits, (l+1)*100)
		types[l] = "full_attention"
	}
	inputs := mkInputs(T)
	specsOwn := model.DeriveLayers(types, 0)
	gotArch, err := DecodeForwardArchQuant(inputs, ql, specsOwn, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArchQuant all-owner: %v", err)
	}
	ref, err := DecodeForwardQuant(inputs, ql, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	if err != nil {
		t.Fatalf("DecodeForwardQuant: %v", err)
	}
	for tok := 0; tok < T; tok++ {
		eqBytes(t, core.Sprintf("quant all-owner vs DecodeForwardQuant tok%d", tok), gotArch[tok], ref[tok])
	}

	// (b) KV-share reroutes attention: 2 layers, layer 1 shares layer 0's cache vs both
	// own. Different layer weights → the shared and owned results must differ.
	ql2 := []QuantizedLayerWeights{
		buildQuantLayer(t, dModel, nHeads, nKV, headDim, dFF, gs, bits, 100),
		buildQuantLayer(t, dModel, nHeads, nKV, headDim, dFF, gs, bits, 200),
	}
	in2 := mkInputs(T)
	specsShare := model.DeriveLayers([]string{"full_attention", "full_attention"}, 1)
	specsBothOwn := model.DeriveLayers([]string{"full_attention", "full_attention"}, 0)
	gotShare, err := DecodeForwardArchQuant(in2, ql2, specsShare, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArchQuant share: %v", err)
	}
	gotBothOwn, err := DecodeForwardArchQuant(in2, ql2, specsBothOwn, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArchQuant both-own: %v", err)
	}
	if !lastTokenDiffers(gotShare, gotBothOwn) {
		t.Fatal("quant KV-share produced the same output as all-owner — sharing did not reroute attention")
	}

	// (c) sliding clips on the quant path: all-sliding W=3 over 6 tokens vs full (W=0).
	const W, T2, maxLen2 = 3, 6, 8
	slideTypes := make([]string, nL)
	for i := range slideTypes {
		slideTypes[i] = "sliding_attention"
	}
	specsSlide := model.DeriveLayers(slideTypes, 0)
	in3 := mkInputs(T2)
	gotSlide, err := DecodeForwardArchQuant(in3, ql, specsSlide, dModel, nHeads, nKV, headDim, maxLen2, dFF, W, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArchQuant sliding: %v", err)
	}
	gotFull, err := DecodeForwardArchQuant(in3, ql, specsSlide, dModel, nHeads, nKV, headDim, maxLen2, dFF, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArchQuant sliding-full: %v", err)
	}
	if !lastTokenDiffers(gotSlide, gotFull) {
		t.Fatal("quant sliding (W=3) matched full attention over 6 tokens — the window did not clip")
	}

	t.Logf("quant arch: all-owner ≡ DecodeForwardQuant byte-for-byte; KV-share reroutes; sliding (W=%d, %d toks) clips — 4-bit on the arch path", W, T2)
}

func TestDecodeForwardArchQuantIntoReusesOutputBacking(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers, maxLen = 64, 1, 1, 64, 128, 32, 1, 4
	const groupSize, bits = 64, 4
	arch := archFixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	inputs := decodeInputsFixture(2, dModel)
	layers := []QuantizedLayerWeights{quantizedLayerFixture(t, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, 3)}
	want, err := DecodeForwardArchQuant(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm)
	if err != nil {
		t.Fatalf("DecodeForwardArchQuant reference: %v", err)
	}
	out := [][]byte{
		bytes.Repeat([]byte{0xa5}, dModel*bf16Size),
		bytes.Repeat([]byte{0x5a}, dModel*bf16Size),
	}
	ptrs := []unsafe.Pointer{unsafe.Pointer(&out[0][0]), unsafe.Pointer(&out[1][0])}

	got, err := DecodeForwardArchQuantInto(out, inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm)
	if err != nil {
		t.Fatalf("DecodeForwardArchQuantInto: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("DecodeForwardArchQuantInto returned %d outputs, want %d", len(got), len(want))
	}
	for tok := range want {
		if len(got[tok]) != dModel*bf16Size || unsafe.Pointer(&got[tok][0]) != ptrs[tok] {
			t.Fatalf("DecodeForwardArchQuantInto token %d did not reuse caller-owned output backing", tok)
		}
		eqBytes(t, "DecodeForwardArchQuantInto token", got[tok], want[tok])
	}
}

func TestDecodeForwardArchQuantMoEAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers, maxLen = 64, 1, 1, 64, 128, 32, 1, 4
	const expertDFF, numExperts, topK = 96, 4, 2
	const groupSize, bits = 32, 4
	arch := archFixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	arch.Layer[0].MoE = true
	inputs := decodeInputsFixture(2, dModel)
	layer := quantizedLayerFixture(t, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, 3)
	moeWeights := quantMoELayerWeightsGuard(t, numExperts, topK, dModel, dFF, expertDFF, groupSize, bits)
	layer.MLPNormW, layer.Gate, layer.Up, layer.Down = nil, QuantWeight{}, QuantWeight{}, QuantWeight{}
	layer.MoE = &moeWeights
	layers := []QuantizedLayerWeights{layer}
	if _, err := DecodeForwardArchQuant(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm); err != nil {
		t.Fatalf("DecodeForwardArchQuant MoE warmup: %v", err)
	}

	var forwardErr error
	allocs := testing.AllocsPerRun(3, func() {
		_, forwardErr = DecodeForwardArchQuant(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm)
	})
	if forwardErr != nil {
		t.Fatalf("DecodeForwardArchQuant MoE: %v", forwardErr)
	}
	if allocs > 30 {
		t.Fatalf("DecodeForwardArchQuant MoE allocations = %.0f, want <= 30", allocs)
	}
}

func TestDecodeForwardArchQuantAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers, maxLen = 64, 1, 1, 64, 128, 32, 1, 4
	const groupSize, bits = 64, 4
	arch := archFixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	inputs := decodeInputsFixture(2, dModel)
	layers := []QuantizedLayerWeights{quantizedLayerFixture(t, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, 3)}
	if _, err := DecodeForwardArchQuant(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm); err != nil {
		t.Fatalf("DecodeForwardArchQuant warmup: %v", err)
	}

	var forwardErr error
	allocs := testing.AllocsPerRun(5, func() {
		_, forwardErr = DecodeForwardArchQuant(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm)
	})
	if forwardErr != nil {
		t.Fatalf("DecodeForwardArchQuant: %v", forwardErr)
	}
	if allocs > 25 {
		t.Fatalf("DecodeForwardArchQuant allocations = %.0f, want <= 25", allocs)
	}
}

func TestBuildQuantArchLayerBufsScratchReusesKVCaches(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers, maxLen = 64, 1, 1, 64, 128, 32, 1, 4
	const groupSize, bits = 64, 4
	arch := archFixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	layers := []QuantizedLayerWeights{quantizedLayerFixture(t, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, 3)}
	setup := getArchQuantLayerBufScratch(nLayers)
	defer putArchQuantLayerBufScratch(setup)

	withAutoreleasePool(func() {
		lb, _, err := buildQuantArchLayerBufsIntoScratch(setup, layers, arch.Layer, dModel, nHeads, nKV, headDim, dFF, maxLen, arch.SlidingWindow, nil)
		if err != nil {
			t.Fatalf("first buildQuantArchLayerBufsIntoScratch: %v", err)
		}
		firstK, firstV := uint64(lb[0].kCache.GetID()), uint64(lb[0].vCache.GetID())
		firstKPtr, firstVPtr := lb[0].kCachePtr, lb[0].vCachePtr
		if firstK == 0 || firstV == 0 || firstKPtr == nil || firstVPtr == nil {
			t.Fatal("first quant arch layer build did not initialise KV cache buffers and pointers")
		}

		lb, _, err = buildQuantArchLayerBufsIntoScratch(setup, layers, arch.Layer, dModel, nHeads, nKV, headDim, dFF, maxLen, arch.SlidingWindow, nil)
		if err != nil {
			t.Fatalf("second buildQuantArchLayerBufsIntoScratch: %v", err)
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

func TestBuildQuantArchLayerBufsResidentMoEViews(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers, maxLen = 64, 1, 1, 64, 128, 32, 1, 4
	const expertDFF, numExperts, topK = 96, 4, 2
	const groupSize, bits = 32, 4
	arch := archFixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	arch.Layer[0].MoE = true
	layer := quantizedLayerFixture(t, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, 3)
	moeWeights := quantMoELayerWeightsGuard(t, numExperts, topK, dModel, dFF, expertDFF, groupSize, bits)
	layer.MLPNormW, layer.Gate, layer.Up, layer.Down = nil, QuantWeight{}, QuantWeight{}, QuantWeight{}
	layer.MoE = &moeWeights
	setup := getArchQuantLayerBufScratch(nLayers)
	defer putArchQuantLayerBufScratch(setup)

	var moe []*MoEQuantLayerWeights
	var buildErr error
	withAutoreleasePool(func() {
		_, moe, buildErr = buildQuantArchLayerBufsIntoScratch(setup, []QuantizedLayerWeights{layer}, arch.Layer, dModel, nHeads, nKV, headDim, dFF, maxLen, arch.SlidingWindow, nil)
	})
	if buildErr != nil {
		t.Fatalf("buildQuantArchLayerBufsIntoScratch: %v", buildErr)
	}
	if len(moe) != 1 || moe[0] == nil {
		t.Fatalf("prepared MoE weights missing: len=%d first=%v", len(moe), len(moe) > 0 && moe[0] != nil)
	}
	w := moe[0]
	weights := []struct {
		name string
		q    QuantWeight
	}{
		{"local gate", w.LocalGate},
		{"local up", w.LocalUp},
		{"local down", w.LocalDown},
		{"router", w.Router},
		{"expert gate", w.ExpGate},
		{"expert up", w.ExpUp},
		{"expert down", w.ExpDown},
	}
	norms := []struct {
		name string
		buf  []byte
		view bufView
	}{
		{"pre ff norm", w.PreFFNormW, w.preFFNormView},
		{"pre ff norm 2", w.PreFFNorm2W, w.preFFNorm2View},
		{"post ff norm 1", w.PostFFNorm1W, w.postFFNorm1View},
		{"post ff norm 2", w.PostFFNorm2W, w.postFFNorm2View},
		{"post ff norm", w.PostFFNormW, w.postFFNormView},
		{"router norm", w.RouterNormWScaled, w.routerNormView},
		{"per expert scale", w.PerExpertScale, w.perExpertScaleView},
	}
	key := func(b []byte) uintptr { return uintptr(unsafe.Pointer(&b[0])) }
	missingViews := make([]string, 0)
	missingResident := make([]string, 0)
	residentBufMu.Lock()
	for _, weight := range weights {
		if weight.q.packedView.buf == nil || weight.q.scalesView.buf == nil || weight.q.biasesView.buf == nil {
			missingViews = append(missingViews, weight.name)
		}
		for _, part := range []struct {
			name string
			buf  []byte
		}{
			{weight.name + ".packed", weight.q.Packed},
			{weight.name + ".scales", weight.q.Scales},
			{weight.name + ".biases", weight.q.Biases},
		} {
			if _, ok := residentBufs[key(part.buf)]; !ok {
				missingResident = append(missingResident, part.name)
			}
		}
	}
	for _, norm := range norms {
		if norm.view.buf == nil {
			missingViews = append(missingViews, norm.name)
		}
		if _, ok := residentBufs[key(norm.buf)]; !ok {
			missingResident = append(missingResident, norm.name)
		}
	}
	residentCount := len(residentBufs)
	residentBufMu.Unlock()
	if len(missingViews) != 0 || len(missingResident) != 0 {
		t.Fatalf("prepared quant MoE resident views missing views=%v resident=%v residentCount=%d", missingViews, missingResident, residentCount)
	}
}

func TestDecodeForwardArchQuantPLEAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers, maxLen = 64, 1, 1, 64, 128, 32, 1, 4
	const pliDim, groupSize, bits = 32, 32, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	arch := archFixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	inputs := decodeInputsFixture(2, dModel)
	qlayers := []QuantizedLayerWeights{quantizedLayerFixture(t, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, 3)}
	qlayers[0].PerLayerGate = quantWeightFixture(t, pliDim, dModel, groupSize, bits, 17)
	qlayers[0].PerLayerProjection = quantWeightFixture(t, dModel, pliDim, groupSize, bits, 23)
	qlayers[0].PostPerLayerInputNormW = toBF16Bytes(syntheticFloat32(dModel, 5))
	pleEmbed := quantWeightFixture(t, vocab, nLayers*pliDim, groupSize, bits, 31)
	ple := ArchPLEQuant{
		TokenIDs:            []int32{1, 2},
		EmbedPerLayer:       pleEmbed.Packed,
		EmbedPerLayerScales: pleEmbed.Scales,
		EmbedPerLayerBiases: pleEmbed.Biases,
		PerLayerModelProjW:  toBF16Bytes(syntheticFloat32(nLayers*pliDim*dModel, 37)),
		PerLayerProjNormW:   toBF16Bytes(syntheticFloat32(pliDim, 41)),
		VocabPLI:            vocab,
		PliDim:              pliDim,
		GroupSize:           groupSize,
		Bits:                bits,
		ProjGroupSize:       groupSize,
		ProjBits:            bits,
	}
	if _, err := DecodeForwardArchQuant(inputs, qlayers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, base, scale, eps, false, ple); err != nil {
		t.Fatalf("DecodeForwardArchQuant PLE warmup: %v", err)
	}

	var forwardErr error
	allocs := testing.AllocsPerRun(5, func() {
		_, forwardErr = DecodeForwardArchQuant(inputs, qlayers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, base, scale, eps, false, ple)
	})
	if forwardErr != nil {
		t.Fatalf("DecodeForwardArchQuant PLE: %v", forwardErr)
	}
	if allocs > 5760 {
		t.Fatalf("DecodeForwardArchQuant PLE allocations = %.0f, want <= 5760", allocs)
	}
}

func TestDecodeForwardArchQuantPLEQuantProjectionAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers, maxLen = 64, 1, 1, 64, 128, 32, 1, 4
	const pliDim, groupSize, bits = 32, 32, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	arch := archFixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	inputs := decodeInputsFixture(2, dModel)
	qlayers := []QuantizedLayerWeights{quantizedLayerFixture(t, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, 3)}
	qlayers[0].PerLayerGate = quantWeightFixture(t, pliDim, dModel, groupSize, bits, 17)
	qlayers[0].PerLayerProjection = quantWeightFixture(t, dModel, pliDim, groupSize, bits, 23)
	qlayers[0].PostPerLayerInputNormW = toBF16Bytes(syntheticFloat32(dModel, 5))
	pleEmbed := quantWeightFixture(t, vocab, nLayers*pliDim, groupSize, bits, 31)
	pleProj := quantWeightFixture(t, nLayers*pliDim, dModel, groupSize, bits, 37)
	ple := ArchPLEQuant{
		TokenIDs:                []int32{1, 2},
		EmbedPerLayer:           pleEmbed.Packed,
		EmbedPerLayerScales:     pleEmbed.Scales,
		EmbedPerLayerBiases:     pleEmbed.Biases,
		PerLayerModelProjW:      pleProj.Packed,
		PerLayerModelProjScales: pleProj.Scales,
		PerLayerModelProjBiases: pleProj.Biases,
		PerLayerProjNormW:       toBF16Bytes(syntheticFloat32(pliDim, 41)),
		VocabPLI:                vocab,
		PliDim:                  pliDim,
		GroupSize:               groupSize,
		Bits:                    bits,
		ProjGroupSize:           groupSize,
		ProjBits:                bits,
	}
	if _, err := DecodeForwardArchQuant(inputs, qlayers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, base, scale, eps, false, ple); err != nil {
		t.Fatalf("DecodeForwardArchQuant PLE quant projection warmup: %v", err)
	}

	var forwardErr error
	allocs := testing.AllocsPerRun(5, func() {
		_, forwardErr = DecodeForwardArchQuant(inputs, qlayers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, base, scale, eps, false, ple)
	})
	if forwardErr != nil {
		t.Fatalf("DecodeForwardArchQuant PLE quant projection: %v", forwardErr)
	}
	if allocs > 5630 {
		t.Fatalf("DecodeForwardArchQuant PLE quant projection allocations = %.0f, want <= 5630", allocs)
	}
}

func TestArchPLEQuantRuntimeResidentBufferAvoidsHostReadback(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, vocab, nLayers = 64, 32, 1
	const pliDim, groupSize, bits = 32, 32, 4
	const eps = float32(1e-5)
	pleEmbed := quantWeightFixture(t, vocab, nLayers*pliDim, groupSize, bits, 31)
	ple := &ArchPLEQuant{
		TokenIDs:            []int32{1},
		EmbedPerLayer:       pleEmbed.Packed,
		EmbedPerLayerScales: pleEmbed.Scales,
		EmbedPerLayerBiases: pleEmbed.Biases,
		PerLayerModelProjW:  toBF16Bytes(syntheticFloat32(nLayers*pliDim*dModel, 37)),
		PerLayerProjNormW:   toBF16Bytes(syntheticFloat32(pliDim, 41)),
		VocabPLI:            vocab,
		PliDim:              pliDim,
		GroupSize:           groupSize,
		Bits:                bits,
		ProjGroupSize:       groupSize,
		ProjBits:            bits,
	}
	runtime, gotDim, err := archPLEQuantRuntime("test", ple, nLayers, len(ple.TokenIDs), dModel, eps)
	if err != nil {
		t.Fatalf("archPLEQuantRuntime: %v", err)
	}
	if gotDim != pliDim {
		t.Fatalf("archPLEQuantRuntime dim = %d, want %d", gotDim, pliDim)
	}
	defer runtime.Close()

	hidden := toBF16Bytes(syntheticFloat32(dModel, 3))
	scratch, err := runtime.ensureScratch(nLayers*pliDim, dModel, float32(1/math.Sqrt(float64(dModel))))
	if err != nil {
		t.Fatalf("ensureScratch: %v", err)
	}
	for i := range scratch.hidden.bytes {
		scratch.hidden.bytes[i] = 0xa5
	}
	wantHidden := append([]byte(nil), scratch.hidden.bytes...)

	want, err := PerLayerInputs(
		ple.EmbedPerLayer, ple.EmbedPerLayerScales, ple.EmbedPerLayerBiases,
		ple.PerLayerModelProjW, nil, nil, ple.PerLayerProjNormW,
		ple.TokenIDs[0], hidden, ple.VocabPLI, nLayers, ple.PliDim, dModel,
		ple.GroupSize, ple.Bits, ple.ProjGroupSize, ple.ProjBits, eps, bufView{},
	)
	if err != nil {
		t.Fatalf("PerLayerInputs reference: %v", err)
	}

	var n int
	var buf metal.MTLBuffer
	var host []byte
	err = withPinnedNoCopyBytes(hidden, func(hiddenBuf metal.MTLBuffer) error {
		var err error
		n, buf, host, err = runtime.computeBuffer(ple.TokenIDs[0], hidden, hiddenBuf)
		return err
	})
	if err != nil {
		t.Fatalf("computeBuffer: %v", err)
	}
	if n != nLayers*pliDim*bf16Size {
		t.Fatalf("computeBuffer bytes = %d, want %d", n, nLayers*pliDim*bf16Size)
	}
	if buf == nil || buf.GetID() == 0 {
		t.Fatal("computeBuffer did not return resident PLE Metal buffer")
	}
	if host != nil {
		t.Fatalf("computeBuffer returned host backing len=%d, want nil resident-buffer path", len(host))
	}
	if runtime.scratch == nil {
		t.Fatal("computeBuffer did not retain reusable PLE scratch")
	}
	if runtime.scratch.outHost != nil {
		t.Fatalf("computeBuffer read back PLE output to host len=%d, want resident buffer only", len(runtime.scratch.outHost))
	}
	if string(runtime.scratch.hidden.bytes) != string(wantHidden) {
		t.Fatal("computeBuffer copied hidden input through host scratch; want pinned resident hidden buffer")
	}
	got := append([]byte(nil), unsafe.Slice((*byte)(buf.Contents()), n)...)
	eqBytes(t, "arch PLE resident hidden-buffer compute", got, want)
}

func TestArchPLEQuantRuntimeQuantProjectionReturnsResidentBuffer(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, vocab, nLayers = 64, 32, 1
	const pliDim, groupSize, bits = 32, 32, 4
	const eps = float32(1e-5)
	pleEmbed := quantWeightFixture(t, vocab, nLayers*pliDim, groupSize, bits, 31)
	pleProj := quantWeightFixture(t, nLayers*pliDim, dModel, groupSize, bits, 37)
	ple := &ArchPLEQuant{
		TokenIDs:                []int32{1},
		EmbedPerLayer:           pleEmbed.Packed,
		EmbedPerLayerScales:     pleEmbed.Scales,
		EmbedPerLayerBiases:     pleEmbed.Biases,
		PerLayerModelProjW:      pleProj.Packed,
		PerLayerModelProjScales: pleProj.Scales,
		PerLayerModelProjBiases: pleProj.Biases,
		PerLayerProjNormW:       toBF16Bytes(syntheticFloat32(pliDim, 41)),
		VocabPLI:                vocab,
		PliDim:                  pliDim,
		GroupSize:               groupSize,
		Bits:                    bits,
		ProjGroupSize:           groupSize,
		ProjBits:                bits,
	}
	runtime, gotDim, err := archPLEQuantRuntime("test", ple, nLayers, len(ple.TokenIDs), dModel, eps)
	if err != nil {
		t.Fatalf("archPLEQuantRuntime: %v", err)
	}
	if gotDim != pliDim {
		t.Fatalf("archPLEQuantRuntime dim = %d, want %d", gotDim, pliDim)
	}
	defer runtime.Close()

	hidden := toBF16Bytes(syntheticFloat32(dModel, 3))
	want, err := PerLayerInputs(
		ple.EmbedPerLayer, ple.EmbedPerLayerScales, ple.EmbedPerLayerBiases,
		ple.PerLayerModelProjW, ple.PerLayerModelProjScales, ple.PerLayerModelProjBiases, ple.PerLayerProjNormW,
		ple.TokenIDs[0], hidden, ple.VocabPLI, nLayers, ple.PliDim, dModel,
		ple.GroupSize, ple.Bits, ple.ProjGroupSize, ple.ProjBits, eps, bufView{},
	)
	if err != nil {
		t.Fatalf("PerLayerInputs reference: %v", err)
	}

	var n int
	var buf metal.MTLBuffer
	var host []byte
	err = withPinnedNoCopyBytes(hidden, func(hiddenBuf metal.MTLBuffer) error {
		var err error
		n, buf, host, err = runtime.computeBuffer(ple.TokenIDs[0], hidden, hiddenBuf)
		return err
	})
	if err != nil {
		t.Fatalf("computeBuffer: %v", err)
	}
	if n != nLayers*pliDim*bf16Size {
		t.Fatalf("computeBuffer bytes = %d, want %d", n, nLayers*pliDim*bf16Size)
	}
	if buf == nil || buf.GetID() == 0 {
		t.Fatal("computeBuffer did not return resident quant PLE Metal buffer")
	}
	if host != nil {
		t.Fatalf("computeBuffer returned host backing len=%d, want nil resident-buffer path", len(host))
	}
	got := append([]byte(nil), unsafe.Slice((*byte)(buf.Contents()), n)...)
	eqBytes(t, "arch PLE quant resident compute", got, want)
}

func TestDecodeForwardArchQuantMoELayer(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim = 64, 1, 1, 64
	const dFF, expertDFF, numExperts, topK = 128, 96, 4, 2
	const gs, bits, maxLen, T = 32, 4, 4, 2
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)

	inputs := decodeInputsFixture(T, dModel)
	denseLayer := quantizedLayerFixture(t, dModel, nHeads, nKV, headDim, dFF, gs, bits, 3)
	moeWeights := quantMoELayerWeightsGuard(t, numExperts, topK, dModel, dFF, expertDFF, gs, bits)
	moeLayer := denseLayer
	moeLayer.MLPNormW, moeLayer.Gate, moeLayer.Up, moeLayer.Down = nil, QuantWeight{}, QuantWeight{}, QuantWeight{}
	moeLayer.MoE = &moeWeights

	denseSpecs := model.DeriveLayers([]string{"full_attention"}, 0)
	moeSpecs := model.DeriveLayers([]string{"full_attention"}, 0)
	moeSpecs[0].MoE = true

	gotMoE, err := DecodeForwardArchQuant(inputs, []QuantizedLayerWeights{moeLayer}, moeSpecs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArchQuant MoE: %v", err)
	}
	gotDense, err := DecodeForwardArchQuant(inputs, []QuantizedLayerWeights{denseLayer}, denseSpecs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArchQuant dense: %v", err)
	}
	if len(gotMoE) != T {
		t.Fatalf("MoE outputs = %d tokens, want %d", len(gotMoE), T)
	}
	for i := range gotMoE {
		if len(gotMoE[i]) != dModel*bf16Size {
			t.Fatalf("MoE token %d has %d bytes, want %d", i, len(gotMoE[i]), dModel*bf16Size)
		}
	}
	if !lastTokenDiffers(gotMoE, gotDense) {
		t.Fatal("quant MoE arch matched dense MLP output; MoE block was not used")
	}

	t.Logf("quant MoE arch: DecodeForwardArchQuant runs the loader-shaped MoE layer through MoEBlockQuant")
}

func TestArchDecodeStateQuantMoEUsesSharedAttentionBuffer(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim = 64, 1, 1, 64
	const dFF, expertDFF, numExperts, topK = 128, 96, 4, 2
	const gs, bits, maxLen = 32, 4, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)

	arch := archFixture(t, dModel, nHeads, nKV, headDim, dFF, 32, 1)
	arch.Layer[0].MoE = true
	layer := quantizedLayerFixture(t, dModel, nHeads, nKV, headDim, dFF, gs, bits, 3)
	moeWeights := quantMoELayerWeightsGuard(t, numExperts, topK, dModel, dFF, expertDFF, gs, bits)
	layer.MLPNormW, layer.Gate, layer.Up, layer.Down = nil, QuantWeight{}, QuantWeight{}, QuantWeight{}
	layer.MoE = &moeWeights
	input := decodeInputsFixture(1, dModel)[0]

	var stepErr error
	withAutoreleasePool(func() {
		setup := getArchQuantLayerBufScratch(1)
		defer putArchQuantLayerBufScratch(setup)
		lb, moeQuant, err := buildQuantArchLayerBufsIntoScratch(setup, []QuantizedLayerWeights{layer}, arch.Layer, dModel, nHeads, nKV, headDim, dFF, maxLen, arch.SlidingWindow, nil)
		if err != nil {
			stepErr = err
			return
		}
		state := newArchDecodeState(arch.Layer, lb, make([]*MoELayerWeights, 1), dModel, nHeads, nKV, headDim, dFF, arch.SlidingWindow, headDim, headDim, base, base, scale, eps, false, maxLen)
		defer state.Close()
		state.moeQuant = moeQuant
		if _, err := state.stepToken(input, 0); err != nil {
			stepErr = err
			return
		}
		if state.hostPinnedScratch != nil {
			t.Fatal("quant MoE arch step allocated host pinned scratch instead of consuming the shared attention buffer")
		}
		if state.coreScratch != nil && state.coreScratch.hostPinned != nil {
			t.Fatal("quant MoE arch step allocated core host pinned scratch instead of consuming the shared attention buffer")
		}
	})
	if stepErr != nil {
		t.Fatalf("quant MoE arch step: %v", stepErr)
	}
}

func TestDecodeForwardArchQuantKeepsFixedWeightsResident(t *testing.T) {
	requireNativeRuntime(t)

	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const dModel, nHeads, nKV, headDim, dFF, maxLen = 64, 1, 1, 64, 128, 4
	const groupSize, bits = 64, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	inputs := decodeInputsFixture(2, dModel)
	layer := quantizedLayerFixture(t, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, 3)
	layers := []QuantizedLayerWeights{layer}
	specs := model.DeriveLayers([]string{"full_attention"}, 0)

	if _, err := DecodeForwardArchQuant(inputs, layers, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false); err != nil {
		t.Fatalf("DecodeForwardArchQuant: %v", err)
	}

	key := func(b []byte) uintptr { return uintptr(unsafe.Pointer(&b[0])) }
	weights := []struct {
		name string
		buf  []byte
	}{
		{"attnNorm", layer.AttnNormW},
		{"mlpNorm", layer.MLPNormW},
		{"q.packed", layer.Q.Packed}, {"q.scales", layer.Q.Scales}, {"q.biases", layer.Q.Biases},
		{"k.packed", layer.K.Packed}, {"k.scales", layer.K.Scales}, {"k.biases", layer.K.Biases},
		{"v.packed", layer.V.Packed}, {"v.scales", layer.V.Scales}, {"v.biases", layer.V.Biases},
		{"o.packed", layer.O.Packed}, {"o.scales", layer.O.Scales}, {"o.biases", layer.O.Biases},
		{"gate.packed", layer.Gate.Packed}, {"gate.scales", layer.Gate.Scales}, {"gate.biases", layer.Gate.Biases},
		{"up.packed", layer.Up.Packed}, {"up.scales", layer.Up.Scales}, {"up.biases", layer.Up.Biases},
		{"down.packed", layer.Down.Packed}, {"down.scales", layer.Down.Scales}, {"down.biases", layer.Down.Biases},
	}

	residentBufMu.Lock()
	got := len(residentBufs)
	missing := make([]string, 0)
	for _, weight := range weights {
		if _, ok := residentBufs[key(weight.buf)]; !ok {
			missing = append(missing, weight.name)
		}
	}
	residentBufMu.Unlock()

	if len(missing) != 0 {
		t.Fatalf("DecodeForwardArchQuant did not keep fixed weights resident (missing=%v resident=%d want>=%d)", missing, got, len(weights))
	}
}

func TestDecodeForwardArchQuantHonoursPerWeightGeometry(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, maxLen = 64, 1, 1, 64, 128, 4
	const groupSize, bits = 64, 4
	const mlpGroupSize, mlpBits = 32, 8
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	inputs := decodeInputsFixture(2, dModel)
	layer := quantizedLayerFixture(t, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, 3)
	layer.Gate = quantWeightFixture(t, dFF, dModel, mlpGroupSize, mlpBits, 20)
	layer.Up = quantWeightFixture(t, dFF, dModel, mlpGroupSize, mlpBits, 22)
	layer.Down = quantWeightFixture(t, dModel, dFF, mlpGroupSize, mlpBits, 26)
	specs := model.DeriveLayers([]string{"full_attention"}, 0)

	got, err := DecodeForwardArchQuant(inputs, []QuantizedLayerWeights{layer}, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArchQuant with per-weight MLP geometry: %v", err)
	}
	ref, err := DecodeForwardQuant(inputs, []QuantizedLayerWeights{layer}, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	if err != nil {
		t.Fatalf("DecodeForwardQuant with per-weight MLP geometry: %v", err)
	}
	for tok := range got {
		eqBytes(t, core.Sprintf("mixed quant arch vs DecodeForwardQuant tok%d", tok), got[tok], ref[tok])
	}
}
