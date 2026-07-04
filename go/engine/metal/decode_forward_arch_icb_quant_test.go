// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"crypto/sha256"
	"os"
	"testing"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

func TestDecodeForwardArchICBQuantAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, maxLen = 64, 1, 1, 64, 128, 4
	const groupSize, bits = 64, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	inputs := decodeInputsFixture(2, dModel)
	layer := quantizedLayerFixture(t, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, 3)
	layers := []QuantizedLayerWeights{layer}
	specs := model.DeriveLayers([]string{"full_attention"}, 0)
	if _, err := DecodeForwardArchICBQuant(inputs, layers, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false); err != nil {
		t.Fatalf("DecodeForwardArchICBQuant warmup: %v", err)
	}

	var forwardErr error
	allocs := testing.AllocsPerRun(5, func() {
		_, forwardErr = DecodeForwardArchICBQuant(inputs, layers, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	})
	if forwardErr != nil {
		t.Fatalf("DecodeForwardArchICBQuant: %v", forwardErr)
	}
	if allocs > 195 {
		t.Fatalf("DecodeForwardArchICBQuant allocations = %.0f, want <= 195", allocs)
	}
}

func TestDecodeForwardArchICBQuantIntoAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers, maxLen = 64, 1, 1, 64, 128, 32, 1, 4
	const groupSize, bits = 64, 4
	arch := archFixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	inputs := decodeInputsFixture(2, dModel)
	layers := []QuantizedLayerWeights{quantizedLayerFixture(t, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, 3)}
	outputs := make([][]byte, len(inputs))
	for i := range outputs {
		outputs[i] = make([]byte, dModel*bf16Size)
	}
	if _, err := DecodeForwardArchICBQuantInto(outputs, inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm); err != nil {
		t.Fatalf("DecodeForwardArchICBQuantInto warmup: %v", err)
	}

	var forwardErr error
	allocs := testing.AllocsPerRun(5, func() {
		_, forwardErr = DecodeForwardArchICBQuantInto(outputs, inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm)
	})
	if forwardErr != nil {
		t.Fatalf("DecodeForwardArchICBQuantInto: %v", forwardErr)
	}
	if allocs > 195 {
		t.Fatalf("DecodeForwardArchICBQuantInto allocations = %.0f, want <= 195", allocs)
	}
}

func TestDecodeForwardArchICBQuantIntoReusesOutputBacking(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers, maxLen = 64, 1, 1, 64, 128, 32, 1, 4
	const groupSize, bits = 64, 4
	arch := archFixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	inputs := decodeInputsFixture(2, dModel)
	layers := []QuantizedLayerWeights{quantizedLayerFixture(t, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, 3)}
	want, err := DecodeForwardArchICBQuant(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm)
	if err != nil {
		t.Fatalf("DecodeForwardArchICBQuant reference: %v", err)
	}
	out := [][]byte{
		bytes.Repeat([]byte{0xa5}, dModel*bf16Size),
		bytes.Repeat([]byte{0x5a}, dModel*bf16Size),
	}
	ptrs := []unsafe.Pointer{unsafe.Pointer(&out[0][0]), unsafe.Pointer(&out[1][0])}

	got, err := DecodeForwardArchICBQuantInto(out, inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm)
	if err != nil {
		t.Fatalf("DecodeForwardArchICBQuantInto: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("DecodeForwardArchICBQuantInto returned %d outputs, want %d", len(got), len(want))
	}
	for tok := range want {
		if len(got[tok]) != dModel*bf16Size || unsafe.Pointer(&got[tok][0]) != ptrs[tok] {
			t.Fatalf("DecodeForwardArchICBQuantInto token %d did not reuse caller-owned output backing", tok)
		}
		eqBytes(t, "DecodeForwardArchICBQuantInto token", got[tok], want[tok])
	}
}

func TestDecodeForwardArchICBQuantMixedDenseProjectionMatchesReencode(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers, maxLen = 64, 1, 1, 64, 128, 32, 1, 4
	const groupSize, bits = 64, 4
	arch := archFixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	inputs := decodeInputsFixture(2, dModel)
	layer := quantizedLayerFixture(t, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, 3)
	layer.Q = QuantWeight{Packed: toBF16Bytes(syntheticFloat32(nHeads*headDim*dModel, 101))}
	layer.Down = QuantWeight{Packed: toBF16Bytes(syntheticFloat32(dModel*dFF, 103))}
	layers := []QuantizedLayerWeights{layer}

	want, err := DecodeForwardArchQuant(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm)
	if err != nil {
		t.Fatalf("DecodeForwardArchQuant mixed dense projection: %v", err)
	}
	got, err := DecodeForwardArchICBQuant(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm)
	if err != nil {
		t.Fatalf("DecodeForwardArchICBQuant mixed dense projection: %v", err)
	}
	for tok := range want {
		eqBytes(t, "mixed dense projection token", got[tok], want[tok])
	}
}

func TestDecodeForwardArchICBQuantIntoPipelinedReusesOutputBacking(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers, maxLen = 64, 1, 1, 64, 128, 32, 1, 8
	const groupSize, bits = 64, 4
	arch := archFixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	inputs := decodeInputsFixture(4, dModel)
	layers := []QuantizedLayerWeights{quantizedLayerFixture(t, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, 3)}
	oldPipe := pipelinedBatchDisabled
	pipelinedBatchDisabled = true
	want, err := DecodeForwardArchICBQuant(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm)
	pipelinedBatchDisabled = oldPipe
	if err != nil {
		t.Fatalf("DecodeForwardArchICBQuant serial reference: %v", err)
	}
	out := make([][]byte, len(inputs))
	ptrs := make([]unsafe.Pointer, len(inputs))
	for tok := range out {
		out[tok] = bytes.Repeat([]byte{byte(0xa5 + tok)}, dModel*bf16Size)
		ptrs[tok] = unsafe.Pointer(&out[tok][0])
	}

	pipelinedBatchDisabled = false
	got, err := DecodeForwardArchICBQuantInto(out, inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm)
	pipelinedBatchDisabled = oldPipe
	if err != nil {
		t.Fatalf("DecodeForwardArchICBQuantInto pipelined: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("DecodeForwardArchICBQuantInto pipelined returned %d outputs, want %d", len(got), len(want))
	}
	for tok := range want {
		if len(got[tok]) != dModel*bf16Size || unsafe.Pointer(&got[tok][0]) != ptrs[tok] {
			t.Fatalf("DecodeForwardArchICBQuantInto pipelined token %d did not reuse caller-owned output backing", tok)
		}
		eqBytes(t, "DecodeForwardArchICBQuantInto pipelined token", got[tok], want[tok])
	}
}

// TestDecodeForwardArchICBQuant gates the stacked fast path — 4-bit qmv weights AND the
// ICB encode-bypass replay, arch-driven. It must equal DecodeForwardArchQuant (the quant
// re-encode arch path) byte-for-byte across every arch axis: all-owner/global, KV-share,
// sliding-window, and KV-share + sliding combined. The all-owner case is also tied to the
// non-arch DecodeForwardICBQuant. MoE layers route through the MoE-capable quant
// re-encode path instead of rejecting the direct ICB API.
func TestDecodeForwardArchICBQuant(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, gs, bits = 512, 8, 4, 64, 1024, 64, 4
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
	buildLayers := func(n int) []QuantizedLayerWeights {
		ls := make([]QuantizedLayerWeights, n)
		for li := range ls {
			ls[li] = buildQuantLayer(t, dModel, nHeads, nKV, headDim, dFF, gs, bits, (li+1)*100)
		}
		return ls
	}

	// check: DecodeForwardArchICBQuant ≡ DecodeForwardArchQuant byte-for-byte.
	check := func(name string, qlayers []QuantizedLayerWeights, specs []model.LayerSpec, T, slidingWindow int) {
		inputs := mkInputs(T)
		got, err := DecodeForwardArchICBQuant(inputs, qlayers, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, slidingWindow, base, scale, eps, false)
		if err != nil {
			t.Fatalf("%s: DecodeForwardArchICBQuant: %v", name, err)
		}
		want, err := DecodeForwardArchQuant(inputs, qlayers, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, slidingWindow, base, scale, eps, false)
		if err != nil {
			t.Fatalf("%s: DecodeForwardArchQuant: %v", name, err)
		}
		for tok := 0; tok < T; tok++ {
			eqBytes(t, core.Sprintf("%s tok%d", name, tok), got[tok], want[tok])
		}
	}

	// (a) all-owner/global — also tie to the non-arch quant ICB (DecodeForwardICBQuant).
	full3 := []string{"full_attention", "full_attention", "full_attention"}
	ql3 := buildLayers(3)
	check("all-owner/global", ql3, model.DeriveLayers(full3, 0), 4, 0)
	{
		inputs := mkInputs(4)
		gotArch, err := DecodeForwardArchICBQuant(inputs, ql3, model.DeriveLayers(full3, 0), dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
		if err != nil {
			t.Fatalf("arch-icb-quant: %v", err)
		}
		gotPlain, err := DecodeForwardICBQuant(inputs, ql3, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
		if err != nil {
			t.Fatalf("DecodeForwardICBQuant: %v", err)
		}
		for tok := 0; tok < 4; tok++ {
			eqBytes(t, core.Sprintf("all-owner vs DecodeForwardICBQuant tok%d", tok), gotArch[tok], gotPlain[tok])
		}
	}

	// (b) KV-share.
	check("kv-share", buildLayers(2), model.DeriveLayers([]string{"full_attention", "full_attention"}, 1), 4, 0)

	// (c) sliding-window W=3 over 6 tokens.
	slide3 := []string{"sliding_attention", "sliding_attention", "sliding_attention"}
	check("sliding-W3", buildLayers(3), model.DeriveLayers(slide3, 0), 6, 3)

	// (d) KV-share + sliding combined.
	mixed := []string{"sliding_attention", "full_attention", "sliding_attention", "full_attention"}
	check("kv-share+sliding", buildLayers(4), model.DeriveLayers(mixed, 2), 6, 3)

	// (e) MoE falls back to the quant re-encode arch path instead of rejecting the API.
	moeSpecs := model.DeriveLayers([]string{"full_attention", "full_attention"}, 0)
	moeSpecs[1].MoE = true
	moeLayers := buildLayers(2)
	moe := quantMoELayerWeightsGuard(t, 4, 2, dModel, dFF, 768, gs, bits)
	moeLayers[1].MoE = &moe
	moeInputs := mkInputs(3)
	gotMoE, err := DecodeForwardArchICBQuant(moeInputs, moeLayers, moeSpecs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArchICBQuant MoE fallback: %v", err)
	}
	wantMoE, err := DecodeForwardArchQuant(moeInputs, moeLayers, moeSpecs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArchQuant MoE: %v", err)
	}
	for tok := range moeInputs {
		eqBytes(t, core.Sprintf("quant moe fallback tok%d", tok), gotMoE[tok], wantMoE[tok])
	}

	t.Logf("stacked quant+ICB arch: replay ≡ DecodeForwardArchQuant byte-for-byte across all-owner/global, KV-share, sliding(W=3), KV-share+sliding; all-owner ≡ DecodeForwardICBQuant; direct MoE ICB API falls back to quant re-encode parity — both levers on the arch path")
}

func TestDecodeForwardArchICBQuantHonoursPerWeightGeometry(t *testing.T) {
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
	layers := []QuantizedLayerWeights{layer}
	specs := model.DeriveLayers([]string{"full_attention"}, 0)

	got, err := DecodeForwardArchICBQuant(inputs, layers, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArchICBQuant with per-weight MLP geometry: %v", err)
	}
	want, err := DecodeForwardArchQuant(inputs, layers, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArchQuant with per-weight MLP geometry: %v", err)
	}
	for tok := range got {
		eqBytes(t, core.Sprintf("mixed quant arch ICB vs DecodeForwardArchQuant tok%d", tok), got[tok], want[tok])
	}
}

func TestDecodeForwardArchICBQuantKeepsFixedWeightsResident(t *testing.T) {
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

	if _, err := DecodeForwardArchICBQuant(inputs, layers, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false); err != nil {
		t.Fatalf("DecodeForwardArchICBQuant: %v", err)
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
		t.Fatalf("DecodeForwardArchICBQuant did not keep fixed weights resident (missing=%v resident=%d want>=%d)", missing, got, len(weights))
	}
}

func TestDecodeForwardArchICBQuantPLE(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, gs, bits = 128, 2, 1, 64, 256, 32, 4
	const vocab, vocabPLI, pliDim = 19, 23, 32
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	const maxLen = 6
	tokenIDs := []int32{1, 5, 3, 7}
	specs := model.DeriveLayers([]string{"full_attention", "full_attention"}, 0)

	embed, embedScales, embedBiases := quantizeProj(t, vocab, dModel, gs, bits, 31)
	inputs, err := EmbedTokensQuant(embed, embedScales, embedBiases, tokenIDs, vocab, dModel, gs, bits, 1)
	if err != nil {
		t.Fatalf("EmbedTokensQuant: %v", err)
	}

	qLayers := make([]QuantizedLayerWeights, len(specs))
	for li := range qLayers {
		qLayers[li] = buildQuantLayer(t, dModel, nHeads, nKV, headDim, dFF, gs, bits, (li+1)*100)
		qLayers[li].PerLayerGate = quantWeightFixture(t, pliDim, dModel, gs, bits, li*10+41)
		qLayers[li].PerLayerProjection = quantWeightFixture(t, dModel, pliDim, gs, bits, li*10+43)
		qLayers[li].PostPerLayerInputNormW = toBF16Bytes(syntheticFloat32(dModel, li*10+47))
		qLayers[li].LayerScalarW = toBF16Bytes([]float32{0.75 + float32(li)*0.125})
	}

	plDim := len(specs) * pliDim
	embedPL, embedPLScales, embedPLBiases := quantizeProj(t, vocabPLI, plDim, gs, bits, 53)
	projPL, projPLScales, projPLBiases := quantizeProj(t, plDim, dModel, gs, bits, 59)
	ple := ArchPLEQuant{
		TokenIDs:      tokenIDs,
		EmbedPerLayer: embedPL, EmbedPerLayerScales: embedPLScales, EmbedPerLayerBiases: embedPLBiases,
		PerLayerModelProjW: projPL, PerLayerModelProjScales: projPLScales, PerLayerModelProjBiases: projPLBiases,
		PerLayerProjNormW: toBF16Bytes(syntheticFloat32(pliDim, 61)),
		VocabPLI:          vocabPLI, PliDim: pliDim,
		GroupSize: gs, Bits: bits, ProjGroupSize: gs, ProjBits: bits,
	}

	got, err := DecodeForwardArchICBQuant(inputs, qLayers, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false, ple)
	if err != nil {
		t.Fatalf("DecodeForwardArchICBQuant PLE: %v", err)
	}
	want, err := DecodeForwardArchQuant(inputs, qLayers, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false, ple)
	if err != nil {
		t.Fatalf("DecodeForwardArchQuant PLE: %v", err)
	}
	h := sha256.New()
	for tok := range tokenIDs {
		eqBytes(t, core.Sprintf("quant PLE ICB tok%d", tok), got[tok], want[tok])
		_, _ = h.Write(got[tok])
	}
	gotHash := core.Sprintf("%x", h.Sum(nil))
	// Golden over the SYNTHETIC fixture's output — the real invariant is the
	// eqBytes above (ICB replay ≡ DecodeForwardArchQuant byte-for-byte); the hash
	// only pins fixture drift. Minted for the pure-Go packAffineQuant fixture
	// (test_helpers_test.go); re-mint deliberately if the fixture changes again.
	const wantHash = "54fa4bbb358da8ab8f922352e0b23335e384752dd68914495c84ae28e8e44298"
	if gotHash != wantHash {
		t.Fatalf("quant PLE ICB hash = %s, want %s", gotHash, wantHash)
	}
	t.Logf("quant PLE arch ICB: replay ≡ DecodeForwardArchQuant byte-for-byte with token-id PerLayerInputs, PLE gate, post norm, and layer scalar; sha256=%s", gotHash)
}
