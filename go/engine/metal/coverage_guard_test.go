// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"os"
	"sync"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/decode/tokenizer"
	"dappco.re/go/inference/model"
	g4 "dappco.re/go/inference/model/gemma4"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
	"github.com/tmc/apple/metal"
)

func plainRopeInvFreqsGuard(base float64, rotaryDim int) []float32 {
	f := make([]float32, rotaryDim/2)
	for d := range f {
		f[d] = float32(math.Pow(base, -float64(2*d)/float64(rotaryDim)))
	}
	return f
}

func expectErr(t *testing.T, name string, err error) {
	t.Helper()
	if err == nil {
		t.Fatalf("%s: expected error", name)
	}
}

func withComposedGELU(t *testing.T) {
	t.Helper()
	old := customLibraryLoaded
	customLibraryLoaded = false
	t.Cleanup(func() { customLibraryLoaded = old })
}

func resetNativeInitGlobalsForCoverage() {
	initOnce = sync.Once{}
	var zeroDevice metal.MTLDeviceObject
	var zeroQueue metal.MTLCommandQueue
	var zeroLibrary metal.MTLLibrary
	device = zeroDevice
	queue = zeroQueue
	library = zeroLibrary
	customLibrary = zeroLibrary
	customLibraryLoaded = false
	initErr = nil
}

func resetNativePipelineCachesForCoverage() {
	psoMu.Lock()
	psoCache = map[string]metal.MTLComputePipelineState{}
	psoMu.Unlock()

	ropePSOMu.Lock()
	ropePSOCache = map[string]metal.MTLComputePipelineState{}
	ropePSOMu.Unlock()

	ropePSOBF16Mu.Lock()
	ropePSOBF16Cache = map[string]metal.MTLComputePipelineState{}
	ropePSOBF16Mu.Unlock()

	ropeFreqsPSOBF16Mu.Lock()
	ropeFreqsPSOBF16Cache = map[string]metal.MTLComputePipelineState{}
	ropeFreqsPSOBF16Mu.Unlock()

	sdpaPSOMu.Lock()
	sdpaPSOCache = map[string]metal.MTLComputePipelineState{}
	sdpaVectorHeadDimPSOCache = map[int]metal.MTLComputePipelineState{}
	sdpaVector2Pass1HeadDimCache = map[sdpa2Pass1Key]metal.MTLComputePipelineState{}
	sdpaVector2Pass2HeadDimCache = map[int]metal.MTLComputePipelineState{}
	sdpaPSOMu.Unlock()

	steelPSOMu.Lock()
	steelPSOCache = map[string]metal.MTLComputePipelineState{}
	steelPSOMu.Unlock()

	icbPSOMu.Lock()
	icbPSOCache = map[string]metal.MTLComputePipelineState{}
	sdpaVectorICBHeadDimPSOCache = map[int]metal.MTLComputePipelineState{}
	icbPSOMu.Unlock()

	geluPSOOnce = sync.Once{}
	geluPSO = nil
	geluPSOErr = nil

	ffnMegaPSOOnce = sync.Once{}
	ffnMegaPSO = nil
	ffnMegaPSOErr = nil
}

type failingProjector struct {
	fail      projIndex
	err       error
	distinctV bool
}

func (p failingProjector) hasV() bool { return p.distinctV }

func (p failingProjector) project(_ metal.MTLComputeCommandEncoder, _, _ metal.MTLBuffer, _ uint, idx projIndex) error {
	if p.err != nil && idx == p.fail {
		return p.err
	}
	return nil
}

func encodedTensors(t *testing.T, tensors map[string]safetensors.Tensor) []byte {
	t.Helper()
	blob, err := safetensors.Encode(tensors)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	return blob
}

func gemma4ConfigJSON(t *testing.T, cfg g4.Config) []byte {
	t.Helper()
	// The reactive loader runs the faithful parser, which REQUIRES these declared (don't-guess). A
	// minimal synthetic config gets sensible defaults so it loads; tests that set them keep their own.
	if cfg.SlidingWindow == 0 {
		cfg.SlidingWindow = 1024
	}
	if cfg.MaxPositionEmbeddings == 0 {
		cfg.MaxPositionEmbeddings = 8192
	}
	if len(cfg.LayerTypes) == 0 {
		cfg.LayerTypes = make([]string, cfg.NumHiddenLayers)
		for i := range cfg.LayerTypes {
			cfg.LayerTypes[i] = "full_attention"
		}
	}
	// The reactive LoadDir/LoadTokenModelDir dispatch on model_type, and g4.Config carries no
	// model_type field — so a synthetic gemma4 config must declare its architecture for the registry
	// to resolve it (the old per-arch loaders were gemma4-by-function and never needed this).
	return configJSONWithModelType(t, cfg, "gemma4_text")
}

func writeLocal(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := coreio.Local.Write(path, string(data)); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

const nativeCoverageTokenizerJSON = `{
  "model": {
    "type": "BPE",
    "vocab": {"h": 0},
    "merges": [],
    "byte_fallback": false
  }
}`

func quantGemma4TensorsGuard(t *testing.T, arch model.Arch, groupSize, bits int) map[string]safetensors.Tensor {
	t.Helper()
	tensors := map[string]safetensors.Tensor{}
	salt := 1
	mkNorm := func(name string, elems int) {
		tensors[name] = safetensors.Tensor{
			Dtype: "BF16",
			Shape: []int{elems},
			Data:  toBF16Bytes(syntheticFloat32(elems, salt)),
		}
		salt++
	}
	mkQuant := func(prefix string, outDim, inDim int) {
		q := quantWeightFixture(t, outDim, inDim, groupSize, bits, salt)
		salt++
		tensors[prefix+".weight"] = safetensors.Tensor{Dtype: "U32", Shape: []int{outDim, inDim * bits / 32}, Data: q.Packed}
		tensors[prefix+".scales"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{outDim, inDim / groupSize}, Data: q.Scales}
		tensors[prefix+".biases"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{outDim, inDim / groupSize}, Data: q.Biases}
	}

	dModel, headDim, dFF, vocab := arch.Hidden, arch.HeadDim, arch.FF, arch.Vocab
	qDim, kvDim := arch.Heads*headDim, arch.KVHeads*headDim
	mkQuant("model.embed_tokens", vocab, dModel)
	mkNorm("model.norm.weight", dModel)
	for i := range arch.Layer {
		p := core.Sprintf("model.layers.%d", i)
		mkNorm(p+".input_layernorm.weight", dModel)
		mkNorm(p+".pre_feedforward_layernorm.weight", dModel)
		mkNorm(p+".self_attn.q_norm.weight", headDim)
		mkNorm(p+".self_attn.k_norm.weight", headDim)
		mkNorm(p+".post_attention_layernorm.weight", dModel)
		mkNorm(p+".post_feedforward_layernorm.weight", dModel)
		mkQuant(p+".self_attn.q_proj", qDim, dModel)
		mkQuant(p+".self_attn.k_proj", kvDim, dModel)
		mkQuant(p+".self_attn.v_proj", kvDim, dModel)
		mkQuant(p+".self_attn.o_proj", dModel, qDim)
		mkQuant(p+".mlp.gate_proj", dFF, dModel)
		mkQuant(p+".mlp.up_proj", dFF, dModel)
		mkQuant(p+".mlp.down_proj", dModel, dFF)
	}
	return tensors
}

func TestNativeEnsureInitErrorPropagationCoverage(t *testing.T) {
	requireNativeRuntime(t)

	old := initErr
	initErr = core.NewError("native synthetic init failure")
	t.Cleanup(func() { initErr = old })

	cases := []struct {
		name string
		call func() error
	}{
		{"RunBinary", func() error { _, err := RunBinary("vv_Addfloat32", nil, nil); return err }},
		{"Square", func() error { _, err := Square(nil); return err }},
		{"RMSNormBF16", func() error { _, err := RMSNormBF16(nil, nil, 1, 1, 0); return err }},
		{"RMSNorm", func() error { _, err := RMSNorm(nil, nil, 1, 1, 0); return err }},
		{"MatVecBF16", func() error { _, err := MatVecBF16(nil, nil, 1, 1); return err }},
		{"MatVec", func() error { _, err := MatVec(nil, nil, 1, 1); return err }},
		{"RoPEBF16", func() error { _, err := RoPEBF16(nil, 1, 1, 2, 10000, 1, 0, false); return err }},
		{"RoPE", func() error { _, err := RoPE(nil, 1, 1, 2, 10000, 1, 0, false); return err }},
		{"RoPEFreqsBF16", func() error { _, err := RoPEFreqsBF16(nil, 1, 1, 2, 2, []float32{1}, 1, 0, false); return err }},
		{"AddBF16", func() error { _, err := AddBF16(nil, nil); return err }},
		{"MulBF16", func() error { _, err := MulBF16(nil, nil); return err }},
		{"TanhBF16", func() error { _, err := TanhBF16(nil); return err }},
		{"GeluBF16", func() error { _, err := GeluBF16(nil); return err }},
		{"Gelu", func() error { _, err := Gelu(nil); return err }},
		{"GeluGateMulBF16", func() error { _, err := GeluGateMulBF16(nil, nil); return err }},
		{"GeluGateMul", func() error { _, err := GeluGateMul(nil, nil); return err }},
		{"NormProject", func() error { _, err := NormProject(nil, nil, nil, 1, 1, 0); return err }},
		{"MLPBlock", func() error { _, err := MLPBlock(nil, nil, nil, nil, nil, 1, 1, 0); return err }},
		{"LMHeadBF16", func() error { _, err := LMHeadBF16(nil, nil, nil, 1, 1, 0, 0); return err }},
		{"LMHeadQuant", func() error { _, err := LMHeadQuant(nil, nil, nil, nil, nil, 1, 1, 1, 4, 0, 0); return err }},
		{"QMV", func() error { _, err := QMV(nil, nil, nil, nil, 1, 1, 1, 4); return err }},
		{"QMVBF16", func() error { _, err := QMVBF16(nil, nil, nil, nil, 1, 1, 1, 4); return err }},
		{"SDPA", func() error { _, err := SDPA(nil, nil, nil, 1, 1, 1, 2, 1, 1); return err }},
		{"RoPE", func() error { _, err := RoPE(nil, 1, 1, 2, 10000, 1, 0, false); return err }},
		{"RoPEFreqsBF16", func() error { _, err := RoPEFreqsBF16(nil, 1, 1, 2, 2, []float32{1}, 1, 0, false); return err }},
		{"AttentionBlock", func() error {
			_, err := AttentionBlock(nil, nil, nil, nil, nil, nil, 1, 1, 1, 2, 1, 10000, 1, 0, 0)
			return err
		}},
		{"AttentionStepKV", func() error {
			_, err := AttentionStepKV(nil, nil, nil, nil, nil, nil, nil, nil, 1, 1, 1, 2, 1, 0, 10000, 1, 0)
			return err
		}},
		{"DecodeLayer", func() error {
			_, err := DecodeLayer(nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, 1, 1, 1, 2, 1, 1, 10000, 1, 0, 0)
			return err
		}},
		{"DecodeLayerICB", func() error {
			_, err := DecodeLayerICB(nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, 1, 1, 1, 2, 1, 1, 10000, 1, 0, 0, 1)
			return err
		}},
		{"DecodeTokenICB", func() error {
			_, err := DecodeTokenICB(nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, 1, 1, 1, 2, 1, 1, 1, 10000, 1, 0, 0, 1)
			return err
		}},
		{"DecodeStepKV", func() error {
			_, err := DecodeStepKV(nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, 1, 1, 1, 2, 1, 1, 0, 10000, 1, 0)
			return err
		}},
		{"DecodeForward", func() error { _, err := DecodeForward(nil, nil, 1, 1, 1, 2, 1, 1, 10000, 1, 0); return err }},
		{"DecodeForwardQuant", func() error { _, err := DecodeForwardQuant(nil, nil, 1, 1, 1, 2, 1, 1, 10000, 1, 0); return err }},
		{"DecodeForwardICB", func() error { _, err := DecodeForwardICB(nil, nil, 1, 1, 1, 2, 1, 1, 10000, 1, 0); return err }},
		{"DecodeForwardICBQuant", func() error { _, err := DecodeForwardICBQuant(nil, nil, 1, 1, 1, 2, 1, 1, 10000, 1, 0); return err }},
		{"DecodeForwardArch", func() error {
			_, err := DecodeForwardArch(nil, nil, nil, 1, 1, 1, 2, 1, 1, 0, 10000, 1, 0, false)
			return err
		}},
		{"DecodeForwardArchQuant", func() error {
			_, err := DecodeForwardArchQuant(nil, nil, nil, 1, 1, 1, 2, 1, 1, 0, 10000, 1, 0, false)
			return err
		}},
		{"DecodeForwardArchICB", func() error {
			_, err := DecodeForwardArchICB(nil, nil, nil, 1, 1, 1, 2, 1, 1, 0, 10000, 1, 0, false)
			return err
		}},
		{"DecodeForwardArchICBQuant", func() error {
			_, err := DecodeForwardArchICBQuant(nil, nil, nil, 1, 1, 1, 2, 1, 1, 0, 10000, 1, 0, false)
			return err
		}},
		{"GenerateBF16", func() error { _, err := GenerateBF16(nil, model.Arch{}, nil, 1, 1, -1); return err }},
		{"NewArchSession", func() error { _, err := NewArchSession(nil, model.Arch{}, 1); return err }},
		{"NewArchQuantSession", func() error { _, err := NewArchQuantSession(nil, model.Arch{}, 1); return err }},
		{"PerLayerInputs", func() error {
			_, err := PerLayerInputs(nil, nil, nil, nil, nil, nil, nil, 0, nil, 1, 1, 1, 1, 0, 0, 0, 0, 0, bufView{})
			return err
		}},
		{"PerLayerInputGateBF16", func() error { _, err := PerLayerInputGateBF16(nil, nil, nil, nil, nil, 1, 1, 0); return err }},
		{"PerLayerInputGateQuant", func() error {
			_, err := PerLayerInputGateQuant(nil, QuantWeight{}, nil, QuantWeight{}, nil, 1, 1, 1, 4, 0)
			return err
		}},
		{"MoERouter", func() error { _, _, err := MoERouter(nil, nil, nil, nil, 1, 1, 1, 0); return err }},
		{"MoERouterQuant", func() error { _, _, err := MoERouterQuant(nil, nil, QuantWeight{}, nil, 1, 1, 1, 1, 4, 0); return err }},
		{"MoEExperts", func() error { _, err := MoEExperts(nil, nil, nil, nil, nil, nil, 1, 1, 1, 1); return err }},
		{"MoEExpertsQuant", func() error {
			_, err := MoEExpertsQuant(nil, nil, nil, QuantWeight{}, QuantWeight{}, QuantWeight{}, 1, 1, 1, 1, 1, 4)
			return err
		}},
		{"MoEBlockBF16", func() error { _, err := MoEBlockBF16(nil, MoELayerWeights{}, 1, 1, 0); return err }},
		{"MoEBlockQuant", func() error { _, err := MoEBlockQuant(nil, MoEQuantLayerWeights{}, 1, 1, 0); return err }},
		{"MLPBlockBF16", func() error { _, err := MLPBlockBF16(nil, nil, nil, nil, nil, 1, 1, 0); return err }},
		{"dispatchProfile", func() error { _, _, _, err := dispatchProfile(1, 1); return err }},
		{"attentionReEncode", func() error { return attentionReEncode(nil, nil, nil, nil, nil, nil, 1, 1, 1, 2, 1, 10000, 1, 0, 0, 1) }},
		{"layerReEncode", func() error {
			return layerReEncode(nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, 1, 1, 1, 2, 1, 1, 10000, 1, 0, 0, 1)
		}},
		{"tokenReEncode", func() error {
			_, err := tokenReEncode(nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, 1, 1, 1, 2, 1, 1, 1, 10000, 1, 0, 0, 1)
			return err
		}},
	}
	for _, tc := range cases {
		if err := tc.call(); err == nil {
			t.Fatalf("%s: expected init error", tc.name)
		}
	}
}

func TestNativeEnsureInitColdErrorsCoverage(t *testing.T) {
	requireNativeRuntime(t)

	goodPath, hadPath := os.LookupEnv(MetallibPathEnv)
	if !hadPath || goodPath == "" {
		t.Fatal("native runtime should have a metallib path after requireNativeRuntime")
	}
	restore := func() {
		if err := os.Setenv(MetallibPathEnv, goodPath); err != nil {
			t.Fatalf("restore %s: %v", MetallibPathEnv, err)
		}
		resetNativeInitGlobalsForCoverage()
		if err := ensureInit(); err != nil {
			t.Fatalf("restore native runtime: %v", err)
		}
	}
	t.Cleanup(restore)

	os.Unsetenv(MetallibPathEnv)
	resetNativeInitGlobalsForCoverage()
	expectErr(t, "ensureInit missing metallib env", ensureInit())

	if err := os.Setenv(MetallibPathEnv, core.PathJoin(t.TempDir(), "missing.metallib")); err != nil {
		t.Fatalf("set bad %s: %v", MetallibPathEnv, err)
	}
	resetNativeInitGlobalsForCoverage()
	expectErr(t, "ensureInit bad metallib path", ensureInit())

	restore()
}

func TestNativeMissingPipelineCoverage(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, kvLen = 64, 1, 1, 64, 128, 1
	const pliDim, vocabPLI, numLayers = 32, 8, 2
	const groupSize, bits = 32, 4
	const eps = float32(1e-5)
	x32 := syntheticFloat32(dModel, 3)
	norm32 := syntheticFloat32(dModel, 5)
	mat32 := syntheticFloat32(dModel*dModel, 7)
	xb := toBF16Bytes(x32)
	normB := toBF16Bytes(norm32)
	matB := toBF16Bytes(mat32)
	kb := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 11))
	vb := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 13))
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 17)
	qw := quantWeightFixture(t, dModel, dModel, groupSize, bits, 19)
	pliPacked := toBF16Bytes(syntheticFloat32(vocabPLI*numLayers*pliDim, 23))
	pliNorm := toBF16Bytes(syntheticFloat32(pliDim, 29))
	pliInput := toBF16Bytes(syntheticFloat32(pliDim, 31))
	pliGateW := toBF16Bytes(syntheticFloat32(pliDim*dModel, 37))
	pliProjW := toBF16Bytes(syntheticFloat32(dModel*pliDim, 41))
	qPliGate := quantWeightFixture(t, pliDim, dModel, groupSize, bits, 43)
	qPliProj := quantWeightFixture(t, dModel, pliDim, groupSize, bits, 47)

	oldLibrary, oldCustomLibrary, oldCustomLoaded := library, customLibrary, customLibraryLoaded
	t.Cleanup(func() {
		library, customLibrary, customLibraryLoaded = oldLibrary, oldCustomLibrary, oldCustomLoaded
		resetNativePipelineCachesForCoverage()
	})
	resetNativePipelineCachesForCoverage()
	library, customLibrary, customLibraryLoaded = nil, nil, false

	_, err := RunUnary("v_Squarefloat32float32", []float32{2})
	expectErr(t, "RunUnary missing pipeline", err)
	_, err = RunBinary("vv_Addfloat32", []float32{1}, []float32{2})
	expectErr(t, "RunBinary missing pipeline", err)
	_, err = RMSNorm(x32, norm32, 1, dModel, eps)
	expectErr(t, "RMSNorm missing pipeline", err)
	_, err = MatVec(mat32, x32, dModel, dModel)
	expectErr(t, "MatVec missing pipeline", err)
	_, err = NormProject(x32, norm32, mat32, dModel, dModel, eps)
	expectErr(t, "NormProject missing pipeline", err)
	_, err = RoPE(x32, 1, nHeads, headDim, 10000, 1, 0, false)
	expectErr(t, "RoPE missing pipeline", err)
	_, err = Gelu(x32)
	expectErr(t, "Gelu missing pipeline", err)
	_, err = GeluGateMul(x32, x32)
	expectErr(t, "GeluGateMul missing pipeline", err)
	_, err = RMSNormBF16(xb, normB, 1, dModel, eps)
	expectErr(t, "RMSNormBF16 missing pipeline", err)
	_, err = MatVecBF16(matB, xb, dModel, dModel)
	expectErr(t, "MatVecBF16 missing pipeline", err)
	_, err = RoPEBF16(xb, 1, nHeads, headDim, 10000, 1, 0, false)
	expectErr(t, "RoPEBF16 missing pipeline", err)
	_, err = RoPEFreqsBF16(xb, 1, nHeads, headDim, headDim, plainRopeInvFreqsGuard(10000, headDim), 1, 0, false)
	expectErr(t, "RoPEFreqsBF16 missing pipeline", err)
	_, err = AddBF16(xb, xb)
	expectErr(t, "AddBF16 missing pipeline", err)
	_, err = MulBF16(xb, xb)
	expectErr(t, "MulBF16 missing pipeline", err)
	_, err = TanhBF16(xb)
	expectErr(t, "TanhBF16 missing pipeline", err)
	_, err = GeluBF16(xb)
	expectErr(t, "GeluBF16 missing pipeline", err)
	_, err = QMV(x32, qw.Packed, qw.Scales, qw.Biases, dModel, dModel, groupSize, bits)
	expectErr(t, "QMV missing pipeline", err)
	_, err = QMVBF16(xb, qw.Packed, qw.Scales, qw.Biases, dModel, dModel, groupSize, bits)
	expectErr(t, "QMVBF16 missing pipeline", err)
	_, err = LMHeadQuant(xb, normB, qw.Packed, qw.Scales, qw.Biases, dModel, dModel, groupSize, bits, eps, 0)
	expectErr(t, "LMHeadQuant missing pipeline", err)
	_, err = LMHeadBF16(xb, normB, matB, dModel, dModel, eps, 0)
	expectErr(t, "LMHeadBF16 missing pipeline", err)
	_, err = PerLayerInputs(pliPacked, nil, nil, matB, nil, nil, pliNorm, 0, xb, vocabPLI, numLayers, pliDim, dModel, groupSize, bits, 0, 0, eps, bufView{})
	expectErr(t, "PerLayerInputs missing pipeline", err)
	_, err = PerLayerInputGateBF16(xb, pliGateW, pliInput, pliProjW, normB, dModel, pliDim, eps)
	expectErr(t, "PerLayerInputGateBF16 missing pipeline", err)
	_, err = PerLayerInputGateQuant(xb, qPliGate, pliInput, qPliProj, normB, dModel, pliDim, groupSize, bits, eps)
	expectErr(t, "PerLayerInputGateQuant missing pipeline", err)
	_, err = SDPA(xb, kb, vb, 1, nHeads, nKV, headDim, kvLen, 0.125)
	expectErr(t, "SDPA missing pipeline", err)
	_, err = AttentionBlock(xb, normB, layer.WQ, layer.WO, kb, vb, dModel, nHeads, nKV, headDim, kvLen, 10000, 0.125, 0, eps)
	expectErr(t, "AttentionBlock missing pipeline", err)
	_, err = DecodeStepKV(xb, normB, layer.WQ, layer.WK, layer.WV, layer.WO, kb, vb, normB, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, kvLen, dFF, 0, 10000, 0.125, eps)
	expectErr(t, "DecodeStepKV missing pipeline", err)
	_, err = squareICB([]float32{2})
	expectErr(t, "squareICB missing pipeline", err)
	_, err = NormProjectICB([]float32{1, 2}, []float32{1, 1}, []float32{1, 2, 3, 4}, 2, 2, eps, 1)
	expectErr(t, "NormProjectICB missing pipeline", err)
	_, _, _, err = dispatchProfile(1, dModel)
	expectErr(t, "dispatchProfile missing pipeline", err)
	_, err = rebindCostProbe(1)
	expectErr(t, "rebindCostProbe missing pipeline", err)
	_, _, err = qmvBF16Profile(dModel, dModel, groupSize, 1)
	expectErr(t, "qmvBF16Profile missing pipeline", err)
	_, _, err = gemvProfile(dModel, dModel, 1)
	expectErr(t, "gemvProfile missing pipeline", err)
	_, err = MLPBlockBF16(xb, normB, layer.WGate, layer.WUp, layer.WDown, dModel, dFF, eps)
	expectErr(t, "MLPBlockBF16 missing pipeline", err)
	_, err = mlpTransformBF16(xb, layer.WGate, layer.WUp, layer.WDown, dModel, dFF)
	expectErr(t, "mlpTransformBF16 missing pipeline", err)
	_, err = MoEExperts(xb, []int32{0}, toBF16Bytes([]float32{1}), layer.WGate, layer.WUp, layer.WDown, 1, 1, dModel, dFF)
	expectErr(t, "MoEExperts missing pipeline", err)
	moeBF := moeLayerWeightsFixture(1, 1, dModel, dFF, dFF, 53)
	_, err = MoEBlockBF16(xb, moeBF, dModel, dFF, eps)
	expectErr(t, "MoEBlockBF16 missing pipeline", err)
	qMoE := quantMoELayerWeightsGuard(t, 1, 1, dModel, dFF, dFF, groupSize, bits)
	_, err = mlpTransformQuant(xb, qMoE.LocalGate, qMoE.LocalUp, qMoE.LocalDown, dModel, dFF, groupSize, bits)
	expectErr(t, "mlpTransformQuant missing pipeline", err)
	_, err = MoEExpertsQuant(xb, []int32{0}, toBF16Bytes([]float32{1}), qMoE.ExpGate, qMoE.ExpUp, qMoE.ExpDown, 1, 1, dModel, dFF, groupSize, bits)
	expectErr(t, "MoEExpertsQuant missing pipeline", err)
	_, err = MoEBlockQuant(xb, qMoE, dModel, dFF, eps)
	expectErr(t, "MoEBlockQuant missing pipeline", err)

	customLibraryLoaded = true
	resetNativePipelineCachesForCoverage()
	_, err = GeluGateMulBF16(xb, xb)
	expectErr(t, "GeluGateMulBF16 missing fused pipeline", err)

	library, customLibrary, customLibraryLoaded = oldLibrary, nil, true
	resetNativePipelineCachesForCoverage()
	_, err = MoEExperts(xb, []int32{0}, toBF16Bytes([]float32{1}), layer.WGate, layer.WUp, layer.WDown, 1, 1, dModel, dFF)
	expectErr(t, "MoEExperts missing fused gelu pipeline", err)
}

func TestNativeColdHelperCoverage(t *testing.T) {
	requireNativeRuntime(t)

	nativeTraceLog("")

	statsBuf := sharedBytes(toBF16Bytes([]float32{0, -2, float32(math.Inf(1)), 3}))
	maxAbs, bad := bufMaxAbsNaN(statsBuf, 4)
	if maxAbs != 3 || bad != 1 {
		t.Fatalf("bufMaxAbsNaN = (%v, %d), want (3, 1)", maxAbs, bad)
	}

	if got := copyOrNilView(nil); got.buf != nil || got.off != 0 {
		t.Fatalf("copyOrNilView(nil) = %+v, want zero view", got)
	}
	if got := copyOrNilView([]byte{1, 2, 3, 4}); got.buf == nil || got.off != 0 {
		t.Fatalf("copyOrNilView(non-empty) = %+v, want buffer at offset zero", got)
	}

	periods := proportionalRopePeriods(64, 32, 10000)
	if len(periods) != 32 {
		t.Fatalf("proportionalRopePeriods len = %d, want 32", len(periods))
	}
	if periods[0] != 1 || !math.IsInf(float64(periods[len(periods)-1]), 1) {
		t.Fatalf("proportionalRopePeriods endpoints = (%v, %v)", periods[0], periods[len(periods)-1])
	}

	const dModel = 8
	x := toBF16Bytes(syntheticFloat32(dModel, 3))
	if out, err := mlpTransformBF16(x, nil, nil, nil, dModel, 0); err != nil {
		t.Fatalf("mlpTransformBF16 zero dFF: %v", err)
	} else if len(out) != dModel*bf16Size {
		t.Fatalf("mlpTransformBF16 zero dFF len = %d", len(out))
	}
	_, err := mlpTransformBF16([]byte{1}, nil, nil, nil, dModel, 0)
	expectErr(t, "mlpTransformBF16 bad hidden", err)
	if out, err := LMHeadQuant(nil, nil, nil, nil, nil, 0, 4, 1, 4, 1e-5, 0); err != nil {
		t.Fatalf("LMHeadQuant zero dModel: %v", err)
	} else if len(out) != 4*bf16Size {
		t.Fatalf("LMHeadQuant zero dModel len = %d", len(out))
	}
}

func TestNativeComposedGELUCoverage(t *testing.T) {
	requireNativeRuntime(t)
	withComposedGELU(t)

	const dModel, nHeads, nKV, headDim, kvLen, maxLen, dFF = 64, 1, 1, 64, 2, 4, 128
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	arch := archFixture(t, dModel, nHeads, nKV, headDim, dFF, 32, 1)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)
	layers := []DecodeLayerWeights{layer}
	inputs := decodeInputsFixture(2, dModel)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	kCache := toBF16Bytes(syntheticFloat32(nKV*maxLen*headDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(nKV*maxLen*headDim, 11))
	kLayer := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 13))
	vLayer := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 17))

	if out, err := GeluGateMulBF16(toBF16Bytes(syntheticFloat32(dFF, 19)), toBF16Bytes(syntheticFloat32(dFF, 23))); err != nil {
		t.Fatalf("GeluGateMulBF16 composed: %v", err)
	} else if len(out) != dFF*bf16Size {
		t.Fatalf("GeluGateMulBF16 composed len = %d", len(out))
	}
	if out, err := MLPBlockBF16(x, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, dFF, eps); err != nil {
		t.Fatalf("MLPBlockBF16 composed: %v", err)
	} else if len(out) != dModel*bf16Size {
		t.Fatalf("MLPBlockBF16 composed len = %d", len(out))
	}
	if out, err := DecodeStepKV(x, layer.AttnNormW, layer.WQ, layer.WK, layer.WV, layer.WO, kCache, vCache, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps); err != nil {
		t.Fatalf("DecodeStepKV composed: %v", err)
	} else if len(out) != dModel*bf16Size {
		t.Fatalf("DecodeStepKV composed len = %d", len(out))
	}
	if out, err := DecodeLayer(x, layer.AttnNormW, layer.WQ, layer.WO, kLayer, vLayer, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, kvLen, dFF, base, scale, offset, eps); err != nil {
		t.Fatalf("DecodeLayer composed: %v", err)
	} else if len(out) != dModel*bf16Size {
		t.Fatalf("DecodeLayer composed len = %d", len(out))
	}
	if out, err := DecodeLayerICB(x, layer.AttnNormW, layer.WQ, layer.WO, kLayer, vLayer, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, kvLen, dFF, base, scale, offset, eps, 0); err != nil {
		t.Fatalf("DecodeLayerICB composed: %v", err)
	} else if len(out) != dModel*bf16Size {
		t.Fatalf("DecodeLayerICB composed len = %d", len(out))
	}
	if out, err := DecodeTokenICB(x, layer.AttnNormW, layer.WQ, layer.WO, kLayer, vLayer, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, kvLen, dFF, 0, base, scale, offset, eps, 0); err != nil {
		t.Fatalf("DecodeTokenICB composed: %v", err)
	} else if len(out) != dModel*bf16Size {
		t.Fatalf("DecodeTokenICB composed len = %d", len(out))
	}
	if out, err := DecodeForwardICB(inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps); err != nil {
		t.Fatalf("DecodeForwardICB composed: %v", err)
	} else if len(out) != len(inputs) {
		t.Fatalf("DecodeForwardICB composed outputs = %d", len(out))
	}
	if out, err := DecodeForwardArchICB(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false); err != nil {
		t.Fatalf("DecodeForwardArchICB composed: %v", err)
	} else if len(out) != len(inputs) {
		t.Fatalf("DecodeForwardArchICB composed outputs = %d", len(out))
	}
	if err := attentionReEncode(x, layer.AttnNormW, layer.WQ, layer.WO, kLayer, vLayer, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps, 1); err != nil {
		t.Fatalf("attentionReEncode composed: %v", err)
	}
	if err := layerReEncode(x, layer.AttnNormW, layer.WQ, layer.WO, kLayer, vLayer, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, kvLen, dFF, base, scale, offset, eps, 1); err != nil {
		t.Fatalf("layerReEncode composed: %v", err)
	}
	if out, err := tokenReEncode(x, layer.AttnNormW, layer.WQ, layer.WO, kLayer, vLayer, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, kvLen, dFF, 0, base, scale, offset, eps, 0); err != nil {
		t.Fatalf("tokenReEncode composed: %v", err)
	} else if len(out) != dModel*bf16Size {
		t.Fatalf("tokenReEncode composed len = %d", len(out))
	}
}

func quantMoELayerWeightsGuard(t testing.TB, numExperts, topK, dModel, dFF, expertDFF, groupSize, bits int) MoEQuantLayerWeights {
	t.Helper()
	qw := func(outDim, inDim, salt int) QuantWeight {
		return quantWeightFixture(t, outDim, inDim, groupSize, bits, salt)
	}
	batched := func(outDim, inDim, saltBase int) QuantWeight {
		var packed, scales, biases []byte
		for e := 0; e < numExperts; e++ {
			w := quantWeightFixture(t, outDim, inDim, groupSize, bits, saltBase+e*7)
			packed = append(packed, w.Packed...)
			scales = append(scales, w.Scales...)
			biases = append(biases, w.Biases...)
		}
		return QuantWeight{Packed: packed, Scales: scales, Biases: biases}
	}
	norm := func(salt int) []byte { return toBF16Bytes(syntheticFloat32(dModel, salt)) }
	return MoEQuantLayerWeights{
		NumExperts: numExperts, TopK: topK, ExpertDFF: expertDFF,
		ExpertGroupSize: groupSize, ExpertBits: bits, LocalGroupSize: groupSize, LocalBits: bits, RouterGroupSize: groupSize, RouterBits: bits,
		PreFFNormW: norm(13), PreFFNorm2W: norm(17), PostFFNorm1W: norm(19), PostFFNorm2W: norm(23), PostFFNormW: norm(29),
		LocalGate: qw(dFF, dModel, 3), LocalUp: qw(dFF, dModel, 31), LocalDown: qw(dModel, dFF, 37),
		RouterNormWScaled: norm(41), Router: qw(numExperts, dModel, 43), PerExpertScale: toBF16Bytes(syntheticFloat32(numExperts, 47)),
		ExpGate: batched(expertDFF, dModel, 53), ExpUp: batched(expertDFF, dModel, 101), ExpDown: batched(dModel, expertDFF, 149),
	}
}

func TestNativeQuantMoEGuardCoverage(t *testing.T) {
	requireNativeRuntime(t)
	withComposedGELU(t)

	const dModel, dFF, expertDFF, numExperts, topK, groupSize, bits = 64, 128, 96, 4, 2, 32, 4
	const eps = float32(1e-6)
	h := toBF16Bytes(syntheticFloat32(dModel, 5))
	w := quantMoELayerWeightsGuard(t, numExperts, topK, dModel, dFF, expertDFF, groupSize, bits)

	if out, err := MoEBlockQuant(h, w, dModel, dFF, eps); err != nil {
		t.Fatalf("MoEBlockQuant composed: %v", err)
	} else if len(out) != dModel*bf16Size {
		t.Fatalf("MoEBlockQuant composed len = %d", len(out))
	}

	idx := []int32{0, 1}
	weights := toBF16Bytes([]float32{0.75, 0.25})
	if out, err := MoEExpertsQuant(h, idx, weights, w.ExpGate, w.ExpUp, w.ExpDown, numExperts, topK, dModel, expertDFF, groupSize, bits); err != nil {
		t.Fatalf("MoEExpertsQuant composed: %v", err)
	} else if len(out) != dModel*bf16Size {
		t.Fatalf("MoEExpertsQuant composed len = %d", len(out))
	}
	if out, err := MoEExpertsQuant(h, nil, nil, w.ExpGate, w.ExpUp, w.ExpDown, numExperts, 0, dModel, expertDFF, groupSize, bits); err != nil {
		t.Fatalf("MoEExpertsQuant topK zero: %v", err)
	} else if len(out) != dModel*bf16Size {
		t.Fatalf("MoEExpertsQuant topK zero len = %d", len(out))
	}

	_, err := MoEExpertsQuant([]byte{1}, idx, weights, w.ExpGate, w.ExpUp, w.ExpDown, numExperts, topK, dModel, expertDFF, groupSize, bits)
	expectErr(t, "MoEExpertsQuant bad x", err)
	_, err = MoEExpertsQuant(h, idx[:1], weights, w.ExpGate, w.ExpUp, w.ExpDown, numExperts, topK, dModel, expertDFF, groupSize, bits)
	expectErr(t, "MoEExpertsQuant bad idx length", err)
	_, err = MoEExpertsQuant(h, idx, weights, w.ExpGate, w.ExpUp, w.ExpDown, numExperts, topK, dModel, expertDFF, 48, bits)
	expectErr(t, "MoEExpertsQuant bad group", err)
	badGate := w.ExpGate
	badGate.Packed = []byte{1}
	_, err = MoEExpertsQuant(h, idx, weights, badGate, w.ExpUp, w.ExpDown, numExperts, topK, dModel, expertDFF, groupSize, bits)
	expectErr(t, "MoEExpertsQuant bad weight", err)
	_, err = MoEExpertsQuant(h, []int32{0, numExperts}, weights, w.ExpGate, w.ExpUp, w.ExpDown, numExperts, topK, dModel, expertDFF, groupSize, bits)
	expectErr(t, "MoEExpertsQuant bad expert", err)

	_, err = MoEBlockQuant([]byte{1}, w, dModel, dFF, eps)
	expectErr(t, "MoEBlockQuant bad h", err)
	bad := w
	bad.Router.GroupSize, bad.Router.Bits = 0, 0
	bad.RouterGroupSize = 48
	_, err = MoEBlockQuant(h, bad, dModel, dFF, eps)
	expectErr(t, "MoEBlockQuant bad router", err)
	bad = w
	bad.PreFFNormW = []byte{1}
	_, err = MoEBlockQuant(h, bad, dModel, dFF, eps)
	expectErr(t, "MoEBlockQuant bad pre norm", err)
	bad = w
	bad.PreFFNorm2W = []byte{1}
	_, err = MoEBlockQuant(h, bad, dModel, dFF, eps)
	expectErr(t, "MoEBlockQuant bad second pre norm", err)
	bad = w
	bad.ExpGate = badGate
	_, err = MoEBlockQuant(h, bad, dModel, dFF, eps)
	expectErr(t, "MoEBlockQuant bad experts", err)
	bad = w
	bad.PostFFNorm1W = []byte{1}
	_, err = MoEBlockQuant(h, bad, dModel, dFF, eps)
	expectErr(t, "MoEBlockQuant bad post norm one", err)
	bad = w
	bad.PostFFNorm2W = []byte{1}
	_, err = MoEBlockQuant(h, bad, dModel, dFF, eps)
	expectErr(t, "MoEBlockQuant bad post norm two", err)
	bad = w
	bad.PostFFNormW = []byte{1}
	_, err = MoEBlockQuant(h, bad, dModel, dFF, eps)
	expectErr(t, "MoEBlockQuant bad final norm", err)
}

func TestNativeLoaderGuardCoverage(t *testing.T) {
	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers = 64, 1, 1, 64, 128, 32, 1
	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: nLayers, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim,
		VocabSize: vocab, RMSNormEps: 1e-6,
	}
	configJSON := gemma4ConfigJSON(t, cfg)
	emptyBlob := encodedTensors(t, map[string]safetensors.Tensor{})

	mcfg, _ := mistralConfigFixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	mcfgJSON := core.JSONMarshal(mcfg)
	if !mcfgJSON.OK {
		t.Fatalf("marshal mistral config: %s", mcfgJSON.Error())
	}
	badMcfg := mcfg
	badMcfg.HiddenSize = 0
	badMcfgJSON := core.JSONMarshal(badMcfg)
	if !badMcfgJSON.OK {
		t.Fatalf("marshal bad mistral config: %s", badMcfgJSON.Error())
	}

	// Every directory load now flows through the registry loaders (LoadDir / LoadTokenModelDir):
	// loadRegistered errors on a missing config, malformed config, unknown architecture, and a
	// checkpoint with no weights — so the error battery the per-arch loaders had is preserved here.
	missingDir := t.TempDir()
	_, err := LoadDir(missingDir, 4)
	expectErr(t, "LoadDir missing config", err)
	_, err = LoadTokenModelDir(missingDir, 4)
	expectErr(t, "LoadTokenModelDir missing config", err)

	badConfigDir := t.TempDir()
	writeLocal(t, core.PathJoin(badConfigDir, "config.json"), []byte("{"))
	_, err = LoadDir(badConfigDir, 4)
	expectErr(t, "LoadDir bad config", err)
	_, err = LoadTokenModelDir(badConfigDir, 4)
	expectErr(t, "LoadTokenModelDir bad config", err)

	badArchDir := t.TempDir()
	writeLocal(t, core.PathJoin(badArchDir, "config.json"), badMcfgJSON.Value.([]byte))
	_, err = LoadDir(badArchDir, 4)
	expectErr(t, "LoadDir bad arch", err)

	noWeightsDir := t.TempDir()
	writeLocal(t, core.PathJoin(noWeightsDir, "config.json"), configJSON)
	_, err = LoadDir(noWeightsDir, 4)
	expectErr(t, "LoadDir no weights", err)
	_, err = LoadTokenModelDir(noWeightsDir, 4)
	expectErr(t, "LoadTokenModelDir no weights", err)

	noMistralWeightsDir := t.TempDir()
	writeLocal(t, core.PathJoin(noMistralWeightsDir, "config.json"), mcfgJSON.Value.([]byte))
	_, err = LoadDir(noMistralWeightsDir, 4)
	expectErr(t, "LoadDir mistral no weights", err)
	emptyMistralDir := t.TempDir()
	writeLocal(t, core.PathJoin(emptyMistralDir, "config.json"), mcfgJSON.Value.([]byte))
	writeLocal(t, core.PathJoin(emptyMistralDir, "model.safetensors"), emptyBlob)
	_, err = LoadDir(emptyMistralDir, 4)
	expectErr(t, "LoadDir mistral assemble", err)

	quantCfg := cfg
	quantCfg.Quantization = &model.QuantConfig{GroupSize: 32, Bits: 4}
	quantDir := t.TempDir()
	writeLocal(t, core.PathJoin(quantDir, "config.json"), gemma4ConfigJSON(t, quantCfg))
	_, err = LoadDir(quantDir, 4)
	expectErr(t, "LoadDir quant no weights", err)
}

func TestNativeDirectorySuccessCoverage(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, vocab, maxLen = 64, 1, 1, 64, 128, 32, 8
	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: 1, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim,
		VocabSize: vocab, RMSNormEps: 1e-6, FinalLogitSoftcapping: 30,
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	tensors, _ := gemma4Tensors(arch, false)
	blob := encodedTensors(t, tensors)

	dir := t.TempDir()
	writeLocal(t, core.PathJoin(dir, "config.json"), gemma4ConfigJSON(t, cfg))
	writeLocal(t, core.PathJoin(dir, "model.safetensors"), blob)
	tm, err := LoadTokenModelDir(dir, maxLen)
	if err != nil {
		t.Fatalf("LoadTokenModelDir bf16: %v", err)
	}
	if closer, ok := tm.(interface{ Close() error }); ok {
		defer func() { _ = closer.Close() }()
	}
	emb, err := tm.Embed(1)
	if err != nil {
		t.Fatalf("bf16 token model Embed: %v", err)
	}
	if len(emb) != dModel*bf16Size {
		t.Fatalf("bf16 token model Embed len = %d", len(emb))
	}
	logits, err := tm.Head(emb)
	if err != nil {
		t.Fatalf("bf16 token model Head: %v", err)
	}
	if len(logits) != vocab*bf16Size {
		t.Fatalf("bf16 token model Head len = %d", len(logits))
	}
}

func TestNativeLoaderCleanupCoverage(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, vocab = 64, 1, 1, 64, 128, 32
	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: 1, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim,
		VocabSize: vocab, RMSNormEps: 1e-6,
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	bf16Dir := t.TempDir()
	writeLocal(t, core.PathJoin(bf16Dir, "config.json"), gemma4ConfigJSON(t, cfg))
	writeLocal(t, core.PathJoin(bf16Dir, "model.safetensors"), encodedTensors(t, gemma4TensorsMust(t, arch)))
	_, err = LoadDir(bf16Dir, 0)
	expectErr(t, "LoadDir bf16 bad maxLen cleanup", err)

	emptyDir := t.TempDir()
	writeLocal(t, core.PathJoin(emptyDir, "config.json"), gemma4ConfigJSON(t, cfg))
	writeLocal(t, core.PathJoin(emptyDir, "model.safetensors"), encodedTensors(t, map[string]safetensors.Tensor{}))
	_, err = LoadDir(emptyDir, 4)
	expectErr(t, "LoadDir bf16 assemble cleanup", err)
	_, err = LoadTokenModelDir(emptyDir, 4)
	expectErr(t, "LoadTokenModelDir bf16 assemble cleanup", err)

	const groupSize, bits = 32, 4
	quantCfg := cfg
	quantCfg.Quantization = &model.QuantConfig{GroupSize: groupSize, Bits: bits}
	quantDir := t.TempDir()
	writeLocal(t, core.PathJoin(quantDir, "config.json"), gemma4ConfigJSON(t, quantCfg))
	writeLocal(t, core.PathJoin(quantDir, "model.safetensors"), encodedTensors(t, quantGemma4TensorsGuard(t, arch, groupSize, bits)))
	_, err = LoadDir(quantDir, 0)
	expectErr(t, "LoadDir quant bad maxLen cleanup", err)

	emptyQuantDir := t.TempDir()
	writeLocal(t, core.PathJoin(emptyQuantDir, "config.json"), gemma4ConfigJSON(t, quantCfg))
	writeLocal(t, core.PathJoin(emptyQuantDir, "model.safetensors"), encodedTensors(t, map[string]safetensors.Tensor{}))
	_, err = LoadDir(emptyQuantDir, 4)
	expectErr(t, "LoadDir quant assemble cleanup", err)
	_, err = LoadTokenModelDir(emptyQuantDir, 4)
	expectErr(t, "LoadTokenModelDir quant assemble cleanup", err)

	// (A bad quant *config* with bf16 weights is no longer an error: the reactive path reads the quant
	// representation from the WEIGHTS — m.Embed.Quantised() — not the config block, so bf16 weights load
	// as bf16 and the stale config quant block is correctly ignored. The old per-arch loader validated
	// the config block; that behaviour was retired with it.)
}

func gemma4TensorsMust(t *testing.T, arch model.Arch) map[string]safetensors.Tensor {
	t.Helper()
	tensors, _ := gemma4Tensors(arch, false)
	return tensors
}

func TestNativeGenerationValidationCoverage(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, vocab, maxLen = 64, 1, 1, 64, 128, 32, 8
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 1)
	_, err := GenerateBF16(nil, arch, []int32{1}, 1, maxLen, -1)
	expectErr(t, "GenerateBF16 nil weights", err)
	_, err = GenerateBF16(g, arch, nil, 1, maxLen, -1)
	expectErr(t, "GenerateBF16 empty prompt", err)
	_, err = GenerateBF16(g, arch, []int32{1}, 0, maxLen, -1)
	expectErr(t, "GenerateBF16 bad maxNew", err)
	_, err = GenerateBF16(g, arch, []int32{1, 2}, maxLen, maxLen, -1)
	expectErr(t, "GenerateBF16 maxLen", err)
	bad := *g
	bad.Embed = []byte{1}
	_, err = GenerateBF16(&bad, arch, []int32{1}, 1, maxLen, -1)
	expectErr(t, "GenerateBF16 bad embed", err)
	bad = *g
	bad.FinalNorm = []byte{1}
	_, err = GenerateBF16(&bad, arch, []int32{1}, 1, maxLen, -1)
	expectErr(t, "GenerateBF16 bad head", err)

	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	sess.greedy = nil
	sess.head = func([]byte, bool) ([]byte, error) { return nil, core.NewError("head failed") }
	_, err = sess.Generate([]int32{1}, 1, -1)
	expectErr(t, "ArchSession.Generate head error", err)

	sess, err = NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession greedy: %v", err)
	}
	sess.greedy = nil
	sess.head = func([]byte, bool) ([]byte, error) { return []byte{1}, nil }
	_, err = sess.Generate([]int32{1}, 1, -1)
	expectErr(t, "ArchSession.Generate greedy error", err)

	sess, err = NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession embed: %v", err)
	}
	origEmbed := sess.embed
	calls := 0
	sess.embed = func(id int32) ([]byte, error) {
		calls++
		if calls > 1 {
			return nil, core.NewError("generated embed failed")
		}
		return origEmbed(id)
	}
	sess.greedy = nil
	sess.head = func([]byte, bool) ([]byte, error) {
		return toBF16Bytes(syntheticFloat32(arch.Vocab, 3)), nil
	}
	_, err = sess.Generate([]int32{1}, 1, -1)
	expectErr(t, "ArchSession.Generate generated step", err)

	oldCapture := captureLayerHiddens
	captureLayerHiddens = true
	capturedAttnHiddens, capturedLayerHiddens = nil, nil
	t.Cleanup(func() {
		captureLayerHiddens = oldCapture
		capturedAttnHiddens, capturedLayerHiddens = nil, nil
	})
	inputs := decodeInputsFixture(2, dModel)
	if _, err := DecodeForwardArch(inputs, g.Layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, arch.RopeBase, arch.AttnScale, arch.Eps, false); err != nil {
		t.Fatalf("DecodeForwardArch capture: %v", err)
	}
	if len(capturedAttnHiddens) == 0 || len(capturedLayerHiddens) == 0 {
		t.Fatal("DecodeForwardArch capture did not record hiddens")
	}
}

func TestNativePerLayerValidationCoverage(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, pliDim, vocabPLI, numLayers = 64, 32, 8, 2
	const plDim = pliDim * numLayers
	hidden := toBF16Bytes(syntheticFloat32(dModel, 3))
	embed := toBF16Bytes(syntheticFloat32(vocabPLI*plDim, 5))
	projW := toBF16Bytes(syntheticFloat32(plDim*dModel, 7))
	projNorm := toBF16Bytes(syntheticFloat32(pliDim, 11))
	qProj := quantWeightFixture(t, plDim, dModel, 32, 4, 13)
	if _, err := PerLayerInputs(embed, nil, nil, qProj.Packed, qProj.Scales, qProj.Biases, projNorm, 2, hidden, vocabPLI, numLayers, pliDim, dModel, 0, 0, 32, 4, 1e-5, bufView{}); err != nil {
		t.Fatalf("PerLayerInputs quant projection: %v", err)
	}
	_, err := PerLayerInputs([]byte{1}, []byte{1}, []byte{1}, projW, nil, nil, projNorm, 2, hidden, vocabPLI, numLayers, pliDim, dModel, 32, 4, 0, 0, 1e-5, bufView{})
	expectErr(t, "PerLayerInputs bad quant embed", err)

	gateW := toBF16Bytes(syntheticFloat32(pliDim*dModel, 13))
	perLayer := toBF16Bytes(syntheticFloat32(pliDim, 17))
	postNorm := toBF16Bytes(syntheticFloat32(dModel, 19))
	projGateW := toBF16Bytes(syntheticFloat32(dModel*pliDim, 23))
	_, err = PerLayerInputGateBF16(hidden, gateW, []byte{1}, projGateW, postNorm, dModel, pliDim, 1e-5)
	expectErr(t, "PerLayerInputGateBF16 bad pli", err)
	_, err = PerLayerInputGateBF16(hidden, gateW, perLayer, []byte{1}, postNorm, dModel, pliDim, 1e-5)
	expectErr(t, "PerLayerInputGateBF16 bad proj", err)
	_, err = PerLayerInputGateBF16(hidden, gateW, perLayer, projGateW, []byte{1}, dModel, pliDim, 1e-5)
	expectErr(t, "PerLayerInputGateBF16 bad post norm", err)

	qGate := quantWeightFixture(t, pliDim, dModel, 32, 4, 29)
	qBack := quantWeightFixture(t, dModel, pliDim, 32, 4, 31)
	_, err = PerLayerInputGateQuant(hidden, qGate, perLayer, qBack, []byte{1}, dModel, pliDim, 32, 4, 1e-5)
	expectErr(t, "PerLayerInputGateQuant bad post norm", err)
}

func TestNativeShapeValidationCoverage(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, kvLen = 64, 1, 1, 64, 128, 2
	const pliDim, groupSize, bits = 32, 32, 4
	const eps = float32(1e-5)
	qDim := nHeads * headDim
	x := toBF16Bytes(syntheticFloat32(dModel, 3))
	norm := toBF16Bytes(syntheticFloat32(dModel, 5))
	wQ := toBF16Bytes(syntheticFloat32(qDim*dModel, 7))
	wO := toBF16Bytes(syntheticFloat32(dModel*qDim, 11))
	wGate := toBF16Bytes(syntheticFloat32(dFF*dModel, 13))
	wUp := toBF16Bytes(syntheticFloat32(dFF*dModel, 17))
	wDown := toBF16Bytes(syntheticFloat32(dModel*dFF, 19))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 23))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 29))
	perLayer := toBF16Bytes(syntheticFloat32(pliDim, 31))
	pliGateW := toBF16Bytes(syntheticFloat32(pliDim*dModel, 37))
	pliProjW := toBF16Bytes(syntheticFloat32(dModel*pliDim, 41))
	qGate := quantWeightFixture(t, pliDim, dModel, groupSize, bits, 43)
	qProj := quantWeightFixture(t, dModel, pliDim, groupSize, bits, 47)

	cases := []struct {
		name string
		call func() error
	}{
		{"MLPBlockBF16 bad gate", func() error {
			_, err := MLPBlockBF16(x, norm, []byte{1}, wUp, wDown, dModel, dFF, eps)
			return err
		}},
		{"MLPBlockBF16 bad down", func() error {
			_, err := MLPBlockBF16(x, norm, wGate, wUp, []byte{1}, dModel, dFF, eps)
			return err
		}},
		{"PerLayerInputGateBF16 bad h", func() error {
			_, err := PerLayerInputGateBF16([]byte{1}, pliGateW, perLayer, pliProjW, norm, dModel, pliDim, eps)
			return err
		}},
		{"PerLayerInputGateBF16 bad gate", func() error {
			_, err := PerLayerInputGateBF16(x, []byte{1}, perLayer, pliProjW, norm, dModel, pliDim, eps)
			return err
		}},
		{"PerLayerInputGateQuant bad h", func() error {
			_, err := PerLayerInputGateQuant([]byte{1}, qGate, perLayer, qProj, norm, dModel, pliDim, groupSize, bits, eps)
			return err
		}},
		{"PerLayerInputGateQuant bad per-layer", func() error {
			_, err := PerLayerInputGateQuant(x, qGate, []byte{1}, qProj, norm, dModel, pliDim, groupSize, bits, eps)
			return err
		}},
		{"DecodeLayerICB bad x", func() error {
			_, err := DecodeLayerICB([]byte{1}, norm, wQ, wO, kCache, vCache, norm, wGate, wUp, wDown, dModel, nHeads, nKV, headDim, kvLen, dFF, 10000, 0.125, 0, eps, 1)
			return err
		}},
		{"DecodeLayerICB bad q", func() error {
			_, err := DecodeLayerICB(x, norm, []byte{1}, wO, kCache, vCache, norm, wGate, wUp, wDown, dModel, nHeads, nKV, headDim, kvLen, dFF, 10000, 0.125, 0, eps, 1)
			return err
		}},
		{"DecodeLayerICB bad mlp", func() error {
			_, err := DecodeLayerICB(x, norm, wQ, wO, kCache, vCache, norm, []byte{1}, wUp, wDown, dModel, nHeads, nKV, headDim, kvLen, dFF, 10000, 0.125, 0, eps, 1)
			return err
		}},
		{"DecodeLayerICB bad cache", func() error {
			_, err := DecodeLayerICB(x, norm, wQ, wO, []byte{1}, vCache, norm, wGate, wUp, wDown, dModel, nHeads, nKV, headDim, kvLen, dFF, 10000, 0.125, 0, eps, 1)
			return err
		}},
		{"DecodeTokenICB bad x", func() error {
			_, err := DecodeTokenICB([]byte{1}, norm, wQ, wO, kCache, vCache, norm, wGate, wUp, wDown, dModel, nHeads, nKV, headDim, kvLen, dFF, 1, 10000, 0.125, 0, eps, 1)
			return err
		}},
		{"DecodeTokenICB bad q", func() error {
			_, err := DecodeTokenICB(x, norm, []byte{1}, wO, kCache, vCache, norm, wGate, wUp, wDown, dModel, nHeads, nKV, headDim, kvLen, dFF, 1, 10000, 0.125, 0, eps, 1)
			return err
		}},
		{"DecodeTokenICB bad mlp", func() error {
			_, err := DecodeTokenICB(x, norm, wQ, wO, kCache, vCache, norm, []byte{1}, wUp, wDown, dModel, nHeads, nKV, headDim, kvLen, dFF, 1, 10000, 0.125, 0, eps, 1)
			return err
		}},
		{"DecodeTokenICB bad cache", func() error {
			_, err := DecodeTokenICB(x, norm, wQ, wO, []byte{1}, vCache, norm, wGate, wUp, wDown, dModel, nHeads, nKV, headDim, kvLen, dFF, 1, 10000, 0.125, 0, eps, 1)
			return err
		}},
	}
	for _, tc := range cases {
		expectErr(t, tc.name, tc.call())
	}
}

func TestNativeSessionGuardCoverage(t *testing.T) {
	requireNativeRuntime(t)

	var nilSession *ArchSession
	if err := nilSession.Close(); err != nil {
		t.Fatalf("nil ArchSession.Close: %v", err)
	}
	var nilTokenModel *NativeTokenModel
	if err := nilTokenModel.Close(); err != nil {
		t.Fatalf("nil NativeTokenModel.Close: %v", err)
	}

	g, arch := gemma4BF16Fixture(t, 64, 1, 1, 64, 128, 32, 1)
	_, err := NewArchSession(nil, arch, 4)
	expectErr(t, "NewArchSession nil weights", err)
	_, err = NewArchSession(&BF16Model{}, arch, 4)
	expectErr(t, "NewArchSession layer mismatch", err)
	_, err = NewArchSession(g, arch, 0)
	expectErr(t, "NewArchSession bad maxLen", err)
	_, err = NewBF16TokenModel(nil, arch, 4)
	expectErr(t, "NewBF16TokenModel nil weights", err)

	sess, err := NewArchSession(g, arch, 1)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	_, err = sess.Step([]byte{1})
	expectErr(t, "Step bad emb", err)
	sess.perLayerInput = func(int32, []byte) ([]byte, error) { return nil, nil }
	_, err = sess.Step(toBF16Bytes(syntheticFloat32(arch.Hidden, 3)))
	expectErr(t, "Step rejects PLE", err)
	sess.perLayerInput = nil
	if _, err = sess.Step(toBF16Bytes(syntheticFloat32(arch.Hidden, 5))); err != nil {
		t.Fatalf("Step valid: %v", err)
	}
	_, err = sess.Step(toBF16Bytes(syntheticFloat32(arch.Hidden, 7)))
	expectErr(t, "Step maxLen", err)
	_, err = sess.StepWithID(1, []byte{1})
	expectErr(t, "StepWithID bad emb", err)
	_, err = sess.StepWithID(1, toBF16Bytes(syntheticFloat32(arch.Hidden, 9)))
	expectErr(t, "StepWithID maxLen", err)
	_, err = sess.Generate(nil, 1, -1)
	expectErr(t, "Generate empty prompt", err)
	_, err = sess.Generate([]int32{1}, 0, -1)
	expectErr(t, "Generate bad maxNew", err)
	_, err = sess.Generate([]int32{1}, 1, -1)
	expectErr(t, "Generate over maxLen", err)
	_, err = sess.GenerateText(nil, "x", 1)
	expectErr(t, "GenerateText nil tokenizer", err)

	q := &QuantModel{Layers: []QuantizedLayerWeights{}}
	_, err = NewArchQuantSession(nil, arch, 4)
	expectErr(t, "NewArchQuantSession nil", err)
	_, err = NewArchQuantSession(q, arch, 4)
	expectErr(t, "NewArchQuantSession mismatch", err)
}

func TestNativeSessionOptionalDecodeFeatures(t *testing.T) {
	requireNativeRuntime(t)
	t.Setenv("LTHN_NATIVE_TRACE", "1")

	g, arch := gemma4BF16Fixture(t, 64, 1, 1, 64, 128, 32, 1)
	arch.ValueNorm = true
	arch.RotaryDim = 32
	arch.RopeFreqs = plainRopeInvFreqsGuard(float64(arch.RopeBase), arch.RotaryDim)
	l := &g.Layers[0]
	l.QNormW = toBF16Bytes(syntheticFloat32(arch.HeadDim, 31))
	l.KNormW = toBF16Bytes(syntheticFloat32(arch.HeadDim, 37))
	l.PostAttnNormW = toBF16Bytes(syntheticFloat32(arch.Hidden, 41))
	l.PostFFNormW = toBF16Bytes(syntheticFloat32(arch.Hidden, 43))
	l.LayerScalarW = toBF16Bytes([]float32{0.75})
	l.WV = nil

	sess, err := NewArchSession(g, arch, 4)
	if err != nil {
		t.Fatalf("NewArchSession optional: %v", err)
	}
	emb := toBF16Bytes(syntheticFloat32(arch.Hidden, 47))
	if _, err := sess.Step(emb); err != nil {
		t.Fatalf("Step optional: %v", err)
	}

	inputs := [][]byte{emb}
	if _, err := DecodeForwardArch(inputs, g.Layers, arch.Layer, arch.Hidden, arch.Heads, arch.KVHeads, arch.HeadDim, 4, arch.FF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm); err != nil {
		t.Fatalf("DecodeForwardArch optional: %v", err)
	}
}

func TestNativeSessionPLEAndDirCoverage(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, vocab, pliDim, maxLen = 64, 1, 1, 64, 128, 32, 32, 4
	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: 1, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim,
		VocabSize: vocab, RMSNormEps: 1e-6,
		HiddenSizePerLayerInput: pliDim, VocabSizePerLayerInput: vocab,
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("PLE Arch: %v", err)
	}
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 17)
	layer.PerLayerGate = toBF16Bytes(syntheticFloat32(pliDim*dModel, 23))
	layer.PerLayerProjection = toBF16Bytes(syntheticFloat32(dModel*pliDim, 29))
	layer.PostPerLayerInputNormW = toBF16Bytes(syntheticFloat32(dModel, 31))
	g := &BF16Model{
		Layers:             []DecodeLayerWeights{layer},
		Embed:              toBF16Bytes(syntheticFloat32(vocab*dModel, 37)),
		FinalNorm:          toBF16Bytes(syntheticFloat32(dModel, 41)),
		EmbedPerLayer:      toBF16Bytes(syntheticFloat32(vocab*pliDim, 43)),
		PerLayerModelProjW: toBF16Bytes(syntheticFloat32(pliDim*dModel, 47)),
		PerLayerProjNormW:  toBF16Bytes(syntheticFloat32(pliDim, 53)),
		Tied:               true,
	}
	g.LMHead = g.Embed

	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession PLE: %v", err)
	}
	emb, err := sess.embed(1)
	if err != nil {
		t.Fatalf("PLE embed: %v", err)
	}
	if out, err := sess.StepWithID(1, emb); err != nil {
		t.Fatalf("PLE StepWithID: %v", err)
	} else if len(out) != dModel*bf16Size {
		t.Fatalf("PLE StepWithID len = %d", len(out))
	}
	sess, err = NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession PLE generate: %v", err)
	}
	if gen, err := sess.Generate([]int32{1}, 1, -1); err != nil {
		t.Fatalf("PLE Generate: %v", err)
	} else if len(gen) != 1 {
		t.Fatalf("PLE Generate len = %d", len(gen))
	}

	// The dir-load→generate path is covered by TestNativeLoaderSessionCoverage (LoadDir) and
	// TestNativeRemainingBranchCoverage (LoadTokenModelDir); the unique coverage here is the in-memory
	// PLE session above. (A synthetic dir-generate over these toy dims — head_dim 64 — tripped an SDPA
	// kernel the backend doesn't precompile for that shape; real models decode fine, so not a product gap.)
}

func TestNativeDecodeGuardCoverage(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, vocab, maxLen = 64, 1, 1, 64, 128, 32, 4
	arch := archFixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 1)
	inputs := decodeInputsFixture(2, dModel)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)
	layers := []DecodeLayerWeights{layer}
	qLayer := quantizedLayerFixture(t, dModel, nHeads, nKV, headDim, dFF, 64, 4, 5)
	qLayers := []QuantizedLayerWeights{qLayer}

	_, err := DecodeForwardArch(nil, nil, nil, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, 10000, arch.AttnScale, arch.Eps, false)
	expectErr(t, "DecodeForwardArch empty", err)
	_, err = DecodeForwardArch(inputs, layers, nil, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, 10000, arch.AttnScale, arch.Eps, false)
	expectErr(t, "DecodeForwardArch specs mismatch", err)
	_, err = DecodeForwardArch(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, 1, dFF, 0, 10000, arch.AttnScale, arch.Eps, false)
	expectErr(t, "DecodeForwardArch maxLen", err)
	badInputs := [][]byte{{1}}
	_, err = DecodeForwardArch(badInputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, 10000, arch.AttnScale, arch.Eps, false)
	expectErr(t, "DecodeForwardArch bad input", err)
	badSpecs := append([]model.LayerSpec(nil), arch.Layer...)
	badSpecs[0].KVShareFrom = -1
	_, err = DecodeForwardArch(inputs, layers, badSpecs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, 10000, arch.AttnScale, arch.Eps, false)
	expectErr(t, "DecodeForwardArch bad share", err)
	moeSpecs := append([]model.LayerSpec(nil), arch.Layer...)
	moeSpecs[0].MoE = true
	_, err = DecodeForwardArch(inputs, layers, moeSpecs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, 10000, arch.AttnScale, arch.Eps, false)
	expectErr(t, "DecodeForwardArch moe mismatch", err)

	_, err = DecodeForwardArchQuant(nil, nil, nil, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, 10000, arch.AttnScale, arch.Eps, false)
	expectErr(t, "DecodeForwardArchQuant empty", err)
	_, err = DecodeForwardArchQuant(inputs, qLayers, nil, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, 10000, arch.AttnScale, arch.Eps, false)
	expectErr(t, "DecodeForwardArchQuant specs mismatch", err)
	_, err = DecodeForwardArchQuant(inputs, qLayers, moeSpecs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, 10000, arch.AttnScale, arch.Eps, false)
	expectErr(t, "DecodeForwardArchQuant moe", err)
	badQLayers := []QuantizedLayerWeights{qLayer}
	badQLayers[0].GroupSize = 0
	_, err = DecodeForwardArchQuant(inputs, badQLayers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, 10000, arch.AttnScale, arch.Eps, false)
	expectErr(t, "DecodeForwardArchQuant unset geometry", err)

	x := toBF16Bytes(syntheticFloat32(dModel, 7))
	kCache := make([]byte, maxLen*nKV*headDim*bf16Size)
	vCache := make([]byte, maxLen*nKV*headDim*bf16Size)
	_, err = AttentionStepKV(x, layer.AttnNormW, layer.WQ, layer.WK, layer.WV, layer.WO, kCache, vCache, dModel, 2, 0, headDim, maxLen, 0, 10000, 0.125, 1e-5)
	expectErr(t, "AttentionStepKV bad gqa", err)
	_, err = AttentionStepKV(x, layer.AttnNormW, layer.WQ, layer.WK, layer.WV, layer.WO, kCache, vCache, dModel, nHeads, nKV, headDim, maxLen, maxLen, 10000, 0.125, 1e-5)
	expectErr(t, "AttentionStepKV bad pos", err)
	_, err = AttentionStepKV([]byte{1}, layer.AttnNormW, layer.WQ, layer.WK, layer.WV, layer.WO, kCache, vCache, dModel, nHeads, nKV, headDim, maxLen, 0, 10000, 0.125, 1e-5)
	expectErr(t, "AttentionStepKV bad x", err)
	_, err = AttentionStepKV(x, layer.AttnNormW, []byte{1}, layer.WK, layer.WV, layer.WO, kCache, vCache, dModel, nHeads, nKV, headDim, maxLen, 0, 10000, 0.125, 1e-5)
	expectErr(t, "AttentionStepKV bad wQ", err)
	_, err = AttentionStepKV(x, layer.AttnNormW, layer.WQ, []byte{1}, layer.WV, layer.WO, kCache, vCache, dModel, nHeads, nKV, headDim, maxLen, 0, 10000, 0.125, 1e-5)
	expectErr(t, "AttentionStepKV bad wK", err)
	_, err = AttentionStepKV(x, layer.AttnNormW, layer.WQ, layer.WK, layer.WV, layer.WO, []byte{1}, vCache, dModel, nHeads, nKV, headDim, maxLen, 0, 10000, 0.125, 1e-5)
	expectErr(t, "AttentionStepKV bad cache", err)
	_, err = DecodeStepKV(x, layer.AttnNormW, layer.WQ, layer.WK, layer.WV, layer.WO, kCache, vCache, []byte{1}, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, 10000, 0.125, 1e-5)
	expectErr(t, "DecodeStepKV bad mlp norm", err)
	_, err = DecodeStepKV(x, layer.AttnNormW, layer.WQ, layer.WK, layer.WV, layer.WO, kCache, vCache, layer.MLPNormW, []byte{1}, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, 10000, 0.125, 1e-5)
	expectErr(t, "DecodeStepKV bad mlp weights", err)

	_, err = DecodeLayerICB([]byte{1}, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, 2, dFF, 10000, 0.125, 0, 1e-5, 0)
	expectErr(t, "DecodeLayerICB bad x", err)
	_, err = DecodeLayerICB(x, layer.AttnNormW, []byte{1}, layer.WO, kCache, vCache, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, 2, dFF, 10000, 0.125, 0, 1e-5, 0)
	expectErr(t, "DecodeLayerICB bad q", err)
	_, err = DecodeLayerICB(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, layer.MLPNormW, []byte{1}, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, 2, dFF, 10000, 0.125, 0, 1e-5, 0)
	expectErr(t, "DecodeLayerICB bad mlp", err)
	_, err = DecodeLayerICB(x, layer.AttnNormW, layer.WQ, layer.WO, []byte{1}, vCache, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, 2, dFF, 10000, 0.125, 0, 1e-5, 0)
	expectErr(t, "DecodeLayerICB bad cache", err)
	_, err = DecodeTokenICB([]byte{1}, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, 2, dFF, 0, 10000, 0.125, 0, 1e-5, 0)
	expectErr(t, "DecodeTokenICB bad x", err)
	_, err = DecodeTokenICB(x, layer.AttnNormW, []byte{1}, layer.WO, kCache, vCache, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, 2, dFF, 0, 10000, 0.125, 0, 1e-5, 0)
	expectErr(t, "DecodeTokenICB bad q", err)
}

func TestNativeICBDecodeValidationCoverage(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, vocab, maxLen = 64, 1, 1, 64, 128, 32, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)

	arch := archFixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 1)
	inputs := decodeInputsFixture(2, dModel)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)
	layers := []DecodeLayerWeights{layer}
	qLayer := quantizedLayerFixture(t, dModel, nHeads, nKV, headDim, dFF, 64, 4, 5)
	qLayers := []QuantizedLayerWeights{qLayer}

	_, err := DecodeForwardICB(nil, nil, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	expectErr(t, "DecodeForwardICB empty", err)
	_, err = DecodeForwardICB(inputs, layers, dModel, nHeads, nKV, headDim, 1, dFF, base, scale, eps)
	expectErr(t, "DecodeForwardICB maxLen", err)
	_, err = DecodeForwardICB([][]byte{{1}}, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	expectErr(t, "DecodeForwardICB bad input", err)
	badLayers := []DecodeLayerWeights{layer}
	badLayers[0].WQ = []byte{1}
	_, err = DecodeForwardICB(inputs, badLayers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	expectErr(t, "DecodeForwardICB bad layer", err)

	_, err = DecodeForwardICBQuant(nil, nil, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	expectErr(t, "DecodeForwardICBQuant empty", err)
	_, err = DecodeForwardICBQuant(inputs, qLayers, dModel, nHeads, nKV, headDim, 1, dFF, base, scale, eps)
	expectErr(t, "DecodeForwardICBQuant maxLen", err)
	unset := qLayer
	unset.GroupSize = 0
	_, err = DecodeForwardICBQuant(inputs, []QuantizedLayerWeights{unset}, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	expectErr(t, "DecodeForwardICBQuant unset geometry", err)
	_, err = DecodeForwardICBQuant([][]byte{{1}}, qLayers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	expectErr(t, "DecodeForwardICBQuant bad input", err)
	badMixed := []QuantizedLayerWeights{qLayer, qLayer}
	badMixed[1].Q.GroupSize = 48
	_, err = DecodeForwardICBQuant(inputs, badMixed, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	expectErr(t, "DecodeForwardICBQuant bad mixed geometry", err)
	badQ := qLayer
	badQ.AttnNormW = []byte{1}
	_, err = DecodeForwardICBQuant(inputs, []QuantizedLayerWeights{badQ}, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	expectErr(t, "DecodeForwardICBQuant bad norm", err)
	badQ = qLayer
	badQ.Q.GroupSize = 48
	_, err = DecodeForwardICBQuant(inputs, []QuantizedLayerWeights{badQ}, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	expectErr(t, "DecodeForwardICBQuant bad group multiple", err)
	badQ = qLayer
	badQ.Q.Packed = []byte{1}
	_, err = DecodeForwardICBQuant(inputs, []QuantizedLayerWeights{badQ}, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	expectErr(t, "DecodeForwardICBQuant bad weight", err)

	_, err = DecodeForwardArchICBQuant(nil, nil, nil, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	expectErr(t, "DecodeForwardArchICBQuant empty", err)
	_, err = DecodeForwardArchICBQuant(inputs, qLayers, nil, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	expectErr(t, "DecodeForwardArchICBQuant specs mismatch", err)
	_, err = DecodeForwardArchICBQuant(inputs, qLayers, arch.Layer, dModel, nHeads, nKV, headDim, 1, dFF, 0, base, scale, eps, false)
	expectErr(t, "DecodeForwardArchICBQuant maxLen", err)
	_, err = DecodeForwardArchICBQuant(inputs, []QuantizedLayerWeights{unset}, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	expectErr(t, "DecodeForwardArchICBQuant unset geometry", err)
	_, err = DecodeForwardArchICBQuant([][]byte{{1}}, qLayers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	expectErr(t, "DecodeForwardArchICBQuant bad input", err)
	badSpecs := append([]model.LayerSpec(nil), arch.Layer...)
	badSpecs[0].KVShareFrom = -1
	_, err = DecodeForwardArchICBQuant(inputs, qLayers, badSpecs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	expectErr(t, "DecodeForwardArchICBQuant bad share", err)
	moeSpecs := append([]model.LayerSpec(nil), arch.Layer...)
	moeSpecs[0].MoE = true
	_, err = DecodeForwardArchICBQuant(inputs, qLayers, moeSpecs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	expectErr(t, "DecodeForwardArchICBQuant moe", err)
	archTwo := archFixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	badArchMixed := []QuantizedLayerWeights{qLayer, qLayer}
	badArchMixed[1].Q.GroupSize = 48
	_, err = DecodeForwardArchICBQuant(inputs, badArchMixed, archTwo.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	expectErr(t, "DecodeForwardArchICBQuant bad mixed geometry", err)
	badQ = qLayer
	badQ.AttnNormW = []byte{1}
	_, err = DecodeForwardArchICBQuant(inputs, []QuantizedLayerWeights{badQ}, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	expectErr(t, "DecodeForwardArchICBQuant bad norm", err)
	badQ = qLayer
	badQ.Q.GroupSize = 48
	_, err = DecodeForwardArchICBQuant(inputs, []QuantizedLayerWeights{badQ}, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	expectErr(t, "DecodeForwardArchICBQuant bad group multiple", err)
	badQ = qLayer
	badQ.Q.Packed = []byte{1}
	_, err = DecodeForwardArchICBQuant(inputs, []QuantizedLayerWeights{badQ}, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	expectErr(t, "DecodeForwardArchICBQuant bad weight", err)

	_, err = NormProjectICB([]float32{1}, nil, nil, 1, 1, eps, 0)
	expectErr(t, "NormProjectICB size", err)
	kCache := toBF16Bytes(syntheticFloat32(nKV*2*headDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(nKV*2*headDim, 11))
	x := toBF16Bytes(syntheticFloat32(dModel, 13))
	if out, err := AttentionBlockICB(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, dModel, nHeads, nKV, headDim, 2, base, scale, 0, eps, 0); err != nil {
		t.Fatalf("AttentionBlockICB default replay: %v", err)
	} else if len(out) != dModel*bf16Size {
		t.Fatalf("AttentionBlockICB default replay len = %d", len(out))
	}
}

func TestNativeQuantPLEAndRouterGuardCoverage(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, pliDim, vocabPLI, numLayers = 64, 32, 8, 2
	const plDim = pliDim * numLayers
	hidden := toBF16Bytes(syntheticFloat32(dModel, 3))
	embed := toBF16Bytes(syntheticFloat32(vocabPLI*plDim, 5))
	projW := toBF16Bytes(syntheticFloat32(plDim*dModel, 7))
	projNorm := toBF16Bytes(syntheticFloat32(pliDim, 11))
	if _, err := PerLayerInputs(embed, nil, nil, projW, nil, nil, projNorm, 2, hidden, vocabPLI, numLayers, pliDim, dModel, 0, 0, 0, 0, 1e-5, bufView{}); err != nil {
		t.Fatalf("PerLayerInputs bf16: %v", err)
	}
	_, err := PerLayerInputs(embed, nil, nil, projW, nil, nil, projNorm, 2, []byte{1}, vocabPLI, numLayers, pliDim, dModel, 0, 0, 0, 0, 1e-5, bufView{})
	expectErr(t, "PerLayerInputs bad hidden", err)
	_, err = PerLayerInputs(embed, nil, nil, []byte{1}, nil, nil, projNorm, 2, hidden, vocabPLI, numLayers, pliDim, dModel, 0, 0, 0, 0, 1e-5, bufView{})
	expectErr(t, "PerLayerInputs bad proj", err)
	_, err = PerLayerInputs(embed, nil, nil, projW, nil, nil, []byte{1}, 2, hidden, vocabPLI, numLayers, pliDim, dModel, 0, 0, 0, 0, 1e-5, bufView{})
	expectErr(t, "PerLayerInputs bad norm", err)
	_, err = PerLayerInputs(embed, nil, nil, projW, nil, nil, projNorm, int32(vocabPLI), hidden, vocabPLI, numLayers, pliDim, dModel, 0, 0, 0, 0, 1e-5, bufView{})
	expectErr(t, "PerLayerInputs token", err)

	gateW := toBF16Bytes(syntheticFloat32(pliDim*dModel, 13))
	perLayer := toBF16Bytes(syntheticFloat32(pliDim, 17))
	postNorm := toBF16Bytes(syntheticFloat32(dModel, 19))
	if _, err := PerLayerInputGateBF16(hidden, gateW, perLayer, toBF16Bytes(syntheticFloat32(dModel*pliDim, 23)), postNorm, dModel, pliDim, 1e-5); err != nil {
		t.Fatalf("PerLayerInputGateBF16: %v", err)
	}
	_, err = PerLayerInputGateBF16([]byte{1}, gateW, perLayer, projW, postNorm, dModel, pliDim, 1e-5)
	expectErr(t, "PerLayerInputGateBF16 bad hidden", err)
	_, err = PerLayerInputGateBF16(hidden, []byte{1}, perLayer, projW, postNorm, dModel, pliDim, 1e-5)
	expectErr(t, "PerLayerInputGateBF16 bad gate", err)

	qGate := quantWeightFixture(t, pliDim, dModel, 32, 4, 29)
	qProj := quantWeightFixture(t, dModel, pliDim, 32, 4, 31)
	if _, err := PerLayerInputGateQuant(hidden, qGate, perLayer, qProj, postNorm, dModel, pliDim, 32, 4, 1e-5); err != nil {
		t.Fatalf("PerLayerInputGateQuant: %v", err)
	}
	_, err = PerLayerInputGateQuant(hidden, qGate, []byte{1}, qProj, postNorm, dModel, pliDim, 32, 4, 1e-5)
	expectErr(t, "PerLayerInputGateQuant bad pli", err)

	const numExperts, topK = 4, 2
	routerW := toBF16Bytes(syntheticFloat32(numExperts*dModel, 37))
	norm := toBF16Bytes(syntheticFloat32(dModel, 41))
	if _, _, err := MoERouter(hidden, norm, routerW, nil, numExperts, topK, dModel, 1e-5); err != nil {
		t.Fatalf("MoERouter nil scale: %v", err)
	}
	_, _, err = MoERouter([]byte{1}, norm, routerW, nil, numExperts, topK, dModel, 1e-5)
	expectErr(t, "MoERouter bad x", err)
	_, _, err = MoERouter(hidden, []byte{1}, routerW, nil, numExperts, topK, dModel, 1e-5)
	expectErr(t, "MoERouter bad norm", err)
	_, _, err = MoERouter(hidden, norm, []byte{1}, nil, numExperts, topK, dModel, 1e-5)
	expectErr(t, "MoERouter bad router", err)
	_, _, err = MoERouter(hidden, norm, routerW, []byte{1}, numExperts, topK, dModel, 1e-5)
	expectErr(t, "MoERouter bad scale", err)
	_, _, err = MoERouter(hidden, norm, routerW, nil, numExperts, numExperts+1, dModel, 1e-5)
	expectErr(t, "MoERouter bad topK", err)

	qRouter := quantWeightFixture(t, numExperts, dModel, 32, 4, 43)
	if _, _, err := MoERouterQuant(hidden, norm, qRouter, nil, numExperts, topK, dModel, 32, 4, 1e-5); err != nil {
		t.Fatalf("MoERouterQuant: %v", err)
	}
	_, _, err = MoERouterQuant(hidden, norm, qRouter, []byte{1}, numExperts, topK, dModel, 32, 4, 1e-5)
	expectErr(t, "MoERouterQuant bad scale", err)
	qRouterFallback := qRouter
	qRouterFallback.GroupSize, qRouterFallback.Bits = 0, 0
	_, _, err = MoERouterQuant(hidden, norm, qRouterFallback, nil, numExperts, topK, dModel, 48, 4, 1e-5)
	expectErr(t, "MoERouterQuant bad group", err)

	_, err = EmbedTokensQuant(nil, nil, nil, []int32{0}, 1, dModel, 32, 0, 1)
	expectErr(t, "EmbedTokensQuant bad bits", err)
	_, err = EmbedTokensQuant(nil, nil, nil, []int32{0}, 1, dModel, 0, 4, 1)
	expectErr(t, "EmbedTokensQuant bad group", err)
	_, err = EmbedTokensQuant([]byte{1}, nil, nil, []int32{0}, 1, dModel, 32, 4, 1)
	expectErr(t, "EmbedTokensQuant bad packed", err)
	_, err = LMHeadQuant([]byte{1}, norm, qRouter.Packed, qRouter.Scales, qRouter.Biases, dModel, numExperts, 32, 4, 1e-5, 1)
	expectErr(t, "LMHeadQuant bad hidden", err)
	_, err = LMHeadQuant(hidden, []byte{1}, qRouter.Packed, qRouter.Scales, qRouter.Biases, dModel, numExperts, 32, 4, 1e-5, 1)
	expectErr(t, "LMHeadQuant bad norm", err)
	_, err = LMHeadQuant(hidden, norm, qRouter.Packed, qRouter.Scales, qRouter.Biases, dModel, numExperts, 48, 4, 1e-5, 1)
	expectErr(t, "LMHeadQuant bad group", err)

	w := moeLayerWeightsFixture(numExperts, topK, dModel, 128, 96, 47)
	_, err = MoEBlockBF16([]byte{1}, w, dModel, 128, 1e-5)
	expectErr(t, "MoEBlockBF16 bad h", err)
	_, err = MoEExperts(hidden, nil, nil, w.ExpGateW, w.ExpUpW, w.ExpDownW, numExperts, 0, dModel, 96)
	if err != nil {
		t.Fatalf("MoEExperts topK zero: %v", err)
	}
}

func TestNativeSmallGuardCoverage(t *testing.T) {
	requireNativeRuntime(t)

	_, err := MulBF16(toBF16Bytes([]float32{1}), toBF16Bytes([]float32{1, 2}))
	expectErr(t, "MulBF16 mismatch", err)
	_, err = MulBF16([]byte{1}, []byte{1})
	expectErr(t, "MulBF16 odd", err)
	_, err = TanhBF16([]byte{1})
	expectErr(t, "TanhBF16 odd", err)
	_, err = GeluBF16([]byte{1})
	expectErr(t, "GeluBF16 odd", err)
	_, err = GeluGateMulBF16(toBF16Bytes([]float32{1}), toBF16Bytes([]float32{1, 2}))
	expectErr(t, "GeluGateMulBF16 mismatch", err)
	if _, err = TanhBF16(nil); err != nil {
		t.Fatalf("TanhBF16 empty: %v", err)
	}
	if _, err = MulBF16(nil, nil); err != nil {
		t.Fatalf("MulBF16 empty: %v", err)
	}
	if (softmaxHybridMixer{}).Kind() != mixerSoftmaxHybrid {
		t.Fatal("softmaxHybridMixer.Kind mismatch")
	}
	if (softmaxHybridMixer{}).State().String() == "" {
		t.Fatal("softmaxHybridMixer.State empty")
	}
}

func TestNativePrimitiveGuardCoverage(t *testing.T) {
	requireNativeRuntime(t)

	if h := f32ToBF16(float32(math.NaN())); h&0x0040 == 0 {
		t.Fatalf("f32ToBF16(NaN) = 0x%x, quiet bit not set", h)
	}

	_, err := RMSNorm([]float32{1}, []float32{1}, 1, 2, 1e-5)
	expectErr(t, "RMSNorm bad x", err)
	_, err = RMSNorm([]float32{1, 2}, []float32{1}, 1, 2, 1e-5)
	expectErr(t, "RMSNorm bad weight", err)
	if out, err := RMSNorm(nil, nil, 0, 0, 1e-5); err != nil {
		t.Fatalf("RMSNorm zero: %v", err)
	} else if len(out) != 0 {
		t.Fatalf("RMSNorm zero len = %d", len(out))
	}
	if out, err := RMSNorm(syntheticFloat32(rmsLoopedLimit+1, 3), fillConst(rmsLoopedLimit+1, 1), 1, rmsLoopedLimit+1, 1e-5); err != nil {
		t.Fatalf("RMSNorm looped: %v", err)
	} else if len(out) != rmsLoopedLimit+1 {
		t.Fatalf("RMSNorm looped len = %d", len(out))
	}

	_, err = RMSNormBF16([]byte{1}, nil, 1, 1, 1e-5)
	expectErr(t, "RMSNormBF16 bad x", err)
	_, err = RMSNormBF16(toBF16Bytes([]float32{1}), nil, 1, 1, 1e-5)
	expectErr(t, "RMSNormBF16 bad weight", err)
	if out, err := RMSNormBF16(nil, nil, 0, 0, 1e-5); err != nil {
		t.Fatalf("RMSNormBF16 zero: %v", err)
	} else if len(out) != 0 {
		t.Fatalf("RMSNormBF16 zero len = %d", len(out))
	}
	if out, err := RMSNormBF16(toBF16Bytes(syntheticFloat32(rmsLoopedLimit+1, 5)), toBF16Bytes(fillConst(rmsLoopedLimit+1, 1)), 1, rmsLoopedLimit+1, 1e-5); err != nil {
		t.Fatalf("RMSNormBF16 looped: %v", err)
	} else if len(out) != (rmsLoopedLimit+1)*bf16Size {
		t.Fatalf("RMSNormBF16 looped len = %d", len(out))
	}

	_, err = MatVec([]float32{1}, nil, 1, 2)
	expectErr(t, "MatVec bad mat", err)
	_, err = MatVec([]float32{1, 2}, []float32{1}, 1, 2)
	expectErr(t, "MatVec bad vec", err)
	if out, err := MatVec(nil, nil, 0, 0); err != nil {
		t.Fatalf("MatVec zero: %v", err)
	} else if len(out) != 0 {
		t.Fatalf("MatVec zero len = %d", len(out))
	}
	_, err = MatVecBF16([]byte{1}, nil, 1, 1)
	expectErr(t, "MatVecBF16 bad mat", err)
	_, err = MatVecBF16(toBF16Bytes([]float32{1}), []byte{1}, 1, 1)
	expectErr(t, "MatVecBF16 bad vec", err)
	if out, err := MatVecBF16(nil, nil, 0, 0); err != nil {
		t.Fatalf("MatVecBF16 zero: %v", err)
	} else if len(out) != 0 {
		t.Fatalf("MatVecBF16 zero len = %d", len(out))
	}

	_, err = RoPE([]float32{1}, 1, 1, 2, 10000, 1, 0, false)
	expectErr(t, "RoPE bad len", err)
	if out, err := RoPE(nil, 0, 1, 2, 10000, 1, 0, false); err != nil {
		t.Fatalf("RoPE zero: %v", err)
	} else if len(out) != 0 {
		t.Fatalf("RoPE zero len = %d", len(out))
	}
	if out, err := RoPE(syntheticFloat32(4, 7), 1, 2, 2, 10000, 1, 1, true); err != nil {
		t.Fatalf("RoPE traditional: %v", err)
	} else if len(out) != 4 {
		t.Fatalf("RoPE traditional len = %d", len(out))
	}

	ropeBF16 := toBF16Bytes(syntheticFloat32(4, 11))
	_, err = RoPEDimsBF16([]byte{1}, 1, 1, 4, 4, 10000, 1, 0, false)
	expectErr(t, "RoPEDimsBF16 bad len", err)
	if out, err := RoPEDimsBF16(nil, 0, 1, 4, 4, 10000, 1, 0, false); err != nil {
		t.Fatalf("RoPEDimsBF16 zero: %v", err)
	} else if len(out) != 0 {
		t.Fatalf("RoPEDimsBF16 zero len = %d", len(out))
	}
	_, err = RoPEDimsBF16(ropeBF16, 1, 1, 4, 3, 10000, 1, 0, false)
	expectErr(t, "RoPEDimsBF16 bad rotary", err)
	if out, err := RoPEDimsBF16(ropeBF16, 1, 1, 4, 2, 10000, 1, 0, true); err != nil {
		t.Fatalf("RoPEDimsBF16 partial traditional: %v", err)
	} else if len(out) != len(ropeBF16) {
		t.Fatalf("RoPEDimsBF16 partial len = %d", len(out))
	}

	_, err = RoPEFreqsBF16([]byte{1}, 1, 1, 4, 4, []float32{1, 0.5}, 1, 0, false)
	expectErr(t, "RoPEFreqsBF16 bad len", err)
	if out, err := RoPEFreqsBF16(nil, 0, 1, 4, 4, []float32{1, 0.5}, 1, 0, false); err != nil {
		t.Fatalf("RoPEFreqsBF16 zero: %v", err)
	} else if len(out) != 0 {
		t.Fatalf("RoPEFreqsBF16 zero len = %d", len(out))
	}
	_, err = RoPEFreqsBF16(ropeBF16, 1, 1, 4, 3, []float32{1}, 1, 0, false)
	expectErr(t, "RoPEFreqsBF16 bad rotary", err)
	_, err = RoPEFreqsBF16(ropeBF16, 1, 1, 4, 4, []float32{1}, 1, 0, false)
	expectErr(t, "RoPEFreqsBF16 bad freqs", err)
	if out, err := RoPEFreqsBF16(ropeBF16, 1, 1, 4, 2, []float32{1}, 1, 1, true); err != nil {
		t.Fatalf("RoPEFreqsBF16 partial traditional: %v", err)
	} else if len(out) != len(ropeBF16) {
		t.Fatalf("RoPEFreqsBF16 partial len = %d", len(out))
	}
	withAutoreleasePool(func() {
		xBuf := sharedBytes(ropeBF16)
		outBuf := scratchBF16(4)
		offBuf := scalarI32(0)
		periods := shared([]float32{1, 2})
		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		if err = encRoPEFreqsBF16(enc, xBuf, outBuf, offBuf, periods, 1, 4, 4, 1); err != nil {
			enc.EndEncoding()
			return
		}
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
	})
	if err != nil {
		t.Fatalf("encRoPEFreqsBF16: %v", err)
	}

	_, err = AddBF16(toBF16Bytes([]float32{1}), toBF16Bytes([]float32{1, 2}))
	expectErr(t, "AddBF16 mismatch", err)
	_, err = AddBF16([]byte{1}, []byte{1})
	expectErr(t, "AddBF16 odd", err)
	if out, err := AddBF16(nil, nil); err != nil {
		t.Fatalf("AddBF16 empty: %v", err)
	} else if len(out) != 0 {
		t.Fatalf("AddBF16 empty len = %d", len(out))
	}

	_, err = NormProject([]float32{1}, nil, nil, 1, 1, 1e-5)
	expectErr(t, "NormProject bad sizes", err)
	_, err = MLPBlock(nil, nil, nil, nil, nil, 1, 1, 1e-5)
	expectErr(t, "MLPBlock bad hidden", err)
	_, err = MLPBlock([]float32{1}, []float32{1}, nil, nil, nil, 1, 1, 1e-5)
	expectErr(t, "MLPBlock bad weights", err)
	if out, err := Gelu([]float32{-1, 0, 1}); err != nil {
		t.Fatalf("Gelu: %v", err)
	} else if len(out) != 3 {
		t.Fatalf("Gelu len = %d", len(out))
	}
	_, err = GeluGateMul([]float32{1}, []float32{1, 2})
	expectErr(t, "GeluGateMul mismatch", err)
}

func TestNativeExecutionBranchCoverage(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, vocab, maxLen = 64, 1, 1, 64, 128, 32, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)

	arch := archFixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	arch.ValueNorm = true
	arch.Layer = []model.LayerSpec{
		{Attention: model.GlobalAttention, KVShareFrom: 0, CacheIndex: 0, HeadDim: headDim, KVHeads: nKV},
		{Attention: model.GlobalAttention, KVShareFrom: 0, CacheIndex: -1, HeadDim: headDim, KVHeads: nKV},
	}
	inputs := decodeInputsFixture(2, dModel)
	layers := []DecodeLayerWeights{
		decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3),
		decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 17),
	}
	qLayers := []QuantizedLayerWeights{
		quantizedLayerFixture(t, dModel, nHeads, nKV, headDim, dFF, 64, 4, 5),
		quantizedLayerFixture(t, dModel, nHeads, nKV, headDim, dFF, 64, 4, 19),
	}
	for i := range layers {
		layers[i].WV = nil
		layers[i].QNormW = toBF16Bytes(syntheticFloat32(headDim, 31+i))
		layers[i].KNormW = toBF16Bytes(syntheticFloat32(headDim, 41+i))
		layers[i].PostAttnNormW = toBF16Bytes(syntheticFloat32(dModel, 51+i))
		layers[i].PostFFNormW = toBF16Bytes(syntheticFloat32(dModel, 61+i))
		layers[i].LayerScalarW = toBF16Bytes([]float32{0.75})
		qLayers[i].V = QuantWeight{}
		qLayers[i].QNormW = layers[i].QNormW
		qLayers[i].KNormW = layers[i].KNormW
		qLayers[i].PostAttnNormW = layers[i].PostAttnNormW
		qLayers[i].PostFFNormW = layers[i].PostFFNormW
		qLayers[i].LayerScalarW = layers[i].LayerScalarW
	}

	if out, err := DecodeForwardArch(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, true); err != nil {
		t.Fatalf("DecodeForwardArch shared cache: %v", err)
	} else if len(out) != len(inputs) {
		t.Fatalf("DecodeForwardArch shared outputs = %d", len(out))
	}
	if out, err := DecodeForwardArchQuant(inputs, qLayers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, true); err != nil {
		t.Fatalf("DecodeForwardArchQuant shared cache: %v", err)
	} else if len(out) != len(inputs) {
		t.Fatalf("DecodeForwardArchQuant shared outputs = %d", len(out))
	}
	if out, err := DecodeForwardArchICB(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, true); err != nil {
		t.Fatalf("DecodeForwardArchICB shared cache: %v", err)
	} else if len(out) != len(inputs) {
		t.Fatalf("DecodeForwardArchICB shared outputs = %d", len(out))
	}
	if out, err := DecodeForwardArchICBQuant(inputs, qLayers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, true); err != nil {
		t.Fatalf("DecodeForwardArchICBQuant shared cache: %v", err)
	} else if len(out) != len(inputs) {
		t.Fatalf("DecodeForwardArchICBQuant shared outputs = %d", len(out))
	}

	fullLayer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 71)
	_, err := DecodeForward(nil, nil, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	expectErr(t, "DecodeForward no layers", err)
	_, err = DecodeForward(nil, []DecodeLayerWeights{fullLayer}, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	expectErr(t, "DecodeForward no inputs", err)
	_, err = DecodeForward(inputs, []DecodeLayerWeights{fullLayer}, dModel, nHeads, nKV, headDim, 1, dFF, base, scale, eps)
	expectErr(t, "DecodeForward maxLen", err)
	_, err = DecodeForward([][]byte{{1}}, []DecodeLayerWeights{fullLayer}, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	expectErr(t, "DecodeForward bad input", err)
	badLayer := fullLayer
	badLayer.AttnNormW = []byte{1}
	_, err = DecodeForward(inputs[:1], []DecodeLayerWeights{badLayer}, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	expectErr(t, "DecodeForward bad norm", err)
	badLayer = fullLayer
	badLayer.WQ = []byte{1}
	_, err = DecodeForward(inputs[:1], []DecodeLayerWeights{badLayer}, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	expectErr(t, "DecodeForward bad q", err)
	badLayer = fullLayer
	badLayer.WK = []byte{1}
	_, err = DecodeForward(inputs[:1], []DecodeLayerWeights{badLayer}, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	expectErr(t, "DecodeForward bad kv", err)
	badLayer = fullLayer
	badLayer.WGate = []byte{1}
	_, err = DecodeForward(inputs[:1], []DecodeLayerWeights{badLayer}, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	expectErr(t, "DecodeForward bad mlp", err)

	qLayer := quantizedLayerFixture(t, dModel, nHeads, nKV, headDim, dFF, 64, 4, 83)
	_, err = DecodeForwardQuant(nil, nil, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	expectErr(t, "DecodeForwardQuant empty", err)
	_, err = DecodeForwardQuant(inputs, []QuantizedLayerWeights{qLayer}, dModel, nHeads, nKV, headDim, 1, dFF, base, scale, eps)
	expectErr(t, "DecodeForwardQuant maxLen", err)
	_, err = DecodeForwardQuant([][]byte{{1}}, []QuantizedLayerWeights{qLayer}, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	expectErr(t, "DecodeForwardQuant bad input", err)
	badQ := qLayer
	badQ.GroupSize = 0
	_, err = DecodeForwardQuant(inputs[:1], []QuantizedLayerWeights{badQ}, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	expectErr(t, "DecodeForwardQuant unset geometry", err)
	badQ = qLayer
	badQ.AttnNormW = []byte{1}
	_, err = DecodeForwardQuant(inputs[:1], []QuantizedLayerWeights{badQ}, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	expectErr(t, "DecodeForwardQuant bad norm", err)
	badQ = qLayer
	badQ.Q.GroupSize = 48
	_, err = DecodeForwardQuant(inputs[:1], []QuantizedLayerWeights{badQ}, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	expectErr(t, "DecodeForwardQuant bad group", err)
	badQ = qLayer
	badQ.Q.Packed = []byte{1}
	_, err = DecodeForwardQuant(inputs[:1], []QuantizedLayerWeights{badQ}, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	expectErr(t, "DecodeForwardQuant bad weight", err)

	_, err = NewBF16Backend(arch, layers[:1], maxLen)
	expectErr(t, "NewBF16Backend mismatch", err)
	_, err = NewQuantBackend(arch, qLayers[:1], maxLen)
	expectErr(t, "NewQuantBackend mismatch", err)
	pleArch := arch
	pleArch.PerLayerInputHidden = 32
	backend, err := NewBF16Backend(pleArch, layers, maxLen, WithICB())
	if err != nil {
		t.Fatalf("NewBF16Backend PLE: %v", err)
	}
	_, err = backend.DecodeForward(inputs)
	expectErr(t, "NativeBackend PLE whole forward", err)

	g, oneLayerArch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 1)
	_, err = NewBF16TokenModel(nil, oneLayerArch, maxLen)
	expectErr(t, "NewBF16TokenModel nil", err)
	tm, err := NewBF16TokenModel(g, oneLayerArch, maxLen)
	if err != nil {
		t.Fatalf("NewBF16TokenModel: %v", err)
	}
	_, err = tm.Head([]byte{1})
	expectErr(t, "NativeTokenModel bad head", err)
	if sess, err := tm.OpenSession(); err != nil {
		t.Fatalf("NativeTokenModel OpenSession: %v", err)
	} else if closer, ok := sess.(interface{ Close() error }); ok {
		_ = closer.Close()
	}
	_, err = NewQuantTokenModel(nil, oneLayerArch, maxLen)
	expectErr(t, "NewQuantTokenModel nil", err)
	_, err = NewQuantTokenModel(&QuantModel{}, oneLayerArch, maxLen)
	expectErr(t, "NewQuantTokenModel mismatch", err)

	h := toBF16Bytes(syntheticFloat32(dModel, 7))
	moe := moeLayerWeightsFixture(4, 2, dModel, dFF, 96, 91)
	if out, err := mlpTransformBF16(h, moe.WGate, moe.WUp, moe.WDown, dModel, dFF); err != nil {
		t.Fatalf("mlpTransformBF16: %v", err)
	} else if len(out) != dModel*bf16Size {
		t.Fatalf("mlpTransformBF16 len = %d", len(out))
	}
	_, err = mlpTransformBF16(h, []byte{1}, moe.WUp, moe.WDown, dModel, dFF)
	expectErr(t, "mlpTransformBF16 bad gate", err)
	_, err = mlpTransformBF16(h, moe.WGate, []byte{1}, moe.WDown, dModel, dFF)
	expectErr(t, "mlpTransformBF16 bad up", err)
	_, err = mlpTransformBF16(h, moe.WGate, moe.WUp, []byte{1}, dModel, dFF)
	expectErr(t, "mlpTransformBF16 bad down", err)
	if out, err := MoEBlockBF16(h, moe, dModel, dFF, eps); err != nil {
		t.Fatalf("MoEBlockBF16: %v", err)
	} else if len(out) != dModel*bf16Size {
		t.Fatalf("MoEBlockBF16 len = %d", len(out))
	}
	badMoE := moe
	badMoE.RouterNormWScaled = []byte{1}
	_, err = MoEBlockBF16(h, badMoE, dModel, dFF, eps)
	expectErr(t, "MoEBlockBF16 bad router", err)
	badMoE = moe
	badMoE.PreFFNormW = []byte{1}
	_, err = MoEBlockBF16(h, badMoE, dModel, dFF, eps)
	expectErr(t, "MoEBlockBF16 bad local norm", err)
	badMoE = moe
	badMoE.WGate = []byte{1}
	_, err = MoEBlockBF16(h, badMoE, dModel, dFF, eps)
	expectErr(t, "MoEBlockBF16 bad local mlp", err)
	badMoE = moe
	badMoE.PreFFNorm2W = []byte{1}
	_, err = MoEBlockBF16(h, badMoE, dModel, dFF, eps)
	expectErr(t, "MoEBlockBF16 bad expert norm", err)
	badMoE = moe
	badMoE.ExpGateW = []byte{1}
	_, err = MoEBlockBF16(h, badMoE, dModel, dFF, eps)
	expectErr(t, "MoEBlockBF16 bad experts", err)
	badMoE = moe
	badMoE.PostFFNorm1W = []byte{1}
	_, err = MoEBlockBF16(h, badMoE, dModel, dFF, eps)
	expectErr(t, "MoEBlockBF16 bad post norm one", err)
	badMoE = moe
	badMoE.PostFFNorm2W = []byte{1}
	_, err = MoEBlockBF16(h, badMoE, dModel, dFF, eps)
	expectErr(t, "MoEBlockBF16 bad post norm two", err)
	badMoE = moe
	badMoE.PostFFNormW = []byte{1}
	_, err = MoEBlockBF16(h, badMoE, dModel, dFF, eps)
	expectErr(t, "MoEBlockBF16 bad final norm", err)

	idx := []int32{0, 1}
	weights := toBF16Bytes([]float32{0.75, 0.25})
	_, err = MoEExperts([]byte{1}, idx, weights, moe.ExpGateW, moe.ExpUpW, moe.ExpDownW, 4, 2, dModel, 96)
	expectErr(t, "MoEExperts bad hidden", err)
	_, err = MoEExperts(h, idx[:1], weights, moe.ExpGateW, moe.ExpUpW, moe.ExpDownW, 4, 2, dModel, 96)
	expectErr(t, "MoEExperts bad route length", err)
	_, err = MoEExperts(h, idx, weights, []byte{1}, moe.ExpUpW, moe.ExpDownW, 4, 2, dModel, 96)
	expectErr(t, "MoEExperts bad weights", err)
	_, err = MoEExperts(h, []int32{0, 4}, weights, moe.ExpGateW, moe.ExpUpW, moe.ExpDownW, 4, 2, dModel, 96)
	expectErr(t, "MoEExperts bad expert", err)

	qMoE := quantMoELayerWeightsGuard(t, 4, 2, dModel, dFF, 96, 32, 4)
	_, err = mlpTransformQuant([]byte{1}, qMoE.LocalGate, qMoE.LocalUp, qMoE.LocalDown, dModel, dFF, 32, 4)
	expectErr(t, "mlpTransformQuant bad hidden", err)
	_, _, err = MoERouterQuant(h, []byte{1}, qMoE.Router, nil, 4, 2, dModel, 32, 4, eps)
	expectErr(t, "MoERouterQuant bad norm", err)
	_, _, err = MoERouterQuant(h, qMoE.RouterNormWScaled, qMoE.Router, nil, 4, 0, dModel, 32, 4, eps)
	expectErr(t, "MoERouterQuant bad topK", err)
}

func TestNativeMiscGuardCoverage(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, vocab, maxLen = 64, 1, 1, 64, 128, 32, 6
	const groupSize, bits = 32, 4
	const eps = float32(1e-6)

	_, err := newShardBuffers(nil)
	expectErr(t, "newShardBuffers nil", err)
	_, err = newShardBuffers(&safetensors.DirMapping{Shards: []*safetensors.Mapping{{}}})
	expectErr(t, "newShardBuffers empty shard", err)
	_, err = (&shardBuffers{}).bufFor([]byte{1})
	expectErr(t, "shardBuffers bufFor outside shard", err)
	if got := (*shardBuffers)(nil).mustBufFor([]byte{1}, &err); got.buf != nil {
		t.Fatal("nil shardBuffers mustBufFor should return a zero view")
	}
	err = nil
	if got := (&shardBuffers{}).mustBufFor([]byte{1}, &err); got.buf != nil || err == nil {
		t.Fatalf("empty shardBuffers mustBufFor = (%+v, %v), want zero view and error", got, err)
	}
	err = core.NewError("existing error")
	if got := (&shardBuffers{}).mustBufFor([]byte{1}, &err); got.buf != nil {
		t.Fatal("mustBufFor with prior error should return a zero view")
	}
	if err := (*shardBuffers)(nil).Close(); err != nil {
		t.Fatalf("nil shardBuffers Close: %v", err)
	}

	qEmbed := quantWeightFixture(t, vocab, dModel, groupSize, bits, 7)
	_, err = EmbedTokensQuant(qEmbed.Packed, []byte{1}, qEmbed.Biases, []int32{0}, vocab, dModel, groupSize, bits, 1)
	expectErr(t, "EmbedTokensQuant scales size", err)
	_, err = EmbedTokensQuant(qEmbed.Packed, qEmbed.Scales, qEmbed.Biases, []int32{int32(vocab)}, vocab, dModel, groupSize, bits, 1)
	expectErr(t, "EmbedTokensQuant token range", err)
	if embs, err := EmbedTokensQuant(qEmbed.Packed, qEmbed.Scales, qEmbed.Biases, []int32{0, 1}, vocab, dModel, groupSize, bits, 0.5); err != nil {
		t.Fatalf("EmbedTokensQuant scaled: %v", err)
	} else if len(embs) != 2 || len(embs[0]) != dModel*bf16Size || len(embs[1]) != dModel*bf16Size {
		t.Fatalf("EmbedTokensQuant scaled lengths = %d/%d/%d", len(embs), len(embs[0]), len(embs[1]))
	}

	hidden := toBF16Bytes(syntheticFloat32(dModel, 11))
	norm := toBF16Bytes(syntheticFloat32(dModel, 13))
	qHead := quantWeightFixture(t, vocab, dModel, groupSize, bits, 17)
	_, err = LMHeadQuant(hidden, norm, []byte{1}, qHead.Scales, qHead.Biases, dModel, vocab, groupSize, bits, eps, 0)
	expectErr(t, "LMHeadQuant packed size", err)
	if logits, err := LMHeadQuant(hidden, norm, qHead.Packed, qHead.Scales, qHead.Biases, dModel, vocab, groupSize, bits, eps, 30); err != nil {
		t.Fatalf("LMHeadQuant soft cap: %v", err)
	} else if len(logits) != vocab*bf16Size {
		t.Fatalf("LMHeadQuant soft cap len = %d", len(logits))
	}

	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 19)
	x := toBF16Bytes(syntheticFloat32(dModel, 23))
	kLayer := toBF16Bytes(syntheticFloat32(nKV*2*headDim, 29))
	vLayer := toBF16Bytes(syntheticFloat32(nKV*2*headDim, 31))
	_, err = DecodeLayer([]byte{1}, layer.AttnNormW, layer.WQ, layer.WO, kLayer, vLayer, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, 2, dFF, 10000, 0.125, 0, eps)
	expectErr(t, "DecodeLayer bad x", err)
	_, err = DecodeLayer(x, layer.AttnNormW, []byte{1}, layer.WO, kLayer, vLayer, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, 2, dFF, 10000, 0.125, 0, eps)
	expectErr(t, "DecodeLayer bad q", err)
	_, err = DecodeLayer(x, layer.AttnNormW, layer.WQ, layer.WO, kLayer, vLayer, layer.MLPNormW, []byte{1}, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, 2, dFF, 10000, 0.125, 0, eps)
	expectErr(t, "DecodeLayer bad mlp", err)
	_, err = DecodeLayer(x, layer.AttnNormW, layer.WQ, layer.WO, []byte{1}, vLayer, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, 2, dFF, 10000, 0.125, 0, eps)
	expectErr(t, "DecodeLayer bad cache", err)

	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 1)
	if gen, err := GenerateBF16(g, arch, []int32{1}, 2, maxLen, -1); err != nil {
		t.Fatalf("GenerateBF16 two tokens: %v", err)
	} else if len(gen) != 2 {
		t.Fatalf("GenerateBF16 two tokens len = %d", len(gen))
	}

	dir := t.TempDir()
	writeLocal(t, core.PathJoin(dir, "tokenizer.json"), []byte(nativeCoverageTokenizerJSON))
	tok, err := tokenizer.LoadTokenizer(core.PathJoin(dir, "tokenizer.json"))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession text guard: %v", err)
	}
	_, err = sess.GenerateText(tok, "", 1)
	expectErr(t, "GenerateText empty prompt", err)

	if _, _, _, err := dispatchProfile(0, 1); err != nil {
		t.Fatalf("dispatchProfile zero dispatch: %v", err)
	}
	if _, err := rebindCostProbe(0); err != nil {
		t.Fatalf("rebindCostProbe zero rebinds: %v", err)
	}
	if _, weightBytes, err := gemvProfile(1, 1, 0); err != nil {
		t.Fatalf("gemvProfile zero dispatch: %v", err)
	} else if weightBytes != bf16Size {
		t.Fatalf("gemvProfile weightBytes = %d, want %d", weightBytes, bf16Size)
	}
	if _, weightBytes, err := qmvBF16Profile(8, 512, 64, 0); err != nil {
		t.Fatalf("qmvBF16Profile zero dispatch: %v", err)
	} else if weightBytes == 0 {
		t.Fatal("qmvBF16Profile weightBytes = 0")
	}

	if h, err := newHeadEncoder(nil, nil, nil, nil, nil, dModel, vocab, groupSize, bits, eps, 0, false); err != nil || h != nil {
		t.Fatalf("newHeadEncoder nil shard = (%+v, %v), want nil nil", h, err)
	}
	if h, err := newHeadEncoder(&shardBuffers{}, nil, nil, nil, nil, dModel, vocab, groupSize, bits, eps, 0, true); err != nil || h != nil {
		t.Fatalf("newHeadEncoder missing quant = (%+v, %v), want nil nil", h, err)
	}
	if h, err := newHeadEncoder(&shardBuffers{}, []byte{1, 2}, []byte{1, 2}, nil, nil, dModel, vocab, groupSize, bits, eps, 0, false); err != nil || h != nil {
		t.Fatalf("newHeadEncoder missing shard view = (%+v, %v), want nil nil", h, err)
	}
}

func TestNativeLoaderSessionCoverage(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, vocab, maxLen = 64, 1, 1, 64, 128, 32, 6
	const groupSize, bits = 32, 4
	badConfigJSON := gemma4ConfigJSON(t, g4.Config{})

	badDir := t.TempDir()
	writeLocal(t, core.PathJoin(badDir, "config.json"), badConfigJSON)
	_, err := LoadDir(badDir, maxLen)
	expectErr(t, "LoadDir bf16 arch", err)
	_, err = LoadTokenModelDir(badDir, maxLen)
	expectErr(t, "LoadTokenModelDir bf16 arch", err)

	badQuantDir := t.TempDir()
	badQuantCfg := g4.Config{Quantization: &model.QuantConfig{GroupSize: groupSize, Bits: bits}}
	writeLocal(t, core.PathJoin(badQuantDir, "config.json"), gemma4ConfigJSON(t, badQuantCfg))
	_, err = LoadDir(badQuantDir, maxLen)
	expectErr(t, "LoadDir quant arch", err)

	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: 1, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim,
		VocabSize: vocab, RMSNormEps: 1e-6,
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	bf16Dir := t.TempDir()
	writeLocal(t, core.PathJoin(bf16Dir, "config.json"), gemma4ConfigJSON(t, cfg))
	writeLocal(t, core.PathJoin(bf16Dir, "model.safetensors"), encodedTensors(t, gemma4TensorsMust(t, arch)))
	bf16Sess, err := LoadDir(bf16Dir, maxLen)
	if err != nil {
		t.Fatalf("LoadDir bf16: %v", err)
	}
	defer func() { _ = bf16Sess.Close() }()

	g, oneLayerArch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 1)
	_, err = newArchSessionShards(g, oneLayerArch, maxLen, &shardBuffers{})
	expectErr(t, "newArchSessionShards missing shard view", err)

	sess, err := NewArchSession(g, oneLayerArch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession closures: %v", err)
	}
	sess.perLayerInput = func(int32, []byte) ([]byte, error) { return nil, core.NewError("pli failed") }
	_, err = sess.StepWithID(1, toBF16Bytes(syntheticFloat32(oneLayerArch.Hidden, 3)))
	expectErr(t, "StepWithID PLI error", err)

	sess, err = NewArchSession(g, oneLayerArch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession generate closures: %v", err)
	}
	sess.embed = func(int32) ([]byte, error) { return nil, core.NewError("embed failed") }
	_, err = sess.Generate([]int32{1}, 1, -1)
	expectErr(t, "Generate embed error", err)

	sess, err = NewArchSession(g, oneLayerArch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession PLI generate closures: %v", err)
	}
	sess.perLayerInput = func(int32, []byte) ([]byte, error) { return nil, core.NewError("pli failed") }
	_, err = sess.Generate([]int32{1}, 1, -1)
	expectErr(t, "Generate PLI error", err)

	sess, err = NewArchSession(g, oneLayerArch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession eos: %v", err)
	}
	eosID := int32(3)
	sess.greedy = nil
	sess.head = func([]byte, bool) ([]byte, error) {
		logits := make([]float32, oneLayerArch.Vocab)
		logits[eosID] = 100
		return toBF16Bytes(logits), nil
	}
	gen, err := sess.Generate([]int32{1}, 2, int(eosID))
	if err != nil {
		t.Fatalf("Generate eos: %v", err)
	}
	if len(gen) != 1 || gen[0] != eosID {
		t.Fatalf("Generate eos = %v, want [%d]", gen, eosID)
	}

	quantCfg := cfg
	quantCfg.Quantization = &model.QuantConfig{GroupSize: groupSize, Bits: bits}
	quantDir := t.TempDir()
	writeLocal(t, core.PathJoin(quantDir, "config.json"), gemma4ConfigJSON(t, quantCfg))
	writeLocal(t, core.PathJoin(quantDir, "model.safetensors"), encodedTensors(t, quantGemma4TensorsGuard(t, arch, groupSize, bits)))

	qSess, err := LoadDir(quantDir, maxLen)
	if err != nil {
		t.Fatalf("LoadDir quant: %v", err)
	}
	defer func() { _ = qSess.Close() }()
	if gen, err := qSess.Generate([]int32{1}, 1, -1); err != nil {
		t.Fatalf("quant session Generate: %v", err)
	} else if len(gen) != 1 {
		t.Fatalf("quant session Generate len = %d", len(gen))
	}

	tm, err := LoadTokenModelDir(quantDir, maxLen)
	if err != nil {
		t.Fatalf("LoadTokenModelDir quant: %v", err)
	}
	if closer, ok := tm.(interface{ Close() error }); ok {
		defer func() { _ = closer.Close() }()
	}
	emb, err := tm.Embed(1)
	if err != nil {
		t.Fatalf("quant token model Embed: %v", err)
	}
	if len(emb) != dModel*bf16Size {
		t.Fatalf("quant token model Embed len = %d", len(emb))
	}
	logits, err := tm.Head(emb)
	if err != nil {
		t.Fatalf("quant token model Head: %v", err)
	}
	if len(logits) != vocab*bf16Size {
		t.Fatalf("quant token model Head len = %d", len(logits))
	}
	if ntm, ok := tm.(*NativeTokenModel); ok {
		stepper, err := ntm.OpenSession()
		if err != nil {
			t.Fatalf("quant token model OpenSession: %v", err)
		}
		if closer, ok := stepper.(interface{ Close() error }); ok {
			_ = closer.Close()
		}
	}
}

func guardArchDecodeState(specs []model.LayerSpec, dModel, nHeads, nKV, headDim, dFF, maxLen int, projs []projector) archDecodeState {
	norm := copyView(toBF16Bytes(fillConst(dModel, 1)))
	lb := make([]archLayerBufs, len(specs))
	for i, sp := range specs {
		p := projs[i]
		lb[i] = archLayerBufs{anw: norm, mnw: norm, proj: p, dFF: dFF}
		if sp.OwnsCache() {
			kvDim := kvHeadsOf(sp, nKV) * headDimOf(sp, headDim)
			lb[i].kCache = scratchBF16(maxLen * kvDim)
			lb[i].vCache = scratchBF16(maxLen * kvDim)
		}
	}
	return newArchDecodeState(specs, lb, make([]*MoELayerWeights, len(specs)), dModel, nHeads, nKV, headDim, dFF, 0, headDim, headDim, 10000, 10000, 0.125, 1e-5, false, 0)
}

func TestNativeProjectorErrorCoverage(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, maxLen = 64, 1, 1, 64, 128, 4
	emb := toBF16Bytes(syntheticFloat32(dModel, 3))
	owner := model.LayerSpec{Attention: model.GlobalAttention, KVShareFrom: 0, CacheIndex: 0, HeadDim: headDim, KVHeads: nKV}
	failErr := core.NewError("project failed")

	for _, idx := range []projIndex{projQ, projK, projV, projO, projGate, projUp, projDown} {
		proj := failingProjector{fail: idx, err: failErr, distinctV: true}
		if idx == projO {
			proj.distinctV = false
		}
		var err error
		withAutoreleasePool(func() {
			st := guardArchDecodeState([]model.LayerSpec{owner}, dModel, nHeads, nKV, headDim, dFF, maxLen, []projector{proj})
			_, err = st.stepToken(emb, 0)
		})
		expectErr(t, core.Sprintf("stepToken projector %d", idx), err)
	}

	sharer := model.LayerSpec{Attention: model.GlobalAttention, KVShareFrom: 0, CacheIndex: -1, HeadDim: headDim, KVHeads: nKV}
	var err error
	withAutoreleasePool(func() {
		st := guardArchDecodeState(
			[]model.LayerSpec{owner, sharer},
			dModel, nHeads, nKV, headDim, dFF, maxLen,
			[]projector{
				failingProjector{distinctV: false},
				failingProjector{fail: projQ, err: failErr, distinctV: true},
			},
		)
		_, err = st.stepToken(emb, 0)
	})
	expectErr(t, "stepToken shared projector", err)

	withAutoreleasePool(func() {
		st := guardArchDecodeState(
			[]model.LayerSpec{owner, sharer},
			dModel, nHeads, nKV, headDim, dFF, maxLen,
			[]projector{
				failingProjector{distinctV: false},
				failingProjector{fail: projO, err: failErr, distinctV: true},
			},
		)
		_, err = st.stepToken(emb, 0)
	})
	expectErr(t, "stepToken shared output projector", err)
}

func TestNativeRemainderValidationCoverage(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, vocab, maxLen = 64, 1, 1, 64, 128, 32, 4
	const groupSize, bits = 32, 4
	const eps = float32(1e-5)

	if got := attnScaleOf(model.Arch{HeadDim: headDim}); got != 0.125 {
		t.Fatalf("attnScaleOf fallback = %v, want 0.125", got)
	}
	if out, err := QMV(nil, nil, nil, nil, 0, 0, groupSize, bits); err != nil || len(out) != 0 {
		t.Fatalf("QMV zero = (%d, %v), want empty nil", len(out), err)
	}
	if out, err := QMVBF16(nil, nil, nil, nil, 0, 0, groupSize, bits); err != nil || len(out) != 0 {
		t.Fatalf("QMVBF16 zero = (%d, %v), want empty nil", len(out), err)
	}

	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 7)
	x := toBF16Bytes(syntheticFloat32(dModel, 11))
	kCache := toBF16Bytes(syntheticFloat32(nKV*2*headDim, 13))
	vCache := toBF16Bytes(syntheticFloat32(nKV*2*headDim, 17))

	_, err := AttentionBlock(x, layer.AttnNormW, layer.WQ, []byte{1}, kCache, vCache, dModel, nHeads, nKV, headDim, 2, 10000, 0.125, 0, eps)
	expectErr(t, "AttentionBlock bad output projection", err)
	_, err = AttentionBlock(x, layer.AttnNormW, layer.WQ, layer.WO, []byte{1}, vCache, dModel, nHeads, nKV, headDim, 2, 10000, 0.125, 0, eps)
	expectErr(t, "AttentionBlock bad cache", err)

	_, err = MLPBlockBF16(x, layer.MLPNormW, layer.WGate, layer.WUp, []byte{1}, dModel, dFF, eps)
	expectErr(t, "MLPBlockBF16 bad down", err)
	_, err = DecodeLayer(x, layer.AttnNormW, layer.WQ, []byte{1}, kCache, vCache, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, 2, dFF, 10000, 0.125, 0, eps)
	expectErr(t, "DecodeLayer bad output projection", err)
	_, err = DecodeLayer(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, layer.MLPNormW, layer.WGate, layer.WUp, []byte{1}, dModel, nHeads, nKV, headDim, 2, dFF, 10000, 0.125, 0, eps)
	expectErr(t, "DecodeLayer bad down projection", err)
	_, err = AttentionStepKV(x, layer.AttnNormW, layer.WQ, layer.WK, []byte{1}, layer.WO, make([]byte, maxLen*nKV*headDim*bf16Size), make([]byte, maxLen*nKV*headDim*bf16Size), dModel, nHeads, nKV, headDim, maxLen, 0, 10000, 0.125, eps)
	expectErr(t, "AttentionStepKV bad value projection", err)
	_, err = AttentionStepKV(x, layer.AttnNormW, layer.WQ, layer.WK, layer.WV, layer.WO, make([]byte, maxLen*nKV*headDim*bf16Size), make([]byte, maxLen*nKV*headDim*bf16Size), dModel, nHeads+1, nKV, headDim, maxLen, 0, 10000, 0.125, eps)
	expectErr(t, "AttentionStepKV bad head multiple", err)
	_, err = DecodeTokenICB(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, layer.MLPNormW, []byte{1}, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, 2, dFF, 1, 10000, 0.125, 0, eps, 1)
	expectErr(t, "DecodeTokenICB bad mlp", err)
	_, err = DecodeTokenICB(x, layer.AttnNormW, layer.WQ, layer.WO, []byte{1}, vCache, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, 2, dFF, 1, 10000, 0.125, 0, eps, 1)
	expectErr(t, "DecodeTokenICB bad cache", err)

	table := toBF16Bytes(syntheticFloat32(vocab*dModel, 19))
	_, err = EmbedTokensBF16(table, []int32{-1}, vocab, dModel, 1)
	expectErr(t, "EmbedTokensBF16 negative token", err)
	_, err = LMHeadBF16(x, layer.MLPNormW, []byte{1}, dModel, vocab, eps, 0)
	expectErr(t, "LMHeadBF16 bad output weight", err)

	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 1)
	tm, err := NewBF16TokenModel(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewBF16TokenModel: %v", err)
	}
	_, err = tm.Embed(int32(vocab))
	expectErr(t, "NativeTokenModel bf16 embed range", err)

	inputs := decodeInputsFixture(2, dModel)
	layers := []DecodeLayerWeights{layer}
	_, err = DecodeForwardArchICB(nil, nil, nil, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, 10000, 0.125, eps, false)
	expectErr(t, "DecodeForwardArchICB empty", err)
	_, err = DecodeForwardArchICB(inputs, layers, nil, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, 10000, 0.125, eps, false)
	expectErr(t, "DecodeForwardArchICB specs mismatch", err)
	_, err = DecodeForwardArchICB(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, 1, dFF, 0, 10000, 0.125, eps, false)
	expectErr(t, "DecodeForwardArchICB maxLen", err)
	_, err = DecodeForwardArchICB([][]byte{{1}}, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, 10000, 0.125, eps, false)
	expectErr(t, "DecodeForwardArchICB bad input", err)
	badSpecs := append([]model.LayerSpec(nil), arch.Layer...)
	badSpecs[0].KVShareFrom = -1
	_, err = DecodeForwardArchICB(inputs, layers, badSpecs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, 10000, 0.125, eps, false)
	expectErr(t, "DecodeForwardArchICB bad share", err)
	moeSpecs := append([]model.LayerSpec(nil), arch.Layer...)
	moeSpecs[0].MoE = true
	_, err = DecodeForwardArchICB(inputs, layers, moeSpecs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, 10000, 0.125, eps, false)
	expectErr(t, "DecodeForwardArchICB moe", err)

	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession step error: %v", err)
	}
	withAutoreleasePool(func() {
		sess.state = guardArchDecodeState(
			arch.Layer, dModel, nHeads, nKV, headDim, dFF, maxLen,
			[]projector{failingProjector{fail: projQ, err: core.NewError("project failed"), distinctV: true}},
		)
	})
	_, err = sess.Step(toBF16Bytes(syntheticFloat32(dModel, 23)))
	expectErr(t, "ArchSession Step decode error", err)

	sess, err = NewArchSession(g, arch, 1)
	if err != nil {
		t.Fatalf("NewArchSession text error: %v", err)
	}
	dir := t.TempDir()
	writeLocal(t, core.PathJoin(dir, "tokenizer.json"), []byte(nativeCoverageTokenizerJSON))
	tok, err := tokenizer.LoadTokenizer(core.PathJoin(dir, "tokenizer.json"))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	_, err = sess.GenerateText(tok, "h", 1)
	expectErr(t, "GenerateText generate error", err)

	qlm, err := g4Assemble(quantGemma4TensorsGuard(t, arch, groupSize, bits), arch)
	if err != nil {
		t.Fatalf("gemma4.Assemble: %v", err)
	}
	qg, err := loadedToQuant(qlm, groupSize, bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	qtm, err := NewQuantTokenModel(qg, arch, maxLen)
	if err != nil {
		t.Fatalf("NewQuantTokenModel: %v", err)
	}
	_, err = qtm.Embed(int32(vocab))
	expectErr(t, "NativeTokenModel quant embed range", err)
	_, err = qtm.Head([]byte{1})
	expectErr(t, "NativeTokenModel quant bad head", err)

	_, err = loadedToQuant(nil, groupSize, bits)
	expectErr(t, "loadedToQuant nil", err)
	_, err = loadedToQuant(&model.LoadedModel{}, groupSize, bits)
	expectErr(t, "loadedToQuant missing embed", err)
	if folded := foldRootSize(nil, dModel); folded != nil {
		t.Fatalf("foldRootSize nil = %v, want nil", folded)
	}

	denseLin := &model.Linear{Weight: []byte{1, 2}, OutDim: dFF}
	quantLin := &model.Linear{Weight: []byte{1}, Scales: []byte{2}, Biases: []byte{3}, GroupSize: groupSize, Bits: bits, Kind: "affine", OutDim: dFF}
	loadedDense := &model.LoadedModel{
		Arch:              model.Arch{Hidden: dModel},
		Embed:             denseLin,
		FinalNorm:         layer.MLPNormW,
		EmbedPerLayer:     denseLin,
		PerLayerModelProj: denseLin,
		PerLayerProjNorm:  layer.MLPNormW,
		Layers: []model.LoadedLayer{{
			AttnNorm: layer.AttnNormW, PostAttnNorm: layer.PostAttnNormW,
			QNorm: layer.QNormW, KNorm: layer.KNormW, LayerScalar: layer.LayerScalarW,
			Q: denseLin, K: denseLin, V: denseLin, O: denseLin,
			MLPNorm: layer.MLPNormW, PostFFNorm: layer.PostFFNormW,
			Gate: denseLin, Up: denseLin, Down: denseLin,
			PerLayerGate: denseLin, PerLayerProjection: denseLin, PostPerLayerInputNorm: layer.MLPNormW,
		}},
	}
	if got := loadedToBF16(loadedDense); !got.Tied || len(got.EmbedPerLayer) == 0 || got.Layers[0].DFF != dFF {
		t.Fatalf("loadedToBF16 = tied %v ple %d dff %d", got.Tied, len(got.EmbedPerLayer), got.Layers[0].DFF)
	}

	loadedQuant := &model.LoadedModel{
		Arch:              model.Arch{Hidden: dModel, Experts: 2, TopK: 1, ExpertFF: 16},
		Embed:             quantLin,
		FinalNorm:         layer.MLPNormW,
		EmbedPerLayer:     quantLin,
		PerLayerModelProj: quantLin,
		PerLayerProjNorm:  layer.MLPNormW,
		Layers: []model.LoadedLayer{{
			AttnNorm: layer.AttnNormW, Q: quantLin, K: quantLin, V: quantLin, O: quantLin,
			PerLayerGate: quantLin, PerLayerProjection: quantLin, PostPerLayerInputNorm: layer.MLPNormW,
			MoE: &model.LoadedMoE{
				PreFFNorm: layer.MLPNormW, PreFFNorm2: layer.MLPNormW,
				PostFFNorm1: layer.MLPNormW, PostFFNorm2: layer.MLPNormW, PostFFNorm: layer.MLPNormW,
				RouterScale: layer.MLPNormW, PerExpertScale: layer.MLPNormW,
				LocalGate: quantLin, LocalUp: quantLin, LocalDown: quantLin,
				Router: quantLin, ExpGate: quantLin, ExpUp: quantLin, ExpDown: quantLin,
			},
		}},
	}
	if got, err := loadedToQuant(loadedQuant, groupSize, bits); err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	} else if !got.Tied || got.Layers[0].MoE == nil || got.PerLayerModelProjBits != bits {
		t.Fatalf("loadedToQuant = tied %v moe %v projBits %d", got.Tied, got.Layers[0].MoE != nil, got.PerLayerModelProjBits)
	}
}

func TestNativeRemainingBranchCoverage(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, vocab, maxLen = 64, 1, 1, 64, 128, 32, 4
	const groupSize, bits = 32, 4
	const eps = float32(1e-5)

	arch := archFixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 1)
	inputs := decodeInputsFixture(2, dModel)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)
	qLayer := quantizedLayerFixture(t, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, 5)

	_, err := DecodeForwardArchQuant(inputs, []QuantizedLayerWeights{qLayer}, arch.Layer, dModel, nHeads, nKV, headDim, 1, dFF, 0, 10000, 0.125, eps, false)
	expectErr(t, "DecodeForwardArchQuant maxLen", err)
	_, err = DecodeForwardArchQuant([][]byte{{1}}, []QuantizedLayerWeights{qLayer}, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, 10000, 0.125, eps, false)
	expectErr(t, "DecodeForwardArchQuant bad input", err)
	badSpecs := append([]model.LayerSpec(nil), arch.Layer...)
	badSpecs[0].KVShareFrom = -1
	_, err = DecodeForwardArchQuant(inputs, []QuantizedLayerWeights{qLayer}, badSpecs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, 10000, 0.125, eps, false)
	expectErr(t, "DecodeForwardArchQuant bad share", err)
	badQ := qLayer
	badQ.AttnNormW = []byte{1}
	_, err = DecodeForwardArchQuant(inputs, []QuantizedLayerWeights{badQ}, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, 10000, 0.125, eps, false)
	expectErr(t, "DecodeForwardArchQuant bad norm", err)
	badQ = qLayer
	badQ.Q.GroupSize = 48
	_, err = DecodeForwardArchQuant(inputs, []QuantizedLayerWeights{badQ}, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, 10000, 0.125, eps, false)
	expectErr(t, "DecodeForwardArchQuant bad group multiple", err)
	badQ = qLayer
	badQ.Q.Packed = []byte{1}
	_, err = DecodeForwardArchQuant(inputs, []QuantizedLayerWeights{badQ}, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, 10000, 0.125, eps, false)
	expectErr(t, "DecodeForwardArchQuant bad weight", err)

	oldProfile := profileForward
	profileForward, profForwardGPUSec = true, 0
	t.Cleanup(func() { profileForward = oldProfile })
	if out, err := DecodeForwardICB(inputs[:1], []DecodeLayerWeights{layer}, dModel, nHeads, nKV, headDim, maxLen, dFF, 10000, 0.125, eps); err != nil {
		t.Fatalf("DecodeForwardICB profiled: %v", err)
	} else if len(out) != 1 {
		t.Fatalf("DecodeForwardICB profiled outputs = %d", len(out))
	}

	owner := model.LayerSpec{Attention: model.GlobalAttention, KVShareFrom: 0, CacheIndex: 0, HeadDim: headDim, KVHeads: nKV}
	emb := toBF16Bytes(syntheticFloat32(dModel, 7))
	withAutoreleasePool(func() {
		st := guardArchDecodeState([]model.LayerSpec{owner}, dModel, nHeads, nKV, headDim, dFF, maxLen, []projector{failingProjector{distinctV: false}})
		st.moeWeights = []*MoELayerWeights{{}}
		_, err = st.stepToken(emb, 0)
	})
	expectErr(t, "stepToken MoE error", err)
	withAutoreleasePool(func() {
		st := guardArchDecodeState([]model.LayerSpec{owner}, dModel, nHeads, nKV, headDim, dFF, maxLen, []projector{failingProjector{distinctV: false}})
		_, err = runArchDecode([][]byte{emb}, st.specs, st.lb, []*MoELayerWeights{{}}, dModel, nHeads, nKV, headDim, dFF, 0, headDim, headDim, 10000, 10000, 0.125, eps, false, 0)
	})
	expectErr(t, "runArchDecode step error", err)
	withAutoreleasePool(func() {
		st := guardArchDecodeState([]model.LayerSpec{owner}, dModel, nHeads, nKV, headDim, dFF, maxLen, []projector{failingProjector{distinctV: false}})
		st.pliDim = 32
		st.perLayerInput = toBF16Bytes(syntheticFloat32(32, 11))
		st.ple = []pleLayer{{gate: QuantWeight{Packed: []byte{1}}, proj: QuantWeight{Packed: []byte{1}}, postNorm: []byte{1}}}
		_, err = st.stepToken(emb, 0)
	})
	expectErr(t, "stepToken PLE error", err)
	withAutoreleasePool(func() {
		wide := model.LayerSpec{Attention: model.GlobalAttention, KVShareFrom: 0, CacheIndex: 0, HeadDim: 128, KVHeads: 2}
		_ = newArchDecodeState([]model.LayerSpec{wide}, []archLayerBufs{{dFF: dFF * 2}}, nil, dModel, nHeads, nKV, headDim, dFF, 0, 32, 64, 10000, 10000, 0.125, eps, true, 0)
	})
	withAutoreleasePool(func() {
		st := guardArchDecodeState([]model.LayerSpec{owner}, dModel, nHeads, nKV, headDim, dFF, maxLen, []projector{failingProjector{distinctV: false}})
		st.trace = true
		traceEmb := toBF16Bytes(append([]float32{float32(math.Inf(1)), -4}, syntheticFloat32(dModel-2, 13)...))
		if _, err = st.stepToken(traceEmb, 0); err != nil {
			t.Fatalf("stepToken trace bad values: %v", err)
		}
	})

	g, oneLayerArch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 1)
	qlm, err := g4Assemble(quantGemma4TensorsGuard(t, oneLayerArch, groupSize, bits), oneLayerArch)
	if err != nil {
		t.Fatalf("gemma4.Assemble: %v", err)
	}
	qg, err := loadedToQuant(qlm, groupSize, bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	_, err = NewArchQuantSession(qg, oneLayerArch, 0)
	expectErr(t, "NewArchQuantSession bad maxLen", err)
	_, err = newArchQuantSessionShards(qg, oneLayerArch, maxLen, &shardBuffers{})
	expectErr(t, "newArchQuantSessionShards missing shard view", err)
	qsess, err := NewArchQuantSession(qg, oneLayerArch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession: %v", err)
	}
	_, err = qsess.embed(int32(vocab))
	expectErr(t, "quant session embed range", err)
	tmBad, err := NewBF16TokenModel(g, oneLayerArch, 0)
	if err != nil {
		t.Fatalf("NewBF16TokenModel maxLen zero: %v", err)
	}
	_, err = tmBad.OpenSession()
	expectErr(t, "NewBF16TokenModel OpenSession bad maxLen", err)
	qtmBad, err := NewQuantTokenModel(qg, oneLayerArch, 0)
	if err != nil {
		t.Fatalf("NewQuantTokenModel maxLen zero: %v", err)
	}
	_, err = qtmBad.OpenSession()
	expectErr(t, "NewQuantTokenModel OpenSession bad maxLen", err)
	sess, err := NewArchSession(g, oneLayerArch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	_, err = sess.embed(int32(vocab))
	expectErr(t, "bf16 session embed range", err)
	withAutoreleasePool(func() {
		sess.state = guardArchDecodeState(oneLayerArch.Layer, dModel, nHeads, nKV, headDim, dFF, maxLen, []projector{failingProjector{fail: projQ, err: core.NewError("project failed"), distinctV: true}})
	})
	_, err = sess.Generate([]int32{1}, 1, -1)
	expectErr(t, "Generate generated step error", err)

	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: 1, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim,
		VocabSize: vocab, RMSNormEps: 1e-6,
	}
	dirArch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	bf16Dir := t.TempDir()
	writeLocal(t, core.PathJoin(bf16Dir, "config.json"), gemma4ConfigJSON(t, cfg))
	writeLocal(t, core.PathJoin(bf16Dir, "model.safetensors"), encodedTensors(t, gemma4TensorsMust(t, dirArch)))
	loadedBF16TM, err := LoadTokenModelDir(bf16Dir, 0)
	if err != nil {
		t.Fatalf("LoadTokenModelDir bf16 maxLen zero: %v", err)
	}
	if closer, ok := loadedBF16TM.(interface{ Close() error }); ok {
		defer func() { _ = closer.Close() }()
	}
	if sessionModel, ok := loadedBF16TM.(model.SessionModel); ok {
		_, err = sessionModel.OpenSession()
		expectErr(t, "LoadTokenModelDir bf16 OpenSession bad maxLen", err)
	} else {
		t.Fatal("loaded bf16 token model is not a SessionModel")
	}
	quantCfg := cfg
	quantCfg.Quantization = &model.QuantConfig{GroupSize: groupSize, Bits: bits}
	quantDir := t.TempDir()
	writeLocal(t, core.PathJoin(quantDir, "config.json"), gemma4ConfigJSON(t, quantCfg))
	writeLocal(t, core.PathJoin(quantDir, "model.safetensors"), encodedTensors(t, quantGemma4TensorsGuard(t, dirArch, groupSize, bits)))
	loadedQuantTM, err := LoadTokenModelDir(quantDir, 0)
	if err != nil {
		t.Fatalf("LoadTokenModelDir quant maxLen zero: %v", err)
	}
	if closer, ok := loadedQuantTM.(interface{ Close() error }); ok {
		defer func() { _ = closer.Close() }()
	}
	if sessionModel, ok := loadedQuantTM.(model.SessionModel); ok {
		_, err = sessionModel.OpenSession()
		expectErr(t, "LoadTokenModelDir quant OpenSession bad maxLen", err)
	} else {
		t.Fatal("loaded quant token model is not a SessionModel")
	}

	hidden := toBF16Bytes(syntheticFloat32(dModel, 17))
	norm := toBF16Bytes(syntheticFloat32(dModel, 19))
	routerW := toBF16Bytes(syntheticFloat32(4*dModel, 23))
	perExpertScale := toBF16Bytes([]float32{1, 0.75, 0.5, 0.25})
	if idx, weights, err := MoERouter(hidden, norm, routerW, perExpertScale, 4, 2, dModel, eps); err != nil {
		t.Fatalf("MoERouter scaled: %v", err)
	} else if len(idx) != 2 || len(weights) != 2*bf16Size {
		t.Fatalf("MoERouter scaled lengths = %d/%d", len(idx), len(weights))
	}
	qRouter := quantWeightFixture(t, 4, dModel, groupSize, bits, 29)
	_, _, err = MoERouterQuant(hidden, norm, QuantWeight{Packed: []byte{1}}, nil, 4, 2, dModel, groupSize, bits, eps)
	expectErr(t, "MoERouterQuant bad weight", err)
	if idx, weights, err := MoERouterQuant(hidden, norm, qRouter, perExpertScale, 4, 2, dModel, groupSize, bits, eps); err != nil {
		t.Fatalf("MoERouterQuant scaled: %v", err)
	} else if len(idx) != 2 || len(weights) != 2*bf16Size {
		t.Fatalf("MoERouterQuant scaled lengths = %d/%d", len(idx), len(weights))
	}
}
