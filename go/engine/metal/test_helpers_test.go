// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"os"
	"runtime/debug"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	g4 "dappco.re/go/inference/model/gemma4"
	"dappco.re/go/inference/model/mistral"
	"dappco.re/go/inference/model/safetensors"
)

func requireNativeRuntime(t testing.TB) {
	t.Helper()
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Fatal(err)
	}
}

func forceNativeGC() {
	debug.FreeOSMemory()
}

func configJSONWithModelType(t testing.TB, cfg any, modelType string) []byte {
	t.Helper()
	js := core.JSONMarshal(cfg)
	if !js.OK {
		t.Fatalf("marshal config: %s", js.Error())
	}
	var m map[string]any
	if r := core.JSONUnmarshal(js.Value.([]byte), &m); !r.OK {
		t.Fatalf("re-parse config for model_type: %s", r.Error())
	}
	if _, ok := m["model_type"]; !ok {
		m["model_type"] = modelType
	}
	out := core.JSONMarshal(m)
	if !out.OK {
		t.Fatalf("re-marshal config: %s", out.Error())
	}
	return out.Value.([]byte)
}

func syntheticFloat32(n, salt int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32((i*salt+7)%101-50) * 0.03125
	}
	return v
}

// quantWeightFixture affine-quantises synthetic float32 data in PURE GO — no cgo, no metal.
// It is the exact inverse of this package's own dequantizeAffineRowsF32/extractAffineCode
// (decode_forward_quant.go, embed_lmhead_quant.go): per-group min/max affine scaling,
// LSB-first bit packing across byte boundaries. Callers only need a SELF-CONSISTENT
// (Packed, Scales, Biases) triple to drive native quantised kernels — none of the ~30
// call sites across the package compare against an external metal oracle, so this
// fixture need not match MLX's own rounding, only this package's own unpacking contract.
func quantWeightFixture(tb testing.TB, outDim, inDim, groupSize, bits, salt int) QuantWeight {
	tb.Helper()
	if err := checkAffineQuantDims(inDim, groupSize, bits); err != nil {
		tb.Fatalf("quantWeightFixture: %v", err)
	}
	packed, scales, biases := packAffineQuant(syntheticFloat32(outDim*inDim, salt), outDim, inDim, groupSize, bits)
	return QuantWeight{Packed: packed, Scales: scales, Biases: biases, GroupSize: groupSize, Bits: bits}
}

// quantizeProj quantises a synthetic [outDim × inDim] bf16 weight in PURE GO — no cgo, no metal —
// returning the same (packed, scales, biases) tuple pkg/model's checkpoint loader expects.
// quantGemma4Tensors below is its only caller.
func quantizeProj(t testing.TB, outDim, inDim, gs, bits, salt int) (packed, scales, biases []byte) {
	t.Helper()
	if err := checkAffineQuantDims(inDim, gs, bits); err != nil {
		t.Fatalf("quantizeProj: %v", err)
	}
	f := make([]float32, outDim*inDim)
	for i := range f {
		f[i] = float32((i*salt+7)%101-50) * 0.02
	}
	return packAffineQuant(f, outDim, inDim, gs, bits)
}

// quantGemma4Tensors builds a full 4-bit gemma4 checkpoint's tensors with synthetic quant weights
// (every projection + the embedding affine-packed via quantizeProj, the norms bf16) — pure Go, no
// cgo/metal needed to produce a self-consistent on-disk-shaped tensor map.
func quantGemma4Tensors(t testing.TB, arch model.Arch, gs, bits int) map[string]safetensors.Tensor {
	t.Helper()
	ts := map[string]safetensors.Tensor{}
	salt := 1
	mkNorm := func(name string, elems int) {
		f := make([]float32, elems)
		for i := range f {
			f[i] = float32((i*salt+13)%97-48) * 0.02
		}
		ts[name] = safetensors.Tensor{Dtype: "BF16", Shape: []int{elems}, Data: toBF16Bytes(f)}
		salt++
	}
	mkQuant := func(prefix string, outDim, inDim int) {
		p, s, b := quantizeProj(t, outDim, inDim, gs, bits, salt)
		salt++
		ts[prefix+".weight"] = safetensors.Tensor{Dtype: "U32", Shape: []int{outDim, inDim * bits / 32}, Data: p}
		ts[prefix+".scales"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{outDim, inDim / gs}, Data: s}
		ts[prefix+".biases"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{outDim, inDim / gs}, Data: b}
	}
	dModel, headDim, dFF, vocab := arch.Hidden, arch.HeadDim, arch.FF, arch.Vocab
	mkQuant("model.embed_tokens", vocab, dModel)
	mkNorm("model.norm.weight", dModel)
	for i := range arch.Layer {
		p := core.Sprintf("model.layers.%d", i)
		mkNorm(p+".input_layernorm.weight", dModel)
		mkNorm(p+".pre_feedforward_layernorm.weight", dModel)
		lhd := headDimOf(arch.Layer[i], headDim)      // per-layer head dim (gemma4 full_attention > sliding)
		lkv := kvHeadsOf(arch.Layer[i], arch.KVHeads) // per-layer kv heads (gemma4 global MQA < sliding GQA)
		lqDim, lkvDim := arch.Heads*lhd, lkv*lhd
		mkNorm(p+".self_attn.q_norm.weight", lhd)
		mkNorm(p+".self_attn.k_norm.weight", lhd)
		mkNorm(p+".post_attention_layernorm.weight", dModel)
		mkNorm(p+".post_feedforward_layernorm.weight", dModel)
		mkQuant(p+".self_attn.q_proj", lqDim, dModel)
		mkQuant(p+".self_attn.k_proj", lkvDim, dModel)
		mkQuant(p+".self_attn.v_proj", lkvDim, dModel)
		mkQuant(p+".self_attn.o_proj", dModel, lqDim)
		mkQuant(p+".mlp.gate_proj", dFF, dModel)
		mkQuant(p+".mlp.up_proj", dFF, dModel)
		mkQuant(p+".mlp.down_proj", dModel, dFF)
	}
	return ts
}

// checkAffineQuantDims validates the shape constraints packAffineQuant (and this package's own
// dequantizeAffineRowsF32) require: cols must be a positive multiple of groupSize, and cols*bits
// must be byte-aligned.
func checkAffineQuantDims(cols, groupSize, bits int) error {
	if groupSize <= 0 || cols%groupSize != 0 {
		return core.NewError(core.Sprintf("cols %d must be a positive multiple of groupSize %d", cols, groupSize))
	}
	if bits <= 0 || bits > 8 || cols*bits%8 != 0 {
		return core.NewError(core.Sprintf("cols %d * bits %d must be byte-aligned (bits in 1..8)", cols, bits))
	}
	return nil
}

// packAffineQuant affine-quantises f (row-major [outDim x inDim]) in PURE GO, matching this
// package's own dequantizeAffineRowsF32/extractAffineCode unpacking contract exactly (LSB-first
// bit packing, per-group min/max affine scaling: code = round((v-lo)/scale), scale=(hi-lo)/maxCode,
// bias=lo). Shared by quantWeightFixture and quantizeProj — no cgo/metal needed to produce a
// self-consistent quantised fixture; callers only need SOMETHING this package's own dequantizer
// reconstructs correctly, not a bit-exact match to MLX's own quantiser.
func packAffineQuant(f []float32, outDim, inDim, groupSize, bits int) (packed, scales, biases []byte) {
	rowPacked := inDim * bits / 8
	groupsPerRow := inDim / groupSize
	rowSB := groupsPerRow * bf16Size
	maxCode := uint32(1)<<uint(bits) - 1
	packed = make([]byte, outDim*rowPacked)
	scales = make([]byte, outDim*rowSB)
	biases = make([]byte, outDim*rowSB)
	for r := range outDim {
		row := f[r*inDim : (r+1)*inDim]
		pRow := packed[r*rowPacked : (r+1)*rowPacked]
		sRow := scales[r*rowSB : (r+1)*rowSB]
		bRow := biases[r*rowSB : (r+1)*rowSB]
		for g := range groupsPerRow {
			group := row[g*groupSize : (g+1)*groupSize]
			lo, hi := group[0], group[0]
			for _, v := range group[1:] {
				if v < lo {
					lo = v
				}
				if v > hi {
					hi = v
				}
			}
			scale := (hi - lo) / float32(maxCode)
			if scale == 0 {
				scale = 1
			}
			// Round scale/bias to bf16 FIRST (their actual storage precision), then compute codes
			// against the ROUNDED values — not the exact float32 ones. dequantizeAffineRowsF32 only
			// ever sees the bf16-rounded scale/bias, so a code chosen against the pre-rounding exact
			// value reconstructs with a small systematic bias (scale_exact-scale_bf16)*code baked
			// into every element. Rounding first keeps pack/unpack consistent with each other.
			sh, bh := f32ToBF16(scale), f32ToBF16(lo)
			scaleR, biasR := bf16ToF32(byte(sh), byte(sh>>8)), bf16ToF32(byte(bh), byte(bh>>8))
			sRow[g*bf16Size], sRow[g*bf16Size+1] = byte(sh), byte(sh>>8)
			bRow[g*bf16Size], bRow[g*bf16Size+1] = byte(bh), byte(bh>>8)
			for c := range groupSize {
				code := min(uint32(math.Round(float64((group[c]-biasR)/scaleR))), maxCode)
				setAffineCode(pRow, (g*groupSize+c)*bits, bits, code)
			}
		}
	}
	return packed, scales, biases
}

// setAffineCode is the exact write-side inverse of extractAffineCode (embed_lmhead_quant.go):
// it writes a bits-wide code at bit offset bitOff within p, LSB-first across byte boundaries,
// preserving any other bits already packed into the touched bytes.
func setAffineCode(p []byte, bitOff, bits int, code uint32) {
	for got := 0; got < bits; {
		bi := (bitOff + got) / 8
		off := (bitOff + got) % 8
		take := min(8-off, bits-got)
		mask := byte((1<<uint(take))-1) << uint(off)
		shifted := byte((code >> uint(got)) << uint(off))
		p[bi] = (p[bi] &^ mask) | (shifted & mask)
		got += take
	}
}

// TestPackAffineQuantRoundTripsThroughDequantizeAffineRowsF32 is the direct correctness check for
// this file's own pure-Go quantiser (packAffineQuant/quantWeightFixture/quantizeProj/quantGemma4Tensors
// — the metal-free replacement for what used to be a real mlxmetal.Quantize call): pack synthetic data,
// then feed the packed/scales/biases into the package's REAL PRODUCTION dequantizer
// (dequantizeAffineRowsF32, diffusion.go) and check the round-tripped values are within one
// quantisation step of the originals. This is the receipt that packAffineQuant is a genuine inverse of
// this package's own unpacking contract, not just self-consistent with itself.
func TestPackAffineQuantRoundTripsThroughDequantizeAffineRowsF32(t *testing.T) {
	for _, tt := range []struct {
		name                 string
		rows, cols, gs, bits int
	}{
		{"gs32_4bit", 5, 64, 32, 4},
		{"gs64_4bit", 3, 128, 64, 4},
		{"gs64_8bit", 4, 128, 64, 8},
		{"gs128_4bit", 2, 256, 128, 4},
	} {
		t.Run(tt.name, func(t *testing.T) {
			f := syntheticFloat32(tt.rows*tt.cols, 17)
			packed, scales, biases := packAffineQuant(f, tt.rows, tt.cols, tt.gs, tt.bits)

			got, err := dequantizeAffineRowsF32(packed, scales, biases, tt.rows, tt.cols, tt.gs, tt.bits)
			if err != nil {
				t.Fatalf("dequantizeAffineRowsF32: %v", err)
			}
			if len(got) != len(f) {
				t.Fatalf("dequantized length = %d, want %d", len(got), len(f))
			}
			maxCode := float32((uint32(1) << uint(tt.bits)) - 1)
			for r := 0; r < tt.rows; r++ {
				for g := 0; g < tt.cols/tt.gs; g++ {
					group := f[r*tt.cols+g*tt.gs : r*tt.cols+(g+1)*tt.gs]
					lo, hi := group[0], group[0]
					for _, v := range group[1:] {
						if v < lo {
							lo = v
						}
						if v > hi {
							hi = v
						}
					}
					scale := (hi - lo) / maxCode
					if scale == 0 {
						scale = 1
					}
					tol := scale/2 + 1e-4 // half a quantisation step, plus bf16 rounding slack
					for c := g * tt.gs; c < (g+1)*tt.gs; c++ {
						i := r*tt.cols + c
						if d := float32(math.Abs(float64(got[i] - f[i]))); d > tol {
							t.Fatalf("%s: row %d col %d: dequant %v vs original %v, diff %v > tol %v", tt.name, r, c, got[i], f[i], d, tol)
						}
					}
				}
			}
		})
	}
}

func decodeInputsFixture(tokens, dModel int) [][]byte {
	inputs := make([][]byte, tokens)
	for i := range inputs {
		inputs[i] = toBF16Bytes(syntheticFloat32(dModel, i+3))
	}
	return inputs
}

func forwardLayer(dModel, nHeads, nKV, headDim, dFF, salt int) DecodeLayerWeights {
	qDim, kvDim := nHeads*headDim, nKV*headDim
	mk := func(n, s int) []byte {
		f := make([]float32, n)
		for i := range f {
			f[i] = float32((i*s+7)%101-50) * 0.02
		}
		return toBF16Bytes(f)
	}
	return DecodeLayerWeights{
		AttnNormW: mk(dModel, salt+13), WQ: mk(qDim*dModel, salt+53),
		WK: mk(kvDim*dModel, salt+71), WV: mk(kvDim*dModel, salt+83), WO: mk(dModel*qDim, salt+17),
		MLPNormW: mk(dModel, salt+19), WGate: mk(dFF*dModel, salt+61),
		WUp: mk(dFF*dModel, salt+29), WDown: mk(dModel*dFF, salt+47),
	}
}

func decodeLayerFixture(dModel, nHeads, nKVHeads, headDim, dFF, salt int) DecodeLayerWeights {
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim
	return DecodeLayerWeights{
		AttnNormW: toBF16Bytes(syntheticFloat32(dModel, salt+1)),
		WQ:        toBF16Bytes(syntheticFloat32(qDim*dModel, salt+3)),
		WK:        toBF16Bytes(syntheticFloat32(kvDim*dModel, salt+5)),
		WV:        toBF16Bytes(syntheticFloat32(kvDim*dModel, salt+7)),
		WO:        toBF16Bytes(syntheticFloat32(dModel*qDim, salt+11)),
		MLPNormW:  toBF16Bytes(syntheticFloat32(dModel, salt+13)),
		WGate:     toBF16Bytes(syntheticFloat32(dFF*dModel, salt+17)),
		WUp:       toBF16Bytes(syntheticFloat32(dFF*dModel, salt+19)),
		WDown:     toBF16Bytes(syntheticFloat32(dModel*dFF, salt+23)),
	}
}

// quantW affine-quantises a synthetic weight in PURE GO (no cgo/metal) and returns it as a
// QuantWeight — the same bytes DecodeForwardQuant and its composed-ops reference both run on, so
// the comparison isolates the composition, not quantisation. Shared by buildQuantLayer here and by
// the metal_runtime-only quantRefForward (decode_forward_metal_test.go).
func quantW(t *testing.T, w []float32, outDim, inDim, gs, bits int) QuantWeight {
	t.Helper()
	if err := checkAffineQuantDims(inDim, gs, bits); err != nil {
		t.Fatalf("quantW: %v", err)
	}
	packed, scales, biases := packAffineQuant(w, outDim, inDim, gs, bits)
	return QuantWeight{Packed: packed, Scales: scales, Biases: biases, GroupSize: gs, Bits: bits}
}

// buildQuantLayer builds one QuantizedLayerWeights with synthetic, salt-varied
// weights — bf16 norms + 7 affine-quantised projections. Pure Go (no cgo/metal).
func buildQuantLayer(t *testing.T, dModel, nHeads, nKV, headDim, dFF, gs, bits, salt int) QuantizedLayerWeights {
	qDim, kvDim := nHeads*headDim, nKV*headDim
	mk := func(n, s int) []float32 {
		f := make([]float32, n)
		for i := range f {
			f[i] = float32((i*s+7)%101-50) * 0.02
		}
		return f
	}
	return QuantizedLayerWeights{
		AttnNormW: toBF16Bytes(mk(dModel, salt+13)),
		MLPNormW:  toBF16Bytes(mk(dModel, salt+19)),
		Q:         quantW(t, mk(qDim*dModel, salt+53), qDim, dModel, gs, bits),
		K:         quantW(t, mk(kvDim*dModel, salt+71), kvDim, dModel, gs, bits),
		V:         quantW(t, mk(kvDim*dModel, salt+83), kvDim, dModel, gs, bits),
		O:         quantW(t, mk(dModel*qDim, salt+17), dModel, qDim, gs, bits),
		Gate:      quantW(t, mk(dFF*dModel, salt+61), dFF, dModel, gs, bits),
		Up:        quantW(t, mk(dFF*dModel, salt+29), dFF, dModel, gs, bits),
		Down:      quantW(t, mk(dModel*dFF, salt+47), dModel, dFF, gs, bits),
		GroupSize: gs, Bits: bits,
	}
}

func quantizedLayerFixture(tb testing.TB, dModel, nHeads, nKVHeads, headDim, dFF, groupSize, bits, salt int) QuantizedLayerWeights {
	tb.Helper()
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim
	return QuantizedLayerWeights{
		AttnNormW: toBF16Bytes(syntheticFloat32(dModel, salt+1)),
		MLPNormW:  toBF16Bytes(syntheticFloat32(dModel, salt+13)),
		Q:         quantWeightFixture(tb, qDim, dModel, groupSize, bits, salt+3),
		K:         quantWeightFixture(tb, kvDim, dModel, groupSize, bits, salt+5),
		V:         quantWeightFixture(tb, kvDim, dModel, groupSize, bits, salt+7),
		O:         quantWeightFixture(tb, dModel, qDim, groupSize, bits, salt+11),
		Gate:      quantWeightFixture(tb, dFF, dModel, groupSize, bits, salt+17),
		Up:        quantWeightFixture(tb, dFF, dModel, groupSize, bits, salt+19),
		Down:      quantWeightFixture(tb, dModel, dFF, groupSize, bits, salt+23),
		GroupSize: groupSize,
		Bits:      bits,
	}
}

func archFixture(tb testing.TB, dModel, nHeads, nKVHeads, headDim, dFF, vocab, nLayers int) model.Arch {
	tb.Helper()
	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: nLayers, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKVHeads, HeadDim: headDim,
		VocabSize: vocab, RMSNormEps: 1e-5, RopeTheta: 10000,
	}
	arch, err := cfg.Arch()
	if err != nil {
		tb.Fatalf("Config.Arch: %v", err)
	}
	return arch
}

func gemma4BF16Fixture(tb testing.TB, dModel, nHeads, nKVHeads, headDim, dFF, vocab, nLayers int) (*BF16Model, model.Arch) {
	tb.Helper()
	arch := archFixture(tb, dModel, nHeads, nKVHeads, headDim, dFF, vocab, nLayers)
	layers := make([]DecodeLayerWeights, len(arch.Layer))
	for i := range layers {
		layers[i] = decodeLayerFixture(dModel, nHeads, nKVHeads, headDim, dFF, (i+1)*100)
	}
	g := &BF16Model{
		Layers:    layers,
		Embed:     toBF16Bytes(syntheticFloat32(vocab*dModel, 11)),
		FinalNorm: toBF16Bytes(syntheticFloat32(dModel, 7)),
	}
	g.LMHead, g.Tied = g.Embed, true
	return g, arch
}

func gemma4TensorFixture(arch model.Arch, withLMHead bool) map[string]safetensors.Tensor {
	tensors := map[string]safetensors.Tensor{}
	salt := 1
	mk := func(name string, elems int) {
		tensors[name] = safetensors.Tensor{
			Dtype: "BF16",
			Shape: []int{elems},
			Data:  toBF16Bytes(syntheticFloat32(elems, salt)),
		}
		salt++
	}
	dModel, headDim, dFF, vocab := arch.Hidden, arch.HeadDim, arch.FF, arch.Vocab
	qDim, kvDim := arch.Heads*headDim, arch.KVHeads*headDim
	mk("model.embed_tokens.weight", vocab*dModel)
	mk("model.norm.weight", dModel)
	if withLMHead {
		mk("lm_head.weight", vocab*dModel)
	}
	for i := range arch.Layer {
		p := core.Sprintf("model.layers.%d", i)
		mk(p+".input_layernorm.weight", dModel)
		mk(p+".self_attn.q_proj.weight", qDim*dModel)
		mk(p+".self_attn.k_proj.weight", kvDim*dModel)
		mk(p+".self_attn.v_proj.weight", kvDim*dModel)
		mk(p+".self_attn.o_proj.weight", dModel*qDim)
		mk(p+".self_attn.q_norm.weight", headDim)
		mk(p+".self_attn.k_norm.weight", headDim)
		mk(p+".post_attention_layernorm.weight", dModel)
		mk(p+".pre_feedforward_layernorm.weight", dModel)
		mk(p+".post_feedforward_layernorm.weight", dModel)
		mk(p+".mlp.gate_proj.weight", dFF*dModel)
		mk(p+".mlp.up_proj.weight", dFF*dModel)
		mk(p+".mlp.down_proj.weight", dModel*dFF)
	}
	return tensors
}

func mistralConfigFixture(tb testing.TB, dModel, nHeads, nKVHeads, headDim, dFF, vocab, nLayers int) (mistral.Config, model.Arch) {
	tb.Helper()
	cfg := mistral.Config{
		HiddenSize: dModel, NumHiddenLayers: nLayers, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKVHeads, HeadDim: headDim,
		VocabSize: vocab, RMSNormEps: 1e-6,
	}
	arch, err := cfg.Arch()
	if err != nil {
		tb.Fatalf("mistral Config.Arch: %v", err)
	}
	return cfg, arch
}

func mistralTensorFixture(tb testing.TB, dModel, nHeads, nKVHeads, headDim, dFF, vocab, nLayers int) map[string]safetensors.Tensor {
	tb.Helper()
	tensors := map[string]safetensors.Tensor{}
	salt := 1
	mk := func(name string, elems int) {
		tensors[name] = safetensors.Tensor{
			Dtype: "BF16",
			Shape: []int{elems},
			Data:  toBF16Bytes(syntheticFloat32(elems, salt)),
		}
		salt++
	}
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim
	mk("language_model.model.embed_tokens.weight", vocab*dModel)
	mk("language_model.model.norm.weight", dModel)
	for i := range nLayers {
		p := core.Sprintf("language_model.model.layers.%d", i)
		mk(p+".input_layernorm.weight", dModel)
		mk(p+".post_attention_layernorm.weight", dModel)
		mk(p+".self_attn.q_proj.weight", qDim*dModel)
		mk(p+".self_attn.k_proj.weight", kvDim*dModel)
		mk(p+".self_attn.v_proj.weight", kvDim*dModel)
		mk(p+".self_attn.o_proj.weight", dModel*qDim)
		mk(p+".mlp.gate_proj.weight", dFF*dModel)
		mk(p+".mlp.up_proj.weight", dFF*dModel)
		mk(p+".mlp.down_proj.weight", dModel*dFF)
	}
	mk("vision_tower.transformer.layers.0.attention.q_proj.weight", dModel*dModel)
	mk("multi_modal_projector.linear_1.weight", dModel*dModel)
	return tensors
}

func moeLayerWeightsFixture(numExperts, topK, dModel, dFF, expertDFF, salt int) MoELayerWeights {
	scale := make([]float32, numExperts)
	for i := range scale {
		scale[i] = 0.5 + float32(i)*0.125
	}
	return MoELayerWeights{
		NumExperts: numExperts, TopK: topK, ExpertDFF: expertDFF,
		PreFFNormW: toBF16Bytes(syntheticFloat32(dModel, salt+1)), PreFFNorm2W: toBF16Bytes(syntheticFloat32(dModel, salt+2)),
		PostFFNorm1W: toBF16Bytes(syntheticFloat32(dModel, salt+3)), PostFFNorm2W: toBF16Bytes(syntheticFloat32(dModel, salt+4)),
		PostFFNormW:       toBF16Bytes(syntheticFloat32(dModel, salt+5)),
		WGate:             toBF16Bytes(syntheticFloat32(dFF*dModel, salt+6)),
		WUp:               toBF16Bytes(syntheticFloat32(dFF*dModel, salt+7)),
		WDown:             toBF16Bytes(syntheticFloat32(dModel*dFF, salt+8)),
		RouterNormWScaled: toBF16Bytes(syntheticFloat32(dModel, salt+9)),
		RouterW:           toBF16Bytes(syntheticFloat32(numExperts*dModel, salt+10)),
		PerExpertScale:    toBF16Bytes(scale),
		ExpGateW:          toBF16Bytes(syntheticFloat32(numExperts*expertDFF*dModel, salt+11)),
		ExpUpW:            toBF16Bytes(syntheticFloat32(numExperts*expertDFF*dModel, salt+12)),
		ExpDownW:          toBF16Bytes(syntheticFloat32(numExperts*dModel*expertDFF, salt+13)),
	}
}

func toBF16Bytes(f []float32) []byte {
	b := make([]byte, len(f)*bf16Size)
	for i, v := range f {
		h := f32ToBF16(v)
		b[i*bf16Size] = byte(h)
		b[i*bf16Size+1] = byte(h >> 8)
	}
	return b
}

func bf16Floats(b []byte) []float32 {
	out := make([]float32, len(b)/bf16Size)
	for i := range out {
		out[i] = bf16ToF32(b[i*bf16Size], b[i*bf16Size+1])
	}
	return out
}

func assertFloat32Near(t *testing.T, name string, got, want []float32, tol float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length mismatch: got %d, want %d", name, len(got), len(want))
	}
	for i := range want {
		if d := float32(math.Abs(float64(got[i] - want[i]))); d > tol {
			t.Fatalf("%s[%d] = %v, want %v (diff %v > %v)", name, i, got[i], want[i], d, tol)
		}
	}
}

func eqBytes(t *testing.T, what string, got, want []byte) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len %d != %d", what, len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s: differ at byte %d: %#x vs %#x", what, i, got[i], want[i])
		}
	}
}

func cosineBF16(a, b []byte) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, na, nb float64
	for i := 0; i+1 < len(a); i += bf16Size {
		av := float64(bf16ToF32(a[i], a[i+1]))
		bv := float64(bf16ToF32(b[i], b[i+1]))
		dot += av * bv
		na += av * av
		nb += bv * bv
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}
