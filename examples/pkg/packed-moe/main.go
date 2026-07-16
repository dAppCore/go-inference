// SPDX-Licence-Identifier: EUPL-1.2

// The packed-expert MoE path: a quantised Mixture-of-Experts checkpoint (Qwen 3.6 — model_type
// qwen3_5_moe) keeps its routed + shared experts PACKED (mlx-affine codes + scales/biases) all
// the way to the matvec — buildMoE used to widen every expert to f32 at load regardless, an ~8x
// blow-up on a grouped checkpoint's dominant tensor class. This example is the lib-level
// acceptance for that fix: build a small synthetic quantised MoE checkpoint in memory (the same
// fixture shape go/model/composed/moe_quant_test.go proves against), load it through the composed
// loader TWO ways — packed, and dense-over-the-exact-dequantised-values — and compare a forward.
//
//	go run ./pkg/packed-moe
package main

import (
	"fmt"
	"math"
	"os"
	"strconv"

	"dappco.re/go/inference/model/composed"
	"dappco.re/go/inference/model/quant/mlxaffine"
	"dappco.re/go/inference/model/safetensors"
)

func main() {
	// Receipt 1 — storage: one routed expert's gate+up+down at a realistic MoE width (D=512,
	// FF=1408), 4-bit codes in groups of 64 — the packed:dequantised-f32 ratio a grouped
	// checkpoint's expert tensors carry. Standalone (no loader): mlxaffine.QuantizeTensor is the
	// same call the loader makes per projection at load time.
	const rD, rFF, rBits, rGS = 512, 1408, 4, 64
	var packedBytes, dequantBytes int
	for i, shape := range [][2]int{{rFF, rD}, {rFF, rD}, {rD, rFF}} { // gate_proj, up_proj, down_proj
		outDim, inDim := shape[0], shape[1]
		w := syn(outDim*inDim, i+1)
		packed, scales, biases, err := mlxaffine.QuantizeTensor(w, outDim, inDim, rBits, rGS)
		if err != nil {
			fmt.Fprintln(os.Stderr, "quantise:", err)
			os.Exit(1)
		}
		packedBytes += len(packed) + len(scales) + len(biases)
		dequantBytes += outDim * inDim * 4 // f32
	}
	ratio := float64(dequantBytes) / float64(packedBytes)
	fmt.Printf("expert storage — one gate+up+down at D=%d FF=%d (%d-bit, groups of %d): packed %d bytes, dequantised-f32 %d bytes, ratio %.2fx\n",
		rD, rFF, rBits, rGS, packedBytes, dequantBytes, ratio)

	// Receipt 2 — correctness: a synthetic 1-layer Qwen-3.6-shaped checkpoint (gated-delta mixer +
	// a 6-routed+1-shared-expert MoE FFN), built two ways from the SAME syn(seed) source weights —
	// "packed" quantises the expert projections to mlx-affine + .scales/.biases siblings; "dense"
	// carries the EXACT dequantised values of those same packed codes (F32, no further rounding —
	// see expertProj) — loaded through the SAME composed.LoadComposed and forwarded. Both paths
	// trace back to the identical quantised numbers, so the gap measured is purely the packed
	// matvec seam's compute-rounding tier, not quantisation error (the wiring-correctness gate
	// TestLoadComposedMoEQuantised and TestMoEMLP_Forward_PackedMatchesDequantised pin at 1e-4).
	tsDense, configDense, err := buildMoECheckpoint(false)
	if err != nil {
		fmt.Fprintln(os.Stderr, "build dense checkpoint:", err)
		os.Exit(1)
	}
	mDense, err := composed.LoadComposed(tsDense, configDense)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load dense:", err)
		os.Exit(1)
	}
	if mDense.Quantised {
		fmt.Fprintln(os.Stderr, "dense checkpoint loaded Quantised=true, want false")
		os.Exit(1)
	}

	tsQuant, configQuant, err := buildMoECheckpoint(true)
	if err != nil {
		fmt.Fprintln(os.Stderr, "build packed checkpoint:", err)
		os.Exit(1)
	}
	mQuant, err := composed.LoadComposed(tsQuant, configQuant)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load packed:", err)
		os.Exit(1)
	}
	if !mQuant.Quantised {
		fmt.Fprintln(os.Stderr, "packed checkpoint loaded Quantised=false, want true")
		os.Exit(1)
	}
	moe, ok := mQuant.Layers[0].MLP.(*composed.MoEMLP)
	if !ok {
		fmt.Fprintf(os.Stderr, "layer 0 FFN is %T, want *composed.MoEMLP\n", mQuant.Layers[0].MLP)
		os.Exit(1)
	}
	for i := range moe.Experts {
		if moe.Experts[i].GateQ == nil || moe.Experts[i].UpQ == nil || moe.Experts[i].DownQ == nil {
			fmt.Fprintf(os.Stderr, "expert %d: not packed — the loader widened it to f32 at load\n", i)
			os.Exit(1)
		}
	}
	if moe.Shared == nil || moe.Shared.GateQ == nil {
		fmt.Fprintln(os.Stderr, "shared expert: not packed")
		os.Exit(1)
	}

	tokens := []int32{1, 5, 3, 0, 7}
	hD, err := composed.NewSession(mDense).Forward(tokens)
	if err != nil {
		fmt.Fprintln(os.Stderr, "dense forward:", err)
		os.Exit(1)
	}
	hQ, err := composed.NewSession(mQuant).Forward(tokens)
	if err != nil {
		fmt.Fprintln(os.Stderr, "packed forward:", err)
		os.Exit(1)
	}
	maxRel := relError(hQ, hD)
	fmt.Printf("packed MoE forward vs dense-reference forward (%d tokens, D=%d, %d routed experts top-%d + shared): max relative error %.3e\n",
		len(tokens), mQuant.D, len(moe.Experts), moe.TopK, maxRel)
}

// buildMoECheckpoint returns a synthetic 1-layer Qwen-3.6-shaped checkpoint (a linear-attention
// gated-delta mixer feeding a 6-routed+1-shared-expert MoE FFN) and its config.json. Every weight
// is generated from syn(n, seed) — the SAME deterministic values whether quantise is false (plain
// bf16 tensors throughout) or true (the routed + shared expert gate/up/down projections are
// additionally packed to mlx-affine U32 codes + .scales/.biases siblings, and the config carries a
// "quantization" block) — so the dense and packed checkpoints this function returns are, modulo
// quantisation error, the SAME model. Mirrors
// go/model/composed/moe_quant_test.go's TestLoadComposedMoEQuantised fixture.
func buildMoECheckpoint(quantise bool) (map[string]safetensors.Tensor, []byte, error) {
	const D, vocab = 8, 32
	const VH, HD, convDim, K, vDim = 4, 8, 64, 4, 32
	const moeFF, nE, sharedFF = 16, 6, 24
	const bits, gs = 4, 8

	ts := map[string]safetensors.Tensor{
		"model.embed_tokens.weight": bf16T(syn(vocab*D, 1), vocab, D),
		"model.norm.weight":         bf16T(syn(D, 2), D),
		"lm_head.weight":            bf16T(syn(vocab*D, 3), vocab, D),
	}
	lp := "model.layers.0."
	ts[lp+"input_layernorm.weight"] = bf16T(syn(D, 1), D)
	ts[lp+"post_attention_layernorm.weight"] = bf16T(syn(D, 2), D)
	gp := lp + "linear_attn."
	ts[gp+"in_proj_qkv.weight"] = bf16T(syn(convDim*D, 20), convDim, D)
	ts[gp+"conv1d.weight"] = bf16T(syn(convDim*K, 21), convDim, 1, K)
	ts[gp+"conv1d.bias"] = bf16T(syn(convDim, 22), convDim)
	ts[gp+"in_proj_a.weight"] = bf16T(syn(VH*D, 23), VH, D)
	ts[gp+"A_log"] = bf16T(syn(VH, 24), VH)
	ts[gp+"dt_bias"] = bf16T(syn(VH, 25), VH)
	ts[gp+"in_proj_b.weight"] = bf16T(syn(VH*D, 26), VH, D)
	ts[gp+"in_proj_z.weight"] = bf16T(syn(vDim*D, 27), vDim, D)
	ts[gp+"norm.weight"] = bf16T(syn(HD, 28), HD)
	ts[gp+"out_proj.weight"] = bf16T(syn(D*vDim, 29), D, vDim)
	mp := lp + "mlp."
	ts[mp+"gate.weight"] = bf16T(syn(nE*D, 30), nE, D)
	for e := range nE {
		ep := mp + "experts." + strconv.Itoa(e) + "."
		gateV := syn(moeFF*D, e*5+40)
		upV := syn(moeFF*D, e*5+41)
		downV := syn(D*moeFF, e*5+42)
		if err := expertProj(ts, ep+"gate_proj.weight", gateV, moeFF, D, bits, gs, quantise); err != nil {
			return nil, nil, err
		}
		if err := expertProj(ts, ep+"up_proj.weight", upV, moeFF, D, bits, gs, quantise); err != nil {
			return nil, nil, err
		}
		if err := expertProj(ts, ep+"down_proj.weight", downV, D, moeFF, bits, gs, quantise); err != nil {
			return nil, nil, err
		}
	}
	sp := mp + "shared_expert."
	sGateV := syn(sharedFF*D, 90)
	sUpV := syn(sharedFF*D, 91)
	sDownV := syn(D*sharedFF, 92)
	if err := expertProj(ts, sp+"gate_proj.weight", sGateV, sharedFF, D, bits, gs, quantise); err != nil {
		return nil, nil, err
	}
	if err := expertProj(ts, sp+"up_proj.weight", sUpV, sharedFF, D, bits, gs, quantise); err != nil {
		return nil, nil, err
	}
	if err := expertProj(ts, sp+"down_proj.weight", sDownV, D, sharedFF, bits, gs, quantise); err != nil {
		return nil, nil, err
	}

	config := `{"hidden_size":8,"num_hidden_layers":1,"intermediate_size":16,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,"vocab_size":32,"rms_norm_eps":1e-5,"num_experts_per_tok":2,"rope_theta":1000000,"partial_rotary_factor":0.5,"layer_types":["linear_attention"]`
	if quantise {
		config += `,"quantization":{"group_size":8,"bits":4}`
	}
	config += `}`
	return ts, []byte(config), nil
}

// expertProj quantises vals ([outDim*inDim] row-major) to its mlx-affine packed form. When packed
// is true, ts[name] becomes the packed U32 codes plus its .scales/.biases siblings — the on-disk
// shape a real quantised checkpoint carries, and what the loader's buildMoE/proj resolution looks
// for. When packed is false, ts[name] instead becomes the EXACT dequantised values (F32 dtype, no
// further rounding) — the reference this example's forward comparison is against, so the measured
// gap is purely the packed matvec seam's compute-rounding tier (matNTQuant rounds to f32 at each
// matmul boundary; the dense path stays f64 throughout — see moe.go's swigluExpertQuantInto doc),
// not the quantisation error a bf16-of-the-original-weights reference would additionally carry.
func expertProj(ts map[string]safetensors.Tensor, name string, vals []float32, outDim, inDim, bits, gs int, packed bool) error {
	p, scales, biases, err := mlxaffine.QuantizeTensor(vals, outDim, inDim, bits, gs)
	if err != nil {
		return err
	}
	if packed {
		ts[name] = safetensors.Tensor{Dtype: "U32", Shape: []int{outDim, mlxaffine.PackedWords(inDim, bits)}, Data: p}
		base := name[:len(name)-len(".weight")]
		ts[base+".scales"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{outDim, inDim / gs}, Data: scales}
		ts[base+".biases"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{outDim, inDim / gs}, Data: biases}
		return nil
	}
	deq, err := mlxaffine.DequantizeTensor(p, scales, biases, outDim, inDim, bits, gs)
	if err != nil {
		return err
	}
	ts[name] = f32T(deq, outDim, inDim)
	return nil
}

// syn generates n deterministic pseudo-random f32 values from seed — a synthetic weight tensor
// with no I/O, the same fixture shape go/model/composed's tests use throughout.
func syn(n, seed int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32((i*seed+7)%101-50) * 0.02
	}
	return out
}

// bf16T rounds vals to bf16 (round-to-nearest-even, matching the mlx/safetensors convention) and
// wraps them as a safetensors.Tensor of the given shape — the checkpoint's on-disk dtype for every
// unquantised weight.
func bf16T(vals []float32, shape ...int) safetensors.Tensor {
	data := make([]byte, len(vals)*2)
	for i, v := range vals {
		bits := math.Float32bits(v)
		r := uint16((bits + 0x7fff + ((bits >> 16) & 1)) >> 16)
		data[2*i], data[2*i+1] = byte(r), byte(r>>8)
	}
	return safetensors.Tensor{Dtype: "BF16", Shape: shape, Data: data}
}

// f32T wraps vals as an exact F32-dtype safetensors.Tensor (4-byte IEEE-754, no rounding) — used
// for expertProj's dense/dequantised reference so the comparison never carries a SECOND rounding
// step (a bf16 re-encode) on top of the dequantisation itself.
func f32T(vals []float32, shape ...int) safetensors.Tensor {
	data := make([]byte, len(vals)*4)
	for i, v := range vals {
		bits := math.Float32bits(v)
		data[4*i], data[4*i+1], data[4*i+2], data[4*i+3] = byte(bits), byte(bits>>8), byte(bits>>16), byte(bits>>24)
	}
	return safetensors.Tensor{Dtype: "F32", Shape: shape, Data: data}
}

// relError is the max relative error between got and want (rel = |got-want| / (1+|want|)) — the
// tolerance basis go/model/composed's packed-vs-dense parity tests use throughout.
func relError(got, want []float32) float64 {
	var maxRel float64
	for i := range want {
		rel := math.Abs(float64(got[i]-want[i])) / (1 + math.Abs(float64(want[i])))
		if rel > maxRel {
			maxRel = rel
		}
	}
	return maxRel
}
