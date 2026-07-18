// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

// assemble_gated_delta.go builds a MixerGatedDelta layer through the reactive factory — the loader half
// of teaching Assemble the named mixer kind (#18), replacing model/composed's buildGatedDelta. The five
// projections (in_proj_qkv/a/b/z + out_proj) load through the factory's own LoadLinear (quant-or-bf16
// universal, the same path gemma's MoE experts use); the small recurrence tensors (conv/A_log/norm/
// dt_bias) stay host f32, unquantised, so the recurrent state math is exact. Geometry is resolved from
// the weight SHAPES (the don't-guess rule), byte-parity with the composed loader it retires.

// tensorFloat32 widens a checkpoint tensor (bf16/f16/f32) to []float32.
func tensorFloat32(t safetensors.Tensor) ([]float32, error) {
	switch t.Dtype {
	case "BF16", "bfloat16":
		out := make([]float32, len(t.Data)/2)
		for i := range out {
			b := uint16(t.Data[2*i]) | uint16(t.Data[2*i+1])<<8
			out[i] = math.Float32frombits(uint32(b) << 16)
		}
		return out, nil
	case "F32", "float32":
		out := make([]float32, len(t.Data)/4)
		for i := range out {
			out[i] = math.Float32frombits(uint32(t.Data[4*i]) | uint32(t.Data[4*i+1])<<8 | uint32(t.Data[4*i+2])<<16 | uint32(t.Data[4*i+3])<<24)
		}
		return out, nil
	}
	return nil, core.NewError("model.tensorFloat32: unsupported dtype " + t.Dtype)
}

// tensorFloat32Opt is tensorFloat32 for an OPTIONAL tensor: nil (no error) when absent (conv1d.bias and
// dt_bias are optional per checkpoint).
func tensorFloat32Opt(t map[string]safetensors.Tensor, name string) []float32 {
	x, ok := t[name]
	if !ok {
		return nil
	}
	f, err := tensorFloat32(x)
	if err != nil {
		return nil
	}
	return f
}

// gdProjection loads one gated-delta projection through LoadLinear and splits the universal *Linear onto
// the GatedDeltaWeights quant XOR bf16 form (mutually exclusive, exactly as composed's proj set one of
// the three). A quant checkpoint yields the packed QuantWeight; a dense bf16 checkpoint the BF16Weight.
func gdProjection(t map[string]safetensors.Tensor, prefix string, inDim int, kind string) (*QuantWeight, *BF16Weight, error) {
	lin := LoadLinear(t, prefix, inDim, kind)
	if lin == nil {
		return nil, nil, core.NewError("model.assembleGatedDelta: missing " + prefix + ".weight")
	}
	if lin.Bits > 0 {
		return &QuantWeight{Packed: lin.Weight, Scales: lin.Scales, Biases: lin.Biases, GroupSize: lin.GroupSize, Bits: lin.Bits, OutDim: lin.OutDim, InDim: lin.InDim}, nil, nil
	}
	return nil, &BF16Weight{Data: lin.Weight, OutDim: lin.OutDim, InDim: lin.InDim}, nil
}

// assembleGatedDelta builds one gated-delta layer's weights + geometry from the checkpoint under prefix
// (e.g. "…layers.N.linear_attn."). d is the model hidden. Geometry: ValueHeads = len(A_log),
// HeadDim = len(norm), convDim/K from conv1d.weight, qDim = (convDim−vDim)/2, KeyHeads = qDim/HeadDim.
func assembleGatedDelta(t map[string]safetensors.Tensor, prefix string, d int, kind string) (*GatedDeltaWeights, GatedDeltaConfig, error) {
	aLogT, ok := t[prefix+"A_log"]
	if !ok || len(aLogT.Shape) != 1 {
		return nil, GatedDeltaConfig{}, core.NewError("model.assembleGatedDelta: missing/!1D " + prefix + "A_log")
	}
	normT, ok := t[prefix+"norm.weight"]
	if !ok || len(normT.Shape) != 1 {
		return nil, GatedDeltaConfig{}, core.NewError("model.assembleGatedDelta: missing/!1D " + prefix + "norm.weight")
	}
	convT, ok := t[prefix+"conv1d.weight"]
	if !ok || len(convT.Shape) == 0 {
		return nil, GatedDeltaConfig{}, core.NewError("model.assembleGatedDelta: missing " + prefix + "conv1d.weight")
	}
	valueHeads, headDim, convDim := aLogT.Shape[0], normT.Shape[0], convT.Shape[0]
	convK := convT.Shape[len(convT.Shape)-1]
	if len(convT.Shape) == 3 && convK == 1 {
		// mlx packs the depthwise conv channel-last ([convDim, K, 1]); torch packs [convDim, 1, K]. The
		// flat bytes are identical (the 1-dim contributes no stride) — only K's slot moves.
		convK = convT.Shape[1]
	}
	vDim := valueHeads * headDim
	if (convDim-vDim)%2 != 0 {
		return nil, GatedDeltaConfig{}, core.NewError(core.Sprintf("model.assembleGatedDelta: convDim %d − vDim %d not even", convDim, vDim))
	}
	qDim := (convDim - vDim) / 2
	if headDim == 0 || qDim%headDim != 0 {
		return nil, GatedDeltaConfig{}, core.NewError("model.assembleGatedDelta: qDim not divisible by headDim")
	}
	keyHeads := qDim / headDim

	qkvQ, qkvB, err := gdProjection(t, prefix+"in_proj_qkv", d, kind)
	if err != nil {
		return nil, GatedDeltaConfig{}, err
	}
	inAQ, inAB, err := gdProjection(t, prefix+"in_proj_a", d, kind)
	if err != nil {
		return nil, GatedDeltaConfig{}, err
	}
	inBQ, inBB, err := gdProjection(t, prefix+"in_proj_b", d, kind)
	if err != nil {
		return nil, GatedDeltaConfig{}, err
	}
	inZQ, inZB, err := gdProjection(t, prefix+"in_proj_z", d, kind)
	if err != nil {
		return nil, GatedDeltaConfig{}, err
	}
	outPQ, outPB, err := gdProjection(t, prefix+"out_proj", vDim, kind)
	if err != nil {
		return nil, GatedDeltaConfig{}, err
	}
	convW, err := tensorFloat32(convT)
	if err != nil {
		return nil, GatedDeltaConfig{}, err
	}
	aLog, err := tensorFloat32(aLogT)
	if err != nil {
		return nil, GatedDeltaConfig{}, err
	}
	norm, err := tensorFloat32(normT)
	if err != nil {
		return nil, GatedDeltaConfig{}, err
	}

	w := &GatedDeltaWeights{
		InProjQKVQ: qkvQ, InProjQKVB: qkvB, ConvWeight: convW, ConvBias: tensorFloat32Opt(t, prefix+"conv1d.bias"),
		InProjAQ: inAQ, InProjAB: inAB, ALog: aLog, DtBias: tensorFloat32Opt(t, prefix+"dt_bias"),
		InProjBQ: inBQ, InProjBB: inBB, InProjZQ: inZQ, InProjZB: inZB,
		Norm: norm, OutProjQ: outPQ, OutProjB: outPB,
	}
	cfg := GatedDeltaConfig{KeyHeads: keyHeads, ValueHeads: valueHeads, HeadDim: headDim, ConvKernel: convK, Eps: 1e-6}
	return w, cfg, nil
}
