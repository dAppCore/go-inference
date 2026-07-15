// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// gemma4Tensors builds the full named bf16 tensor set for arch, each tensor filled with a
// distinct byte (recorded in fills) so a wrong field assignment is detectable. withLMHead
// adds a separate lm_head.weight (untied); otherwise the model ties to the embedding. Shared
// fixture for the bf16 directory/session tests (the hand-coded AssembleGemma4BF16 it used to
// gate is gone — pkg/model/gemma4.Assemble owns the name mapping now, with its own tests).
func gemma4Tensors(arch model.Arch, withLMHead bool) (map[string]safetensors.Tensor, map[string]byte) {
	ts := map[string]safetensors.Tensor{}
	fills := map[string]byte{}
	next := byte(1)
	mk := func(name string, shape ...int) {
		elems := 1
		for _, dim := range shape {
			elems *= dim
		}
		data := make([]byte, elems*bf16Size)
		for j := range data {
			data[j] = next
		}
		ts[name] = safetensors.Tensor{Dtype: "BF16", Shape: shape, Data: data}
		fills[name] = next
		next++
	}
	dModel, dFF, vocab := arch.Hidden, arch.FF, arch.Vocab
	mk("model.embed_tokens.weight", vocab, dModel)
	mk("model.norm.weight", dModel)
	if withLMHead {
		mk("lm_head.weight", vocab, dModel)
	}
	for i := range arch.Layer {
		p := core.Sprintf("model.layers.%d", i)
		lhd := headDimOf(arch.Layer[i], arch.HeadDim)
		lkv := kvHeadsOf(arch.Layer[i], arch.KVHeads)
		qDim, kvDim := arch.Heads*lhd, lkv*lhd
		mk(p+".input_layernorm.weight", dModel)
		mk(p+".self_attn.q_proj.weight", qDim, dModel)
		mk(p+".self_attn.k_proj.weight", kvDim, dModel)
		mk(p+".self_attn.v_proj.weight", kvDim, dModel)
		mk(p+".self_attn.o_proj.weight", dModel, qDim)
		mk(p+".self_attn.q_norm.weight", lhd)
		mk(p+".self_attn.k_norm.weight", lhd)
		mk(p+".post_attention_layernorm.weight", dModel)
		mk(p+".pre_feedforward_layernorm.weight", dModel)
		mk(p+".post_feedforward_layernorm.weight", dModel)
		mk(p+".mlp.gate_proj.weight", dFF, dModel)
		mk(p+".mlp.up_proj.weight", dFF, dModel)
		mk(p+".mlp.down_proj.weight", dModel, dFF)
	}
	return ts, fills
}

// g4Assemble runs the engine's generic assembler with gemma4's weight layout — gemma4 no longer owns an
// Assemble (model.Assemble does), so the native tests that build a gemma4 LoadedModel from a synthetic
// tensor set go through this.
func g4Assemble(ts map[string]safetensors.Tensor, arch model.Arch) (*model.LoadedModel, error) {
	return model.Assemble(ts, arch, model.StandardWeightNames())
}
