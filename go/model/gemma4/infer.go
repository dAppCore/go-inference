// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/safetensors"
)

// infer.go — gemma4's weight-shape inference. The SELECTION is gemma4's (which attention-typed layer,
// the gemma4 weight names); the dim-from-shape READ is the engine's (pkg/model: WeightAny / InferHeadDim
// / InferOutFeaturesPerN), so other arches reuse the engine with their own names + patterns rather than
// re-rolling it.

// inferGemma4HeadDim reads the head dim of the first `target`-attention layer from its q_proj rows.
// gemma4 carries distinct head dims for sliding vs full (global) layers, so the caller resolves each by
// passing the matching attention type.
func inferGemma4HeadDim(weights map[string]safetensors.Tensor, layerTypes []string, numAttentionHeads int, target string) int {
	for i, layerType := range layerTypes {
		if layerType != target {
			continue
		}
		if hd := model.InferHeadDim(weights, core.Sprintf("model.layers.%d.self_attn.q_proj.weight", i), numAttentionHeads); hd > 0 {
			return hd
		}
	}
	return 0
}

// inferGemma4PerLayerInputSize reads the gemma4 per-layer-input width — the per-layer projection's
// out-features ÷ layer count (the E2B/E4B PLE tower).
func inferGemma4PerLayerInputSize(weights map[string]safetensors.Tensor, numHiddenLayers int) int {
	return model.InferOutFeaturesPerN(weights, "model.per_layer_model_projection.weight", numHiddenLayers)
}

// InferFromWeights resolves, in place, the dims gemma4 reads from the weight SHAPES rather than the
// config (the don't-guess rule): per-layer head dims (sliding vs full) from the q_proj rows, vocab from
// the embedding rows, the PLE width from the per-layer projection — with the hidden/heads fallback only
// as a last resort, and the PLE tower disabled unless complete. The engine loader (model.Load) calls
// this between Parse and Arch(); it satisfies model.ArchConfig. Relocated from the former gemma4.Load.
func (c *Gemma4TextConfig) InferFromWeights(weights map[string]safetensors.Tensor) {
	if hd := inferGemma4HeadDim(weights, c.LayerTypes, int(c.NumAttentionHeads), "sliding_attention"); hd > 0 {
		c.HeadDim = int32(hd)
	}
	if hd := inferGemma4HeadDim(weights, c.LayerTypes, int(c.NumAttentionHeads), "full_attention"); hd > 0 {
		c.GlobalHeadDim = int32(hd)
	}
	if c.HeadDim == 0 && c.HiddenSize > 0 && c.NumAttentionHeads > 0 {
		c.HeadDim = c.HiddenSize / c.NumAttentionHeads
	}
	if c.VocabSize == 0 {
		if w, ok := model.WeightAny(weights, "model.embed_tokens.weight", "model.embed_tokens"); ok && len(w.Shape) > 0 && w.Shape[0] > 0 {
			c.VocabSize = int32(w.Shape[0])
		}
	}
	if c.VocabSizePerLayerInput == 0 {
		c.VocabSizePerLayerInput = c.VocabSize
	}
	if pl := inferGemma4PerLayerInputSize(weights, int(c.NumHiddenLayers)); pl > 0 {
		c.HiddenSizePerLayerInput = int32(pl)
	}
	if c.HiddenSizePerLayerInput > 0 { // the PLE tower must be complete, else disable it
		_, e1 := model.WeightAny(weights, "model.embed_tokens_per_layer.weight")
		_, e2 := model.WeightAny(weights, "model.per_layer_model_projection.weight")
		_, e3 := model.WeightAny(weights, "model.per_layer_projection_norm.weight")
		if !e1 || !e2 || !e3 {
			c.HiddenSizePerLayerInput = 0
		}
	}
	gemma4FinaliseEmbeddingScales(c)
}
