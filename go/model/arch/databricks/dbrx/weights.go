// SPDX-Licence-Identifier: EUPL-1.2

package dbrx

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// packedExpertsPrefix names the synthetic per-layer tensor packExperts writes. DBRX's real checkpoint
// (databricks/dbrx-instruct) ships every layer's routed experts as ONE FUSED tensor per role
// (transformer.blocks.{l}.ffn.experts.mlp.w1/v1/w2 — every expert's [ffn_hidden_size, d_model] matrix
// concatenated back-to-back in ONE tensor's bytes, no per-expert split at all). NormalizeWeights already
// decodes that fused layout into per-expert tensors for the Composed route
// (model.layers.{l}.mlp.experts.{e}.{gate,up,down}_proj.weight — see register.go); model.Assemble's generic
// MoE loader (assembleMoE) wants the OPPOSITE shape again — ONE tensor per role per layer — so packExperts
// re-concatenates NormalizeWeights' per-expert output back into that packed [numExperts·outDim, inDim]
// convention. This is the same byte layout mixtral.packExperts synthesises from Mixtral's genuinely-separate
// per-expert tensors (see that package's weights.go for the full convention writeup), and the same
// "experts_packed" naming discipline: it can never collide with a real checkpoint name, nor with
// NormalizeWeights' own "mlp.experts.<index>." per-expert names.
//
// Chaining through NormalizeWeights — rather than re-deriving the fused-tensor slice+transpose split a
// second time directly against the raw transformer.blocks.* tensors — means packExperts reuses the
// ALREADY-tested decode of DBRX's fused w1/v1/w2 layout (register_test.go); the only new logic here is the
// re-concatenation, identical in shape to mixtral's.
const packedExpertsPrefix = ".mlp.experts_packed"

// FactoryWeightNames returns the DBRX tensor layout for model.Assemble, once NormalizeConfig has run:
// NormalizeWeights already renames the fused checkpoint onto the standard llama-shaped attention names
// (self_attn.{q,k,v,o}_proj, input_layernorm, post_attention_layernorm as the pre-MoE norm — no gemma-style
// post-attention sandwich norm), so FactoryWeightNames only overrides the FFN/MoE roles: the dense MLP
// fields stay at StandardWeightNames' defaults but are never read (every DBRX layer is MoE — see
// Config.Arch), and the MoE block points at packExperts' synthesised packed tensors. DBRX has no shared
// expert, so SharedGate/Up/Down/SharedSigmoid stay "" (nil-safe — see model.MoEWeightNames).
func FactoryWeightNames() model.WeightNames {
	w := model.StandardWeightNames()
	w.MLPNorm = ".post_attention_layernorm.weight"
	w.PostAttnNorm = ""
	w.MoE = model.MoEWeightNames{
		PreFFNorm: ".post_attention_layernorm.weight",
		Router:    ".mlp.gate",
		ExpGate:   packedExpertsPrefix + ".gate_proj",
		ExpUp:     packedExpertsPrefix + ".up_proj",
		ExpDown:   packedExpertsPrefix + ".down_proj",
	}
	return w
}

// expertRoles is the fixed gate/up/down triple NormalizeWeights writes per expert, in a DETERMINISTIC
// order — map iteration is not, and packExperts must visit the same roles every call for reproducible
// output.
var expertRoles = []string{"gate_proj", "up_proj", "down_proj"}

// packExperts synthesises the packed per-layer expert tensors packedExpertsPrefix names, from the
// PER-EXPERT tensors NormalizeWeights produces (model.layers.{l}.mlp.experts.{e}.{role}.weight): for every
// layer in [0,numLayers) and every role, it concatenates the numExperts per-expert 2-D tensors row-major
// into one [numExperts·outDim, inDim] tensor — expert e occupies rows [e·outDim, (e+1)·outDim), so a
// downstream row-slice by expert index reproduces expert e's original matrix byte-for-byte. Every source
// tensor also stays in the returned map UNCHANGED (the packed tensors are ADDITIONS, never replacements):
// spec.Composed still reads the original per-expert names through its own NormalizeWeights call, untouched
// by this hook.
//
// Returns an error naming the exact tensor/layer/role that failed (a genuinely malformed or partial
// checkpoint, or a NormalizeWeights input that never carried the fused experts) so a direct caller gets a
// precise diagnosis; the NormalizeConfig hook that calls this at load time cannot propagate an error (its
// signature returns no error — see ArchSpec.NormalizeConfig), so it passes the normalised-but-unpacked
// tensors through on failure instead, and the resulting nil ExpGate/ExpUp/ExpDown surfaces at decode as an
// ordinary missing-weight condition rather than a silently wrong pack.
func packExperts(in map[string]safetensors.Tensor, numLayers, numExperts int) (map[string]safetensors.Tensor, error) {
	if numLayers <= 0 || numExperts <= 0 {
		return nil, core.NewError("dbrx.packExperts: n_layers and moe_num_experts must be > 0")
	}
	out := make(map[string]safetensors.Tensor, len(in)+numLayers*len(expertRoles)*3)
	for name, tensor := range in {
		out[name] = tensor
	}
	for layer := 0; layer < numLayers; layer++ {
		prefix := core.Sprintf("model.layers.%d.mlp.experts.", layer)
		for _, role := range expertRoles {
			weight, scales, biases, err := packExpertRole(in, prefix, role, numExperts)
			if err != nil {
				return nil, err
			}
			packedName := core.Sprintf("model.layers.%d%s.%s", layer, packedExpertsPrefix, role)
			out[packedName+".weight"] = *weight
			if scales != nil {
				out[packedName+".scales"] = *scales
				out[packedName+".biases"] = *biases
			}
		}
	}
	return out, nil
}

// packExpertRole concatenates one role's numExperts per-expert 2-D tensors (prefix+"{e}."+role+".weight")
// into one [numExperts·outDim, inDim] tensor, row-major — and, when the checkpoint ships this role
// QUANTISED (a sibling "<expert>.<role>.scales" tensor present), the matching packed .scales/.biases
// pair alongside it, stacked the identical way. Concatenating scales/biases along the same output-row
// axis as the weight reproduces exactly the shape model.Assemble's quant loader (LoadLinear/
// affineGeometry) already reads off a NATIVELY packed checkpoint — gemma4's switch_glu.*.weight/
// .scales/.biases ship as one [experts·outDim, inDim] weight beside a matching [experts·outDim,
// nGroups] scales/biases pair per role; this synthesises the identical convention from DBRX's
// per-expert triples (post-NormalizeWeights), so LoadLinear derives the SAME groupSize/bits from the
// packed shapes with zero engine-side change. A role with no ".scales" sibling packs .weight only
// (the existing bf16 behaviour, byte-identical). Every expert must share the first expert's shape
// and dtype for whichever components are present — NormalizeWeights' own decode of a real checkpoint
// always produces that; a mismatch means a malformed/partial checkpoint or a hand-built input,
// reported by the exact tensor name that broke the pattern.
func packExpertRole(in map[string]safetensors.Tensor, prefix, role string, numExperts int) (weight, scales, biases *safetensors.Tensor, err error) {
	weight, err = packExpertComponent(in, prefix, role, "weight", numExperts)
	if err != nil {
		return nil, nil, nil, err
	}
	if _, quantised := in[prefix+"0."+role+".scales"]; !quantised {
		return weight, nil, nil, nil
	}
	if scales, err = packExpertComponent(in, prefix, role, "scales", numExperts); err != nil {
		return nil, nil, nil, err
	}
	if biases, err = packExpertComponent(in, prefix, role, "biases", numExperts); err != nil {
		return nil, nil, nil, err
	}
	return weight, scales, biases, nil
}

// packExpertComponent concatenates one role's numExperts per-expert 2-D tensors
// (prefix+"{e}."+role+"."+component) into one [numExperts·outDim, lastDim] tensor, row-major — the
// shape-check-and-stack shared by the weight, scales and biases components. Every expert must share
// the first expert's shape and dtype; a mismatch means a malformed/partial checkpoint or a hand-built
// input, reported by the exact tensor name that broke the pattern.
func packExpertComponent(in map[string]safetensors.Tensor, prefix, role, component string, numExperts int) (*safetensors.Tensor, error) {
	firstName := prefix + "0." + role + "." + component
	first, ok := in[firstName]
	if !ok {
		return nil, core.NewError("dbrx.packExperts: " + firstName + " absent")
	}
	if len(first.Shape) != 2 {
		return nil, core.NewError("dbrx.packExperts: " + firstName + " is not 2-D")
	}
	outDim, lastDim := first.Shape[0], first.Shape[1]
	rowBytes := len(first.Data)
	data := make([]byte, 0, rowBytes*numExperts)
	for e := 0; e < numExperts; e++ {
		name := core.Sprintf("%s%d.%s.%s", prefix, e, role, component)
		t, ok := in[name]
		if !ok {
			return nil, core.NewError("dbrx.packExperts: " + name + " absent")
		}
		if len(t.Shape) != 2 || t.Shape[0] != outDim || t.Shape[1] != lastDim {
			return nil, core.NewError("dbrx.packExperts: " + name + " shape mismatch")
		}
		if t.Dtype != first.Dtype {
			return nil, core.NewError("dbrx.packExperts: " + name + " dtype mismatch")
		}
		if len(t.Data) != rowBytes {
			return nil, core.NewError("dbrx.packExperts: " + name + " byte length mismatch")
		}
		data = append(data, t.Data...)
	}
	return &safetensors.Tensor{Dtype: first.Dtype, Shape: []int{outDim * numExperts, lastDim}, Data: data}, nil
}
