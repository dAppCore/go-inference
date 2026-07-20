// SPDX-Licence-Identifier: EUPL-1.2

package mixtral

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// packedExpertsPrefix names the synthetic per-layer tensor packExperts writes. Real Mixtral checkpoints
// (mistralai/Mixtral-8x7B-v0.1) ship each expert as its OWN 2-D tensor
// (.block_sparse_moe.experts.{i}.w1/w2/w3.weight — see NormalizeWeights), but model.Assemble's generic MoE
// loader (assembleMoE) looks up ONE tensor per role per layer — the "every expert lives in one tensor"
// convention gpt_oss and qwen3_5_moe's real checkpoints already ship natively (see gptoss.WeightNames:
// "a 3-D-shaped safetensors tensor ([experts, outDim, inDim], row-major) is byte-identical to the
// 2-D-flattened [experts*outDim, inDim] shape the engine already assumes"). Mixtral does not ship that
// way, so packExperts SYNTHESISES it at load time: N per-expert matrices concatenated row-major into one
// [N·outDim, inDim] tensor — the same byte layout gpt_oss's native 3-D tensor already has. Named
// "experts_packed" (not "switch_mlp"/"switch_glu", which name a checkpoint's OWN native packed tensor) so
// it can never collide with a real checkpoint name.
const packedExpertsPrefix = ".block_sparse_moe.experts_packed"

// FactoryWeightNames returns the Mixtral tensor layout for model.Assemble: StandardWeightNames' attention
// projections as-is (Mixtral is llama-shaped there — self_attn.{q,k,v,o}_proj, input_layernorm, no
// QK-norm, no partial rotary), the llama/mistral 2-norm FFN override (post_attention_layernorm is the
// pre-MoE norm; there is no gemma-style post-attention sandwich norm — mirrors gptoss.WeightNames exactly,
// same architecture family), and the MoE block pointed at packExperts' synthesised packed tensors. Mixtral
// has no shared expert, so SharedGate/Up/Down/SharedSigmoid stay "" (nil-safe — see MoEWeightNames).
//
// Named distinctly from the package's existing WeightNames() (which returns the composed-path Names
// alias templates — Router/ExpertGate/ExpertDown/ExpertUp, consumed by NormalizeWeights) so the two
// weight-name conventions — one per Mixtral's two load routes — stay unambiguous at the call site.
func FactoryWeightNames() model.WeightNames {
	w := model.StandardWeightNames()
	w.MLPNorm = ".post_attention_layernorm.weight"
	w.PostAttnNorm = ""
	w.MoE = model.MoEWeightNames{
		PreFFNorm: ".post_attention_layernorm.weight",
		Router:    ".block_sparse_moe.gate",
		ExpGate:   packedExpertsPrefix + ".gate_proj",
		ExpUp:     packedExpertsPrefix + ".up_proj",
		ExpDown:   packedExpertsPrefix + ".down_proj",
	}
	return w
}

// expertRole is one routed-expert projection: the real per-expert HF suffix (w1/w2/w3) and the packed
// tensor's role suffix (matching FactoryWeightNames' MoE.Exp{Gate,Up,Down}).
type expertRole struct{ hfSuffix, packedSuffix string }

// expertRoles is the fixed gate/up/down triple in a DETERMINISTIC order — map iteration is not, and
// packExperts must visit the same roles every call for reproducible output.
var expertRoles = []expertRole{{"w1", "gate_proj"}, {"w3", "up_proj"}, {"w2", "down_proj"}}

// packExperts synthesises the packed per-layer expert tensors packedExpertsPrefix names: for every layer
// in [0,numLayers) and every role (gate/up/down), it concatenates the numExperts per-expert 2-D tensors
// (.block_sparse_moe.experts.{e}.{w1,w2,w3}.weight) row-major into one [numExperts·outDim, inDim] tensor —
// expert e occupies rows [e·outDim, (e+1)·outDim), so a downstream row-slice by expert index reproduces
// expert e's original matrix byte-for-byte. Every source tensor also stays in the returned map UNCHANGED
// (the packed tensors are ADDITIONS, never replacements): spec.Composed still reads the original
// per-expert names through NormalizeWeights, untouched by this hook.
//
// Returns an error naming the exact tensor/layer/role that failed (a genuinely malformed or partial
// checkpoint) so a direct caller gets a precise diagnosis; the NormalizeConfig hook that calls this at
// load time cannot propagate an error (its signature returns no error — see ArchSpec.NormalizeConfig), so
// it passes tensors through UNPACKED on failure instead, and the resulting nil ExpGate/ExpUp/ExpDown
// surfaces at decode as an ordinary missing-weight condition rather than a silently wrong pack.
func packExperts(in map[string]safetensors.Tensor, numLayers, numExperts int) (map[string]safetensors.Tensor, error) {
	if numLayers <= 0 || numExperts <= 0 {
		return nil, core.NewError("mixtral.packExperts: num_hidden_layers and num_local_experts must be > 0")
	}
	out := make(map[string]safetensors.Tensor, len(in)+numLayers*len(expertRoles)*3)
	for name, tensor := range in {
		out[name] = tensor
	}
	for layer := 0; layer < numLayers; layer++ {
		prefix := core.Sprintf("model.layers.%d.block_sparse_moe.experts.", layer)
		for _, role := range expertRoles {
			weight, scales, biases, err := packExpertRole(in, prefix, role.hfSuffix, numExperts)
			if err != nil {
				return nil, err
			}
			packedName := core.Sprintf("model.layers.%d%s.%s", layer, packedExpertsPrefix, role.packedSuffix)
			out[packedName+".weight"] = *weight
			if scales != nil {
				out[packedName+".scales"] = *scales
				out[packedName+".biases"] = *biases
			}
		}
	}
	return out, nil
}

// packExpertRole concatenates one role's numExperts per-expert 2-D tensors (prefix+"{e}."+hfSuffix+
// ".weight") into one [numExperts·outDim, inDim] tensor, row-major — and, when the checkpoint ships
// this role QUANTISED (a sibling "<expert>.<hfSuffix>.scales" tensor present), the matching packed
// .scales/.biases pair alongside it, stacked the identical way. Concatenating scales/biases along
// the same output-row axis as the weight reproduces exactly the shape model.Assemble's quant loader
// (LoadLinear/affineGeometry) already reads off a NATIVELY packed checkpoint — gemma4's
// switch_glu.*.weight/.scales/.biases ship as one [experts·outDim, inDim] weight beside a matching
// [experts·outDim, nGroups] scales/biases pair per role; this synthesises the identical convention
// from Mixtral's per-expert triples, so LoadLinear derives the SAME groupSize/bits from the packed
// shapes with zero engine-side change. A role with no ".scales" sibling packs .weight only (the
// existing bf16 behaviour, byte-identical). Every expert must share the first expert's shape and
// dtype for whichever components are present — a real checkpoint always does; a mismatch means a
// malformed or partial checkpoint, reported by the exact tensor name that broke the pattern.
func packExpertRole(in map[string]safetensors.Tensor, prefix, hfSuffix string, numExperts int) (weight, scales, biases *safetensors.Tensor, err error) {
	weight, err = packExpertComponent(in, prefix, hfSuffix, "weight", numExperts)
	if err != nil {
		return nil, nil, nil, err
	}
	if _, quantised := in[prefix+"0."+hfSuffix+".scales"]; !quantised {
		return weight, nil, nil, nil
	}
	if scales, err = packExpertComponent(in, prefix, hfSuffix, "scales", numExperts); err != nil {
		return nil, nil, nil, err
	}
	if biases, err = packExpertComponent(in, prefix, hfSuffix, "biases", numExperts); err != nil {
		return nil, nil, nil, err
	}
	return weight, scales, biases, nil
}

// packExpertComponent concatenates one role's numExperts per-expert 2-D tensors
// (prefix+"{e}."+hfSuffix+"."+component) into one [numExperts·outDim, lastDim] tensor, row-major —
// the shape-check-and-stack shared by the weight, scales and biases components. Every expert must
// share the first expert's shape and dtype; a mismatch means a malformed or partial checkpoint,
// reported by the exact tensor name that broke the pattern.
func packExpertComponent(in map[string]safetensors.Tensor, prefix, hfSuffix, component string, numExperts int) (*safetensors.Tensor, error) {
	firstName := prefix + "0." + hfSuffix + "." + component
	first, ok := in[firstName]
	if !ok {
		return nil, core.NewError("mixtral.packExperts: " + firstName + " absent")
	}
	if len(first.Shape) != 2 {
		return nil, core.NewError("mixtral.packExperts: " + firstName + " is not 2-D")
	}
	outDim, lastDim := first.Shape[0], first.Shape[1]
	rowBytes := len(first.Data)
	data := make([]byte, 0, rowBytes*numExperts)
	for e := 0; e < numExperts; e++ {
		name := core.Sprintf("%s%d.%s.%s", prefix, e, hfSuffix, component)
		t, ok := in[name]
		if !ok {
			return nil, core.NewError("mixtral.packExperts: " + name + " absent")
		}
		if len(t.Shape) != 2 || t.Shape[0] != outDim || t.Shape[1] != lastDim {
			return nil, core.NewError("mixtral.packExperts: " + name + " shape mismatch")
		}
		if t.Dtype != first.Dtype {
			return nil, core.NewError("mixtral.packExperts: " + name + " dtype mismatch")
		}
		if len(t.Data) != rowBytes {
			return nil, core.NewError("mixtral.packExperts: " + name + " byte length mismatch")
		}
		data = append(data, t.Data...)
	}
	return &safetensors.Tensor{Dtype: first.Dtype, Shape: []int{outDim * numExperts, lastDim}, Data: data}, nil
}
