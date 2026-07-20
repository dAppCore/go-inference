// SPDX-Licence-Identifier: EUPL-1.2

package olmoe

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// packedExpertsPrefix names the synthetic per-layer tensor packExperts writes. OLMoE's real checkpoint
// (allenai/OLMoE-1B-7B-0924) ships each routed expert as its OWN 2-D tensor
// (.mlp.experts.{i}.gate_proj/up_proj/down_proj.weight — the same per-expert HF convention Mixtral uses,
// just under the plain ".mlp." prefix rather than ".block_sparse_moe."; the Composed route below consumes
// this layout directly, no renaming needed). model.Assemble's generic MoE loader (assembleMoE) looks up ONE
// tensor per role per layer, so packExperts SYNTHESISES that at load time exactly as mixtral.packExperts
// does for Mixtral's structurally identical per-expert layout (see that package's weights.go for the full
// convention writeup): numExperts per-expert matrices concatenated row-major into one
// [numExperts·outDim, inDim] tensor. Named "experts_packed" so it can never collide with a real checkpoint
// tensor name.
const packedExpertsPrefix = ".mlp.experts_packed"

// FactoryWeightNames returns the OLMoE tensor layout for model.Assemble: StandardWeightNames' attention
// projections AND its QK-norm suffixes as-is (OLMoE carries per-head q/k RMSNorm under the standard
// self_attn.{q,k}_norm.weight names — see the tinyOLMoEWeights fixture in integration_test.go), the
// llama/mistral 2-norm FFN override (post_attention_layernorm is the pre-MoE norm; there is no gemma-style
// post-attention sandwich norm), and the MoE block pointed at packExperts' synthesised packed tensors.
// OLMoE has no shared expert, so SharedGate/Up/Down/SharedSigmoid stay "" (nil-safe — see
// model.MoEWeightNames).
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

// expertRoles is the fixed gate/up/down triple in a DETERMINISTIC order — map iteration is not, and
// packExperts must visit the same roles every call for reproducible output.
var expertRoles = []string{"gate_proj", "up_proj", "down_proj"}

// packExperts synthesises the packed per-layer expert tensors packedExpertsPrefix names: for every layer in
// [0,numLayers) and every role (gate/up/down), it concatenates the numExperts per-expert 2-D tensors
// (model.layers.{l}.mlp.experts.{e}.{role}.weight) row-major into one [numExperts·outDim, inDim] tensor —
// expert e occupies rows [e·outDim, (e+1)·outDim), so a downstream row-slice by expert index reproduces
// expert e's original matrix byte-for-byte. Every source tensor also stays in the returned map UNCHANGED
// (the packed tensors are ADDITIONS, never replacements): spec.Composed still reads the original per-expert
// names untouched by this hook.
//
// Returns an error naming the exact tensor/layer/role that failed (a genuinely malformed or partial
// checkpoint) so a direct caller gets a precise diagnosis; the NormalizeConfig hook that calls this at load
// time cannot propagate an error (its signature returns no error — see ArchSpec.NormalizeConfig), so it
// passes tensors through UNPACKED on failure instead, and the resulting nil ExpGate/ExpUp/ExpDown surfaces
// at decode as an ordinary missing-weight condition rather than a silently wrong pack.
func packExperts(in map[string]safetensors.Tensor, numLayers, numExperts int) (map[string]safetensors.Tensor, error) {
	if numLayers <= 0 || numExperts <= 0 {
		return nil, core.NewError("olmoe.packExperts: num_hidden_layers and num_experts must be > 0")
	}
	out := make(map[string]safetensors.Tensor, len(in)+numLayers*len(expertRoles))
	for name, tensor := range in {
		out[name] = tensor
	}
	for layer := 0; layer < numLayers; layer++ {
		prefix := core.Sprintf("model.layers.%d.mlp.experts.", layer)
		for _, role := range expertRoles {
			packed, err := packExpertRole(in, prefix, role, numExperts)
			if err != nil {
				return nil, err
			}
			out[core.Sprintf("model.layers.%d%s.%s.weight", layer, packedExpertsPrefix, role)] = *packed
		}
	}
	return out, nil
}

// packExpertRole concatenates one role's numExperts per-expert 2-D tensors (prefix+"{e}."+role+".weight")
// into one [numExperts·outDim, inDim] tensor, row-major. Every expert must share the first expert's shape
// and dtype — a real checkpoint always does; a mismatch means a malformed or partial checkpoint, reported
// by the exact tensor name that broke the pattern.
func packExpertRole(in map[string]safetensors.Tensor, prefix, role string, numExperts int) (*safetensors.Tensor, error) {
	firstName := prefix + "0." + role + ".weight"
	first, ok := in[firstName]
	if !ok {
		return nil, core.NewError("olmoe.packExperts: " + firstName + " absent")
	}
	if len(first.Shape) != 2 {
		return nil, core.NewError("olmoe.packExperts: " + firstName + " is not 2-D")
	}
	outDim, inDim := first.Shape[0], first.Shape[1]
	rowBytes := len(first.Data)
	data := make([]byte, 0, rowBytes*numExperts)
	for e := 0; e < numExperts; e++ {
		name := core.Sprintf("%s%d.%s.weight", prefix, e, role)
		t, ok := in[name]
		if !ok {
			return nil, core.NewError("olmoe.packExperts: " + name + " absent")
		}
		if len(t.Shape) != 2 || t.Shape[0] != outDim || t.Shape[1] != inDim {
			return nil, core.NewError("olmoe.packExperts: " + name + " shape mismatch")
		}
		if t.Dtype != first.Dtype {
			return nil, core.NewError("olmoe.packExperts: " + name + " dtype mismatch")
		}
		if len(t.Data) != rowBytes {
			return nil, core.NewError("olmoe.packExperts: " + name + " byte length mismatch")
		}
		data = append(data, t.Data...)
	}
	return &safetensors.Tensor{Dtype: first.Dtype, Shape: []int{outDim * numExperts, inDim}, Data: data}, nil
}
