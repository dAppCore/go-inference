// SPDX-Licence-Identifier: EUPL-1.2

package qwenmoe

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// packedExpertsPrefix names the synthetic per-layer tensor packExperts writes — see mixtral.packExperts'
// doc for why synthesis is needed at all (model.Assemble's generic MoE loader wants ONE tensor per role
// per layer; real Qwen2-MoE/Qwen3-MoE checkpoints ship each expert as its OWN 2-D tensor,
// mlp.experts.{i}.{gate,up,down}_proj.weight — confirmed against the real Qwen1.5-MoE-A2.7B and
// Qwen3-30B-A3B safetensors index fixtures in testdata/, via TestWeightMap_Good). Named "experts_packed"
// (not "switch_mlp", the mlx-lm community repack's OWN batched layout composed.buildMoE already
// recognises separately) so it can never collide with a real checkpoint name.
const packedExpertsPrefix = ".mlp.experts_packed"

// FactoryWeightNames returns the Qwen2-MoE/Qwen3-MoE tensor layout for model.Assemble: StandardWeightNames'
// attention projections as-is (the qwenmoe family is llama-shaped there — self_attn.{q,k,v,o}_proj,
// input_layernorm; Qwen3-MoE additionally ships self_attn.{q,k}_norm, which StandardWeightNames' default
// QNorm/KNorm names already cover and Qwen2-MoE simply lacks — nil-safe either way), the llama/qwen 2-norm
// FFN override (post_attention_layernorm is the pre-MoE norm; there is no gemma-style post-attention
// sandwich norm — mirrors mixtral.FactoryWeightNames/granitemoe.FactoryWeightNames), the routed MoE block
// pointed at packExperts' synthesised packed tensors, and the shared-expert block pointed straight at the
// checkpoint's own mlp.shared_expert.* tensors (a single dense trio per layer, not per-expert, so no
// packing needed there). Qwen1.5-MoE/Qwen2-MoE carry the shared expert; Qwen3-MoE dropped it (see
// testdata's Qwen-Qwen3-30B-A3B fixture: no shared_expert_gate.weight in its weight map) — both are
// declared unconditionally here because LoadLinear is nil-safe on an absent name, exactly like
// QNorm/KNorm above.
//
// FIXED (#57, was a KNOWN LIMITATION here): SharedDown's declared inDim now derives from
// arch.SharedExpertFF (populated below from shared_expert_intermediate_size) instead of assuming it
// equals arch.ExpertFF (the ROUTED experts' intermediate size) — model.Arch carries a distinct
// shared-expert-FF field precisely because real Qwen2-MoE checkpoints need one: Qwen1.5-MoE-A2.7B ships
// moe_intermediate_size=1408 but shared_expert_intermediate_size=5632 (testdata/
// Qwen-Qwen1.5-MoE-A2.7B-config.json) — a 4x mismatch. For a DENSE (bf16) checkpoint the old assumption
// was harmless (engine/metal's qw() carries only Weight/Scales/Biases/GroupSize/Bits through a
// QuantWeight, dropping Linear.InDim entirely); for a QUANTISED shared expert the wrong inDim derived
// the wrong affine GroupSize/Bits (model.LoadLinear's affineGeometry) — see
// TestTinyQwen2MoEFactoryLoad_QuantisedSharedExpert_Good (load_test.go) for the byte-level regression
// proof. Numeric parity against a real checkpoint stays out of scope regardless (host-side
// synthetic-tensor tests only, per the #50 archzoo brief).
//
// STILL OPEN (out of THIS change's file fence — #57 follow-up): engine/metal's non-fused shared-expert
// decode (arch_qwen_moe.go's encQwenMoEHalf) independently hardcodes "shared FF == moe.ExpertDFF" when
// sizing its MoEExpertsQuantSiLU dispatch — a decode-time correctness gap for a checkpoint whose shared
// and routed FF genuinely differ, separate from (and not fixed by) the InDim/GroupSize/Bits correction
// above. arch_qwen_fused.go's chain lane is UNAFFECTED (qwenChainMoE already derives sharedFF from the
// packed byte stride, never from arch.ExpertFF).
func FactoryWeightNames() model.WeightNames {
	w := model.StandardWeightNames()
	w.MLPNorm = ".post_attention_layernorm.weight"
	w.PostAttnNorm = ""
	w.MoE = model.MoEWeightNames{
		PreFFNorm:     ".post_attention_layernorm.weight",
		Router:        ".mlp.gate",
		ExpGate:       packedExpertsPrefix + ".gate_proj",
		ExpUp:         packedExpertsPrefix + ".up_proj",
		ExpDown:       packedExpertsPrefix + ".down_proj",
		SharedGate:    ".mlp.shared_expert.gate_proj",
		SharedUp:      ".mlp.shared_expert.up_proj",
		SharedDown:    ".mlp.shared_expert.down_proj",
		SharedSigmoid: ".mlp.shared_expert_gate",
	}
	return w
}

// expertRole is one routed-expert projection: the real per-expert HF suffix and the packed tensor's role
// suffix. Unlike Mixtral (whose w1/w2/w3 checkpoint names differ from the packed gate_proj/up_proj/
// down_proj roles), Qwen2-MoE/Qwen3-MoE already name each expert's tensors gate_proj/up_proj/down_proj —
// hfSuffix and packedSuffix are identical strings here, kept as two fields anyway (rather than one) so
// this mirrors mixtral.expertRole exactly and stays a mechanical find-and-adapt for the next arch.
type expertRole struct{ hfSuffix, packedSuffix string }

// expertRoles is the fixed gate/up/down triple in a DETERMINISTIC order — map iteration is not, and
// packExperts must visit the same roles every call for reproducible output.
var expertRoles = []expertRole{{"gate_proj", "gate_proj"}, {"up_proj", "up_proj"}, {"down_proj", "down_proj"}}

// packExperts synthesises the packed per-layer routed-expert tensors packedExpertsPrefix names: for every
// layer in [0,numLayers) and every role (gate/up/down), it concatenates the numExperts per-expert 2-D
// tensors (mlp.experts.{e}.{gate,up,down}_proj.weight) row-major into one [numExperts·outDim, inDim]
// tensor — expert e occupies rows [e·outDim, (e+1)·outDim), matching mixtral.packExperts' byte-layout
// contract exactly. Every source tensor also stays in the returned map UNCHANGED (the packed tensors are
// ADDITIONS, never replacements): spec.Composed still reads the original per-expert (and shared-expert)
// names, untouched by this hook. Structurally mirrors mixtral.packExperts; kept as this package's own copy
// (not a shared helper) so each arch package stays self-contained per house style — only the prefix
// ("model.layers.%d.mlp.experts." vs "…block_sparse_moe.experts.") and role-suffix set differ.
//
// Returns an error naming the exact tensor/layer/role that failed (a genuinely malformed or partial
// checkpoint); the NormalizeConfig hook that calls this at load time cannot propagate an error (its
// signature returns no error — see ArchSpec.NormalizeConfig), so it passes tensors through UNPACKED on
// failure instead, and the resulting nil ExpGate/ExpUp/ExpDown surfaces at decode as an ordinary
// missing-weight condition rather than a silently wrong pack.
func packExperts(in map[string]safetensors.Tensor, numLayers, numExperts int) (map[string]safetensors.Tensor, error) {
	if numLayers <= 0 || numExperts <= 0 {
		return nil, core.NewError("qwenmoe.packExperts: num_hidden_layers and num_experts must be > 0")
	}
	out := make(map[string]safetensors.Tensor, len(in)+numLayers*len(expertRoles)*3)
	for name, tensor := range in {
		out[name] = tensor
	}
	for layer := 0; layer < numLayers; layer++ {
		prefix := core.Sprintf("model.layers.%d.mlp.experts.", layer)
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
// from Qwen-MoE's per-expert triples, so LoadLinear derives the SAME groupSize/bits from the packed
// shapes with zero engine-side change. A role with no ".scales" sibling packs .weight only (the
// existing bf16 behaviour, byte-identical). Every expert must share the first expert's shape and
// dtype for whichever components are present — a real checkpoint always does; a mismatch means a
// malformed or partial checkpoint, reported by the exact tensor name that broke the pattern. Mirrors
// mixtral.packExpertRole.
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
		return nil, core.NewError("qwenmoe.packExperts: " + firstName + " absent")
	}
	if len(first.Shape) != 2 {
		return nil, core.NewError("qwenmoe.packExperts: " + firstName + " is not 2-D")
	}
	outDim, lastDim := first.Shape[0], first.Shape[1]
	rowBytes := len(first.Data)
	data := make([]byte, 0, rowBytes*numExperts)
	for e := 0; e < numExperts; e++ {
		name := core.Sprintf("%s%d.%s.%s", prefix, e, hfSuffix, component)
		t, ok := in[name]
		if !ok {
			return nil, core.NewError("qwenmoe.packExperts: " + name + " absent")
		}
		if len(t.Shape) != 2 || t.Shape[0] != outDim || t.Shape[1] != lastDim {
			return nil, core.NewError("qwenmoe.packExperts: " + name + " shape mismatch")
		}
		if t.Dtype != first.Dtype {
			return nil, core.NewError("qwenmoe.packExperts: " + name + " dtype mismatch")
		}
		if len(t.Data) != rowBytes {
			return nil, core.NewError("qwenmoe.packExperts: " + name + " byte length mismatch")
		}
		data = append(data, t.Data...)
	}
	return &safetensors.Tensor{Dtype: first.Dtype, Shape: []int{outDim * numExperts, lastDim}, Data: data}, nil
}
