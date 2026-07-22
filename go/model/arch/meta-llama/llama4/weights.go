// SPDX-Licence-Identifier: EUPL-1.2

package llama4

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// packedExpertsPrefix names the synthetic per-layer tensor packExperts writes. Llama 4's real
// checkpoint (meta-llama/Llama-4-Scout-17B-16E-Instruct) ships every MoE layer's routed experts as
// ONE FUSED 3-D tensor per role (language_model.model.layers.{l}.feed_forward.experts.gate_up_proj/
// down_proj — see testdata/'s index excerpt), in the checkpoint's bmm [experts, inDim, outDim]
// layout. NormalizeWeights already decodes that (transposing each expert's chunk to the engine's
// [outDim, inDim] convention) into per-expert tensors for the Composed route
// (…mlp.experts.{e}.{gate,up,down}_proj.weight — see register.go); model.Assemble's generic MoE
// loader (assembleMoE) wants the OPPOSITE shape again — ONE tensor per role per layer — so
// packExperts re-concatenates NormalizeWeights' per-expert output back into that packed
// [numExperts·outDim, inDim] convention. This is the same byte layout dbrx.packExperts/
// mixtral.packExperts synthesise from their own per-expert tensors (see dbrx/weights.go for the
// full convention writeup), and the same "experts_packed" naming discipline: it can never collide
// with a real checkpoint name, nor with NormalizeWeights' own "mlp.experts.<index>." per-expert
// names.
//
// Chaining through NormalizeWeights — rather than re-deriving the packed-tensor slice+transpose
// split a second time directly against the raw feed_forward.experts.* tensors — means packExperts
// reuses the ALREADY-TESTED decode (register_test.go); the only new logic here is the
// re-concatenation, identical in shape to dbrx's and mixtral's.
const packedExpertsPrefix = ".mlp.experts_packed"

// FactoryWeightNames returns the Llama 4 tensor layout for model.Assemble, once NormalizeConfig has
// run: NormalizeWeights only ever renames the ".feed_forward." branch onto ".mlp." (router included:
// feed_forward.router.weight -> mlp.gate.weight) — it never touches attention (self_attn.{q,k,v,o}_
// proj), the layer norms (input_layernorm / post_attention_layernorm), or the top-level embed/norm/
// lm_head names, because the real checkpoint already ships those llama-shaped. FactoryWeightNames
// therefore only overrides the FFN/MoE roles, exactly as dbrx/mixtral/granitemoe/qwenmoe do: the
// llama 2-norm coupling (post_attention_layernorm is the pre-MoE norm; there is no gemma-style
// post-attention sandwich norm), the MoE block pointed at packExperts' synthesised packed tensors,
// and — unlike dbrx/mixtral/granitemoe, but like Qwen1.5-MoE — the shared-expert trio Llama 4 always
// carries (Config.Arch hardcodes SharedExperts: 1; every published Llama4TextConfig has one).
//
// SharedSigmoid stays "" (nil-safe — see model.MoEWeightNames): unlike Qwen1.5-MoE's
// shared_expert_gate.weight, Llama 4's checkpoint carries no sigmoid gate on the shared expert (the
// testdata/ index excerpt lists only the gate/up/down trio) — composed/moe.go documents the same
// ungated fallback for an absent gate tensor.
//
// Every name below is UNPREFIXED (the StandardWeightNames "model.layers.%d" convention), even
// though the real checkpoint wraps every text-tower tensor under "language_model." (a multimodal
// Llama4ForConditionalGeneration artefact — see testdata/'s index excerpt). model.Assemble strips
// that wrapper prefix itself (model.NormalizeWrapperNames), so bare lookups resolve regardless of
// nesting; packExperts below does the same stripping explicitly, since it runs at NormalizeConfig
// time, before Assemble's internal pass. The vision tower (vision_config, when present) is
// deliberately outside this map — see the package doc comment: this is the text lane only.
func FactoryWeightNames() model.WeightNames {
	w := model.StandardWeightNames()
	w.MLPNorm = ".post_attention_layernorm.weight"
	w.PostAttnNorm = ""
	w.MoE = model.MoEWeightNames{
		PreFFNorm:  ".post_attention_layernorm.weight",
		Router:     ".mlp.gate",
		ExpGate:    packedExpertsPrefix + ".gate_proj",
		ExpUp:      packedExpertsPrefix + ".up_proj",
		ExpDown:    packedExpertsPrefix + ".down_proj",
		SharedGate: ".mlp.shared_expert.gate_proj",
		SharedUp:   ".mlp.shared_expert.up_proj",
		SharedDown: ".mlp.shared_expert.down_proj",
	}
	return w
}

// expertRoles is the fixed gate/up/down triple NormalizeWeights writes per expert, in a
// DETERMINISTIC order — map iteration is not, and packExperts must visit the same roles every call
// for reproducible output. NormalizeWeights already renames Llama 4's checkpoint onto these exact
// role suffixes, so — like dbrx, and unlike mixtral (whose checkpoint suffixes are w1/w2/w3) —
// there is no separate "real HF suffix" to track: the source and packed role names are identical
// strings.
var expertRoles = []string{"gate_proj", "up_proj", "down_proj"}

// packExperts synthesises the packed per-layer expert tensors packedExpertsPrefix names, from the
// PER-EXPERT tensors NormalizeWeights produces (…mlp.experts.{e}.{role}.weight): for every MoE layer
// in arch.Layer and every role, it concatenates the arch.Experts per-expert 2-D tensors row-major
// into one [arch.Experts·outDim, inDim] tensor — expert e occupies rows [e·outDim, (e+1)·outDim), so
// a downstream row-slice by expert index reproduces expert e's original matrix byte-for-byte. Every
// source tensor also stays in the returned map UNCHANGED (the packed tensors are ADDITIONS, never
// replacements): spec.Composed still reads the original per-expert names through its own
// NormalizeWeights call, untouched by this hook.
//
// Unlike dbrx/mixtral/qwenmoe (every layer routed), Llama 4 interleaves dense and MoE layers
// (moe_layers / interleave_moe_layer_step — see Config.Arch): a layer with spec.MoE == false is
// SKIPPED, not packed — it carries no expert tensors at all, and probing for them would fail loudly
// on a perfectly valid dense layer. This is the wrinkle none of the all-MoE exemplars had to handle.
//
// in may or may not still carry Llama 4's "language_model." multimodal wrapper prefix (see
// FactoryWeightNames' doc) — packExperts normalises it via model.NormalizeWrapperNames before
// reading, so the hardcoded "model.layers.%d.mlp.experts." lookup below resolves either way, and
// writes its packed output under that same unprefixed convention (FactoryWeightNames' LayerPrefix).
//
// Returns an error naming the exact tensor/layer/role that failed (a genuinely malformed or partial
// checkpoint, or a NormalizeWeights input that never carried the packed experts) so a direct caller
// gets a precise diagnosis; the NormalizeConfig hook that calls this at load time cannot propagate an
// error (its signature returns no error — see ArchSpec.NormalizeConfig), so it passes the
// normalised-but-unpacked tensors through on failure instead, and the resulting nil
// ExpGate/ExpUp/ExpDown surfaces at decode as an ordinary missing-weight condition rather than a
// silently wrong pack.
func packExperts(in map[string]safetensors.Tensor, arch model.Arch) (map[string]safetensors.Tensor, error) {
	if arch.Experts <= 0 {
		return nil, core.NewError("llama4.packExperts: num_local_experts must be > 0")
	}
	in = model.NormalizeWrapperNames(in)
	out := make(map[string]safetensors.Tensor, len(in)+len(arch.Layer)*len(expertRoles))
	for name, tensor := range in {
		out[name] = tensor
	}
	for layer, spec := range arch.Layer {
		if !spec.MoE {
			continue // dense layer (llama4's interleaved moe_layers) — nothing to pack
		}
		prefix := core.Sprintf("model.layers.%d.mlp.experts.", layer)
		for _, role := range expertRoles {
			packed, err := packExpertRole(in, prefix, role, arch.Experts)
			if err != nil {
				return nil, err
			}
			out[core.Sprintf("model.layers.%d%s.%s.weight", layer, packedExpertsPrefix, role)] = *packed
		}
	}
	return out, nil
}

// packExpertRole concatenates one role's numExperts per-expert 2-D tensors (prefix+"{e}."+role+
// ".weight") into one [numExperts·outDim, inDim] tensor, row-major. Every expert must share the
// first expert's shape and dtype — NormalizeWeights' own decode of a real checkpoint always produces
// that; a mismatch means a malformed/partial checkpoint or a hand-built input, reported by the exact
// tensor name that broke the pattern. Mirrors dbrx.packExpertRole/mixtral.packExpertRole.
func packExpertRole(in map[string]safetensors.Tensor, prefix, role string, numExperts int) (*safetensors.Tensor, error) {
	firstName := prefix + "0." + role + ".weight"
	first, ok := in[firstName]
	if !ok {
		return nil, core.NewError("llama4.packExperts: " + firstName + " absent")
	}
	if len(first.Shape) != 2 {
		return nil, core.NewError("llama4.packExperts: " + firstName + " is not 2-D")
	}
	outDim, inDim := first.Shape[0], first.Shape[1]
	rowBytes := len(first.Data)
	data := make([]byte, 0, rowBytes*numExperts)
	for e := 0; e < numExperts; e++ {
		name := core.Sprintf("%s%d.%s.weight", prefix, e, role)
		t, ok := in[name]
		if !ok {
			return nil, core.NewError("llama4.packExperts: " + name + " absent")
		}
		if len(t.Shape) != 2 || t.Shape[0] != outDim || t.Shape[1] != inDim {
			return nil, core.NewError("llama4.packExperts: " + name + " shape mismatch")
		}
		if t.Dtype != first.Dtype {
			return nil, core.NewError("llama4.packExperts: " + name + " dtype mismatch")
		}
		if len(t.Data) != rowBytes {
			return nil, core.NewError("llama4.packExperts: " + name + " byte length mismatch")
		}
		data = append(data, t.Data...)
	}
	return &safetensors.Tensor{Dtype: first.Dtype, Shape: []int{outDim * numExperts, inDim}, Data: data}, nil
}
