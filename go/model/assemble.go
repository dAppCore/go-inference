// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

// assemble.go is the engine's generic weight assembler: ONE arch.Layer-driven loop that maps a tensor
// set onto the neutral LoadedModel, with the weight NAMES supplied as data (WeightNames) rather than
// hard-coded per architecture. The per-weight quant decision lives in LoadLinear (it reads .scales +
// the affine geometry from the shapes), so the same loop serves bf16 / 4 / 5 / 6 / 8-bit / mixed; and
// because every arch-specific weight is loaded nil-safe (absent → nil → the executor skips it), the same
// loop serves any architecture, from the full weight set down to a minimal subset.

// WeightNames maps each weight ROLE to its tensor name. Model-level fields are full names; the per-layer
// fields are SUFFIXES joined onto Sprintf(LayerPrefix, i) (mirroring the original per-arch assembler). A
// "" field = the weight is absent for this arch → loaded nil. StandardWeightNames is the canonical
// layout; an arch overrides only the names that differ.
type WeightNames struct {
	Embed, EmbedNorm, PositionEmbed, LMHead, FinalNorm string // model-level
	EmbedPerLayer, PerLayerModelProj, PerLayerProjNorm string // PLE tower (E2B/E4B)
	LayerPrefix                                        string // "model.layers.%d" — the %d carrier
	AttnNorm, PostAttnNorm, QNorm, KNorm, LayerScalar  string // per-layer norms (suffixes)
	Q, K, V, O                                         string // attention projections (suffixes)
	MLPNorm, Gate, Up, Down, PostFFNorm                string // dense MLP (suffixes)
	PerLayerGate, PerLayerProjection                   string // PLE per-layer (suffixes)
	PostPerLayerInputNorm                              string
	MoE                                                MoEWeightNames
	// NormBiasOne folds the gemma "(1 + weight)" RMSNorm convention into every norm weight at load
	// (see norm_bias.go), so the plain RMSNorm kernel reproduces gemma's (1+w)·rms(x). gemma/gemma2/
	// gemma3/gemma4 set it; mistral and non-gemma arches leave it false.
	NormBiasOne bool
}

// MoEWeightNames maps a MoE layer's weight roles (per-layer suffixes), mirroring LoadedMoE.
type MoEWeightNames struct {
	PreFFNorm, PreFFNorm2, PostFFNorm1, PostFFNorm2, PostFFNorm               string
	RouterScale, PerExpertScale                                               string
	LocalGate, LocalUp, LocalDown, Router, ExpGate, ExpUp, ExpGateUp, ExpDown string
}

// StandardWeightNames returns the canonical HF weight layout — the full superset. An arch with that
// layout uses it as-is; an architecture with different names (e.g. a pre-MLP norm named
// post_attention_layernorm) overrides only those, and the weights it lacks stay "" → nil.
func StandardWeightNames() WeightNames {
	return WeightNames{
		Embed: "model.embed_tokens", LMHead: "lm_head", FinalNorm: "model.norm.weight",
		EmbedPerLayer: "model.embed_tokens_per_layer", PerLayerModelProj: "model.per_layer_model_projection",
		PerLayerProjNorm: "model.per_layer_projection_norm.weight",
		LayerPrefix:      "model.layers.%d",
		AttnNorm:         ".input_layernorm.weight", PostAttnNorm: ".post_attention_layernorm.weight",
		QNorm: ".self_attn.q_norm.weight", KNorm: ".self_attn.k_norm.weight", LayerScalar: ".layer_scalar",
		Q: ".self_attn.q_proj", K: ".self_attn.k_proj", V: ".self_attn.v_proj", O: ".self_attn.o_proj",
		MLPNorm: ".pre_feedforward_layernorm.weight", Gate: ".mlp.gate_proj", Up: ".mlp.up_proj", Down: ".mlp.down_proj",
		PostFFNorm:   ".post_feedforward_layernorm.weight",
		PerLayerGate: ".per_layer_input_gate", PerLayerProjection: ".per_layer_projection",
		PostPerLayerInputNorm: ".post_per_layer_input_norm.weight",
		MoE: MoEWeightNames{
			PreFFNorm: ".pre_feedforward_layernorm.weight", PreFFNorm2: ".pre_feedforward_layernorm_2.weight",
			PostFFNorm1: ".post_feedforward_layernorm_1.weight", PostFFNorm2: ".post_feedforward_layernorm_2.weight",
			PostFFNorm:  ".post_feedforward_layernorm.weight",
			RouterScale: ".router.scale", PerExpertScale: ".router.per_expert_scale",
			LocalGate: ".mlp.gate_proj", LocalUp: ".mlp.up_proj", LocalDown: ".mlp.down_proj",
			Router: ".router.proj", ExpGate: ".experts.switch_glu.gate_proj",
			ExpUp: ".experts.switch_glu.up_proj", ExpGateUp: ".experts.switch_glu.gate_up_proj",
			ExpDown: ".experts.switch_glu.down_proj",
		},
	}
}

// Assemble builds the LoadedModel from a tensor set, the derived Arch, and the arch's weight names. It
// is the former per-arch assembler with the names lifted to data: the loop reads arch.Layer (OwnsCache / MoE
// / PerLayerInputHidden) for STRUCTURE and names for the tensor lookups, so it is the single assembler
// every architecture and quant shares.
func Assemble(tensors map[string]safetensors.Tensor, arch Arch, names WeightNames) (*LoadedModel, error) {
	const kind = "affine"
	t := NormalizeWrapperNames(tensors)
	d := arch.Hidden
	lin := func(name string, inDim int) *Linear { return LoadLinear(t, name, inDim, kind) }
	var foldErr error
	norm := func(name string) []byte {
		x, ok := t[name]
		if !ok {
			return nil
		}
		if names.NormBiasOne {
			folded, err := foldNormBiasOne(x.Data, x.Dtype)
			if err != nil {
				foldErr = err
				return x.Data
			}
			return folded
		}
		return x.Data
	}

	m := &LoadedModel{Arch: arch, EmbedNorm: norm(names.EmbedNorm), FinalNorm: norm(names.FinalNorm)}
	m.Embed = lin(names.Embed, d)
	m.PositionEmbed = lin(names.PositionEmbed, d)
	if m.Embed == nil {
		return nil, core.NewError("model.Assemble: " + names.Embed + " absent")
	}
	m.LMHead = lin(names.LMHead, d) // nil ⇒ tied to Embed
	if arch.TieWordEmbeddings != nil {
		if *arch.TieWordEmbeddings && m.LMHead != nil {
			return nil, core.NewError("model.Assemble: config declares tied word embeddings but lm_head is present")
		}
		if !*arch.TieWordEmbeddings && m.LMHead == nil {
			return nil, core.NewError("model.Assemble: config declares untied word embeddings but lm_head is absent")
		}
	}

	if arch.PerLayerInputHidden > 0 {
		plDim := len(arch.Layer) * arch.PerLayerInputHidden
		m.EmbedPerLayer = lin(names.EmbedPerLayer, plDim)
		m.PerLayerModelProj = lin(names.PerLayerModelProj, d)
		m.PerLayerProjNorm = norm(names.PerLayerProjNorm)
	}

	m.Layers = make([]LoadedLayer, len(arch.Layer))
	for i := range arch.Layer {
		p := core.Sprintf(names.LayerPrefix, i)
		spec := arch.Layer[i]
		qDim := arch.Heads * spec.HeadDim // o_proj input width (global layers have a larger head_dim)
		L := &m.Layers[i]
		L.AttnNorm = norm(p + names.AttnNorm)
		L.PostAttnNorm = norm(p + names.PostAttnNorm)
		L.QNorm = norm(p + names.QNorm)
		L.KNorm = norm(p + names.KNorm)
		L.LayerScalar = norm(p + names.LayerScalar)
		L.Q = lin(p+names.Q, d)
		if spec.OwnsCache() { // KV-shared layers carry no own k/v; v is also absent on K==V layers (lin → nil)
			L.K = lin(p+names.K, d)
			L.V = lin(p+names.V, d)
		}
		L.O = lin(p+names.O, qDim)

		if spec.MoE {
			L.MoE = assembleMoE(t, p, arch, names.MoE, lin, norm, kind)
		} else {
			L.MLPNorm = norm(p + names.MLPNorm)
			L.Gate = lin(p+names.Gate, d)
			L.Up = lin(p+names.Up, d)
			ff := arch.FF
			if L.Gate != nil { // per-layer FFN width (MatFormer): read from the gate's output rows
				ff = L.Gate.OutDim
			}
			L.Down = lin(p+names.Down, ff)
			L.PostFFNorm = norm(p + names.PostFFNorm)
		}

		if arch.PerLayerInputHidden > 0 {
			L.PerLayerGate = lin(p+names.PerLayerGate, d)
			L.PerLayerProjection = lin(p+names.PerLayerProjection, arch.PerLayerInputHidden)
			L.PostPerLayerInputNorm = norm(p + names.PostPerLayerInputNorm)
		}
	}
	if foldErr != nil {
		return nil, foldErr
	}
	if err := m.ValidateRequired(arch); err != nil {
		return nil, err
	}
	return m, nil
}

// assembleMoE builds a MoE layer's dual-branch FFN (local dense MLP + sparse experts).
func assembleMoE(t map[string]safetensors.Tensor, p string, arch Arch, names MoEWeightNames, lin func(string, int) *Linear, norm func(string) []byte, kind string) *LoadedMoE {
	d := arch.Hidden
	expGate := lin(p+names.ExpGate, d)
	expUp := lin(p+names.ExpUp, d)
	return &LoadedMoE{
		PreFFNorm:      norm(p + names.PreFFNorm),
		PreFFNorm2:     norm(p + names.PreFFNorm2),
		PostFFNorm1:    norm(p + names.PostFFNorm1),
		PostFFNorm2:    norm(p + names.PostFFNorm2),
		PostFFNorm:     norm(p + names.PostFFNorm),
		RouterScale:    norm(p + names.RouterScale),
		PerExpertScale: norm(p + names.PerExpertScale),
		LocalGate:      lin(p+names.LocalGate, d),
		LocalUp:        lin(p+names.LocalUp, d),
		LocalDown:      lin(p+names.LocalDown, arch.ExpertFF),
		Router:         lin(p+names.Router, d),
		ExpGate:        expGate,
		ExpUp:          expUp,
		ExpGateUp:      lin(p+names.ExpGateUp, d),
		ExpDown:        lin(p+names.ExpDown, arch.ExpertFF),
	}
}
