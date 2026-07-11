// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model/qwen3"
	"dappco.re/go/inference/model/safetensors"
)

// loader.go builds a ComposedModel from a hybrid checkpoint (Qwen 3.6), the native port of metal's
// composed.buildComposed: parse the config, dispatch each layer by layer_type to its mixer (linear_attn →
// gated-delta, self_attn → attention), wire the SwiGLU MLP + the two norms, and resolve the
// model.language_model. multimodal-wrapper prefix. The gated-delta geometry is derived from the weight
// shapes (as metal does); the attention geometry from the config.

// The loaderConfig type + its effective()/ropeTheta()/partialRotary() helpers live in config.go.

// tensorF32 widens a bf16/f32 safetensors tensor to a flat f32 slice.
func tensorF32(t safetensors.Tensor) ([]float32, error) {
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
	return nil, core.NewError("composed.tensorF32: unsupported dtype " + t.Dtype)
}

// LoadComposed assembles a ComposedModel from a hybrid checkpoint's tensors + its config.json bytes.
func LoadComposed(tensors map[string]safetensors.Tensor, configJSON []byte) (*ComposedModel, error) {
	var raw loaderConfig
	if r := core.JSONUnmarshal(configJSON, &raw); !r.OK {
		return nil, core.NewError("composed.LoadComposed: config.json parse failed")
	}
	cfg := raw.effective()
	if cfg.HiddenSize <= 0 || cfg.NumHiddenLayers <= 0 {
		return nil, core.NewError("composed.LoadComposed: hidden_size and num_hidden_layers required")
	}

	// Resolve the weight prefix (multimodal wrapper nests under model.language_model.).
	prefix := "model."
	if _, ok := tensors["model.language_model.embed_tokens.weight"]; ok {
		prefix = "model.language_model."
	}
	get := func(name string) (safetensors.Tensor, bool) { t, ok := tensors[name]; return t, ok }
	f32 := func(name string) ([]float32, error) {
		t, ok := get(name)
		if !ok {
			return nil, core.NewError("composed.LoadComposed: missing " + name)
		}
		return tensorF32(t)
	}
	f32opt := func(name string) []float32 {
		if t, ok := get(name); ok {
			if v, e := tensorF32(t); e == nil {
				return v
			}
		}
		return nil
	}

	embedT, ok := get(prefix + "embed_tokens.weight")
	if !ok || len(embedT.Shape) != 2 {
		return nil, core.NewError("composed.LoadComposed: missing/!2D embed_tokens.weight")
	}
	embed, err := tensorF32(embedT)
	if err != nil {
		return nil, err
	}
	D := embedT.Shape[1]
	vocab := embedT.Shape[0]
	normF, err := f32(prefix + "norm.weight")
	if err != nil {
		return nil, err
	}
	output := f32opt("lm_head.weight") // untied; nil ⇒ tied to embed

	kinds, err := resolveKinds(cfg)
	if err != nil {
		return nil, err
	}

	m := &ComposedModel{Embed: embed, NormF: normF, Output: output, D: D, Vocab: vocab, Eps: cfg.RMSNormEps}
	if m.Eps == 0 {
		m.Eps = 1e-6
	}
	for i := 0; i < cfg.NumHiddenLayers; i++ {
		lp := prefix + core.Sprintf("layers.%d.", i)
		inNorm, err := f32(lp + "input_layernorm.weight")
		if err != nil {
			return nil, err
		}
		postNorm, err := f32(lp + "post_attention_layernorm.weight")
		if err != nil {
			return nil, err
		}
		ffn, err := buildFFN(get, f32, lp+"mlp.", cfg, D)
		if err != nil {
			return nil, core.E("composed.LoadComposed", core.Sprintf("layer %d ffn", i), err)
		}

		var mixer Mixer
		if kinds[i] == "full_attention" {
			mixer, err = buildAttn(f32, f32opt, lp+"self_attn.", cfg, D)
		} else {
			mixer, err = buildGatedDelta(get, f32, f32opt, lp+"linear_attn.", cfg, D)
		}
		if err != nil {
			return nil, core.E("composed.LoadComposed", core.Sprintf("layer %d (%s)", i, kinds[i]), err)
		}
		m.Layers = append(m.Layers, Layer{
			InputNorm: inNorm, Mixer: mixer, PostAttnNorm: postNorm, MLP: ffn,
		})
	}
	return m, nil
}

// resolveKinds maps each layer to "full_attention" or "linear_attention" from layer_types (preferred) or
// full_attention_interval (every Nth layer is full).
func resolveKinds(cfg *loaderConfig) ([]string, error) {
	n := cfg.NumHiddenLayers
	out := make([]string, n)
	if len(cfg.LayerTypes) == n {
		copy(out, cfg.LayerTypes)
		return out, nil
	}
	if cfg.FullAttentionInterval > 0 {
		for i := range out {
			if (i+1)%cfg.FullAttentionInterval == 0 {
				out[i] = "full_attention"
			} else {
				out[i] = "linear_attention"
			}
		}
		return out, nil
	}
	return nil, core.NewError("composed.resolveKinds: need layer_types or full_attention_interval")
}

// buildAttn builds a full-attention mixer; geometry from the config.
func buildAttn(f32 func(string) ([]float32, error), f32opt func(string) []float32, sp string, cfg *loaderConfig, D int) (Mixer, error) {
	q, err := f32(sp + "q_proj.weight")
	if err != nil {
		return nil, err
	}
	k, err := f32(sp + "k_proj.weight")
	if err != nil {
		return nil, err
	}
	v, err := f32(sp + "v_proj.weight")
	if err != nil {
		return nil, err
	}
	o, err := f32(sp + "o_proj.weight")
	if err != nil {
		return nil, err
	}
	heads := cfg.NumAttentionHeads
	headDim := cfg.HeadDim
	if headDim == 0 && heads > 0 {
		headDim = (len(q) / D) / heads
	}
	kvHeads := cfg.NumKeyValueHeads
	if kvHeads == 0 {
		kvHeads = heads
	}
	rd := int(cfg.partialRotary() * float32(headDim))
	if rd%2 != 0 {
		rd--
	}
	return NewAttnMixer(&AttnWeights{
		QProj: q, KProj: k, VProj: v, OProj: o,
		QNorm: f32opt(sp + "q_norm.weight"), KNorm: f32opt(sp + "k_norm.weight"),
	}, AttnConfig{Heads: heads, KVHeads: kvHeads, HeadDim: headDim, RotaryDim: rd, RopeTheta: cfg.ropeTheta(), NormEps: cfg.RMSNormEps}), nil
}

// buildGatedDelta builds a gated-delta mixer; geometry derived from the weight shapes (as metal does):
// ValueHeads = len(A_log), HeadDim = len(norm), convDim/K from conv1d.weight, qDim = (convDim−vDim)/2,
// KeyHeads = qDim/HeadDim.
func buildGatedDelta(get func(string) (safetensors.Tensor, bool), f32 func(string) ([]float32, error), f32opt func(string) []float32, sp string, lcfg *loaderConfig, D int) (Mixer, error) {
	aLogT, ok := get(sp + "A_log")
	if !ok || len(aLogT.Shape) != 1 {
		return nil, core.NewError("missing/!1D A_log")
	}
	normT, ok := get(sp + "norm.weight")
	if !ok || len(normT.Shape) != 1 {
		return nil, core.NewError("missing/!1D norm.weight")
	}
	convT, ok := get(sp + "conv1d.weight")
	if !ok || len(convT.Shape) == 0 {
		return nil, core.NewError("missing conv1d.weight")
	}
	valueHeads := aLogT.Shape[0]
	headDim := normT.Shape[0]
	convDim := convT.Shape[0]
	convK := convT.Shape[len(convT.Shape)-1]
	vDim := valueHeads * headDim
	if (convDim-vDim)%2 != 0 {
		return nil, core.NewError(core.Sprintf("gated-delta geometry: convDim %d − vDim %d not even", convDim, vDim))
	}
	qDim := (convDim - vDim) / 2
	if headDim == 0 || qDim%headDim != 0 {
		return nil, core.NewError("gated-delta geometry: qDim not divisible by headDim")
	}
	keyHeads := qDim / headDim
	if err := lcfg.validateLinearGeometry(keyHeads, valueHeads, headDim, convK); err != nil {
		return nil, err
	}

	qkv, err := f32(sp + "in_proj_qkv.weight")
	if err != nil {
		return nil, err
	}
	convW, err := tensorF32(convT) // [convDim,1,K] contiguous = [convDim,K]
	if err != nil {
		return nil, err
	}
	aLog, err := tensorF32(aLogT)
	if err != nil {
		return nil, err
	}
	norm, err := tensorF32(normT)
	if err != nil {
		return nil, err
	}
	inA, err := f32(sp + "in_proj_a.weight")
	if err != nil {
		return nil, err
	}
	inB, err := f32(sp + "in_proj_b.weight")
	if err != nil {
		return nil, err
	}
	inZ, err := f32(sp + "in_proj_z.weight")
	if err != nil {
		return nil, err
	}
	outP, err := f32(sp + "out_proj.weight")
	if err != nil {
		return nil, err
	}
	w := &qwen3.GatedDeltaWeights{
		InProjQKV: qkv, ConvWeight: convW, ConvBias: f32opt(sp + "conv1d.bias"),
		InProjA: inA, ALog: aLog, DtBias: f32opt(sp + "dt_bias"),
		InProjB: inB, InProjZ: inZ, Norm: norm, OutProj: outP,
	}
	cfg := qwen3.GatedDeltaConfig{KeyHeads: keyHeads, ValueHeads: valueHeads, HeadDim: headDim, ConvKernel: convK, Eps: 1e-6}
	return NewGatedDeltaMixer(w, cfg), nil
}

// buildFFN builds a layer's feed-forward: a MoE (qwen3_6_moe) when expert weights are present, else a
// dense SwiGLU MLP. sp is the "…mlp." prefix.
func buildFFN(get func(string) (safetensors.Tensor, bool), f32 func(string) ([]float32, error), sp string, cfg *loaderConfig, D int) (FFN, error) {
	if _, ok := get(sp + "experts.0.gate_proj.weight"); ok {
		return buildMoE(get, f32, sp, cfg, D)
	}
	gate, err := f32(sp + "gate_proj.weight")
	if err != nil {
		return nil, err
	}
	up, err := f32(sp + "up_proj.weight")
	if err != nil {
		return nil, err
	}
	down, err := f32(sp + "down_proj.weight")
	if err != nil {
		return nil, err
	}
	return &MLP{Gate: gate, Up: up, Down: down, FF: len(gate) / D}, nil
}

// buildMoE loads the MoE FFN: router (mlp.gate.weight), the experts (mlp.experts.E.*), and the optional
// shared expert (mlp.shared_expert.*). TopK = num_experts_per_tok.
func buildMoE(get func(string) (safetensors.Tensor, bool), f32 func(string) ([]float32, error), sp string, cfg *loaderConfig, D int) (FFN, error) {
	router, err := f32(sp + "gate.weight")
	if err != nil {
		return nil, err
	}
	expert := func(p string) (MoEExpert, error) {
		g, e1 := f32(p + "gate_proj.weight")
		u, e2 := f32(p + "up_proj.weight")
		d, e3 := f32(p + "down_proj.weight")
		for _, e := range []error{e1, e2, e3} {
			if e != nil {
				return MoEExpert{}, e
			}
		}
		return MoEExpert{Gate: g, Up: u, Down: d}, nil
	}
	var experts []MoEExpert
	for e := 0; ; e++ {
		ep := sp + core.Sprintf("experts.%d.", e)
		if _, ok := get(ep + "gate_proj.weight"); !ok {
			break
		}
		ex, err := expert(ep)
		if err != nil {
			return nil, err
		}
		experts = append(experts, ex)
	}
	if len(experts) == 0 {
		return nil, core.NewError("composed.buildMoE: experts.0 present but none loaded")
	}
	var shared *MoEExpert
	if _, ok := get(sp + "shared_expert.gate_proj.weight"); ok {
		ex, err := expert(sp + "shared_expert.")
		if err != nil {
			return nil, err
		}
		shared = &ex
	}
	topK := cfg.NumExpertsPerTok
	if topK <= 0 {
		topK = 8
	}
	return &MoEMLP{Router: router, Experts: experts, Shared: shared, TopK: topK}, nil
}
