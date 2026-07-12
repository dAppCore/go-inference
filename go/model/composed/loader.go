// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/quant/mlxaffine"
	"dappco.re/go/inference/model/qwen3"
	"dappco.re/go/inference/model/safetensors"
)

// loader.go builds a ComposedModel from a hybrid checkpoint (Qwen 3.6), the native port of metal's
// composed.buildComposed: parse the config, dispatch each layer by layer_type to its mixer (linear_attn →
// gated-delta, self_attn → attention), wire the SwiGLU MLP + the two norms, and resolve the wrapper
// prefixes (model.language_model. via the prefix probe; the language_model.model. nesting real Qwen 3.6
// packs ship via model.NormalizeWrapperNames). Quantised checkpoints (mlx affine packed-uint32 weights
// with .scales/.biases siblings) dequantise host-side to f32 at load. The gated-delta geometry is derived
// from the weight shapes (as metal does); the attention geometry from the config.

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

// quantBlock extracts the checkpoint's mlx quantization block (top-level, else nested under
// text_config in the multimodal wrapper); nil for an unquantised (bf16/f32) checkpoint.
func quantBlock(configJSON []byte) *model.QuantConfig {
	var probe struct {
		Quantization *model.QuantConfig `json:"quantization"`
		TextConfig   struct {
			Quantization *model.QuantConfig `json:"quantization"`
		} `json:"text_config"`
	}
	if r := core.JSONUnmarshal(configJSON, &probe); !r.OK {
		return nil
	}
	if probe.Quantization != nil {
		return probe.Quantization
	}
	return probe.TextConfig.Quantization
}

// tensorAsF32 widens t (looked up as name) to a flat f32 slice. A quantised tensor — mlx
// affine's packed-uint32 weight with .scales/.biases siblings — dequantises host-side, its
// (groupSize, bits) read from the quantization block and cross-checked against the packed
// shape so a mismatched config/checkpoint pairing fails loudly rather than mis-loading.
func tensorAsF32(tensors map[string]safetensors.Tensor, name string, t safetensors.Tensor, quant *model.QuantConfig) ([]float32, error) {
	base := name
	if core.HasSuffix(base, ".weight") {
		base = base[:len(base)-len(".weight")]
	}
	scalesT, sOK := tensors[base+".scales"]
	biasesT, bOK := tensors[base+".biases"]
	if !sOK || !bOK {
		return tensorF32(t)
	}
	if quant == nil {
		return nil, core.NewError("composed.tensorAsF32: " + name + " carries .scales/.biases but the config has no quantization block")
	}
	if len(t.Shape) != 2 || len(scalesT.Shape) != 2 {
		return nil, core.NewError("composed.tensorAsF32: quantised " + name + " is not 2-D")
	}
	gs, bits := quant.For(base)
	outDim, packedCols := t.Shape[0], t.Shape[1]
	inDim := scalesT.Shape[1] * gs
	if packedCols*32 != inDim*bits {
		return nil, core.NewError(core.Sprintf("composed.tensorAsF32: %s packed cols %d ≠ inDim %d·bits %d/32 (groupSize %d)", name, packedCols, inDim, bits, gs))
	}
	return mlxaffine.DequantizeTensor(t.Data, scalesT.Data, biasesT.Data, outDim, inDim, bits, gs)
}

// LoadComposed assembles a ComposedModel from a hybrid checkpoint's tensors + its config.json bytes.
func LoadComposed(tensors map[string]safetensors.Tensor, configJSON []byte) (*ComposedModel, error) {
	return loadComposed(tensors, configJSON, nil)
}

// LoadComposedWithArch assembles a composed model while consuming the neutral MoE
// policy declared by an architecture package. The policy is validated against the
// checkpoint instead of re-assuming one family's router behaviour from tensor names.
func LoadComposedWithArch(tensors map[string]safetensors.Tensor, configJSON []byte, arch model.Arch) (*ComposedModel, error) {
	return loadComposed(tensors, configJSON, &arch)
}

func loadComposed(tensors map[string]safetensors.Tensor, configJSON []byte, arch *model.Arch) (*ComposedModel, error) {
	var raw loaderConfig
	if r := core.JSONUnmarshal(configJSON, &raw); !r.OK {
		return nil, core.NewError("composed.LoadComposed: config.json parse failed")
	}
	cfg := raw.effective()
	if cfg.HiddenSize <= 0 || cfg.NumHiddenLayers <= 0 {
		return nil, core.NewError("composed.LoadComposed: hidden_size and num_hidden_layers required")
	}

	// Real Qwen 3.6 packs nest the text model under language_model. (language_model.model.layers…);
	// the normaliser adds the stripped model.… aliases so the bare lookups below work either way.
	tensors = model.NormalizeWrapperNames(tensors)
	quant := quantBlock(configJSON)

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
		return tensorAsF32(tensors, name, t, quant)
	}
	f32opt := func(name string) []float32 {
		if t, ok := get(name); ok {
			if v, e := tensorAsF32(tensors, name, t, quant); e == nil {
				return v
			}
		}
		return nil
	}

	embedT, ok := get(prefix + "embed_tokens.weight")
	if !ok || len(embedT.Shape) != 2 {
		return nil, core.NewError("composed.LoadComposed: missing/!2D embed_tokens.weight")
	}
	embed, err := tensorAsF32(tensors, prefix+"embed_tokens.weight", embedT, quant)
	if err != nil {
		return nil, err
	}
	vocab := embedT.Shape[0]
	// Logical width from the dequantised length — a quantised embed's Shape[1] is the
	// bits-compressed packed-word count, not the hidden size.
	D := len(embed) / vocab
	normF, err := f32(prefix + "norm.weight")
	if err != nil {
		return nil, err
	}
	output := f32opt("lm_head.weight") // untied; nil ⇒ tied to embed

	kinds, err := resolveKinds(cfg)
	if err != nil {
		return nil, err
	}

	isCohere := cfg.ModelType == "cohere" || cfg.ModelType == "cohere2"
<<<<<<< HEAD
	m := &ComposedModel{Embed: embed, NormF: normF, Output: output, D: D, Vocab: vocab, Eps: cfg.RMSNormEps, LayerNorm: isCohere, ParallelResidual: isCohere, LogitScale: cfg.LogitScale}
	if arch != nil {
		m.EmbedScale = arch.EmbedScale
		m.LogitsScaling = arch.LogitsScaling
		m.ResidualScale = arch.ResidualMultiplier
	}
=======
	m := &ComposedModel{Embed: embed, NormF: normF, Output: output, D: D, Vocab: vocab, Eps: cfg.RMSNormEps, LayerNorm: isCohere || cfg.UseLayerNorm, ParallelResidual: isCohere, LogitScale: cfg.LogitScale}
>>>>>>> lane/moe-dbrx
	if isCohere {
		m.Eps = cfg.LayerNormEps
		if m.Eps == 0 {
			m.Eps = 1e-5
		}
		if m.LogitScale == 0 {
			m.LogitScale = 0.0625
		}
	}
	if m.Eps == 0 {
		m.Eps = 1e-6
	}
	for i := 0; i < cfg.NumHiddenLayers; i++ {
		lp := prefix + core.Sprintf("layers.%d.", i)
		inNorm, err := f32(lp + "input_layernorm.weight")
		if err != nil {
			return nil, err
		}
		var postNorm []float32
		if !isCohere {
			postNorm, err = f32(lp + "post_attention_layernorm.weight")
			if err != nil {
				return nil, err
			}
		}
		ffn, err := buildFFN(get, f32, lp+"mlp.", cfg, arch, D)
		if err != nil {
			return nil, core.E("composed.LoadComposed", core.Sprintf("layer %d ffn", i), err)
		}

		var mixer Mixer
		if kinds[i] == "full_attention" || kinds[i] == "sliding_attention" {
<<<<<<< HEAD
<<<<<<< HEAD
			mixer, err = buildAttn(f32, f32opt, lp+"self_attn.", cfg, arch, D, kinds[i])
=======
			mixer, err = buildAttn(f32, f32opt, lp+"self_attn.", cfg, arch, i, D, kinds[i])
>>>>>>> lane/llama4-text
=======
			mixer, err = buildAttn(f32, f32opt, lp+"self_attn.", cfg, arch, D, kinds[i])
>>>>>>> lane/moe-dbrx
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
	if cfg.ModelType == "cohere2" && cfg.SlidingWindow > 0 {
		pattern := cfg.SlidingWindowPattern
		if pattern == 0 {
			pattern = 4
		}
		for i := range out {
			out[i] = "sliding_attention"
			if (i+1)%pattern == 0 {
				out[i] = "full_attention"
			}
		}
		return out, nil
	}
	// Dense-attention families such as Llama and Mixtral omit both hybrid selectors:
	// every layer is full attention.
	for i := range out {
		out[i] = "full_attention"
	}
	return out, nil
}

// buildAttn builds a full-attention mixer; geometry from the config.
<<<<<<< HEAD
<<<<<<< HEAD
func buildAttn(f32 func(string) ([]float32, error), f32opt func(string) []float32, sp string, cfg *loaderConfig, arch *model.Arch, D int, kind string) (Mixer, error) {
=======
func buildAttn(f32 func(string) ([]float32, error), f32opt func(string) []float32, sp string, cfg *loaderConfig, arch *model.Arch, layer, D int, kind string) (Mixer, error) {
>>>>>>> lane/llama4-text
=======
func buildAttn(f32 func(string) ([]float32, error), f32opt func(string) []float32, sp string, cfg *loaderConfig, arch *model.Arch, D int, kind string) (Mixer, error) {
>>>>>>> lane/moe-dbrx
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
		if cfg.AttnOutputGate {
			headDim /= 2 // gated q_proj emits [q ; gate], so its rows are 2·heads·headDim
		}
	}
	kvHeads := cfg.NumKeyValueHeads
	if kvHeads == 0 {
		kvHeads = heads
	}
	rd := int(cfg.partialRotary() * float32(headDim))
	if cfg.ModelType == "cohere2" && kind == "full_attention" {
		rd = 0
	}
	if rd%2 != 0 {
		rd--
	}
	qk := model.QKNone
	if cfg.ModelType == "cohere" && cfg.UseQKNorm != nil && *cfg.UseQKNorm {
		qk = model.QKLayerNorm
	}
	if arch != nil {
		qk = arch.QKNormalization
		if layer < len(arch.Layer) && arch.Layer[layer].DisableRotary {
			rd = 0
		}
	}
	window := 0
	if kind == "sliding_attention" {
		window = cfg.SlidingWindow
	}
	qkvClip := cfg.QKVClip
	if arch != nil && arch.QKVClip > 0 {
		qkvClip = arch.QKVClip
	}
	return NewAttnMixer(&AttnWeights{
		QProj: q, KProj: k, VProj: v, OProj: o,
		QNorm: f32opt(sp + "q_norm.weight"), KNorm: f32opt(sp + "k_norm.weight"),
<<<<<<< HEAD
	}, AttnConfig{Heads: heads, KVHeads: kvHeads, HeadDim: headDim, RotaryDim: rd, RopeTheta: cfg.ropeTheta(), Scale: func() float32 {
		if arch != nil {
			return arch.AttnScale
		}
		return 0
	}(), NormEps: func() float32 {
=======
	}, AttnConfig{Heads: heads, KVHeads: kvHeads, HeadDim: headDim, RotaryDim: rd, RopeTheta: cfg.ropeTheta(), QKVClip: qkvClip, NormEps: func() float32 {
>>>>>>> lane/moe-dbrx
		if cfg.LayerNormEps > 0 {
			return cfg.LayerNormEps
		}
		return cfg.RMSNormEps
	}(), OutputGate: cfg.AttnOutputGate, QKNormalization: qk, SlidingWindow: window}), nil
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
	if len(convT.Shape) == 3 && convK == 1 {
		// mlx packs the depthwise conv channel-last ([convDim, K, 1]); torch packs [convDim, 1, K].
		// The flat bytes are identical (the 1-dim contributes no stride) — only K's slot moves.
		convK = convT.Shape[1]
	}
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
func buildFFN(get func(string) (safetensors.Tensor, bool), f32 func(string) ([]float32, error), sp string, cfg *loaderConfig, arch *model.Arch, D int) (FFN, error) {
	if _, ok := get(sp + "experts.0.gate_proj.weight"); ok {
		return buildMoE(get, f32, sp, cfg, arch, D)
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
func buildMoE(get func(string) (safetensors.Tensor, bool), f32 func(string) ([]float32, error), sp string, cfg *loaderConfig, arch *model.Arch, D int) (FFN, error) {
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
	if arch != nil {
		if arch.MoEGating != "" && arch.MoEGating != model.MoEGatingSoftmax && arch.MoEGating != model.MoEGatingSigmoid {
			return nil, core.NewError("composed.buildMoE: unsupported router score function " + string(arch.MoEGating))
		}
		if arch.Experts > 0 && len(experts) != arch.Experts {
			return nil, core.NewError(core.Sprintf("composed.buildMoE: checkpoint experts %d != architecture %d", len(experts), arch.Experts))
		}
	}
	var shared *MoEExpert
	var sharedGate []float32
	if _, ok := get(sp + "shared_expert.gate_proj.weight"); ok {
		ex, err := expert(sp + "shared_expert.")
		if err != nil {
			return nil, err
		}
		shared = &ex
		// shared_expert_gate is the reference's sigmoid gate on the shared expert (Linear(hidden,1)
		// ⇒ [D]); optional so a checkpoint without it (or a synthetic test) adds the shared expert
		// ungated.
		if t, ok := get(sp + "shared_expert_gate.weight"); ok {
			if v, e := tensorF32(t); e == nil {
				sharedGate = v
			}
		}
	}
	sharedExperts := 0
	if shared != nil {
		sharedExperts = 1
	}
	if arch != nil && sharedExperts != arch.SharedExperts {
		return nil, core.NewError(core.Sprintf("composed.buildMoE: checkpoint shared experts %d != architecture %d", sharedExperts, arch.SharedExperts))
	}
	topK := cfg.NumExpertsPerTok
	normTopK := cfg.normTopKProb()
	if arch != nil {
		topK = arch.TopK
		normTopK = arch.NormaliseMoETopK
	}
	if topK <= 0 {
		topK = 8
	}
	gating := model.MoEGatingSoftmax
	if arch != nil && arch.MoEGating != "" {
		gating = arch.MoEGating
	}
	return &MoEMLP{Router: router, Experts: experts, Shared: shared, SharedGate: sharedGate, TopK: topK, NormTopKProb: normTopK, Gating: gating}, nil
}
