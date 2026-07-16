// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/arch/Qwen/qwen3"
	"dappco.re/go/inference/model/quant/mlxaffine"
	"dappco.re/go/inference/model/safetensors"
)

// loader.go builds a ComposedModel from a hybrid checkpoint (Qwen 3.6), the native port of metal's
// composed.buildComposed: parse the config, dispatch each layer by layer_type to its mixer (linear_attn →
// gated-delta, self_attn → attention), wire the SwiGLU MLP + the two norms, and resolve the wrapper
// prefixes (model.language_model. via the prefix probe; the language_model.model. nesting real Qwen 3.6
// packs ship via model.NormalizeWrapperNames). The gated-delta geometry is derived from the weight shapes
// (as metal does); the attention geometry from the config.
//
// Quantised checkpoints (mlx affine packed-uint32 weights with .scales/.biases siblings) keep their 2-D
// PROJECTION weights PACKED — carried as model.QuantWeight to the engine's quant matvec seam rather than
// widened (a 27B checkpoint dequantised to f32 is ~110 GB, dead on arrival). The small tensors (norms,
// conv kernels, delta/gating params, biases) stay host f32 as before — the mixers' state math is exact
// host f32. The embedding table stays packed too (dequantised one row per token at gather time). Bonsai's
// 1-bit packs are widened to 2-bit at load (RepackB1ToB2, exact) so the engine dispatches stock kernels.
// This lane is TEXT-ONLY: vision_tower.* tensors are never looked up (skipped by name), so a multimodal
// wrapper's image tower is dropped at load and image input keeps refusing through the text-only machinery.

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
	case "F16", "float16":
		out := make([]float32, len(t.Data)/2)
		for i := range out {
			out[i] = f16ToF32(uint16(t.Data[2*i]) | uint16(t.Data[2*i+1])<<8)
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

// f16ToF32 decodes an IEEE-754 half (F16: sign 1, exp 5, mantissa 10) to float32. mlx 1-bit packs (e.g.
// prism-ml Bonsai) store their norms + quant scales/biases as F16 where the 4-bit packs use BF16.
func f16ToF32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1f
	man := uint32(h & 0x3ff)
	switch exp {
	case 0:
		if man == 0 {
			return math.Float32frombits(sign) // ±0
		}
		exp = 1 // subnormal: normalise into a f32 normal
		for man&0x400 == 0 {
			man <<= 1
			exp--
		}
		man &= 0x3ff
		return math.Float32frombits(sign | (exp+112)<<23 | man<<13)
	case 0x1f:
		return math.Float32frombits(sign | 0x7f800000 | man<<13) // inf/nan
	default:
		return math.Float32frombits(sign | (exp+112)<<23 | man<<13)
	}
}

// bf16Bytes returns t's data as BF16 bytes — the tier the engine's affine_qmv/qmm_t kernels and
// mlxaffine.DequantizeTensor read quant scales/biases at. A BF16 tensor passes verbatim; an F16 tensor
// (Bonsai's sidecars) is decoded to f32 and re-rounded to bf16 (round-to-nearest-even); an F32 tensor is
// rounded down. The F16→BF16 step drops 3 mantissa bits of the per-group scale — negligible against a
// 1-bit weight's own error, and it normalises the whole lane onto one scale dtype.
func bf16Bytes(t safetensors.Tensor) ([]byte, error) {
	f32ToBF16 := func(b uint32) (byte, byte) {
		r := uint16((b + 0x7fff + ((b >> 16) & 1)) >> 16)
		return byte(r), byte(r >> 8)
	}
	switch t.Dtype {
	case "BF16", "bfloat16":
		return append([]byte(nil), t.Data...), nil
	case "F16", "float16":
		out := make([]byte, len(t.Data))
		for i := 0; i < len(t.Data)/2; i++ {
			f := f16ToF32(uint16(t.Data[2*i]) | uint16(t.Data[2*i+1])<<8)
			out[2*i], out[2*i+1] = f32ToBF16(math.Float32bits(f))
		}
		return out, nil
	case "F32", "float32":
		out := make([]byte, len(t.Data)/2)
		for i := 0; i < len(t.Data)/4; i++ {
			b := uint32(t.Data[4*i]) | uint32(t.Data[4*i+1])<<8 | uint32(t.Data[4*i+2])<<16 | uint32(t.Data[4*i+3])<<24
			out[2*i], out[2*i+1] = f32ToBF16(b)
		}
		return out, nil
	}
	return nil, core.NewError("composed.bf16Bytes: unsupported scales/biases dtype " + t.Dtype)
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

// tensorAsQuant returns name's projection weight kept PACKED when it is an mlx affine quantised tensor
// (a packed-uint32 weight with .scales/.biases siblings), else (nil, false, nil) so the caller widens the
// dense tensor. The (groupSize, bits) come from the quantization block and are cross-checked against the
// packed shape so a mismatched config/checkpoint pairing fails loudly. Bonsai's 1-bit packs are widened to
// 2-bit here (RepackB1ToB2 — exact: w = scale·q + bias unchanged) so the engine dispatches the stock b_2
// kernels rather than a b_1 kernel that does not ship.
//
// zeroCopy chooses the packed-weight memory. false (the default LoadComposed contract): the QuantWeight
// OWNS a heap copy, so it outlives the input tensors — the legacy metal loader relies on this (it unmaps
// the shard mmap right after the build). true (the LoadComposedDir path): the packed bytes ALIAS t.Data —
// an mmap view when the checkpoint was mapped — with NO heap copy, cutting the load-time RSS by the packed
// weight (the dominant term of a quant checkpoint); the model then owns that mapping and unmaps it on Close.
// The returned bool reports whether the weight aliases t.Data (true only when zeroCopy AND the codes were
// not repacked — a b1→b2 repack rewrites into owned heap buffers, so a repacked weight never aliases).
func tensorAsQuant(tensors map[string]safetensors.Tensor, name string, t safetensors.Tensor, quant *model.QuantConfig, zeroCopy bool) (*model.QuantWeight, bool, error) {
	base := name
	if core.HasSuffix(base, ".weight") {
		base = base[:len(base)-len(".weight")]
	}
	scalesT, sOK := tensors[base+".scales"]
	biasesT, bOK := tensors[base+".biases"]
	if !sOK || !bOK {
		return nil, false, nil // dense tensor — caller widens with tensorF32
	}
	if quant == nil {
		return nil, false, core.NewError("composed.tensorAsQuant: " + name + " carries .scales/.biases but the config has no quantization block")
	}
	if len(t.Shape) != 2 || len(scalesT.Shape) != 2 {
		return nil, false, core.NewError("composed.tensorAsQuant: quantised " + name + " is not 2-D")
	}
	gs, bits := quant.For(base)
	outDim, packedCols := t.Shape[0], t.Shape[1]
	inDim := scalesT.Shape[1] * gs
	if packedCols*32 != inDim*bits {
		return nil, false, core.NewError(core.Sprintf("composed.tensorAsQuant: %s packed cols %d ≠ inDim %d·bits %d/32 (groupSize %d)", name, packedCols, inDim, bits, gs))
	}
	var packed []byte
	aliased := false
	if zeroCopy {
		packed, aliased = t.Data, true // view the (mmap'd) checkpoint bytes — no heap copy
	} else {
		packed = append([]byte(nil), t.Data...) // owned copy — the QuantWeight outlives the input tensors
	}
	scales, err := bf16Bytes(scalesT) // normalise F16 sidecars (Bonsai) to the bf16 tier the kernels read
	if err != nil {
		return nil, false, core.E("composed.tensorAsQuant", name+" scales", err)
	}
	biases, err := bf16Bytes(biasesT)
	if err != nil {
		return nil, false, core.E("composed.tensorAsQuant", name+" biases", err)
	}
	if bits == 1 {
		if packed, scales, biases, err = mlxaffine.RepackB1ToB2(packed, scales, biases, outDim, inDim, gs); err != nil {
			return nil, false, core.E("composed.tensorAsQuant", name+" b1→b2 repack", err)
		}
		bits = 2
		aliased = false // repacked into owned heap buffers — no longer a view into the checkpoint
	}
	return &model.QuantWeight{Packed: packed, Scales: scales, Biases: biases, Bits: bits, GroupSize: gs, OutDim: outDim, InDim: inDim}, aliased, nil
}

// LoadComposed assembles a ComposedModel from a hybrid checkpoint's tensors + its config.json bytes. The
// packed quant weights are COPIED to owned heap buffers, so the model outlives its input tensors — the
// contract the legacy metal loader depends on (it unmaps the shard mmap right after the build). The
// registry path (model.LoadComposedDir) instead uses the zero-copy build, which aliases the mmap.
func LoadComposed(tensors map[string]safetensors.Tensor, configJSON []byte) (*ComposedModel, error) {
	return loadComposed(tensors, configJSON, nil, false)
}

// LoadComposedWithArch assembles a composed model while consuming the neutral MoE policy declared by an
// architecture package. The policy is validated against the checkpoint instead of re-assuming one family's
// router behaviour from tensor names. Like LoadComposed, packed weights are COPIED to owned buffers (not
// aliased) — the owned-copy contract for a caller that unmaps the checkpoint right after the build. The
// registry path (model.LoadComposedDir → the MoE arch hooks) instead uses LoadComposedWithArchMmap, which
// aliases the mmap zero-copy.
func LoadComposedWithArch(tensors map[string]safetensors.Tensor, configJSON []byte, arch model.Arch) (*ComposedModel, error) {
	return loadComposed(tensors, configJSON, &arch, false)
}

// LoadComposedWithArchMmap is the ZERO-COPY arch-aware build the registry (model.LoadComposedDir) uses for
// the MoE architectures (dbrx, olmoe, qwenmoe, granitemoe, llama4). Like LoadComposedWithArch it consumes
// the arch's neutral MoE policy, but the packed quant PROJECTION weights (attention q/k/v/o, embed, lm_head)
// VIEW the mapped checkpoint — QuantWeight.Packed aliases the input tensors' mmap region — instead of being
// copied to the heap, so a quant MoE checkpoint's load-time RSS drops by those packed weights. The model
// takes ownership of the mapping through the SAME RetainMmap handshake as the base path and unmaps it on
// Close/finalize; when nothing aliases (a dense checkpoint, or an all-1-bit pack repacked to owned heap)
// RetainMmap declines and LoadComposedDir unmaps immediately. The MoE expert weights themselves stay on the
// dequant-to-f32 path (buildFFN routes experts through buildMoE — packed-expert MoE is a later slice), so the
// aliased weights are the dense-projection quant weights. Use LoadComposedWithArch (copying) when the caller
// unmaps the checkpoint right after the build.
func LoadComposedWithArchMmap(tensors map[string]safetensors.Tensor, configJSON []byte, arch model.Arch) (*ComposedModel, error) {
	return loadComposed(tensors, configJSON, &arch, true)
}

// loadComposed builds the model. zeroCopy true keeps the packed quant weights as VIEWS into the input
// tensors (an mmap region on the LoadComposedDir path) instead of copying them to the heap — the RSS win.
// It sets ComposedModel.mmapAliased when any weight ends up aliasing, so the loader knows to retain the
// checkpoint mapping for the model's lifetime (model.LoadComposedDir hands it over via RetainMmap).
func loadComposed(tensors map[string]safetensors.Tensor, configJSON []byte, arch *model.Arch, zeroCopy bool) (*ComposedModel, error) {
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
	// anyAlias records whether any packed weight ended up as a view into the input tensors (zero-copy),
	// so the model can retain the checkpoint mapping for its lifetime rather than let it be unmapped.
	var anyAlias bool
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
	// proj resolves a 2-D projection weight: PACKED (quant checkpoint) or widened to f32 — exactly one of
	// the two returns is non-nil. Small unquantised tensors (norms, conv, biases) keep the f32/f32opt path.
	proj := func(name string) ([]float32, *model.QuantWeight, error) {
		t, ok := get(name)
		if !ok {
			return nil, nil, core.NewError("composed.LoadComposed: missing " + name)
		}
		qw, aliased, err := tensorAsQuant(tensors, name, t, quant, zeroCopy)
		if err != nil {
			return nil, nil, err
		}
		if qw != nil {
			anyAlias = anyAlias || aliased
			return nil, qw, nil
		}
		f, err := tensorF32(t)
		return f, nil, err
	}
	projOpt := func(name string) ([]float32, *model.QuantWeight) {
		if _, ok := get(name); !ok {
			return nil, nil
		}
		f, qw, err := proj(name)
		if err != nil {
			return nil, nil
		}
		return f, qw
	}

	embedT, ok := get(prefix + "embed_tokens.weight")
	if !ok || len(embedT.Shape) != 2 {
		return nil, core.NewError("composed.LoadComposed: missing/!2D embed_tokens.weight")
	}
	vocab := embedT.Shape[0]
	embed, embedQ, err := proj(prefix + "embed_tokens.weight")
	if err != nil {
		return nil, err
	}
	// Logical width: a dense embed's dequantised length / vocab; a packed embed's InDim (its Shape[1] is
	// the bits-compressed packed-word count, not the hidden size).
	D := 0
	if embedQ != nil {
		D = embedQ.InDim
	} else {
		D = len(embed) / vocab
	}
	normF, err := f32(prefix + "norm.weight")
	if err != nil {
		return nil, err
	}
	output, outputQ := projOpt("lm_head.weight") // untied; nil/nil ⇒ tied to embed

	kinds, err := resolveKinds(cfg)
	if err != nil {
		return nil, err
	}

	isCohere := cfg.ModelType == "cohere" || cfg.ModelType == "cohere2"
	m := &ComposedModel{Embed: embed, EmbedQ: embedQ, NormF: normF, Output: output, OutputQ: outputQ, D: D, Vocab: vocab, Eps: cfg.RMSNormEps, LayerNorm: isCohere || cfg.UseLayerNorm, ParallelResidual: isCohere, LogitScale: cfg.LogitScale, Quantised: quant != nil}
	if arch != nil {
		m.EmbedScale = arch.EmbedScale
		m.LogitsScaling = arch.LogitsScaling
		m.ResidualScale = arch.ResidualMultiplier
	}
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

	// Vision: additive-only. buildVisionTower probes tensors for vision_tower.*/multi_modal_projector.*
	// and returns (nil, nil) when neither is present, so a text-only checkpoint (the whole suite before
	// this) loads with m.Vision nil exactly as it always did. raw (not cfg) carries the WRAPPER-level
	// vision_config + image/video token ids — cfg is already narrowed to the text_config side.
	if vision, verr := buildVisionTower(tensors, raw.VisionConfig, D); verr != nil {
		return nil, core.E("composed.LoadComposed", "vision tower", verr)
	} else if vision != nil {
		m.Vision = vision
		m.ImageTokenID = int32(raw.ImageTokenID)
		// The Qwen-VL family's own stable special-token spellings (config.json carries only the numeric
		// ids, not the text — see composed.ChatMLDialect's identical hardcoding for the ChatML turn
		// markers). Every composed arch that ships vision_tower.* tensors today is a Qwen 3.6-family
		// checkpoint (register.go's ModelTypes), so gating on Vision's presence rather than model_type is
		// equivalent and keeps this loader arch-agnostic.
		m.VisionBeginToken, m.VisionToken, m.VisionEndToken = qwenVisionBeginToken, qwenVisionToken, qwenVisionEndToken
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
		ffn, err := buildFFN(get, proj, f32, lp+"mlp.", cfg, arch, D)
		if err != nil {
			return nil, core.E("composed.LoadComposed", core.Sprintf("layer %d ffn", i), err)
		}

		var mixer Mixer
		if kinds[i] == "full_attention" || kinds[i] == "sliding_attention" {
			mixer, err = buildAttn(proj, f32opt, lp+"self_attn.", cfg, arch, i, D, kinds[i])
		} else {
			mixer, err = buildGatedDelta(get, proj, f32opt, lp+"linear_attn.", cfg, D)
		}
		if err != nil {
			return nil, core.E("composed.LoadComposed", core.Sprintf("layer %d (%s)", i, kinds[i]), err)
		}
		m.Layers = append(m.Layers, Layer{
			InputNorm: inNorm, Mixer: mixer, PostAttnNorm: postNorm, MLP: ffn,
		})
	}
	m.mmapAliased = anyAlias
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

// projFn resolves a 2-D projection weight to either its widened f32 slice OR a packed
// model.QuantWeight (exactly one non-nil) — the closure the loader threads into the mixer/FFN
// builders so a quant checkpoint keeps its projections packed while a dense one widens.
type projFn func(string) ([]float32, *model.QuantWeight, error)

// buildAttn builds a full-attention mixer; geometry from the config.
func buildAttn(proj projFn, f32opt func(string) []float32, sp string, cfg *loaderConfig, arch *model.Arch, layer, D int, kind string) (Mixer, error) {
	qF, qQ, err := proj(sp + "q_proj.weight")
	if err != nil {
		return nil, err
	}
	kF, kQ, err := proj(sp + "k_proj.weight")
	if err != nil {
		return nil, err
	}
	vF, vQ, err := proj(sp + "v_proj.weight")
	if err != nil {
		return nil, err
	}
	oF, oQ, err := proj(sp + "o_proj.weight")
	if err != nil {
		return nil, err
	}
	heads := cfg.NumAttentionHeads
	headDim := cfg.HeadDim
	if headDim == 0 && heads > 0 {
		qCols := len(qF) / D // rows of q_proj = heads·headDim (2× when gated); a packed q_proj reads it from OutDim
		if qQ != nil {
			qCols = qQ.OutDim
		}
		headDim = qCols / heads
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
		QProj: qF, KProj: kF, VProj: vF, OProj: oF,
		QProjQ: qQ, KProjQ: kQ, VProjQ: vQ, OProjQ: oQ,
		QNorm: f32opt(sp + "q_norm.weight"), KNorm: f32opt(sp + "k_norm.weight"),
	}, AttnConfig{Heads: heads, KVHeads: kvHeads, HeadDim: headDim, RotaryDim: rd, RopeTheta: cfg.ropeTheta(), QKVClip: qkvClip, NormEps: func() float32 {
		if cfg.LayerNormEps > 0 {
			return cfg.LayerNormEps
		}
		return cfg.RMSNormEps
	}(), OutputGate: cfg.AttnOutputGate, QKNormalization: qk, SlidingWindow: window}), nil
}

// buildGatedDelta builds a gated-delta mixer; geometry derived from the weight shapes (as metal does):
// ValueHeads = len(A_log), HeadDim = len(norm), convDim/K from conv1d.weight, qDim = (convDim−vDim)/2,
// KeyHeads = qDim/HeadDim.
func buildGatedDelta(get func(string) (safetensors.Tensor, bool), proj projFn, f32opt func(string) []float32, sp string, lcfg *loaderConfig, D int) (Mixer, error) {
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

	qkvF, qkvQ, err := proj(sp + "in_proj_qkv.weight")
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
	inAF, inAQ, err := proj(sp + "in_proj_a.weight")
	if err != nil {
		return nil, err
	}
	inBF, inBQ, err := proj(sp + "in_proj_b.weight")
	if err != nil {
		return nil, err
	}
	inZF, inZQ, err := proj(sp + "in_proj_z.weight")
	if err != nil {
		return nil, err
	}
	outPF, outPQ, err := proj(sp + "out_proj.weight")
	if err != nil {
		return nil, err
	}
	w := &qwen3.GatedDeltaWeights{
		InProjQKV: qkvF, InProjQKVQ: qkvQ, ConvWeight: convW, ConvBias: f32opt(sp + "conv1d.bias"),
		InProjA: inAF, InProjAQ: inAQ, ALog: aLog, DtBias: f32opt(sp + "dt_bias"),
		InProjB: inBF, InProjBQ: inBQ, InProjZ: inZF, InProjZQ: inZQ, Norm: norm, OutProj: outPF, OutProjQ: outPQ,
	}
	cfg := qwen3.GatedDeltaConfig{KeyHeads: keyHeads, ValueHeads: valueHeads, HeadDim: headDim, ConvKernel: convK, Eps: 1e-6}
	return NewGatedDeltaMixer(w, cfg), nil
}

// buildFFN builds a layer's feed-forward: a MoE (qwen3_6_moe) when expert weights are present, else a
// dense SwiGLU MLP. sp is the "…mlp." prefix.
func buildFFN(get func(string) (safetensors.Tensor, bool), proj projFn, f32 func(string) ([]float32, error), sp string, cfg *loaderConfig, arch *model.Arch, D int) (FFN, error) {
	if _, ok := get(sp + "experts.0.gate_proj.weight"); ok {
		// MoE stays on the dequant path (quant MoE is a later slice; the dense acceptance models never reach here).
		return buildMoE(get, f32, sp, cfg, arch, D)
	}
	gateF, gateQ, err := proj(sp + "gate_proj.weight")
	if err != nil {
		return nil, err
	}
	upF, upQ, err := proj(sp + "up_proj.weight")
	if err != nil {
		return nil, err
	}
	downF, downQ, err := proj(sp + "down_proj.weight")
	if err != nil {
		return nil, err
	}
	ff := len(gateF) / D // gate_proj rows = FF; a packed gate reads it from OutDim
	if gateQ != nil {
		ff = gateQ.OutDim
	}
	return &MLP{Gate: gateF, Up: upF, Down: downF, GateQ: gateQ, UpQ: upQ, DownQ: downQ, FF: ff}, nil
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
