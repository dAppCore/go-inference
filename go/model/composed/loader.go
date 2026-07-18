// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
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
// host f32. The embedding table stays packed too (dequantised one row per token at gather time). A MoE
// FFN's routed + shared experts (buildMoE) resolve through the SAME proj closure as the dense MLP's
// gate/up/down, so a quant checkpoint's experts stay packed too — the dominant tensor class of a grouped
// checkpoint (an ~8x blow-up widened), and the actual blocker for serving bigger-than-RAM sparse MoE.
// Bonsai's 1-bit packs are widened to 2-bit at load (RepackB1ToB2, exact) so the engine dispatches stock
// kernels.
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

// composedNormsZeroCentred reports whether this checkpoint stores its RMSNorm gammas zero-centred
// (the official Qwen 3.5/3.6 export): true when any mtp.* head tensor survives or any conv1d.weight
// still has the torch [ch,1,K] layout — exactly mlx-lm qwen3_5.sanitize's should_shift_norm_weights
// test, which the conversions this loader also serves have already been through (mtp stripped, conv
// moved, norms pre-baked).
func composedNormsZeroCentred(tensors map[string]safetensors.Tensor) bool {
	for name, t := range tensors {
		if core.HasPrefix(name, "mtp.") || core.Contains(name, ".mtp.") {
			return true
		}
		if core.HasSuffix(name, "conv1d.weight") && len(t.Shape) == 3 && t.Shape[2] != 1 {
			return true
		}
	}
	return false
}

// qwenHybridModelType names the model_type family whose exports store zero-centred norms — the same
// arch scoping mlx-lm gets from its per-class sanitize (register.go's qwen ids, minus the generic
// composed/hybrid catch-alls, which no official Qwen export carries).
func qwenHybridModelType(mt string) bool {
	switch mt {
	case "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text", "qwen3_6", "qwen3_6_moe", "qwen3_next":
		return true
	}
	return false
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

// LoadComposedMmap is the nil-arch ZERO-COPY build model.LoadComposedDir uses for the Qwen 3.6 gated-delta
// hybrid: like LoadComposed but the packed quant weights VIEW the mapped checkpoint instead of being copied
// to the heap (the RSS win), the model taking ownership through the RetainMmap handshake. It is the exact
// build the qwen3_5 Composed hook ran inline before the qwen35 arch package took over the registration —
// kept exported so that hook (now delegating here) preserves the zero-copy behaviour byte-for-byte.
func LoadComposedMmap(tensors map[string]safetensors.Tensor, configJSON []byte) (*ComposedModel, error) {
	return loadComposed(tensors, configJSON, nil, true)
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
// the arch's neutral MoE policy, but every packed quant weight — the dense PROJECTIONS (attention q/k/v/o,
// embed, lm_head) AND the routed + shared MoE EXPERTS (mlp.experts.E.*, mlp.shared_expert.*, resolved
// through the same proj closure as the dense MLP's gate/up/down) — VIEWS the mapped checkpoint
// (QuantWeight.Packed aliases the input tensors' mmap region) instead of being copied to the heap, so a
// quant MoE checkpoint's load-time RSS drops by the packed weight — for a grouped checkpoint the experts
// are the dominant tensor class, so this is the whole win, not a rounding error on it. The model takes
// ownership of the mapping through the SAME RetainMmap handshake as the base path and unmaps it on
// Close/finalize; when nothing aliases (a dense checkpoint, or an all-1-bit pack repacked to owned heap)
// RetainMmap declines and LoadComposedDir unmaps immediately. Use LoadComposedWithArch (copying) when the
// caller unmaps the checkpoint right after the build.
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
	proj := func(name string) ([]float32, *model.QuantWeight, *model.BF16Weight, error) {
		t, ok := get(name)
		if !ok {
			return nil, nil, nil, core.NewError("composed.LoadComposed: missing " + name)
		}
		qw, aliased, err := tensorAsQuant(tensors, name, t, quant, zeroCopy)
		if err != nil {
			return nil, nil, nil, err
		}
		if qw != nil {
			anyAlias = anyAlias || aliased
			return nil, qw, nil, nil
		}
		// Dense bf16 2-D projections stay the checkpoint's own bytes (#26): a zero-copy view of the
		// mmap region (retained via anyAlias) — the widen-to-f32 this replaces doubled both the
		// resident set and the bytes streamed per token. Other dtypes (f32/f16) keep widening.
		if (t.Dtype == "BF16" || t.Dtype == "bfloat16") && len(t.Shape) == 2 {
			data := t.Data
			if !zeroCopy {
				data = append([]byte(nil), t.Data...)
			} else {
				anyAlias = true
			}
			return nil, nil, &model.BF16Weight{Data: data, OutDim: t.Shape[0], InDim: t.Shape[1]}, nil
		}
		f, err := tensorF32(t)
		return f, nil, nil, err
	}
	projOpt := func(name string) ([]float32, *model.QuantWeight, *model.BF16Weight) {
		if _, ok := get(name); !ok {
			return nil, nil, nil
		}
		f, qw, bw, err := proj(name)
		if err != nil {
			return nil, nil, nil
		}
		return f, qw, bw
	}

	// Zero-centred norm shift (#24): official Qwen 3.5/3.6 checkpoints store every layer / final /
	// q/k RMSNorm gamma as an OFFSET from one (y = x·(1+w)/rms); the mlx-community conversions ship
	// the same weights pre-baked (mlx-lm's qwen3_5 sanitize adds the 1 and strips the provenance
	// marks before saving). Serving an official checkpoint with plain ·w garbles every token, so the
	// load mirrors mlx-lm's exact provenance test: an UNCONVERTED checkpoint still carries its mtp.*
	// head and/or the torch conv1d layout [ch,1,K]. Only the four norm slots below shift — the
	// gated-delta block's own norm (linear_attn.norm.weight) is stored plain in both forms.
	normShift := float32(0)
	if qwenHybridModelType(raw.ModelType) || qwenHybridModelType(cfg.ModelType) {
		if composedNormsZeroCentred(tensors) {
			normShift = 1
		}
	}
	normF32 := func(name string) ([]float32, error) {
		v, err := f32(name)
		if err != nil || normShift == 0 {
			return v, err
		}
		for i := range v {
			v[i] += normShift
		}
		return v, nil
	}
	normF32opt := func(name string) []float32 {
		v := f32opt(name)
		if v != nil && normShift != 0 {
			for i := range v {
				v[i] += normShift
			}
		}
		return v
	}

	embedT, ok := get(prefix + "embed_tokens.weight")
	if !ok || len(embedT.Shape) != 2 {
		return nil, core.NewError("composed.LoadComposed: missing/!2D embed_tokens.weight")
	}
	vocab := embedT.Shape[0]
	embed, embedQ, embedB, err := proj(prefix + "embed_tokens.weight")
	if err != nil {
		return nil, err
	}
	// Logical width: a dense embed's dequantised length / vocab; a packed embed's InDim (its Shape[1] is
	// the bits-compressed packed-word count, not the hidden size).
	D := 0
	switch {
	case embedQ != nil:
		D = embedQ.InDim
	case embedB != nil:
		D = embedB.InDim
	default:
		D = len(embed) / vocab
	}
	normF, err := normF32(prefix + "norm.weight")
	if err != nil {
		return nil, err
	}
	output, outputQ, outputB := projOpt("lm_head.weight") // untied; all nil ⇒ tied to embed

	kinds, err := resolveKinds(cfg)
	if err != nil {
		return nil, err
	}

	isCohere := cfg.ModelType == "cohere" || cfg.ModelType == "cohere2"
	m := &ComposedModel{Embed: embed, EmbedQ: embedQ, EmbedB: embedB, NormF: normF, Output: output, OutputQ: outputQ, OutputB: outputB, D: D, Vocab: vocab, Eps: cfg.RMSNormEps, LayerNorm: isCohere || cfg.UseLayerNorm, ParallelResidual: isCohere, LogitScale: cfg.LogitScale, Quantised: quant != nil, BF16Resident: embedB != nil}
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

	// Vision: additive-only. buildVisionTowerQuant probes tensors for vision_tower.*/multi_modal_projector.*
	// (either supported layout — see vision_loader.go's file doc comment) and returns (nil, nil) when
	// neither is present, so a text-only checkpoint (the whole suite before this) loads with m.Vision nil
	// exactly as it always did. raw (not cfg) carries the WRAPPER-level vision_config + image/video token
	// ids — cfg is already narrowed to the text_config side. quant/zeroCopy are the SAME checkpoint quant
	// block and zero-copy choice the text stack's projections resolve through (proj/f32 above), so a
	// quantised vision tower stays packed exactly like the text stack does.
	if vision, visionAlias, verr := buildVisionTowerQuant(tensors, raw.VisionConfig, D, quant, zeroCopy); verr != nil {
		return nil, core.E("composed.LoadComposed", "vision tower", verr)
	} else if vision != nil {
		m.Vision = vision
		anyAlias = anyAlias || visionAlias
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
		inNorm, err := normF32(lp + "input_layernorm.weight")
		if err != nil {
			return nil, err
		}
		var postNorm []float32
		if !isCohere {
			postNorm, err = normF32(lp + "post_attention_layernorm.weight")
			if err != nil {
				return nil, err
			}
		}
		ffn, err := buildFFN(get, proj, f32, lp+"mlp.", cfg, arch, D)
		if err != nil {
			return nil, core.E("composed.LoadComposed", core.Sprintf("layer %d ffn", i), err)
		}
		// The batched routed-MoE quant tensors are zero-copy VIEWS into the switch_mlp tensors (buildMoE
		// bypasses the proj closure that tracks aliasing for the dense projections). On the mmap
		// (LoadComposedDir) path the model must retain the mapping — anyAlias drives RetainMmap; on the
		// owned-copy path (LoadComposed unmaps the checkpoint after the build) they are copied out so no
		// view dangles. Mirrors proj/tensorAsQuant's zeroCopy handling for the dense weights.
		if moe, isMoE := ffn.(*MoEMLP); isMoE && moe.GateBatchedQ != nil {
			if zeroCopy {
				anyAlias = true
			} else {
				moe.ownBatchedQuant()
			}
		}

		var mixer Mixer
		if kinds[i] == "full_attention" || kinds[i] == "sliding_attention" {
			mixer, err = buildAttn(proj, normF32opt, lp+"self_attn.", cfg, arch, i, D, kinds[i])
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
type projFn func(string) ([]float32, *model.QuantWeight, *model.BF16Weight, error)

// widenProjFn wraps a projFn for builders with no bf16 dispatch yet (mamba2, rwkv7, the MoE
// experts): a bf16 view widens to f32 on the spot — the pre-#26 behaviour, scoped to exactly the
// consumers that still need it.
func widenProjFn(proj projFn) projFn {
	return func(name string) ([]float32, *model.QuantWeight, *model.BF16Weight, error) {
		f, qw, bw, err := proj(name)
		if err != nil || bw == nil {
			return f, qw, bw, err
		}
		wf := make([]float32, bw.OutDim*bw.InDim)
		for i := range wf {
			wf[i] = math.Float32frombits(uint32(uint16(bw.Data[2*i])|uint16(bw.Data[2*i+1])<<8) << 16)
		}
		return wf, nil, nil, nil
	}
}

// buildAttn builds a full-attention mixer; geometry from the config.
func buildAttn(proj projFn, f32opt func(string) []float32, sp string, cfg *loaderConfig, arch *model.Arch, layer, D int, kind string) (Mixer, error) {
	qF, qQ, qB, err := proj(sp + "q_proj.weight")
	if err != nil {
		return nil, err
	}
	kF, kQ, kB, err := proj(sp + "k_proj.weight")
	if err != nil {
		return nil, err
	}
	vF, vQ, vB, err := proj(sp + "v_proj.weight")
	if err != nil {
		return nil, err
	}
	oF, oQ, oB, err := proj(sp + "o_proj.weight")
	if err != nil {
		return nil, err
	}
	heads := cfg.NumAttentionHeads
	headDim := cfg.HeadDim
	if headDim == 0 && heads > 0 {
		qCols := len(qF) / D // rows of q_proj = heads·headDim (2× when gated); packed/bf16 forms read it from OutDim
		if qQ != nil {
			qCols = qQ.OutDim
		}
		if qB != nil {
			qCols = qB.OutDim
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
		QProjB: qB, KProjB: kB, VProjB: vB, OProjB: oB,
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

	qkvF, qkvQ, qkvB, err := proj(sp + "in_proj_qkv.weight")
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
	inAF, inAQ, inAB, err := proj(sp + "in_proj_a.weight")
	if err != nil {
		return nil, err
	}
	inBF, inBQ, inBB, err := proj(sp + "in_proj_b.weight")
	if err != nil {
		return nil, err
	}
	inZF, inZQ, inZB, err := proj(sp + "in_proj_z.weight")
	if err != nil {
		return nil, err
	}
	outPF, outPQ, outPB, err := proj(sp + "out_proj.weight")
	if err != nil {
		return nil, err
	}
	w := &model.GatedDeltaWeights{
		InProjQKV: qkvF, InProjQKVQ: qkvQ, InProjQKVB: qkvB, ConvWeight: convW, ConvBias: f32opt(sp + "conv1d.bias"),
		InProjA: inAF, InProjAQ: inAQ, InProjAB: inAB, ALog: aLog, DtBias: f32opt(sp + "dt_bias"),
		InProjB: inBF, InProjBQ: inBQ, InProjBB: inBB, InProjZ: inZF, InProjZQ: inZQ, InProjZB: inZB,
		Norm: norm, OutProj: outPF, OutProjQ: outPQ, OutProjB: outPB,
	}
	cfg := model.GatedDeltaConfig{KeyHeads: keyHeads, ValueHeads: valueHeads, HeadDim: headDim, ConvKernel: convK, Eps: 1e-6}
	return NewGatedDeltaMixer(w, cfg), nil
}

// sliceBatchedExpertQuant carves expert e's 2-D MLX-affine quant weight out of a batched mlx switch_mlp
// tensor: <base>.weight [numExperts, outDim, inDim·bits/32] (U32 packed) with the [numExperts, outDim,
// inDim/groupSize] bf16 <base>.scales/.biases siblings — the layout mlx-lm emits for a fused MoE (all
// experts in one tensor per projection, gate/up carried separately). bits and groupSize are read off the
// packed/scales widths against inDim (the projection's contraction dim: D for gate/up, FF for down). Expert
// e's bytes are COPIED (owned) so the QuantWeight outlives the checkpoint mapping — the composed lane's
// per-expert twin of engine/metal's view of a batched switch_glu tensor.
func sliceBatchedExpertQuant(get func(string) (safetensors.Tensor, bool), base string, e, outDim, inDim int) (*model.QuantWeight, error) {
	wT, ok := get(base + ".weight")
	sT, sOK := get(base + ".scales")
	bT, bOK := get(base + ".biases")
	if !ok || !sOK || !bOK {
		return nil, core.NewError("composed.sliceBatchedExpertQuant: " + base + " missing .weight/.scales/.biases")
	}
	if len(wT.Shape) != 3 || len(sT.Shape) != 3 || wT.Shape[1] != outDim || wT.Shape[0] == 0 {
		return nil, core.NewError("composed.sliceBatchedExpertQuant: " + base + " is not a [numExperts, outDim, packed] quant tensor")
	}
	nE, packedCols, groups := wT.Shape[0], wT.Shape[2], sT.Shape[2]
	if packedCols*32%inDim != 0 || groups == 0 || inDim%groups != 0 {
		return nil, core.NewError(core.Sprintf("composed.sliceBatchedExpertQuant: %s geometry (packed %d, groups %d, inDim %d)", base, packedCols, groups, inDim))
	}
	slice := func(data []byte) ([]byte, error) {
		per := len(data) / nE
		if per == 0 || (e+1)*per > len(data) {
			return nil, core.NewError(core.Sprintf("composed.sliceBatchedExpertQuant: %s expert %d out of range", base, e))
		}
		return append([]byte(nil), data[e*per:(e+1)*per]...), nil
	}
	packed, err := slice(wT.Data)
	if err != nil {
		return nil, err
	}
	scales, err := slice(sT.Data)
	if err != nil {
		return nil, err
	}
	biases, err := slice(bT.Data)
	if err != nil {
		return nil, err
	}
	return &model.QuantWeight{Packed: packed, Scales: scales, Biases: biases, Bits: packedCols * 32 / inDim, GroupSize: inDim / groups, OutDim: outDim, InDim: inDim}, nil
}

// batchedExpertQuant wraps the WHOLE mlx switch_mlp batched tensor for one projection as a single packed
// model.QuantWeight — <base>.weight [numExperts, outDim, inDim·bits/32] (U32) with its [numExperts, outDim,
// inDim/groupSize] bf16 <base>.scales/.biases siblings, every expert concatenated — the batched form the
// engine's single-dispatch routed-MoE kernel (composed.MoEExpertsDevice) consumes. bits/groupSize are read
// off the packed/scales widths against inDim (D for gate/up, FF for down), exactly as sliceBatchedExpertQuant
// derives them per expert. Packed/Scales/Biases ZERO-COPY the (mmap'd) checkpoint tensors — the loader owns
// the retention decision (LoadComposedDir keeps the mapping via anyAlias; the owned-copy path deep-copies via
// MoEMLP.ownBatchedQuant). OutDim/InDim are one expert's logical dims; Packed holds numExperts of them.
func batchedExpertQuant(get func(string) (safetensors.Tensor, bool), base string, inDim int) (*model.QuantWeight, error) {
	wT, ok := get(base + ".weight")
	sT, sOK := get(base + ".scales")
	bT, bOK := get(base + ".biases")
	if !ok || !sOK || !bOK {
		return nil, core.NewError("composed.batchedExpertQuant: " + base + " missing .weight/.scales/.biases")
	}
	if len(wT.Shape) != 3 || len(sT.Shape) != 3 || wT.Shape[0] == 0 {
		return nil, core.NewError("composed.batchedExpertQuant: " + base + " is not a [numExperts, outDim, packed] quant tensor")
	}
	outDim, packedCols, groups := wT.Shape[1], wT.Shape[2], sT.Shape[2]
	if packedCols*32%inDim != 0 || groups == 0 || inDim%groups != 0 {
		return nil, core.NewError(core.Sprintf("composed.batchedExpertQuant: %s geometry (packed %d, groups %d, inDim %d)", base, packedCols, groups, inDim))
	}
	return &model.QuantWeight{
		Packed: wT.Data, Scales: sT.Data, Biases: bT.Data,
		Bits: packedCols * 32 / inDim, GroupSize: inDim / groups, OutDim: outDim, InDim: inDim,
	}, nil
}

// buildFFN builds a layer's feed-forward: a MoE (qwen3_6_moe) when expert weights are present, else a
// dense SwiGLU MLP. sp is the "…mlp." prefix. Experts arrive either per-expert (experts.N.gate_proj — a
// pre-split checkpoint) or batched (switch_mlp.gate_proj [numExperts,…] — the mlx-lm quantised layout).
func buildFFN(get func(string) (safetensors.Tensor, bool), proj projFn, f32 func(string) ([]float32, error), sp string, cfg *loaderConfig, arch *model.Arch, D int) (FFN, error) {
	_, perExpert := get(sp + "experts.0.gate_proj.weight")
	_, batched := get(sp + "switch_mlp.gate_proj.weight")
	if perExpert || batched {
		// A quant checkpoint's experts resolve PACKED (model.QuantWeight): per-expert through proj, or
		// sliced out of the batched switch_mlp tensors — either way the dense MLP path below is skipped.
		return buildMoE(get, widenProjFn(proj), f32, sp, cfg, arch, D)
	}
	gateF, gateQ, gateB, err := proj(sp + "gate_proj.weight")
	if err != nil {
		return nil, err
	}
	upF, upQ, upB, err := proj(sp + "up_proj.weight")
	if err != nil {
		return nil, err
	}
	downF, downQ, downB, err := proj(sp + "down_proj.weight")
	if err != nil {
		return nil, err
	}
	ff := len(gateF) / D // gate_proj rows = FF; packed/bf16 forms read it from OutDim
	if gateQ != nil {
		ff = gateQ.OutDim
	}
	if gateB != nil {
		ff = gateB.OutDim
	}
	return &MLP{Gate: gateF, Up: upF, Down: downF, GateQ: gateQ, UpQ: upQ, DownQ: downQ, GateB: gateB, UpB: upB, DownB: downB, FF: ff}, nil
}

// buildMoE loads the MoE FFN: router (mlp.gate.weight — always host f32; small enough that real
// checkpoints don't quantise it), the experts (mlp.experts.E.*) and the optional shared expert
// (mlp.shared_expert.*), both resolved through proj — PACKED (model.QuantWeight) on a quant checkpoint,
// f32 otherwise — exactly like the dense MLP's gate/up/down (buildFFN). TopK = num_experts_per_tok.
func buildMoE(get func(string) (safetensors.Tensor, bool), proj projFn, f32 func(string) ([]float32, error), sp string, cfg *loaderConfig, arch *model.Arch, D int) (FFN, error) {
	router, err := f32(sp + "gate.weight")
	if err != nil {
		return nil, err
	}
	expert := func(p string) (MoEExpert, error) {
		g, gQ, _, e1 := proj(p + "gate_proj.weight")
		u, uQ, _, e2 := proj(p + "up_proj.weight")
		d, dQ, _, e3 := proj(p + "down_proj.weight")
		for _, e := range []error{e1, e2, e3} {
			if e != nil {
				return MoEExpert{}, e
			}
		}
		return MoEExpert{Gate: g, Up: u, Down: d, GateQ: gQ, UpQ: uQ, DownQ: dQ}, nil
	}
	var experts []MoEExpert
	var gateBatchedQ, upBatchedQ, downBatchedQ *model.QuantWeight
	var moeBits, moeGroupSize int
	if gT, ok := get(sp + "switch_mlp.gate_proj.weight"); ok {
		// mlx batched layout: switch_mlp.{gate,up,down}_proj are single [numExperts, outDim, packed]
		// tensors — slice each expert's own 2-D quant weight out (gate/up: FF×D, down: D×FF).
		nE, ff := gT.Shape[0], gT.Shape[1]
		for e := 0; e < nE; e++ {
			gQ, err := sliceBatchedExpertQuant(get, sp+"switch_mlp.gate_proj", e, ff, D)
			if err != nil {
				return nil, err
			}
			uQ, err := sliceBatchedExpertQuant(get, sp+"switch_mlp.up_proj", e, ff, D)
			if err != nil {
				return nil, err
			}
			dQ, err := sliceBatchedExpertQuant(get, sp+"switch_mlp.down_proj", e, D, ff)
			if err != nil {
				return nil, err
			}
			experts = append(experts, MoEExpert{GateQ: gQ, UpQ: uQ, DownQ: dQ})
		}
		// ALSO keep the WHOLE switch_mlp tensors as three batched quant weights (zero-copy views) — the
		// form the single-dispatch routed-MoE kernel consumes, collapsing this layer's topK×3 per-expert
		// submits into one device call. The per-expert slices above stay the host fallback. loadComposed
		// owns the retention decision for these views (anyAlias on a zero-copy load, else ownBatchedQuant).
		if gateBatchedQ, err = batchedExpertQuant(get, sp+"switch_mlp.gate_proj", D); err != nil {
			return nil, err
		}
		if upBatchedQ, err = batchedExpertQuant(get, sp+"switch_mlp.up_proj", D); err != nil {
			return nil, err
		}
		if downBatchedQ, err = batchedExpertQuant(get, sp+"switch_mlp.down_proj", ff); err != nil {
			return nil, err
		}
		moeBits, moeGroupSize = gateBatchedQ.Bits, gateBatchedQ.GroupSize
	} else {
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
		if _, ok := get(sp + "shared_expert_gate.weight"); ok {
			// f32() dequantises when the gate carries .scales/.biases (mlx quantises even this [1,D] tensor).
			if v, e := f32(sp + "shared_expert_gate.weight"); e == nil {
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
	// Fuse each packed expert's gate+up into one [gate‖up] projection when the arch opts in
	// (Arch.FuseExpertGateUp) — one quant matvec per routed expert instead of two, mirroring the metal
	// lane's fusedExperts path. Off by default: it materialises the concat on the heap (trading the
	// gate/up mmap zero-copy the composed lane exists to keep), so it is a per-model opt-in the device
	// bench justifies, not automatic. No-op on a dense (f32) checkpoint.
	if arch != nil && arch.FuseExpertGateUp {
		for i := range experts {
			fuseExpertGateUp(&experts[i])
		}
		if shared != nil {
			fuseExpertGateUp(shared)
		}
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
	return &MoEMLP{Router: router, Experts: experts, Shared: shared, SharedGate: sharedGate, TopK: topK, NormTopKProb: normTopK, Gating: gating,
		GateBatchedQ: gateBatchedQ, UpBatchedQ: upBatchedQ, DownBatchedQ: downBatchedQ, MoEBits: moeBits, MoEGroupSize: moeGroupSize}, nil
}
