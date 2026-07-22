// SPDX-Licence-Identifier: EUPL-1.2

package qwen35

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/quant/mlxaffine"
	"dappco.re/go/inference/model/safetensors"
)

// vision_loader.go builds a *VisionTower from a checkpoint's tensors — the factory-native port of the
// retired composed engine's buildVisionTowerQuant (model/composed/vision_loader.go at b1f6c21a^),
// keeping its house style: probe by name, derive geometry from the WEIGHTS themselves wherever a
// tensor shape settles the question unambiguously, and only fall back to config.json when a dimension
// has no tensor to derive it from (PatchSize — pixel geometry) or the checkpoint carries no
// disambiguating tensor (NumHeads, only when there is no q_norm to read HeadDim off directly).
//
// TWO checkpoint conventions resolve here, distinguished purely by which tensor names are present
// (never a model_type switch):
//
//   - the REAL layout (verified against mlx-community/Qwen3.6-27B-4bit — 333 vision tensors, 21
//     patterns; vision_real_test.go is the reconciliation receipt): a FUSED single attn.qkv linear
//     (split by output row at load — splitFusedQKV), a plain 2-linear GELU mlp.linear_fc1/linear_fc2
//     (no SwiGLU), full LayerNorm-with-bias norms, NO q_norm/k_norm, and a LEARNED additive
//     vision_tower.pos_embed.weight alongside the per-block 2-D rotary positions:
//
//     vision_tower.patch_embed.proj.weight/.bias      (weight may be >2-D — e.g. [Hidden,T,P,P,C] —
//     patchDim is the FLATTENED product of every dim after Hidden; the flat bytes are identical to a
//     [Hidden,PatchDim] tensor, and that trailing order matches ImageToPatchGrid's own patch-row
//     layout)
//     vision_tower.pos_embed.weight                   [NumPositions, Hidden]
//     vision_tower.blocks.<i>.norm1/.norm2.weight/.bias
//     vision_tower.blocks.<i>.attn.qkv.weight/.bias   [3·Hidden, Hidden] (plain MHA — equal thirds)
//     vision_tower.blocks.<i>.attn.proj.weight/.bias
//     vision_tower.blocks.<i>.mlp.linear_fc1/.linear_fc2.weight/.bias
//     vision_tower.merger.norm.weight/.bias
//     vision_tower.merger.linear_fc1/.linear_fc2.weight/.bias
//
//   - the GUESSED layout (separate q/k/v/o projections, SwiGLU MLP, optional RMS q/k-norms, 2-D
//     rotary positions only) the composed package shipped before a real checkpoint was available:
//     vision_tower.patch_embed(.proj).weight/.bias, blocks.<i>.attn.q_proj/.k_proj/.v_proj/.o_proj,
//     blocks.<i>.mlp.gate_proj/.up_proj/.down_proj, multi_modal_projector.norm(/.ln_q) +
//     linear_1(.mlp.0)/linear_2(.mlp.2).
//
// Both conventions produce the SAME payload shape — the divergences are load-time-only facts
// (VisionMLPWeights.GELU, VisionTowerConfig.LearnedPositions), never a second forward path.
//
// Quantisation: the payload is ALWAYS host f32. A dense tensor widens through
// safetensors.DecodeFloatData; a PACKED projection (mlx affine — a .scales/.biases sibling pair)
// dequantises at load through mlxaffine.DequantizeTensor — the SAME primitive the composed reference's
// host quant matvec (matNTQuantHost) called per output row, so the values the forward dots against are
// identical to that reference's host path; only the composed lane's optional device qmv tier is not
// reproduced (it differed from the host reference at the parity gate's tolerance). Small vectors
// (biases, norms, the learned position table) are never quantised in any observed scheme.
//
// A checkpoint carrying neither layout's patch_embed tensor returns (nil, nil) — the text-only path,
// unchanged behaviour.

// LoadVisionTower probes tensors for a Qwen-VL-family vision tower + merger and, when found, assembles
// a *VisionTower with every geometric field DERIVED from the tensors' own shapes. configJSON is the
// checkpoint's config.json (the same bytes the factory's Parse consumed) — it supplies the pixel
// geometry (vision_config.patch_size), the quant block, the text hidden width for the merger
// cross-check, and the image token id. Returns (nil, nil) — not an error — when no patch_embed tensor
// is present: the checkpoint is text-only.
func LoadVisionTower(tensors map[string]safetensors.Tensor, configJSON []byte) (*VisionTower, error) {
	var cfg Config
	if r := core.JSONUnmarshal(configJSON, &cfg); !r.OK {
		return nil, core.NewError("qwen35.LoadVisionTower: config.json parse failed")
	}
	return buildVisionTower(tensors, &cfg)
}

// buildVisionTower is LoadVisionTower over an already-parsed Config.
func buildVisionTower(tensors map[string]safetensors.Tensor, cfg *Config) (*VisionTower, error) {
	patchName, patchT, ok := weightAnyName(tensors, "vision_tower.patch_embed.weight", "vision_tower.patch_embed.proj.weight")
	if !ok {
		return nil, nil // text-only checkpoint
	}
	if len(patchT.Shape) < 2 {
		return nil, core.NewError("qwen35.buildVisionTower: patch_embed weight has no input dimensions")
	}
	quant := cfg.ResolvedQuant()
	vc := cfg.VisionConfig
	hidden := patchT.Shape[0]
	// patchDim is patch_embed's LOGICAL input width. For a dense tensor this flattens Shape[1:]
	// (row-major, byte-identical to a flat [Hidden,PatchDim] tensor); for a PACKED tensor Shape[1] is
	// the compressed word count, so the width is recovered from the .scales sibling instead.
	patchDim, err := visionProjInDim(tensors, patchName, patchT, quant)
	if err != nil {
		return nil, core.E("qwen35.buildVisionTower", "patch_embed weight", err)
	}
	patchLin, err := visionProj(tensors, patchName, patchT, hidden, patchDim, quant)
	if err != nil {
		return nil, core.E("qwen35.buildVisionTower", "patch_embed weight", err)
	}
	patchLin.B = optionalVisionVec(tensors, biasSibling(patchName))

	patchSize, inChannels := 0, 3
	ropeTheta, eps := float32(10000), float32(1e-6)
	if vc != nil {
		if vc.PatchSize > 0 {
			patchSize = vc.PatchSize
		}
		if vc.InChannels > 0 {
			inChannels = vc.InChannels
		}
		if vc.RopeTheta > 0 {
			ropeTheta = vc.RopeTheta
		}
		if vc.RMSNormEps > 0 {
			eps = vc.RMSNormEps
		}
	}
	if patchSize <= 0 {
		return nil, core.NewError("qwen35.buildVisionTower: vision_tower present but vision_config.patch_size is missing")
	}
	perFrame := inChannels * patchSize * patchSize
	if perFrame <= 0 || patchDim%perFrame != 0 {
		return nil, core.NewError(core.Sprintf("qwen35.buildVisionTower: patch_embed input width %d is not a multiple of in_channels·patch_size² %d", patchDim, perFrame))
	}
	temporal := patchDim / perFrame

	blocks, headDim, numHeads, numKVHeads, err := buildVisionBlocks(tensors, hidden, vc, quant)
	if err != nil {
		return nil, err
	}

	textHidden := cfg.effective().HiddenSize
	merger, mergeSize, err := buildVisionMerger(tensors, hidden, textHidden, vc, quant)
	if err != nil {
		return nil, err
	}

	// The REAL layout's learned absolute position embedding — additive, added once after the patch
	// embed, ALONGSIDE the per-block 2-D rotary embedding (the reference convention). Never packed in
	// any observed checkpoint (an additive lookup table, not a GEMM weight).
	posEmbed := optionalVisionVec(tensors, "vision_tower.pos_embed.weight")
	if len(posEmbed) > 0 {
		positions := len(posEmbed) / hidden
		if len(posEmbed)%hidden != 0 || isqrt(positions)*isqrt(positions) != positions {
			return nil, core.NewError(core.Sprintf("qwen35.buildVisionTower: pos_embed length %d is not a square grid of %d-wide rows", len(posEmbed), hidden))
		}
	}

	return &VisionTower{
		Patch:    patchLin,
		PosEmbed: posEmbed,
		Blocks:   blocks,
		Merger:   *merger,
		Cfg: VisionTowerConfig{
			Hidden: hidden, PatchDim: patchDim,
			NumHeads: numHeads, NumKVHeads: numKVHeads, HeadDim: headDim,
			PatchSize: patchSize, InChannels: inChannels, TemporalPatchSize: temporal,
			MergeSize: mergeSize, TextHidden: textHidden,
			RopeTheta: ropeTheta, Eps: eps,
			LearnedPositions: len(posEmbed) > 0,
			ImageTokenID:     int32(cfg.ImageTokenID),
		},
	}, nil
}

// optionalVisionVec returns tensors[name] widened to f32, or nil when absent/unwidenable — the
// bias/qk-norm/norm/learned-position-table probe this file uses throughout. Never quant-aware: none of
// the small vectors this resolves are ever packed in a known checkpoint.
func optionalVisionVec(tensors map[string]safetensors.Tensor, name string) []float32 {
	if t, ok := tensors[name]; ok {
		if v, err := safetensors.DecodeFloatData(t.Dtype, t.Data, numel(t.Shape)); err == nil {
			return v
		}
	}
	return nil
}

// numel is the element count of a shape (1 for rank-0).
func numel(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

// weightAnyName resolves the FIRST present name of several alias spellings, reporting which matched —
// needed wherever the result feeds visionProj/biasSibling, which derive sibling keys from the matched
// name itself.
func weightAnyName(tensors map[string]safetensors.Tensor, names ...string) (name string, t safetensors.Tensor, ok bool) {
	for _, n := range names {
		if tv, found := tensors[n]; found {
			return n, tv, true
		}
	}
	return "", safetensors.Tensor{}, false
}

// biasSibling turns a resolved "<x>.weight" tensor name into its "<x>.bias" sibling.
func biasSibling(name string) string {
	if core.HasSuffix(name, ".weight") {
		return name[:len(name)-len(".weight")] + ".bias"
	}
	return name + ".bias"
}

// bf16SidecarDtype reports whether a quant sidecar tensor's dtype is the bf16 tier
// mlxaffine.DequantizeTensor reads. Every known mlx-affine qwen pack ships BF16 sidecars; anything
// else refuses loudly rather than dequantising garbage.
func bf16SidecarDtype(dtype string) bool { return dtype == "BF16" || dtype == "bfloat16" }

// visionProj resolves name (a projection weight, already looked up as t) to an f32 VisionLinear.
// outDim/inDim are the projection's LOGICAL shape, supplied by the caller rather than read off
// t.Shape, so a tensor whose dense form ships >2-D (the real layout's patch_embed) resolves the same
// way a genuinely 2-D one does, and a PACKED tensor (whose own Shape[1] is the compressed word count)
// is validated against the logical width. A packed weight (a .scales/.biases sibling pair) dequantises
// whole through mlxaffine.DequantizeTensor — value-identical to the composed reference's per-row host
// dequant (the same function over the same rows).
func visionProj(tensors map[string]safetensors.Tensor, name string, t safetensors.Tensor, outDim, inDim int, quant *model.QuantConfig) (VisionLinear, error) {
	base := name
	if core.HasSuffix(base, ".weight") {
		base = base[:len(base)-len(".weight")]
	}
	scalesT, sOK := tensors[base+".scales"]
	biasesT, bOK := tensors[base+".biases"]
	if sOK && bOK {
		if quant == nil {
			return VisionLinear{}, core.NewError("qwen35.buildVisionTower: " + name + " carries .scales/.biases but the config has no quantization block")
		}
		if len(t.Shape) != 2 || len(scalesT.Shape) != 2 {
			return VisionLinear{}, core.NewError("qwen35.buildVisionTower: quantised " + name + " is not 2-D")
		}
		if !bf16SidecarDtype(scalesT.Dtype) || !bf16SidecarDtype(biasesT.Dtype) {
			return VisionLinear{}, core.NewError("qwen35.buildVisionTower: " + name + " quant sidecars are not BF16 (unsupported pack)")
		}
		gs, bits := quant.For(base)
		if gs <= 0 || bits <= 0 {
			return VisionLinear{}, core.NewError("qwen35.buildVisionTower: " + name + " is quantised but the config's group_size/bits are not positive")
		}
		packedCols := t.Shape[1]
		derivedIn := scalesT.Shape[1] * gs
		if derivedIn != inDim {
			return VisionLinear{}, core.NewError(core.Sprintf("qwen35.buildVisionTower: %s packed input width %d != expected %d", name, derivedIn, inDim))
		}
		if t.Shape[0] != outDim {
			return VisionLinear{}, core.NewError(core.Sprintf("qwen35.buildVisionTower: %s packed rows %d != expected %d", name, t.Shape[0], outDim))
		}
		if packedCols*32 != inDim*bits {
			return VisionLinear{}, core.NewError(core.Sprintf("qwen35.buildVisionTower: %s packed cols %d != inDim %d·bits %d/32", name, packedCols, inDim, bits))
		}
		w, err := mlxaffine.DequantizeTensor(t.Data, scalesT.Data, biasesT.Data, outDim, inDim, bits, gs)
		if err != nil {
			return VisionLinear{}, core.E("qwen35.buildVisionTower", name+" dequantise", err)
		}
		return VisionLinear{W: w, Out: outDim, In: inDim}, nil
	}
	f, err := safetensors.DecodeFloatData(t.Dtype, t.Data, numel(t.Shape))
	if err != nil {
		return VisionLinear{}, core.E("qwen35.buildVisionTower", name+" widen", err)
	}
	if len(f) != outDim*inDim {
		return VisionLinear{}, core.NewError(core.Sprintf("qwen35.buildVisionTower: %s width %d != expected %d·%d", name, len(f), outDim, inDim))
	}
	return VisionLinear{W: f, Out: outDim, In: inDim}, nil
}

// visionProjInDim returns name's LOGICAL input width — needed for the projections whose width the
// caller cannot know independently (patch_embed — possibly >2-D dense; the merger's linear_1, whose
// width DERIVES the merge size). For a dense t this flattens Shape[1:]; for a PACKED t (a .scales
// sibling) Shape[1] is the compressed word count, so the width is recovered from the sibling's shape
// (nGroups·groupSize) — the same derivation the composed reference used.
func visionProjInDim(tensors map[string]safetensors.Tensor, name string, t safetensors.Tensor, quant *model.QuantConfig) (int, error) {
	base := name
	if core.HasSuffix(base, ".weight") {
		base = base[:len(base)-len(".weight")]
	}
	scalesT, sOK := tensors[base+".scales"]
	if !sOK {
		return numel(t.Shape[1:]), nil
	}
	if quant == nil {
		return 0, core.NewError("qwen35.buildVisionTower: " + name + " carries .scales but the config has no quantization block")
	}
	if len(scalesT.Shape) != 2 {
		return 0, core.NewError("qwen35.buildVisionTower: " + name + " .scales is not 2-D")
	}
	gs, _ := quant.For(base)
	if gs <= 0 {
		return 0, core.NewError("qwen35.buildVisionTower: " + name + " is quantised but its group_size is not positive")
	}
	return scalesT.Shape[1] * gs, nil
}

// splitFusedQKV splits one fused vision_tower.blocks.<i>.attn.qkv VisionLinear (Out = 3·PerHead, In =
// Hidden — the REAL layout's single QKV linear) into three [PerHead,Hidden] projections by OUTPUT ROW:
// rows [0,PerHead) are Q (every head), [PerHead,2·PerHead) K, [2·PerHead,3·PerHead) V. This is the row
// order torch's reshape(L, 3, heads, headDim).permute(...) convention guarantees for a fused qkv
// Linear — each of the 3 groups occupies a contiguous OUTPUT-ROW band, so splitting by row is exact.
// Assumes q/k/v are EQUAL width (plain MHA) — the fused-qkv convention carries no per-branch
// head-count signal, and no known Qwen-VL checkpoint ships an uneven split (the real checkpoint's
// vision_config carries no num_key_value_heads at all).
func splitFusedQKV(fused VisionLinear) (q, k, v VisionLinear, err error) {
	if fused.Out <= 0 || fused.Out%3 != 0 {
		return VisionLinear{}, VisionLinear{}, VisionLinear{}, core.NewError(core.Sprintf("qwen35.buildVisionTower: fused qkv width %d is not divisible by 3", fused.Out))
	}
	per := fused.Out / 3
	band := func(i int) VisionLinear {
		lin := VisionLinear{Out: per, In: fused.In, W: fused.W[i*per*fused.In : (i+1)*per*fused.In]}
		if len(fused.B) > 0 {
			lin.B = fused.B[i*per : (i+1)*per]
		}
		return lin
	}
	return band(0), band(1), band(2), nil
}

// resolveVisionAttnGeometry derives (fused, headDim, numHeads, numKVHeads) from ONE block's attention
// tensors — fused reports whether this checkpoint uses the REAL layout's single attn.qkv linear (true)
// or the GUESSED layout's separate attn.q_proj/k_proj/v_proj (false); the caller runs this ONCE at
// block 0 and loads every later block under the same decided convention. HeadDim prefers q_norm's own
// width (unambiguous — a per-head RMSNorm scale is exactly HeadDim wide); only when no q_norm tensor
// exists does it fall back to vision_config.num_heads. The fused layout has no per-branch head-count
// signal, so numKVHeads==numHeads always on that path (plain MHA).
func resolveVisionAttnGeometry(tensors map[string]safetensors.Tensor, bp string, vc *VisionArchConfig, qNorm []float32) (fused bool, headDim, numHeads, numKVHeads int, err error) {
	if fusedT, ok := tensors[bp+"attn.qkv.weight"]; ok {
		if len(fusedT.Shape) != 2 {
			return false, 0, 0, 0, core.NewError(core.Sprintf("qwen35.buildVisionTower: %sattn.qkv is not 2-D", bp))
		}
		if fusedT.Shape[0] <= 0 || fusedT.Shape[0]%3 != 0 {
			return false, 0, 0, 0, core.NewError(core.Sprintf("qwen35.buildVisionTower: %sattn.qkv width %d is not divisible by 3", bp, fusedT.Shape[0]))
		}
		perQKV := fusedT.Shape[0] / 3
		switch {
		case len(qNorm) > 0:
			headDim = len(qNorm)
		case vc != nil && vc.NumHeads > 0:
			headDim = perQKV / vc.NumHeads
		}
		if headDim <= 0 || perQKV%headDim != 0 {
			return false, 0, 0, 0, core.NewError("qwen35.buildVisionTower: cannot derive attention head_dim (no q_norm tensor and no usable vision_config.num_heads)")
		}
		return true, headDim, perQKV / headDim, perQKV / headDim, nil
	}
	qT, ok := tensors[bp+"attn.q_proj.weight"]
	if !ok {
		return false, 0, 0, 0, core.NewError(core.Sprintf("qwen35.buildVisionTower: %sattn.qkv/attn.q_proj weight is missing", bp))
	}
	if len(qT.Shape) != 2 {
		return false, 0, 0, 0, core.NewError(core.Sprintf("qwen35.buildVisionTower: %sq_proj is not 2-D", bp))
	}
	kT, ok := tensors[bp+"attn.k_proj.weight"]
	if !ok {
		return false, 0, 0, 0, core.NewError(core.Sprintf("qwen35.buildVisionTower: %smissing attn.k_proj.weight", bp))
	}
	switch {
	case len(qNorm) > 0:
		headDim = len(qNorm)
	case vc != nil && vc.NumHeads > 0:
		headDim = qT.Shape[0] / vc.NumHeads
	}
	if headDim <= 0 || qT.Shape[0]%headDim != 0 {
		return false, 0, 0, 0, core.NewError("qwen35.buildVisionTower: cannot derive attention head_dim (no q_norm tensor and no usable vision_config.num_heads)")
	}
	numHeads = qT.Shape[0] / headDim
	if kT.Shape[0]%headDim != 0 {
		return false, 0, 0, 0, core.NewError(core.Sprintf("qwen35.buildVisionTower: k_proj rows %d not a multiple of derived head_dim %d", kT.Shape[0], headDim))
	}
	numKVHeads = kT.Shape[0] / headDim
	if vc != nil && vc.NumKeyValueHeads > 0 && vc.NumKeyValueHeads != numKVHeads {
		return false, 0, 0, 0, core.NewError(core.Sprintf("qwen35.buildVisionTower: derived kv heads %d != config num_key_value_heads %d", numKVHeads, vc.NumKeyValueHeads))
	}
	return false, headDim, numHeads, numKVHeads, nil
}

// loadBlockQKV resolves one block's Q/K/V projections under the ALREADY-DECIDED fused/separate
// convention (resolveVisionAttnGeometry, called once at block 0). A block whose tensors don't match
// the decided convention fails loudly here, catching a checkpoint that mixes conventions across
// blocks.
func loadBlockQKV(tensors map[string]safetensors.Tensor, bp string, hidden int, fused bool, quant *model.QuantConfig) (q, k, v VisionLinear, err error) {
	if fused {
		fusedT, ok := tensors[bp+"attn.qkv.weight"]
		if !ok {
			return VisionLinear{}, VisionLinear{}, VisionLinear{}, core.NewError(core.Sprintf("qwen35.buildVisionTower: %smissing attn.qkv.weight", bp))
		}
		fusedLin, ferr := visionProj(tensors, bp+"attn.qkv.weight", fusedT, fusedT.Shape[0], hidden, quant)
		if ferr != nil {
			return VisionLinear{}, VisionLinear{}, VisionLinear{}, core.E("qwen35.buildVisionTower", bp+"attn.qkv", ferr)
		}
		fusedLin.B = optionalVisionVec(tensors, bp+"attn.qkv.bias")
		q, k, v, ferr = splitFusedQKV(fusedLin)
		if ferr != nil {
			return VisionLinear{}, VisionLinear{}, VisionLinear{}, core.E("qwen35.buildVisionTower", bp+"attn.qkv split", ferr)
		}
		return q, k, v, nil
	}
	qT, ok := tensors[bp+"attn.q_proj.weight"]
	if !ok {
		return VisionLinear{}, VisionLinear{}, VisionLinear{}, core.NewError(core.Sprintf("qwen35.buildVisionTower: %smissing attn.q_proj.weight", bp))
	}
	kT, ok := tensors[bp+"attn.k_proj.weight"]
	if !ok {
		return VisionLinear{}, VisionLinear{}, VisionLinear{}, core.NewError(core.Sprintf("qwen35.buildVisionTower: %smissing attn.k_proj.weight", bp))
	}
	vT, ok := tensors[bp+"attn.v_proj.weight"]
	if !ok {
		return VisionLinear{}, VisionLinear{}, VisionLinear{}, core.NewError(core.Sprintf("qwen35.buildVisionTower: %smissing attn.v_proj.weight", bp))
	}
	// inDim is `hidden` (the known tower width) for all three — q/k/v read the SAME hidden-wide input,
	// so the logical inDim is known independent of any (possibly packed) tensor's own Shape[1].
	qLin, err := visionProj(tensors, bp+"attn.q_proj.weight", qT, qT.Shape[0], hidden, quant)
	if err != nil {
		return VisionLinear{}, VisionLinear{}, VisionLinear{}, core.E("qwen35.buildVisionTower", bp+"attn.q_proj", err)
	}
	qLin.B = optionalVisionVec(tensors, bp+"attn.q_proj.bias")
	kLin, err := visionProj(tensors, bp+"attn.k_proj.weight", kT, kT.Shape[0], hidden, quant)
	if err != nil {
		return VisionLinear{}, VisionLinear{}, VisionLinear{}, core.E("qwen35.buildVisionTower", bp+"attn.k_proj", err)
	}
	kLin.B = optionalVisionVec(tensors, bp+"attn.k_proj.bias")
	vLin, err := visionProj(tensors, bp+"attn.v_proj.weight", vT, vT.Shape[0], hidden, quant)
	if err != nil {
		return VisionLinear{}, VisionLinear{}, VisionLinear{}, core.E("qwen35.buildVisionTower", bp+"attn.v_proj", err)
	}
	vLin.B = optionalVisionVec(tensors, bp+"attn.v_proj.bias")
	return qLin, kLin, vLin, nil
}

// loadBlockMLP resolves one block's feed-forward — the GUESSED layout's SwiGLU (mlp.gate_proj/up_proj/
// down_proj, gelu=false) when gate_proj is present, else the REAL layout's plain 2-linear GELU
// (mlp.linear_fc1/linear_fc2, gelu=true). ff is the hidden FF width the caller checks for uniformity
// across blocks.
func loadBlockMLP(tensors map[string]safetensors.Tensor, bp string, hidden int, quant *model.QuantConfig) (w VisionMLPWeights, ff int, gelu bool, err error) {
	if gateT, ok := tensors[bp+"mlp.gate_proj.weight"]; ok {
		if len(gateT.Shape) != 2 {
			return VisionMLPWeights{}, 0, false, core.NewError(core.Sprintf("qwen35.buildVisionTower: %smlp.gate_proj is not 2-D", bp))
		}
		upT, ok := tensors[bp+"mlp.up_proj.weight"]
		if !ok {
			return VisionMLPWeights{}, 0, false, core.NewError(core.Sprintf("qwen35.buildVisionTower: %smissing mlp.up_proj.weight", bp))
		}
		downT, ok := tensors[bp+"mlp.down_proj.weight"]
		if !ok {
			return VisionMLPWeights{}, 0, false, core.NewError(core.Sprintf("qwen35.buildVisionTower: %smissing mlp.down_proj.weight", bp))
		}
		blockFF := gateT.Shape[0]
		gate, err := visionProj(tensors, bp+"mlp.gate_proj.weight", gateT, blockFF, hidden, quant)
		if err != nil {
			return VisionMLPWeights{}, 0, false, core.E("qwen35.buildVisionTower", bp+"mlp.gate_proj", err)
		}
		gate.B = optionalVisionVec(tensors, bp+"mlp.gate_proj.bias")
		up, err := visionProj(tensors, bp+"mlp.up_proj.weight", upT, blockFF, hidden, quant)
		if err != nil {
			return VisionMLPWeights{}, 0, false, core.E("qwen35.buildVisionTower", bp+"mlp.up_proj", err)
		}
		up.B = optionalVisionVec(tensors, bp+"mlp.up_proj.bias")
		down, err := visionProj(tensors, bp+"mlp.down_proj.weight", downT, hidden, blockFF, quant)
		if err != nil {
			return VisionMLPWeights{}, 0, false, core.E("qwen35.buildVisionTower", bp+"mlp.down_proj", err)
		}
		down.B = optionalVisionVec(tensors, bp+"mlp.down_proj.bias")
		return VisionMLPWeights{Gate: gate, Up: up, Down: down}, blockFF, false, nil
	}
	fc1T, ok := tensors[bp+"mlp.linear_fc1.weight"]
	if !ok {
		return VisionMLPWeights{}, 0, false, core.NewError(core.Sprintf("qwen35.buildVisionTower: %smlp.gate_proj/mlp.linear_fc1 weight is missing", bp))
	}
	if len(fc1T.Shape) != 2 {
		return VisionMLPWeights{}, 0, false, core.NewError(core.Sprintf("qwen35.buildVisionTower: %smlp.linear_fc1 is not 2-D", bp))
	}
	fc2T, ok := tensors[bp+"mlp.linear_fc2.weight"]
	if !ok {
		return VisionMLPWeights{}, 0, false, core.NewError(core.Sprintf("qwen35.buildVisionTower: %smissing mlp.linear_fc2.weight", bp))
	}
	blockFF := fc1T.Shape[0]
	fc1, err := visionProj(tensors, bp+"mlp.linear_fc1.weight", fc1T, blockFF, hidden, quant)
	if err != nil {
		return VisionMLPWeights{}, 0, false, core.E("qwen35.buildVisionTower", bp+"mlp.linear_fc1", err)
	}
	fc1.B = optionalVisionVec(tensors, bp+"mlp.linear_fc1.bias")
	fc2, err := visionProj(tensors, bp+"mlp.linear_fc2.weight", fc2T, hidden, blockFF, quant)
	if err != nil {
		return VisionMLPWeights{}, 0, false, core.E("qwen35.buildVisionTower", bp+"mlp.linear_fc2", err)
	}
	fc2.B = optionalVisionVec(tensors, bp+"mlp.linear_fc2.bias")
	return VisionMLPWeights{FC1: fc1, FC2: fc2, GELU: true}, blockFF, true, nil
}

// buildVisionBlocks loads vision_tower.blocks.<i>.* for i=0,1,2,… until a block's attn.qkv AND
// attn.q_proj are both missing (the counting-probe shape the factory's MoE assembly also uses). The
// fused-vs-separate attention convention and the SwiGLU-vs-GELU MLP convention are each decided ONCE
// at block 0 and every later block is loaded assuming the SAME one — a block whose tensors don't match
// fails loudly, catching a checkpoint that mixes conventions across blocks.
func buildVisionBlocks(tensors map[string]safetensors.Tensor, hidden int, vc *VisionArchConfig, quant *model.QuantConfig) (blocks []VisionBlock, headDim, numHeads, numKVHeads int, err error) {
	var ff int
	var gelu, fused bool
	for i := 0; ; i++ {
		bp := core.Sprintf("vision_tower.blocks.%d.", i)
		_, hasFused := tensors[bp+"attn.qkv.weight"]
		_, hasSeparate := tensors[bp+"attn.q_proj.weight"]
		if !hasFused && !hasSeparate {
			break
		}

		qNorm := optionalVisionVec(tensors, bp+"attn.q_norm.weight")
		kNorm := optionalVisionVec(tensors, bp+"attn.k_norm.weight")

		if i == 0 {
			fused, headDim, numHeads, numKVHeads, err = resolveVisionAttnGeometry(tensors, bp, vc, qNorm)
			if err != nil {
				return nil, 0, 0, 0, err
			}
		}
		q, k, v, err := loadBlockQKV(tensors, bp, hidden, fused, quant)
		if err != nil {
			return nil, 0, 0, 0, core.E("qwen35.buildVisionTower", core.Sprintf("block %d", i), err)
		}

		oName, oT, ok := weightAnyName(tensors, bp+"attn.o_proj.weight", bp+"attn.proj.weight")
		if !ok {
			return nil, 0, 0, 0, core.NewError(core.Sprintf("qwen35.buildVisionTower: block %d missing attn.o_proj/attn.proj weight", i))
		}
		if len(oT.Shape) != 2 {
			return nil, 0, 0, 0, core.NewError(core.Sprintf("qwen35.buildVisionTower: block %d attn output proj is not 2-D", i))
		}
		// inDim is q.Out (the concatenated multi-head width, identical in EITHER convention), NOT
		// oT.Shape[1] — a packed tensor's own Shape[1] is the compressed word count.
		o, err := visionProj(tensors, oName, oT, oT.Shape[0], q.Out, quant)
		if err != nil {
			return nil, 0, 0, 0, core.E("qwen35.buildVisionTower", core.Sprintf("block %d attn output proj", i), err)
		}
		o.B = optionalVisionVec(tensors, biasSibling(oName))

		mlpW, blockFF, mlpGELU, err := loadBlockMLP(tensors, bp, hidden, quant)
		if err != nil {
			return nil, 0, 0, 0, core.E("qwen35.buildVisionTower", core.Sprintf("block %d mlp", i), err)
		}
		if i == 0 {
			ff, gelu = blockFF, mlpGELU
		} else if blockFF != ff || mlpGELU != gelu {
			return nil, 0, 0, 0, core.NewError(core.Sprintf("qwen35.buildVisionTower: block %d MLP shape (FF %d, gelu %v) != block 0's (FF %d, gelu %v)", i, blockFF, mlpGELU, ff, gelu))
		}

		norm1W := optionalVisionVec(tensors, bp+"norm1.weight")
		norm2W := optionalVisionVec(tensors, bp+"norm2.weight")
		if norm1W == nil || norm2W == nil {
			return nil, 0, 0, 0, core.NewError(core.Sprintf("qwen35.buildVisionTower: block %d missing norm1/norm2 weight", i))
		}

		blocks = append(blocks, VisionBlock{
			Norm1W: norm1W, Norm1B: optionalVisionVec(tensors, bp+"norm1.bias"),
			Norm2W: norm2W, Norm2B: optionalVisionVec(tensors, bp+"norm2.bias"),
			Attn: VisionAttnWeights{Q: q, K: k, V: v, O: o, QNorm: qNorm, KNorm: kNorm},
			MLP:  mlpW,
		})
	}
	if len(blocks) == 0 {
		return nil, 0, 0, 0, core.NewError("qwen35.buildVisionTower: vision_tower.patch_embed present but no vision_tower.blocks.0.* found")
	}
	return blocks, headDim, numHeads, numKVHeads, nil
}

// buildVisionMerger loads the vision-to-text merger/projector under either alias family — the GUESSED
// layout's multi_modal_projector.* (probing .mlp.0/.mlp.2 as further aliases) or the REAL layout's
// vision_tower.merger.*. The spatial merge size is DERIVED from linear_1's own width as a multiple of
// hidden (must be a perfect square) rather than trusted from config; a present
// vision_config.spatial_merge_size is cross-validated against the derived value, and linear_2's output
// width is cross-validated against the text hidden the same way.
func buildVisionMerger(tensors map[string]safetensors.Tensor, hidden, textHidden int, vc *VisionArchConfig, quant *model.QuantConfig) (merger *VisionMerger, mergeSize int, err error) {
	l1Name, l1T, ok := weightAnyName(tensors,
		"multi_modal_projector.linear_1.weight", "multi_modal_projector.mlp.0.weight",
		"vision_tower.merger.linear_fc1.weight")
	if !ok {
		return nil, 0, core.NewError("qwen35.buildVisionTower: vision_tower present but multi_modal_projector.linear_1(.mlp.0)/vision_tower.merger.linear_fc1 weight is missing")
	}
	if len(l1T.Shape) != 2 {
		return nil, 0, core.NewError("qwen35.buildVisionTower: merger linear_1 is not 2-D")
	}
	// l1In can't be read off l1T.Shape[1] directly on a packed checkpoint (compressed word count) and
	// the merge size is DERIVED from it — the same visionProjInDim recovery patch_embed uses.
	l1Out := l1T.Shape[0]
	l1In, err := visionProjInDim(tensors, l1Name, l1T, quant)
	if err != nil {
		return nil, 0, core.E("qwen35.buildVisionTower", "merger linear_1", err)
	}
	if hidden <= 0 || l1In%hidden != 0 {
		return nil, 0, core.NewError(core.Sprintf("qwen35.buildVisionTower: merger linear_1 input %d is not a multiple of the tower hidden size %d", l1In, hidden))
	}
	mergeSq := l1In / hidden
	mergeSize = isqrt(mergeSq)
	if mergeSize <= 0 || mergeSize*mergeSize != mergeSq {
		return nil, 0, core.NewError(core.Sprintf("qwen35.buildVisionTower: merger linear_1 input %d is not hidden(%d)·mergeSize² for an integer mergeSize", l1In, hidden))
	}
	if vc != nil && vc.SpatialMergeSize > 0 && vc.SpatialMergeSize != mergeSize {
		return nil, 0, core.NewError(core.Sprintf("qwen35.buildVisionTower: derived merge size %d != config spatial_merge_size %d", mergeSize, vc.SpatialMergeSize))
	}
	l1Lin, err := visionProj(tensors, l1Name, l1T, l1Out, l1In, quant)
	if err != nil {
		return nil, 0, core.E("qwen35.buildVisionTower", "merger linear_1", err)
	}
	l1Lin.B = optionalVisionVec(tensors, biasSibling(l1Name))

	l2Name, l2T, ok := weightAnyName(tensors,
		"multi_modal_projector.linear_2.weight", "multi_modal_projector.mlp.2.weight",
		"vision_tower.merger.linear_fc2.weight")
	if !ok {
		return nil, 0, core.NewError("qwen35.buildVisionTower: vision_tower present but multi_modal_projector.linear_2(.mlp.2)/vision_tower.merger.linear_fc2 weight is missing")
	}
	if len(l2T.Shape) != 2 {
		return nil, 0, core.NewError("qwen35.buildVisionTower: merger linear_2 is not 2-D")
	}
	if textHidden > 0 && l2T.Shape[0] != textHidden {
		return nil, 0, core.NewError(core.Sprintf("qwen35.buildVisionTower: merger output width %d != text model hidden size %d (config/checkpoint mismatch)", l2T.Shape[0], textHidden))
	}
	// inDim is l1Out (linear_2 chains directly off linear_1's output rows — never packed, always
	// trustworthy), NOT l2T.Shape[1].
	l2Lin, err := visionProj(tensors, l2Name, l2T, l2T.Shape[0], l1Out, quant)
	if err != nil {
		return nil, 0, core.E("qwen35.buildVisionTower", "merger linear_2", err)
	}
	l2Lin.B = optionalVisionVec(tensors, biasSibling(l2Name))

	normW := optionalVisionVec(tensors, "multi_modal_projector.norm.weight")
	normB := optionalVisionVec(tensors, "multi_modal_projector.norm.bias")
	if normW == nil {
		normW = optionalVisionVec(tensors, "multi_modal_projector.ln_q.weight")
		normB = optionalVisionVec(tensors, "multi_modal_projector.ln_q.bias")
	}
	if normW == nil {
		normW = optionalVisionVec(tensors, "vision_tower.merger.norm.weight")
		normB = optionalVisionVec(tensors, "vision_tower.merger.norm.bias")
	}
	if normW == nil {
		return nil, 0, core.NewError("qwen35.buildVisionTower: merger pre-norm weight (norm.weight / ln_q.weight) is missing")
	}

	return &VisionMerger{
		NormW: normW, NormB: normB,
		L1: l1Lin, L2: l2Lin,
	}, mergeSize, nil
}
