// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/quant/mlxaffine"
	"dappco.re/go/inference/model/safetensors"
)

// vision_loader.go builds a *visionTower from a checkpoint's tensors, mirroring loader.go's own house
// style (buildAttn/buildGatedDelta/buildFFN): probe by name, derive geometry from the WEIGHTS themselves
// wherever a tensor shape settles the question unambiguously, and only fall back to config.json when a
// dimension has no tensor to derive it from (PatchSize — pixel geometry) or the checkpoint carries no
// disambiguating tensor (NumHeads, only when there is no q_norm to read HeadDim off directly). This is
// deliberately MORE weight-derived than the text loader needs to be: "Qwen3.5/3.6" vision_config field
// names are not yet a settled, load-bearing contract the way num_hidden_layers/hidden_size are for the
// text stack, so deriving Hidden/PatchDim/Depth/HeadDim/NumKVHeads/FF/MergeSize from tensor shapes (like
// buildGatedDelta already does for the linear-attention geometry) is the robust choice, not a shortcut.
//
// TWO checkpoint conventions resolve here, distinguished purely by which tensor names are present (never a
// model_type switch):
//
//   - the "GUESSED" layout (separate q/k/v/o projections, SwiGLU MLP, RMS q/k-norms, 2-D rotary positions)
//     this package shipped before a real checkpoint was available to check it against:
//
//     vision_tower.patch_embed(.proj).weight/.bias           [Hidden, InChannels·TemporalPatchSize·PatchSize²]
//     vision_tower.blocks.<i>.norm1/.norm2.weight/.bias        [Hidden]
//     vision_tower.blocks.<i>.attn.q_proj/.k_proj/.v_proj/.o_proj.weight/.bias
//     vision_tower.blocks.<i>.attn.q_norm/.k_norm.weight       [HeadDim]                     (optional)
//     vision_tower.blocks.<i>.mlp.gate_proj/.up_proj/.down_proj.weight/.bias
//     multi_modal_projector.norm(/.ln_q).weight/.bias          [Hidden]
//     multi_modal_projector.linear_1(.mlp.0)/.linear_2(.mlp.2).weight/.bias
//
//   - the REAL layout (verified against mlx-community/Qwen3.6-27B-4bit — 333 vision tensors, 21 patterns;
//     see vision_loader_real_test.go's reconciliation receipt): a FUSED single attn.qkv linear (split by
//     output row at load — splitFusedQKV), a plain 2-linear GELU mlp.linear_fc1/linear_fc2 (no SwiGLU), full
//     LayerNorm-with-bias norms (already this package's only norm shape — layerNormRowsWithBias in
//     vision.go), NO q_norm/k_norm, and a LEARNED additive vision_tower.pos_embed.weight instead of 2-D
//     rotary positions:
//
//     vision_tower.patch_embed.proj.weight/.bias                (weight may be >2-D — e.g. [Hidden,T,P,P,C]
//                                                                 — patchDim is the FLATTENED product of every
//                                                                 dim after Hidden; the flat bytes are
//                                                                 identical to a [Hidden,PatchDim] tensor,
//                                                                 and that trailing order — T,then row,then
//                                                                 col,then channel — matches
//                                                                 imageToPatchGrid's own patch-row layout)
//     vision_tower.pos_embed.weight                             [NumPositions, Hidden]
//     vision_tower.blocks.<i>.norm1/.norm2.weight/.bias         [Hidden]
//     vision_tower.blocks.<i>.attn.qkv.weight/.bias             [3·PerHead, Hidden] (PerHead=Hidden, plain
//                                                                 MHA — split into equal Q/K/V thirds by
//                                                                 OUTPUT ROW, see splitFusedQKV)
//     vision_tower.blocks.<i>.attn.proj.weight/.bias            [Hidden, Hidden]
//     vision_tower.blocks.<i>.mlp.linear_fc1/.linear_fc2.weight/.bias
//     vision_tower.merger.norm.weight/.bias                     [Hidden]
//     vision_tower.merger.linear_fc1/.linear_fc2.weight/.bias
//
// Both conventions share the SAME forward math in vision.go (visionBlock.forward, visionAttentionForward,
// visionMLPForward, visionTowerForward) — the divergences are load-time-only facts: which tensor names
// resolved selects a MODE on the loaded weights (visionMLPWeights.GELU, visionTowerCfg.LearnedPositions),
// never a second code path through the maths.
//
// Quantisation: on a quantised checkpoint (mlx affine packed-uint32 weights with .scales/.biases siblings)
// every 2-D PROJECTION weight (patch embed, each block's attention/MLP linears, the merger's linears)
// resolves through visionProj — the SAME tensorAsQuant/tensorAsF32 split loader.go's own proj/f32 closures
// run for the composed TEXT stack's projections — so a quantised vision tower stays PACKED (visionLinear.WQ,
// model.QuantWeight, matNTQuant — vision.go) exactly like the text stack does, rather than tensorF32
// hard-failing on a dtype it doesn't recognise. Small vectors (biases, norms, the learned position table)
// are never quantised in any observed scheme (an additive lookup/bias, not a GEMM weight) and stay on the
// plain host-f32 path via optionalVisionVec, mirroring which fields loader.go's own f32/f32opt (as opposed
// to proj) resolve for the text stack.
//
// A checkpoint carrying neither layout's patch_embed tensor returns (nil, nil) — the text-only path,
// unchanged from before this file existed.

// optionalVisionVec returns tensors[name] widened to f32, or nil when absent/unwidenable — the bias/
// qk-norm/norm/learned-position-table probe this file uses throughout, mirroring loader.go's f32opt for the
// text stack. Never quant-aware (see the file doc comment): none of the small vectors this resolves are
// ever packed in a known checkpoint.
func optionalVisionVec(tensors map[string]safetensors.Tensor, name string) []float32 {
	if t, ok := tensors[name]; ok {
		if v, err := tensorF32(t); err == nil {
			return v
		}
	}
	return nil
}

// weightAnyName is model.WeightAny plus the MATCHED name — needed wherever the result feeds visionProj
// (and, through it, tensorAsQuant/tensorAsF32 in loader.go), which derive a quantised tensor's .scales/
// .biases sibling keys from the LOOKED-UP name itself; model.WeightAny alone reports which tensor matched
// but discards which of several alias spellings it was, so a caller resolving an aliased name (patch_embed,
// the merger's linears, the attention output projection) couldn't compute the right sibling keys from it.
func weightAnyName(tensors map[string]safetensors.Tensor, names ...string) (name string, t safetensors.Tensor, ok bool) {
	for _, n := range names {
		if tv, found := tensors[n]; found {
			return n, tv, true
		}
	}
	return "", safetensors.Tensor{}, false
}

// biasSibling turns a resolved "<x>.weight" tensor name into its "<x>.bias" sibling — for the alias-probed
// projections (patch_embed, the merger's linears, the attention output projection) whose bias lives under
// whichever alias actually matched; optionalVisionVec then probes it exactly like every statically-named
// bias elsewhere in this file.
func biasSibling(name string) string {
	if core.HasSuffix(name, ".weight") {
		return name[:len(name)-len(".weight")] + ".bias"
	}
	return name + ".bias"
}

// isqrt returns the integer square root of n (⌊√n⌋), or -1 for n<0 — used to recover the merger's spatial
// merge size from its first linear's width as a multiple of Hidden (must be a perfect square).
func isqrt(n int) int {
	if n < 0 {
		return -1
	}
	r := int(math.Sqrt(float64(n)))
	for r*r > n {
		r--
	}
	for (r+1)*(r+1) <= n {
		r++
	}
	return r
}

// visionProj resolves name (a 2-D projection weight, already looked up as t) to a visionLinear: PACKED
// (WQ set, W nil) when the checkpoint carries name's .scales/.biases siblings, else widened to host f32 (W
// set, WQ nil) — never both. outDim/inDim are the projection's LOGICAL shape, supplied by the caller rather
// than read off t.Shape, so a tensor whose dense form ships >2-D (the real layout's patch_embed — see the
// file doc comment) resolves the same way a genuinely 2-D one does: tensorAsQuant only inspects t.Shape
// when a PACKED sibling pair exists, and MLX affine packing always flattens to 2-D regardless of the source
// tensor's dense rank, so the shape check never fires for a >2-D dense tensor (no scales/biases ⇒ it
// returns early). Runs the SAME tensorAsQuant/tensorF32 pair loader.go's own proj closure calls for the
// composed text stack, so a quantised vision projection stays packed through to matNTQuant exactly like a
// quantised text projection does. alias reports whether the packed form aliases the input tensors' memory
// (zeroCopy AND not repacked), mirroring tensorAsQuant's own contract.
func visionProj(tensors map[string]safetensors.Tensor, name string, t safetensors.Tensor, outDim, inDim int, quant *model.QuantConfig, zeroCopy bool) (lin visionLinear, alias bool, err error) {
	qw, aliased, err := tensorAsQuant(tensors, name, t, quant, zeroCopy)
	if err != nil {
		return visionLinear{}, false, err
	}
	if qw != nil {
		if qw.OutDim != outDim || qw.InDim != inDim {
			return visionLinear{}, false, core.NewError(core.Sprintf("composed.buildVisionTower: %s packed geometry %dx%d != expected %dx%d", name, qw.OutDim, qw.InDim, outDim, inDim))
		}
		return visionLinear{WQ: qw, Out: outDim, In: inDim}, aliased, nil
	}
	f, err := tensorF32(t)
	if err != nil {
		return visionLinear{}, false, err
	}
	if len(f) != outDim*inDim {
		return visionLinear{}, false, core.NewError(core.Sprintf("composed.buildVisionTower: %s width %d != expected %d·%d", name, len(f), outDim, inDim))
	}
	return visionLinear{W: f, Out: outDim, In: inDim}, false, nil
}

// visionProjInDim returns name's LOGICAL input width — needed ONLY for patch_embed, the one vision
// projection whose DENSE form ships >2-D (e.g. [Hidden,Temporal,PatchH,PatchW,Channel] — see the file doc
// comment); every other vision projection is genuinely 2-D even dense, so t.Shape[1] alone is already its
// logical width and callers use it directly. For a dense t this flattens Shape[1:] (row-major, so the flat
// bytes are unaffected by how many dims that product spans). For a PACKED t (name's .scales sibling
// present) t.Shape[1] is the COMPRESSED packed-word count, not the logical width — model.LoadLinear's own
// doc comment: "a packed weight's columns differ" — so this instead recovers it from the .scales sibling's
// shape (nGroups·groupSize), the SAME derivation tensorAsF32/tensorAsQuant use internally (loader.go).
func visionProjInDim(tensors map[string]safetensors.Tensor, name string, t safetensors.Tensor, quant *model.QuantConfig) (int, error) {
	base := name
	if core.HasSuffix(base, ".weight") {
		base = base[:len(base)-len(".weight")]
	}
	scalesT, sOK := tensors[base+".scales"]
	if !sOK {
		dim := 1
		for _, d := range t.Shape[1:] {
			dim *= d
		}
		return dim, nil
	}
	if quant == nil {
		return 0, core.NewError("composed.buildVisionTower: " + name + " carries .scales but the config has no quantization block")
	}
	if len(scalesT.Shape) != 2 {
		return 0, core.NewError("composed.buildVisionTower: " + name + " .scales is not 2-D")
	}
	gs, _ := quant.For(base)
	if gs <= 0 {
		return 0, core.NewError("composed.buildVisionTower: " + name + " is quantised but its group_size is not positive")
	}
	return scalesT.Shape[1] * gs, nil
}

// splitFusedQKV splits one fused vision_tower.blocks.<i>.attn.qkv visionLinear (Out = 3·PerHead, In =
// Hidden — the REAL layout's single QKV linear) into three [PerHead,Hidden] projections by OUTPUT ROW: rows
// [0,PerHead) are Q (every head), [PerHead,2·PerHead) K, [2·PerHead,3·PerHead) V. This is the row order
// torch's reshape(L, 3, heads, headDim).permute(...) convention guarantees for a fused qkv Linear — each of
// the 3 groups occupies a Hidden-wide contiguous OUTPUT-ROW band before the next group starts, so splitting
// by row is exact, not an approximation (see the reference Qwen2VLVisionAttention/Qwen3VLVisionAttention
// .qkv). A DENSE weight splits by a flat row-slice of W; a PACKED (quantised) weight splits the SAME way
// over Packed/Scales/Biases, since MLX affine keeps each output row byte-contiguous (wordsPerRow packed
// words, groupsPerRow scale/bias pairs — the same row-offset arithmetic composed.go's matNTQuantHost already
// uses) regardless of quantisation: row-contiguity, not element width, is what makes a row-band split valid.
// Assumes q/k/v are EQUAL width (plain MHA) — the fused-qkv convention carries no per-branch head-count
// signal to derive an uneven GQA split from, and no known Qwen-VL checkpoint ships one (confirmed against
// the real checkpoint: vision_config carries no num_key_value_heads at all).
func splitFusedQKV(fused visionLinear) (q, k, v visionLinear, err error) {
	if fused.Out <= 0 || fused.Out%3 != 0 {
		return visionLinear{}, visionLinear{}, visionLinear{}, core.NewError(core.Sprintf("composed.buildVisionTower: fused qkv width %d is not divisible by 3", fused.Out))
	}
	per := fused.Out / 3
	band := func(i int) visionLinear {
		lin := visionLinear{Out: per, In: fused.In}
		if fused.WQ != nil {
			wordsPerRow := mlxaffine.PackedWords(fused.In, fused.WQ.Bits)
			groupsPerRow := fused.In / fused.WQ.GroupSize
			lin.WQ = &model.QuantWeight{
				Packed:    fused.WQ.Packed[i*per*wordsPerRow*4 : (i+1)*per*wordsPerRow*4],
				Scales:    fused.WQ.Scales[i*per*groupsPerRow*2 : (i+1)*per*groupsPerRow*2],
				Biases:    fused.WQ.Biases[i*per*groupsPerRow*2 : (i+1)*per*groupsPerRow*2],
				Bits:      fused.WQ.Bits,
				GroupSize: fused.WQ.GroupSize,
				OutDim:    per,
				InDim:     fused.In,
			}
		} else {
			lin.W = fused.W[i*per*fused.In : (i+1)*per*fused.In]
		}
		if len(fused.B) > 0 {
			lin.B = fused.B[i*per : (i+1)*per]
		}
		return lin
	}
	return band(0), band(1), band(2), nil
}

// resolveVisionAttnGeometry derives (fused, headDim, numHeads, numKVHeads) from ONE block's attention
// tensors — fused reports whether this checkpoint uses the REAL layout's single attn.qkv linear (true) or
// the GUESSED layout's separate attn.q_proj/k_proj/v_proj (false); the caller (buildVisionBlocksQuant) runs
// this ONCE at block 0 and loads every later block under the same decided convention (loadBlockQKV). HeadDim
// prefers q_norm's own width (unambiguous — a per-head RMSNorm scale is exactly HeadDim wide); only when no
// q_norm tensor exists does it fall back to vision_config.num_heads (mirrors buildAttn's identical text-side
// fallback). The fused layout has no per-branch head-count signal to derive an uneven q/k/v split from, so
// numKVHeads==numHeads always on that path (plain MHA — see splitFusedQKV).
func resolveVisionAttnGeometry(tensors map[string]safetensors.Tensor, bp string, vc *visionConfig, qNorm []float32) (fused bool, headDim, numHeads, numKVHeads int, err error) {
	if fusedT, ok := tensors[bp+"attn.qkv.weight"]; ok {
		if len(fusedT.Shape) != 2 {
			return false, 0, 0, 0, core.NewError(core.Sprintf("composed.buildVisionTower: %sattn.qkv is not 2-D", bp))
		}
		if fusedT.Shape[0] <= 0 || fusedT.Shape[0]%3 != 0 {
			return false, 0, 0, 0, core.NewError(core.Sprintf("composed.buildVisionTower: %sattn.qkv width %d is not divisible by 3", bp, fusedT.Shape[0]))
		}
		perQKV := fusedT.Shape[0] / 3
		switch {
		case len(qNorm) > 0:
			headDim = len(qNorm)
		case vc != nil && vc.NumHeads > 0:
			headDim = perQKV / vc.NumHeads
		}
		if headDim <= 0 || perQKV%headDim != 0 {
			return false, 0, 0, 0, core.NewError("composed.buildVisionTower: cannot derive attention head_dim (no q_norm tensor and no usable vision_config.num_heads)")
		}
		return true, headDim, perQKV / headDim, perQKV / headDim, nil
	}
	qT, ok := tensors[bp+"attn.q_proj.weight"]
	if !ok {
		return false, 0, 0, 0, core.NewError(core.Sprintf("composed.buildVisionTower: %sattn.qkv/attn.q_proj weight is missing", bp))
	}
	if len(qT.Shape) != 2 {
		return false, 0, 0, 0, core.NewError(core.Sprintf("composed.buildVisionTower: %sq_proj is not 2-D", bp))
	}
	kT, ok := tensors[bp+"attn.k_proj.weight"]
	if !ok {
		return false, 0, 0, 0, core.NewError(core.Sprintf("composed.buildVisionTower: %smissing attn.k_proj.weight", bp))
	}
	switch {
	case len(qNorm) > 0:
		headDim = len(qNorm)
	case vc != nil && vc.NumHeads > 0:
		headDim = qT.Shape[0] / vc.NumHeads
	}
	if headDim <= 0 || qT.Shape[0]%headDim != 0 {
		return false, 0, 0, 0, core.NewError("composed.buildVisionTower: cannot derive attention head_dim (no q_norm tensor and no usable vision_config.num_heads)")
	}
	numHeads = qT.Shape[0] / headDim
	if kT.Shape[0]%headDim != 0 {
		return false, 0, 0, 0, core.NewError(core.Sprintf("composed.buildVisionTower: k_proj rows %d not a multiple of derived head_dim %d", kT.Shape[0], headDim))
	}
	numKVHeads = kT.Shape[0] / headDim
	if vc != nil && vc.NumKeyValueHeads > 0 && vc.NumKeyValueHeads != numKVHeads {
		return false, 0, 0, 0, core.NewError(core.Sprintf("composed.buildVisionTower: derived kv heads %d != config num_key_value_heads %d", numKVHeads, vc.NumKeyValueHeads))
	}
	return false, headDim, numHeads, numKVHeads, nil
}

// loadBlockQKV resolves one block's Q/K/V projections under the ALREADY-DECIDED fused/separate convention
// (resolveVisionAttnGeometry, called once at block 0) — the REAL layout's single attn.qkv (split by output
// row, splitFusedQKV) or the GUESSED layout's separate attn.q_proj/k_proj/v_proj. A block whose tensors
// don't match the decided convention fails loudly here (a missing-key error), catching a checkpoint that
// mixes conventions across blocks.
func loadBlockQKV(tensors map[string]safetensors.Tensor, bp string, hidden int, fused bool, quant *model.QuantConfig, zeroCopy bool) (q, k, v visionLinear, alias bool, err error) {
	if fused {
		fusedT, ok := tensors[bp+"attn.qkv.weight"]
		if !ok {
			return visionLinear{}, visionLinear{}, visionLinear{}, false, core.NewError(core.Sprintf("composed.buildVisionTower: %smissing attn.qkv.weight", bp))
		}
		fusedLin, fa, ferr := visionProj(tensors, bp+"attn.qkv.weight", fusedT, fusedT.Shape[0], hidden, quant, zeroCopy)
		if ferr != nil {
			return visionLinear{}, visionLinear{}, visionLinear{}, false, core.E("composed.buildVisionTower", bp+"attn.qkv", ferr)
		}
		fusedLin.B = optionalVisionVec(tensors, bp+"attn.qkv.bias")
		q, k, v, ferr = splitFusedQKV(fusedLin)
		if ferr != nil {
			return visionLinear{}, visionLinear{}, visionLinear{}, false, core.E("composed.buildVisionTower", bp+"attn.qkv split", ferr)
		}
		return q, k, v, fa, nil
	}
	qT, ok := tensors[bp+"attn.q_proj.weight"]
	if !ok {
		return visionLinear{}, visionLinear{}, visionLinear{}, false, core.NewError(core.Sprintf("composed.buildVisionTower: %smissing attn.q_proj.weight", bp))
	}
	kT, ok := tensors[bp+"attn.k_proj.weight"]
	if !ok {
		return visionLinear{}, visionLinear{}, visionLinear{}, false, core.NewError(core.Sprintf("composed.buildVisionTower: %smissing attn.k_proj.weight", bp))
	}
	vT, ok := tensors[bp+"attn.v_proj.weight"]
	if !ok {
		return visionLinear{}, visionLinear{}, visionLinear{}, false, core.NewError(core.Sprintf("composed.buildVisionTower: %smissing attn.v_proj.weight", bp))
	}
	// inDim is `hidden` (the caller's own known tower Hidden), NOT qT/kT/vT.Shape[1]: a packed tensor's
	// own Shape[1] is the compressed word count (see visionProjInDim's doc comment for the general rule) —
	// q/k/v all read the SAME hidden-wide input, so the true logical inDim is already known independent of
	// whichever tensor's shape we're looking at.
	qLin, qa, err := visionProj(tensors, bp+"attn.q_proj.weight", qT, qT.Shape[0], hidden, quant, zeroCopy)
	if err != nil {
		return visionLinear{}, visionLinear{}, visionLinear{}, false, core.E("composed.buildVisionTower", bp+"attn.q_proj", err)
	}
	qLin.B = optionalVisionVec(tensors, bp+"attn.q_proj.bias")
	kLin, ka, err := visionProj(tensors, bp+"attn.k_proj.weight", kT, kT.Shape[0], hidden, quant, zeroCopy)
	if err != nil {
		return visionLinear{}, visionLinear{}, visionLinear{}, false, core.E("composed.buildVisionTower", bp+"attn.k_proj", err)
	}
	kLin.B = optionalVisionVec(tensors, bp+"attn.k_proj.bias")
	vLin, va, err := visionProj(tensors, bp+"attn.v_proj.weight", vT, vT.Shape[0], hidden, quant, zeroCopy)
	if err != nil {
		return visionLinear{}, visionLinear{}, visionLinear{}, false, core.E("composed.buildVisionTower", bp+"attn.v_proj", err)
	}
	vLin.B = optionalVisionVec(tensors, bp+"attn.v_proj.bias")
	return qLin, kLin, vLin, qa || ka || va, nil
}

// loadBlockMLP resolves one block's feed-forward — the GUESSED layout's SwiGLU (mlp.gate_proj/up_proj/
// down_proj, gelu=false) when gate_proj is present, else the REAL layout's plain 2-linear GELU
// (mlp.linear_fc1/linear_fc2, gelu=true). ff is the hidden FF width (gate's or fc1's Out) the caller checks
// for uniformity across blocks.
func loadBlockMLP(tensors map[string]safetensors.Tensor, bp string, hidden int, quant *model.QuantConfig, zeroCopy bool) (w visionMLPWeights, ff int, gelu, alias bool, err error) {
	if gateT, ok := tensors[bp+"mlp.gate_proj.weight"]; ok {
		if len(gateT.Shape) != 2 {
			return visionMLPWeights{}, 0, false, false, core.NewError(core.Sprintf("composed.buildVisionTower: %smlp.gate_proj is not 2-D", bp))
		}
		upT, ok := tensors[bp+"mlp.up_proj.weight"]
		if !ok {
			return visionMLPWeights{}, 0, false, false, core.NewError(core.Sprintf("composed.buildVisionTower: %smissing mlp.up_proj.weight", bp))
		}
		downT, ok := tensors[bp+"mlp.down_proj.weight"]
		if !ok {
			return visionMLPWeights{}, 0, false, false, core.NewError(core.Sprintf("composed.buildVisionTower: %smissing mlp.down_proj.weight", bp))
		}
		blockFF := gateT.Shape[0]
		gate, ga, err := visionProj(tensors, bp+"mlp.gate_proj.weight", gateT, blockFF, hidden, quant, zeroCopy)
		if err != nil {
			return visionMLPWeights{}, 0, false, false, core.E("composed.buildVisionTower", bp+"mlp.gate_proj", err)
		}
		gate.B = optionalVisionVec(tensors, bp+"mlp.gate_proj.bias")
		up, ua, err := visionProj(tensors, bp+"mlp.up_proj.weight", upT, blockFF, hidden, quant, zeroCopy)
		if err != nil {
			return visionMLPWeights{}, 0, false, false, core.E("composed.buildVisionTower", bp+"mlp.up_proj", err)
		}
		up.B = optionalVisionVec(tensors, bp+"mlp.up_proj.bias")
		down, da, err := visionProj(tensors, bp+"mlp.down_proj.weight", downT, hidden, blockFF, quant, zeroCopy)
		if err != nil {
			return visionMLPWeights{}, 0, false, false, core.E("composed.buildVisionTower", bp+"mlp.down_proj", err)
		}
		down.B = optionalVisionVec(tensors, bp+"mlp.down_proj.bias")
		return visionMLPWeights{Gate: gate, Up: up, Down: down}, blockFF, false, ga || ua || da, nil
	}
	fc1T, ok := tensors[bp+"mlp.linear_fc1.weight"]
	if !ok {
		return visionMLPWeights{}, 0, false, false, core.NewError(core.Sprintf("composed.buildVisionTower: %smlp.gate_proj/mlp.linear_fc1 weight is missing", bp))
	}
	if len(fc1T.Shape) != 2 {
		return visionMLPWeights{}, 0, false, false, core.NewError(core.Sprintf("composed.buildVisionTower: %smlp.linear_fc1 is not 2-D", bp))
	}
	fc2T, ok := tensors[bp+"mlp.linear_fc2.weight"]
	if !ok {
		return visionMLPWeights{}, 0, false, false, core.NewError(core.Sprintf("composed.buildVisionTower: %smissing mlp.linear_fc2.weight", bp))
	}
	blockFF := fc1T.Shape[0]
	fc1, fa, err := visionProj(tensors, bp+"mlp.linear_fc1.weight", fc1T, blockFF, hidden, quant, zeroCopy)
	if err != nil {
		return visionMLPWeights{}, 0, false, false, core.E("composed.buildVisionTower", bp+"mlp.linear_fc1", err)
	}
	fc1.B = optionalVisionVec(tensors, bp+"mlp.linear_fc1.bias")
	fc2, f2a, err := visionProj(tensors, bp+"mlp.linear_fc2.weight", fc2T, hidden, blockFF, quant, zeroCopy)
	if err != nil {
		return visionMLPWeights{}, 0, false, false, core.E("composed.buildVisionTower", bp+"mlp.linear_fc2", err)
	}
	fc2.B = optionalVisionVec(tensors, bp+"mlp.linear_fc2.bias")
	return visionMLPWeights{FC1: fc1, FC2: fc2, GELU: true}, blockFF, true, fa || f2a, nil
}

// buildVisionTower probes tensors for a Qwen-VL-family vision tower + merger (either supported layout — see
// the file doc comment) and, when found, assembles a *visionTower with every geometric field DERIVED from
// the tensors' own shapes. Returns (nil, nil) — not an error — when no patch_embed tensor is present: the
// checkpoint is text-only and loadComposed's caller leaves ComposedModel.Vision nil, loading exactly as
// before this file existed. This is the non-quant, owned-copy entry (mirrors LoadComposed vs
// LoadComposedWithArchMmap); loadComposed itself calls buildVisionTowerQuant directly with the checkpoint's
// real quant block + zero-copy choice.
func buildVisionTower(tensors map[string]safetensors.Tensor, vc *visionConfig, textHidden int) (*visionTower, error) {
	tower, _, err := buildVisionTowerQuant(tensors, vc, textHidden, nil, false)
	return tower, err
}

// buildVisionBlocks is buildVisionBlocksQuant with no quantisation and no zero-copy aliasing — see
// buildVisionTower's identical relationship to buildVisionTowerQuant.
func buildVisionBlocks(tensors map[string]safetensors.Tensor, hidden int, vc *visionConfig) (blocks []visionBlock, headDim, numHeads, numKVHeads int, err error) {
	blocks, headDim, numHeads, numKVHeads, _, err = buildVisionBlocksQuant(tensors, hidden, vc, nil, false)
	return blocks, headDim, numHeads, numKVHeads, err
}

// buildVisionMerger is buildVisionMergerQuant with no quantisation and no zero-copy aliasing — see
// buildVisionTower's identical relationship to buildVisionTowerQuant.
func buildVisionMerger(tensors map[string]safetensors.Tensor, hidden, textHidden int, vc *visionConfig) (*visionMerger, int, error) {
	merger, mergeSize, _, err := buildVisionMergerQuant(tensors, hidden, textHidden, vc, nil, false)
	return merger, mergeSize, err
}

// buildVisionTowerQuant is buildVisionTower plus the checkpoint's quant block and the zero-copy/owned-copy
// choice loadComposed already makes for the text stack: every 2-D projection weight (patch embed, each
// block's attention/MLP linears, the merger's linears) resolves through visionProj — the SAME
// tensorAsQuant/tensorAsF32 pair the text side's proj/f32 closures call (loader.go) — so a quantised vision
// tower stays packed exactly like the text stack does. alias reports whether any vision weight ended up
// aliasing the input tensors' memory (zeroCopy AND not repacked), mirroring loadComposed's own anyAlias so
// the caller can fold it into the model's retained mapping.
func buildVisionTowerQuant(tensors map[string]safetensors.Tensor, vc *visionConfig, textHidden int, quant *model.QuantConfig, zeroCopy bool) (tower *visionTower, alias bool, err error) {
	patchName, patchT, ok := weightAnyName(tensors, "vision_tower.patch_embed.weight", "vision_tower.patch_embed.proj.weight")
	if !ok {
		return nil, false, nil // text-only checkpoint
	}
	if len(patchT.Shape) < 2 {
		return nil, false, core.NewError("composed.buildVisionTower: patch_embed weight has no input dimensions")
	}
	hidden := patchT.Shape[0]
	// patchDim is patch_embed's LOGICAL input width — see visionProjInDim's doc comment for why this
	// can't simply be "flatten Shape[1:]" once a quantised checkpoint is in play (a packed tensor's own
	// Shape[1] is the compressed word count, not the logical width). For the dense case this reduces to
	// exactly that flatten, and row-major flattening the real layout's >2-D dense shape
	// ([Hidden,Temporal,PatchH,PatchW,Channel]) is byte-identical to an equivalent flat [Hidden,PatchDim]
	// tensor — see the file doc comment for why that flatten order matches imageToPatchGrid's own
	// patch-row layout exactly.
	patchDim, err := visionProjInDim(tensors, patchName, patchT, quant)
	if err != nil {
		return nil, false, core.E("composed.buildVisionTower", "patch_embed weight", err)
	}
	patchLin, patchAlias, err := visionProj(tensors, patchName, patchT, hidden, patchDim, quant, zeroCopy)
	if err != nil {
		return nil, false, core.E("composed.buildVisionTower", "patch_embed weight", err)
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
		return nil, false, core.NewError("composed.buildVisionTower: vision_tower present but vision_config.patch_size is missing")
	}
	perFrame := inChannels * patchSize * patchSize
	if perFrame <= 0 || patchDim%perFrame != 0 {
		return nil, false, core.NewError(core.Sprintf("composed.buildVisionTower: patch_embed input width %d is not a multiple of in_channels·patch_size² %d", patchDim, perFrame))
	}
	temporal := patchDim / perFrame

	blocks, headDim, numHeads, numKVHeads, blocksAlias, err := buildVisionBlocksQuant(tensors, hidden, vc, quant, zeroCopy)
	if err != nil {
		return nil, false, err
	}

	merger, mergeSize, mergerAlias, err := buildVisionMergerQuant(tensors, hidden, textHidden, vc, quant, zeroCopy)
	if err != nil {
		return nil, false, err
	}

	// The REAL layout's learned absolute position embedding (vision_tower.pos_embed.weight) — additive,
	// added once after the patch embed (visionTowerForward) instead of the GUESSED layout's per-block 2-D
	// rotary embedding. Never packed in any observed checkpoint (an additive lookup table, not a GEMM
	// weight) — resolved on the plain host-f32 path like every other small vision vector.
	posEmbed := optionalVisionVec(tensors, "vision_tower.pos_embed.weight")

	return &visionTower{
		Patch:    patchLin,
		PosEmbed: posEmbed,
		Blocks:   blocks,
		Merger:   *merger,
		Cfg: visionTowerCfg{
			Hidden: hidden, PatchDim: patchDim,
			NumHeads: numHeads, NumKVHeads: numKVHeads, HeadDim: headDim,
			PatchSize: patchSize, InChannels: inChannels, TemporalPatchSize: temporal,
			MergeSize: mergeSize, TextHidden: textHidden,
			RopeTheta: ropeTheta, Eps: eps,
			LearnedPositions: len(posEmbed) > 0,
			// FF is carried on the MLP linears themselves (Gate.Out/FC1.Out), not on visionTowerCfg — the
			// text stack's own MLP type follows the same shape (FF lives on the *MLP, not passed alongside).
		},
	}, patchAlias || blocksAlias || mergerAlias, nil
}

// buildVisionBlocksQuant loads vision_tower.blocks.<i>.* for i=0,1,2,… until a block's attn.qkv AND
// attn.q_proj are both missing (the same counting-probe shape buildMoE uses for experts.<e>.*). The
// fused-vs-separate attention convention and the SwiGLU-vs-GELU MLP convention are each decided ONCE at
// block 0 (resolveVisionAttnGeometry / loadBlockMLP's own gate_proj probe) and every later block is loaded
// assuming the SAME one — a block whose tensors don't match fails loudly (loadBlockQKV's missing-key error,
// or the FF/gelu uniformity check below), catching a checkpoint that mixes conventions across blocks.
func buildVisionBlocksQuant(tensors map[string]safetensors.Tensor, hidden int, vc *visionConfig, quant *model.QuantConfig, zeroCopy bool) (blocks []visionBlock, headDim, numHeads, numKVHeads int, alias bool, err error) {
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
				return nil, 0, 0, 0, false, err
			}
		}
		q, k, v, qkvAlias, err := loadBlockQKV(tensors, bp, hidden, fused, quant, zeroCopy)
		if err != nil {
			return nil, 0, 0, 0, false, core.E("composed.buildVisionTower", core.Sprintf("block %d", i), err)
		}

		oName, oT, ok := weightAnyName(tensors, bp+"attn.o_proj.weight", bp+"attn.proj.weight")
		if !ok {
			return nil, 0, 0, 0, false, core.NewError(core.Sprintf("composed.buildVisionTower: block %d missing attn.o_proj/attn.proj weight", i))
		}
		if len(oT.Shape) != 2 {
			return nil, 0, 0, 0, false, core.NewError(core.Sprintf("composed.buildVisionTower: block %d attn output proj is not 2-D", i))
		}
		// inDim is q.Out (the concatenated multi-head width numHeads·HeadDim, already known from
		// loadBlockQKV's split — identical in EITHER convention), NOT oT.Shape[1]: see visionProjInDim's
		// doc comment for why a packed tensor's own Shape[1] can't be trusted as the logical width.
		o, oAlias, err := visionProj(tensors, oName, oT, oT.Shape[0], q.Out, quant, zeroCopy)
		if err != nil {
			return nil, 0, 0, 0, false, core.E("composed.buildVisionTower", core.Sprintf("block %d attn output proj", i), err)
		}
		o.B = optionalVisionVec(tensors, biasSibling(oName))

		mlpW, blockFF, mlpGELU, mlpAlias, err := loadBlockMLP(tensors, bp, hidden, quant, zeroCopy)
		if err != nil {
			return nil, 0, 0, 0, false, core.E("composed.buildVisionTower", core.Sprintf("block %d mlp", i), err)
		}
		if i == 0 {
			ff, gelu = blockFF, mlpGELU
		} else if blockFF != ff || mlpGELU != gelu {
			return nil, 0, 0, 0, false, core.NewError(core.Sprintf("composed.buildVisionTower: block %d MLP shape (FF %d, gelu %v) != block 0's (FF %d, gelu %v)", i, blockFF, mlpGELU, ff, gelu))
		}

		norm1W := optionalVisionVec(tensors, bp+"norm1.weight")
		norm2W := optionalVisionVec(tensors, bp+"norm2.weight")
		if norm1W == nil || norm2W == nil {
			return nil, 0, 0, 0, false, core.NewError(core.Sprintf("composed.buildVisionTower: block %d missing norm1/norm2 weight", i))
		}

		blocks = append(blocks, visionBlock{
			Norm1W: norm1W, Norm1B: optionalVisionVec(tensors, bp+"norm1.bias"),
			Norm2W: norm2W, Norm2B: optionalVisionVec(tensors, bp+"norm2.bias"),
			Attn: visionAttnWeights{Q: q, K: k, V: v, O: o, QNorm: qNorm, KNorm: kNorm},
			MLP:  mlpW,
		})
		alias = alias || qkvAlias || oAlias || mlpAlias
	}
	if len(blocks) == 0 {
		return nil, 0, 0, 0, false, core.NewError("composed.buildVisionTower: vision_tower.patch_embed present but no vision_tower.blocks.0.* found")
	}
	return blocks, headDim, numHeads, numKVHeads, alias, nil
}

// buildVisionMergerQuant loads the vision-to-text merger/projector under either alias family — the GUESSED
// layout's multi_modal_projector.* (probing .mlp.0/.mlp.2 as further aliases) or the REAL layout's
// vision_tower.merger.* — resolving the two big linears through visionProj (packed on a quantised
// checkpoint, else host f32; see the file doc comment). The spatial merge size is DERIVED from linear_1's
// own width as a multiple of hidden (must be a perfect square — mergeSize·mergeSize patches concatenate
// into linear_1's input) rather than trusted from config; when vision_config.spatial_merge_size is also
// present it is cross-validated against the derived value (a mismatch fails loudly). linear_2's output width
// is cross-validated against textHidden the same way.
func buildVisionMergerQuant(tensors map[string]safetensors.Tensor, hidden, textHidden int, vc *visionConfig, quant *model.QuantConfig, zeroCopy bool) (merger *visionMerger, mergeSize int, alias bool, err error) {
	l1Name, l1T, ok := weightAnyName(tensors,
		"multi_modal_projector.linear_1.weight", "multi_modal_projector.mlp.0.weight",
		"vision_tower.merger.linear_fc1.weight")
	if !ok {
		return nil, 0, false, core.NewError("composed.buildVisionTower: vision_tower present but multi_modal_projector.linear_1(.mlp.0)/vision_tower.merger.linear_fc1 weight is missing")
	}
	if len(l1T.Shape) != 2 {
		return nil, 0, false, core.NewError("composed.buildVisionTower: merger linear_1 is not 2-D")
	}
	// l1In (unlike l1Out) can't be read off l1T.Shape[1] directly: mergeSize is DERIVED from it below, and
	// a packed tensor's own Shape[1] is the compressed word count, not the logical width — the same fact
	// visionProjInDim exists for patch_embed to handle, reused here since the merger has no independently
	// known "previous layer" width to fall back on the way q/k/v/o do (hidden, q.Out).
	l1Out := l1T.Shape[0]
	l1In, err := visionProjInDim(tensors, l1Name, l1T, quant)
	if err != nil {
		return nil, 0, false, core.E("composed.buildVisionTower", "merger linear_1", err)
	}
	if hidden <= 0 || l1In%hidden != 0 {
		return nil, 0, false, core.NewError(core.Sprintf("composed.buildVisionTower: merger linear_1 input %d is not a multiple of the tower hidden size %d", l1In, hidden))
	}
	mergeSq := l1In / hidden
	mergeSize = isqrt(mergeSq)
	if mergeSize <= 0 || mergeSize*mergeSize != mergeSq {
		return nil, 0, false, core.NewError(core.Sprintf("composed.buildVisionTower: merger linear_1 input %d is not hidden(%d)·mergeSize² for an integer mergeSize", l1In, hidden))
	}
	if vc != nil && vc.SpatialMergeSize > 0 && vc.SpatialMergeSize != mergeSize {
		return nil, 0, false, core.NewError(core.Sprintf("composed.buildVisionTower: derived merge size %d != config spatial_merge_size %d", mergeSize, vc.SpatialMergeSize))
	}
	l1Lin, l1Alias, err := visionProj(tensors, l1Name, l1T, l1Out, l1In, quant, zeroCopy)
	if err != nil {
		return nil, 0, false, core.E("composed.buildVisionTower", "merger linear_1", err)
	}
	l1Lin.B = optionalVisionVec(tensors, biasSibling(l1Name))

	l2Name, l2T, ok := weightAnyName(tensors,
		"multi_modal_projector.linear_2.weight", "multi_modal_projector.mlp.2.weight",
		"vision_tower.merger.linear_fc2.weight")
	if !ok {
		return nil, 0, false, core.NewError("composed.buildVisionTower: vision_tower present but multi_modal_projector.linear_2(.mlp.2)/vision_tower.merger.linear_fc2 weight is missing")
	}
	if len(l2T.Shape) != 2 {
		return nil, 0, false, core.NewError("composed.buildVisionTower: merger linear_2 is not 2-D")
	}
	if textHidden > 0 && l2T.Shape[0] != textHidden {
		return nil, 0, false, core.NewError(core.Sprintf("composed.buildVisionTower: merger output width %d != text model hidden size %d (config/checkpoint mismatch)", l2T.Shape[0], textHidden))
	}
	// inDim is l1Out (linear_2 chains directly off linear_1's output — rows, never packed, so ALWAYS
	// trustworthy), NOT l2T.Shape[1]: see visionProjInDim's doc comment for why a packed tensor's own
	// Shape[1] can't be trusted as the logical width.
	l2Lin, l2Alias, err := visionProj(tensors, l2Name, l2T, l2T.Shape[0], l1Out, quant, zeroCopy)
	if err != nil {
		return nil, 0, false, core.E("composed.buildVisionTower", "merger linear_2", err)
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
		return nil, 0, false, core.NewError("composed.buildVisionTower: merger pre-norm weight (norm.weight / ln_q.weight) is missing")
	}

	return &visionMerger{
		NormW: normW, NormB: normB,
		L1: l1Lin, L2: l2Lin,
	}, mergeSize, l1Alias || l2Alias, nil
}
