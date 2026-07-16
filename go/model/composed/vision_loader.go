// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
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
// Expected tensor names (wrapper root, sibling to the language_model.* prefix loadComposed resolves):
//
//	vision_tower.patch_embed.weight                    [Hidden, InChannels·TemporalPatchSize·PatchSize²]
//	vision_tower.patch_embed.bias                       [Hidden]                                    (optional)
//	vision_tower.blocks.<i>.norm1.weight / .bias         [Hidden]
//	vision_tower.blocks.<i>.norm2.weight / .bias         [Hidden]
//	vision_tower.blocks.<i>.attn.q_proj.weight / .bias   [NumHeads·HeadDim,   Hidden]  (.bias optional)
//	vision_tower.blocks.<i>.attn.k_proj.weight / .bias   [NumKVHeads·HeadDim, Hidden]  (.bias optional)
//	vision_tower.blocks.<i>.attn.v_proj.weight / .bias   [NumKVHeads·HeadDim, Hidden]  (.bias optional)
//	vision_tower.blocks.<i>.attn.o_proj.weight / .bias   [Hidden, NumHeads·HeadDim]    (.bias optional)
//	vision_tower.blocks.<i>.attn.q_norm.weight           [HeadDim]                     (optional)
//	vision_tower.blocks.<i>.attn.k_norm.weight           [HeadDim]                     (optional)
//	vision_tower.blocks.<i>.mlp.gate_proj.weight / .bias [FF, Hidden]                  (.bias optional)
//	vision_tower.blocks.<i>.mlp.up_proj.weight / .bias   [FF, Hidden]                  (.bias optional)
//	vision_tower.blocks.<i>.mlp.down_proj.weight / .bias [Hidden, FF]                  (.bias optional)
//	multi_modal_projector.norm.weight / .bias            [Hidden]           (probes .ln_q.* as an alias)
//	multi_modal_projector.linear_1.weight / .bias        [Hidden·M², Hidden·M²] (probes .mlp.0.* as an alias)
//	multi_modal_projector.linear_2.weight / .bias         [TextHidden, Hidden·M²] (probes .mlp.2.* as an alias)
//
// A checkpoint carrying none of these (no vision_tower.patch_embed(.proj).weight) returns (nil, nil) — the
// text-only path, unchanged from before this file existed.

// optionalVisionVec returns tensors[name] widened to f32, or nil when absent/unwidenable — the bias/
// qk-norm probe this file uses throughout, mirroring loader.go's f32opt for the text stack.
func optionalVisionVec(tensors map[string]safetensors.Tensor, name string) []float32 {
	if t, ok := tensors[name]; ok {
		if v, err := tensorF32(t); err == nil {
			return v
		}
	}
	return nil
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

// buildVisionTower probes tensors for a Qwen-VL-family vision tower + merger and, when found, assembles a
// *visionTower with every geometric field DERIVED from the tensors' own shapes (see the file doc comment).
// Returns (nil, nil) — not an error — when vision_tower.patch_embed(.proj).weight is absent: the checkpoint
// is text-only and loadComposed's caller leaves ComposedModel.Vision nil, loading exactly as before this
// file existed. vc is the checkpoint's parsed vision_config (nil when the config carries none — the loader
// still tries the tensor probe, since a config's vision_config presence and a checkpoint's actual tensors
// are, in principle, independent facts); textHidden is the text model's own D, cross-checked against the
// merger's output width so a config/checkpoint mismatch fails loudly rather than mis-loading.
func buildVisionTower(tensors map[string]safetensors.Tensor, vc *visionConfig, textHidden int) (*visionTower, error) {
	patchT, ok := model.WeightAny(tensors, "vision_tower.patch_embed.weight", "vision_tower.patch_embed.proj.weight")
	if !ok {
		return nil, nil // text-only checkpoint
	}
	if len(patchT.Shape) != 2 {
		return nil, core.NewError("composed.buildVisionTower: patch_embed weight is not 2-D")
	}
	hidden, patchDim := patchT.Shape[0], patchT.Shape[1]
	patchW, err := tensorF32(patchT)
	if err != nil {
		return nil, core.E("composed.buildVisionTower", "patch_embed weight", err)
	}
	patchB := optionalVisionVec(tensors, "vision_tower.patch_embed.bias")

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
		return nil, core.NewError("composed.buildVisionTower: vision_tower present but vision_config.patch_size is missing")
	}
	perFrame := inChannels * patchSize * patchSize
	if perFrame <= 0 || patchDim%perFrame != 0 {
		return nil, core.NewError(core.Sprintf("composed.buildVisionTower: patch_embed input width %d is not a multiple of in_channels·patch_size² %d", patchDim, perFrame))
	}
	temporal := patchDim / perFrame

	blocks, headDim, numHeads, numKVHeads, err := buildVisionBlocks(tensors, hidden, vc)
	if err != nil {
		return nil, err
	}

	merger, mergeSize, err := buildVisionMerger(tensors, hidden, textHidden, vc)
	if err != nil {
		return nil, err
	}

	return &visionTower{
		Patch:  visionLinear{W: patchW, B: patchB, Out: hidden, In: patchDim},
		Blocks: blocks,
		Merger: *merger,
		Cfg: visionTowerCfg{
			Hidden: hidden, PatchDim: patchDim,
			NumHeads: numHeads, NumKVHeads: numKVHeads, HeadDim: headDim,
			PatchSize: patchSize, InChannels: inChannels, TemporalPatchSize: temporal,
			MergeSize: mergeSize, TextHidden: textHidden,
			RopeTheta: ropeTheta, Eps: eps,
			// FF is carried on the MLP linears themselves (Gate.Out/Up.Out), not on visionTowerCfg — the
			// text stack's own MLP type follows the same shape (FF lives on the *MLP, not passed alongside).
		},
	}, nil
}

// buildVisionBlocks loads vision_tower.blocks.<i>.* for i=0,1,2,… until a block's q_proj is missing (the
// same counting-probe shape buildMoE uses for experts.<e>.*). HeadDim/NumHeads/NumKVHeads are derived ONCE
// from block 0 and assumed uniform across layers (the text stack makes the same per-model, not per-layer,
// geometry assumption). HeadDim prefers the block's own q_norm width (unambiguous — a per-head RMSNorm
// scale is exactly HeadDim wide); only when no q_norm tensor exists does it fall back to
// vision_config.num_heads splitting q_proj's output rows, mirroring buildAttn's identical fallback for the
// text attention mixer.
func buildVisionBlocks(tensors map[string]safetensors.Tensor, hidden int, vc *visionConfig) (blocks []visionBlock, headDim, numHeads, numKVHeads int, err error) {
	var ff int // the first block's MLP width — every later block is checked against it (uniform MLP width)
	for i := 0; ; i++ {
		bp := core.Sprintf("vision_tower.blocks.%d.", i)
		qT, ok := tensors[bp+"attn.q_proj.weight"]
		if !ok {
			break
		}
		if len(qT.Shape) != 2 {
			return nil, 0, 0, 0, core.NewError(core.Sprintf("composed.buildVisionTower: block %d q_proj is not 2-D", i))
		}
		kT, ok := tensors[bp+"attn.k_proj.weight"]
		if !ok {
			return nil, 0, 0, 0, core.NewError(core.Sprintf("composed.buildVisionTower: block %d missing attn.k_proj.weight", i))
		}
		vT, ok := tensors[bp+"attn.v_proj.weight"]
		if !ok {
			return nil, 0, 0, 0, core.NewError(core.Sprintf("composed.buildVisionTower: block %d missing attn.v_proj.weight", i))
		}
		oT, ok := tensors[bp+"attn.o_proj.weight"]
		if !ok {
			return nil, 0, 0, 0, core.NewError(core.Sprintf("composed.buildVisionTower: block %d missing attn.o_proj.weight", i))
		}
		qNorm := optionalVisionVec(tensors, bp+"attn.q_norm.weight")
		kNorm := optionalVisionVec(tensors, bp+"attn.k_norm.weight")

		if i == 0 {
			switch {
			case len(qNorm) > 0:
				headDim = len(qNorm)
			case vc != nil && vc.NumHeads > 0:
				headDim = qT.Shape[0] / vc.NumHeads
			}
			if headDim <= 0 || qT.Shape[0]%headDim != 0 {
				return nil, 0, 0, 0, core.NewError("composed.buildVisionTower: cannot derive attention head_dim (no q_norm tensor and no usable vision_config.num_heads)")
			}
			numHeads = qT.Shape[0] / headDim
			if kT.Shape[0]%headDim != 0 {
				return nil, 0, 0, 0, core.NewError(core.Sprintf("composed.buildVisionTower: k_proj rows %d not a multiple of derived head_dim %d", kT.Shape[0], headDim))
			}
			numKVHeads = kT.Shape[0] / headDim
			if vc != nil && vc.NumKeyValueHeads > 0 && vc.NumKeyValueHeads != numKVHeads {
				return nil, 0, 0, 0, core.NewError(core.Sprintf("composed.buildVisionTower: derived kv heads %d != config num_key_value_heads %d", numKVHeads, vc.NumKeyValueHeads))
			}
		}

		qW, err := tensorF32(qT)
		if err != nil {
			return nil, 0, 0, 0, core.E("composed.buildVisionTower", core.Sprintf("block %d q_proj", i), err)
		}
		kW, err := tensorF32(kT)
		if err != nil {
			return nil, 0, 0, 0, core.E("composed.buildVisionTower", core.Sprintf("block %d k_proj", i), err)
		}
		vW, err := tensorF32(vT)
		if err != nil {
			return nil, 0, 0, 0, core.E("composed.buildVisionTower", core.Sprintf("block %d v_proj", i), err)
		}
		oW, err := tensorF32(oT)
		if err != nil {
			return nil, 0, 0, 0, core.E("composed.buildVisionTower", core.Sprintf("block %d o_proj", i), err)
		}

		gateT, ok := tensors[bp+"mlp.gate_proj.weight"]
		if !ok {
			return nil, 0, 0, 0, core.NewError(core.Sprintf("composed.buildVisionTower: block %d missing mlp.gate_proj.weight", i))
		}
		upT, ok := tensors[bp+"mlp.up_proj.weight"]
		if !ok {
			return nil, 0, 0, 0, core.NewError(core.Sprintf("composed.buildVisionTower: block %d missing mlp.up_proj.weight", i))
		}
		downT, ok := tensors[bp+"mlp.down_proj.weight"]
		if !ok {
			return nil, 0, 0, 0, core.NewError(core.Sprintf("composed.buildVisionTower: block %d missing mlp.down_proj.weight", i))
		}
		if len(gateT.Shape) != 2 {
			return nil, 0, 0, 0, core.NewError(core.Sprintf("composed.buildVisionTower: block %d mlp.gate_proj is not 2-D", i))
		}
		blockFF := gateT.Shape[0]
		if i == 0 {
			ff = blockFF
		} else if blockFF != ff {
			return nil, 0, 0, 0, core.NewError(core.Sprintf("composed.buildVisionTower: block %d FF width %d != block 0's %d (non-uniform MLP width is not supported)", i, blockFF, ff))
		}
		gateW, err := tensorF32(gateT)
		if err != nil {
			return nil, 0, 0, 0, core.E("composed.buildVisionTower", core.Sprintf("block %d mlp.gate_proj", i), err)
		}
		upW, err := tensorF32(upT)
		if err != nil {
			return nil, 0, 0, 0, core.E("composed.buildVisionTower", core.Sprintf("block %d mlp.up_proj", i), err)
		}
		downW, err := tensorF32(downT)
		if err != nil {
			return nil, 0, 0, 0, core.E("composed.buildVisionTower", core.Sprintf("block %d mlp.down_proj", i), err)
		}

		norm1W := optionalVisionVec(tensors, bp+"norm1.weight")
		norm2W := optionalVisionVec(tensors, bp+"norm2.weight")
		if norm1W == nil || norm2W == nil {
			return nil, 0, 0, 0, core.NewError(core.Sprintf("composed.buildVisionTower: block %d missing norm1/norm2 weight", i))
		}

		blocks = append(blocks, visionBlock{
			Norm1W: norm1W, Norm1B: optionalVisionVec(tensors, bp+"norm1.bias"),
			Norm2W: norm2W, Norm2B: optionalVisionVec(tensors, bp+"norm2.bias"),
			Attn: visionAttnWeights{
				Q:     visionLinear{W: qW, B: optionalVisionVec(tensors, bp+"attn.q_proj.bias"), Out: qT.Shape[0], In: hidden},
				K:     visionLinear{W: kW, B: optionalVisionVec(tensors, bp+"attn.k_proj.bias"), Out: kT.Shape[0], In: hidden},
				V:     visionLinear{W: vW, B: optionalVisionVec(tensors, bp+"attn.v_proj.bias"), Out: vT.Shape[0], In: hidden},
				O:     visionLinear{W: oW, B: optionalVisionVec(tensors, bp+"attn.o_proj.bias"), Out: oT.Shape[0], In: oT.Shape[1]},
				QNorm: qNorm, KNorm: kNorm,
			},
			MLP: visionMLPWeights{
				Gate: visionLinear{W: gateW, B: optionalVisionVec(tensors, bp+"mlp.gate_proj.bias"), Out: blockFF, In: hidden},
				Up:   visionLinear{W: upW, B: optionalVisionVec(tensors, bp+"mlp.up_proj.bias"), Out: blockFF, In: hidden},
				Down: visionLinear{W: downW, B: optionalVisionVec(tensors, bp+"mlp.down_proj.bias"), Out: hidden, In: blockFF},
			},
		})
	}
	if len(blocks) == 0 {
		return nil, 0, 0, 0, core.NewError("composed.buildVisionTower: vision_tower.patch_embed present but no vision_tower.blocks.0.* found")
	}
	return blocks, headDim, numHeads, numKVHeads, nil
}

// buildVisionMerger loads the vision-to-text merger/projector. The spatial merge size is DERIVED from
// linear_1's own width as a multiple of hidden (must be a perfect square — mergeSize·mergeSize patches
// concatenate into linear_1's input) rather than trusted from config; when vision_config.spatial_merge_size
// is also present it is cross-validated against the derived value (a mismatch fails loudly). linear_2's
// output width is cross-validated against textHidden the same way.
func buildVisionMerger(tensors map[string]safetensors.Tensor, hidden, textHidden int, vc *visionConfig) (*visionMerger, int, error) {
	l1T, ok := model.WeightAny(tensors, "multi_modal_projector.linear_1.weight", "multi_modal_projector.mlp.0.weight")
	if !ok {
		return nil, 0, core.NewError("composed.buildVisionTower: vision_tower present but multi_modal_projector.linear_1(.mlp.0).weight is missing")
	}
	if len(l1T.Shape) != 2 {
		return nil, 0, core.NewError("composed.buildVisionTower: merger linear_1 is not 2-D")
	}
	l1Out, l1In := l1T.Shape[0], l1T.Shape[1]
	if hidden <= 0 || l1In%hidden != 0 {
		return nil, 0, core.NewError(core.Sprintf("composed.buildVisionTower: merger linear_1 input %d is not a multiple of the tower hidden size %d", l1In, hidden))
	}
	mergeSq := l1In / hidden
	mergeSize := isqrt(mergeSq)
	if mergeSize <= 0 || mergeSize*mergeSize != mergeSq {
		return nil, 0, core.NewError(core.Sprintf("composed.buildVisionTower: merger linear_1 input %d is not hidden(%d)·mergeSize² for an integer mergeSize", l1In, hidden))
	}
	if vc != nil && vc.SpatialMergeSize > 0 && vc.SpatialMergeSize != mergeSize {
		return nil, 0, core.NewError(core.Sprintf("composed.buildVisionTower: derived merge size %d != config spatial_merge_size %d", mergeSize, vc.SpatialMergeSize))
	}
	l1W, err := tensorF32(l1T)
	if err != nil {
		return nil, 0, core.E("composed.buildVisionTower", "merger linear_1", err)
	}

	l2T, ok := model.WeightAny(tensors, "multi_modal_projector.linear_2.weight", "multi_modal_projector.mlp.2.weight")
	if !ok {
		return nil, 0, core.NewError("composed.buildVisionTower: vision_tower present but multi_modal_projector.linear_2(.mlp.2).weight is missing")
	}
	if len(l2T.Shape) != 2 {
		return nil, 0, core.NewError("composed.buildVisionTower: merger linear_2 is not 2-D")
	}
	if textHidden > 0 && l2T.Shape[0] != textHidden {
		return nil, 0, core.NewError(core.Sprintf("composed.buildVisionTower: merger output width %d != text model hidden size %d (config/checkpoint mismatch)", l2T.Shape[0], textHidden))
	}
	l2W, err := tensorF32(l2T)
	if err != nil {
		return nil, 0, core.E("composed.buildVisionTower", "merger linear_2", err)
	}

	normW := optionalVisionVec(tensors, "multi_modal_projector.norm.weight")
	normB := optionalVisionVec(tensors, "multi_modal_projector.norm.bias")
	if normW == nil {
		normW = optionalVisionVec(tensors, "multi_modal_projector.ln_q.weight")
		normB = optionalVisionVec(tensors, "multi_modal_projector.ln_q.bias")
	}
	if normW == nil {
		return nil, 0, core.NewError("composed.buildVisionTower: merger pre-norm weight (norm.weight / ln_q.weight) is missing")
	}

	return &visionMerger{
		NormW: normW, NormB: normB,
		L1: visionLinear{W: l1W, B: optionalVisionVec(tensors, "multi_modal_projector.linear_1.bias"), Out: l1Out, In: l1In},
		L2: visionLinear{W: l2W, B: optionalVisionVec(tensors, "multi_modal_projector.linear_2.bias"), Out: l2T.Shape[0], In: l2T.Shape[1]},
	}, mergeSize, nil
}
