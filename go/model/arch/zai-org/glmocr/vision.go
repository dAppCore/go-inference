// SPDX-Licence-Identifier: EUPL-1.2

package glmocr

import core "dappco.re/go"

// vision.go is GlmOcrVisionModel ported host-side: patch_embed (a Linear — see weights.go's
// doc comment for why the reference's Conv3d collapses to one) → 2D rotary table → depth
// pre-RMSNorm blocks (per-head-RMSNorm'd, roped, non-causal self-attention + SwiGLU MLP) →
// post_layernorm → a spatial_merge_size×spatial_merge_size downsample (Conv2d, likewise a
// Linear over a gathered block) → the GLU patch merger, producing one embedding per merged
// spatial_merge_size² block — the text decoder's image-token embeddings. Ported directly from
// modeling_glm_ocr.py's GlmOcrVisionModel.forward; see that file's read notes in rope.go/
// weights.go's doc comments for the exact line-by-line correspondence this package pins goldens
// against (testdata/block_goldens.json's "vision" section).

func visionAttnForward(x []float32, T, hidden, heads, headDim int, w VisionAttnWeights, cos, sin [][]float32, eps float32) []float32 {
	q := linearForward(x, w.Q, T)
	k := linearForward(x, w.K, T)
	v := linearForward(x, w.V, T)
	for t := range T {
		for h := range heads {
			off := t*hidden + h*headDim
			qh := rmsNormForward(q[off:off+headDim], w.QNorm, 1, headDim, eps)
			kh := rmsNormForward(k[off:off+headDim], w.KNorm, 1, headDim, eps)
			copy(q[off:off+headDim], applyRopeVision(qh, cos[t], sin[t]))
			copy(k[off:off+headDim], applyRopeVision(kh, cos[t], sin[t]))
		}
	}
	attnOut := mhaCore(q, k, v, T, heads, heads, headDim, false)
	return linearForward(attnOut, w.Proj, T)
}

func visionBlockForward(x []float32, T, hidden, ff, heads, headDim int, w VisionBlockWeights, cos, sin [][]float32, eps float32) []float32 {
	normed1 := rmsNormForward(x, w.Norm1, T, hidden, eps)
	attnOut := visionAttnForward(normed1, T, hidden, heads, headDim, w.Attn, cos, sin, eps)
	h1 := addVec(x, attnOut)

	normed2 := rmsNormForward(h1, w.Norm2, T, hidden, eps)
	mlpOut := swiGLUForward(normed2, w.MLP.Gate, w.MLP.Up, w.MLP.Down, T)
	return addVec(h1, mlpOut)
}

// downsampleForward is GlmOcrVisionModel's Conv2d(kernel=stride=merge) over consecutive
// merge² blocks of the (already merge-grouped — see visionPosIDs' doc comment) patch sequence,
// flattened to a Linear over a per-block gathered vector in the weight's [outChan,inChan,mh,mw]
// row-major order (inChan outermost, so channel c's four (mh,mw) values are NOT contiguous in
// the source sequence — this gather gathers them into contiguous per-channel runs the way the
// Conv2d weight expects).
func downsampleForward(hidden []float32, T, hiddenV, merge int, w LinearWeights) []float32 {
	blockSize := merge * merge
	numBlocks := T / blockSize
	flatDim := hiddenV * blockSize
	out := make([]float32, numBlocks*w.Out)
	gathered := make([]float32, flatDim)
	for b := range numBlocks {
		base := b * blockSize
		for ic := range hiddenV {
			for mh := range merge {
				for mw := range merge {
					gathered[ic*blockSize+mh*merge+mw] = hidden[(base+mh*merge+mw)*hiddenV+ic]
				}
			}
		}
		copy(out[b*w.Out:(b+1)*w.Out], linearForward(gathered, w, 1))
	}
	return out
}

func visionMergerForward(x []float32, T int, w VisionMergerWeights) []float32 {
	h1 := linearForward(x, w.Proj, T)
	h1 = layerNormForward(h1, w.PostProjectionNorm, T, w.Proj.Out)
	for i := range h1 {
		h1[i] = geluExact(h1[i])
	}
	return swiGLUForward(h1, w.Gate, w.Up, w.Down, T)
}

// VisionForward runs the full vision tower over one decoded image's patch grid, returning the
// merged image embeddings ([]float32, flat [numMerged, VisionConfig.OutHiddenSize]) and
// numMerged (the count the text decoder's image-token span must equal).
func VisionForward(patches *PatchGrid, w *VisionWeights, vc *VisionConfig) ([]float32, int, error) {
	if patches == nil || w == nil || vc == nil {
		return nil, 0, core.NewError("glmocr.VisionForward: nil patches/weights/config")
	}
	T := len(patches.Patches) / patches.PatchDim
	if T == 0 || T*patches.PatchDim != len(patches.Patches) {
		return nil, 0, core.NewError("glmocr.VisionForward: patch grid element count does not match PatchDim")
	}
	headDim := vc.HiddenSize / vc.NumHeads

	hidden := linearForward(patches.Patches, w.PatchEmbed, T)

	hpos, wpos := visionPosIDs(patches.GridT, patches.GridH, patches.GridW, vc.SpatialMergeSize)
	if len(hpos) != T {
		return nil, 0, core.NewError(core.Sprintf("glmocr.VisionForward: grid_thw (%d,%d,%d) implies %d patches, patch grid carries %d", patches.GridT, patches.GridH, patches.GridW, len(hpos), T))
	}
	cos, sin := visionCosSin(hpos, wpos, headDim, 10000.0)

	for _, blk := range w.Blocks {
		hidden = visionBlockForward(hidden, T, vc.HiddenSize, vc.IntermediateSize, vc.NumHeads, headDim, blk, cos, sin, vc.RMSNormEps)
	}
	hidden = rmsNormForward(hidden, w.PostLayernorm, T, vc.HiddenSize, vc.RMSNormEps)

	merge := vc.SpatialMergeSize
	if T%(merge*merge) != 0 {
		return nil, 0, core.NewError(core.Sprintf("glmocr.VisionForward: %d patches is not divisible by spatial_merge_size² (%d)", T, merge*merge))
	}
	downsampled := downsampleForward(hidden, T, vc.HiddenSize, merge, w.Downsample)
	numMerged := T / (merge * merge)
	merged := visionMergerForward(downsampled, numMerged, w.Merger)
	return merged, numMerged, nil
}
