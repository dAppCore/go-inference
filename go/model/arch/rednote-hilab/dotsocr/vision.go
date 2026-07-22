// SPDX-Licence-Identifier: EUPL-1.2

package dotsocr

import (
	"math"

	core "dappco.re/go"
)

// vision.go is DotsVisionTransformer ported host-side: a linear patch embed (see weights.go's
// doc comment for why the real Conv2d folds into a dense projection) + RMSNorm → 2D-rotary
// full-attention transformer blocks (NOT causal — every patch attends to every other patch of
// the SAME image; this package only ever encodes one static image per call, so there is no
// cu_seqlens segmentation to reproduce: the reference's cu_seqlens boundary machinery exists for
// batching multiple images/frames into one sequence, which OCR's single-image-per-call contract
// never needs) → an optional post-trunk RMSNorm → PatchMerger (LayerNorm → Linear → GELU →
// Linear), which groups every spatial_merge_size² consecutive patches into one output token.
// Ported directly from modeling_dots_vision.py's DotsVisionTransformer.forward.

// mergerLNEps is the vision PatchMerger's ln_q epsilon — hard-coded 1e-6 in the reference
// (`LayerNorm(context_dim, eps=1e-6)`), independent of vision_config.rms_norm_eps (which governs
// every RMSNorm in the tower, including patch_embed's norm and each block's norm1/norm2).
const mergerLNEps = 1e-6

// visionRotaryTable computes the vision tower's 2D rotary cos/sin half-tables, one [freqDim]
// pair per patch, in the SAME (block_h, block_w, i_h, i_w) iteration order the reference's
// get_pos_ids_by_grid produces (confirmed against modeling_dots_vision.py: reshape/permute over
// (h/m, m, w/m, m) with the merge-size-block axes OUTSIDE the within-block axes) — this order
// must match the patchify order image.go's patchify produces (see its doc comment), since
// position n's rotary table entry is applied to pixel patch n. freqDim = headDim/4: the
// reference's VisionRotaryEmbedding(headDim/2) yields an invFreq of length (headDim/2)/2, and
// rot_pos_emb concatenates an h-half and a w-half of that length into the headDim/2 "cosHalf"
// applyRotaryHalf consumes (each covering ONE spatial axis — h occupies [0,freqDim), w occupies
// [freqDim,2*freqDim)=[freqDim,headDim/2)).
func visionRotaryTable(gridH, gridW, mergeSize, headDim int) (cosHalf, sinHalf [][]float32) {
	freqDim := headDim / 4
	invFreq := make([]float64, freqDim)
	for j := range freqDim {
		invFreq[j] = 1.0 / math.Pow(10000, float64(2*j)/float64(headDim/2))
	}
	n := gridH * gridW
	cosHalf = make([][]float32, n)
	sinHalf = make([][]float32, n)
	idx := 0
	for blockH := 0; blockH < gridH/mergeSize; blockH++ {
		for blockW := 0; blockW < gridW/mergeSize; blockW++ {
			for ih := range mergeSize {
				for iw := range mergeSize {
					hpos := blockH*mergeSize + ih
					wpos := blockW*mergeSize + iw
					half := 2 * freqDim
					c := make([]float32, half)
					s := make([]float32, half)
					for j := range freqDim {
						hAngle := float64(hpos) * invFreq[j]
						wAngle := float64(wpos) * invFreq[j]
						c[j], s[j] = float32(math.Cos(hAngle)), float32(math.Sin(hAngle))
						c[freqDim+j], s[freqDim+j] = float32(math.Cos(wAngle)), float32(math.Sin(wAngle))
					}
					cosHalf[idx], sinHalf[idx] = c, s
					idx++
				}
			}
		}
	}
	return cosHalf, sinHalf
}

// visionAttentionForward is one DotsVisionBlock's self-attention: project q/k/v from the fused
// weights, apply the 2D rotary embedding to q/k per head, full (non-causal, whole-image) scaled
// dot-product attention, out_proj. Scale is applied to the score matrix after the dot product
// (matches VisionAttention.forward's `matmul(q,k)/sqrt(head_dim)`, mathematically equivalent to
// whisper's pre-scaled-Q convention).
func visionAttentionForward(x []float32, n, d, heads int, w VisionAttnWeights, cosHalf, sinHalf [][]float32) ([]float32, error) {
	if d%heads != 0 {
		return nil, core.NewError("dotsocr.visionAttentionForward: embed_dim not divisible by num_attention_heads")
	}
	headDim := d / heads
	q := linearForward(x, w.Q, n)
	k := linearForward(x, w.K, n)
	v := linearForward(x, w.V, n)
	for i := range n {
		for h := range heads {
			off := i*d + h*headDim
			applyRotaryHalf(q[off:off+headDim], cosHalf[i], sinHalf[i])
			applyRotaryHalf(k[off:off+headDim], cosHalf[i], sinHalf[i])
		}
	}

	scale := 1.0 / math.Sqrt(float64(headDim))
	out := make([]float32, n*d)
	scores := make([]float64, n)
	for h := range heads {
		off := h * headDim
		for i := range n {
			qi := q[i*d+off : i*d+off+headDim]
			maxScore := math.Inf(-1)
			for j := range n {
				kj := k[j*d+off : j*d+off+headDim]
				var dot float64
				for c := range headDim {
					dot += float64(qi[c]) * float64(kj[c])
				}
				dot *= scale
				scores[j] = dot
				if dot > maxScore {
					maxScore = dot
				}
			}
			var sum float64
			for j := range n {
				e := math.Exp(scores[j] - maxScore)
				scores[j] = e
				sum += e
			}
			oi := out[i*d+off : i*d+off+headDim]
			for c := range headDim {
				var acc float64
				for j := range n {
					acc += scores[j] * float64(v[j*d+off+c])
				}
				oi[c] = float32(acc / sum)
			}
		}
	}
	return linearForward(out, w.Proj, n), nil
}

// visionBlockForward is one DotsVisionBlock: pre-norm attention, pre-norm SwiGLU FFN, both
// residual — mirrors DotsVisionBlock.forward exactly.
func visionBlockForward(x []float32, n, d, heads int, blk VisionBlockWeights, cosHalf, sinHalf [][]float32, eps float32) ([]float32, error) {
	normed := rmsNormForward(x, blk.Norm1, n, d, eps)
	attnOut, err := visionAttentionForward(normed, n, d, heads, blk.Attn, cosHalf, sinHalf)
	if err != nil {
		return nil, err
	}
	x = addRows(x, attnOut)

	normed = rmsNormForward(x, blk.Norm2, n, d, eps)
	ff := swiGLU(normed, blk.Gate, blk.Up, blk.Down, n)
	return addRows(x, ff), nil
}

// patchMerger is PatchMerger.forward: LayerNorm (hard-coded eps, see mergerLNEps) then a
// GELU-gated 2-layer MLP over every spatial_merge_size²-consecutive-row group. The "view(-1,
// mergedDim)" reshape in the reference is a NO-OP on a row-major flat buffer — grouping
// groupSize=mergeSize² consecutive [d]-rows into one [groupSize*d]-row reinterprets the SAME
// bytes, so this reads normed as [numGroups, mergedDim] directly without any data movement,
// PROVIDED image.go's patchify already ordered patches so consecutive groupSize rows are one
// spatial block (see visionRotaryTable's doc comment — the same ordering requirement).
func patchMerger(x []float32, n, embedDim, mergeSize int, w *VisionWeights) ([]float32, error) {
	groupSize := mergeSize * mergeSize
	if n%groupSize != 0 {
		return nil, core.NewError(core.Sprintf("dotsocr.patchMerger: %d patches is not a multiple of merge_size²=%d", n, groupSize))
	}
	normed := layerNormForward(x, w.MergerLNQ, n, embedDim, mergerLNEps)
	numGroups := n / groupSize
	hidden := linearForward(normed, w.MergerFC1, numGroups)
	hidden = geluRow(hidden)
	return linearForward(hidden, w.MergerFC2, numGroups), nil
}

// EncodeImage runs the full vision tower over one image's already-patchified pixel values
// (pixelValues is [nPatches, patchDim] flat, patchDim = NumChannels*TemporalPatchSize*
// PatchSize²  — see image.go's Patchify) and its patch grid [gridT=1, gridH, gridW], returning
// the merged vision embeddings [ (gridH/merge)·(gridW/merge), VisionConfig.HiddenSize ] ready to
// scatter into the text decoder's input embeddings at the image-token positions (ocr.go).
// gridT must be 1: this package only supports one static image per call (OCR's own contract;
// video/multi-frame temporal batching is out of scope, see visionRotaryTable's doc comment).
func EncodeImage(pixelValues []float32, gridT, gridH, gridW int, w *Weights, cfg *Config) ([]float32, error) {
	if w == nil || cfg == nil || cfg.VisionConfig == nil {
		return nil, core.NewError("dotsocr.EncodeImage: nil weights/config")
	}
	vc := cfg.VisionConfig
	if gridT != 1 {
		return nil, core.NewError(core.Sprintf("dotsocr.EncodeImage: grid_t=%d, only single-frame (grid_t=1) images are supported", gridT))
	}
	nPatches := gridT * gridH * gridW
	patchDim := vc.NumChannels * vc.TemporalPatchSize * vc.PatchSize * vc.PatchSize
	if len(pixelValues) != nPatches*patchDim {
		return nil, core.NewError(core.Sprintf("dotsocr.EncodeImage: pixel_values has %d elements, want %d (%d patches × %d)", len(pixelValues), nPatches*patchDim, nPatches, patchDim))
	}
	if gridH%vc.SpatialMergeSize != 0 || gridW%vc.SpatialMergeSize != 0 {
		return nil, core.NewError(core.Sprintf("dotsocr.EncodeImage: grid %d×%d is not a multiple of spatial_merge_size=%d", gridH, gridW, vc.SpatialMergeSize))
	}

	d := vc.EmbedDim
	hidden := linearForward(pixelValues, w.Vision.PatchEmbed, nPatches)
	hidden = rmsNormForward(hidden, w.Vision.PatchEmbedNorm, nPatches, d, vc.RMSNormEps)

	headDim := d / vc.NumAttentionHeads
	cosHalf, sinHalf := visionRotaryTable(gridH, gridW, vc.SpatialMergeSize, headDim)

	var err error
	for _, blk := range w.Vision.Blocks {
		hidden, err = visionBlockForward(hidden, nPatches, d, vc.NumAttentionHeads, blk, cosHalf, sinHalf, vc.RMSNormEps)
		if err != nil {
			return nil, err
		}
	}
	if vc.PostNorm {
		hidden = rmsNormForward(hidden, w.Vision.PostTrunkNorm, nPatches, d, vc.RMSNormEps)
	}
	return patchMerger(hidden, nPatches, d, vc.SpatialMergeSize, &w.Vision)
}
