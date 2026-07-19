// SPDX-Licence-Identifier: EUPL-1.2

package deepseekvl2

import (
	"math"

	core "dappco.re/go"
)

// vision_clip.go is deepencoder.VitModel ported host-side: CLIPVisionEmbeddings (prepend the
// learned CLS token, add the fixed absolute position table — patch_embeds comes in EXTERNALLY,
// from SAM's own tower output, never CLIP's own conv; see vision.go's doc comment) → one
// pre-transformer LayerNorm → clipNumLayers pre-norm bidirectional blocks (combined-QKV
// self-attention, quick-GELU 2-layer MLP) → the raw stack output (no final norm — see
// weights_clip.go's doc comment). Like vision_sam.go's block/attention functions, dim/numHeads
// are derived from the weights/an explicit parameter rather than the package's clipHidden/
// clipNumHeads constants, so the same code serves both the real tower and
// vision_clip_test.go's toy-scale goldens.

// clipEmbeddings assembles the [numPositions,dim] token sequence from an externally-supplied flat
// patch-token grid (SAM's [16,16,1024] output at real scale — 256 tokens — or a toy golden's
// synthetic patch_embeds): CLS embedding first, then the patch tokens, then the FIXED position
// table added directly (no interpolation — see weights_clip.go's doc comment: this checkpoint's
// pretrained 16×16 grid always matches SAM's output grid exactly for the v1 "Base" resolution
// mode, see ocr.go). dim/numPositions are derived from w's own tensor shapes.
func clipEmbeddings(patchEmbeds []float32, w CLIPWeights) ([]float32, error) {
	dim := len(w.ClassEmbedding)
	if dim == 0 || len(w.PositionEmbedding)%dim != 0 {
		return nil, core.NewError("deepseekvl2.clipEmbeddings: empty or malformed CLIP weights (class embedding/position table)")
	}
	numPositions := len(w.PositionEmbedding) / dim
	patches := len(patchEmbeds) / dim
	if patches*dim != len(patchEmbeds) {
		return nil, core.NewError("deepseekvl2.clipEmbeddings: patch embed buffer is not a whole number of tokens")
	}
	if patches+1 != numPositions {
		return nil, core.NewError(core.Sprintf("deepseekvl2.clipEmbeddings: got %d patch tokens (+1 CLS = %d positions), want %d — the general resize/interpolated-position-table path is not implemented in this v1 lane", patches, patches+1, numPositions))
	}
	out := make([]float32, numPositions*dim)
	copy(out[0:dim], w.ClassEmbedding)
	copy(out[dim:], patchEmbeds)
	return addRows(out, w.PositionEmbedding), nil
}

// clipAttentionForward runs one block's bidirectional multi-head self-attention: a single
// combined QKV projection (deepencoder.NoTPAttention.qkv_proj) split into heads, standard
// 1/√headDim-scaled softmax attention (no relative-position bias — that's a SAM-only mechanism),
// out_proj.
func clipAttentionForward(x []float32, numHeads int, b CLIPBlockWeights) []float32 {
	dim := len(b.OutBias)
	headDim := dim / numHeads
	t := len(x) / dim
	qkv := linear(x, b.QKVWeight, dim, 3*dim, b.QKVBias)
	q := make([]float32, t*dim)
	k := make([]float32, t*dim)
	v := make([]float32, t*dim)
	for i := range t {
		row := qkv[i*3*dim : (i+1)*3*dim]
		copy(q[i*dim:(i+1)*dim], row[0:dim])
		copy(k[i*dim:(i+1)*dim], row[dim:2*dim])
		copy(v[i*dim:(i+1)*dim], row[2*dim:3*dim])
	}

	scale := 1.0 / math.Sqrt(float64(headDim))
	out := make([]float32, t*dim)
	// Parallelise over the flattened (head, query-token) space — scores allocated per-unit, never
	// shared across goroutines (mathops.go's parallelFor doc comment).
	parallelFor(numHeads*t, func(unit int) {
		h, qi := unit/t, unit%t
		off := h * headDim
		scores := make([]float64, t)
		qVec := q[qi*dim+off : qi*dim+off+headDim]
		var maxScore = math.Inf(-1)
		for ki := range t {
			kVec := k[ki*dim+off : ki*dim+off+headDim]
			var dot float64
			for d := range headDim {
				dot += float64(qVec[d]) * float64(kVec[d])
			}
			sc := dot * scale
			scores[ki] = sc
			if sc > maxScore {
				maxScore = sc
			}
		}
		var sum float64
		for ki := range t {
			e := math.Exp(scores[ki] - maxScore)
			scores[ki] = e
			sum += e
		}
		oVec := out[qi*dim+off : qi*dim+off+headDim]
		for d := range headDim {
			var acc float64
			for ki := range t {
				acc += scores[ki] * float64(v[ki*dim+off+d])
			}
			oVec[d] = float32(acc / sum)
		}
	})
	return linear(out, b.OutWeight, dim, dim, b.OutBias)
}

// clipBlockForward runs one pre-norm block: LayerNorm → attention → residual, LayerNorm →
// quick-GELU MLP → residual (deepencoder.NoTPTransformerBlock.forward). dim/ffnHidden are derived
// from the weights, numHeads is an explicit parameter.
func clipBlockForward(x []float32, numHeads int, b CLIPBlockWeights) []float32 {
	dim := len(b.Norm1W)
	ffnHidden := len(b.FC1Bias)
	normed := layerNormBias(x, b.Norm1W, b.Norm1B, dim, clipLayerNormEps)
	hidden := addRows(x, clipAttentionForward(normed, numHeads, b))
	normed2 := layerNormBias(hidden, b.Norm2W, b.Norm2B, dim, clipLayerNormEps)
	mlp := linear(quickGELURow(linear(normed2, b.FC1Weight, dim, ffnHidden, b.FC1Bias)), b.FC2Weight, ffnHidden, dim, b.FC2Bias)
	return addRows(hidden, mlp)
}

// CLIPForward runs the whole CLIP-L tower given SAM's [16,16,1024] flat patch grid (256 tokens,
// patch_embeds — never CLIP's own conv, see vision.go's doc comment), returning the raw
// [257,1024] stack output (CLS row included at index 0 — VisionForward drops it, matching the
// reference's clip_out[:,1:] slice).
func CLIPForward(patchEmbeds []float32, w CLIPWeights) ([]float32, error) {
	hidden, err := clipEmbeddings(patchEmbeds, w)
	if err != nil {
		return nil, err
	}
	dim := len(w.ClassEmbedding)
	hidden = layerNormBias(hidden, w.PreLNWeight, w.PreLNBias, dim, clipLayerNormEps)
	for _, b := range w.Blocks {
		hidden = clipBlockForward(hidden, clipNumHeads, b)
	}
	return hidden, nil
}
