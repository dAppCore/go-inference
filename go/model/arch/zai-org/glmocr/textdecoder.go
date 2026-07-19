// SPDX-Licence-Identifier: EUPL-1.2

package glmocr

import core "dappco.re/go"

// textdecoder.go is GlmOcrTextModel ported host-side: token embedding (image spans are
// overwritten with the vision tower's merged embeddings BEFORE this runs — see ocr.go) → per-
// position 3D mrope frequencies → NumHiddenLayers GLM-4-sandwich-norm decoder layers (causal
// GQA self-attention with GLM's interleaved-pair rope, then a SwiGLU MLP — see weights.go's
// TextLayerWeights doc comment for the exact residual/norm order) → a final RMSNorm. lm_head
// (a separate, untied projection — weights.go's doc comment) is applied by the caller only to
// the LAST row, since generation only ever needs the next token's logits. This package always
// recomputes the whole sequence on every decode step (no growing KV cache — the whisper
// template's optional perf slice; see decoder.go's DecodeLogitsStep there for the pattern a
// future speed pass would follow) — host-f32 correctness-first, per the design's own phasing.

func textAttnForward(x []float32, T, hidden, heads, kvHeads, headDim int, w TextAttnWeights, freqsPerPos [][]float32) []float32 {
	q := linearForward(x, w.Q, T)
	k := linearForward(x, w.K, T)
	v := linearForward(x, w.V, T)
	qDim, kvDim := heads*headDim, kvHeads*headDim
	for t := range T {
		freqs := freqsPerPos[t]
		for h := range heads {
			off := t*qDim + h*headDim
			copy(q[off:off+headDim], applyRopeTextPair(q[off:off+headDim], freqs))
		}
		for h := range kvHeads {
			off := t*kvDim + h*headDim
			copy(k[off:off+headDim], applyRopeTextPair(k[off:off+headDim], freqs))
		}
	}
	attnOut := mhaCore(q, k, v, T, heads, kvHeads, headDim, true)
	return linearForward(attnOut, w.O, T)
}

func textLayerForward(x []float32, T, hidden, ff, heads, kvHeads, headDim int, w TextLayerWeights, freqsPerPos [][]float32, eps float32) []float32 {
	residual := x
	normed := rmsNormForward(x, w.InputNorm, T, hidden, eps)
	attnOut := textAttnForward(normed, T, hidden, heads, kvHeads, headDim, w.Attn, freqsPerPos)
	attnOut = rmsNormForward(attnOut, w.PostSelfAttnNorm, T, hidden, eps)
	h1 := addVec(residual, attnOut)

	residual = h1
	normed = rmsNormForward(h1, w.PostAttnNorm, T, hidden, eps)
	mlpOut := swiGLUForward(normed, w.MLP.Gate, w.MLP.Up, w.MLP.Down, T)
	mlpOut = rmsNormForward(mlpOut, w.PostMLPNorm, T, hidden, eps)
	return addVec(residual, mlpOut)
}

// TextForward runs the full text decoder stack over hiddenIn ([]float32, flat [T,HiddenSize] —
// token embeddings with any image span already scattered in), returning the final-normed
// hidden states (same shape). tPos/hPos/wPos are the 3D mrope position ids for each of the T
// positions (prompt.PositionIDs' output).
func TextForward(hiddenIn []float32, T int, tc *TextConfig, w *TextWeights, tPos, hPos, wPos []int) ([]float32, error) {
	if tc == nil || w == nil {
		return nil, core.NewError("glmocr.TextForward: nil config/weights")
	}
	if tc.RopeParameters == nil || len(tc.RopeParameters.MropeSection) == 0 {
		return nil, core.NewError("glmocr.TextForward: text_config.rope_parameters.mrope_section is required")
	}
	if len(tPos) != T || len(hPos) != T || len(wPos) != T {
		return nil, core.NewError("glmocr.TextForward: position id slices must have length T")
	}
	headDim := tc.HeadDim
	theta := tc.RopeParameters.RopeTheta
	section := tc.RopeParameters.MropeSection

	freqsPerPos := make([][]float32, T)
	for t := range T {
		freqsPerPos[t] = textRotaryFreqPos(tPos[t], hPos[t], wPos[t], headDim, theta, section)
	}

	hidden := hiddenIn
	for _, layer := range w.Layers {
		hidden = textLayerForward(hidden, T, tc.HiddenSize, tc.IntermediateSize, tc.NumAttentionHeads, tc.NumKeyValueHeads, headDim, layer, freqsPerPos, tc.RMSNormEps)
	}
	return rmsNormForward(hidden, w.FinalNorm, T, tc.HiddenSize, tc.RMSNormEps), nil
}

// embedTokens looks up ids' rows in the [VocabSize,HiddenSize] embedding table.
func embedTokens(ids []int32, table []float32, hidden int) ([]float32, error) {
	out := make([]float32, len(ids)*hidden)
	vocab := len(table) / hidden
	for i, id := range ids {
		if int(id) < 0 || int(id) >= vocab {
			return nil, core.NewError(core.Sprintf("glmocr.embedTokens: token id %d at position %d is out of vocab range [0,%d)", id, i, vocab))
		}
		copy(out[i*hidden:(i+1)*hidden], table[int(id)*hidden:(int(id)+1)*hidden])
	}
	return out, nil
}

// argmax32 returns the index of the largest value in x (the greedy decode step) — ties resolve
// to the FIRST maximal index, matching torch.argmax's documented tie-break.
func argmax32(x []float32) int {
	best := 0
	for i := 1; i < len(x); i++ {
		if x[i] > x[best] {
			best = i
		}
	}
	return best
}
