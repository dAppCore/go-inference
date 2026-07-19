// SPDX-Licence-Identifier: EUPL-1.2

package glmocr

import "math"

// rope.go carries GLM-OCR's TWO distinct rotary position embeddings — the vision tower's 2D
// (height/width) split-half rope (apply_rotary_pos_emb_vision) and the text decoder's 3D
// (temporal/height/width) multimodal rope with GLM's interleaved-pair rotation
// (apply_rotary_pos_emb + apply_mrope) — ported directly from transformers'
// modeling_glm_ocr.py. The two are NOT the same math: vision rotates the two HALVES of a
// head's channels against each other (rotate_half); text rotates ADJACENT PAIRS
// (rotate_half_llm) — mixing them up produces coherent-but-wrong output, not a crash.

// visionPosIDs enumerates one image's patch sequence in the exact nested order GLM-OCR's real
// patch pipeline produces (image.go's DecodeAndPatchify walks this SAME order when it flattens
// pixels to patch vectors, so the two are guaranteed consistent by sharing this one function —
// see GlmOcrVisionModel.rot_pos_emb's hpos_ids/wpos_ids construction, which reshapes/permutes
// to this identical (t, blockRow, blockCol, mergeRow, mergeCol) nesting): for each temporal
// frame, each spatial_merge_size×spatial_merge_size block (row-major), each position within
// that block (row-major) — hpos/wpos are that patch's row/col in the FULL (pre-merge) patch
// grid. gridH/gridW must already be divisible by merge (image.go only ever calls this with
// dimensions it derived that way).
func visionPosIDs(gridT, gridH, gridW, merge int) (hpos, wpos []int) {
	gh, gw := gridH/merge, gridW/merge
	n := gridT * gh * gw * merge * merge
	hpos = make([]int, 0, n)
	wpos = make([]int, 0, n)
	for range gridT {
		for bh := range gh {
			for bw := range gw {
				for mh := range merge {
					for mw := range merge {
						hpos = append(hpos, bh*merge+mh)
						wpos = append(wpos, bw*merge+mw)
					}
				}
			}
		}
	}
	return hpos, wpos
}

// visionRotaryFreqs computes GlmOcrVisionRotaryEmbedding(headDim/2)'s frequency table: inv_freq
// has length headDim/4 (arange(0,headDim/2,2) — the module's OWN internal halving, on top of
// the headDim/2 it was constructed with), freqs[pos][i] = pos·inv_freq[i], for pos in
// [0,maxPos). theta is GlmOcrVisionRotaryEmbedding's hardcoded default (10000.0) — GLM-OCR
// never overrides it, and vision_config carries no rope_theta field to read.
func visionRotaryFreqs(maxPos, headDim int, theta float32) [][]float32 {
	moduleDim := headDim / 2
	n := moduleDim / 2
	invFreq := make([]float32, n)
	for i := range n {
		invFreq[i] = float32(1.0 / math.Pow(float64(theta), float64(2*i)/float64(moduleDim)))
	}
	table := make([][]float32, maxPos)
	for p := range maxPos {
		row := make([]float32, n)
		for i := range n {
			row[i] = float32(p) * invFreq[i]
		}
		table[p] = row
	}
	return table
}

// visionCosSin builds per-patch (cos,sin), each [headDim], from that patch's (hpos,wpos) —
// GlmOcrVisionModel.rot_pos_emb + GlmOcrVisionModel.forward's emb=cat(rotary,rotary): the
// quarter-width h-frequency and w-frequency rows are concatenated (h then w) and that
// half-width result duplicated to fill headDim, THEN cos/sin taken — see
// apply_rotary_pos_emb_vision, which applies these via the SPLIT-HALF rotate_half (not text's
// interleaved-pair rotation).
func visionCosSin(hpos, wpos []int, headDim int, theta float32) (cos, sin [][]float32) {
	maxPos := 0
	for i := range hpos {
		if hpos[i]+1 > maxPos {
			maxPos = hpos[i] + 1
		}
		if wpos[i]+1 > maxPos {
			maxPos = wpos[i] + 1
		}
	}
	table := visionRotaryFreqs(maxPos, headDim, theta)
	n := len(hpos)
	quarter := headDim / 4
	cos = make([][]float32, n)
	sin = make([][]float32, n)
	for i := range n {
		hf, wf := table[hpos[i]], table[wpos[i]]
		full := make([]float32, headDim)
		copy(full[0:quarter], hf)
		copy(full[quarter:2*quarter], wf)
		copy(full[2*quarter:3*quarter], hf)
		copy(full[3*quarter:4*quarter], wf)
		c := make([]float32, headDim)
		s := make([]float32, headDim)
		for j, v := range full {
			c[j] = float32(math.Cos(float64(v)))
			s[j] = float32(math.Sin(float64(v)))
		}
		cos[i], sin[i] = c, s
	}
	return cos, sin
}

// rotateHalf negates+swaps the two HALVES of x (the vision tower's rotation: pairs are
// (i, i+len(x)/2)) — distinct from rotateHalfLLM's adjacent-pair convention below.
func rotateHalf(x []float32) []float32 {
	half := len(x) / 2
	out := make([]float32, len(x))
	for i := range half {
		out[i] = -x[half+i]
		out[half+i] = x[i]
	}
	return out
}

// applyRopeVision rotates one head's channel vector x[headDim] by the vision tower's split-half
// convention: out = x·cos + rotateHalf(x)·sin.
func applyRopeVision(x, cos, sin []float32) []float32 {
	rh := rotateHalf(x)
	out := make([]float32, len(x))
	for i := range x {
		out[i] = x[i]*cos[i] + rh[i]*sin[i]
	}
	return out
}

// textRotaryFreqPos computes GlmOcrTextRotaryEmbedding's per-position frequency vector
// (length headDim/2, partial_rotary_factor is always 1.0 for GLM-OCR so this covers the whole
// head) for ONE token whose 3D position is (tPos,hPos,wPos): frequency index k picks its
// position id by which mrope_section band it falls in, round-robin temporal→height→width
// (apply_mrope's `chunk[i % 3]` — GLM-OCR's rope_parameters always carries exactly 3 sections,
// so i%3 == i and the mapping is simply section 0 → temporal, 1 → height, 2 → width). A plain
// (non-multimodal) text run passes tPos==hPos==wPos, collapsing this to ordinary 1D rope.
func textRotaryFreqPos(tPos, hPos, wPos, headDim int, theta float32, mropeSection []int) []float32 {
	half := headDim / 2
	freqs := make([]float32, half)
	band, bandEnd := 0, 0
	if len(mropeSection) > 0 {
		bandEnd = mropeSection[0]
	}
	for k := range half {
		for band < len(mropeSection)-1 && k >= bandEnd {
			band++
			bandEnd += mropeSection[band]
		}
		var pos int
		switch band % 3 {
		case 0:
			pos = tPos
		case 1:
			pos = hPos
		default:
			pos = wPos
		}
		invFreq := 1.0 / math.Pow(float64(theta), float64(2*k)/float64(headDim))
		freqs[k] = float32(float64(pos) * invFreq)
	}
	return freqs
}

// applyRopeTextPair rotates x[headDim] by GLM's interleaved-pair convention (rotate_half_llm):
// pair k is channels (2k,2k+1), rotated by angle freqs[k] — the algebraic simplification of
// apply_rotary_pos_emb's cos[...,:half].repeat_interleave(2) + rotate_half_llm (see the file
// doc comment): output[2k] = x[2k]·cos(f_k) − x[2k+1]·sin(f_k), output[2k+1] = x[2k+1]·cos(f_k)
// + x[2k]·sin(f_k).
func applyRopeTextPair(x, freqs []float32) []float32 {
	out := make([]float32, len(x))
	for k, f := range freqs {
		c := float32(math.Cos(float64(f)))
		s := float32(math.Sin(float64(f)))
		a, b := x[2*k], x[2*k+1]
		out[2*k] = a*c - b*s
		out[2*k+1] = b*c + a*s
	}
	return out
}
