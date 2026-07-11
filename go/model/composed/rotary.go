// SPDX-Licence-Identifier: EUPL-1.2

package composed

import "math"

// rotary.go encodes the Qwen 3.6 partial + interleaved mRoPE convention (rope_parameters:
// partial_rotary_factor 0.25, mrope_interleaved true, mrope_section [11,11,10]) and — the load-bearing
// fact for the text decode — PROVES it reduces to standard partial rotary for pure-text positions.
//
// The transformers qwen3_5 reference builds per-frequency angles from a 3D position (temporal T, height H,
// width W) and interleaves them across the rotary pairs (apply_interleaved_mrope): starting from the T
// angles, the stride-3 slices at offset 1 and offset 2 are overwritten with the H and W angles up to
// mrope_section[dim]*3 — the [T,H,W,T,H,W,…] interleave. apply_rotary_pos_emb then applies the standard
// rotate_half over the first rotary_dim = int(head_dim·partial_rotary_factor) dims, leaving the rest
// unrotated (partial rotary).
//
// For a pure-TEXT position all three position dims equal the text position p, so every T/H/W angle is
// p·inv_freq[i] and the interleave is a no-op: the angles collapse to the standard 1D partial-rotary
// angles. The composed attention forward therefore uses the reduced form (applyRotaryHalf, pinned by the
// attention goldens); the full interleave here is the convention for a future multimodal (3D-position) cut
// and the subject of the reduction property below. Full numeric verification against real checkpoint
// outputs is a checkpoint-smoke item (no weights here).

// rotaryInvFreq returns the inv_freq table for a partial-rotary block: inv_freq[i] = theta^(-2i/rotaryDim)
// for i in [0, rotaryDim/2). This is the exact frequency progression applyRotaryHalf computes inline.
func rotaryInvFreq(rotaryDim int, theta float64) []float64 {
	n := rotaryDim / 2
	out := make([]float64, n)
	for i := range out {
		out[i] = 1.0 / math.Pow(theta, float64(2*i)/float64(rotaryDim))
	}
	return out
}

// mRoPESectionOf reports which position dimension (0=T, 1=H, 2=W) the interleaved mRoPE assigns to rotary
// frequency index i, per apply_interleaved_mrope: the offset-1 stride-3 slice (up to section[1]*3) is H,
// the offset-2 stride-3 slice (up to section[2]*3) is W, and everything else — including all multiples of
// 3 and any index past a section's span — is T. The H and W slices never overlap (offsets 1 and 2, stride
// 3), so the assignment is order-independent.
func mRoPESectionOf(i int, section [3]int) int {
	if i%3 == 1 && i < section[1]*3 {
		return 1
	}
	if i%3 == 2 && i < section[2]*3 {
		return 2
	}
	return 0
}

// mRoPEInterleavedFreqs builds the interleaved mRoPE angle for each rotary pair from a 3D position
// (posT/H/W) and inv_freq. For a pure-text position (posT == posH == posW == p) every entry is p·inv_freq
// — the standard 1D partial-rotary angle — whatever the section split, which is the reduction the text
// decode relies on (see TestMRoPEReducesToPartialRotary).
func mRoPEInterleavedFreqs(posT, posH, posW float64, invFreq []float64, section [3]int) []float64 {
	pos := [3]float64{posT, posH, posW}
	out := make([]float64, len(invFreq))
	for i := range invFreq {
		out[i] = pos[mRoPESectionOf(i, section)] * invFreq[i]
	}
	return out
}

// applyRotaryAngles rotates the first 2·len(angles) dims of x by the given per-pair angles (rotate_half:
// pair i with i+half), leaving the rest unrotated — the angle-driven sibling of applyRotaryHalf. It lets a
// caller drive the rotation from mRoPE angles directly; for text those angles are p·inv_freq, so this is
// bit-identical to applyRotaryHalf(x, p, rotaryDim, theta).
func applyRotaryAngles(x []float32, angles []float64) {
	half := len(angles)
	for i := range half {
		c, s := math.Cos(angles[i]), math.Sin(angles[i])
		a, b := float64(x[i]), float64(x[i+half])
		x[i] = float32(a*c - b*s)
		x[i+half] = float32(b*c + a*s)
	}
}
