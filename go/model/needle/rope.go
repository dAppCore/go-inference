// SPDX-Licence-Identifier: EUPL-1.2

package needle

import "math"

// ropeTable holds precomputed cos/sin for positions 0..maxPos-1, laid out
// [pos*headDim + d] with the reference's emb = cat(freqs, freqs) duplication so
// entry d and d+half share an angle. Needle uses the GPT-NeoX "rotate_half"
// convention (a contiguous split, not interleaved pairs), applied to q and k
// *after* per-head QK-norm, on self-attention only (cross-attention passes no
// RoPE).
type ropeTable struct {
	headDim int
	cos     []float32
	sin     []float32
}

// newRopeTable builds the cos/sin tables for a given head width, base theta and
// maximum position.
//
//	rt := newRopeTable(64, 10000, 128)
func newRopeTable(headDim int, theta float64, maxPos int) *ropeTable {
	half := headDim / 2
	invFreq := make([]float64, half)
	for j := range half {
		invFreq[j] = 1.0 / math.Pow(theta, float64(2*j)/float64(headDim))
	}
	rt := &ropeTable{
		headDim: headDim,
		cos:     make([]float32, maxPos*headDim),
		sin:     make([]float32, maxPos*headDim),
	}
	for p := range maxPos {
		for j := range half {
			angle := float64(p) * invFreq[j]
			c := float32(math.Cos(angle))
			s := float32(math.Sin(angle))
			rt.cos[p*headDim+j] = c
			rt.cos[p*headDim+j+half] = c
			rt.sin[p*headDim+j] = s
			rt.sin[p*headDim+j+half] = s
		}
	}
	return rt
}

// apply rotates one head vector (length headDim) in place at position pos:
//
//	out[d] = x[d]·cos[d] + rotate_half(x)[d]·sin[d]
//	rotate_half(x)[d] = -x[d+half] (d < half) or x[d-half] (d >= half)
func (rt *ropeTable) apply(vec []float32, pos int) {
	half := rt.headDim / 2
	base := pos * rt.headDim
	rotated := make([]float32, rt.headDim)
	for d := range half {
		rotated[d] = -vec[d+half]
		rotated[d+half] = vec[d]
	}
	for d := range rt.headDim {
		vec[d] = vec[d]*rt.cos[base+d] + rotated[d]*rt.sin[base+d]
	}
}
