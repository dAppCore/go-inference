// SPDX-Licence-Identifier: EUPL-1.2

package turboquant

import "math"

// qjlConst is √(π/2) — the QJL reconstruction's normalising constant, from
// E[|Z|] = √(2/π) for a standard normal Z.
var qjlConst = math.Sqrt(math.Pi / 2)

// QProdEncoded is one row's Q_prod payload: TurboQuant's inner-product-
// preserving two-stage codec. Stage 1 is Q_mse at TotalBits-1 bits; stage 2
// is a 1-bit-per-coordinate QJL sign sketch of the stage-1 residual. A
// zero-norm row (Gamma == 0) carries no information in either stage.
type QProdEncoded struct {
	Gamma         float32
	Stage1Indices []byte // Q_mse indices at TotalBits-1 bits
	Rho           float32
	Signs         []byte // packed QJL sign bits, 1 bit/coordinate
	D             int
	TotalBits     int
}

// EncodeQProd quantises row x with TurboQuant's two-stage inner-product-
// preserving codec at totalBits total bits per coordinate (b ∈ {1,2,3,4}):
//
//  1. Stage 1 is Q_mse at (totalBits-1) bits, giving reconstruction ũ_mse in
//     the row's unit-norm space; residual r = u - ũ_mse.
//  2. Stage 2 is QJL: a seeded i.i.d. N(0,1) matrix S, storing sign(S·r) (1
//     bit/coordinate) plus ρ = ||r||₂.
//
// totalBits == 1 makes stage 1 a 0-bit (single-centroid, always-0 by
// symmetry) quantiser — the whole 1-bit budget goes to the QJL sign sketch.
//
//	e := EncodeQProd([]float32{3, 4}, 3, 42)
//	x := DecodeQProd(e, 42)
func EncodeQProd(x []float32, totalBits int, seed uint64) QProdEncoded {
	d := len(x)
	xf := toFloat64(x)
	gamma := l2Norm(xf)
	stage1Bits := totalBits - 1
	if gamma == 0 {
		return QProdEncoded{D: d, TotalBits: totalBits}
	}
	u := scaled(xf, 1/gamma)

	stage1Indices := quantiseUnit(u, stage1Bits, seed)
	uMSE := dequantiseUnit(stage1Indices, d, stage1Bits, seed)
	r := subtract(u, uMSE)
	rho := l2Norm(r)

	enc := QProdEncoded{
		Gamma:         float32(gamma),
		Stage1Indices: stage1Indices,
		D:             d,
		TotalBits:     totalBits,
	}
	if rho == 0 {
		return enc
	}
	s := qjlMatrixFor(seed, d)
	sr := s.mulVec(r)
	signs := make([]bool, d)
	for i, v := range sr {
		signs[i] = v >= 0
	}
	enc.Rho = float32(rho)
	enc.Signs = packSigns(signs)
	return enc
}

// DecodeQProd reverses EncodeQProd: reconstructs stage 1's ũ_mse, adds the
// QJL contribution (√(π/2)/d)·ρ·Sᵀ·q, and rescales by Gamma. seed must match
// the seed Encode used.
//
//	e := EncodeQProd([]float32{3, 4}, 3, 7)
//	x := DecodeQProd(e, 7)
func DecodeQProd(e QProdEncoded, seed uint64) []float32 {
	if e.Gamma == 0 {
		return make([]float32, e.D)
	}
	stage1Bits := e.TotalBits - 1
	uMSE := dequantiseUnit(e.Stage1Indices, e.D, stage1Bits, seed)
	u := uMSE
	if e.Rho > 0 {
		s := qjlMatrixFor(seed, e.D)
		q := unpackSigns(e.Signs, e.D)
		stq := s.mulVecT(q)
		c := qjlConst / float64(e.D) * float64(e.Rho)
		qjl := scaled(stq, c)
		u = add(uMSE, qjl)
	}
	return toFloat32(scaled(u, float64(e.Gamma)))
}

// MarshalQProd serialises e to its wire form: Gamma (4 bytes LE) ++ the
// stage-1 packed indices ++ Rho (4 bytes LE) ++ the packed QJL sign bits. D
// and TotalBits are not carried — same contract as MarshalQMSE.
//
//	data := MarshalQProd(EncodeQProd(row, 3, 42))
func MarshalQProd(e QProdEncoded) []byte {
	out := make([]byte, 4+len(e.Stage1Indices)+4+len(e.Signs))
	putFloat32LE(out, e.Gamma)
	off := 4
	off += copy(out[off:], e.Stage1Indices)
	putFloat32LE(out[off:], e.Rho)
	off += 4
	copy(out[off:], e.Signs)
	return out
}

// UnmarshalQProd reverses MarshalQProd, given the row dimension d and total
// bit width totalBits the encoder used.
//
//	e := UnmarshalQProd(data, 128, 3)
func UnmarshalQProd(data []byte, d, totalBits int) QProdEncoded {
	stage1Bits := totalBits - 1
	stage1Len := packedByteLen(d, stage1Bits)
	if len(data) < 4+stage1Len+4 {
		return QProdEncoded{D: d, TotalBits: totalBits}
	}
	gamma := getFloat32LE(data)
	off := 4
	stage1 := append([]byte(nil), data[off:off+stage1Len]...)
	off += stage1Len
	rho := getFloat32LE(data[off:])
	off += 4
	signs := append([]byte(nil), data[off:]...)
	return QProdEncoded{
		Gamma:         gamma,
		Stage1Indices: stage1,
		Rho:           rho,
		Signs:         signs,
		D:             d,
		TotalBits:     totalBits,
	}
}
