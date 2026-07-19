// SPDX-Licence-Identifier: EUPL-1.2

package turboquant

// QMSEEncoded is one row's Q_mse payload: the row's f32 norm plus the packed
// per-coordinate Lloyd-Max centroid indices in the rotated basis. A
// zero-norm row (Gamma == 0) carries no information in Indices — Decode
// short-circuits it to an all-zero row.
type QMSEEncoded struct {
	Gamma   float32
	Indices []byte
	D       int
	Bits    int
}

// quantiseUnit rotates u by Π (rotationFor(seed, len(u))) into y = Π·u and
// quantises each coordinate of y independently to its nearest Lloyd-Max
// centroid at bits width — the shared core of both Q_mse and Q_prod's
// stage 1. u is assumed unit-ish: the Lloyd-Max centroids are calibrated
// against the ||u||=1 sphere-marginal density, so a caller must normalise
// before calling this.
func quantiseUnit(u []float64, bits int, seed uint64) []byte {
	d := len(u)
	pi := rotationFor(seed, d)
	y := pi.mulVec(u)
	centroids := centroidsFor(d, bits)
	indices := make([]int, d)
	for i, yi := range y {
		indices[i] = nearestCentroid(yi, centroids)
	}
	return packBits(indices, bits)
}

// dequantiseUnit reverses quantiseUnit: unpacks the indices, maps each to
// its Lloyd-Max centroid, and un-rotates by Πᵀ. Returns the reconstruction
// ũ in the same unit-ish space u occupied.
func dequantiseUnit(indices []byte, d, bits int, seed uint64) []float64 {
	centroids := centroidsFor(d, bits)
	idx := unpackBits(indices, d, bits)
	y := make([]float64, d)
	for i, ix := range idx {
		y[i] = centroids[ix]
	}
	pi := rotationFor(seed, d)
	return pi.mulVecT(y)
}

// EncodeQMSE quantises row x with the MSE-optimal TurboQuant codec at bits
// bits per coordinate (b ∈ {1,2,3,4}): normalise by the row's own L2 norm,
// rotate by a seeded random orthogonal matrix, then quantise each rotated
// coordinate independently against the Lloyd-Max centroids solved for the
// sphere-marginal density at this dimension and bit width.
//
// A zero row (every element exactly 0) encodes as Gamma: 0 with no
// meaningful indices — there is no direction to rotate.
//
//	e := EncodeQMSE([]float32{3, 4}, 2, 42)
//	x := DecodeQMSE(e, 42) // ≈ {3, 4}, MSE-bounded by the 2-bit codebook
func EncodeQMSE(x []float32, bits int, seed uint64) QMSEEncoded {
	d := len(x)
	xf := toFloat64(x)
	gamma := l2Norm(xf)
	if gamma == 0 {
		return QMSEEncoded{D: d, Bits: bits}
	}
	u := scaled(xf, 1/gamma)
	return QMSEEncoded{
		Gamma:   float32(gamma),
		Indices: quantiseUnit(u, bits, seed),
		D:       d,
		Bits:    bits,
	}
}

// DecodeQMSE reverses EncodeQMSE: unpacks the centroid indices, un-rotates,
// and rescales by Gamma. seed must match the seed Encode used — it is not
// carried in QMSEEncoded because a caller measuring many rows under one
// codec instance already fixes it once (see QMSECodec).
//
//	e := EncodeQMSE([]float32{3, 4}, 4, 7)
//	x := DecodeQMSE(e, 7)
func DecodeQMSE(e QMSEEncoded, seed uint64) []float32 {
	if e.Gamma == 0 {
		return make([]float32, e.D)
	}
	u := dequantiseUnit(e.Indices, e.D, e.Bits, seed)
	return toFloat32(scaled(u, float64(e.Gamma)))
}

// MarshalQMSE serialises e to its wire form: Gamma as 4 little-endian bytes
// followed by the packed indices verbatim. D and Bits are not carried in the
// payload — a decoder must already know them (they are per-codec-instance
// constants; see QMSECodec), exactly like the row dimension itself.
//
//	data := MarshalQMSE(EncodeQMSE(row, 2, 42))
func MarshalQMSE(e QMSEEncoded) []byte {
	out := make([]byte, 4+len(e.Indices))
	putFloat32LE(out, e.Gamma)
	copy(out[4:], e.Indices)
	return out
}

// UnmarshalQMSE reverses MarshalQMSE, given the row dimension d and bit
// width bits the encoder used.
//
//	e := UnmarshalQMSE(data, 128, 2)
func UnmarshalQMSE(data []byte, d, bits int) QMSEEncoded {
	if len(data) < 4 {
		return QMSEEncoded{D: d, Bits: bits}
	}
	gamma := getFloat32LE(data)
	indices := append([]byte(nil), data[4:]...)
	return QMSEEncoded{Gamma: gamma, Indices: indices, D: d, Bits: bits}
}
