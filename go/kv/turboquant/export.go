// SPDX-Licence-Identifier: EUPL-1.2

package turboquant

// export.go — additive, engine-neutral accessors for RFC #41 slice S2 (the device kernel work in
// go-inference's engine/metal): a cross-target parity prover must upload the EXACT SAME rotation
// matrix and centroid table this package's Encode/Decode functions use internally, or "byte-parity
// against S1" is meaningless — it would be comparing a device kernel against a DIFFERENT random Π
// and a DIFFERENT Lloyd-Max solve, not against this reference. These three functions expose the
// already-cached values (never regenerated, never renumbered) without moving any of the package's
// existing exported surface — EncodeQMSE/DecodeQMSE/EncodeQProd/DecodeQProd are untouched.

// RotationMatrix returns the deterministic d×d orthogonal rotation Π for seed, flattened row-major
// float64 (Π[i][j] at index i*d+j) — the SAME cached matrix EncodeQMSE/EncodeQProd generate
// internally via rotationFor. A defensive copy: mutating the result never corrupts the package's
// cache. Exported so a downstream device target (engine/metal's TurboQuant kernels) can upload the
// identical rotation data and prove index-level parity against this package's Encode/Decode as the
// correctness oracle.
//
//	pi := turboquant.RotationMatrix(42, 128) // len(pi) == 128*128
func RotationMatrix(seed uint64, d int) []float64 {
	m := rotationFor(seed, d)
	out := make([]float64, len(m.data))
	copy(out, m.data)
	return out
}

// Centroids returns the cached Lloyd-Max centroids for (d, bits), sorted ascending — the SAME table
// quantiseUnit/dequantiseUnit use internally. A defensive copy, like RotationMatrix. Exported for the
// same cross-target parity need.
//
//	c := turboquant.Centroids(128, 2) // 4 centroids
func Centroids(d, bits int) []float64 {
	c := centroidsFor(d, bits)
	out := make([]float64, len(c))
	copy(out, c)
	return out
}

// UnpackIndices reverses packBits for n values of the given bit width — exported so a cross-target
// parity test can compare S1's packed wire indices coordinate-by-coordinate (e.g. against a device
// kernel's own packed output) rather than only as an opaque byte blob.
//
//	turboquant.UnpackIndices([]byte{0x15}, 2, 3) // []int{5, 2}
func UnpackIndices(data []byte, n, bits int) []int {
	return unpackBits(data, n, bits)
}
