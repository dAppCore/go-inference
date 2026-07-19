// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"

	"dappco.re/go/inference/kv/turboquant"
)

// turboquant_device_test.go proves the S2 device kernels (kernels/lthn_turboquant.metal) against the
// S1 host reference (kv/turboquant) — the same Π and centroids on both sides, via the additive
// kv/turboquant.RotationMatrix/Centroids/UnpackIndices accessors, so any divergence is the kernels'
// own arithmetic, not a different random matrix or a different Lloyd-Max solve.
//
// Index-level byte parity (device argmin vs host argmin) is gated as a DIAGNOSTIC with a generous
// bound, not asserted bit-exact: kv/turboquant accumulates the rotation and the Lloyd-Max distance
// comparison in float64; Metal has no double-precision type at all, so the device kernel is
// necessarily float32 throughout. That gap is far smaller than a typical Lloyd-Max cell width, but a
// coordinate landing within a few ULP of a cell boundary can tip the nearest-centroid argmin either
// way — a structural property of comparing f32 to f64, not a bug, and asserting zero mismatches would
// make the test flaky rather than correct (see the file-level design note by TestTurboQuantRotateQuantDevice_Good).
// The HARD gates are the round-trip / dequant comparisons, which tolerate that rounding gap by
// construction (bf16-scale tolerance, MSE band) while still catching a real bug (wrong Π, wrong
// centroid table, wrong pack/unpack order) as a large, structural divergence.

func tqRequireKernel(t *testing.T, bits, d int) {
	t.Helper()
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set — device TurboQuant kernels")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("Metal runtime unavailable — device TurboQuant kernels: %v", err)
	}
	if !tqRotateQuantUsable(bits, d) {
		t.Skipf("lthn_tq_rotate_quant_b%d unavailable in this metallib (d=%d)", bits, d)
	}
	if !tqDequantUnrotateUsable(bits, d) {
		t.Skipf("lthn_tq_dequant_unrotate_b%d unavailable in this metallib (d=%d)", bits, d)
	}
}

// tqGenRows fills numRows rows of dimension d with i.i.d. standard-normal components — representative
// of the sphere-marginal density kv/turboquant's Lloyd-Max centroids are actually solved against.
func tqGenRows(seed int64, numRows, d int) []float32 {
	rng := rand.New(rand.NewSource(seed))
	out := make([]float32, numRows*d)
	for i := range out {
		out[i] = float32(rng.NormFloat64())
	}
	return out
}

// tqF32 narrows a float64 slice (kv/turboquant.RotationMatrix/Centroids) to the float32 the device
// buffers need.
func tqF32(x []float64) []float32 {
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = float32(v)
	}
	return out
}

// tqRowMSE is the mean squared error between two equal-length rows.
func tqRowMSE(orig, recon []float32) float64 {
	var sum float64
	for i := range orig {
		d := float64(orig[i]) - float64(recon[i])
		sum += d * d
	}
	return sum / float64(len(orig))
}

// tqBF16Tol is the house device-vs-host tolerance formula (composed_quant_backend_test.go) applied
// per-element: 1% of magnitude plus a floor, appropriate for an f32-device/f64-host comparison after
// a bf16-scale rounding budget.
func tqBF16Tol(want float32) float64 {
	return 1e-2 * (1 + math.Abs(float64(want)))
}

// TestTurboQuantRotateQuantDevice_Good proves the encode kernel's gamma and its nearest-centroid
// index choice track kv/turboquant.EncodeQMSE at the SAME (Π, centroids) — see the file header for
// why index agreement is gated as a generous-but-meaningful rate rather than asserted bit-exact.
func TestTurboQuantRotateQuantDevice_Good(t *testing.T) {
	const seed, numRows = uint64(42), 24
	for _, bits := range []int{2, 3, 4} {
		for _, d := range []int{64, 128} {
			t.Run(fmt.Sprintf("b%d_d%d", bits, d), func(t *testing.T) {
				tqRequireKernel(t, bits, d)
				rows := tqGenRows(int64(seed), numRows, d)
				pi := tqF32(turboquant.RotationMatrix(seed, d))
				centroids := tqF32(turboquant.Centroids(d, bits))

				gamma, packed, err := TurboQuantRotateQuantDevice(rows, pi, centroids, bits, numRows, d)
				if err != nil {
					t.Fatalf("TurboQuantRotateQuantDevice: %v", err)
				}
				if len(gamma) != numRows {
					t.Fatalf("gamma len = %d, want %d", len(gamma), numRows)
				}
				bytesPerRow := tqBytesPerRow(bits, d)
				if len(packed) != numRows*bytesPerRow {
					t.Fatalf("packed len = %d, want %d", len(packed), numRows*bytesPerRow)
				}

				var mismatches, total int
				for r := 0; r < numRows; r++ {
					row := rows[r*d : (r+1)*d]
					he := turboquant.EncodeQMSE(row, bits, seed)
					if diff := math.Abs(float64(gamma[r]) - float64(he.Gamma)); diff > 1e-3*(1+math.Abs(float64(he.Gamma))) {
						t.Errorf("row %d: device gamma %v vs host %v (Δ=%v)", r, gamma[r], he.Gamma, diff)
					}
					devIdx := turboquant.UnpackIndices(packed[r*bytesPerRow:(r+1)*bytesPerRow], d, bits)
					hostIdx := turboquant.UnpackIndices(he.Indices, d, bits)
					for c := 0; c < d; c++ {
						total++
						if devIdx[c] != hostIdx[c] {
							mismatches++
						}
					}
				}
				rate := float64(mismatches) / float64(total)
				t.Logf("b%d d%d: index agreement %d/%d coordinates mismatched (%.4f%%)", bits, d, mismatches, total, rate*100)
				if rate > 0.02 {
					t.Fatalf("b%d d%d: index mismatch rate %.4f%% exceeds the f32/f64 boundary-noise budget (2%%) — likely a real bug, not rounding", bits, d, rate*100)
				}
			})
		}
	}
}

// TestTurboQuantRotateQuantDevice_Bad pins the rejection shapes: an uninstantiated bit width, an
// out-of-range dimension, and mismatched slice sizes all error before any dispatch.
func TestTurboQuantRotateQuantDevice_Bad(t *testing.T) {
	const bits, d, numRows = 4, 64, 2
	tqRequireKernel(t, bits, d)
	pi := make([]float32, d*d)
	centroids := make([]float32, 1<<uint(bits))
	rows := make([]float32, numRows*d)

	if _, _, err := TurboQuantRotateQuantDevice(rows, pi, centroids, 5, numRows, d); err == nil {
		t.Fatal("bits=5 (no instantiation) must error")
	}
	if _, _, err := TurboQuantRotateQuantDevice(rows, pi, centroids, bits, numRows, 0); err == nil {
		t.Fatal("d=0 must error")
	}
	if _, _, err := TurboQuantRotateQuantDevice(rows, pi, centroids, bits, numRows, 512); err == nil {
		t.Fatal("d=512 (exceeds the kernel's fixed threadgroup span) must error")
	}
	if _, _, err := TurboQuantRotateQuantDevice(rows[:d], pi, centroids, bits, numRows, d); err == nil {
		t.Fatal("rows size mismatch must error")
	}
	if _, _, err := TurboQuantRotateQuantDevice(rows, pi[:d], centroids, bits, numRows, d); err == nil {
		t.Fatal("pi size mismatch must error")
	}
	if _, _, err := TurboQuantRotateQuantDevice(rows, pi, centroids[:1], bits, numRows, d); err == nil {
		t.Fatal("centroids size mismatch must error")
	}
	if tqRotateQuantUsable(2, -1) {
		t.Fatal("d=-1 must not be usable")
	}
	if tqRotateQuantUsable(7, d) {
		t.Fatal("bits=7 must not be usable")
	}
}

// TestTurboQuantDequantUnrotateDevice_Good isolates the decode kernel from any encode-side index
// disagreement: it feeds the HOST's own packed indices and gamma (kv/turboquant.EncodeQMSE's output)
// into the device dequant kernel and checks the reconstruction matches kv/turboquant.DecodeQMSE
// within bf16 tolerance — at that point the only remaining difference between the two sides is the
// unrotate matmul's f32-device-vs-f64-host arithmetic, no index-choice ambiguity at all.
func TestTurboQuantDequantUnrotateDevice_Good(t *testing.T) {
	const seed, numRows = uint64(7), 16
	for _, bits := range []int{2, 3, 4} {
		for _, d := range []int{64, 128} {
			t.Run(fmt.Sprintf("b%d_d%d", bits, d), func(t *testing.T) {
				tqRequireKernel(t, bits, d)
				rows := tqGenRows(99, numRows, d)
				pi := tqF32(turboquant.RotationMatrix(seed, d))
				centroids := tqF32(turboquant.Centroids(d, bits))
				bytesPerRow := tqBytesPerRow(bits, d)

				packed := make([]byte, numRows*bytesPerRow)
				gamma := make([]float32, numRows)
				want := make([]float32, numRows*d)
				for r := 0; r < numRows; r++ {
					row := rows[r*d : (r+1)*d]
					he := turboquant.EncodeQMSE(row, bits, seed)
					copy(packed[r*bytesPerRow:(r+1)*bytesPerRow], he.Indices)
					gamma[r] = he.Gamma
					copy(want[r*d:(r+1)*d], turboquant.DecodeQMSE(he, seed))
				}

				got, err := TurboQuantDequantUnrotateDevice(packed, pi, centroids, gamma, bits, numRows, d)
				if err != nil {
					t.Fatalf("TurboQuantDequantUnrotateDevice: %v", err)
				}
				if len(got) != len(want) {
					t.Fatalf("len = %d, want %d", len(got), len(want))
				}
				for i := range want {
					if diff := math.Abs(float64(got[i] - want[i])); diff > tqBF16Tol(want[i]) {
						t.Fatalf("elem %d: device %v vs host %v (Δ=%v exceeds bf16 tol)", i, got[i], want[i], diff)
					}
				}
				t.Logf("b%d d%d: device dequant of the host's own packed indices matches host DecodeQMSE within bf16 tol", bits, d)
			})
		}
	}
}

// TestTurboQuantDequantUnrotateDevice_Bad pins the rejection shapes, mirroring
// TestTurboQuantRotateQuantDevice_Bad for the decode kernel.
func TestTurboQuantDequantUnrotateDevice_Bad(t *testing.T) {
	const bits, d, numRows = 4, 64, 2
	tqRequireKernel(t, bits, d)
	pi := make([]float32, d*d)
	centroids := make([]float32, 1<<uint(bits))
	gamma := make([]float32, numRows)
	packed := make([]byte, numRows*tqBytesPerRow(bits, d))

	if _, err := TurboQuantDequantUnrotateDevice(packed, pi, centroids, gamma, 6, numRows, d); err == nil {
		t.Fatal("bits=6 (no instantiation) must error")
	}
	if _, err := TurboQuantDequantUnrotateDevice(packed, pi, centroids, gamma, bits, numRows, 0); err == nil {
		t.Fatal("d=0 must error")
	}
	if _, err := TurboQuantDequantUnrotateDevice(packed[:1], pi, centroids, gamma, bits, numRows, d); err == nil {
		t.Fatal("packed size mismatch must error")
	}
	if _, err := TurboQuantDequantUnrotateDevice(packed, pi, centroids, gamma[:1], bits, numRows, d); err == nil {
		t.Fatal("gamma size mismatch must error")
	}
	if tqDequantUnrotateUsable(bits, 0) {
		t.Fatal("d=0 must not be usable")
	}
}

// TestTurboQuantDeviceRoundTrip_Good is the primary parity gate: the whole device round trip (encode
// then decode) against kv/turboquant's own round trip (EncodeQMSE then DecodeQMSE), on the SAME rows
// — the distortion oracle from the task brief. Device and host MSE must land in the same band; a
// device implementation with a real bug (wrong rotation direction, wrong centroid table, a transposed
// pack) would land WAY outside a 3x band, not just drift within it.
func TestTurboQuantDeviceRoundTrip_Good(t *testing.T) {
	const seed, numRows = uint64(123), 32
	for _, bits := range []int{2, 3, 4} {
		for _, d := range []int{64, 128} {
			t.Run(fmt.Sprintf("b%d_d%d", bits, d), func(t *testing.T) {
				tqRequireKernel(t, bits, d)
				rows := tqGenRows(555, numRows, d)
				pi := tqF32(turboquant.RotationMatrix(seed, d))
				centroids := tqF32(turboquant.Centroids(d, bits))

				devGamma, devPacked, err := TurboQuantRotateQuantDevice(rows, pi, centroids, bits, numRows, d)
				if err != nil {
					t.Fatalf("TurboQuantRotateQuantDevice: %v", err)
				}
				devRecon, err := TurboQuantDequantUnrotateDevice(devPacked, pi, centroids, devGamma, bits, numRows, d)
				if err != nil {
					t.Fatalf("TurboQuantDequantUnrotateDevice: %v", err)
				}

				var devMSE, hostMSE float64
				for r := 0; r < numRows; r++ {
					row := rows[r*d : (r+1)*d]
					he := turboquant.EncodeQMSE(row, bits, seed)
					hostRecon := turboquant.DecodeQMSE(he, seed)
					devMSE += tqRowMSE(row, devRecon[r*d:(r+1)*d])
					hostMSE += tqRowMSE(row, hostRecon)
				}
				devMSE /= float64(numRows)
				hostMSE /= float64(numRows)
				t.Logf("b%d d%d: round-trip MSE device=%.6e host=%.6e", bits, d, devMSE, hostMSE)
				if hostMSE <= 0 {
					t.Fatalf("host MSE is non-positive (%.6e) — test fixture is degenerate", hostMSE)
				}
				if ratio := devMSE / hostMSE; ratio < 1.0/3 || ratio > 3.0 {
					t.Fatalf("b%d d%d: device round-trip MSE %.6e is not in the same band as S1's %.6e (ratio %.3f, want in [1/3,3])", bits, d, devMSE, hostMSE, ratio)
				}
			})
		}
	}
}

// TestTurboQuantDeviceRoundTrip_Ugly checks the all-zero row: kv/turboquant.EncodeQMSE never defines
// an index for it (Gamma==0 short-circuits before any rotation/quantisation), so the device kernel's
// own arbitrary index choice for that degenerate row must still round-trip to exactly zero — the
// dequant's ×gamma stage makes the choice of index irrelevant whenever gamma is 0.
func TestTurboQuantDeviceRoundTrip_Ugly(t *testing.T) {
	const bits, d, numRows = 3, 64, 1
	tqRequireKernel(t, bits, d)
	seed := uint64(11)
	pi := tqF32(turboquant.RotationMatrix(seed, d))
	centroids := tqF32(turboquant.Centroids(d, bits))
	rows := make([]float32, d) // all zero

	gamma, packed, err := TurboQuantRotateQuantDevice(rows, pi, centroids, bits, numRows, d)
	if err != nil {
		t.Fatalf("TurboQuantRotateQuantDevice(zero row): %v", err)
	}
	if gamma[0] != 0 {
		t.Fatalf("gamma = %v, want 0 for an all-zero row", gamma[0])
	}
	recon, err := TurboQuantDequantUnrotateDevice(packed, pi, centroids, gamma, bits, numRows, d)
	if err != nil {
		t.Fatalf("TurboQuantDequantUnrotateDevice(zero row): %v", err)
	}
	for i, v := range recon {
		if v != 0 {
			t.Fatalf("recon[%d] = %v, want 0 (gamma=0 must zero the row regardless of the arbitrary packed index)", i, v)
		}
	}
}
