// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"math/rand"
	"os"
	"testing"

	"dappco.re/go/inference/kv/turboquant"
)

// sdpa_vector_tq_test.go — the S3 kernel gate: the TurboQuant live-KV store
// and the code-reading decode SDPA pair against host float references
// computing THE SAME math on dequantised rows (the byte-band idiom of
// sdpa_sinks_test.go / the q8 tests). The reference dequantises the device's
// own codes (γ·Πᵀ·centroid, f64 accumulation over the SAME f32 Π/centroid
// values the kernels read) and runs a plain f64 decode attention; rotation
// invariance makes that equal to the kernel's rotated-space computation up to
// fp rounding, so the band is arithmetic drift only — a wrong Π, a wrong
// table, a wrong unpack, or a wrong γ application shows as a structural,
// orders-larger divergence.
//
// Measured bands (this box, 2026-07-19, n=64/96 single-pass + n=knee+512
// 2-pass, hd ∈ {128, 256, 512}, modes (4,4)/(4,3)/(3,3)/(2,2), kv ∈ {1, 2}):
// max |Δ| vs the f64 oracle is ≤ 0.0041 in 33 of 36 cases; the outliers are
// hd=512 (4,4): single-pass kv=2 0.0369, 2-pass 0.0253, and 2-pass (4,3)
// 0.0137 — near-tied softmax weights where the bf16 rounding of the
// pre-rotated q tips a dominant score. The asserted band 0.08 carries ~2×
// margin over the worst measured case.

func tqKVRequire(t *testing.T, kBits, vBits, headDim int) {
	t.Helper()
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set — TurboQuant live-KV kernels")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("Metal runtime unavailable: %v", err)
	}
	if _, err := tqKVStorePipeline(kBits); err != nil {
		t.Skipf("lthn_tq_kv_store_bf16_b%d unavailable: %v", kBits, err)
	}
	if _, err := sdpaVectorTQPipeline(headDim, kBits, vBits); err != nil {
		t.Skipf("lthn_sdpa_vector_tq_bf16_%d unavailable: %v", headDim, err)
	}
}

// tqKVDequantRowsHost expands packed codes + γ back to rows — f64 accumulation
// over the lane's own f32 Π/centroids: k̃ = γ·Πᵀ·centroid[idx].
func tqKVDequantRowsHost(codes []byte, gammas []float32, rows, heads, d, bits int) [][]float64 {
	pi := tqKVPiF32(d)
	cent := tqKVCentroidsF32(d, bits)
	bytesPerHead := tqBytesPerRow(bits, d)
	out := make([][]float64, rows*heads)
	for r := 0; r < rows*heads; r++ {
		idx := turboquant.UnpackIndices(codes[r*bytesPerHead:(r+1)*bytesPerHead], d, bits)
		y := make([]float64, d)
		for i, ix := range idx {
			y[i] = float64(cent[ix])
		}
		x := make([]float64, d)
		g := float64(gammas[r])
		for o := 0; o < d; o++ {
			var acc float64
			for j := 0; j < d; j++ {
				acc += float64(pi[j*d+o]) * y[j] // Πᵀ — column o
			}
			x[o] = acc * g
		}
		out[r] = x
	}
	return out
}

// tqKVSDPAHostRef runs the plain f64 decode attention (one query row per
// head, GQA) over already-dequantised K/V rows — the oracle the kernel chain
// must match through rotation invariance.
func tqKVSDPAHostRef(q []float64, kRows, vRows [][]float64, nHeads, nKVHeads, d, n int, scale float64) []float64 {
	out := make([]float64, nHeads*d)
	for h := 0; h < nHeads; h++ {
		kv := h / (nHeads / nKVHeads)
		scores := make([]float64, n)
		maxS := math.Inf(-1)
		for i := 0; i < n; i++ {
			k := kRows[i*nKVHeads+kv]
			var dot float64
			for c := 0; c < d; c++ {
				dot += q[h*d+c] * k[c]
			}
			scores[i] = dot * scale
			if scores[i] > maxS {
				maxS = scores[i]
			}
		}
		var denom float64
		for i := range scores {
			scores[i] = math.Exp(scores[i] - maxS)
			denom += scores[i]
		}
		for i := 0; i < n; i++ {
			w := scores[i] / denom
			v := vRows[i*nKVHeads+kv]
			for c := 0; c < d; c++ {
				out[h*d+c] += w * v[c]
			}
		}
	}
	return out
}

// tqKVGenBF16 generates rows*heads bf16 rows of dimension d (standard normal,
// bf16-rounded) plus their f64 view — both sides of every gate see identical
// input values.
func tqKVGenBF16(seed int64, count, d int) (bf16 []byte, f64 []float64) {
	rng := rand.New(rand.NewSource(seed))
	f := make([]float32, count*d)
	for i := range f {
		f[i] = float32(rng.NormFloat64())
	}
	bf16 = toBF16Bytes(f)
	back := bf16ToF32Slice(bf16)
	f64 = make([]float64, len(back))
	for i, v := range back {
		f64[i] = float64(v)
	}
	return bf16, f64
}

// tqKVModes are the mode contract's (kBits, vBits) pairs: 4 / 3.5 / 3 / 2.
var tqKVModes = [][2]int{{4, 4}, {4, 3}, {3, 3}, {2, 2}}

// TestTurboQuantKVStoreDevice_Good gates the append kernel against
// kv/turboquant.EncodeQMSE on the SAME bf16-rounded rows: γ must track the f64
// reference within f32-accumulation slack, and the device's own dequantised
// reconstruction must sit inside the codec's MSE band (the S2 round-trip
// method, widened to bf16 inputs and the 512 head dim).
func TestTurboQuantKVStoreDevice_Good(t *testing.T) {
	for _, d := range []int{128, 256, 512} {
		for _, bits := range []int{2, 3, 4} {
			tqKVRequire(t, bits, bits, d)
			const heads = 4
			stage, ref := tqKVGenBF16(int64(1000+d+bits), heads, d)
			codes, gammas, err := TurboQuantKVStoreDevice(stage, heads, d, bits)
			if err != nil {
				t.Fatalf("d=%d b=%d: TurboQuantKVStoreDevice: %v", d, bits, err)
			}
			deq := tqKVDequantRowsHost(codes, gammas, 1, heads, d, bits)
			for h := 0; h < heads; h++ {
				row := ref[h*d : (h+1)*d]
				var norm float64
				for _, v := range row {
					norm += v * v
				}
				norm = math.Sqrt(norm)
				he := turboquant.EncodeQMSE(bf16ToF32Slice(stage[h*d*bf16Size:(h+1)*d*bf16Size]), bits, tqKVSeed)
				if diff := math.Abs(float64(gammas[h]) - float64(he.Gamma)); diff > 1e-3*math.Max(1, norm) {
					t.Fatalf("d=%d b=%d head %d: device γ %v vs EncodeQMSE γ %v (diff %g)", d, bits, h, gammas[h], he.Gamma, diff)
				}
				// reconstruction error inside the codec band: relative MSE well
				// below 1 and shrinking with bits (b=4 ≲ 0.02, b=2 ≲ 0.2).
				var mse float64
				for c := 0; c < d; c++ {
					e := deq[h][c] - row[c]
					mse += e * e
				}
				rel := mse / (norm * norm)
				bound := map[int]float64{2: 0.35, 3: 0.12, 4: 0.05}[bits]
				if rel > bound {
					t.Fatalf("d=%d b=%d head %d: reconstruction relative MSE %g exceeds %g", d, bits, h, rel, bound)
				}
			}
		}
	}
}

// TestTurboQuantKVStoreDevice_Bad proves the geometry refusals: bit widths
// outside {2,3,4}, a head dim past the 512 cap, and a size mismatch.
func TestTurboQuantKVStoreDevice_Bad(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("Metal runtime unavailable: %v", err)
	}
	if _, _, err := TurboQuantKVStoreDevice(make([]byte, 2*128*2), 2, 128, 5); err == nil {
		t.Fatal("expected bits=5 to refuse")
	}
	if _, _, err := TurboQuantKVStoreDevice(make([]byte, 1024*2), 1, 1024, 4); err == nil {
		t.Fatal("expected d=1024 (past the 512 threadgroup cap) to refuse")
	}
	if _, _, err := TurboQuantKVStoreDevice(make([]byte, 7), 1, 128, 4); err == nil {
		t.Fatal("expected staging size mismatch to refuse")
	}
}

// TestTurboQuantSDPADevice_Good gates the SINGLE-PASS code-reading SDPA chain
// (q pre-rotation → lthn_sdpa_vector_tq → output unrotation) against the f64
// oracle on dequantised rows, across head dims 128/256/512, every mode
// contract bit pairing, and MQA (kv=1) + GQA (kv=2) geometries.
func TestTurboQuantSDPADevice_Good(t *testing.T) {
	for _, d := range []int{128, 256, 512} {
		for _, mode := range tqKVModes {
			kBits, vBits := mode[0], mode[1]
			tqKVRequire(t, kBits, vBits, d)
			for _, nKV := range []int{1, 2} {
				const nHeads = 4
				n := 64 + nKV*32
				scale := float32(1 / math.Sqrt(float64(d)))

				kStage, _ := tqKVGenBF16(int64(2000+d+kBits), n*nKV, d)
				vStage, _ := tqKVGenBF16(int64(3000+d+vBits), n*nKV, d)
				qBytes, qRef := tqKVGenBF16(int64(4000+d), nHeads, d)

				kCodes, kGammas, err := TurboQuantKVStoreDevice(kStage, n*nKV, d, kBits)
				if err != nil {
					t.Fatalf("K store: %v", err)
				}
				vCodes, vGammas, err := TurboQuantKVStoreDevice(vStage, n*nKV, d, vBits)
				if err != nil {
					t.Fatalf("V store: %v", err)
				}

				got, err := TurboQuantSDPADevice(qBytes, kCodes, vCodes, kGammas, vGammas, nHeads, nKV, d, n, kBits, vBits, scale, false)
				if err != nil {
					t.Fatalf("TurboQuantSDPADevice: %v", err)
				}

				kRows := tqKVDequantRowsHost(kCodes, kGammas, n, nKV, d, kBits)
				vRows := tqKVDequantRowsHost(vCodes, vGammas, n, nKV, d, vBits)
				want := tqKVSDPAHostRef(qRef, kRows, vRows, nHeads, nKV, d, n, float64(scale))

				gotF := bf16ToF32Slice(got)
				var maxDiff float64
				for i := range want {
					if diff := math.Abs(float64(gotF[i]) - want[i]); diff > maxDiff {
						maxDiff = diff
					}
				}
				t.Logf("d=%d k%dv%d kv=%d single-pass: max |Δ| vs oracle %g", d, kBits, vBits, nKV, maxDiff)
				if maxDiff > 0.08 {
					t.Fatalf("d=%d k%dv%d kv=%d: max |Δ| vs dequantised-rows oracle %g exceeds 0.08", d, kBits, vBits, nKV, maxDiff)
				}
			}
		}
	}
}

// TestTurboQuantSDPADevice_TwoPass_Good is the same oracle gate through the
// 2-pass pair (lthn_sdpa_vector_2pass_1_tq + MLX's unchanged pass-2 merge) at
// a depth past the 2-pass knee.
func TestTurboQuantSDPADevice_TwoPass_Good(t *testing.T) {
	for _, d := range []int{128, 256, 512} {
		for _, mode := range tqKVModes {
			kBits, vBits := mode[0], mode[1]
			tqKVRequire(t, kBits, vBits, d)
			const nHeads, nKV = 4, 1
			n := sdpa2PassMinKV + 512
			scale := float32(1 / math.Sqrt(float64(d)))

			kStage, _ := tqKVGenBF16(int64(5000+d+kBits), n*nKV, d)
			vStage, _ := tqKVGenBF16(int64(6000+d+vBits), n*nKV, d)
			qBytes, qRef := tqKVGenBF16(int64(7000+d), nHeads, d)

			kCodes, kGammas, err := TurboQuantKVStoreDevice(kStage, n*nKV, d, kBits)
			if err != nil {
				t.Fatalf("K store: %v", err)
			}
			vCodes, vGammas, err := TurboQuantKVStoreDevice(vStage, n*nKV, d, vBits)
			if err != nil {
				t.Fatalf("V store: %v", err)
			}

			got, err := TurboQuantSDPADevice(qBytes, kCodes, vCodes, kGammas, vGammas, nHeads, nKV, d, n, kBits, vBits, scale, true)
			if err != nil {
				t.Fatalf("TurboQuantSDPADevice(twoPass): %v", err)
			}

			kRows := tqKVDequantRowsHost(kCodes, kGammas, n, nKV, d, kBits)
			vRows := tqKVDequantRowsHost(vCodes, vGammas, n, nKV, d, vBits)
			want := tqKVSDPAHostRef(qRef, kRows, vRows, nHeads, nKV, d, n, float64(scale))

			gotF := bf16ToF32Slice(got)
			var maxDiff float64
			for i := range want {
				if diff := math.Abs(float64(gotF[i]) - want[i]); diff > maxDiff {
					maxDiff = diff
				}
			}
			t.Logf("d=%d k%dv%d 2-pass: max |Δ| vs oracle %g", d, kBits, vBits, maxDiff)
			if maxDiff > 0.08 {
				t.Fatalf("d=%d k%dv%d 2-pass: max |Δ| vs dequantised-rows oracle %g exceeds 0.08", d, kBits, vBits, maxDiff)
			}
		}
	}
}

// TestTurboQuantSDPADevice_TwoPassShallow_Good pins the LIVE prefill regime the
// deep gate misses: N far below the block fan (E2B kv=1 bakes 256 blocks; the
// first decode steps run N ≤ 32), so almost every pass-1 block is EMPTY and the
// merge must reconstruct from a handful of live partials. gqa=8 matches the
// served E2B head shape.
func TestTurboQuantSDPADevice_TwoPassShallow_Good(t *testing.T) {
	for _, d := range []int{512, 256} {
		const kBits, vBits = 4, 4
		tqKVRequire(t, kBits, vBits, d)
		const nHeads, nKV, n = 8, 1, 16
		scale := float32(1 / math.Sqrt(float64(d)))

		kStage, _ := tqKVGenBF16(int64(11000+d), n*nKV, d)
		vStage, _ := tqKVGenBF16(int64(12000+d), n*nKV, d)
		qBytes, qRef := tqKVGenBF16(int64(13000+d), nHeads, d)

		kCodes, kGammas, err := TurboQuantKVStoreDevice(kStage, n*nKV, d, kBits)
		if err != nil {
			t.Fatalf("K store: %v", err)
		}
		vCodes, vGammas, err := TurboQuantKVStoreDevice(vStage, n*nKV, d, vBits)
		if err != nil {
			t.Fatalf("V store: %v", err)
		}
		got, err := TurboQuantSDPADevice(qBytes, kCodes, vCodes, kGammas, vGammas, nHeads, nKV, d, n, kBits, vBits, scale, true)
		if err != nil {
			t.Fatalf("TurboQuantSDPADevice(twoPass, shallow): %v", err)
		}
		kRows := tqKVDequantRowsHost(kCodes, kGammas, n, nKV, d, kBits)
		vRows := tqKVDequantRowsHost(vCodes, vGammas, n, nKV, d, vBits)
		want := tqKVSDPAHostRef(qRef, kRows, vRows, nHeads, nKV, d, n, float64(scale))
		gotF := bf16ToF32Slice(got)
		var maxDiff float64
		for i := range want {
			if diff := math.Abs(float64(gotF[i]) - want[i]); diff > maxDiff {
				maxDiff = diff
			}
		}
		t.Logf("d=%d shallow 2-pass (N=%d, gqa=8): max |Δ| vs oracle %g", d, n, maxDiff)
		if maxDiff > 0.08 {
			t.Fatalf("d=%d shallow 2-pass: max |Δ| %g exceeds 0.08", d, maxDiff)
		}
	}
}

// TestTurboQuantSDPADevice_Bad proves the driver guards: unsupported head dim,
// unsupported bits, a GQA factor that does not divide, and a size mismatch.
func TestTurboQuantSDPADevice_Bad(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("Metal runtime unavailable: %v", err)
	}
	q := make([]byte, 4*128*bf16Size)
	if _, err := TurboQuantSDPADevice(q, nil, nil, nil, nil, 4, 1, 96, 4, 4, 4, 1, false); err == nil {
		t.Fatal("expected head dim 96 (no instantiation) to refuse")
	}
	if _, err := TurboQuantSDPADevice(q, nil, nil, nil, nil, 4, 1, 128, 4, 1, 4, 1, false); err == nil {
		t.Fatal("expected kBits=1 to refuse")
	}
	if _, err := TurboQuantSDPADevice(q, nil, nil, nil, nil, 4, 3, 128, 4, 4, 4, 1, false); err == nil {
		t.Fatal("expected nHeads%nKVHeads != 0 to refuse")
	}
	if _, err := TurboQuantSDPADevice(q, make([]byte, 7), nil, nil, nil, 4, 1, 128, 4, 4, 4, 1, false); err == nil {
		t.Fatal("expected code-plane size mismatch to refuse")
	}
}

// TestTurboQuantSDPADevice_Ugly pins the zero-row boundary: an all-zero K/V
// row stores γ=0 and must contribute exactly a zero value vector with a
// uniform-score share — no NaN, no drift — matching the host oracle on the
// same dequantised (zero) rows.
func TestTurboQuantSDPADevice_Ugly(t *testing.T) {
	const d, nHeads, nKV, n, kBits, vBits = 128, 2, 1, 8, 4, 4
	tqKVRequire(t, kBits, vBits, d)
	scale := float32(1 / math.Sqrt(float64(d)))

	kStage, _ := tqKVGenBF16(8000, n, d)
	vStage, _ := tqKVGenBF16(9000, n, d)
	// zero out row 3 on both sides
	for i := 3 * d * bf16Size; i < 4*d*bf16Size; i++ {
		kStage[i], vStage[i] = 0, 0
	}
	qBytes, qRef := tqKVGenBF16(10000, nHeads, d)

	kCodes, kGammas, err := TurboQuantKVStoreDevice(kStage, n, d, kBits)
	if err != nil {
		t.Fatalf("K store: %v", err)
	}
	vCodes, vGammas, err := TurboQuantKVStoreDevice(vStage, n, d, vBits)
	if err != nil {
		t.Fatalf("V store: %v", err)
	}
	if kGammas[3] != 0 || vGammas[3] != 0 {
		t.Fatalf("zero row stored γ (%v, %v), want 0", kGammas[3], vGammas[3])
	}

	got, err := TurboQuantSDPADevice(qBytes, kCodes, vCodes, kGammas, vGammas, nHeads, nKV, d, n, kBits, vBits, scale, false)
	if err != nil {
		t.Fatalf("TurboQuantSDPADevice: %v", err)
	}
	gotF := bf16ToF32Slice(got)
	for i, v := range gotF {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("out[%d] = %v — zero-γ row produced a non-finite output", i, v)
		}
	}
	kRows := tqKVDequantRowsHost(kCodes, kGammas, n, nKV, d, kBits)
	vRows := tqKVDequantRowsHost(vCodes, vGammas, n, nKV, d, vBits)
	want := tqKVSDPAHostRef(qRef, kRows, vRows, nHeads, nKV, d, n, float64(scale))
	for i := range want {
		if diff := math.Abs(float64(gotF[i]) - want[i]); diff > 0.05 {
			t.Fatalf("out[%d] = %v, oracle %v (diff %g)", i, gotF[i], want[i], diff)
		}
	}
}

// TestEmitSDPAVectorTQ_Good asserts the single-pass TQ binding ABI on the
// recording sink: γ planes at 11/12, centroid tables at 13/14, gqa at 4, the
// inline N at 5 — and nothing at the q8 lane's scale indices beyond them.
func TestEmitSDPAVectorTQ_Good(t *testing.T) {
	rec := &recordingDispatchSink{}
	emitSDPAVectorTQ(rec, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, 8, 2, 64, 128, 512, 96, 384, 0.125)
	for _, idx := range []uint{11, 12, 13, 14} {
		if !rec.boundBuf(idx) {
			t.Fatalf("emitSDPAVectorTQ did not bind buffer(%d)", idx)
		}
	}
	if got := rec.i32[4]; got != 4 {
		t.Fatalf("gqa_factor(4) = %d, want 4", got)
	}
	if got := rec.i32[5]; got != 64 {
		t.Fatalf("N(5) = %d, want 64", got)
	}
}

// TestEmitSDPAVector2Pass1TQ_Good asserts the pass-1 TQ ABI: kCentroids at 6
// (the MLX ABI's free slot), N inline at 7, γ planes at 13/14, vCentroids at
// 15 — and NOTHING at 16, which the recorded arch ICB's
// maxKernelBufferBindCount=16 turns into a silent no-op (the S3 bring-up bug).
func TestEmitSDPAVector2Pass1TQ_Good(t *testing.T) {
	rec := &recordingDispatchSink{}
	emitSDPAVector2Pass1TQ(rec, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, 8, 2, 4096, 32, 128, 512, 96, 384, 0.125)
	for _, idx := range []uint{6, 13, 14, 15} {
		if !rec.boundBuf(idx) {
			t.Fatalf("emitSDPAVector2Pass1TQ did not bind buffer(%d)", idx)
		}
	}
	if rec.boundBuf(16) {
		t.Fatal("emitSDPAVector2Pass1TQ bound buffer(16) — past the ICB's maxKernelBufferBindCount, a silent no-op")
	}
	if got := rec.i32[7]; got != 4096 {
		t.Fatalf("N(7) = %d, want 4096", got)
	}
}

// TestEmitTQKVStore_Good asserts the store ABI: staging fixed at 0, Π at 1,
// centroids at 2, the rebindable codes/γ outputs at 3/4, d inline at 5.
func TestEmitTQKVStore_Good(t *testing.T) {
	rec := &recordingDispatchSink{}
	emitTQKVStore(rec, nil, nil, nil, nil, nil, 0, nil, 0, 2, 512)
	for _, idx := range []uint{0, 1, 2, 3, 4} {
		if !rec.boundBuf(idx) {
			t.Fatalf("emitTQKVStore did not bind buffer(%d)", idx)
		}
	}
	if got := rec.i32[5]; got != 512 {
		t.Fatalf("d(5) = %d, want 512", got)
	}
}
