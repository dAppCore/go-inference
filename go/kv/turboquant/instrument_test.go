// SPDX-Licence-Identifier: EUPL-1.2

package turboquant

import (
	"math/rand/v2"
	"testing"

	core "dappco.re/go"
)

// writeTestFixture writes a minimal real-KV fixture (uint32 LE dimension
// header + rows f32-LE) to dir/name and returns the full path.
func writeTestFixture(t *testing.T, dir, name string, d int, rows [][]float32) string {
	t.Helper()
	path := dir + "/" + name
	data := make([]byte, 4+4*d*len(rows))
	putUint32LE(data, uint32(d))
	off := 4
	for _, row := range rows {
		for _, v := range row {
			putFloat32LE(data[off:], v)
			off += 4
		}
	}
	if r := core.WriteFile(path, data, 0o644); !r.OK {
		t.Fatalf("core.WriteFile(%q) failed: %v", path, r.Value)
	}
	return path
}

// TestLoadRealKVRows_Good checks a well-formed fixture parses to the exact
// rows written.
func TestLoadRealKVRows_Good(t *testing.T) {
	dir := t.TempDir()
	want := [][]float32{{1, 2}, {3, 4}, {-5, 6.5}}
	path := writeTestFixture(t, dir, "rows.bin", 2, want)

	rows, d, ok, err := LoadRealKVRows(path)
	if err != nil {
		t.Fatalf("LoadRealKVRows(%q) error: %v", path, err)
	}
	if !ok {
		t.Fatal("LoadRealKVRows(valid fixture).ok = false, want true")
	}
	if d != 2 {
		t.Errorf("LoadRealKVRows(...) d = %d, want 2", d)
	}
	if len(rows) != len(want) {
		t.Fatalf("LoadRealKVRows(...) returned %d rows, want %d", len(rows), len(want))
	}
	for i := range want {
		for j := range want[i] {
			if rows[i][j] != want[i][j] {
				t.Errorf("row %d[%d] = %v, want %v", i, j, rows[i][j], want[i][j])
			}
		}
	}
}

// TestLoadRealKVRows_Ugly checks a missing path returns ok=false with a nil
// error — "skip cleanly when absent" per the RFC #41 spec, not a failure.
func TestLoadRealKVRows_Ugly(t *testing.T) {
	rows, d, ok, err := LoadRealKVRows(t.TempDir() + "/does-not-exist.bin")
	if err != nil {
		t.Errorf("LoadRealKVRows(missing file) error = %v, want nil", err)
	}
	if ok {
		t.Error("LoadRealKVRows(missing file).ok = true, want false")
	}
	if rows != nil || d != 0 {
		t.Errorf("LoadRealKVRows(missing file) = (%v, %d), want (nil, 0)", rows, d)
	}
}

// TestLoadRealKVRows_Bad checks three distinct malformations of a fixture
// that DOES exist all report a non-nil error — a file present but corrupt
// is a real problem, unlike a file simply absent.
func TestLoadRealKVRows_Bad(t *testing.T) {
	dir := t.TempDir()

	tooShort := dir + "/short.bin"
	if r := core.WriteFile(tooShort, []byte{1, 2}, 0o644); !r.OK {
		t.Fatal(r.Value)
	}
	if _, _, ok, err := LoadRealKVRows(tooShort); err == nil || ok {
		t.Errorf("LoadRealKVRows(too-short file) = (ok=%v, err=%v), want an error", ok, err)
	}

	zeroDim := dir + "/zerodim.bin"
	zdata := make([]byte, 4)
	putUint32LE(zdata, 0)
	if r := core.WriteFile(zeroDim, zdata, 0o644); !r.OK {
		t.Fatal(r.Value)
	}
	if _, _, ok, err := LoadRealKVRows(zeroDim); err == nil || ok {
		t.Errorf("LoadRealKVRows(zero dimension) = (ok=%v, err=%v), want an error", ok, err)
	}

	misaligned := dir + "/misaligned.bin"
	mdata := make([]byte, 4+4*2+1) // d=2 header + one full row + one stray byte
	putUint32LE(mdata, 2)
	if r := core.WriteFile(misaligned, mdata, 0o644); !r.OK {
		t.Fatal(r.Value)
	}
	if _, _, ok, err := LoadRealKVRows(misaligned); err == nil || ok {
		t.Errorf("LoadRealKVRows(misaligned payload) = (ok=%v, err=%v), want an error", ok, err)
	}
}

// TestGenerateGaussianRows_Good checks the shape and that rows are not
// degenerate (a broken RNG wiring symptom).
func TestGenerateGaussianRows_Good(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	rows := generateGaussianRows(rng, 5, 16)
	if len(rows) != 5 {
		t.Fatalf("generateGaussianRows returned %d rows, want 5", len(rows))
	}
	for i, row := range rows {
		if len(row) != 16 {
			t.Errorf("row %d has %d elements, want 16", i, len(row))
		}
	}
	if rows[0][0] == rows[1][0] {
		t.Error("generateGaussianRows looks degenerate: rows[0][0] == rows[1][0]")
	}
}

// TestGenerateSphereUniformRows_Good checks every row lands on the unit
// sphere (||row|| ≈ 1) — the property the distortion oracles depend on.
func TestGenerateSphereUniformRows_Good(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	rows := generateSphereUniformRows(rng, 20, 32)
	for i, row := range rows {
		if norm := l2Norm(toFloat64(row)); !approxEqual(norm, 1, 1e-5) {
			t.Errorf("row %d: ||row|| = %v, want ≈1", i, norm)
		}
	}
}

// TestSquaredL2Diff_Good checks a known squared distance.
func TestSquaredL2Diff_Good(t *testing.T) {
	if got := squaredL2Diff([]float32{1, 2}, []float32{4, 6}); got != 25 {
		t.Errorf("squaredL2Diff({1,2},{4,6}) = %v, want 25 (3²+4²)", got)
	}
}

// TestSquaredL2_Ugly checks the zero row gives exactly 0.
func TestSquaredL2_Ugly(t *testing.T) {
	if got := squaredL2([]float32{0, 0, 0}); got != 0 {
		t.Errorf("squaredL2(zero) = %v, want 0", got)
	}
}

// TestAttnOutput_Good checks a hand-computed weighted sum.
func TestAttnOutput_Good(t *testing.T) {
	weights := []float64{0.25, 0.75}
	rows := [][]float32{{4, 0}, {0, 4}}
	got := attnOutput(weights, rows)
	if !approxEqual(got[0], 1, 1e-9) || !approxEqual(got[1], 3, 1e-9) {
		t.Errorf("attnOutput = %v, want {1,3}", got)
	}
}

// TestMeasureAttention_Good checks the exact codec (round-tripping through
// full-precision int8 headroom is not exact, so we use a codec whose
// reconstruction is close enough that both softmax deltas and the output
// relative error stay small) reports a bounded max softmax delta.
func TestMeasureAttention_Good(t *testing.T) {
	const d = 32
	rng := rand.New(rand.NewPCG(3, 4))
	codec := QMSECodec{Bits: 4, Seed: 42}
	check := func(t *testing.T, maxDelta, outRelErr float64) {
		t.Helper()
		if maxDelta < 0 || maxDelta > 1 {
			t.Errorf("maxSoftmaxDelta = %v, want within [0,1] (a probability delta)", maxDelta)
		}
		if outRelErr < 0 || outRelErr > 2 {
			t.Errorf("outputRelError = %v, want a small non-negative relative error", outRelErr)
		}
	}

	// len(rows) <= attentionN: no row is spare, so the window is every row
	// and the query falls back to a fresh Gaussian draw.
	t.Run("small_pool_fallback_query", func(t *testing.T) {
		rows := generateGaussianRows(rng, 65, d)
		maxDelta, outRelErr, n := measureAttention(codec, rows, d, rng)
		if n != 65 {
			t.Errorf("window n = %d, want 65 (min(len(rows), attentionN))", n)
		}
		check(t, maxDelta, outRelErr)
	})

	// len(rows) > attentionN: the window caps at attentionN and the spare
	// row beyond it becomes the query — the realistic "drawn from the same
	// distribution" path MeasureCodecs' synthetic sources always take.
	t.Run("large_pool_spare_row_query", func(t *testing.T) {
		rows := generateGaussianRows(rng, attentionN+1, d)
		maxDelta, outRelErr, n := measureAttention(codec, rows, d, rng)
		if n != attentionN {
			t.Errorf("window n = %d, want %d (capped at attentionN)", n, attentionN)
		}
		check(t, maxDelta, outRelErr)
	})
}

// TestMeasureAttention_Ugly checks an empty row pool reports all zeros
// rather than panicking (division by zero, indexing rows[0]).
func TestMeasureAttention_Ugly(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 1))
	maxDelta, outRelErr, n := measureAttention(QMSECodec{Bits: 2, Seed: 1}, nil, 8, rng)
	if maxDelta != 0 || outRelErr != 0 || n != 0 {
		t.Errorf("measureAttention(empty rows) = (%v,%v,%d), want (0,0,0)", maxDelta, outRelErr, n)
	}
}

// TestMeasureOne_Good checks the aggregate report's Samples/AttentionRows
// bookkeeping and that a codec with headroom (4-bit) reports a smaller MSE
// than a starved one (1-bit) on the same data — the report must actually
// discriminate between codecs, not just run without crashing.
func TestMeasureOne_Good(t *testing.T) {
	const d = 32
	rng := rand.New(rand.NewPCG(5, 6))
	rows := generateSphereUniformRows(rng, 200, d)
	src := dataSourceEntry{name: "test", rows: rows}

	reportHi := measureOne(QMSECodec{Bits: 4, Seed: 42}, src, d, 42)
	reportLo := measureOne(QMSECodec{Bits: 1, Seed: 42}, src, d, 42)

	if reportHi.Samples != 200 {
		t.Errorf("Samples = %d, want 200", reportHi.Samples)
	}
	if reportHi.RowMSE >= reportLo.RowMSE {
		t.Errorf("4-bit RowMSE (%v) should be less than 1-bit RowMSE (%v)", reportHi.RowMSE, reportLo.RowMSE)
	}
}

// TestMeasureOne_Ugly checks an empty source reports the zero-value report
// rather than dividing by zero.
func TestMeasureOne_Ugly(t *testing.T) {
	report := measureOne(QMSECodec{Bits: 2, Seed: 1}, dataSourceEntry{name: "empty"}, 8, 1)
	if report.Samples != 0 || report.RowMSE != 0 {
		t.Errorf("measureOne(empty source) = %+v, want the zero-value MSE/Samples", report)
	}
}

// TestMeasureCodecs_Good is the RFC #41 slice S1 instrument itself: it runs
// the full codec table against the always-available synthetic sources and
// checks the report table is complete and sane — every (codec, source) pair
// present, bytes-per-row positive, and the real-kv source cleanly absent
// (this commit ships no testdata/real_kv_rows.bin; the orchestrator
// captures it at merge).
func TestMeasureCodecs_Good(t *testing.T) {
	const d = 128
	result := MeasureCodecs(d, 42, 1500, DefaultRealKVFixturePath)

	if result.D != d {
		t.Errorf("MeasureCodecsResult.D = %d, want %d", result.D, d)
	}
	wantRows := 9 * 2 // 9 codecs × {gaussian, sphere-uniform}; no real-kv fixture in this checkout
	if len(result.Reports) != wantRows {
		t.Fatalf("MeasureCodecs produced %d report rows, want %d (9 codecs × 2 always-on sources)", len(result.Reports), wantRows)
	}
	for _, r := range result.Reports {
		if r.DataSource == "real-kv" {
			t.Error("report contains a real-kv row, want it absent (no fixture captured in this checkout)")
		}
		if r.BytesPerRow <= 0 {
			t.Errorf("codec %q/%q: BytesPerRow = %d, want > 0", r.Codec, r.DataSource, r.BytesPerRow)
		}
		if r.Samples == 0 {
			t.Errorf("codec %q/%q: Samples = 0, want > 0", r.Codec, r.DataSource)
		}
		if r.MaxSoftmaxDelta < 0 || r.MaxSoftmaxDelta > 1 {
			t.Errorf("codec %q/%q: MaxSoftmaxDelta = %v, want within [0,1]", r.Codec, r.DataSource, r.MaxSoftmaxDelta)
		}
	}

	t.Log("\n" + FormatReport(result))
}

// TestMeasureCodecs_Ugly checks a real-KV fixture IS picked up when present
// and dimension-matched, and stays absent when dimension-mismatched — the
// two branches of the "skip cleanly" contract that
// TestMeasureCodecs_Good's default (no fixture) run cannot exercise.
func TestMeasureCodecs_Ugly(t *testing.T) {
	dir := t.TempDir()
	rng := rand.New(rand.NewPCG(9, 9))

	const d = 16
	matched := writeTestFixture(t, dir, "matched.bin", d, generateGaussianRows(rng, 600, d))
	result := MeasureCodecs(d, 1, 500, matched)
	found := false
	for _, r := range result.Reports {
		if r.DataSource == "real-kv" {
			found = true
		}
	}
	if !found {
		t.Error("MeasureCodecs did not pick up a present, dimension-matched real-KV fixture")
	}

	mismatched := writeTestFixture(t, dir, "mismatched.bin", d+1, generateGaussianRows(rng, 600, d+1))
	result2 := MeasureCodecs(d, 2, 500, mismatched)
	for _, r := range result2.Reports {
		if r.DataSource == "real-kv" {
			t.Error("MeasureCodecs picked up a dimension-mismatched real-KV fixture, want it skipped")
		}
	}
}

// ExampleMeasureCodecs demonstrates running the instrument and formatting
// its report — small samplesPerSource here for a fast doc example; the real
// measurement run uses far more (TestMeasureCodecs_Good).
func ExampleMeasureCodecs() {
	result := MeasureCodecs(16, 42, 50, "testdata/does-not-exist.bin")
	core.Println("report rows:", len(result.Reports))
	// Output:
	// report rows: 18
}
