// SPDX-Licence-Identifier: EUPL-1.2

package turboquant

import (
	"math/rand/v2"
	"testing"

	core "dappco.re/go"
)

// TestMeasureReal_Good checks the report shape and the codec-discriminates-
// bit-width invariant (more bits -> less distortion) on synthetic
// sphere-uniform K/V pools — always available, no fixture dependency.
func TestMeasureReal_Good(t *testing.T) {
	const d, n = 32, 300
	rng := rand.New(rand.NewPCG(11, 12))
	keys := generateSphereUniformRows(rng, n, d)
	values := generateSphereUniformRows(rng, n, d)

	result, err := MeasureReal(keys, values, 42)
	if err != nil {
		t.Fatalf("MeasureReal: %v", err)
	}
	if result.D != d {
		t.Errorf("D = %d, want %d", result.D, d)
	}
	wantRows := 3*2 + 2 // {2,3,4} bits x {K,V} + 2 mixed rows
	if len(result.Reports) != wantRows {
		t.Fatalf("Reports = %d rows, want %d", len(result.Reports), wantRows)
	}

	bitsOf := func(side string, bits int) CodecReport {
		name := core.Sprintf("TurboQuant-Qmse-b%d", bits)
		for _, r := range result.Reports {
			if r.Codec == name && r.DataSource == side {
				return r
			}
		}
		t.Fatalf("no report for %s/%s", name, side)
		return CodecReport{}
	}
	for _, side := range []string{"real-kv-K", "real-kv-V"} {
		b2, b4 := bitsOf(side, 2), bitsOf(side, 4)
		if b4.RowMSERelative >= b2.RowMSERelative {
			t.Errorf("%s: 4-bit relative MSE (%v) should be less than 2-bit (%v)", side, b4.RowMSERelative, b2.RowMSERelative)
		}
		if b2.Samples != n {
			t.Errorf("%s: Samples = %d, want %d", side, b2.Samples, n)
		}
	}
	if result.Mixed.Samples != 2*n {
		t.Errorf("Mixed.Samples = %d, want %d (both sides pooled)", result.Mixed.Samples, 2*n)
	}
	if result.Mixed.RowMSERelative <= 0 {
		t.Errorf("Mixed.RowMSERelative = %v, want > 0", result.Mixed.RowMSERelative)
	}

	t.Log("\n" + FormatRealReport(result))
}

// TestMeasureReal_Bad checks the four input-validation error paths: ragged
// key rows, ragged value rows, and an empty side on either input.
func TestMeasureReal_Bad(t *testing.T) {
	good := [][]float32{{1, 2}, {3, 4}}
	ragged := [][]float32{{1, 2}, {3}}

	t.Run("ragged_keys", func(t *testing.T) {
		if _, err := MeasureReal(ragged, good, 1); err == nil {
			t.Fatal("MeasureReal accepted ragged key rows")
		}
	})
	t.Run("ragged_values", func(t *testing.T) {
		if _, err := MeasureReal(good, ragged, 1); err == nil {
			t.Fatal("MeasureReal accepted ragged value rows")
		}
	})
	t.Run("empty_keys", func(t *testing.T) {
		if _, err := MeasureReal(nil, good, 1); err == nil {
			t.Fatal("MeasureReal accepted no key rows")
		}
	})
	t.Run("empty_values", func(t *testing.T) {
		if _, err := MeasureReal(good, nil, 1); err == nil {
			t.Fatal("MeasureReal accepted no value rows")
		}
	})
}

// TestMeasureReal_Ugly runs the ACTUAL captured real-KV fixture (see
// go/kv/turboquant/testdata/, written by engine/metal's live capture
// harness) when present — real cache rows are neither Gaussian nor
// sphere-uniform, the "surprising but valid" data MeasureCodecs' synthetic
// sources never exercise. A fresh checkout without the captured fixture
// skips cleanly (LoadRealKVRows' own "ok=false, no error" contract), so this
// test never blocks a checkout that hasn't run the live capture.
func TestMeasureReal_Ugly(t *testing.T) {
	keys, dk, okK, err := LoadRealKVRows("testdata/real_kv_keys.bin")
	if err != nil {
		t.Fatalf("LoadRealKVRows(keys): %v", err)
	}
	values, dv, okV, err := LoadRealKVRows("testdata/real_kv_values.bin")
	if err != nil {
		t.Fatalf("LoadRealKVRows(values): %v", err)
	}
	if !okK || !okV {
		t.Skip("no captured real-KV fixture in this checkout (go/kv/turboquant/testdata/real_kv_{keys,values}.bin)")
	}
	if dk != dv {
		t.Fatalf("captured key/value dimension mismatch: %d vs %d", dk, dv)
	}

	result, err := MeasureReal(keys, values, 42)
	if err != nil {
		t.Fatalf("MeasureReal(captured real-KV): %v", err)
	}
	if result.D != dk {
		t.Errorf("D = %d, want %d", result.D, dk)
	}
	for _, r := range result.Reports {
		if r.RowMSERelative < 0 {
			t.Errorf("%s/%s: RowMSERelative = %v, want >= 0", r.Codec, r.DataSource, r.RowMSERelative)
		}
	}
	t.Log("\n" + FormatRealReport(result))
}

// TestSaveRealKVRows_Good checks a round trip through LoadRealKVRows
// reproduces the exact rows written.
func TestSaveRealKVRows_Good(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/rows.bin"
	want := [][]float32{{1, 2, 3}, {-4, 5, 6.5}}

	if err := SaveRealKVRows(path, want); err != nil {
		t.Fatalf("SaveRealKVRows: %v", err)
	}
	got, d, ok, err := LoadRealKVRows(path)
	if err != nil {
		t.Fatalf("LoadRealKVRows: %v", err)
	}
	if !ok {
		t.Fatal("LoadRealKVRows(saved fixture).ok = false, want true")
	}
	if d != 3 {
		t.Errorf("d = %d, want 3", d)
	}
	if len(got) != len(want) {
		t.Fatalf("round-tripped %d rows, want %d", len(got), len(want))
	}
	for i := range want {
		for j := range want[i] {
			if got[i][j] != want[i][j] {
				t.Errorf("row %d[%d] = %v, want %v", i, j, got[i][j], want[i][j])
			}
		}
	}
}

// TestSaveRealKVRows_Bad checks empty input and ragged rows both error
// rather than writing a malformed fixture.
func TestSaveRealKVRows_Bad(t *testing.T) {
	dir := t.TempDir()
	t.Run("no_rows", func(t *testing.T) {
		if err := SaveRealKVRows(dir+"/empty.bin", nil); err == nil {
			t.Fatal("SaveRealKVRows accepted no rows")
		}
	})
	t.Run("ragged_rows", func(t *testing.T) {
		if err := SaveRealKVRows(dir+"/ragged.bin", [][]float32{{1, 2}, {3}}); err == nil {
			t.Fatal("SaveRealKVRows accepted ragged rows")
		}
	})
}

// ExampleMeasureReal demonstrates measuring a captured (here, hand-written)
// K/V row set and formatting its report.
func ExampleMeasureReal() {
	keys := [][]float32{{1, 0}, {0, 1}, {1, 1}, {-1, 0.5}}
	values := [][]float32{{2, 0}, {0, 2}, {1, -1}, {0.5, 0.5}}
	result, err := MeasureReal(keys, values, 42)
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("report rows:", len(result.Reports))
	// Output:
	// report rows: 8
}
