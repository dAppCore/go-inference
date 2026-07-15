// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

// TestAddOutputProjBias proves the audio tower's output_proj.bias is applied to every projected row
// (defect-catcher: the earlier port dropped it, corrupting every clip — the e2b bias is non-negligible,
// max|abs| 14.875). A nil/short bias must leave the rows untouched so bias-free packs stay byte-identical.
func TestAddOutputProjBias(t *testing.T) {
	const rows, outDim = 3, 4
	biasVals := []float32{1, 2, 3, 4} // exactly representable in BF16 → exact round-trip
	bias := f32ToBf16Slice(biasVals)

	out := make([]float32, rows*outDim) // start from zeros so each cell must equal its bias
	addOutputProjBias(out, bias, rows, outDim)
	for r := range rows {
		for c := range outDim {
			if got := out[r*outDim+c]; got != biasVals[c] {
				t.Fatalf("row %d col %d = %v, want %v (bias not applied)", r, c, got, biasVals[c])
			}
		}
	}

	// nil bias is a no-op — the rows stay byte-identical.
	base := []float32{0.5, -1, 2.5, 4, 0, -0.25, 8, 1.5, -3, 6, 0.75, -2}
	ref := append([]float32(nil), base...)
	addOutputProjBias(base, nil, rows, outDim)
	for i := range base {
		if base[i] != ref[i] {
			t.Fatalf("nil bias mutated index %d: %v != %v", i, base[i], ref[i])
		}
	}
}
