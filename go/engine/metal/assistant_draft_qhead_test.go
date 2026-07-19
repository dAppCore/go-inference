// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"
)

func TestQuantiseAffine4RowsBF16_Good(t *testing.T) {
	const rows, cols = 8, 128
	w := toBF16Bytes(syntheticFloat32(rows*cols, 4409))
	packed, scales, biases, err := quantiseAffine4RowsBF16(w, rows, cols)
	if err != nil {
		t.Fatalf("quantiseAffine4RowsBF16: %v", err)
	}
	if want := rows * cols / 2; len(packed) != want {
		t.Fatalf("packed bytes = %d, want %d", len(packed), want)
	}
	if want := rows * (cols / assistantQHeadGroupSize) * bf16Size; len(scales) != want || len(biases) != want {
		t.Fatalf("scales/biases bytes = %d/%d, want %d", len(scales), len(biases), want)
	}
	// round-trip through the package's affine oracle: every element must
	// reconstruct within half a quantisation step of its group (plus bf16
	// storage rounding on the source values).
	got, err := dequantizeAffineRowsF32(packed, scales, biases, rows, cols, assistantQHeadGroupSize, 4)
	if err != nil {
		t.Fatalf("dequantizeAffineRowsF32: %v", err)
	}
	for r := range rows {
		sRow := scales[r*(cols/assistantQHeadGroupSize)*bf16Size:]
		for c := range cols {
			src := bf16ToF32(w[(r*cols+c)*bf16Size], w[(r*cols+c)*bf16Size+1])
			g := c / assistantQHeadGroupSize
			scale := bf16ToF32(sRow[g*bf16Size], sRow[g*bf16Size+1])
			tol := float64(scale)/2 + 1e-2
			if diff := math.Abs(float64(got[r*cols+c] - src)); diff > tol {
				t.Fatalf("row %d col %d: reconstructed %f vs source %f (|diff| %f > tol %f)", r, c, got[r*cols+c], src, diff, tol)
			}
		}
	}
}

func TestQuantiseAffine4RowsBF16_FlatGroup_Good(t *testing.T) {
	// a flat group (scale 0) must code 0 everywhere and carry the value in the
	// bias exactly (up to bf16 storage).
	const rows, cols = 1, assistantQHeadGroupSize
	vals := make([]float32, cols)
	for i := range vals {
		vals[i] = 0.8125 // exactly representable in bf16
	}
	w := toBF16Bytes(vals)
	packed, scales, biases, err := quantiseAffine4RowsBF16(w, rows, cols)
	if err != nil {
		t.Fatalf("quantiseAffine4RowsBF16 flat: %v", err)
	}
	got, err := dequantizeAffineRowsF32(packed, scales, biases, rows, cols, assistantQHeadGroupSize, 4)
	if err != nil {
		t.Fatalf("dequantizeAffineRowsF32 flat: %v", err)
	}
	for c := range cols {
		if got[c] != 0.8125 {
			t.Fatalf("flat group col %d = %f, want 0.8125 exactly", c, got[c])
		}
	}
}

func TestQuantiseAffine4RowsBF16_Bad(t *testing.T) {
	w := toBF16Bytes(syntheticFloat32(2*assistantQHeadGroupSize, 5003))
	if _, _, _, err := quantiseAffine4RowsBF16(w, 0, assistantQHeadGroupSize); err == nil {
		t.Fatal("accepted zero rows")
	}
	if _, _, _, err := quantiseAffine4RowsBF16(w, 1, assistantQHeadGroupSize+1); err == nil {
		t.Fatal("accepted a column count off the group size")
	}
	if _, _, _, err := quantiseAffine4RowsBF16(w[:2], 2, assistantQHeadGroupSize); err == nil {
		t.Fatal("accepted a short matrix")
	}
}

func TestAffine4Code_Good(t *testing.T) {
	if got := affine4Code(1.0, 0, 1); got != 1 {
		t.Fatalf("affine4Code(1,0,1) = %d, want 1", got)
	}
	if got := affine4Code(99, 0, 1); got != 15 {
		t.Fatalf("clamp high = %d, want 15", got)
	}
	if got := affine4Code(-99, 0, 1); got != 0 {
		t.Fatalf("clamp low = %d, want 0", got)
	}
	if got := affine4Code(123, 0, 0); got != 0 {
		t.Fatalf("flat group code = %d, want 0", got)
	}
}
