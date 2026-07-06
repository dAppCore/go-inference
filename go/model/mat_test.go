// SPDX-Licence-Identifier: EUPL-1.2

package model

import "testing"

// TestMat_MatNT_Good covers the ordinary [M×K]·[N×K]ᵀ case: a known 2×3 input against a
// 2×3 weight (N=2 output rows) checked against hand-computed dot products.
func TestMat_MatNT_Good(t *testing.T) {
	// in is [2×3]: row0=[1,2,3], row1=[4,5,6]
	in := []float32{1, 2, 3, 4, 5, 6}
	// w is [2×3] (transposed layout: 2 output rows, 3 cols each): row0=[1,0,0], row1=[0,1,0]
	w := []float32{1, 0, 0, 0, 1, 0}
	got := MatNT(in, w, 2, 3, 2)
	want := []float32{1, 2, 4, 5} // out[m][n] = in[m]·w[n]
	if len(got) != len(want) {
		t.Fatalf("MatNT len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("MatNT[%d] = %v, want %v (full %v)", i, got[i], want[i], got)
		}
	}
}

// TestMat_MatNT_Bad covers a degenerate M=0 (no rows to project): the loop bodies never
// run and MatNT returns a correctly-sized (empty) output rather than panicking.
func TestMat_MatNT_Bad(t *testing.T) {
	got := MatNT(nil, []float32{1, 2}, 0, 2, 1)
	if len(got) != 0 {
		t.Fatalf("MatNT(M=0) = %v, want empty", got)
	}
}

// TestMat_MatNT_Ugly covers K=0 (an empty contraction dimension): every dot product
// accumulates zero terms, so the output is all zeros at the requested M×N shape — the
// arithmetic degenerates cleanly rather than indexing out of range.
func TestMat_MatNT_Ugly(t *testing.T) {
	got := MatNT([]float32{}, []float32{}, 2, 0, 2)
	want := []float32{0, 0, 0, 0}
	if len(got) != len(want) {
		t.Fatalf("MatNT(K=0) len = %d, want %d", len(got), len(want))
	}
	for i, v := range got {
		if v != 0 {
			t.Fatalf("MatNT(K=0)[%d] = %v, want 0", i, v)
		}
	}
}
