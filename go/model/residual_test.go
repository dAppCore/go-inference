// SPDX-Licence-Identifier: EUPL-1.2

package model

import "testing"

func TestParallelResidual_Good(t *testing.T) {
	r := ParallelResidual([]float32{0.25, -0.5, 1.5}, []float32{1, 2, -3}, []float32{-0.75, 0.125, 4})
	if !r.OK {
		t.Fatal(r.Error())
	}
	want := []float32{0.5, 1.625, 2.5}
	got := r.Value.([]float32)
	for i, v := range got {
		if v != want[i] {
			t.Fatalf("output[%d] = %g, want %g", i, v, want[i])
		}
	}
}

func TestParallelResidual_Bad(t *testing.T) {
	if r := ParallelResidual([]float32{1}, nil, []float32{2}); r.OK {
		t.Fatal("mismatched inputs accepted")
	}
}

func TestParallelResidual_Ugly(t *testing.T) {
	r := ParallelResidual(nil, nil, nil)
	if !r.OK || len(r.Value.([]float32)) != 0 {
		t.Fatalf("empty result = %#v", r)
	}
}

func TestApplyResidualOrder_Good(t *testing.T) {
	norm := func(x []float32) []float32 { return []float32{2 * x[0], 2 * x[1]} }
	attention := func(x []float32) []float32 { return []float32{x[1], x[0]} }
	feedForward := func(x []float32) []float32 { return []float32{x[0] / 2, -x[1] / 2} }
	r := ApplyResidualOrder(NormPlacementPre, []float32{1, 2}, norm, norm, attention, feedForward)
	if !r.OK {
		t.Fatal(r.Error())
	}
	want := []float32{10, 0}
	for i, got := range r.Value.([]float32) {
		if got != want[i] {
			t.Fatalf("pre-norm output[%d] = %g, want %g", i, got, want[i])
		}
	}
}

func TestApplyResidualOrder_Bad(t *testing.T) {
	id := func(x []float32) []float32 { return x }
	if r := ApplyResidualOrder(NormPlacement("middle"), []float32{1}, id, id, id, id); r.OK {
		t.Fatal("undeclared norm placement accepted")
	}
}

func TestApplyResidualOrder_Ugly(t *testing.T) {
	id := func(x []float32) []float32 { return x }
	short := func([]float32) []float32 { return nil }
	if r := ApplyResidualOrder(NormPlacementPost, []float32{1}, id, id, short, id); r.OK {
		t.Fatal("mismatched sublayer output accepted")
	}
	if r := ApplyResidualOrder(NormPlacementPre, []float32{1}, nil, id, id, id); r.OK {
		t.Fatal("nil norm accepted")
	}
	if r := ApplyResidualOrder(NormPlacementPost, []float32{1}, id, id, id, short); r.OK {
		t.Fatal("mismatched feed-forward output accepted")
	}
}
