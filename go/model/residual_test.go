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
