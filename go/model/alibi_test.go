// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"math"
	"testing"
)

func TestALiBiSlopes_Golden(t *testing.T) {
	want := []float32{0.25, 0.0625, 0.015625, 0.00390625, 0.5, 0.125}
	got := ALiBiSlopes(6)
	if len(got) != len(want) {
		t.Fatalf("ALiBiSlopes(6) length = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("slope[%d] = %g, want %g", i, got[i], want[i])
		}
	}
}

func TestALiBiSlopes_Bad(t *testing.T) {
	if got := ALiBiSlopes(-1); got != nil {
		t.Fatalf("ALiBiSlopes(-1) = %v, want nil", got)
	}
}

func TestApplyALiBi_Golden(t *testing.T) {
	scores := []float64{1, 2, 3, 4}
	ApplyALiBi(scores, 0.5, 3, 0)
	want := []float64{-0.5, 1, 2.5, 4}
	for i := range want {
		if math.Abs(scores[i]-want[i]) > 1e-12 {
			t.Errorf("scores[%d] = %g, want %g", i, scores[i], want[i])
		}
	}
}

func TestApplyALiBi_Ugly(t *testing.T) {
	ApplyALiBi(nil, 1, 0, 0)
}
