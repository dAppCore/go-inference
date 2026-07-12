// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"math"
	"testing"
)

func TestLayerNorm_Golden(t *testing.T) {
	x := []float32{1, 2, 3}
	if err := LayerNorm(x, []float32{1, 1, 1}, []float32{0, 0, 0}, 1, 3, 0); err != nil {
		t.Fatal(err)
	}
	w := []float32{-1.2247449, 0, 1.2247449}
	for i := range w {
		if math.Abs(float64(x[i]-w[i])) > 1e-6 {
			t.Fatalf("x[%d]=%v want %v", i, x[i], w[i])
		}
	}
}
func TestLayerNorm_Bad(t *testing.T) {
	if LayerNorm([]float32{1}, nil, nil, 1, 1, 0) == nil {
		t.Fatal("bad affine shape accepted")
	}
}
func TestGELUNew_Golden(t *testing.T) {
	x := []float32{-1, 0, 1}
	GELUNew(x)
	w := []float32{-0.158808, 0, 0.841192}
	for i := range w {
		if math.Abs(float64(x[i]-w[i])) > 1e-5 {
			t.Fatalf("x[%d]=%v", i, x[i])
		}
	}
}
