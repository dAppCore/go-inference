// SPDX-Licence-Identifier: EUPL-1.2

package bert

import (
	"math"
	"testing"
)

// TestPool_Good_CLS returns the first token's hidden state as a copy.
func TestPool_Good_CLS(t *testing.T) {
	hidden := [][]float32{{1, 2, 3}, {4, 5, 6}}
	got, err := pool(PoolingCLS, hidden)
	if err != nil {
		t.Fatalf("pool: %v", err)
	}
	want := []float32{1, 2, 3}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("pool cls = %v, want %v", got, want)
		}
	}
	// Mutating the result must not touch the source hidden state.
	got[0] = 99
	if hidden[0][0] != 1 {
		t.Fatal("pool cls returned a view, not a copy")
	}
}

// TestPool_Good_Mean averages across tokens element-wise.
func TestPool_Good_Mean(t *testing.T) {
	hidden := [][]float32{{1, 2}, {3, 4}, {5, 6}}
	got, err := pool(PoolingMean, hidden)
	if err != nil {
		t.Fatalf("pool: %v", err)
	}
	want := []float32{3, 4}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Fatalf("pool mean = %v, want %v", got, want)
		}
	}
}

// TestPool_Bad_Empty errors when there is nothing to pool.
func TestPool_Bad_Empty(t *testing.T) {
	if _, err := pool(PoolingCLS, nil); err == nil {
		t.Fatal("expected an error pooling an empty hidden state")
	}
}

// TestPool_Ugly_UnknownMode refuses an unrecognised pooling mode loudly.
func TestPool_Ugly_UnknownMode(t *testing.T) {
	if _, err := pool(Pooling("max"), [][]float32{{1}}); err == nil {
		t.Fatal("expected an error for an unknown pooling mode")
	}
}

// TestL2Normalise_Good scales a vector to unit length.
func TestL2Normalise_Good(t *testing.T) {
	vec := []float32{3, 4}
	l2Normalise(vec)
	if math.Abs(float64(vec[0]-0.6)) > 1e-6 || math.Abs(float64(vec[1]-0.8)) > 1e-6 {
		t.Fatalf("l2Normalise = %v, want [0.6 0.8]", vec)
	}
}

// TestL2Normalise_Ugly_Zero leaves a zero vector untouched instead of dividing by zero.
func TestL2Normalise_Ugly_Zero(t *testing.T) {
	vec := []float32{0, 0, 0}
	l2Normalise(vec)
	for _, v := range vec {
		if v != 0 {
			t.Fatalf("l2Normalise of zero vector = %v, want zeros", vec)
		}
	}
}
