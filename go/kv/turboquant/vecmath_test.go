// SPDX-Licence-Identifier: EUPL-1.2

package turboquant

import (
	"math"
	"testing"
)

const testEpsilon = 1e-9

func approxEqual(a, b, eps float64) bool { return math.Abs(a-b) <= eps }

// TestToFloat64_Good widens a float32 row and checks values survive exactly
// for the small integers/halves used across this package's tests.
func TestToFloat64_Good(t *testing.T) {
	got := toFloat64([]float32{1, -2, 0.5})
	want := []float64{1, -2, 0.5}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("toFloat64()[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

// TestToFloat32_Good narrows a float64 row back to float32.
func TestToFloat32_Good(t *testing.T) {
	got := toFloat32([]float64{1, -2, 0.5})
	want := []float32{1, -2, 0.5}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("toFloat32()[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

// TestL2Norm_Good checks the classic 3-4-5 triangle.
func TestL2Norm_Good(t *testing.T) {
	if got := l2Norm([]float64{3, 4}); got != 5 {
		t.Errorf("l2Norm({3,4}) = %v, want 5", got)
	}
}

// TestL2Norm_Ugly checks the zero vector norms to exactly 0.
func TestL2Norm_Ugly(t *testing.T) {
	if got := l2Norm([]float64{0, 0, 0}); got != 0 {
		t.Errorf("l2Norm(zero) = %v, want 0", got)
	}
}

// TestScaled_Good checks element-wise scaling and that the input is not
// mutated.
func TestScaled_Good(t *testing.T) {
	x := []float64{1, 2}
	got := scaled(x, 0.5)
	if got[0] != 0.5 || got[1] != 1 {
		t.Errorf("scaled({1,2}, 0.5) = %v, want {0.5,1}", got)
	}
	if x[0] != 1 || x[1] != 2 {
		t.Errorf("scaled mutated its input: %v", x)
	}
}

// TestSubtract_Good checks element-wise subtraction.
func TestSubtract_Good(t *testing.T) {
	got := subtract([]float64{3, 5}, []float64{1, 1})
	if got[0] != 2 || got[1] != 4 {
		t.Errorf("subtract({3,5},{1,1}) = %v, want {2,4}", got)
	}
}

// TestSubtract_Bad checks the mismatched-length contract panics rather than
// silently truncating.
func TestSubtract_Bad(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("subtract with mismatched lengths did not panic")
		}
	}()
	subtract([]float64{1, 2}, []float64{1})
}

// TestAdd_Good checks element-wise addition.
func TestAdd_Good(t *testing.T) {
	got := add([]float64{1, 2}, []float64{3, 4})
	if got[0] != 4 || got[1] != 6 {
		t.Errorf("add({1,2},{3,4}) = %v, want {4,6}", got)
	}
}

// TestAdd_Bad checks the mismatched-length contract panics.
func TestAdd_Bad(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("add with mismatched lengths did not panic")
		}
	}()
	add([]float64{1, 2}, []float64{1})
}

// TestDot_Good checks a known inner product.
func TestDot_Good(t *testing.T) {
	if got := dot([]float64{1, 2, 3}, []float64{4, 5, 6}); got != 32 {
		t.Errorf("dot({1,2,3},{4,5,6}) = %v, want 32", got)
	}
}

// TestDot_Bad checks the mismatched-length contract panics.
func TestDot_Bad(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("dot with mismatched lengths did not panic")
		}
	}()
	dot([]float64{1, 2}, []float64{1})
}

// TestSoftmax_Good checks the output sums to 1 and is monotone with the
// input ordering.
func TestSoftmax_Good(t *testing.T) {
	got := softmax([]float64{1, 2, 3})
	var sum float64
	for _, v := range got {
		sum += v
	}
	if !approxEqual(sum, 1, testEpsilon) {
		t.Errorf("softmax({1,2,3}) sums to %v, want 1", sum)
	}
	if !(got[0] < got[1] && got[1] < got[2]) {
		t.Errorf("softmax({1,2,3}) = %v, want monotone increasing", got)
	}
}

// TestSoftmax_Ugly checks a large-magnitude input stays numerically stable
// (no NaN/Inf from the max-subtraction) and an empty input returns empty.
func TestSoftmax_Ugly(t *testing.T) {
	got := softmax([]float64{1000, 1001, 1002})
	for i, v := range got {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("softmax(large) produced non-finite value at %d: %v", i, v)
		}
	}
	if empty := softmax(nil); len(empty) != 0 {
		t.Errorf("softmax(nil) = %v, want empty", empty)
	}
}
