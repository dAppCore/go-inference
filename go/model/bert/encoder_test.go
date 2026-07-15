// SPDX-Licence-Identifier: EUPL-1.2

package bert

import (
	"math"
	"reflect"
	"testing"
)

// TestLinear_Good computes y = x·Wᵀ + b for a [out,in] row-major weight.
func TestLinear_Good(t *testing.T) {
	// x = [1,2,3]; W rows = [[1,0,0],[0,1,1]]; b = [10,20]
	// y = [1*1, 2*1+3*1] + b = [11, 25]
	x := []float32{1, 2, 3}
	weight := []float32{1, 0, 0, 0, 1, 1}
	bias := []float32{10, 20}
	got := linear(x, weight, bias, 3, 2)
	want := []float32{11, 25}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Fatalf("linear = %v, want %v", got, want)
		}
	}
}

// TestLinear_Good_NilBias omits the bias add when bias is nil.
func TestLinear_Good_NilBias(t *testing.T) {
	got := linear([]float32{2, 3}, []float32{1, 1}, nil, 2, 1)
	if math.Abs(float64(got[0]-5)) > 1e-6 {
		t.Fatalf("linear nil bias = %v, want [5]", got)
	}
}

// TestLinearBatch_Good preserves linear's result bits while projecting all rows.
func TestLinearBatch_Good(t *testing.T) {
	x := [][]float32{{1, 2, 3}, {-4, 5, -6}}
	weight := []float32{0.25, -0.5, 0.75, -1.25, 1.5, -1.75}
	bias := []float32{0.125, -0.25}
	got := linearBatch(x, weight, bias, 3, 2)
	for i := range x {
		want := linear(x[i], weight, bias, 3, 2)
		if !reflect.DeepEqual(got[i], want) {
			t.Fatalf("linearBatch row %d = %v, want bit-identical %v", i, got[i], want)
		}
	}
}

// TestLayerNorm_Good normalises to zero mean / unit variance then applies affine.
func TestLayerNorm_Good(t *testing.T) {
	// x = [1,2,3]; mean=2; var=2/3; with identity weight, zero bias.
	x := []float32{1, 2, 3}
	weight := []float32{1, 1, 1}
	bias := []float32{0, 0, 0}
	got := layerNorm(x, weight, bias, 1e-12)
	inv := 1.0 / math.Sqrt(2.0/3.0)
	want := []float64{-inv, 0, inv}
	for i := range want {
		if math.Abs(float64(got[i])-want[i]) > 1e-5 {
			t.Fatalf("layerNorm = %v, want ~%v", got, want)
		}
	}
}

// TestLayerNorm_Good_Affine applies the weight and bias after normalisation.
func TestLayerNorm_Good_Affine(t *testing.T) {
	x := []float32{1, 2, 3}
	got := layerNorm(x, []float32{2, 2, 2}, []float32{1, 1, 1}, 1e-12)
	// middle element normalises to 0, so weight*0 + bias = 1.
	if math.Abs(float64(got[1]-1)) > 1e-5 {
		t.Fatalf("layerNorm affine middle = %v, want 1", got[1])
	}
}

// TestGELU_Good matches the exact erf formula at reference points.
func TestGELU_Good(t *testing.T) {
	cases := map[float32]float64{
		0:  0,
		1:  0.5 * 1 * (1 + math.Erf(1/math.Sqrt2)),
		-1: 0.5 * -1 * (1 + math.Erf(-1/math.Sqrt2)),
	}
	for in, want := range cases {
		if got := gelu(in); math.Abs(float64(got)-want) > 1e-6 {
			t.Fatalf("gelu(%v) = %v, want %v", in, got, want)
		}
	}
}

// TestSoftmax_Good produces a distribution that sums to one.
func TestSoftmax_Good(t *testing.T) {
	scores := []float64{1, 2, 3}
	softmax(scores)
	var sum float64
	for _, s := range scores {
		sum += s
		if s <= 0 {
			t.Fatalf("softmax produced a non-positive weight: %v", scores)
		}
	}
	if math.Abs(sum-1) > 1e-9 {
		t.Fatalf("softmax sum = %v, want 1", sum)
	}
	if !(scores[2] > scores[1] && scores[1] > scores[0]) {
		t.Fatalf("softmax lost ordering: %v", scores)
	}
}

// TestSoftmax_Ugly_LargeValues stays finite under a large score via the max-shift.
func TestSoftmax_Ugly_LargeValues(t *testing.T) {
	scores := []float64{1000, 1001, 1002}
	softmax(scores)
	for _, s := range scores {
		if math.IsNaN(s) || math.IsInf(s, 0) {
			t.Fatalf("softmax overflowed: %v", scores)
		}
	}
}
