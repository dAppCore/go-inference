// SPDX-Licence-Identifier: EUPL-1.2

package awq

import (
	"math"
	"testing"
)

func TestQuantize_Good(t *testing.T) {
	const rows, columns = 32, 64
	values := make([]float32, rows*columns)
	for i := range values {
		values[i] = 0.025*float32(i%17-8) + 0.0001*float32(i)
	}
	got, err := Quantize(values, rows, columns, Options{Bits: 4, GroupSize: 32, ZeroPoint: true})
	if err != nil {
		t.Fatalf("Quantize() error = %v", err)
	}
	if got.QWeightShape != [2]int{64, 4} || got.QZerosShape != [2]int{2, 4} || got.ScalesShape != [2]int{2, 32} {
		t.Fatalf("Quantize() shapes = qweight %v qzeros %v scales %v", got.QWeightShape, got.QZerosShape, got.ScalesShape)
	}
	dequantized, err := Dequantize(got)
	if err != nil {
		t.Fatalf("Dequantize() error = %v", err)
	}
	var maxError float64
	for i := range values {
		maxError = math.Max(maxError, math.Abs(float64(values[i]-dequantized[i])))
	}
	if maxError > float64(got.MaxScale) {
		t.Fatalf("maximum error %g exceeds one quantisation step %g", maxError, got.MaxScale)
	}
}

func TestQuantize_Bad(t *testing.T) {
	if _, err := Quantize(make([]float32, 31*64), 31, 64, Options{Bits: 4, GroupSize: 32, ZeroPoint: true}); err == nil {
		t.Fatal("Quantize() error = nil, want row packing diagnostic")
	}
}

func TestQuantize_Ugly(t *testing.T) {
	if _, err := Quantize(nil, 0, 0, Options{}); err == nil {
		t.Fatal("Quantize(nil) error = nil")
	}
}

func TestDequantize_Bad(t *testing.T) {
	if _, err := Dequantize(Tensor{}); err == nil {
		t.Fatal("Dequantize(Tensor{}) error = nil")
	}
}
