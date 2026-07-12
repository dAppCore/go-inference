// SPDX-Licence-Identifier: EUPL-1.2

package gptq

import (
	"math"
	"testing"
)

func TestQuantize_Good(t *testing.T) {
	const rows, columns = 32, 64
	values := make([]float32, rows*columns)
	for row := range rows {
		for column := range columns {
			values[row*columns+column] = 0.025*float32((row+column)%17-8) + 0.0001*float32(row*columns+column)
		}
	}
	got, err := Quantize(values, rows, columns, Options{Bits: 4, GroupSize: 32, Symmetric: true})
	if err != nil {
		t.Fatalf("Quantize() error = %v", err)
	}
	if got.QWeightShape != [2]int{8, 32} || got.QZerosShape != [2]int{2, 4} || got.ScalesShape != [2]int{2, 32} || len(got.GIdx) != columns {
		t.Fatalf("Quantize() shapes = qweight %v qzeros %v scales %v g_idx %d", got.QWeightShape, got.QZerosShape, got.ScalesShape, len(got.GIdx))
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
	hessian := make([]float64, columns*columns)
	for i := range columns {
		hessian[i*columns+i] = float64(columns - i)
	}
	ordered, err := Quantize(values, rows, columns, Options{Bits: 4, GroupSize: 32, Symmetric: true, DescAct: true, Hessian: hessian})
	if err != nil || len(ordered.QWeight) != 8*32 {
		t.Fatalf("Quantize(Hessian) = %d words, %v", len(ordered.QWeight), err)
	}
}

func TestQuantize_Bad(t *testing.T) {
	if _, err := Quantize(make([]float32, 31*64), 31, 64, Options{Bits: 4, GroupSize: 32}); err == nil {
		t.Fatal("Quantize() error = nil, want row packing diagnostic")
	}
	if _, err := Quantize(make([]float32, 32*64), 32, 64, Options{Bits: 4, GroupSize: 32, Hessian: []float64{1}}); err == nil {
		t.Fatal("Quantize(bad Hessian) error = nil")
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
