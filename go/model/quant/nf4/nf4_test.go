// SPDX-Licence-Identifier: EUPL-1.2
package nf4

import (
	"math"
	"testing"
)

func TestQuantize_Good(t *testing.T) {
	values := make([]float32, 65)
	for i := range values {
		values[i] = float32(i-32) / 32
	}
	q, err := Quantize(values, []int{5, 13})
	if err != nil {
		t.Fatal(err)
	}
	got, err := Dequantize(q)
	if err != nil {
		t.Fatal(err)
	}
	for i := range values {
		if math.Abs(float64(got[i]-values[i])) > .16 {
			t.Fatalf("value %d error %g", i, got[i]-values[i])
		}
	}
}
func TestQuantize_Bad(t *testing.T) {
	if _, err := Quantize(nil, nil); err == nil {
		t.Fatal("Quantize(nil) error = nil")
	}
}
func TestQuantize_Ugly(t *testing.T) {
	if _, err := Quantize([]float32{1}, []int{2}); err == nil {
		t.Fatal("Quantize(shape mismatch) error = nil")
	}
}
func TestDequantize_Good(t *testing.T) {
	got, err := Dequantize(Tensor{Data: []byte{0x0f}, Absmax: []float32{2}, Shape: []int{2}})
	if err != nil || got[0] != -2 || got[1] != 2 {
		t.Fatalf("Dequantize = %v, %v", got, err)
	}
}
func TestDequantize_Bad(t *testing.T) {
	if _, err := Dequantize(Tensor{}); err == nil {
		t.Fatal("Dequantize(empty) error = nil")
	}
}
func TestDequantize_Ugly(t *testing.T) {
	if _, err := Dequantize(Tensor{Data: []byte{0}, Shape: []int{3}}); err == nil {
		t.Fatal("Dequantize(malformed) error = nil")
	}
}
