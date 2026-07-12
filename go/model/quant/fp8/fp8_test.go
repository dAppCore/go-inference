// SPDX-Licence-Identifier: EUPL-1.2
package fp8

import (
	"math"
	"testing"
)

func TestQuantize_Good(t *testing.T) {
	values := []float32{-2, -1, -.1, 0, .1, 1, 2}
	q, err := Quantize(values)
	if err != nil {
		t.Fatal(err)
	}
	got, err := Dequantize(q)
	if err != nil {
		t.Fatal(err)
	}
	for i := range values {
		if math.Abs(float64(got[i]-values[i])) > float64(q.Scale)*32 {
			t.Fatalf("value %d error", i)
		}
	}
}
func TestQuantize_Bad(t *testing.T) {
	if _, err := Quantize(nil); err == nil {
		t.Fatal("Quantize(nil) error = nil")
	}
}
func TestQuantize_Ugly(t *testing.T) {
	if _, err := Quantize([]float32{float32(math.NaN())}); err == nil {
		t.Fatal("Quantize(NaN) error = nil")
	}
}
func TestDequantize_Good(t *testing.T) {
	got, err := Dequantize(Tensor{Data: []byte{0x38}, Scale: 1})
	if err != nil || got[0] != 1 {
		t.Fatalf("Dequantize = %v, %v", got, err)
	}
}
func TestDequantize_Bad(t *testing.T) {
	if _, err := Dequantize(Tensor{}); err == nil {
		t.Fatal("Dequantize(empty) error = nil")
	}
}
func TestDequantize_Ugly(t *testing.T) {
	if _, err := Dequantize(Tensor{Data: []byte{1}, Scale: -1}); err == nil {
		t.Fatal("Dequantize(negative scale) error = nil")
	}
}
