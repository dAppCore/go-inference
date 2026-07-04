// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/inference/model"
)

func TestQWMapsModelLinear(t *testing.T) {
	if got := qw(nil); got.Packed != nil || got.GroupSize != 0 || got.Bits != 0 {
		t.Fatalf("qw(nil) = %+v, want zero QuantWeight", got)
	}
	lin := &model.Linear{
		Weight:    []byte{1, 2, 3},
		Scales:    []byte{4, 5},
		Biases:    []byte{6, 7},
		GroupSize: 64,
		Bits:      4,
	}
	got := qw(lin)
	if string(got.Packed) != string(lin.Weight) || string(got.Scales) != string(lin.Scales) || string(got.Biases) != string(lin.Biases) {
		t.Fatalf("qw did not preserve linear byte slices: got %+v", got)
	}
	if got.GroupSize != 64 || got.Bits != 4 {
		t.Fatalf("qw geometry = gs%d bits%d, want gs64 bits4", got.GroupSize, got.Bits)
	}
}

func TestLoadedToQuantRejectsNilModel(t *testing.T) {
	if _, err := loadedToQuant(nil, 64, 4); err == nil {
		t.Fatal("expected loadedToQuant to reject a nil model")
	}
}
