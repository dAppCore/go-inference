// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"dappco.re/go/inference/model/safetensors"
	"testing"
)

func TestTransposeTensor2D_Golden(t *testing.T) {
	got, err := TransposeTensor2D(safetensors.Tensor{Dtype: "U8", Shape: []int{2, 3}, Data: []byte{1, 2, 3, 4, 5, 6}})
	if err != nil {
		t.Fatal(err)
	}
	want := []byte{1, 4, 2, 5, 3, 6}
	for i := range want {
		if got.Data[i] != want[i] {
			t.Fatalf("data[%d]=%d want %d", i, got.Data[i], want[i])
		}
	}
	if got.Shape[0] != 3 || got.Shape[1] != 2 {
		t.Fatalf("shape=%v", got.Shape)
	}
}

func TestTransposeTensor2D_Bad(t *testing.T) {
	if _, err := TransposeTensor2D(safetensors.Tensor{Shape: []int{2}}); err == nil {
		t.Fatal("rank-one accepted")
	}
}
