// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"slices"
	"testing"

	"dappco.re/go/inference/model/safetensors"
)

func TestFusedQKV_SplitContiguousGateUp_Good(t *testing.T) {
	in := map[string]safetensors.Tensor{"f.weight": {Dtype: "F32", Shape: []int{4, 2}, Data: []byte{1, 2, 3, 4, 5, 6, 7, 8}}}
	out := SplitContiguousGateUp(in, "f", "g", "u")
	if out["g.weight"].Shape[0] != 2 || out["u.weight"].Data[0] != 5 {
		t.Fatalf("split = %+v", out)
	}
}
func TestFusedQKV_SplitContiguousGateUp_Bad(t *testing.T) {
	in := map[string]safetensors.Tensor{"f.weight": {Shape: []int{3, 2}, Data: []byte{1, 2, 3}}}
	if out := SplitContiguousGateUp(in, "f", "g", "u"); len(out) != 1 {
		t.Fatal("malformed tensor split")
	}
}
func TestFusedQKV_SplitContiguousGateUp_Ugly(t *testing.T) {
	in := map[string]safetensors.Tensor{}
	if out := SplitContiguousGateUp(in, "", "", ""); len(out) != 0 {
		t.Fatal("empty map changed")
	}
}

func TestSplitInterleavedQKV_Good(t *testing.T) {
	in := map[string]safetensors.Tensor{"f.weight": {Dtype: "U8", Shape: []int{6, 1}, Data: []byte{1, 2, 3, 4, 5, 6}}}
	got := SplitInterleavedQKV(in, "f", "q", "k", "v", 2, 1)
	if !slices.Equal(got["q.weight"].Data, []byte{1, 4}) || !slices.Equal(got["k.weight"].Data, []byte{2, 5}) || !slices.Equal(got["v.weight"].Data, []byte{3, 6}) {
		t.Fatalf("interleaved split = q%v k%v v%v", got["q.weight"].Data, got["k.weight"].Data, got["v.weight"].Data)
	}
}

func TestSplitContiguousQKV_Good(t *testing.T) {
	in := map[string]safetensors.Tensor{"f.weight": {Dtype: "U8", Shape: []int{4, 1}, Data: []byte{1, 2, 3, 4}}}
	got := SplitContiguousQKV(in, "f", "q", "k", "v", 2, 1)
	if !slices.Equal(got["q.weight"].Data, []byte{1, 2}) || got["k.weight"].Data[0] != 3 || got["v.weight"].Data[0] != 4 {
		t.Fatalf("contiguous split = q%v k%v v%v", got["q.weight"].Data, got["k.weight"].Data, got["v.weight"].Data)
	}
}

func TestSplitGroupedQKV_Good(t *testing.T) {
	in := map[string]safetensors.Tensor{"f.weight": {Dtype: "U8", Shape: []int{8, 1}, Data: []byte{1, 2, 3, 4, 5, 6, 7, 8}}}
	got := SplitGroupedQKV(in, "f", "q", "k", "v", 4, 2, 1)
	if !slices.Equal(got["q.weight"].Data, []byte{1, 2, 5, 6}) || !slices.Equal(got["k.weight"].Data, []byte{3, 7}) || !slices.Equal(got["v.weight"].Data, []byte{4, 8}) {
		t.Fatalf("grouped split = q%v k%v v%v", got["q.weight"].Data, got["k.weight"].Data, got["v.weight"].Data)
	}
}
