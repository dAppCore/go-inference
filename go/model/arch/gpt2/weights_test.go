// SPDX-Licence-Identifier: EUPL-1.2

package gpt2

import (
	"dappco.re/go/inference/model/safetensors"
	"testing"
)

// Names are from the public safetensors indexes for openai-community/gpt2,
// AI-Sweden-Models/gpt-sw3-1.3b and bigcode/tiny_starcoder_py respectively.
func TestWeights_NormalizeWeights_Good(t *testing.T) {
	in := map[string]safetensors.Tensor{"transformer.h.0.attn.c_attn.weight": {Dtype: "U8", Shape: []int{2, 6}, Data: []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}}}
	out := NormalizeWeights(in)
	for _, n := range []string{"transformer.h.0.attn.q_proj.weight", "transformer.h.0.attn.k_proj.weight", "transformer.h.0.attn.v_proj.weight"} {
		if out[n].Shape[0] != 2 || out[n].Shape[1] != 2 {
			t.Fatalf("%s=%v", n, out[n].Shape)
		}
	}
}

func TestWeights_NormalizeWeights_Bad(t *testing.T) {
	if len(NormalizeWeights(nil)) != 0 {
		t.Fatal("nil changed")
	}
}
func TestWeights_NormalizeWeights_Ugly(t *testing.T) {
	in := map[string]safetensors.Tensor{"transformer.h.0.attn.c_attn.weight": {Shape: []int{2}}}
	if len(NormalizeWeights(in)) != 1 {
		t.Fatal("malformed tensor should remain untouched")
	}
}
