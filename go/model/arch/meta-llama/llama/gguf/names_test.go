// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import "testing"

func TestNames_llamaCanonicalTensorName_Good(t *testing.T) {
	got, err := llamaCanonicalTensorName("model.layers.7.self_attn.q_proj.weight")
	if err != nil || got != "blk.7.attn_q.weight" {
		t.Fatalf("llamaCanonicalTensorName = %q, %v; want blk.7.attn_q.weight, nil", got, err)
	}
}

func TestNames_llamaCanonicalTensorName_Bad(t *testing.T) {
	if _, err := llamaCanonicalTensorName("model.layers.0.unknown.weight"); err == nil {
		t.Fatal("llamaCanonicalTensorName unknown tensor: want error")
	}
}

func TestNames_llamaCanonicalTensorName_Ugly(t *testing.T) {
	if _, err := llamaCanonicalTensorName(""); err == nil {
		t.Fatal("llamaCanonicalTensorName empty tensor: want error")
	}
}

func TestNames_llamaGGUFShape_Good(t *testing.T) {
	got := llamaGGUFShape([]uint64{2048, 1536})
	if len(got) != 2 || got[0] != 1536 || got[1] != 2048 {
		t.Fatalf("llamaGGUFShape = %v, want [1536 2048]", got)
	}
}
