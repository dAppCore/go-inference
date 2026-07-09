// SPDX-Licence-Identifier: EUPL-1.2

package mistral

import (
	"testing"

	"dappco.re/go/inference/model"
)

// TestMistralRegistered confirms every declared mistral model_type alias resolves to the
// registered spec carrying mistral's two weight-layout overrides — the pre-MLP norm is
// Mistral's post_attention_layernorm and there is no gemma-style post-attention norm — with
// plain RMSNorm (NormBiasOne false), and that the registered Parse round-trips a minimal
// config into an Arch. The qwen3 twin (TestQwen3Registered) pins the same facts for qwen3;
// mistral's registration previously had no coverage at all.
func TestMistralRegistered(t *testing.T) {
	var spec model.ArchSpec
	for _, mt := range []string{"mistral3", "ministral3", "mistral", "ministral"} {
		s, ok := model.LookupArch(mt)
		if !ok {
			t.Fatalf("mistral not registered under model_type %q", mt)
		}
		if s.Weights.MLPNorm != ".post_attention_layernorm.weight" {
			t.Errorf("%s: MLPNorm = %q, want mistral-style post_attention_layernorm", mt, s.Weights.MLPNorm)
		}
		if s.Weights.PostAttnNorm != "" {
			t.Errorf("%s: PostAttnNorm = %q, want empty (mistral has no gemma post-attn norm)", mt, s.Weights.PostAttnNorm)
		}
		if s.Weights.NormBiasOne {
			t.Errorf("%s: NormBiasOne must be false (mistral is plain RMSNorm, not gemma (1+w))", mt)
		}
		spec = s
	}

	if _, err := spec.Parse([]byte(`not json`)); err == nil {
		t.Error("registered Parse must reject malformed config JSON")
	}
	cfg, err := spec.Parse([]byte(`{"model_type":"ministral3","hidden_size":2048,"num_hidden_layers":4,
		"intermediate_size":8192,"num_attention_heads":16,"num_key_value_heads":4,"head_dim":128,
		"vocab_size":131072}`))
	if err != nil {
		t.Fatalf("registered Parse: %v", err)
	}
	a, err := cfg.Arch()
	if err != nil {
		t.Fatalf("registered Arch: %v", err)
	}
	if len(a.Layer) != 4 || a.HeadDim != 128 || a.Hidden != 2048 {
		t.Fatalf("parsed arch wrong: layers=%d headDim=%d hidden=%d", len(a.Layer), a.HeadDim, a.Hidden)
	}
	t.Log("mistral registered: all four aliases resolve, weight overrides + plain RMSNorm pinned, Parse→Arch works")
}
