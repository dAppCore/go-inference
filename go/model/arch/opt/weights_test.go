// SPDX-Licence-Identifier: EUPL-1.2

package opt

import "testing"

// Names are from the public BAAI/OPI-Galactica-6.7B safetensors index, an OPT checkpoint:
// https://huggingface.co/BAAI/OPI-Galactica-6.7B/blob/main/model.safetensors.index.json
func TestWeights_WeightNames_Good(t *testing.T) {
	w := WeightNames()
	want := map[string]string{
		"embed": "model.decoder.embed_tokens", "position": "model.decoder.embed_positions",
		"final_norm": "model.decoder.final_layer_norm.weight", "layer": "model.decoder.layers.%d",
		"attention_norm": ".self_attn_layer_norm.weight", "q": ".self_attn.q_proj",
		"k": ".self_attn.k_proj", "v": ".self_attn.v_proj", "o": ".self_attn.out_proj",
		"mlp_norm": ".final_layer_norm.weight", "up": ".fc1", "down": ".fc2",
	}
	got := map[string]string{"embed": w.Embed, "position": w.PositionEmbed, "final_norm": w.FinalNorm,
		"layer": w.LayerPrefix, "attention_norm": w.AttnNorm, "q": w.Q, "k": w.K, "v": w.V,
		"o": w.O, "mlp_norm": w.MLPNorm, "up": w.Up, "down": w.Down}
	for role, name := range want {
		if got[role] != name {
			t.Fatalf("%s=%q want %q", role, got[role], name)
		}
	}
}

func TestWeights_WeightNames_Bad(t *testing.T) {
	if WeightNames().LMHead != "" {
		t.Fatal("OPT checkpoint unexpectedly declares an independent lm_head")
	}
}

func TestWeights_WeightNames_Ugly(t *testing.T) {
	w := WeightNames()
	if w.EmbedProjectionIn == "" || w.EmbedProjectionOut == "" {
		t.Fatal("projected embedding checkpoints are unsupported")
	}
}
