// SPDX-Licence-Identifier: EUPL-1.2

package mistral_test

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/mistral"
)

// the real Ministral-3-3B-Base-2512 config shape: the multimodal wrapper (text arch nested
// under text_config), YaRN rope, full attention (sliding_window null), tied embeddings.
const ministral3B = `{
  "model_type": "mistral3",
  "architectures": ["Mistral3ForConditionalGeneration"],
  "vision_config": {"hidden_size": 1024},
  "text_config": {
    "model_type": "ministral3",
    "hidden_size": 3072, "num_hidden_layers": 26,
    "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
    "intermediate_size": 9216, "vocab_size": 131072, "rms_norm_eps": 1e-05,
    "sliding_window": null, "tie_word_embeddings": true,
    "rope_parameters": {"rope_type": "yarn", "rope_theta": 1000000.0, "factor": 16.0,
      "beta_fast": 32.0, "beta_slow": 1.0, "original_max_position_embeddings": 16384}
  }
}`

func TestConfigArchMinistral3B(t *testing.T) {
	var cfg mistral.Config
	if r := core.JSONUnmarshal([]byte(ministral3B), &cfg); !r.OK {
		t.Fatalf("unmarshal: %s", r.Error())
	}
	arch, err := cfg.Arch() // resolves the text_config wrapper
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	// neutral transformer dims lifted from the nested text_config
	if arch.Hidden != 3072 || arch.Heads != 32 || arch.KVHeads != 8 || arch.HeadDim != 128 {
		t.Fatalf("dims: hidden %d heads %d kv %d headDim %d", arch.Hidden, arch.Heads, arch.KVHeads, arch.HeadDim)
	}
	if arch.FF != 9216 || arch.Vocab != 131072 || arch.Eps != 1e-5 {
		t.Fatalf("ff %d vocab %d eps %g", arch.FF, arch.Vocab, arch.Eps)
	}
	// full rotary (RotaryDim == HeadDim), rope_theta as the base, no scaling knob
	if arch.RotaryDim != 128 || arch.RotaryDimLocal != 128 {
		t.Fatalf("partial rotary leaked: %d/%d (want full 128)", arch.RotaryDim, arch.RotaryDimLocal)
	}
	if arch.RopeBase != 1_000_000 || arch.RopeScale != 1 {
		t.Fatalf("rope base %g scale %g", arch.RopeBase, arch.RopeScale)
	}
	// every gemma4-specific extra OFF
	if arch.SlidingWindow != 0 {
		t.Fatalf("sliding window leaked: %d (Ministral is full attention)", arch.SlidingWindow)
	}
	if arch.SoftCap != 0 {
		t.Fatalf("soft-cap leaked: %g", arch.SoftCap)
	}
	if arch.HasMoE() {
		t.Fatal("MoE leaked into a dense Mistral arch")
	}
	if arch.PerLayerInputHidden != 0 || arch.PerLayerInputVocab != 0 {
		t.Fatal("per-layer-input tower leaked into Mistral")
	}
	// 26 layers, all full attention, each owning its own KV cache (no sliding, no KV-share)
	if len(arch.Layer) != 26 {
		t.Fatalf("layers: %d (want 26)", len(arch.Layer))
	}
	for i, l := range arch.Layer {
		if l.Attention != model.GlobalAttention {
			t.Fatalf("layer %d not full attention", i)
		}
		if !l.OwnsCache() || l.CacheIndex != i || l.KVShareFrom != i {
			t.Fatalf("layer %d not a cache owner: cacheIdx %d shareFrom %d", i, l.CacheIndex, l.KVShareFrom)
		}
		if l.MoE {
			t.Fatalf("layer %d marked MoE", i)
		}
	}
	// rope_type "yarn" → the arch carries the YaRN inv-freqs (full rotary, so
	// head_dim/2 of them), matching the standalone computation.
	if len(arch.RopeFreqs) != 128/2 {
		t.Fatalf("RopeFreqs len %d, want 64 (YaRN, head_dim/2)", len(arch.RopeFreqs))
	}
	wantFreqs := mistral.YaRNInvFreqs(1_000_000, 16, 32, 1, 16384, 128)
	for i := range wantFreqs {
		if relDiff(arch.RopeFreqs[i], wantFreqs[i]) > 1e-5 {
			t.Fatalf("RopeFreqs[%d]=%g != YaRNInvFreqs %g", i, arch.RopeFreqs[i], wantFreqs[i])
		}
	}
}

// TestConfigArchNoYaRN_Good proves a non-yarn rope leaves RopeFreqs nil (the
// backend then derives uniformly from RopeBase) — the dense families are
// unaffected.
func TestConfigArchNoYaRN_Good(t *testing.T) {
	const cfg = `{"hidden_size":256,"num_hidden_layers":2,"num_attention_heads":4,"head_dim":64,` +
		`"rope_parameters":{"rope_type":"default","rope_theta":1000000}}`
	var c mistral.Config
	if r := core.JSONUnmarshal([]byte(cfg), &c); !r.OK {
		t.Fatalf("unmarshal: %s", r.Error())
	}
	arch, err := c.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if arch.RopeFreqs != nil {
		t.Fatalf("default rope must not set RopeFreqs, got len %d", len(arch.RopeFreqs))
	}
	if arch.RopeBase != 1_000_000 {
		t.Errorf("ropeBase %g, want 1e6", arch.RopeBase)
	}
}

func TestConfigArchDefaults(t *testing.T) {
	// minimal config: head_dim, num_key_value_heads, eps, rope all absent → derived defaults.
	const minimal = `{"hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 8, "intermediate_size": 128, "vocab_size": 100}`
	var cfg mistral.Config
	if r := core.JSONUnmarshal([]byte(minimal), &cfg); !r.OK {
		t.Fatalf("unmarshal: %s", r.Error())
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if arch.HeadDim != 8 { // hidden/heads = 64/8
		t.Fatalf("headDim default %d (want 8)", arch.HeadDim)
	}
	if arch.KVHeads != 8 { // == heads when absent (MHA)
		t.Fatalf("kvHeads default %d (want 8)", arch.KVHeads)
	}
	if arch.Eps != 1e-5 {
		t.Fatalf("eps default %g (want 1e-5)", arch.Eps)
	}
	if arch.RopeBase != 1_000_000 {
		t.Fatalf("rope base default %g (want 1e6)", arch.RopeBase)
	}
	if arch.RotaryDim != 8 { // full rotary == headDim
		t.Fatalf("rotaryDim %d (want 8)", arch.RotaryDim)
	}
}

func TestConfigArchErrors(t *testing.T) {
	for _, tc := range []struct {
		name, json string
	}{
		{"no hidden", `{"num_hidden_layers": 2, "num_attention_heads": 8}`},
		{"no layers", `{"hidden_size": 64, "num_attention_heads": 8}`},
		{"heads not multiple of kv", `{"hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 8, "num_key_value_heads": 3}`},
		{"headDim absent, hidden indivisible", `{"hidden_size": 65, "num_hidden_layers": 2, "num_attention_heads": 8}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var cfg mistral.Config
			if r := core.JSONUnmarshal([]byte(tc.json), &cfg); !r.OK {
				t.Fatalf("unmarshal: %s", r.Error())
			}
			if _, err := cfg.Arch(); err == nil {
				t.Fatal("expected an error")
			}
		})
	}
}
