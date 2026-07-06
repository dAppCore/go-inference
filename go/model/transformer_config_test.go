// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"testing"

	core "dappco.re/go"
)

// TestTransformerConfig_JSON_Good covers the config.json round-trip every arch's
// embedded TransformerConfig relies on: the JSON tags read the HF field names into
// the neutral dims, and marshalling back preserves them.
func TestTransformerConfig_JSON_Good(t *testing.T) {
	const in = `{"model_type":"faketest","hidden_size":2048,"num_hidden_layers":24,` +
		`"intermediate_size":8192,"num_attention_heads":8,"num_key_value_heads":2,` +
		`"head_dim":256,"vocab_size":262144,"rms_norm_eps":1e-6,"max_position_embeddings":32768}`
	var cfg TransformerConfig
	if r := core.JSONUnmarshal([]byte(in), &cfg); !r.OK {
		t.Fatalf("JSONUnmarshal: %s", r.Error())
	}
	if cfg.ModelType != "faketest" || cfg.HiddenSize != 2048 || cfg.NumHiddenLayers != 24 ||
		cfg.IntermediateSize != 8192 || cfg.NumAttentionHeads != 8 || cfg.NumKeyValueHeads != 2 ||
		cfg.HeadDim != 256 || cfg.VocabSize != 262144 || cfg.MaxPositionEmbeddings != 32768 {
		t.Fatalf("TransformerConfig = %+v, want every field parsed from its HF JSON key", cfg)
	}
	if cfg.RMSNormEps != 1e-6 {
		t.Fatalf("RMSNormEps = %v, want 1e-6", cfg.RMSNormEps)
	}
	r := core.JSONMarshal(cfg)
	if !r.OK {
		t.Fatalf("JSONMarshal: %s", r.Error())
	}
	var round TransformerConfig
	if r := core.JSONUnmarshal(r.Value.([]byte), &round); !r.OK {
		t.Fatalf("JSONUnmarshal(round-trip): %s", r.Error())
	}
	if round != cfg {
		t.Fatalf("round-tripped TransformerConfig = %+v, want %+v", round, cfg)
	}
}

// TestTransformerConfig_JSON_Bad covers a config.json that omits every field: the
// struct decodes to its zero value rather than erroring — an arch's own parser is
// responsible for validating required dims, not the neutral embed itself.
func TestTransformerConfig_JSON_Bad(t *testing.T) {
	var cfg TransformerConfig
	if r := core.JSONUnmarshal([]byte(`{}`), &cfg); !r.OK {
		t.Fatalf("JSONUnmarshal({}): %s", r.Error())
	}
	if cfg != (TransformerConfig{}) {
		t.Fatalf("TransformerConfig from {} = %+v, want the zero value", cfg)
	}
}

// TestTransformerConfig_JSON_Ugly covers malformed JSON: the decode itself must
// fail cleanly (a core.Result error) rather than leaving a partially-populated
// struct silently.
func TestTransformerConfig_JSON_Ugly(t *testing.T) {
	var cfg TransformerConfig
	if r := core.JSONUnmarshal([]byte(`not json`), &cfg); r.OK {
		t.Fatalf("JSONUnmarshal(malformed) = ok, want an error")
	}
}
