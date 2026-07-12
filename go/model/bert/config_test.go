// SPDX-Licence-Identifier: EUPL-1.2

package bert

import "testing"

// TestParseConfig_Good decodes a well-formed BertModel config and fills defaults.
func TestParseConfig_Good(t *testing.T) {
	data := []byte(`{
		"model_type": "bert",
		"hidden_size": 384,
		"num_hidden_layers": 12,
		"num_attention_heads": 12,
		"intermediate_size": 1536,
		"vocab_size": 30522,
		"max_position_embeddings": 512,
		"type_vocab_size": 2,
		"layer_norm_eps": 1e-12,
		"hidden_act": "gelu",
		"pad_token_id": 0
	}`)
	cfg, err := ParseConfig(data)
	if err != nil {
		t.Fatalf("ParseConfig: %v", err)
	}
	if cfg.HiddenSize != 384 || cfg.NumAttentionHeads != 12 {
		t.Fatalf("unexpected config: %+v", cfg)
	}
	if cfg.HeadDim() != 32 {
		t.Fatalf("HeadDim() = %d, want 32", cfg.HeadDim())
	}
}

// TestParseConfig_Good_Defaults fills layer_norm_eps and hidden_act when absent.
func TestParseConfig_Good_Defaults(t *testing.T) {
	data := []byte(`{"hidden_size":8,"num_hidden_layers":1,"num_attention_heads":2,"intermediate_size":16,"vocab_size":10,"max_position_embeddings":16,"type_vocab_size":2}`)
	cfg, err := ParseConfig(data)
	if err != nil {
		t.Fatalf("ParseConfig: %v", err)
	}
	if cfg.LayerNormEps != 1e-12 {
		t.Errorf("default layer_norm_eps = %v, want 1e-12", cfg.LayerNormEps)
	}
	if cfg.HiddenAct != "gelu" {
		t.Errorf("default hidden_act = %q, want gelu", cfg.HiddenAct)
	}
}

// TestParseConfig_Bad_NotDivisible rejects a hidden size the heads cannot split.
func TestParseConfig_Bad_NotDivisible(t *testing.T) {
	data := []byte(`{"hidden_size":10,"num_hidden_layers":1,"num_attention_heads":3,"intermediate_size":16,"vocab_size":10,"max_position_embeddings":16,"type_vocab_size":2}`)
	if _, err := ParseConfig(data); err == nil {
		t.Fatal("expected an error for hidden_size not divisible by heads")
	}
}

// TestParseConfig_Ugly_InvalidJSON surfaces a decode error rather than a zero config.
func TestParseConfig_Ugly_InvalidJSON(t *testing.T) {
	if _, err := ParseConfig([]byte(`{not json`)); err == nil {
		t.Fatal("expected a decode error for malformed JSON")
	}
}
