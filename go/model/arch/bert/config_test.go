// SPDX-Licence-Identifier: EUPL-1.2

package bert

import "testing"

func TestConfig_HeadDim_Good(t *testing.T) {
	cfg := Config{HiddenSize: 384, NumAttentionHeads: 12}
	if got := cfg.HeadDim(); got != 32 {
		t.Fatalf("HeadDim() = %d, want 32", got)
	}
}

// TestConfig_HeadDim_Bad proves the zero-head-count guard: HeadDim returns 0
// rather than dividing by zero.
func TestConfig_HeadDim_Bad(t *testing.T) {
	if got := (Config{}).HeadDim(); got != 0 {
		t.Fatalf("HeadDim() = %d, want 0 for a zero head count", got)
	}
}

// TestConfig_HeadDim_Ugly proves HeadDim itself does NOT validate divisibility
// — a hidden_size not evenly divisible by heads truncates (integer division)
// rather than erroring; that guard lives in Config.validate/ParseConfig, not
// here. Distinct from _Bad's divide-by-zero guard.
func TestConfig_HeadDim_Ugly(t *testing.T) {
	cfg := Config{HiddenSize: 10, NumAttentionHeads: 3}
	if got := cfg.HeadDim(); got != 3 {
		t.Fatalf("HeadDim() = %d, want 3 (10/3 truncated, HeadDim doesn't itself validate)", got)
	}
}

// TestConfig_ParseConfig_Good decodes a well-formed BertModel config and fills defaults.
func TestConfig_ParseConfig_Good(t *testing.T) {
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

func TestConfig_IsCrossEncoder_Good(t *testing.T) {
	cfg, err := ParseConfig([]byte(`{"architectures":["BertForSequenceClassification"],"id2label":{"0":"LABEL_0"},"hidden_size":8,"num_hidden_layers":1,"num_attention_heads":2,"intermediate_size":16,"vocab_size":10,"max_position_embeddings":16,"type_vocab_size":2}`))
	if err != nil {
		t.Fatalf("ParseConfig: %v", err)
	}
	if cfg.NumLabels != 1 || !cfg.IsCrossEncoder() {
		t.Fatalf("config = %+v, want scalar cross-encoder", cfg)
	}
}

func TestConfig_IsCrossEncoder_Bad(t *testing.T) {
	cfg := Config{NumLabels: 2, Architectures: []string{"BertForSequenceClassification"}}
	if cfg.IsCrossEncoder() {
		t.Fatal("two-label classifier detected as scalar cross-encoder")
	}
}

func TestConfig_IsCrossEncoder_Ugly(t *testing.T) {
	if (Config{NumLabels: 1}).IsCrossEncoder() {
		t.Fatal("architecture-free config detected as cross-encoder")
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

// TestConfig_ParseConfig_Bad rejects a hidden size the heads cannot split.
func TestConfig_ParseConfig_Bad(t *testing.T) {
	data := []byte(`{"hidden_size":10,"num_hidden_layers":1,"num_attention_heads":3,"intermediate_size":16,"vocab_size":10,"max_position_embeddings":16,"type_vocab_size":2}`)
	if _, err := ParseConfig(data); err == nil {
		t.Fatal("expected an error for hidden_size not divisible by heads")
	}
}

// TestConfig_ParseConfig_Ugly surfaces a decode error rather than a zero config.
func TestConfig_ParseConfig_Ugly(t *testing.T) {
	if _, err := ParseConfig([]byte(`{not json`)); err == nil {
		t.Fatal("expected a decode error for malformed JSON")
	}
}
