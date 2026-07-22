// SPDX-Licence-Identifier: EUPL-1.2

package gptneox

import (
	"testing"

	core "dappco.re/go"
)

// Config receipts are from the named checkpoints' public config.json files:
// https://huggingface.co/EleutherAI/pythia-70m/blob/main/config.json
// https://huggingface.co/EleutherAI/gpt-j-6b/blob/main/config.json
// https://huggingface.co/EleutherAI/gpt-neo-125m/blob/main/config.json
func parseConfig(t *testing.T, raw string) *Config {
	t.Helper()
	var cfg Config
	if r := core.JSONUnmarshal([]byte(raw), &cfg); !r.OK {
		t.Fatal(r.Error())
	}
	return &cfg
}

func TestConfig_Arch_Good(t *testing.T) {
	cfg := parseConfig(t, `{"model_type":"gpt_neox","hidden_size":512,"intermediate_size":2048,"num_hidden_layers":6,"num_attention_heads":8,"vocab_size":50304,"layer_norm_eps":1e-5,"rotary_emb_base":10000,"rotary_pct":0.25,"use_parallel_residual":true}`)
	a, err := cfg.Arch()
	if err != nil {
		t.Fatal(err)
	}
	if a.HeadDim != 64 || a.RotaryDim != 16 || !a.ParallelResidual || len(a.Layer) != 6 {
		t.Fatalf("Pythia arch = %#v", a)
	}
}

func TestConfig_Arch_Bad(t *testing.T) {
	cfg := parseConfig(t, `{"model_type":"gpt_neox","hidden_size":513,"num_hidden_layers":6,"num_attention_heads":8,"vocab_size":50304}`)
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("indivisible hidden size accepted")
	}
}

func TestConfig_Arch_Ugly(t *testing.T) {
	cfg := parseConfig(t, `{"model_type":"gpt_neox","hidden_size":8,"intermediate_size":16,"num_hidden_layers":1,"num_attention_heads":2,"vocab_size":9,"rotary_pct":0.1}`)
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("zero rounded rotary dimension accepted")
	}
}

func TestConfig_InferFromWeights_Good(t *testing.T) {
	cfg := Config{HiddenSize: 8}
	cfg.InferFromWeights(nil)
	if cfg.HiddenSize != 8 {
		t.Fatalf("InferFromWeights changed config: %+v", cfg)
	}
}

func TestConfig_InferFromWeights_Bad(t *testing.T) {
	cfg := Config{}
	cfg.InferFromWeights(nil)
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("empty config became valid after InferFromWeights")
	}
}

// TestConfig_InferFromWeights_Ugly proves the no-op does not paper over the
// zero-rounded-rotary-dimension guard — distinct from _Bad's all-zero rejection.
func TestConfig_InferFromWeights_Ugly(t *testing.T) {
	cfg := Config{ModelType: "gpt_neox", HiddenSize: 8, IntermediateSize: 16, NumHiddenLayers: 1, NumAttentionHeads: 2, VocabSize: 9, RotaryPct: 0.1}
	cfg.InferFromWeights(nil)
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("zero-rounded rotary dimension became valid after InferFromWeights")
	}
}

func TestConfig_GPTJ_Good(t *testing.T) {
	cfg := parseConfig(t, `{"model_type":"gptj","n_embd":4096,"n_inner":16384,"n_layer":28,"n_head":16,"vocab_size":50400,"rotary_dim":64,"layer_norm_epsilon":1e-5}`)
	a, err := cfg.Arch()
	if err != nil {
		t.Fatal(err)
	}
	if a.HeadDim != 256 || a.RotaryDim != 64 || !a.ParallelResidual {
		t.Fatalf("GPT-J arch = %#v", a)
	}
}

func TestConfig_GPTNeo_Good(t *testing.T) {
	layers := `["global","local","global","local","global","local","global","local","global","local","global","local"]`
	cfg := parseConfig(t, `{"model_type":"gpt_neo","n_embd":768,"n_inner":3072,"n_layer":12,"n_head":12,"vocab_size":50257,"window_size":256,"attention_layers":`+layers+`}`)
	a, err := cfg.Arch()
	if err != nil {
		t.Fatal(err)
	}
	if a.SlidingWindow != 256 || a.Layer[1].Attention != 1 || a.ParallelResidual {
		t.Fatalf("GPT-Neo arch = %#v", a)
	}
}
