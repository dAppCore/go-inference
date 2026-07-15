// SPDX-Licence-Identifier: EUPL-1.2

package gpt2

import (
	core "dappco.re/go"
	"testing"
)

// Fixtures are verbatim architecture fields from the cited public config.json files.
// Sources: https://huggingface.co/openai-community/gpt2/blob/main/config.json
// https://huggingface.co/AI-Sweden-Models/gpt-sw3-1.3b/blob/main/config.json
// https://huggingface.co/bigcode/tiny_starcoder_py/blob/main/config.json
func TestConfig_RealCheckpointFixtures_Good(t *testing.T) {
	fixtures := []struct {
		name, json                string
		hidden, heads, layers, kv int
	}{
		{"GPT-2", `{"model_type":"gpt2","n_embd":768,"n_head":12,"n_layer":12,"n_positions":1024,"vocab_size":50257,"activation_function":"gelu_new","layer_norm_epsilon":0.00001}`, 768, 12, 12, 12},
		{"GPT-SW3", `{"model_type":"gpt2","n_embd":2048,"n_head":32,"n_inner":8192,"n_layer":24,"n_positions":2048,"vocab_size":64000,"activation_function":"gelu","layer_norm_epsilon":0.00001}`, 2048, 32, 24, 32},
		{"StarCoder", `{"model_type":"gpt_bigcode","n_embd":768,"n_head":12,"n_inner":3072,"n_layer":20,"n_positions":8192,"vocab_size":49152,"activation_function":"gelu_pytorch_tanh","multi_query":true}`, 768, 12, 20, 1},
	}
	for _, f := range fixtures {
		t.Run(f.name, func(t *testing.T) {
			var c Config
			if r := core.JSONUnmarshal([]byte(f.json), &c); !r.OK {
				t.Fatal("parse")
			}
			a, err := c.Arch()
			if err != nil {
				t.Fatal(err)
			}
			if a.Hidden != f.hidden || a.Heads != f.heads || len(a.Layer) != f.layers || a.KVHeads != f.kv || !a.LearnedAbsolutePositions {
				t.Fatalf("arch=%+v", a)
			}
		})
	}
}

func TestConfig_RealCheckpointFixtures_Bad(t *testing.T) {
	if _, err := (&Config{}).Arch(); err == nil {
		t.Fatal("empty config accepted")
	}
}
func TestConfig_RealCheckpointFixtures_Ugly(t *testing.T) {
	if _, err := (&Config{Hidden: 7, Heads: 2, Layers: 1, Positions: 1, Vocab: 1}).Arch(); err == nil {
		t.Fatal("indivisible heads accepted")
	}
}
