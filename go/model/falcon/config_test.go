// SPDX-Licence-Identifier: EUPL-1.2

package falcon

import (
	"testing"

	core "dappco.re/go"
)

// Fixture source: https://huggingface.co/tiiuae/falcon-rw-1b/blob/main/config.json
func TestConfigFalconRW1B_Good(t *testing.T) {
	var cfg Config
	if r := core.JSONUnmarshal([]byte(falconRW1BConfig), &cfg); !r.OK {
		t.Fatalf("parse fixture: %v", r.Value)
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if arch.Hidden != 2048 || arch.Heads != 32 || arch.KVHeads != 32 || !arch.ALiBi || arch.ParallelResidual || len(arch.Layer) != 24 {
		t.Fatalf("Falcon-RW-1B arch = %+v", arch)
	}
}

func TestConfigParallelAttention_Good(t *testing.T) {
	arch, err := (Config{HiddenSize: 64, NumHiddenLayers: 2, NumAttentionHeads: 8, VocabSize: 100, ParallelAttn: true}).Arch()
	if err != nil || !arch.ParallelResidual {
		t.Fatalf("parallel attention Arch = %+v, %v", arch, err)
	}
}

func TestConfigMultiQuery_Good(t *testing.T) {
	cfg := Config{HiddenSize: 64, NumHiddenLayers: 2, NumAttentionHeads: 8, VocabSize: 100, MultiQuery: true, ALiBi: true}
	arch, err := cfg.Arch()
	if err != nil || arch.KVHeads != 1 {
		t.Fatalf("multi-query Arch = %+v, %v", arch, err)
	}
}

func TestConfigNewDecoderArchitecture_Good(t *testing.T) {
	cfg := Config{HiddenSize: 64, NumHiddenLayers: 2, NumAttentionHeads: 8, NumKVHeads: 2, VocabSize: 100, NewDecoderArchitecture: true, ALiBi: true}
	arch, err := cfg.Arch()
	if err != nil || arch.KVHeads != 2 {
		t.Fatalf("new decoder Arch = %+v, %v", arch, err)
	}
}

func TestConfigArch_Bad(t *testing.T) {
	if _, err := (Config{}).Arch(); err == nil {
		t.Fatal("empty config accepted")
	}
}

const falconRW1BConfig = `{"alibi":true,"hidden_size":2048,"layer_norm_epsilon":1e-5,"model_type":"falcon","multi_query":false,"new_decoder_architecture":false,"num_attention_heads":32,"num_hidden_layers":24,"parallel_attn":false,"vocab_size":50304}`
