// SPDX-Licence-Identifier: EUPL-1.2

package bloom

import (
	"testing"

	core "dappco.re/go"
)

// Fixture source: https://huggingface.co/bigscience/bloom-560m/blob/main/config.json
func TestConfigBLOOM560M_Good(t *testing.T) {
	var cfg Config
	if r := core.JSONUnmarshal([]byte(bloom560MConfig), &cfg); !r.OK {
		t.Fatalf("parse fixture: %v", r.Value)
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if arch.Hidden != 1024 || arch.Heads != 16 || arch.FF != 4096 || !arch.ALiBi || len(arch.Layer) != 24 {
		t.Fatalf("BLOOM-560m arch = %+v", arch)
	}
}

func TestConfigArch_Bad(t *testing.T) {
	if _, err := (Config{}).Arch(); err == nil {
		t.Fatal("empty config accepted")
	}
}

const bloom560MConfig = `{"layer_norm_epsilon":1e-5,"model_type":"bloom","n_embed":1024,"n_inner":null,"n_layer":24,"num_attention_heads":16,"offset_alibi":100,"vocab_size":250880}`
