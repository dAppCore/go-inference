// SPDX-Licence-Identifier: EUPL-1.2

package dflash_test

import (
	"encoding/binary"
	"math"
	"testing"

	core "dappco.re/go"
	dflash "dappco.re/go/inference/model/arch/z-lab/dflash"
	"dappco.re/go/inference/model/safetensors"
)

// dflash_test.go proves the z-lab drafter checkpoint contract: ParseConfig on
// the REAL checkpoint's config shape (fields evidenced from
// z-lab/Qwen3-4B-DFlash-b16 — the marker/block fields verbatim from its
// config.json, the decoder dims from the in-tree oracle fixture computed from
// its real weights; see docs/design-dflash-survey.md), Assemble on synthetic
// tensors with exact-shape validation, and Load end-to-end through a written
// bf16 safetensors file.

// zlabRealConfig is a faithful z-lab/Qwen3-4B-DFlash-b16 config.json, trimmed
// to the fields this package reads. Provenance per field: architectures /
// block_size / dflash_config / model_type / num_hidden_layers / hidden_size are
// verbatim from the real file (decode/dflash's TestParseConfig_ZLabNative
// carries the same bytes); head_dim / num_attention_heads /
// num_key_value_heads / rms_norm_eps / rope_theta match the in-tree oracle
// fixture (decode/dflash/testdata/zlab_qwen3_4b_oracle.json) computed FROM the
// real checkpoint; intermediate_size 9728 is the real weight shape the survey
// recorded. Nothing guessed.
const zlabRealConfig = `{
	"architectures": ["DFlashDraftModel"],
	"block_size": 16,
	"dflash_config": {"mask_token_id": 151669, "target_layer_ids": [1, 9, 17, 25, 33]},
	"head_dim": 128,
	"hidden_size": 2560,
	"intermediate_size": 9728,
	"model_type": "qwen3",
	"num_attention_heads": 32,
	"num_hidden_layers": 5,
	"num_key_value_heads": 8,
	"num_target_layers": 36,
	"rms_norm_eps": 1e-06,
	"rope_theta": 1000000
}`

func TestParseConfig_Good(t *testing.T) {
	cfg, ok := dflash.ParseConfig([]byte(zlabRealConfig))
	if !ok {
		t.Fatal("the real z-lab config shape must parse ok")
	}
	if cfg.Hidden != 2560 || cfg.Heads != 32 || cfg.KVHeads != 8 || cfg.HeadDim != 128 {
		t.Fatalf("decoder geometry wrong: %+v", cfg)
	}
	if cfg.Intermediate != 9728 || cfg.NumLayers != 5 {
		t.Fatalf("mlp/layer geometry wrong: %+v", cfg)
	}
	if cfg.Eps != 1e-6 || cfg.RopeTheta != 1_000_000 {
		t.Fatalf("eps/theta wrong: eps=%v theta=%v", cfg.Eps, cfg.RopeTheta)
	}
	if cfg.Block.BlockSize != 16 || cfg.Block.MaskTokenID != 151669 {
		t.Fatalf("block contract wrong: %+v", cfg.Block)
	}
	if cfg.NumAux() != 5 {
		t.Fatalf("NumAux = %d, want 5 (len target_layer_ids)", cfg.NumAux())
	}
}

// TestParseConfig_Defaults_Ugly: a minimal-but-valid z-lab config omitting the
// optional fields — kv heads default to heads, head_dim derives from
// hidden/heads, eps and rope_theta take the qwen3 defaults.
func TestParseConfig_Defaults_Ugly(t *testing.T) {
	data := []byte(`{
		"architectures": ["DFlashDraftModel"],
		"dflash_config": {"target_layer_ids": [1, 3]},
		"hidden_size": 8,
		"intermediate_size": 12,
		"model_type": "qwen3",
		"num_attention_heads": 4,
		"num_hidden_layers": 2
	}`)
	cfg, ok := dflash.ParseConfig(data)
	if !ok {
		t.Fatal("a minimal z-lab config must parse ok")
	}
	if cfg.KVHeads != 4 {
		t.Fatalf("KVHeads default = %d, want heads (4)", cfg.KVHeads)
	}
	if cfg.HeadDim != 2 {
		t.Fatalf("HeadDim derived = %d, want hidden/heads (2)", cfg.HeadDim)
	}
	if cfg.Eps != 1e-6 || cfg.RopeTheta != 1_000_000 {
		t.Fatalf("defaults wrong: eps=%v theta=%v", cfg.Eps, cfg.RopeTheta)
	}
}

func TestParseConfig_Bad(t *testing.T) {
	t.Run("plain qwen3, no dflash marker", func(t *testing.T) {
		if _, ok := dflash.ParseConfig([]byte(`{"model_type":"qwen3","hidden_size":2560,"num_attention_heads":32,"num_hidden_layers":36}`)); ok {
			t.Fatal("a plain qwen3 text model must NOT parse as a dflash drafter")
		}
	})
	t.Run("speculators nesting (no flat decoder dims)", func(t *testing.T) {
		// The RedHatAI convention: marker present, decoder nested under
		// transformer_layer_config — recognised as DFlash by decode/dflash, but
		// NOT loadable by this package (different payload: reduced head, d2t).
		data := []byte(`{
			"speculators_model_type": "dflash",
			"architectures": ["DFlashDraftModel"],
			"aux_hidden_state_layer_ids": [10, 60, 110],
			"transformer_layer_config": {"hidden_size": 4096, "num_attention_heads": 32}
		}`)
		if _, ok := dflash.ParseConfig(data); ok {
			t.Fatal("a speculators-convention config must parse ok=false here (nested decoder dims)")
		}
	})
	t.Run("unparseable bytes", func(t *testing.T) {
		if _, ok := dflash.ParseConfig([]byte(`{nope`)); ok {
			t.Fatal("garbage must not parse")
		}
	})
}

// Tiny synthetic geometry for Assemble/Load: hidden 8, 4 heads x headDim 2,
// kv 2, intermediate 12, 2 layers, 2 fused target layers.
const (
	tHidden = 8
	tHeads  = 4
	tKV     = 2
	tHD     = 2
	tInter  = 12
	tLayers = 2
	tAux    = 2
)

func tinyConfig() dflash.Config {
	cfg, ok := dflash.ParseConfig([]byte(`{
		"architectures": ["DFlashDraftModel"],
		"block_size": 4,
		"dflash_config": {"mask_token_id": 7, "target_layer_ids": [1, 3]},
		"head_dim": 2,
		"hidden_size": 8,
		"intermediate_size": 12,
		"model_type": "qwen3",
		"num_attention_heads": 4,
		"num_hidden_layers": 2,
		"num_key_value_heads": 2
	}`))
	if !ok {
		panic("tinyConfig must parse")
	}
	return cfg
}

// f32Tensor builds an F32 safetensors.Tensor of the given shape with seeded
// deterministic values (never constant — the vacuous-fixture trap).
func f32Tensor(seed uint32, shape ...int) safetensors.Tensor {
	n := 1
	for _, d := range shape {
		n *= d
	}
	data := make([]byte, 4*n)
	s := seed*2654435761 + 11
	for i := 0; i < n; i++ {
		s = s*1664525 + 1013904223
		v := float32(int32(s>>9)%1000)/4000 - 0.125
		binary.LittleEndian.PutUint32(data[4*i:], math.Float32bits(v))
	}
	return safetensors.Tensor{Dtype: "F32", Shape: append([]int(nil), shape...), Data: data}
}

// tinyTensors builds the complete tensor set for tinyConfig — every name the
// real checkpoint carries, nothing else.
func tinyTensors() map[string]safetensors.Tensor {
	ts := map[string]safetensors.Tensor{}
	seed := uint32(1)
	next := func(shape ...int) safetensors.Tensor {
		seed++
		return f32Tensor(seed, shape...)
	}
	ts["fc.weight"] = next(tHidden, tAux*tHidden)
	ts["hidden_norm.weight"] = next(tHidden)
	ts["norm.weight"] = next(tHidden)
	qDim, kvDim := tHeads*tHD, tKV*tHD
	for li := 0; li < tLayers; li++ {
		p := core.Sprintf("layers.%d.", li)
		ts[p+"input_layernorm.weight"] = next(tHidden)
		ts[p+"post_attention_layernorm.weight"] = next(tHidden)
		ts[p+"self_attn.q_proj.weight"] = next(qDim, tHidden)
		ts[p+"self_attn.k_proj.weight"] = next(kvDim, tHidden)
		ts[p+"self_attn.v_proj.weight"] = next(kvDim, tHidden)
		ts[p+"self_attn.o_proj.weight"] = next(tHidden, qDim)
		ts[p+"self_attn.q_norm.weight"] = next(tHD)
		ts[p+"self_attn.k_norm.weight"] = next(tHD)
		ts[p+"mlp.gate_proj.weight"] = next(tInter, tHidden)
		ts[p+"mlp.up_proj.weight"] = next(tInter, tHidden)
		ts[p+"mlp.down_proj.weight"] = next(tHidden, tInter)
	}
	return ts
}

func TestAssemble_Good(t *testing.T) {
	m, err := dflash.Assemble(tinyTensors(), tinyConfig())
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	if len(m.Layers) != tLayers {
		t.Fatalf("layers = %d, want %d", len(m.Layers), tLayers)
	}
	if len(m.FC) != tHidden*tAux*tHidden || len(m.HiddenNorm) != tHidden || len(m.FinalNorm) != tHidden {
		t.Fatalf("fused-context tensor lengths wrong: fc=%d hn=%d fn=%d", len(m.FC), len(m.HiddenNorm), len(m.FinalNorm))
	}
	l := m.Layers[1]
	if len(l.Q) != tHeads*tHD*tHidden || len(l.K) != tKV*tHD*tHidden || len(l.O) != tHidden*tHeads*tHD {
		t.Fatalf("attn tensor lengths wrong: q=%d k=%d o=%d", len(l.Q), len(l.K), len(l.O))
	}
	if len(l.Gate) != tInter*tHidden || len(l.Down) != tHidden*tInter {
		t.Fatalf("mlp tensor lengths wrong: gate=%d down=%d", len(l.Gate), len(l.Down))
	}
	if len(l.QNorm) != tHD || len(l.KNorm) != tHD {
		t.Fatalf("qk norm lengths wrong: q=%d k=%d", len(l.QNorm), len(l.KNorm))
	}
	// The widen must carry real values, not zero-fill.
	var nonZero bool
	for _, v := range m.FC {
		if v != 0 {
			nonZero = true
			break
		}
	}
	if !nonZero {
		t.Fatal("fc.weight widened to all zeros — the widen path is broken")
	}
}

func TestAssemble_Bad(t *testing.T) {
	cfg := tinyConfig()
	t.Run("missing tensor is named", func(t *testing.T) {
		ts := tinyTensors()
		delete(ts, "layers.1.mlp.down_proj.weight")
		_, err := dflash.Assemble(ts, cfg)
		if err == nil {
			t.Fatal("a missing tensor must fail")
		}
		if !core.Contains(err.Error(), "layers.1.mlp.down_proj.weight") {
			t.Fatalf("error must name the missing tensor, got: %v", err)
		}
	})
	t.Run("mis-shaped tensor is named", func(t *testing.T) {
		ts := tinyTensors()
		ts["fc.weight"] = f32Tensor(99, tHidden, tHidden) // wrong input width
		_, err := dflash.Assemble(ts, cfg)
		if err == nil {
			t.Fatal("a mis-shaped tensor must fail")
		}
		if !core.Contains(err.Error(), "fc.weight") {
			t.Fatalf("error must name the mis-shaped tensor, got: %v", err)
		}
	})
	t.Run("incomplete geometry", func(t *testing.T) {
		if _, err := dflash.Assemble(tinyTensors(), dflash.Config{}); err == nil {
			t.Fatal("a zero-value config must fail")
		}
	})
	t.Run("GQA mismatch", func(t *testing.T) {
		bad := cfg
		bad.KVHeads = 3 // 4 heads not a multiple of 3
		if _, err := dflash.Assemble(tinyTensors(), bad); err == nil {
			t.Fatal("heads not a multiple of kv_heads must fail")
		}
	})
	t.Run("no fused target layers", func(t *testing.T) {
		bad := cfg
		bad.Block.AuxHiddenLayerIDs = nil
		if _, err := dflash.Assemble(tinyTensors(), bad); err == nil {
			t.Fatal("a config with no target_layer_ids must fail")
		}
	})
}

// bf16Bytes narrows f32 values to bf16 little-endian bytes (truncation — the
// values used are chosen bf16-exact so the round trip is lossless).
func bf16Bytes(vals []float32) []byte {
	out := make([]byte, 2*len(vals))
	for i, v := range vals {
		bits := math.Float32bits(v) >> 16
		out[2*i] = byte(bits)
		out[2*i+1] = byte(bits >> 8)
	}
	return out
}

// writeTinyCheckpoint writes a loadable bf16 z-lab drafter checkpoint dir:
// config.json + model.safetensors with every required tensor, values seeded
// from bf16-exact steps so Load's widen is checkable exactly.
func writeTinyCheckpoint(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	if r := core.WriteFile(core.PathJoin(dir, "config.json"), []byte(`{
		"architectures": ["DFlashDraftModel"],
		"block_size": 4,
		"dflash_config": {"mask_token_id": 7, "target_layer_ids": [1, 3]},
		"head_dim": 2,
		"hidden_size": 8,
		"intermediate_size": 12,
		"model_type": "qwen3",
		"num_attention_heads": 4,
		"num_hidden_layers": 2,
		"num_key_value_heads": 2
	}`), 0o644); !r.OK {
		t.Fatalf("write config.json: %v", r.Err())
	}
	infos := map[string]safetensors.SafetensorsTensorInfo{}
	data := map[string][]byte{}
	for name, tensor := range tinyTensors() {
		n := len(tensor.Data) / 4
		vals := make([]float32, n)
		for i := range vals {
			vals[i] = float32(i%13)*0.25 - 1.5 // bf16-exact quarter steps
		}
		infos[name] = safetensors.SafetensorsTensorInfo{Dtype: "BF16", Shape: append([]int(nil), tensor.Shape...)}
		data[name] = bf16Bytes(vals)
	}
	if r := safetensors.WriteSafetensors(core.PathJoin(dir, "model.safetensors"), infos, data); !r.OK {
		t.Fatalf("write model.safetensors: %v", r.Err())
	}
	return dir
}

func TestLoad_Good(t *testing.T) {
	dir := writeTinyCheckpoint(t)
	m, err := dflash.Load(dir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if m.Cfg.Hidden != tHidden || m.Cfg.NumLayers != tLayers || m.Cfg.NumAux() != tAux {
		t.Fatalf("loaded geometry wrong: %+v", m.Cfg)
	}
	// The bf16 widen must reproduce the bf16-exact written values.
	want := float32(1%13)*0.25 - 1.5
	if m.FC[1] != want {
		t.Fatalf("FC[1] = %v, want %v (bf16-exact round trip)", m.FC[1], want)
	}
}

func TestLoad_Bad(t *testing.T) {
	t.Run("no config.json", func(t *testing.T) {
		if _, err := dflash.Load(t.TempDir()); err == nil {
			t.Fatal("an empty dir must fail to load")
		}
	})
	t.Run("not a dflash drafter", func(t *testing.T) {
		dir := t.TempDir()
		if r := core.WriteFile(core.PathJoin(dir, "config.json"), []byte(`{"model_type":"qwen3","hidden_size":8,"num_attention_heads":4,"num_hidden_layers":2}`), 0o644); !r.OK {
			t.Fatalf("write config: %v", r.Err())
		}
		if _, err := dflash.Load(dir); err == nil {
			t.Fatal("a plain qwen3 config must be refused with the named decline")
		}
	})
}
