// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"os"
	"path/filepath"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

// minimalGemma4Tensors builds a complete dense bf16 gemma4 tensor set for arch — just the required
// weights at the right shapes, distinct fills not needed (the validation only checks presence).
func minimalGemma4Tensors(arch model.Arch) map[string]safetensors.Tensor {
	ts := map[string]safetensors.Tensor{}
	bf := func(n int) safetensors.Tensor {
		return safetensors.Tensor{Dtype: "BF16", Shape: []int{n}, Data: make([]byte, n*2)}
	}
	mat := func(out, in int) safetensors.Tensor {
		return safetensors.Tensor{Dtype: "BF16", Shape: []int{out, in}, Data: make([]byte, out*in*2)}
	}
	d := arch.Hidden
	ts["model.embed_tokens.weight"] = mat(arch.Vocab, d)
	ts["model.norm.weight"] = bf(d)
	for i := range arch.Layer {
		p := core.Sprintf("model.layers.%d", i)
		ts[p+".input_layernorm.weight"] = bf(d)
		ts[p+".self_attn.q_proj.weight"] = mat(arch.Heads*arch.HeadDim, d)
		ts[p+".self_attn.k_proj.weight"] = mat(arch.KVHeads*arch.HeadDim, d)
		ts[p+".self_attn.v_proj.weight"] = mat(arch.KVHeads*arch.HeadDim, d)
		ts[p+".self_attn.o_proj.weight"] = mat(d, arch.Heads*arch.HeadDim)
		ts[p+".pre_feedforward_layernorm.weight"] = bf(d)
		ts[p+".mlp.gate_proj.weight"] = mat(arch.FF, d)
		ts[p+".mlp.up_proj.weight"] = mat(arch.FF, d)
		ts[p+".mlp.down_proj.weight"] = mat(d, arch.FF)
		ts[p+".post_feedforward_layernorm.weight"] = bf(d)
	}
	return ts
}

// TestAssembleValidatesRequired gates the presence validation: a complete set assembles, and a set
// missing a required weight (q_proj) is rejected with a clean error rather than a nil-deref later.
func TestAssembleValidatesRequired(t *testing.T) {
	arch, err := Config{
		HiddenSize: 64, NumHiddenLayers: 2, IntermediateSize: 128,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 16, VocabSize: 32, RMSNormEps: 1e-6,
	}.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	ts := minimalGemma4Tensors(arch)
	if _, err := gemma4Assemble(ts, arch); err != nil {
		t.Fatalf("Assemble of a complete set: %v", err)
	}
	delete(ts, "model.layers.0.self_attn.q_proj.weight")
	if _, err := gemma4Assemble(ts, arch); err == nil {
		t.Fatal("expected an error on a missing required q_proj")
	}
}

func TestLoadDiffusionGemmaDecoderTrunk_Good(t *testing.T) {
	arch, err := Config{
		HiddenSize: 64, NumHiddenLayers: 2, IntermediateSize: 128,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 16, GlobalHeadDim: 16,
		VocabSize: 32, RMSNormEps: 1e-6, SlidingWindow: 4,
		LayerTypes: []string{"sliding_attention", "full_attention"},
	}.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	decoder := make(map[string]safetensors.Tensor)
	for name, tensor := range minimalGemma4Tensors(arch) {
		decoder["model.decoder."+name] = tensor
	}
	bf := func(n int) safetensors.Tensor {
		return safetensors.Tensor{Dtype: "BF16", Shape: []int{n}, Data: make([]byte, n*2)}
	}
	mat := func(out, in int) safetensors.Tensor {
		return safetensors.Tensor{Dtype: "BF16", Shape: []int{out, in}, Data: make([]byte, out*in*2)}
	}
	decoder["self_conditioning.pre_norm.weight"] = bf(64)
	decoder["self_conditioning.gate_proj.weight"] = mat(128, 64)
	decoder["self_conditioning.up_proj.weight"] = mat(128, 64)
	decoder["self_conditioning.down_proj.weight"] = mat(64, 128)
	decoder["model.encoder.language_model.layers.0.layer_scalar"] = bf(1)
	decoder["model.encoder.language_model.layers.1.layer_scalar"] = bf(1)

	dir := t.TempDir()
	configJSON := `{
		"model_type": "diffusion_gemma",
		"hidden_size": 64,
		"num_hidden_layers": 2,
		"intermediate_size": 128,
		"num_attention_heads": 2,
		"num_key_value_heads": 1,
		"head_dim": 16,
		"global_head_dim": 16,
		"vocab_size": 32,
		"max_position_embeddings": 16,
		"rms_norm_eps": 1e-6,
		"sliding_window": 4,
		"canvas_length": 4,
		"eos_token_id": [1, 2],
		"layer_types": ["sliding_attention", "full_attention"]
	}`
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), configJSON); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	blob, err := safetensors.Encode(decoder)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if err := coreio.Local.Write(core.PathJoin(dir, "model.safetensors"), string(blob)); err != nil {
		t.Fatalf("write model.safetensors: %v", err)
	}

	loaded, dm, err := model.Load(dir)
	if err != nil {
		t.Fatalf("model.Load(diffusion_gemma trunk): %v", err)
	}
	defer func() { _ = dm.Close() }()
	if loaded.Embed == nil || loaded.FinalNorm == nil || len(loaded.Layers) != 2 {
		t.Fatalf("loaded diffusion trunk incomplete: embed=%v finalNorm=%v layers=%d", loaded.Embed != nil, loaded.FinalNorm != nil, len(loaded.Layers))
	}
	if loaded.Layers[0].Q == nil || loaded.Layers[1].O == nil {
		t.Fatalf("loaded diffusion trunk missing attention projections: layer0.Q=%v layer1.O=%v", loaded.Layers[0].Q != nil, loaded.Layers[1].O != nil)
	}
	if loaded.Diffusion == nil {
		t.Fatalf("loaded diffusion trunk missing neutral diffusion extras")
	}
	if loaded.Diffusion.CanvasLength != 4 {
		t.Fatalf("diffusion canvas length = %d, want 4", loaded.Diffusion.CanvasLength)
	}
	if len(loaded.Diffusion.EOSTokens) != 2 || loaded.Diffusion.EOSTokens[0] != 1 || loaded.Diffusion.EOSTokens[1] != 2 {
		t.Fatalf("diffusion eos tokens = %v, want [1 2]", loaded.Diffusion.EOSTokens)
	}
	if len(loaded.Diffusion.EncoderLayerScalars) != 2 {
		t.Fatalf("diffusion encoder scalars = %d, want 2", len(loaded.Diffusion.EncoderLayerScalars))
	}
	if loaded.Diffusion.SelfCondPreNorm == nil || loaded.Diffusion.SelfCondGate == nil || loaded.Diffusion.SelfCondUp == nil || loaded.Diffusion.SelfCondDown == nil {
		t.Fatalf("loaded diffusion self-conditioning block incomplete: %+v", loaded.Diffusion)
	}
}

// gemma4Snapshot resolves an HF-cache snapshot dir for repo, or "" when not cached.
func gemma4Snapshot(repo string) string {
	base := filepath.Join(os.Getenv("HOME"), ".cache/huggingface/hub", repo, "snapshots")
	ents, err := os.ReadDir(base)
	if err != nil {
		return ""
	}
	for _, e := range ents {
		if e.IsDir() {
			d := filepath.Join(base, e.Name())
			if _, err := os.Stat(filepath.Join(d, "config.json")); err == nil {
				return d
			}
		}
	}
	return ""
}

// TestParseConfigRealFamily round-trips the REAL per-size config.json files (the HF-cache
// snapshots, config only — no weights, no GPU) through parseGemma4Config → Arch and asserts
// the model-card truth for the WHOLE family. The gemma-4 sizes are NOT scaled twins — E2B is
// 35 layers / MQA / full-every-5th / 20 shared-KV; E4B 42 / 2 KV heads / full-every-6th / 18
// shared; 12B is the gemma4_unified encoder arch (48 layers, 16/8 heads, window 1024, dense
// 15360, no experts); 26B-A4B is the MoE (128 experts top-8 at per-expert FF 704 — NOT the
// dense 2112); 31B is the deepest dense (60 layers, 32/16). Any parser change that assumes a
// universal window, a universal full-attention period, uniform KV heads, or treats 12B as "a
// bigger E2B" fails here against the actual checkpoints. Skips per size when not cached.
func TestParseConfigRealFamily(t *testing.T) {
	type sizeTruth struct {
		repo                    string
		modelType               string // the TOP-LEVEL model_type (the registry dispatch id)
		layers, hidden, heads   int
		kvHeads, globalKVHeads  int
		window, kvShared        int
		ff                      int // dense intermediate_size
		experts, topK, expertFF int // 0/0/0 = dense
		firstFull, fullCount    int // sliding/full schedule: index of the first full_attention layer + total fulls
		kEqV                    bool
		quantGS, quantBits      int  // 0/0 = bf16 pack (no quantization block)
		quantOverrides          bool // per-module mixed-precision overrides present (26B QAT)
		perLayerInputHidden     int  // the E2B/E4B PLE tower width; 0 = absent
	}
	cases := []sizeTruth{
		{repo: "models--google--gemma-4-e2b-it", modelType: "gemma4",
			layers: 35, hidden: 1536, heads: 8, kvHeads: 1, globalKVHeads: 1,
			window: 512, kvShared: 20, ff: 6144, firstFull: 4, fullCount: 7, perLayerInputHidden: 256},
		{repo: "models--google--gemma-4-e4b-it", modelType: "gemma4",
			layers: 42, hidden: 2560, heads: 8, kvHeads: 2, globalKVHeads: 2,
			window: 512, kvShared: 18, ff: 10240, firstFull: 5, fullCount: 7, perLayerInputHidden: 256},
		{repo: "models--google--gemma-4-12B-it", modelType: "gemma4_unified",
			layers: 48, hidden: 3840, heads: 16, kvHeads: 8, globalKVHeads: 1,
			window: 1024, kvShared: 0, ff: 15360, firstFull: 5, fullCount: 8, kEqV: true},
		{repo: "models--mlx-community--gemma-4-26B-A4B-it-qat-4bit", modelType: "gemma4",
			layers: 30, hidden: 2816, heads: 16, kvHeads: 8, globalKVHeads: 2,
			window: 1024, kvShared: 0, ff: 2112, experts: 128, topK: 8, expertFF: 704,
			firstFull: 5, fullCount: 5, kEqV: true, quantGS: 64, quantBits: 4, quantOverrides: true},
		{repo: "models--mlx-community--gemma-4-31B-it-4bit", modelType: "gemma4",
			layers: 60, hidden: 5376, heads: 32, kvHeads: 16, globalKVHeads: 4,
			window: 1024, kvShared: 0, ff: 21504, firstFull: 5, fullCount: 10,
			kEqV: true, quantGS: 64, quantBits: 4},
	}
	for _, c := range cases {
		t.Run(c.repo, func(t *testing.T) {
			dir := gemma4Snapshot(c.repo)
			if dir == "" {
				t.Skipf("%s not cached", c.repo)
			}
			raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
			if err != nil {
				t.Fatalf("read real config: %v", err)
			}
			cfg, err := ParseConfig(raw)
			if err != nil {
				t.Fatalf("parseGemma4Config on the real checkpoint: %v", err)
			}
			if cfg.ModelType != c.modelType {
				t.Fatalf("ModelType = %q, want %q (the top-level dispatch id)", cfg.ModelType, c.modelType)
			}
			if _, ok := model.LookupArch(cfg.ModelType); !ok {
				t.Fatalf("model_type %q not resolvable in the arch registry — the real pack would not dispatch", cfg.ModelType)
			}
			a, err := cfg.Arch()
			if err != nil {
				t.Fatalf("Arch from the real config: %v", err)
			}
			if len(a.Layer) != c.layers || a.Hidden != c.hidden || a.Heads != c.heads {
				t.Fatalf("core dims = %d layers/%d hidden/%d heads, want %d/%d/%d", len(a.Layer), a.Hidden, a.Heads, c.layers, c.hidden, c.heads)
			}
			if a.KVHeads != c.kvHeads || a.GlobalKVHeads != c.globalKVHeads {
				t.Fatalf("KV heads = %d sliding/%d global, want %d/%d", a.KVHeads, a.GlobalKVHeads, c.kvHeads, c.globalKVHeads)
			}
			if a.HeadDim != 256 || a.GlobalHeadDim != 512 {
				t.Fatalf("head dims = %d/%d, want 256 sliding / 512 global (every gemma-4 size)", a.HeadDim, a.GlobalHeadDim)
			}
			if a.SlidingWindow != c.window {
				t.Fatalf("SlidingWindow = %d, want %d — the window is per-size, never universal", a.SlidingWindow, c.window)
			}
			if a.FF != c.ff || a.Experts != c.experts || a.TopK != c.topK || a.ExpertFF != c.expertFF {
				t.Fatalf("FFN = dense %d / experts %d top-%d @ %d, want %d / %d top-%d @ %d",
					a.FF, a.Experts, a.TopK, a.ExpertFF, c.ff, c.experts, c.topK, c.expertFF)
			}
			if a.SoftCap != 30 {
				t.Fatalf("SoftCap = %v, want 30 (final_logit_softcapping, every size)", a.SoftCap)
			}
			if a.AttentionKEqV != c.kEqV {
				t.Fatalf("AttentionKEqV = %v, want %v", a.AttentionKEqV, c.kEqV)
			}
			if a.PerLayerInputHidden != c.perLayerInputHidden {
				t.Fatalf("PerLayerInputHidden = %d, want %d (the PLE tower is E2B/E4B-only)", a.PerLayerInputHidden, c.perLayerInputHidden)
			}
			firstFull, fulls := -1, 0
			for i, l := range a.Layer {
				if l.Attention == model.GlobalAttention {
					if firstFull < 0 {
						firstFull = i
					}
					fulls++
				}
			}
			if firstFull != c.firstFull || fulls != c.fullCount {
				t.Fatalf("full-attention schedule: first at %d, %d total; want first %d, %d total — the period is per-size", firstFull, fulls, c.firstFull, c.fullCount)
			}
			owners := 0
			for _, l := range a.Layer {
				if l.OwnsCache() {
					owners++
				}
			}
			if wantOwners := c.layers - c.kvShared; owners != wantOwners {
				t.Fatalf("cache owners = %d, want %d (%d layers − %d kv-shared)", owners, wantOwners, c.layers, c.kvShared)
			}
			q := cfg.Quantization
			if c.quantGS == 0 {
				if q != nil {
					t.Fatalf("bf16 pack carries a quant block: %+v", q)
				}
			} else {
				if q == nil || q.GroupSize != c.quantGS || q.Bits != c.quantBits {
					t.Fatalf("quant = %+v, want gs %d / bits %d", q, c.quantGS, c.quantBits)
				}
				if c.quantOverrides != (len(q.Overrides) > 0) {
					t.Fatalf("quant overrides present = %v (%d), want %v (26B QAT mixes 8-bit mlp/router)", len(q.Overrides) > 0, len(q.Overrides), c.quantOverrides)
				}
			}
			t.Logf("%s: %d layers · %d/%d/%d heads · window %d · full@%d×%d · owners %d · experts %d — real config round-trips",
				c.repo, len(a.Layer), a.Heads, a.KVHeads, a.GlobalKVHeads, a.SlidingWindow, c.firstFull, fulls, owners, a.Experts)
		})
	}
}

// TestLoad_EFamily_QuantAgnostic loads e2b (4-bit) and e4b (qat-4-bit) through the SINGLE shared
// assembler and asserts the things native used to re-bug per model: KV-shared layers carry no own
// K, the MatFormer per-layer FFN width is read from the gate shape, and — the headline — e4b's
// per_layer_model_projection is seen as quantised while e2b's is bf16, with NO per-weight branch.
// AX-11: mmap metadata only, no compute / no GPU.
func TestLoad_EFamily_QuantAgnostic(t *testing.T) {
	cases := []struct {
		key, repo     string
		wantProjQuant bool // per_layer_model_projection: e2b bf16, e4b 4-bit (the bug case)
	}{
		{"e2b", "models--mlx-community--gemma-4-E2B-it-4bit", false},
		{"e4b", "models--mlx-community--gemma-4-E4B-it-qat-4bit", true},
	}
	for _, c := range cases {
		t.Run(c.key, func(t *testing.T) {
			dir := gemma4Snapshot(c.repo)
			if dir == "" {
				t.Skipf("%s not cached", c.key)
			}
			m, dm, err := model.Load(dir)
			if err != nil {
				t.Fatalf("Load: %v", err)
			}
			defer dm.Close()

			if len(m.Layers) != len(m.Arch.Layer) {
				t.Fatalf("layers %d != arch %d", len(m.Layers), len(m.Arch.Layer))
			}
			if m.Embed == nil || m.FinalNorm == nil {
				t.Fatal("embed or final norm missing")
			}
			owners, ffs := 0, map[int]int{}
			for i, L := range m.Layers {
				spec := m.Arch.Layer[i]
				if L.Q == nil || L.AttnNorm == nil {
					t.Fatalf("layer %d missing Q / AttnNorm", i)
				}
				if spec.OwnsCache() {
					owners++
					if L.K == nil {
						t.Fatalf("cache-owner layer %d missing K", i)
					}
				} else if L.K != nil {
					t.Fatalf("KV-shared layer %d has its own K — KV-share broken", i)
				}
				if L.MoE == nil { // dense MLP
					if L.Gate == nil || L.Gate.OutDim <= 0 {
						t.Fatalf("layer %d gate FFN width not read from shape", i)
					}
					ffs[L.Gate.OutDim]++
				}
			}
			if m.PerLayerModelProj == nil {
				t.Fatal("PLE per_layer_model_projection missing")
			}
			if got := m.PerLayerModelProj.Quantised(); got != c.wantProjQuant {
				t.Fatalf("per_layer_model_projection Quantised()=%v, want %v", got, c.wantProjQuant)
			}
			t.Logf("%s: %d layers · %d cache owners (%d shared) · FFN widths %v · PLE-proj quantised=%v",
				c.key, len(m.Layers), owners, len(m.Layers)-owners, ffs, m.PerLayerModelProj.Quantised())
		})
	}
}
