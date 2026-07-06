// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

// mLinearStub is a non-nil model.Linear for the validateRequired / Tied direct-call branches —
// those checks only test pointer presence, so an empty Linear suffices.
var mLinearStub = model.Linear{}

// This file closes the coverage gaps the original config_test.go / load_test.go left open:
// the small pure-arch helpers (HasMoE / MaxKVHeads / MaxHeadDim's else / model.DeriveLayers clamps),
// the remaining Config branches (ResolvedQuant fallback+nil, quant override skips, the
// global-kv-heads + expertFF-fallback Arch paths), and — the headline — the on-disk Load entry
// driven off a synthetic checkpoint written to t.TempDir() (no model load, no GPU, AX-11). All
// white-box (package gemma4) so the unexported validateRequired / model.DeriveLayers are reachable.

// TestHasMoE covers Arch.HasMoE in both directions: a dense arch reports false, a MoE arch true.
func TestHasMoE(t *testing.T) {
	dense, err := Config{
		HiddenSize: 64, NumHiddenLayers: 2, IntermediateSize: 128,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 16, VocabSize: 32,
	}.Arch()
	if err != nil {
		t.Fatalf("dense Arch: %v", err)
	}
	if dense.HasMoE() {
		t.Fatal("dense arch reported HasMoE() = true")
	}

	moe, err := Config{
		HiddenSize: 64, NumHiddenLayers: 2, IntermediateSize: 128,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 16, VocabSize: 32,
		EnableMoEBlock: true, NumExperts: 4, TopKExperts: 2, MoEIntermediateSize: 96,
	}.Arch()
	if err != nil {
		t.Fatalf("moe Arch: %v", err)
	}
	if !moe.HasMoE() {
		t.Fatal("moe arch reported HasMoE() = false")
	}
	t.Logf("HasMoE: dense=false, moe=true")
}

// TestMaxHeadDimAndKVHeads covers both branches of MaxHeadDim and MaxKVHeads: the no-distinction
// case (global == sliding → return the sliding value) and the gemma4 case (global larger → return
// the global value). The original suite only hit MaxHeadDim's larger-global branch.
func TestMaxHeadDimAndKVHeads(t *testing.T) {
	// Uniform: GlobalHeadDim/GlobalKVHeads default to HeadDim/KVHeads, so the else branch returns them.
	uniform := model.Arch{HeadDim: 256, KVHeads: 4, GlobalHeadDim: 256, GlobalKVHeads: 4}
	if uniform.MaxHeadDim() != 256 {
		t.Fatalf("uniform MaxHeadDim = %d, want 256 (else branch)", uniform.MaxHeadDim())
	}
	if uniform.MaxKVHeads() != 4 {
		t.Fatalf("uniform MaxKVHeads = %d, want 4 (else branch)", uniform.MaxKVHeads())
	}

	// gemma4: full_attention uses a larger head_dim and may carry more KV heads.
	split := model.Arch{HeadDim: 256, KVHeads: 1, GlobalHeadDim: 512, GlobalKVHeads: 2}
	if split.MaxHeadDim() != 512 {
		t.Fatalf("split MaxHeadDim = %d, want 512", split.MaxHeadDim())
	}
	if split.MaxKVHeads() != 2 {
		t.Fatalf("split MaxKVHeads = %d, want 2", split.MaxKVHeads())
	}
	t.Logf("MaxHeadDim/MaxKVHeads: uniform→256/4 (else), split→512/2 (global)")
}

// TestDeriveLayersClamps covers the two firstShared clamps model.DeriveLayers guards: numKVShared > n
// (firstShared < 0 → clamp to 0, every layer shares) and numKVShared < 0 (firstShared > n → clamp
// to n, every layer owns). The original suite only exercised the in-range path.
func TestDeriveLayersClamps(t *testing.T) {
	types := []string{"full_attention", "full_attention", "sliding_attention"}

	// numKVShared > n → firstShared = n - 5 = -2 → clamp 0 → no layer is in the owner-by-position
	// region; each promotes only as the first of its type (the toy edge).
	over := model.DeriveLayers(types, 5)
	if len(over) != 3 {
		t.Fatalf("over: got %d specs", len(over))
	}
	// layer 0 (full) is the first full → promoted owner; layer 1 (full) shares 0; layer 2 (sliding)
	// is the first sliding → promoted owner.
	if !over[0].OwnsCache() || over[1].OwnsCache() || !over[2].OwnsCache() {
		t.Fatalf("over ownership wrong: %+v", over)
	}
	if over[1].KVShareFrom != 0 {
		t.Fatalf("over: layer 1 should share layer 0, got KVShareFrom=%d", over[1].KVShareFrom)
	}

	// numKVShared < 0 → firstShared = n - (-2) = 5 > n → clamp n → every layer owns.
	under := model.DeriveLayers(types, -2)
	for i, s := range under {
		if !s.OwnsCache() {
			t.Fatalf("under: layer %d should own its cache (clamp to all-own), got %+v", i, s)
		}
	}
	t.Logf("model.DeriveLayers clamps: numKVShared>n → all-share-by-type, numKVShared<0 → all-own")
}

// TestResolvedQuantFallbackAndNil covers ResolvedQuant's two uncovered branches: the nested
// text_config fallback (top-level nil, quant under text_config) and the all-nil bf16 case. The
// original suite only hit the top-level-present branch.
func TestResolvedQuantFallbackAndNil(t *testing.T) {
	// Fallback: top-level quant nil, text_config carries it.
	nested := Config{TextConfig: &Config{Quantization: &model.QuantConfig{GroupSize: 32, Bits: 8}}}
	if q := nested.ResolvedQuant(); q == nil || q.GroupSize != 32 || q.Bits != 8 {
		t.Fatalf("nested fallback ResolvedQuant = %+v, want gs32/b8", q)
	}

	// bf16: no quant anywhere → nil.
	if q := (Config{}).ResolvedQuant(); q != nil {
		t.Fatalf("bf16 ResolvedQuant = %+v, want nil", q)
	}

	// text_config present but itself quant-free → still nil.
	if q := (Config{TextConfig: &Config{}}).ResolvedQuant(); q != nil {
		t.Fatalf("text_config-without-quant ResolvedQuant = %+v, want nil", q)
	}
	t.Logf("ResolvedQuant: text_config fallback resolves, bf16 → nil")
}

// TestQuantUnmarshalSkips covers the QuantConfig.UnmarshalJSON branches the original mixed-precision
// test missed: a parse failure (returns an error), and a per-module override whose bits == 0 (skipped,
// not recorded). The covered original test already hits the scalar + valid-override + "mode" paths.
func TestQuantUnmarshalSkips(t *testing.T) {
	// Parse failure: malformed JSON for the quantization block.
	var bad model.QuantConfig
	if err := bad.UnmarshalJSON([]byte(`{not json`)); err == nil {
		t.Fatal("expected an error unmarshalling malformed quant JSON")
	}

	// bits == 0 override is skipped; a real override is kept.
	var q model.QuantConfig
	if err := q.UnmarshalJSON([]byte(`{"group_size":64,"bits":4,
		"model.layers.0.mlp.gate_proj":{"group_size":64,"bits":0},
		"model.layers.0.mlp.up_proj":{"group_size":32,"bits":8}}`)); err != nil {
		t.Fatalf("valid quant unmarshal: %v", err)
	}
	if _, ok := q.Overrides["model.layers.0.mlp.gate_proj"]; ok {
		t.Fatal("a bits==0 override should be skipped, not recorded")
	}
	if o, ok := q.Overrides["model.layers.0.mlp.up_proj"]; !ok || o.Bits != 8 {
		t.Fatalf("the bits==8 override should be kept, got %+v ok=%v", o, ok)
	}
	t.Logf("QuantConfig.UnmarshalJSON: malformed → error, bits==0 override skipped, bits!=0 kept")
}

// TestArchGlobalKVHeadsError covers the num_attention_heads % num_global_key_value_heads validation
// (the per-type-KV branch the original error suite didn't reach).
func TestArchGlobalKVHeadsError(t *testing.T) {
	_, err := Config{
		HiddenSize: 256, NumHiddenLayers: 2, IntermediateSize: 512,
		NumAttentionHeads: 8, NumKeyValueHeads: 2, HeadDim: 32,
		NumGlobalKeyValueHeads: 3, // 8 % 3 != 0
		VocabSize:              100,
	}.Arch()
	if err == nil {
		t.Fatal("expected an error when num_attention_heads is not a multiple of num_global_key_value_heads")
	}
	t.Logf("Arch: rejects num_global_key_value_heads that doesn't divide num_attention_heads")
}

// TestArchExpertFFFallback covers the MoE expertFF == 0 fallback: when moe_intermediate_size is
// absent, ExpertFF falls back to the dense intermediate_size.
func TestArchExpertFFFallback(t *testing.T) {
	a, err := Config{
		HiddenSize: 64, NumHiddenLayers: 2, IntermediateSize: 128,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 16, VocabSize: 32,
		EnableMoEBlock: true, NumExperts: 4, TopKExperts: 2, // no MoEIntermediateSize
	}.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if a.ExpertFF != 128 {
		t.Fatalf("ExpertFF = %d, want 128 (fallback to intermediate_size)", a.ExpertFF)
	}
	t.Logf("Arch: ExpertFF falls back to intermediate_size (%d) when moe_intermediate_size absent", a.ExpertFF)
}

// TestTiedReportsLMHead covers model.LoadedModel.Tied in both directions.
func TestTiedReportsLMHead(t *testing.T) {
	if !(&model.LoadedModel{LMHead: nil}).Tied() {
		t.Fatal("LMHead nil should report Tied() = true")
	}
	if (&model.LoadedModel{LMHead: &mLinearStub}).Tied() {
		t.Fatal("a separate LMHead should report Tied() = false")
	}
	t.Logf("Tied: nil LMHead → true (tied to embed), separate LMHead → false")
}

// TestValidateRequiredDirectBranches covers the validateRequired branches not reachable through
// Assemble. m.Embed == nil is shadowed by Assemble's own embed check (it returns before
// validateRequired), so it can only be hit by calling validateRequired directly on a hand-built
// model; the final-norm / cache-owner-K / dense-MLP branches are also exercised here directly for a
// focused signal (Assemble-driven coverage of them is in TestAssembleMissingWeightBranches).
func TestValidateRequiredDirectBranches(t *testing.T) {
	arch, err := Config{
		HiddenSize: 64, NumHiddenLayers: 1, IntermediateSize: 128,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 16, VocabSize: 32,
	}.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}

	// Embed nil — the branch Assemble shadows.
	if err := (&model.LoadedModel{Embed: nil}).ValidateRequired(arch); err == nil {
		t.Fatal("validateRequired should reject a nil Embed")
	}

	// Final norm nil (Embed present).
	if err := (&model.LoadedModel{Embed: &mLinearStub}).ValidateRequired(arch); err == nil {
		t.Fatal("validateRequired should reject a nil FinalNorm")
	}

	// A layer missing AttnNorm/Q/O.
	m := &model.LoadedModel{Embed: &mLinearStub, FinalNorm: []byte{1}, Layers: []model.LoadedLayer{{}}}
	if err := m.ValidateRequired(arch); err == nil {
		t.Fatal("validateRequired should reject a layer missing input_layernorm/q_proj/o_proj")
	}

	// Cache-owner layer present but missing K.
	m = &model.LoadedModel{Embed: &mLinearStub, FinalNorm: []byte{1}, Layers: []model.LoadedLayer{{
		AttnNorm: []byte{1}, Q: &mLinearStub, O: &mLinearStub, // owner layer, K nil
		MLPNorm: []byte{1}, Gate: &mLinearStub, Up: &mLinearStub, Down: &mLinearStub,
	}}}
	if !arch.Layer[0].OwnsCache() {
		t.Fatal("test arch's layer 0 must own its cache for this branch")
	}
	if err := m.ValidateRequired(arch); err == nil {
		t.Fatal("validateRequired should reject a cache-owner layer missing k_proj")
	}

	// Dense layer (MoE nil) missing a required MLP weight (MLPNorm).
	m = &model.LoadedModel{Embed: &mLinearStub, FinalNorm: []byte{1}, Layers: []model.LoadedLayer{{
		AttnNorm: []byte{1}, Q: &mLinearStub, K: &mLinearStub, O: &mLinearStub,
		Gate: &mLinearStub, Up: &mLinearStub, Down: &mLinearStub, // MLPNorm missing
	}}}
	if err := m.ValidateRequired(arch); err == nil {
		t.Fatal("validateRequired should reject a dense layer missing a required MLP weight")
	}
	t.Logf("validateRequired: nil embed / nil final-norm / missing attn / missing owner-K / missing dense-MLP all rejected")
}

// TestAssembleMoEBranch covers the MoE assembly path (load.go:150 + assembleMoE, both 0% before):
// a MoE-flagged arch assembled from the minimal tensor set succeeds with the MoE sub-linears nil
// (validateRequired has no MoE-weight checks), and the resulting layer carries a non-nil MoE.
func TestAssembleMoEBranch(t *testing.T) {
	arch, err := Config{
		HiddenSize: 64, NumHiddenLayers: 2, IntermediateSize: 128,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 16, VocabSize: 32, RMSNormEps: 1e-6,
		EnableMoEBlock: true, NumExperts: 4, TopKExperts: 2, MoEIntermediateSize: 96,
	}.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	ts := minimalGemma4Tensors(arch)
	m, err := gemma4Assemble(ts, arch)
	if err != nil {
		t.Fatalf("Assemble of a MoE arch (sparse weights absent): %v", err)
	}
	for i, L := range m.Layers {
		if L.MoE == nil {
			t.Fatalf("layer %d should carry a non-nil MoE (gemma4 applies MoE uniformly)", i)
		}
	}
	t.Logf("Assemble: MoE branch builds a model.LoadedMoE per layer (assembleMoE reached)")
}

// TestAssembleMissingWeightBranches drives the Assemble-reachable validateRequired failures by
// deleting one required weight at a time from a complete dense set, plus the missing-final-norm and
// missing-owner-K cases.
func TestAssembleMissingWeightBranches(t *testing.T) {
	arch, err := Config{
		HiddenSize: 64, NumHiddenLayers: 2, IntermediateSize: 128,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 16, VocabSize: 32, RMSNormEps: 1e-6,
	}.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	for _, name := range []string{
		"model.norm.weight",                               // final norm → validateRequired:185
		"model.layers.0.self_attn.o_proj.weight",          // o_proj → :190
		"model.layers.0.self_attn.k_proj.weight",          // cache-owner K → :193
		"model.layers.0.pre_feedforward_layernorm.weight", // dense MLP norm → :196
		"model.layers.0.mlp.down_proj.weight",             // dense MLP down → :196
	} {
		ts := minimalGemma4Tensors(arch)
		delete(ts, name)
		if _, err := gemma4Assemble(ts, arch); err == nil {
			t.Fatalf("Assemble should fail with %q deleted", name)
		}
	}

	// Missing embed → the early Assemble check (load.go:118), not validateRequired.
	ts := minimalGemma4Tensors(arch)
	delete(ts, "model.embed_tokens.weight")
	if _, err := gemma4Assemble(ts, arch); err == nil {
		t.Fatal("Assemble should fail with the embedding deleted")
	}
	t.Logf("Assemble: each deleted required weight (embed/final-norm/o_proj/owner-K/dense-MLP) rejected")
}

// TestLoadFromDir is the on-disk Load entry: a synthetic bf16 checkpoint (config.json +
// model.safetensors) written to t.TempDir() loads, returns a model.LoadedModel whose layer count matches
// the arch, and the DirMapping closes clean. AX-11: mmap metadata only, no compute / GPU. Also
// covers Load's error paths (missing config, bad config, missing weights dir, malformed arch).
func TestLoadFromDir(t *testing.T) {
	cfg := Config{
		HiddenSize: 64, NumHiddenLayers: 2, IntermediateSize: 128,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 16, VocabSize: 32, RMSNormEps: 1e-6,
		SlidingWindow: 32, MaxPositionEmbeddings: 128, NumKVSharedLayers: 1,
		LayerTypes: []string{"full_attention", "sliding_attention"},
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	dir := writeGemma4Dir(t, cfg, minimalGemma4Tensors(arch))

	m, dm, err := model.Load(dir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer dm.Close()
	if len(m.Layers) != len(arch.Layer) {
		t.Fatalf("loaded %d layers, arch has %d", len(m.Layers), len(arch.Layer))
	}
	if m.Embed == nil || m.FinalNorm == nil {
		t.Fatal("loaded model missing embed / final norm")
	}
	if !m.Tied() {
		t.Fatal("a checkpoint with no lm_head weight should load tied")
	}
	if err := dm.Close(); err != nil {
		t.Fatalf("DirMapping.Close: %v", err)
	}
	t.Logf("Load: synthetic %d-layer bf16 checkpoint loaded + closed clean", len(m.Layers))
}

func TestLoadFromDirCarriesVisionPayload_Good(t *testing.T) {
	cfg := Config{
		HiddenSize: 64, NumHiddenLayers: 1, IntermediateSize: 128,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 16, VocabSize: 32, RMSNormEps: 1e-6,
		SlidingWindow: 32, MaxPositionEmbeddings: 128,
		LayerTypes: []string{"full_attention"},
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	ts := minimalGemma4Tensors(arch)
	addMinimalVisionTensors(ts, 64, 1)
	dir := writeGemma4Dir(t, cfg, ts)

	m, dm, err := model.Load(dir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer dm.Close()
	if m.Vision == nil {
		t.Fatal("loaded model Vision = nil, want gathered vision payload")
	}
	if len(m.Vision.Layers) != 1 {
		t.Fatalf("vision layers = %d, want 1", len(m.Vision.Layers))
	}
	if m.Vision.PatchEmbedding == nil || m.Vision.Projector.Projection.Weight == nil {
		t.Fatal("vision payload missing patch embedding or projector")
	}
}

// TestLoadErrors covers Load's four error branches: missing config.json, unparseable config.json,
// no safetensors in the dir, and a config that parses but fails Arch validation.
func TestLoadErrors(t *testing.T) {
	t.Run("missing config", func(t *testing.T) {
		if _, _, err := model.Load(t.TempDir()); err == nil {
			t.Fatal("Load should fail when config.json is absent")
		}
	})

	t.Run("unparseable config", func(t *testing.T) {
		dir := t.TempDir()
		writeFile(t, dir, "config.json", `{not valid json`)
		if _, _, err := model.Load(dir); err == nil {
			t.Fatal("Load should fail on an unparseable config.json")
		}
	})

	t.Run("config fails Arch", func(t *testing.T) {
		dir := t.TempDir()
		// hidden_size 0 → Arch validation rejects it before the weights are touched.
		writeFile(t, dir, "config.json", `{"hidden_size":0,"num_hidden_layers":2,"num_attention_heads":8}`)
		if _, _, err := model.Load(dir); err == nil {
			t.Fatal("Load should surface the Arch validation error")
		}
	})

	t.Run("no safetensors", func(t *testing.T) {
		dir := t.TempDir()
		cfg := Config{
			HiddenSize: 64, NumHiddenLayers: 1, IntermediateSize: 128,
			NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 16, VocabSize: 32, RMSNormEps: 1e-6,
		}
		cj := core.JSONMarshal(cfg)
		if !cj.OK {
			t.Fatal("marshal config")
		}
		writeFile(t, dir, "config.json", string(cj.Value.([]byte)))
		// config present, but no model.safetensors / index → LoadDirMmap fails.
		if _, _, err := model.Load(dir); err == nil {
			t.Fatal("Load should fail when no safetensors file is present")
		}
	})

	t.Run("incomplete weights", func(t *testing.T) {
		cfg := Config{
			HiddenSize: 64, NumHiddenLayers: 1, IntermediateSize: 128,
			NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 16, VocabSize: 32, RMSNormEps: 1e-6,
		}
		arch, err := cfg.Arch()
		if err != nil {
			t.Fatalf("Arch: %v", err)
		}
		ts := minimalGemma4Tensors(arch)
		delete(ts, "model.layers.0.self_attn.q_proj.weight") // Assemble rejects → Load closes the mmap + errors
		dir := writeGemma4Dir(t, cfg, ts)
		if _, _, err := model.Load(dir); err == nil {
			t.Fatal("Load should fail (and close the mmap) when Assemble rejects an incomplete set")
		}
	})
	t.Logf("Load errors: missing/unparseable/invalid config, no safetensors, incomplete weights all surfaced")
}

// writeGemma4Dir writes config.json + a single model.safetensors (the tensor set) to a fresh temp dir
// and returns it — the on-disk shape model.Load reads. It stamps model_type (gemma4_text) onto the
// config the way a real gemma4 checkpoint declares it, since Config carries none and model.Load
// dispatches on it.
func writeGemma4Dir(t *testing.T, cfg Config, ts map[string]safetensors.Tensor) string {
	t.Helper()
	dir := t.TempDir()
	cj := core.JSONMarshal(cfg)
	if !cj.OK {
		t.Fatal("marshal config")
	}
	var m map[string]any
	if r := core.JSONUnmarshal(cj.Value.([]byte), &m); !r.OK {
		t.Fatal("re-parse config for model_type")
	}
	m["model_type"] = "gemma4_text"
	if cj = core.JSONMarshal(m); !cj.OK {
		t.Fatal("re-marshal config")
	}
	writeFile(t, dir, "config.json", string(cj.Value.([]byte)))
	blob, err := safetensors.Encode(ts)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	writeFile(t, dir, "model.safetensors", string(blob))
	return dir
}

func addMinimalVisionTensors(ts map[string]safetensors.Tensor, hidden, layers int) {
	bf := func(n int) safetensors.Tensor {
		return safetensors.Tensor{Dtype: "BF16", Shape: []int{n}, Data: make([]byte, n*2)}
	}
	mat := func(out, in int) safetensors.Tensor {
		return safetensors.Tensor{Dtype: "BF16", Shape: []int{out, in}, Data: make([]byte, out*in*2)}
	}
	ts["vision_tower.embeddings.patch_embedding.weight"] = mat(hidden, 588)
	for i := range layers {
		p := core.Sprintf("vision_tower.encoder.layers.%d", i)
		for _, n := range []string{".input_layernorm", ".post_attention_layernorm", ".pre_feedforward_layernorm", ".post_feedforward_layernorm", ".self_attn.q_norm", ".self_attn.k_norm"} {
			ts[p+n+".weight"] = bf(hidden)
		}
		for _, n := range []string{".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj", ".self_attn.o_proj"} {
			ts[p+n+".weight"] = mat(hidden, hidden)
		}
		ts[p+".mlp.gate_proj.weight"] = mat(hidden*4, hidden)
		ts[p+".mlp.up_proj.weight"] = mat(hidden*4, hidden)
		ts[p+".mlp.down_proj.weight"] = mat(hidden, hidden*4)
	}
	ts["multi_modal_projector.proj.weight"] = mat(hidden, hidden)
}

func writeFile(t *testing.T, dir, name, content string) {
	t.Helper()
	if err := coreio.Local.Write(core.PathJoin(dir, name), content); err != nil {
		t.Fatalf("write %s: %v", name, err)
	}
}
