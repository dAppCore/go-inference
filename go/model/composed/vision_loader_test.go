// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"testing"

	"dappco.re/go/inference/model/safetensors"
)

// addVisionTensors appends a synthetic 2-block-shaped Qwen-VL vision_tower.*/multi_modal_projector.*
// tensor set to ts — Hidden=8, NumHeads=2, HeadDim=4, NumKVHeads=kvHeads, PatchSize=2, InChannels=3
// (PatchDim=12), FF=16, MergeSize=2 — matching mkVisionTower's geometry so loader-derived fields can be
// checked against known values. textHidden sizes the merger's output (must equal the paired text model's D
// for LoadComposed's cross-check to pass).
func addVisionTensors(ts map[string]safetensors.Tensor, nBlocks, kvHeads, textHidden int, withQKNorm bool) {
	const hidden, numHeads, headDim, ff, patchDim = 8, 2, 4, 16, 12
	ts["vision_tower.patch_embed.weight"] = bf16T(syn(hidden*patchDim, 9001), hidden, patchDim)
	for i := range nBlocks {
		bp := "vision_tower.blocks." + itoa(i) + "."
		s := 9100 + i*50
		ts[bp+"norm1.weight"] = bf16T(syn(hidden, s+1), hidden)
		ts[bp+"norm2.weight"] = bf16T(syn(hidden, s+2), hidden)
		ts[bp+"attn.q_proj.weight"] = bf16T(syn(numHeads*headDim*hidden, s+3), numHeads*headDim, hidden)
		ts[bp+"attn.k_proj.weight"] = bf16T(syn(kvHeads*headDim*hidden, s+4), kvHeads*headDim, hidden)
		ts[bp+"attn.v_proj.weight"] = bf16T(syn(kvHeads*headDim*hidden, s+5), kvHeads*headDim, hidden)
		ts[bp+"attn.o_proj.weight"] = bf16T(syn(hidden*numHeads*headDim, s+6), hidden, numHeads*headDim)
		if withQKNorm {
			ts[bp+"attn.q_norm.weight"] = bf16T(syn(headDim, s+7), headDim)
			ts[bp+"attn.k_norm.weight"] = bf16T(syn(headDim, s+8), headDim)
		}
		ts[bp+"mlp.gate_proj.weight"] = bf16T(syn(ff*hidden, s+9), ff, hidden)
		ts[bp+"mlp.up_proj.weight"] = bf16T(syn(ff*hidden, s+10), ff, hidden)
		ts[bp+"mlp.down_proj.weight"] = bf16T(syn(hidden*ff, s+11), hidden, ff)
	}
	const mergeSize = 2
	mergedIn := hidden * mergeSize * mergeSize
	ts["multi_modal_projector.norm.weight"] = bf16T(syn(hidden, 9500), hidden)
	ts["multi_modal_projector.linear_1.weight"] = bf16T(syn(mergedIn*mergedIn, 9501), mergedIn, mergedIn)
	ts["multi_modal_projector.linear_2.weight"] = bf16T(syn(textHidden*mergedIn, 9502), textHidden, mergedIn)
}

func TestBuildVisionTower_TextOnlyNil(t *testing.T) {
	ts, _ := mkHybridCheckpoint() // the existing text-only fixture — no vision_tower.* tensors at all
	tower, err := buildVisionTower(ts, nil, 8)
	if err != nil {
		t.Fatalf("buildVisionTower(text-only): %v", err)
	}
	if tower != nil {
		t.Fatal("buildVisionTower(text-only): want nil tower, got one — a text-only checkpoint must load unaffected")
	}
}

func TestBuildVisionTower_Good(t *testing.T) {
	ts := map[string]safetensors.Tensor{}
	addVisionTensors(ts, 2, 2, 8, true) // MHA (kvHeads=2), with q_norm/k_norm so headDim derives from them
	vc := &visionConfig{PatchSize: 2}
	tower, err := buildVisionTower(ts, vc, 8)
	if err != nil {
		t.Fatalf("buildVisionTower: %v", err)
	}
	if tower == nil {
		t.Fatal("buildVisionTower: want a tower, got nil")
	}
	if tower.Cfg.Hidden != 8 || tower.Cfg.PatchDim != 12 {
		t.Fatalf("Cfg.Hidden/PatchDim = %d/%d, want 8/12", tower.Cfg.Hidden, tower.Cfg.PatchDim)
	}
	if tower.Cfg.NumHeads != 2 || tower.Cfg.HeadDim != 4 || tower.Cfg.NumKVHeads != 2 {
		t.Fatalf("Cfg.NumHeads/HeadDim/NumKVHeads = %d/%d/%d, want 2/4/2 (derived from q_norm+q_proj+k_proj shapes)",
			tower.Cfg.NumHeads, tower.Cfg.HeadDim, tower.Cfg.NumKVHeads)
	}
	if tower.Cfg.TemporalPatchSize != 1 {
		t.Fatalf("Cfg.TemporalPatchSize = %d, want 1 (PatchDim 12 = 3·2·2·1)", tower.Cfg.TemporalPatchSize)
	}
	if tower.Cfg.MergeSize != 2 {
		t.Fatalf("Cfg.MergeSize = %d, want 2 (derived from merger linear_1 width)", tower.Cfg.MergeSize)
	}
	if tower.Cfg.TextHidden != 8 {
		t.Fatalf("Cfg.TextHidden = %d, want 8", tower.Cfg.TextHidden)
	}
	if len(tower.Blocks) != 2 {
		t.Fatalf("len(Blocks) = %d, want 2 (counting-probe over vision_tower.blocks.<i>.*)", len(tower.Blocks))
	}
	for i, b := range tower.Blocks {
		if len(b.Attn.QNorm) != 4 || len(b.Attn.KNorm) != 4 {
			t.Fatalf("block %d QNorm/KNorm len = %d/%d, want 4/4", i, len(b.Attn.QNorm), len(b.Attn.KNorm))
		}
	}
}

// TestBuildVisionTower_GoodNoQKNormFallsBackToConfig covers the OTHER head_dim derivation path: no q_norm
// tensor in the checkpoint, so buildVisionBlocks falls back to vision_config.num_heads (mirroring
// buildAttn's identical fallback for the text attention mixer).
func TestBuildVisionTower_GoodNoQKNormFallsBackToConfig(t *testing.T) {
	ts := map[string]safetensors.Tensor{}
	addVisionTensors(ts, 1, 1, 8, false) // GQA (kvHeads=1), no q_norm/k_norm
	vc := &visionConfig{PatchSize: 2, NumHeads: 2}
	tower, err := buildVisionTower(ts, vc, 8)
	if err != nil {
		t.Fatalf("buildVisionTower: %v", err)
	}
	if tower.Cfg.HeadDim != 4 || tower.Cfg.NumHeads != 2 || tower.Cfg.NumKVHeads != 1 {
		t.Fatalf("Cfg.HeadDim/NumHeads/NumKVHeads = %d/%d/%d, want 4/2/1 (config num_heads fallback + GQA)",
			tower.Cfg.HeadDim, tower.Cfg.NumHeads, tower.Cfg.NumKVHeads)
	}
}

func TestBuildVisionTower_BadMissingPatchSize(t *testing.T) {
	ts := map[string]safetensors.Tensor{}
	addVisionTensors(ts, 1, 2, 8, true)
	if _, err := buildVisionTower(ts, nil, 8); err == nil {
		t.Fatal("buildVisionTower: want an error when vision_config.patch_size is missing, got nil")
	}
}

func TestBuildVisionTower_BadHeadDimUnderivable(t *testing.T) {
	ts := map[string]safetensors.Tensor{}
	addVisionTensors(ts, 1, 2, 8, false) // no q_norm
	vc := &visionConfig{PatchSize: 2}    // and no NumHeads fallback either
	if _, err := buildVisionTower(ts, vc, 8); err == nil {
		t.Fatal("buildVisionTower: want an error when head_dim cannot be derived, got nil")
	}
}

func TestBuildVisionTower_BadMergerTextHiddenMismatch(t *testing.T) {
	ts := map[string]safetensors.Tensor{}
	addVisionTensors(ts, 1, 2, 8, true)
	vc := &visionConfig{PatchSize: 2}
	if _, err := buildVisionTower(ts, vc, 99); err == nil {
		t.Fatal("buildVisionTower: want an error when the merger output width != textHidden, got nil")
	}
}

func TestBuildVisionBlocks_BadMissingKProj(t *testing.T) {
	ts := map[string]safetensors.Tensor{}
	addVisionTensors(ts, 1, 2, 8, true)
	delete(ts, "vision_tower.blocks.0.attn.k_proj.weight")
	if _, _, _, _, err := buildVisionBlocks(ts, 8, &visionConfig{PatchSize: 2}); err == nil {
		t.Fatal("buildVisionBlocks: want an error for a missing k_proj, got nil")
	}
}

func TestBuildVisionMerger_BadNonSquareMerge(t *testing.T) {
	ts := map[string]safetensors.Tensor{}
	// linear_1 input width 24 is a multiple of hidden(8) by 3 — not a perfect square.
	ts["multi_modal_projector.linear_1.weight"] = bf16T(syn(24*24, 1), 24, 24)
	ts["multi_modal_projector.linear_2.weight"] = bf16T(syn(8*24, 2), 8, 24)
	ts["multi_modal_projector.norm.weight"] = bf16T(syn(8, 3), 8)
	if _, _, err := buildVisionMerger(ts, 8, 8, nil); err == nil {
		t.Fatal("buildVisionMerger: want an error for a non-square merge size, got nil")
	}
}

func TestOptionalVisionVec_Good(t *testing.T) {
	ts := map[string]safetensors.Tensor{"x": bf16T([]float32{1, 2, 3}, 3)}
	got := optionalVisionVec(ts, "x")
	if len(got) != 3 || got[0] != 1 || got[1] != 2 || got[2] != 3 {
		t.Fatalf("optionalVisionVec = %v, want [1 2 3]", got)
	}
}

func TestOptionalVisionVec_Bad(t *testing.T) {
	if got := optionalVisionVec(map[string]safetensors.Tensor{}, "missing"); got != nil {
		t.Fatalf("optionalVisionVec(missing) = %v, want nil", got)
	}
}

// TestLoadComposedVisionTower_Good is the flagship integration: a checkpoint that carries BOTH the Qwen
// 3.6 hybrid text stack AND a vision tower loads both, and the text stack still forwards correctly — vision
// loading is additive-only, never disturbing the existing text path.
func TestLoadComposedVisionTower_Good(t *testing.T) {
	ts, _ := mkHybridCheckpoint() // D=8, 4 layers (gated-delta + full attention), vocab 32
	addVisionTensors(ts, 2, 2, 8, true)
	cfg := []byte(`{"hidden_size":8,"num_hidden_layers":4,"intermediate_size":16,"num_attention_heads":4,
		"num_key_value_heads":2,"head_dim":8,"vocab_size":32,"rms_norm_eps":1e-5,"rope_theta":1000000,
		"partial_rotary_factor":0.5,"full_attention_interval":2,"image_token_id":1234,
		"vision_config":{"patch_size":2}}`)

	m, err := LoadComposed(ts, cfg)
	if err != nil {
		t.Fatalf("LoadComposed: %v", err)
	}
	if len(m.Layers) != 4 {
		t.Fatalf("len(m.Layers) = %d, want 4 (vision loading must not disturb the text stack)", len(m.Layers))
	}
	if m.Vision == nil {
		t.Fatal("m.Vision is nil, want a loaded tower")
	}
	if len(m.Vision.Blocks) != 2 {
		t.Fatalf("len(m.Vision.Blocks) = %d, want 2", len(m.Vision.Blocks))
	}
	if m.ImageTokenID != 1234 {
		t.Fatalf("m.ImageTokenID = %d, want 1234", m.ImageTokenID)
	}
	if m.VisionBeginToken != qwenVisionBeginToken || m.VisionToken != qwenVisionToken || m.VisionEndToken != qwenVisionEndToken {
		t.Fatalf("vision placeholder tokens = %q/%q/%q, want the Qwen-VL family literals", m.VisionBeginToken, m.VisionToken, m.VisionEndToken)
	}
	// The base hybrid stack forwards exactly as it does without a vision tensor in sight (loader_test.go's
	// own TestComposedHybridQwen35Integration proves the SAME checkpoint shape text-only; this proves the
	// vision-carrying twin still forwards).
	if _, err := NewSession(m).Forward([]int32{1, 5, 9}); err != nil {
		t.Fatalf("text forward on a vision-carrying model: %v", err)
	}
}

// TestLoadComposedVisionTower_TextOnlyFieldsZero is done-gate #1's direct proof for the NEW fields
// specifically: a text-only checkpoint (no vision_tower.* tensors) loads with every vision field at its
// zero value — Vision nil, ImageTokenID 0, no placeholder tokens.
func TestLoadComposedVisionTower_TextOnlyFieldsZero(t *testing.T) {
	ts, cfg := mkHybridCheckpoint()
	m, err := LoadComposed(ts, cfg)
	if err != nil {
		t.Fatalf("LoadComposed: %v", err)
	}
	if m.Vision != nil {
		t.Fatal("m.Vision is non-nil for a text-only checkpoint")
	}
	if m.ImageTokenID != 0 {
		t.Fatalf("m.ImageTokenID = %d, want 0", m.ImageTokenID)
	}
	if m.VisionBeginToken != "" || m.VisionToken != "" || m.VisionEndToken != "" {
		t.Fatalf("vision placeholder tokens = %q/%q/%q, want all empty", m.VisionBeginToken, m.VisionToken, m.VisionEndToken)
	}
}
