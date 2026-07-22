// SPDX-Licence-Identifier: EUPL-1.2

package qwen35

import (
	"testing"

	"dappco.re/go/inference/model/quant/mlxaffine"
	"dappco.re/go/inference/model/safetensors"
)

// vision_loader_test.go proves LoadVisionTower's assembly on synthetic tensors at small dimensions —
// the zoo-lane pattern (dbrx/qwenmoe load_test): the REAL fused-qkv/GELU/pos_embed layout, the GUESSED
// separate-projection/SwiGLU layout, the text-only nil return, the packed-quant path, and the loud
// failure modes (mixed conventions, geometry mismatches). The name-coverage receipt against the real
// mlx-community/Qwen3.6-27B-4bit snapshot lives in vision_real_test.go.

// tinyVisionDims is the synthetic tower geometry every test here shares: hidden 8, patch 2, 3
// channels, temporal 2 (patchDim 3·2·2·2 = 24), 2 heads (headDim 4), FF 12, merge 2, text hidden 16,
// 2 blocks, a 4×4 learned position table.
const (
	tvHidden     = 8
	tvPatchDim   = 24
	tvFF         = 12
	tvTextHidden = 16
	tvMergeIn    = tvHidden * 4 // hidden · mergeSize²
	tvPositions  = 16           // 4×4 square table
)

// tvConfigJSON is the multimodal-wrapper config the tiny towers load under: image_token_id at the
// wrapper root, text hidden nested under text_config, pixel geometry under vision_config.
const tvConfigJSON = `{"model_type":"qwen3_5","image_token_id":777,"video_token_id":778,` +
	`"vision_start_token_id":775,"vision_end_token_id":776,` +
	`"text_config":{"model_type":"qwen3_5_text","hidden_size":16,"num_hidden_layers":1,"num_attention_heads":2,"vocab_size":32},` +
	`"vision_config":{"patch_size":2,"in_channels":3,"num_heads":2,"spatial_merge_size":2,"rms_norm_eps":1e-6}}`

// tvSeq fills a deterministic pseudo-random f32 tensor (tiny LCG — no math/rand seeding drift).
func tvSeq(n int, seed uint32) []float32 {
	out := make([]float32, n)
	s := seed*2654435761 + 1
	for i := range out {
		s = s*1664525 + 1013904223
		out[i] = float32(int32(s>>9)%1000) / 1000
	}
	return out
}

func tvTensor(shape []int, seed uint32) safetensors.Tensor {
	return safetensors.Tensor{Dtype: "F32", Shape: shape, Data: safetensors.EncodeFloat32(tvSeq(numel(shape), seed))}
}

// tinyRealLayoutTensors is the REAL-layout synthetic tower: fused attn.qkv, GELU
// mlp.linear_fc1/linear_fc2, LayerNorm-with-bias norms, a learned pos_embed, the
// vision_tower.merger.* projector.
func tinyRealLayoutTensors(blocks int) map[string]safetensors.Tensor {
	t := map[string]safetensors.Tensor{
		// >2-D dense patch_embed — the real layout's [Hidden,T,P,P,C] shape whose row-major flatten
		// is byte-identical to [Hidden,PatchDim] (the _Ugly shape, exercised on the happy path).
		"vision_tower.patch_embed.proj.weight": tvTensor([]int{tvHidden, 2, 2, 2, 3}, 1),
		"vision_tower.patch_embed.proj.bias":   tvTensor([]int{tvHidden}, 2),
		"vision_tower.pos_embed.weight":        tvTensor([]int{tvPositions, tvHidden}, 3),
		"vision_tower.merger.norm.weight":      tvTensor([]int{tvHidden}, 4),
		"vision_tower.merger.norm.bias":        tvTensor([]int{tvHidden}, 5),
		"vision_tower.merger.linear_fc1.weight": tvTensor([]int{tvMergeIn, tvMergeIn}, 6),
		"vision_tower.merger.linear_fc1.bias":   tvTensor([]int{tvMergeIn}, 7),
		"vision_tower.merger.linear_fc2.weight": tvTensor([]int{tvTextHidden, tvMergeIn}, 8),
		"vision_tower.merger.linear_fc2.bias":   tvTensor([]int{tvTextHidden}, 9),
	}
	for i := range blocks {
		bp := "vision_tower.blocks." + string(rune('0'+i)) + "."
		seed := uint32(100 * (i + 1))
		t[bp+"norm1.weight"] = tvTensor([]int{tvHidden}, seed+1)
		t[bp+"norm1.bias"] = tvTensor([]int{tvHidden}, seed+2)
		t[bp+"norm2.weight"] = tvTensor([]int{tvHidden}, seed+3)
		t[bp+"norm2.bias"] = tvTensor([]int{tvHidden}, seed+4)
		t[bp+"attn.qkv.weight"] = tvTensor([]int{3 * tvHidden, tvHidden}, seed+5)
		t[bp+"attn.qkv.bias"] = tvTensor([]int{3 * tvHidden}, seed+6)
		t[bp+"attn.proj.weight"] = tvTensor([]int{tvHidden, tvHidden}, seed+7)
		t[bp+"attn.proj.bias"] = tvTensor([]int{tvHidden}, seed+8)
		t[bp+"mlp.linear_fc1.weight"] = tvTensor([]int{tvFF, tvHidden}, seed+9)
		t[bp+"mlp.linear_fc1.bias"] = tvTensor([]int{tvFF}, seed+10)
		t[bp+"mlp.linear_fc2.weight"] = tvTensor([]int{tvHidden, tvFF}, seed+11)
		t[bp+"mlp.linear_fc2.bias"] = tvTensor([]int{tvHidden}, seed+12)
	}
	return t
}

// tinyGuessedLayoutTensors is the GUESSED-layout synthetic tower: separate q/k/v/o (GQA — 2 query
// heads, 1 kv head), per-head q/k RMS norms, SwiGLU MLP, no pos_embed, the multi_modal_projector.*
// aliases.
func tinyGuessedLayoutTensors() map[string]safetensors.Tensor {
	const headDim, kvOut = 4, 4 // 1 kv head · headDim
	t := map[string]safetensors.Tensor{
		"vision_tower.patch_embed.weight":       tvTensor([]int{tvHidden, tvPatchDim}, 11),
		"vision_tower.patch_embed.bias":         tvTensor([]int{tvHidden}, 12),
		"multi_modal_projector.norm.weight":     tvTensor([]int{tvHidden}, 13),
		"multi_modal_projector.norm.bias":       tvTensor([]int{tvHidden}, 14),
		"multi_modal_projector.linear_1.weight": tvTensor([]int{tvMergeIn, tvMergeIn}, 15),
		"multi_modal_projector.linear_2.weight": tvTensor([]int{tvTextHidden, tvMergeIn}, 16),
	}
	bp := "vision_tower.blocks.0."
	t[bp+"norm1.weight"] = tvTensor([]int{tvHidden}, 21)
	t[bp+"norm2.weight"] = tvTensor([]int{tvHidden}, 22)
	t[bp+"attn.q_proj.weight"] = tvTensor([]int{tvHidden, tvHidden}, 23)
	t[bp+"attn.k_proj.weight"] = tvTensor([]int{kvOut, tvHidden}, 24)
	t[bp+"attn.v_proj.weight"] = tvTensor([]int{kvOut, tvHidden}, 25)
	t[bp+"attn.o_proj.weight"] = tvTensor([]int{tvHidden, tvHidden}, 26)
	t[bp+"attn.q_norm.weight"] = tvTensor([]int{headDim}, 27)
	t[bp+"attn.k_norm.weight"] = tvTensor([]int{headDim}, 28)
	t[bp+"mlp.gate_proj.weight"] = tvTensor([]int{tvFF, tvHidden}, 29)
	t[bp+"mlp.up_proj.weight"] = tvTensor([]int{tvFF, tvHidden}, 30)
	t[bp+"mlp.down_proj.weight"] = tvTensor([]int{tvHidden, tvFF}, 31)
	return t
}

func TestLoadVisionTower_RealLayout_Good(t *testing.T) {
	tower, err := LoadVisionTower(tinyRealLayoutTensors(2), []byte(tvConfigJSON))
	if err != nil {
		t.Fatalf("LoadVisionTower: %v", err)
	}
	if tower == nil {
		t.Fatal("LoadVisionTower returned nil tower for a vision checkpoint")
	}
	cfg := tower.Cfg
	if cfg.Hidden != tvHidden || cfg.PatchDim != tvPatchDim {
		t.Fatalf("derived hidden/patchDim = %d/%d, want %d/%d", cfg.Hidden, cfg.PatchDim, tvHidden, tvPatchDim)
	}
	if cfg.TemporalPatchSize != 2 {
		t.Fatalf("derived temporal patch size = %d, want 2 (patchDim %d / in_channels·patch² 12)", cfg.TemporalPatchSize, tvPatchDim)
	}
	if cfg.NumHeads != 2 || cfg.NumKVHeads != 2 || cfg.HeadDim != 4 {
		t.Fatalf("attention geometry = %d heads / %d kv / headDim %d, want 2/2/4 (fused qkv is plain MHA)", cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim)
	}
	if cfg.MergeSize != 2 {
		t.Fatalf("derived merge size = %d, want 2 (merger linear_fc1 input %d / hidden %d)", cfg.MergeSize, tvMergeIn, tvHidden)
	}
	if cfg.TextHidden != tvTextHidden {
		t.Fatalf("text hidden = %d, want %d (nested text_config.hidden_size)", cfg.TextHidden, tvTextHidden)
	}
	if !cfg.LearnedPositions || len(tower.PosEmbed) != tvPositions*tvHidden {
		t.Fatalf("learned positions = %v with table len %d, want true with %d", cfg.LearnedPositions, len(tower.PosEmbed), tvPositions*tvHidden)
	}
	if cfg.ImageTokenID != 777 {
		t.Fatalf("image token id = %d, want 777 (wrapper-root image_token_id)", cfg.ImageTokenID)
	}
	if len(tower.Blocks) != 2 {
		t.Fatalf("block count = %d, want 2", len(tower.Blocks))
	}
	b0 := tower.Blocks[0]
	if !b0.MLP.GELU || b0.MLP.FC1.Out != tvFF || b0.MLP.FC2.Out != tvHidden {
		t.Fatalf("block 0 MLP = gelu %v fc1 %dx%d fc2 %dx%d, want the 2-linear GELU shape", b0.MLP.GELU, b0.MLP.FC1.Out, b0.MLP.FC1.In, b0.MLP.FC2.Out, b0.MLP.FC2.In)
	}
	if b0.Norm1B == nil || b0.Norm2B == nil {
		t.Fatal("block 0 norm biases missing — the real layout's LayerNorm carries weight AND bias")
	}
	if b0.Attn.QNorm != nil || b0.Attn.KNorm != nil {
		t.Fatal("block 0 carries q/k norms — the real layout ships none (probe must not invent them)")
	}
	// Fused-split exactness: Q/K/V are the equal output-row thirds of the fused tensor, and the bias
	// splits the same way.
	fused := tvSeq(3*tvHidden*tvHidden, 105)
	fusedB := tvSeq(3*tvHidden, 106)
	for band, lin := range []VisionLinear{b0.Attn.Q, b0.Attn.K, b0.Attn.V} {
		if lin.Out != tvHidden || lin.In != tvHidden {
			t.Fatalf("band %d shape %dx%d, want %dx%d", band, lin.Out, lin.In, tvHidden, tvHidden)
		}
		for i, w := range lin.W {
			if w != fused[band*tvHidden*tvHidden+i] {
				t.Fatalf("band %d weight[%d] = %v, want the fused row-band value %v", band, i, w, fused[band*tvHidden*tvHidden+i])
			}
		}
		for i, b := range lin.B {
			if b != fusedB[band*tvHidden+i] {
				t.Fatalf("band %d bias[%d] = %v, want %v", band, i, b, fusedB[band*tvHidden+i])
			}
		}
	}
}

func TestLoadVisionTower_GuessedLayout_Good(t *testing.T) {
	tower, err := LoadVisionTower(tinyGuessedLayoutTensors(), []byte(tvConfigJSON))
	if err != nil {
		t.Fatalf("LoadVisionTower: %v", err)
	}
	if tower == nil {
		t.Fatal("LoadVisionTower returned nil tower for a vision checkpoint")
	}
	cfg := tower.Cfg
	if cfg.NumHeads != 2 || cfg.NumKVHeads != 1 || cfg.HeadDim != 4 {
		t.Fatalf("attention geometry = %d heads / %d kv / headDim %d, want 2/1/4 (headDim from q_norm width, kv from k_proj rows)", cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim)
	}
	if cfg.LearnedPositions || tower.PosEmbed != nil {
		t.Fatal("guessed layout derived learned positions — no pos_embed tensor ships in it")
	}
	b0 := tower.Blocks[0]
	if b0.MLP.GELU {
		t.Fatal("guessed layout resolved the GELU MLP — gate_proj presence must select SwiGLU")
	}
	if b0.Attn.QNorm == nil || b0.Attn.KNorm == nil {
		t.Fatal("guessed layout dropped the shipped q/k norms")
	}
	if b0.Norm1B != nil {
		t.Fatal("guessed layout invented a norm1 bias the checkpoint does not ship")
	}
}

func TestLoadVisionTower_TextOnly_Good(t *testing.T) {
	tower, err := LoadVisionTower(map[string]safetensors.Tensor{
		"model.embed_tokens.weight": tvTensor([]int{32, tvTextHidden}, 40),
	}, []byte(tvConfigJSON))
	if err != nil {
		t.Fatalf("LoadVisionTower on a text-only tensor set: %v", err)
	}
	if tower != nil {
		t.Fatal("text-only checkpoint produced a vision tower — the patch_embed probe must gate")
	}
}

// TestLoadVisionTower_Quantised_Good packs the fused qkv through the REAL mlx-affine quantiser and
// proves the loader dequantises it back to the packed values (mlxaffine round-trip), keeping every
// other projection dense — the per-weight quant decision.
func TestLoadVisionTower_Quantised_Good(t *testing.T) {
	tensors := tinyRealLayoutTensors(1)
	const bits, groupSize = 4, 8
	orig := tvSeq(3*tvHidden*tvHidden, 105) // block 0's qkv seed in tinyRealLayoutTensors
	packed, scales, biases, err := mlxaffine.QuantizeTensor(orig, 3*tvHidden, tvHidden, bits, groupSize)
	if err != nil {
		t.Fatalf("QuantizeTensor: %v", err)
	}
	tensors["vision_tower.blocks.0.attn.qkv.weight"] = safetensors.Tensor{Dtype: "U32", Shape: []int{3 * tvHidden, tvHidden * bits / 32}, Data: packed}
	tensors["vision_tower.blocks.0.attn.qkv.scales"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{3 * tvHidden, tvHidden / groupSize}, Data: scales}
	tensors["vision_tower.blocks.0.attn.qkv.biases"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{3 * tvHidden, tvHidden / groupSize}, Data: biases}

	quantCfg := `{"model_type":"qwen3_5","image_token_id":777,` +
		`"quantization":{"group_size":8,"bits":4},` +
		`"text_config":{"model_type":"qwen3_5_text","hidden_size":16,"num_hidden_layers":1,"num_attention_heads":2,"vocab_size":32},` +
		`"vision_config":{"patch_size":2,"in_channels":3,"num_heads":2,"spatial_merge_size":2}}`
	tower, err := LoadVisionTower(tensors, []byte(quantCfg))
	if err != nil {
		t.Fatalf("LoadVisionTower (packed qkv): %v", err)
	}
	want, err := mlxaffine.DequantizeTensor(packed, scales, biases, 3*tvHidden, tvHidden, bits, groupSize)
	if err != nil {
		t.Fatalf("DequantizeTensor: %v", err)
	}
	q := tower.Blocks[0].Attn.Q
	for i, w := range q.W {
		if w != want[i] {
			t.Fatalf("packed qkv Q[%d] = %v, want the dequantised value %v — the loader must land the exact DequantizeTensor output", i, w, want[i])
		}
	}
	if got := tower.Blocks[0].Attn.O; got.W == nil || got.Out != tvHidden {
		t.Fatal("dense sibling projections must stay on the plain widen path")
	}
}

func TestLoadVisionTower_BadQKVWidth_Bad(t *testing.T) {
	tensors := tinyRealLayoutTensors(1)
	tensors["vision_tower.blocks.0.attn.qkv.weight"] = tvTensor([]int{3*tvHidden + 1, tvHidden}, 50)
	if _, err := LoadVisionTower(tensors, []byte(tvConfigJSON)); err == nil {
		t.Fatal("a fused qkv width not divisible by 3 must fail loudly")
	}
}

func TestLoadVisionTower_MergerMismatch_Bad(t *testing.T) {
	tensors := tinyRealLayoutTensors(1)
	// linear_fc1 input 3·hidden — not hidden·M² for any integer M.
	tensors["vision_tower.merger.linear_fc1.weight"] = tvTensor([]int{tvMergeIn, 3 * tvHidden}, 51)
	if _, err := LoadVisionTower(tensors, []byte(tvConfigJSON)); err == nil {
		t.Fatal("a merger width that is not hidden·mergeSize² must fail loudly")
	}
}

func TestLoadVisionTower_SpatialMergeConfigMismatch_Bad(t *testing.T) {
	cfg := `{"model_type":"qwen3_5","text_config":{"model_type":"qwen3_5_text","hidden_size":16},` +
		`"vision_config":{"patch_size":2,"in_channels":3,"num_heads":2,"spatial_merge_size":3}}`
	if _, err := LoadVisionTower(tinyRealLayoutTensors(1), []byte(cfg)); err == nil {
		t.Fatal("a config spatial_merge_size disagreeing with the derived merge size must fail loudly")
	}
}

func TestLoadVisionTower_TextHiddenMismatch_Bad(t *testing.T) {
	cfg := `{"model_type":"qwen3_5","text_config":{"model_type":"qwen3_5_text","hidden_size":24},` +
		`"vision_config":{"patch_size":2,"in_channels":3,"num_heads":2,"spatial_merge_size":2}}`
	if _, err := LoadVisionTower(tinyRealLayoutTensors(1), []byte(cfg)); err == nil {
		t.Fatal("a merger output width disagreeing with the text hidden size must fail loudly")
	}
}

func TestLoadVisionTower_MissingPatchSize_Bad(t *testing.T) {
	cfg := `{"model_type":"qwen3_5","text_config":{"model_type":"qwen3_5_text","hidden_size":16}}`
	if _, err := LoadVisionTower(tinyRealLayoutTensors(1), []byte(cfg)); err == nil {
		t.Fatal("a vision tower without vision_config.patch_size must fail loudly (pixel geometry has no tensor to derive it from)")
	}
}

// TestLoadVisionTower_MixedConventions_Ugly: block 0 resolves the REAL fused convention, block 1
// ships only the GUESSED separate q_proj — the counting probe still SEES block 1 (q_proj present) but
// loading it under the decided fused convention fails loudly on the missing attn.qkv.
func TestLoadVisionTower_MixedConventions_Ugly(t *testing.T) {
	tensors := tinyRealLayoutTensors(1)
	bp := "vision_tower.blocks.1."
	tensors[bp+"norm1.weight"] = tvTensor([]int{tvHidden}, 60)
	tensors[bp+"norm2.weight"] = tvTensor([]int{tvHidden}, 61)
	tensors[bp+"attn.q_proj.weight"] = tvTensor([]int{tvHidden, tvHidden}, 62)
	tensors[bp+"attn.k_proj.weight"] = tvTensor([]int{tvHidden, tvHidden}, 63)
	tensors[bp+"attn.v_proj.weight"] = tvTensor([]int{tvHidden, tvHidden}, 64)
	tensors[bp+"attn.o_proj.weight"] = tvTensor([]int{tvHidden, tvHidden}, 65)
	tensors[bp+"mlp.gate_proj.weight"] = tvTensor([]int{tvFF, tvHidden}, 66)
	tensors[bp+"mlp.up_proj.weight"] = tvTensor([]int{tvFF, tvHidden}, 67)
	tensors[bp+"mlp.down_proj.weight"] = tvTensor([]int{tvHidden, tvFF}, 68)
	if _, err := LoadVisionTower(tensors, []byte(tvConfigJSON)); err == nil {
		t.Fatal("a checkpoint mixing attention conventions across blocks must fail loudly")
	}
}
