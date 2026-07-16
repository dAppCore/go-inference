// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"testing"

	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/quant/mlxaffine"
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

// ---------------------------------------------------------------------------------------------------------
// The REAL layout (verified against mlx-community/Qwen3.6-27B-4bit — see vision_loader_real_test.go's
// reconciliation receipt): fused attn.qkv, plain 2-linear GELU mlp.linear_fc1/linear_fc2,
// vision_tower.merger.* naming, a learned vision_tower.pos_embed.weight, and NO q_norm/k_norm.
// ---------------------------------------------------------------------------------------------------------

// addRealLayoutVisionTensors appends a synthetic nBlocks-deep REAL-layout tensor set to ts — Hidden=8,
// NumHeads=2, HeadDim=4 (perQKV=8=Hidden, plain MHA — the fused-qkv convention's only shape), PatchSize=2,
// InChannels=3 (PatchDim=12), FF=16, MergeSize=2, NumPositions=16 (a 4x4 grid, matching the 4x4 grids the
// forward-level tests below patchify) — matching mkRealLayoutVisionTower-shaped expectations. textHidden
// sizes the merger's output.
func addRealLayoutVisionTensors(ts map[string]safetensors.Tensor, nBlocks, textHidden int) {
	const hidden, numHeads, headDim, ff, patchDim, numPositions = 8, 2, 4, 16, 12, 16
	const perQKV = numHeads * headDim // = hidden: the fused-qkv convention is plain MHA, never GQA
	ts["vision_tower.patch_embed.proj.weight"] = bf16T(syn(hidden*patchDim, 8001), hidden, patchDim)
	ts["vision_tower.patch_embed.proj.bias"] = bf16T(syn(hidden, 8002), hidden)
	ts["vision_tower.pos_embed.weight"] = bf16T(syn(numPositions*hidden, 8003), numPositions, hidden)
	for i := range nBlocks {
		bp := "vision_tower.blocks." + itoa(i) + "."
		s := 8100 + i*50
		ts[bp+"norm1.weight"] = bf16T(syn(hidden, s+1), hidden)
		ts[bp+"norm1.bias"] = bf16T(syn(hidden, s+2), hidden)
		ts[bp+"norm2.weight"] = bf16T(syn(hidden, s+3), hidden)
		ts[bp+"norm2.bias"] = bf16T(syn(hidden, s+4), hidden)
		ts[bp+"attn.qkv.weight"] = bf16T(syn(3*perQKV*hidden, s+5), 3*perQKV, hidden)
		ts[bp+"attn.qkv.bias"] = bf16T(syn(3*perQKV, s+6), 3*perQKV)
		ts[bp+"attn.proj.weight"] = bf16T(syn(hidden*perQKV, s+7), hidden, perQKV)
		ts[bp+"attn.proj.bias"] = bf16T(syn(hidden, s+8), hidden)
		ts[bp+"mlp.linear_fc1.weight"] = bf16T(syn(ff*hidden, s+9), ff, hidden)
		ts[bp+"mlp.linear_fc1.bias"] = bf16T(syn(ff, s+10), ff)
		ts[bp+"mlp.linear_fc2.weight"] = bf16T(syn(hidden*ff, s+11), hidden, ff)
		ts[bp+"mlp.linear_fc2.bias"] = bf16T(syn(hidden, s+12), hidden)
	}
	const mergeSize = 2
	mergedIn := hidden * mergeSize * mergeSize
	ts["vision_tower.merger.norm.weight"] = bf16T(syn(hidden, 8500), hidden)
	ts["vision_tower.merger.norm.bias"] = bf16T(syn(hidden, 8501), hidden)
	ts["vision_tower.merger.linear_fc1.weight"] = bf16T(syn(mergedIn*mergedIn, 8502), mergedIn, mergedIn)
	ts["vision_tower.merger.linear_fc1.bias"] = bf16T(syn(mergedIn, 8503), mergedIn)
	ts["vision_tower.merger.linear_fc2.weight"] = bf16T(syn(textHidden*mergedIn, 8504), textHidden, mergedIn)
	ts["vision_tower.merger.linear_fc2.bias"] = bf16T(syn(textHidden, 8505), textHidden)
}

// dequantiseVisionTowerInPlace rewrites tower's packed projections to their dense f32 equivalents in
// place — the vision-tower twin of composed_quant_test.go's dequantiseInPlace, used to prove the packed
// forward path (matNTQuant → matNTQuantHost) matches the dense reference bit-for-bit. Safe to call on a
// visionBlock in either MLP mode: deq no-ops on a zero-value visionLinear (WQ nil), so the unused
// Gate/Up/Down (GELU mode) or FC1/FC2 (SwiGLU mode) fields are untouched.
func dequantiseVisionTowerInPlace(t *testing.T, tower *visionTower) {
	t.Helper()
	deq := func(lin *visionLinear) {
		if lin.WQ == nil {
			return
		}
		v, err := mlxaffine.DequantizeTensor(lin.WQ.Packed, lin.WQ.Scales, lin.WQ.Biases, lin.WQ.OutDim, lin.WQ.InDim, lin.WQ.Bits, lin.WQ.GroupSize)
		if err != nil {
			t.Fatalf("dequantise: %v", err)
		}
		lin.W, lin.WQ = v, nil
	}
	deq(&tower.Patch)
	for i := range tower.Blocks {
		b := &tower.Blocks[i]
		deq(&b.Attn.Q)
		deq(&b.Attn.K)
		deq(&b.Attn.V)
		deq(&b.Attn.O)
		deq(&b.MLP.Gate)
		deq(&b.MLP.Up)
		deq(&b.MLP.Down)
		deq(&b.MLP.FC1)
		deq(&b.MLP.FC2)
	}
	deq(&tower.Merger.L1)
	deq(&tower.Merger.L2)
}

func TestWeightAnyName_Good(t *testing.T) {
	ts := map[string]safetensors.Tensor{"b": bf16T([]float32{1}, 1)}
	name, tv, ok := weightAnyName(ts, "a", "b", "c")
	if !ok || name != "b" || len(tv.Data) == 0 {
		t.Fatalf("weightAnyName(a,b,c) = (%q,%v,%v), want (\"b\",<tensor>,true)", name, tv, ok)
	}
}

func TestWeightAnyName_Bad(t *testing.T) {
	if name, _, ok := weightAnyName(map[string]safetensors.Tensor{}, "a", "b"); ok {
		t.Fatalf("weightAnyName: want ok=false when none of the names are present, got (%q,true)", name)
	}
}

func TestBiasSibling_Good(t *testing.T) {
	if got := biasSibling("vision_tower.patch_embed.proj.weight"); got != "vision_tower.patch_embed.proj.bias" {
		t.Fatalf("biasSibling = %q, want %q", got, "vision_tower.patch_embed.proj.bias")
	}
	if got := biasSibling("no_weight_suffix"); got != "no_weight_suffix.bias" {
		t.Fatalf("biasSibling(no .weight suffix) = %q, want %q", got, "no_weight_suffix.bias")
	}
}

func TestVisionProj_Good(t *testing.T) {
	ts := map[string]safetensors.Tensor{"w.weight": bf16T(syn(6, 1), 2, 3)}
	lin, alias, err := visionProj(ts, "w.weight", ts["w.weight"], 2, 3, nil, false)
	if err != nil {
		t.Fatalf("visionProj: %v", err)
	}
	if alias {
		t.Fatal("alias = true, want false (the dense path never aliases)")
	}
	if lin.WQ != nil || lin.W == nil || lin.Out != 2 || lin.In != 3 {
		t.Fatalf("visionProj = %+v, want a dense [2,3] visionLinear", lin)
	}
}

func TestVisionProj_Quant(t *testing.T) {
	const out, in, bits, gs = 4, 8, 4, 8
	dense := syn(out*in, 1)
	packed, scales, biases, err := mlxaffine.QuantizeTensor(dense, out, in, bits, gs)
	if err != nil {
		t.Fatalf("QuantizeTensor: %v", err)
	}
	ts := map[string]safetensors.Tensor{
		"w.weight": {Dtype: "U32", Shape: []int{out, mlxaffine.PackedWords(in, bits)}, Data: packed},
		"w.scales": {Dtype: "BF16", Shape: []int{out, in / gs}, Data: scales},
		"w.biases": {Dtype: "BF16", Shape: []int{out, in / gs}, Data: biases},
	}
	quant := &model.QuantConfig{GroupSize: gs, Bits: bits}
	lin, alias, err := visionProj(ts, "w.weight", ts["w.weight"], out, in, quant, false)
	if err != nil {
		t.Fatalf("visionProj: %v", err)
	}
	if alias {
		t.Fatal("alias = true, want false (zeroCopy=false)")
	}
	if lin.WQ == nil || lin.W != nil {
		t.Fatalf("visionProj = %+v, want a packed (WQ set, W nil) visionLinear", lin)
	}
	if lin.WQ.OutDim != out || lin.WQ.InDim != in {
		t.Fatalf("visionProj.WQ.OutDim/InDim = %d/%d, want %d/%d", lin.WQ.OutDim, lin.WQ.InDim, out, in)
	}
}

func TestVisionProj_Bad(t *testing.T) {
	ts := map[string]safetensors.Tensor{"w.weight": bf16T(syn(6, 1), 2, 3)}
	if _, _, err := visionProj(ts, "w.weight", ts["w.weight"], 2, 99, nil, false); err == nil {
		t.Fatal("visionProj: want an error for an outDim*inDim width mismatch, got nil")
	}
}

// TestSplitFusedQKV_Good proves the dense split is an EXACT row-band slice: q/k/v(rows/bias) reconstructed
// from the fused visionLinear equal the original per-branch slices bit-for-bit.
func TestSplitFusedQKV_Good(t *testing.T) {
	const per, in = 3, 5
	q, k, v := syn(per*in, 1), syn(per*in, 2), syn(per*in, 3)
	qB, kB, vB := syn(per, 4), syn(per, 5), syn(per, 6)
	fused := visionLinear{
		W:   append(append(append([]float32(nil), q...), k...), v...),
		B:   append(append(append([]float32(nil), qB...), kB...), vB...),
		Out: 3 * per, In: in,
	}
	gotQ, gotK, gotV, err := splitFusedQKV(fused)
	if err != nil {
		t.Fatalf("splitFusedQKV: %v", err)
	}
	check := func(name string, got visionLinear, wantW, wantB []float32) {
		t.Helper()
		if got.Out != per || got.In != in {
			t.Fatalf("%s: Out/In = %d/%d, want %d/%d", name, got.Out, got.In, per, in)
		}
		for i := range wantW {
			if got.W[i] != wantW[i] {
				t.Fatalf("%s.W[%d] = %v, want %v", name, i, got.W[i], wantW[i])
			}
		}
		for i := range wantB {
			if got.B[i] != wantB[i] {
				t.Fatalf("%s.B[%d] = %v, want %v", name, i, got.B[i], wantB[i])
			}
		}
	}
	check("q", gotQ, q, qB)
	check("k", gotK, k, kB)
	check("v", gotV, v, vB)
}

// TestSplitFusedQKV_Quant proves the PACKED split lands exactly on group boundaries: dequantising each
// split band must equal the corresponding row-slice of the WHOLE fused tensor dequantised once — the same
// "split then dequantise" == "dequantise then slice" identity a correct byte-offset row-band split gives,
// avoiding any re-quantisation-rounding ambiguity a "quantise the sub-slice directly" comparison would risk.
func TestSplitFusedQKV_Quant(t *testing.T) {
	const per, in, bits, gs = 4, 8, 4, 8
	fusedDense := syn(3*per*in, 555)
	packed, scales, biases, err := mlxaffine.QuantizeTensor(fusedDense, 3*per, in, bits, gs)
	if err != nil {
		t.Fatalf("QuantizeTensor: %v", err)
	}
	wantWhole, err := mlxaffine.DequantizeTensor(packed, scales, biases, 3*per, in, bits, gs)
	if err != nil {
		t.Fatalf("DequantizeTensor: %v", err)
	}
	fused := visionLinear{
		WQ:  &model.QuantWeight{Packed: packed, Scales: scales, Biases: biases, Bits: bits, GroupSize: gs, OutDim: 3 * per, InDim: in},
		Out: 3 * per, In: in,
	}
	gotQ, gotK, gotV, err := splitFusedQKV(fused)
	if err != nil {
		t.Fatalf("splitFusedQKV: %v", err)
	}
	check := func(name string, got visionLinear, wantRows []float32) {
		t.Helper()
		if got.WQ == nil {
			t.Fatalf("%s: WQ is nil, want packed", name)
		}
		deq, err := mlxaffine.DequantizeTensor(got.WQ.Packed, got.WQ.Scales, got.WQ.Biases, got.WQ.OutDim, got.WQ.InDim, got.WQ.Bits, got.WQ.GroupSize)
		if err != nil {
			t.Fatalf("%s: dequantise split band: %v", name, err)
		}
		if len(deq) != len(wantRows) {
			t.Fatalf("%s: len = %d, want %d", name, len(deq), len(wantRows))
		}
		for i := range wantRows {
			if deq[i] != wantRows[i] {
				t.Fatalf("%s[%d] = %v, want %v", name, i, deq[i], wantRows[i])
			}
		}
	}
	check("q", gotQ, wantWhole[0:per*in])
	check("k", gotK, wantWhole[per*in:2*per*in])
	check("v", gotV, wantWhole[2*per*in:3*per*in])
}

func TestSplitFusedQKV_Bad(t *testing.T) {
	fused := visionLinear{W: syn(10, 1), Out: 10, In: 1} // 10 is not divisible by 3
	if _, _, _, err := splitFusedQKV(fused); err == nil {
		t.Fatal("splitFusedQKV: want an error when Out is not divisible by 3, got nil")
	}
}

func TestResolveVisionAttnGeometry_GoodFused(t *testing.T) {
	ts := map[string]safetensors.Tensor{"vision_tower.blocks.0.attn.qkv.weight": bf16T(syn(24*8, 1), 24, 8)}
	fused, headDim, numHeads, numKVHeads, err := resolveVisionAttnGeometry(ts, "vision_tower.blocks.0.", &visionConfig{NumHeads: 2}, nil)
	if err != nil {
		t.Fatalf("resolveVisionAttnGeometry: %v", err)
	}
	if !fused {
		t.Fatal("fused = false, want true (attn.qkv.weight present)")
	}
	if headDim != 4 || numHeads != 2 || numKVHeads != 2 {
		t.Fatalf("headDim/numHeads/numKVHeads = %d/%d/%d, want 4/2/2 (config num_heads fallback, plain MHA)", headDim, numHeads, numKVHeads)
	}
}

// TestResolveVisionAttnGeometry_GoodFusedQNorm covers the OTHER head_dim derivation path for the fused
// layout: a q_norm tensor settles HeadDim directly, no vision_config.num_heads needed.
func TestResolveVisionAttnGeometry_GoodFusedQNorm(t *testing.T) {
	ts := map[string]safetensors.Tensor{"vision_tower.blocks.0.attn.qkv.weight": bf16T(syn(24*8, 1), 24, 8)}
	qNorm := make([]float32, 4)
	fused, headDim, numHeads, numKVHeads, err := resolveVisionAttnGeometry(ts, "vision_tower.blocks.0.", nil, qNorm)
	if err != nil {
		t.Fatalf("resolveVisionAttnGeometry: %v", err)
	}
	if !fused || headDim != 4 || numHeads != 2 || numKVHeads != 2 {
		t.Fatalf("(fused,headDim,numHeads,numKVHeads) = (%v,%d,%d,%d), want (true,4,2,2)", fused, headDim, numHeads, numKVHeads)
	}
}

func TestResolveVisionAttnGeometry_GoodSeparate(t *testing.T) {
	ts := map[string]safetensors.Tensor{
		"vision_tower.blocks.0.attn.q_proj.weight": bf16T(syn(8*8, 1), 8, 8),
		"vision_tower.blocks.0.attn.k_proj.weight": bf16T(syn(4*8, 2), 4, 8),
	}
	fused, headDim, numHeads, numKVHeads, err := resolveVisionAttnGeometry(ts, "vision_tower.blocks.0.", &visionConfig{NumHeads: 2}, nil)
	if err != nil {
		t.Fatalf("resolveVisionAttnGeometry: %v", err)
	}
	if fused {
		t.Fatal("fused = true, want false (separate q_proj/k_proj)")
	}
	if headDim != 4 || numHeads != 2 || numKVHeads != 1 {
		t.Fatalf("headDim/numHeads/numKVHeads = %d/%d/%d, want 4/2/1 (GQA)", headDim, numHeads, numKVHeads)
	}
}

func TestResolveVisionAttnGeometry_Bad(t *testing.T) {
	if _, _, _, _, err := resolveVisionAttnGeometry(map[string]safetensors.Tensor{}, "vision_tower.blocks.0.", nil, nil); err == nil {
		t.Fatal("resolveVisionAttnGeometry: want an error when neither attn.qkv nor attn.q_proj is present, got nil")
	}
}

func TestResolveVisionAttnGeometry_BadFusedNotDivisibleBy3(t *testing.T) {
	ts := map[string]safetensors.Tensor{"vision_tower.blocks.0.attn.qkv.weight": bf16T(syn(10*8, 1), 10, 8)}
	if _, _, _, _, err := resolveVisionAttnGeometry(ts, "vision_tower.blocks.0.", &visionConfig{NumHeads: 2}, nil); err == nil {
		t.Fatal("resolveVisionAttnGeometry: want an error when fused qkv width is not divisible by 3, got nil")
	}
}

func TestLoadBlockQKV_Fused(t *testing.T) {
	ts := map[string]safetensors.Tensor{
		"vision_tower.blocks.0.attn.qkv.weight": bf16T(syn(24*8, 1), 24, 8),
		"vision_tower.blocks.0.attn.qkv.bias":   bf16T(syn(24, 2), 24),
	}
	q, k, v, alias, err := loadBlockQKV(ts, "vision_tower.blocks.0.", 8, true, nil, false)
	if err != nil {
		t.Fatalf("loadBlockQKV: %v", err)
	}
	if alias {
		t.Fatal("alias = true, want false")
	}
	if q.Out != 8 || k.Out != 8 || v.Out != 8 || q.In != 8 || k.In != 8 || v.In != 8 {
		t.Fatalf("q/k/v Out/In = %d/%d %d/%d %d/%d, want 8/8 8/8 8/8", q.Out, q.In, k.Out, k.In, v.Out, v.In)
	}
	if len(q.B) != 8 || len(k.B) != 8 || len(v.B) != 8 {
		t.Fatalf("len(q.B)/len(k.B)/len(v.B) = %d/%d/%d, want 8/8/8", len(q.B), len(k.B), len(v.B))
	}
}

func TestLoadBlockQKV_Separate(t *testing.T) {
	ts := map[string]safetensors.Tensor{
		"vision_tower.blocks.0.attn.q_proj.weight": bf16T(syn(8*8, 1), 8, 8),
		"vision_tower.blocks.0.attn.k_proj.weight": bf16T(syn(4*8, 2), 4, 8),
		"vision_tower.blocks.0.attn.v_proj.weight": bf16T(syn(4*8, 3), 4, 8),
	}
	q, k, v, alias, err := loadBlockQKV(ts, "vision_tower.blocks.0.", 8, false, nil, false)
	if err != nil {
		t.Fatalf("loadBlockQKV: %v", err)
	}
	if alias {
		t.Fatal("alias = true, want false")
	}
	if q.Out != 8 || k.Out != 4 || v.Out != 4 {
		t.Fatalf("q/k/v.Out = %d/%d/%d, want 8/4/4", q.Out, k.Out, v.Out)
	}
}

func TestLoadBlockQKV_Bad(t *testing.T) {
	if _, _, _, _, err := loadBlockQKV(map[string]safetensors.Tensor{}, "vision_tower.blocks.0.", 8, true, nil, false); err == nil {
		t.Fatal("loadBlockQKV: want an error when attn.qkv.weight is missing (fused=true), got nil")
	}
	if _, _, _, _, err := loadBlockQKV(map[string]safetensors.Tensor{}, "vision_tower.blocks.0.", 8, false, nil, false); err == nil {
		t.Fatal("loadBlockQKV: want an error when attn.q_proj.weight is missing (fused=false), got nil")
	}
}

func TestLoadBlockMLP_SwiGLU(t *testing.T) {
	ts := map[string]safetensors.Tensor{
		"vision_tower.blocks.0.mlp.gate_proj.weight": bf16T(syn(16*8, 1), 16, 8),
		"vision_tower.blocks.0.mlp.up_proj.weight":   bf16T(syn(16*8, 2), 16, 8),
		"vision_tower.blocks.0.mlp.down_proj.weight": bf16T(syn(8*16, 3), 8, 16),
	}
	w, ff, gelu, alias, err := loadBlockMLP(ts, "vision_tower.blocks.0.", 8, nil, false)
	if err != nil {
		t.Fatalf("loadBlockMLP: %v", err)
	}
	if alias {
		t.Fatal("alias = true, want false")
	}
	if gelu {
		t.Fatal("gelu = true, want false (gate_proj present ⇒ SwiGLU)")
	}
	if ff != 16 {
		t.Fatalf("ff = %d, want 16", ff)
	}
	if w.Gate.W == nil || w.FC1.W != nil {
		t.Fatalf("w = %+v, want Gate set and FC1 unset", w)
	}
}

func TestLoadBlockMLP_GELU(t *testing.T) {
	ts := map[string]safetensors.Tensor{
		"vision_tower.blocks.0.mlp.linear_fc1.weight": bf16T(syn(16*8, 1), 16, 8),
		"vision_tower.blocks.0.mlp.linear_fc2.weight": bf16T(syn(8*16, 2), 8, 16),
	}
	w, ff, gelu, alias, err := loadBlockMLP(ts, "vision_tower.blocks.0.", 8, nil, false)
	if err != nil {
		t.Fatalf("loadBlockMLP: %v", err)
	}
	if alias {
		t.Fatal("alias = true, want false")
	}
	if !gelu {
		t.Fatal("gelu = false, want true (no gate_proj, linear_fc1 present)")
	}
	if ff != 16 {
		t.Fatalf("ff = %d, want 16", ff)
	}
	if w.FC1.W == nil || w.Gate.W != nil {
		t.Fatalf("w = %+v, want FC1 set and Gate unset", w)
	}
}

func TestLoadBlockMLP_Bad(t *testing.T) {
	if _, _, _, _, err := loadBlockMLP(map[string]safetensors.Tensor{}, "vision_tower.blocks.0.", 8, nil, false); err == nil {
		t.Fatal("loadBlockMLP: want an error when neither mlp.gate_proj nor mlp.linear_fc1 is present, got nil")
	}
}

// TestBuildVisionTowerQuant_RealLayoutGood is the REAL layout's flagship integration test — the twin of
// TestBuildVisionTower_Good for the fused-qkv/GELU-MLP/learned-position/vision_tower.merger.* convention:
// every divergence from the guessed layout in one fixture (task divergences 1-5; divergence 6, quant, gets
// its own test below).
func TestBuildVisionTowerQuant_RealLayoutGood(t *testing.T) {
	ts := map[string]safetensors.Tensor{}
	addRealLayoutVisionTensors(ts, 2, 8)
	vc := &visionConfig{PatchSize: 2, NumHeads: 2}
	tower, alias, err := buildVisionTowerQuant(ts, vc, 8, nil, false)
	if err != nil {
		t.Fatalf("buildVisionTowerQuant: %v", err)
	}
	if tower == nil {
		t.Fatal("buildVisionTowerQuant: want a tower, got nil")
	}
	if alias {
		t.Fatal("alias = true, want false (dense, non-zero-copy)")
	}
	if tower.Cfg.Hidden != 8 || tower.Cfg.PatchDim != 12 {
		t.Fatalf("Cfg.Hidden/PatchDim = %d/%d, want 8/12", tower.Cfg.Hidden, tower.Cfg.PatchDim)
	}
	if tower.Cfg.NumHeads != 2 || tower.Cfg.HeadDim != 4 || tower.Cfg.NumKVHeads != 2 {
		t.Fatalf("Cfg.NumHeads/HeadDim/NumKVHeads = %d/%d/%d, want 2/4/2 (fused-qkv derivation, plain MHA)",
			tower.Cfg.NumHeads, tower.Cfg.HeadDim, tower.Cfg.NumKVHeads)
	}
	if !tower.Cfg.LearnedPositions {
		t.Fatal("Cfg.LearnedPositions = false, want true (vision_tower.pos_embed.weight present)")
	}
	if len(tower.PosEmbed) != 16*8 {
		t.Fatalf("len(tower.PosEmbed) = %d, want %d (16 positions x Hidden 8)", len(tower.PosEmbed), 16*8)
	}
	if tower.Cfg.MergeSize != 2 {
		t.Fatalf("Cfg.MergeSize = %d, want 2 (derived from vision_tower.merger.linear_fc1 width)", tower.Cfg.MergeSize)
	}
	if len(tower.Blocks) != 2 {
		t.Fatalf("len(Blocks) = %d, want 2", len(tower.Blocks))
	}
	for i, b := range tower.Blocks {
		if len(b.Attn.QNorm) != 0 || len(b.Attn.KNorm) != 0 {
			t.Fatalf("block %d QNorm/KNorm len = %d/%d, want 0/0 (the real layout ships neither — divergence 5)", i, len(b.Attn.QNorm), len(b.Attn.KNorm))
		}
		if !b.MLP.GELU {
			t.Fatalf("block %d MLP.GELU = false, want true", i)
		}
		if b.Attn.Q.Out != 8 || b.Attn.K.Out != 8 || b.Attn.V.Out != 8 || b.Attn.Q.In != 8 {
			t.Fatalf("block %d Q/K/V Out/In = %d/%d %d/%d %d/%d, want 8/8 8/8 8/8 (split from the fused attn.qkv)",
				i, b.Attn.Q.Out, b.Attn.Q.In, b.Attn.K.Out, b.Attn.K.In, b.Attn.V.Out, b.Attn.V.In)
		}
		if b.Attn.O.Out != 8 || b.Attn.O.In != 8 {
			t.Fatalf("block %d O.Out/In = %d/%d, want 8/8 (attn.proj alias)", i, b.Attn.O.Out, b.Attn.O.In)
		}
		if b.Norm1B == nil || b.Norm2B == nil {
			t.Fatalf("block %d Norm1B/Norm2B is nil, want the LayerNorm bias populated", i)
		}
	}

	// The tower must actually FORWARD end-to-end: patchify a synthetic 8x8 image (PatchSize=2 ⇒ a 4x4
	// grid, matching PosEmbed's 16-position table) and run it through the whole stack.
	patches := syn(4*4*tower.Cfg.PatchDim, 4321)
	features, softTokens, ferr := visionTowerForward(patches, 4, 4, tower)
	if ferr != nil {
		t.Fatalf("visionTowerForward: %v", ferr)
	}
	if softTokens != 4 || len(features) != softTokens*8 {
		t.Fatalf("softTokens/len(features) = %d/%d, want 4/%d", softTokens, len(features), 4*8)
	}
}

// TestBuildVisionTowerQuant_RealLayoutQuantised is divergence 6's proof: every 2-D projection in the REAL
// layout (patch embed, each block's fused qkv/output proj/GELU fc1/fc2, the merger's two linears) gets
// mlx-affine-quantised, and the loader must resolve every one of them PACKED (WQ set, not widened) — then
// the forward output must be bit-identical to the SAME tower with every packed weight dequantised back to
// dense (matNTQuantHost's own contract; composed_quant_test.go proves the identical property for the text
// stack).
func TestBuildVisionTowerQuant_RealLayoutQuantised(t *testing.T) {
	ts := map[string]safetensors.Tensor{}
	addRealLayoutVisionTensors(ts, 2, 8)
	// bits=8 (not 4): 4-bit packing needs groupSize a multiple of 8 codes-per-word, which patchDim=12
	// cannot satisfy (12 is not a multiple of 8); 8-bit needs only a multiple of 4, and gs=4 divides every
	// quantised inDim here (patchDim 12, hidden 8, ff 16, mergedIn 32).
	const bits, gs = 8, 4
	names := []string{
		"vision_tower.patch_embed.proj.weight",
		"vision_tower.merger.linear_fc1.weight", "vision_tower.merger.linear_fc2.weight",
	}
	for i := range 2 {
		bp := "vision_tower.blocks." + itoa(i) + "."
		names = append(names, bp+"attn.qkv.weight", bp+"attn.proj.weight", bp+"mlp.linear_fc1.weight", bp+"mlp.linear_fc2.weight")
	}
	for _, name := range names {
		quantiseInPlace(t, ts, name, bits, gs)
	}

	vc := &visionConfig{PatchSize: 2, NumHeads: 2}
	quant := &model.QuantConfig{GroupSize: gs, Bits: bits}
	tower, alias, err := buildVisionTowerQuant(ts, vc, 8, quant, false)
	if err != nil {
		t.Fatalf("buildVisionTowerQuant: %v", err)
	}
	if alias {
		t.Fatal("alias = true, want false (owned-copy path — zeroCopy=false)")
	}
	if tower.Patch.WQ == nil {
		t.Fatal("tower.Patch.WQ is nil, want packed")
	}
	for i, b := range tower.Blocks {
		if b.Attn.Q.WQ == nil || b.Attn.K.WQ == nil || b.Attn.V.WQ == nil || b.Attn.O.WQ == nil {
			t.Fatalf("block %d attn WQ not all set: Q=%v K=%v V=%v O=%v", i, b.Attn.Q.WQ != nil, b.Attn.K.WQ != nil, b.Attn.V.WQ != nil, b.Attn.O.WQ != nil)
		}
		if b.MLP.FC1.WQ == nil || b.MLP.FC2.WQ == nil {
			t.Fatalf("block %d mlp WQ not all set: FC1=%v FC2=%v", i, b.MLP.FC1.WQ != nil, b.MLP.FC2.WQ != nil)
		}
	}
	if tower.Merger.L1.WQ == nil || tower.Merger.L2.WQ == nil {
		t.Fatal("merger WQ not all set")
	}

	patches := syn(4*4*tower.Cfg.PatchDim, 7654)
	featQ, softQ, err := visionTowerForward(append([]float32(nil), patches...), 4, 4, tower)
	if err != nil {
		t.Fatalf("visionTowerForward (packed): %v", err)
	}
	dequantiseVisionTowerInPlace(t, tower)
	featD, softD, err := visionTowerForward(append([]float32(nil), patches...), 4, 4, tower)
	if err != nil {
		t.Fatalf("visionTowerForward (dequantised): %v", err)
	}
	if softQ != softD {
		t.Fatalf("softTokens packed=%d dequantised=%d, want equal", softQ, softD)
	}
	if len(featQ) != len(featD) {
		t.Fatalf("len(features) packed=%d dequantised=%d, want equal", len(featQ), len(featD))
	}
	for i := range featD {
		if featQ[i] != featD[i] {
			t.Fatalf("features[%d] packed=%v != dequantised=%v", i, featQ[i], featD[i])
		}
	}
}

// TestBuildVisionTowerQuant_RealLayoutZeroCopyAlias proves the zeroCopy=true path aliases the input
// tensors' packed bytes (alias=true) — the RSS-saving contract model.LoadComposedDir depends on — while the
// SAME fixture with zeroCopy=false copies to owned buffers (alias=false), mirroring loader.go's own
// tensorAsQuant zeroCopy contract at the vision seam.
func TestBuildVisionTowerQuant_RealLayoutZeroCopyAlias(t *testing.T) {
	ts := map[string]safetensors.Tensor{}
	addRealLayoutVisionTensors(ts, 1, 8)
	// bits=8 (not 4): 4-bit packing needs groupSize a multiple of 8 codes-per-word, which patchDim=12
	// cannot satisfy (12 is not a multiple of 8); 8-bit needs only a multiple of 4, and gs=4 divides every
	// quantised inDim here (patchDim 12, hidden 8, ff 16, mergedIn 32).
	const bits, gs = 8, 4
	quantiseInPlace(t, ts, "vision_tower.patch_embed.proj.weight", bits, gs)
	vc := &visionConfig{PatchSize: 2, NumHeads: 2}
	quant := &model.QuantConfig{GroupSize: gs, Bits: bits}

	_, aliasCopied, err := buildVisionTowerQuant(ts, vc, 8, quant, false)
	if err != nil {
		t.Fatalf("buildVisionTowerQuant (owned-copy): %v", err)
	}
	if aliasCopied {
		t.Fatal("alias = true with zeroCopy=false, want false")
	}

	_, aliasZeroCopy, err := buildVisionTowerQuant(ts, vc, 8, quant, true)
	if err != nil {
		t.Fatalf("buildVisionTowerQuant (zero-copy): %v", err)
	}
	if !aliasZeroCopy {
		t.Fatal("alias = false with zeroCopy=true and a quantised patch_embed, want true")
	}
}

// TestBuildVisionBlocksQuant_MixedLayoutBad pins that a checkpoint mixing the two conventions ACROSS
// blocks (block 0 fused, block 1 separate) fails loudly rather than silently mis-loading block 1 under
// block 0's decided convention.
func TestBuildVisionBlocksQuant_MixedLayoutBad(t *testing.T) {
	ts := map[string]safetensors.Tensor{}
	addRealLayoutVisionTensors(ts, 1, 8) // block 0: fused qkv, GELU mlp
	ts["vision_tower.blocks.1.attn.q_proj.weight"] = bf16T(syn(8*8, 101), 8, 8)
	ts["vision_tower.blocks.1.attn.k_proj.weight"] = bf16T(syn(8*8, 102), 8, 8)
	ts["vision_tower.blocks.1.attn.v_proj.weight"] = bf16T(syn(8*8, 103), 8, 8)
	ts["vision_tower.blocks.1.attn.o_proj.weight"] = bf16T(syn(8*8, 104), 8, 8)
	ts["vision_tower.blocks.1.norm1.weight"] = bf16T(syn(8, 105), 8)
	ts["vision_tower.blocks.1.norm2.weight"] = bf16T(syn(8, 106), 8)
	ts["vision_tower.blocks.1.mlp.gate_proj.weight"] = bf16T(syn(16*8, 107), 16, 8)
	ts["vision_tower.blocks.1.mlp.up_proj.weight"] = bf16T(syn(16*8, 108), 16, 8)
	ts["vision_tower.blocks.1.mlp.down_proj.weight"] = bf16T(syn(8*16, 109), 8, 16)
	vc := &visionConfig{NumHeads: 2}
	if _, _, _, _, _, err := buildVisionBlocksQuant(ts, 8, vc, nil, false); err == nil {
		t.Fatal("buildVisionBlocksQuant: want an error when block 1's attention convention differs from block 0's, got nil")
	}
}

// TestLoadComposedVisionTower_RealLayoutGood is the full LoadComposed integration for the REAL layout,
// mirroring TestLoadComposedVisionTower_Good: a checkpoint carrying BOTH the Qwen 3.6 hybrid text stack AND
// a REAL-layout vision tower loads both, and the text stack still forwards correctly.
func TestLoadComposedVisionTower_RealLayoutGood(t *testing.T) {
	ts, _ := mkHybridCheckpoint() // D=8, 4 layers (gated-delta + full attention), vocab 32
	addRealLayoutVisionTensors(ts, 2, 8)
	cfg := []byte(`{"hidden_size":8,"num_hidden_layers":4,"intermediate_size":16,"num_attention_heads":4,
		"num_key_value_heads":2,"head_dim":8,"vocab_size":32,"rms_norm_eps":1e-5,"rope_theta":1000000,
		"partial_rotary_factor":0.5,"full_attention_interval":2,"image_token_id":1234,
		"vision_config":{"patch_size":2,"num_heads":2}}`)

	m, err := LoadComposed(ts, cfg)
	if err != nil {
		t.Fatalf("LoadComposed: %v", err)
	}
	if len(m.Layers) != 4 {
		t.Fatalf("len(m.Layers) = %d, want 4 (vision loading must not disturb the text stack)", len(m.Layers))
	}
	if m.Vision == nil {
		t.Fatal("m.Vision is nil, want a loaded REAL-layout tower")
	}
	if len(m.Vision.Blocks) != 2 {
		t.Fatalf("len(m.Vision.Blocks) = %d, want 2", len(m.Vision.Blocks))
	}
	if !m.Vision.Cfg.LearnedPositions {
		t.Fatal("m.Vision.Cfg.LearnedPositions = false, want true")
	}
	if m.ImageTokenID != 1234 {
		t.Fatalf("m.ImageTokenID = %d, want 1234", m.ImageTokenID)
	}
	if _, err := NewSession(m).Forward([]int32{1, 5, 9}); err != nil {
		t.Fatalf("text forward on a REAL-layout vision-carrying model: %v", err)
	}
}
