// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"context"
	"image"
	"image/color"
	"image/png"
	"iter"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/internal/enginegate"
	"dappco.re/go/inference/model/arch/Qwen/qwen35"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

// qwenVisionChatter is the Chat/Err slice of the served engine.TextModel these receipts drive.
type qwenVisionChatter interface {
	Chat(context.Context, []inference.Message, ...inference.GenerateOption) iter.Seq[inference.Token]
	Err() core.Result
}

// qwen_vision_model_test.go proves the serve glue around the qwen tower: the load-seam attach
// (loadQwenVisionTower over a mapped checkpoint dir — with-tower, text-only, non-qwen, malformed),
// the NativeTokenModel dispatch (AcceptsImageInput / placeholder id + block / ProjectImage), the
// REFUSAL contract on a real text-only qwen snapshot, and the live end-to-end image turn on the real
// vision-towered mlx-community/Qwen3.6-27B-4bit — both real-checkpoint receipts behind the
// local-HF-cache skip (enginegate.HFModelPath), so a checkout without the weights stays green.

// qwenVisionTestPNG encodes a w×h PNG with a bright square on a dark field — decodable content for
// the projection receipts.
func qwenVisionTestPNG(t testing.TB, w, h int) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			c := color.RGBA{R: 16, G: 16, B: 24, A: 255}
			if x >= w/4 && x < 3*w/4 && y >= h/4 && y < 3*h/4 {
				c = color.RGBA{R: 230, G: 220, B: 40, A: 255}
			}
			img.Set(x, y, c)
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("encode png: %v", err)
	}
	return buf.Bytes()
}

// qwenVisionDirConfig is the tiny multimodal-wrapper config the seam fixtures load under — the same
// geometry as the qwen35 package's loader tests (hidden 8, patch 2, merge 2, text hidden 16).
const qwenVisionDirConfig = `{"model_type":"qwen3_5","image_token_id":777,` +
	`"text_config":{"model_type":"qwen3_5_text","hidden_size":16,"num_hidden_layers":1,"num_attention_heads":2,"vocab_size":32},` +
	`"vision_config":{"patch_size":2,"in_channels":3,"num_heads":2,"spatial_merge_size":2,"rms_norm_eps":1e-6}}`

// writeQwenVisionDir materialises a checkpoint dir carrying config.json + the given tensors — enough
// for the load-seam probe (loadQwenVisionTower reads dm.Tensors + config.json only).
func writeQwenVisionDir(t *testing.T, config string, tensors map[string]safetensors.Tensor) string {
	t.Helper()
	dir := t.TempDir()
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), config); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	blob, err := safetensors.Encode(tensors)
	if err != nil {
		t.Fatalf("encode weights: %v", err)
	}
	if err := coreio.Local.Write(core.PathJoin(dir, "model.safetensors"), string(blob)); err != nil {
		t.Fatalf("write model.safetensors: %v", err)
	}
	return dir
}

func qwenVisionF32Tensor(shape []int, seed uint32) safetensors.Tensor {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return safetensors.Tensor{Dtype: "F32", Shape: shape, Data: safetensors.EncodeFloat32(qtSeq(n, seed))}
}

// qwenVisionDirTensors is the REAL-layout tiny tower as checkpoint tensors (1 block; geometry
// mirrors the qwen35 loader tests) plus a token stub — the with-tower seam fixture.
func qwenVisionDirTensors() map[string]safetensors.Tensor {
	const hidden, ff, mergeIn, text = 8, 12, 32, 16
	t := map[string]safetensors.Tensor{
		"model.embed_tokens.weight":             qwenVisionF32Tensor([]int{32, text}, 90),
		"vision_tower.patch_embed.proj.weight":  qwenVisionF32Tensor([]int{hidden, 2, 2, 2, 3}, 1),
		"vision_tower.patch_embed.proj.bias":    qwenVisionF32Tensor([]int{hidden}, 2),
		"vision_tower.pos_embed.weight":         qwenVisionF32Tensor([]int{16, hidden}, 3),
		"vision_tower.merger.norm.weight":       qwenVisionF32Tensor([]int{hidden}, 4),
		"vision_tower.merger.norm.bias":         qwenVisionF32Tensor([]int{hidden}, 5),
		"vision_tower.merger.linear_fc1.weight": qwenVisionF32Tensor([]int{mergeIn, mergeIn}, 6),
		"vision_tower.merger.linear_fc1.bias":   qwenVisionF32Tensor([]int{mergeIn}, 7),
		"vision_tower.merger.linear_fc2.weight": qwenVisionF32Tensor([]int{text, mergeIn}, 8),
		"vision_tower.merger.linear_fc2.bias":   qwenVisionF32Tensor([]int{text}, 9),
	}
	bp := "vision_tower.blocks.0."
	t[bp+"norm1.weight"] = qwenVisionF32Tensor([]int{hidden}, 11)
	t[bp+"norm1.bias"] = qwenVisionF32Tensor([]int{hidden}, 12)
	t[bp+"norm2.weight"] = qwenVisionF32Tensor([]int{hidden}, 13)
	t[bp+"norm2.bias"] = qwenVisionF32Tensor([]int{hidden}, 14)
	t[bp+"attn.qkv.weight"] = qwenVisionF32Tensor([]int{3 * hidden, hidden}, 15)
	t[bp+"attn.qkv.bias"] = qwenVisionF32Tensor([]int{3 * hidden}, 16)
	t[bp+"attn.proj.weight"] = qwenVisionF32Tensor([]int{hidden, hidden}, 17)
	t[bp+"attn.proj.bias"] = qwenVisionF32Tensor([]int{hidden}, 18)
	t[bp+"mlp.linear_fc1.weight"] = qwenVisionF32Tensor([]int{ff, hidden}, 19)
	t[bp+"mlp.linear_fc1.bias"] = qwenVisionF32Tensor([]int{ff}, 20)
	t[bp+"mlp.linear_fc2.weight"] = qwenVisionF32Tensor([]int{hidden, ff}, 21)
	t[bp+"mlp.linear_fc2.bias"] = qwenVisionF32Tensor([]int{hidden}, 22)
	return t
}

func TestLoadQwenVisionTower_Good(t *testing.T) {
	dir := writeQwenVisionDir(t, qwenVisionDirConfig, qwenVisionDirTensors())
	dm, err := safetensors.LoadDirMmap(dir)
	if err != nil {
		t.Fatalf("LoadDirMmap: %v", err)
	}
	defer func() { _ = dm.Close() }()
	tower, err := loadQwenVisionTower(dir, dm)
	if err != nil {
		t.Fatalf("loadQwenVisionTower: %v", err)
	}
	if tower == nil {
		t.Fatal("vision-towered qwen checkpoint attached no tower")
	}
	if tower.Cfg.ImageTokenID != 777 || tower.Cfg.TextHidden != 16 || tower.Cfg.MergeSize != 2 {
		t.Fatalf("tower cfg = id %d text %d merge %d, want 777/16/2", tower.Cfg.ImageTokenID, tower.Cfg.TextHidden, tower.Cfg.MergeSize)
	}
}

func TestLoadQwenVisionTower_TextOnly_Good(t *testing.T) {
	dir := writeQwenVisionDir(t, qwenVisionDirConfig, map[string]safetensors.Tensor{
		"model.embed_tokens.weight": qwenVisionF32Tensor([]int{32, 16}, 90),
	})
	dm, err := safetensors.LoadDirMmap(dir)
	if err != nil {
		t.Fatalf("LoadDirMmap: %v", err)
	}
	defer func() { _ = dm.Close() }()
	tower, err := loadQwenVisionTower(dir, dm)
	if err != nil {
		t.Fatalf("loadQwenVisionTower on a text-only checkpoint: %v", err)
	}
	if tower != nil {
		t.Fatal("text-only qwen checkpoint grew a vision tower")
	}
}

func TestLoadQwenVisionTower_NonQwen_Good(t *testing.T) {
	// A non-qwen model_type never probes — even with vision_tower.* names present.
	cfg := `{"model_type":"llama","hidden_size":16,"num_hidden_layers":1,"num_attention_heads":2,"vocab_size":32}`
	dir := writeQwenVisionDir(t, cfg, qwenVisionDirTensors())
	dm, err := safetensors.LoadDirMmap(dir)
	if err != nil {
		t.Fatalf("LoadDirMmap: %v", err)
	}
	defer func() { _ = dm.Close() }()
	tower, err := loadQwenVisionTower(dir, dm)
	if err != nil || tower != nil {
		t.Fatalf("non-qwen arch must attach nothing, quietly; got tower=%v err=%v", tower != nil, err)
	}
}

func TestLoadQwenVisionTower_Malformed_Bad(t *testing.T) {
	tensors := qwenVisionDirTensors()
	delete(tensors, "vision_tower.merger.linear_fc1.weight") // tower present, merger broken
	delete(tensors, "vision_tower.merger.linear_fc1.bias")
	dir := writeQwenVisionDir(t, qwenVisionDirConfig, tensors)
	dm, err := safetensors.LoadDirMmap(dir)
	if err != nil {
		t.Fatalf("LoadDirMmap: %v", err)
	}
	defer func() { _ = dm.Close() }()
	if _, err := loadQwenVisionTower(dir, dm); err == nil {
		t.Fatal("a PRESENT but malformed vision tower must fail the load loudly, not silently serve text-only")
	}
}

// TestQwenVisionTokenModelDispatch_Good proves the NativeTokenModel vision entry points route to the
// qwen tower: the live AcceptsImageInput probe, the config-derived placeholder id, the family
// placeholder block spellings, and a full ProjectImage (patchify → tower → bf16 rows).
func TestQwenVisionTokenModelDispatch_Good(t *testing.T) {
	tower := qtRealTower(1)
	m := &NativeTokenModel{qwenVision: tower}
	if !m.AcceptsImageInput() {
		t.Fatal("AcceptsImageInput must be true with a loaded qwen tower")
	}
	if got := m.ImagePlaceholderTokenID(); got != 777 {
		t.Fatalf("ImagePlaceholderTokenID = %d, want 777", got)
	}
	block := m.ImagePlaceholderBlock(2)
	want := qwen35.VisionBeginToken + qwen35.VisionPadToken + qwen35.VisionPadToken + qwen35.VisionEndToken
	if block != want {
		t.Fatalf("ImagePlaceholderBlock = %q, want %q", block, want)
	}
	// A 4×4 PNG at patch 1 (qtRealTower's PatchSize) crops to grid 4×4 → merge 2 → 4 soft tokens.
	features, softTokens, err := m.ProjectImage(qwenVisionTestPNG(t, 4, 4))
	if err != nil {
		t.Fatalf("ProjectImage: %v", err)
	}
	if softTokens != 4 {
		t.Fatalf("softTokens = %d, want 4 ((4/2)·(4/2))", softTokens)
	}
	if len(features) != softTokens*tower.Cfg.TextHidden*2 {
		t.Fatalf("feature bytes = %d, want %d (softTokens·TextHidden·bf16)", len(features), softTokens*tower.Cfg.TextHidden*2)
	}
}

// TestQwenVisionRefusal_RealTextOnly is the refusal contract on a REAL text-only qwen snapshot: the
// load succeeds, AcceptsImages answers false, and an image turn gets the engine's clean refusal —
// never a crash, never a fabricated answer. Skips when the snapshot is not in the local HF cache.
func TestQwenVisionRefusal_RealTextOnly(t *testing.T) {
	if testing.Short() {
		t.Skip("real-checkpoint load in -short")
	}
	// The mlx-community OptiQ pack — the engine's own proven 0.8B serve target; the official
	// Qwen/Qwen3.5-0.8B snapshot in this cache ships a nonstandard shard name the directory loader
	// does not resolve (a pre-existing condition, not this lane's).
	snap := enginegate.HFModelPath(t, "mlx-community/Qwen3.5-0.8B-OptiQ-4bit")
	res := metalBackend{}.LoadModel(snap, inference.WithContextLen(2048))
	if !res.OK {
		t.Fatalf("LoadModel(%s): %v", snap, res.Err())
	}
	tm := res.Value.(inference.TextModel)
	defer func() { _ = tm.Close() }()
	vm, ok := tm.(inference.VisionModel)
	if !ok {
		t.Fatal("engine.TextModel must satisfy the inference.VisionModel probe surface")
	}
	if vm.AcceptsImages() {
		t.Fatal("a text-only qwen checkpoint must not accept images")
	}
	chatter, ok := tm.(qwenVisionChatter)
	if !ok {
		t.Fatal("loaded model does not expose Chat/Err")
	}
	seq := chatter.Chat(context.Background(), []inference.Message{{
		Role: "user", Content: "What is in this image?", Images: [][]byte{qwenVisionTestPNG(t, 64, 64)},
	}}, inference.WithMaxTokens(8))
	tokens := 0
	seq(func(inference.Token) bool { tokens++; return true })
	if tokens != 0 {
		t.Fatalf("image turn on a text-only model yielded %d token(s) — it must refuse, not answer", tokens)
	}
	err := chatter.Err()
	if err.OK {
		t.Fatal("image turn on a text-only model must surface an error")
	}
	if !core.Contains(core.Sprintf("%v", err.Err()), "does not accept image input") {
		t.Fatalf("refusal error = %v, want the engine's clean not-a-vision-model refusal", err.Err())
	}
}

// TestQwenVisionImageTurn_RealCheckpoint is the LIVE end-to-end receipt on the real vision-towered
// snapshot: load mlx-community/Qwen3.6-27B-4bit through the registered backend, confirm the tower
// attached (AcceptsImages true — the exact turn that used to hit the not-a-vision-model refusal, #59
// item 1), and serve one image chat turn through the whole chain (project → placeholder block →
// ChatML render → placeholder-run verification → feature splice → PrefillTokenEmbeddings → decode).
// Skips when the snapshot is not in the local HF cache. This is a HEAVY test (a ~15 GB checkpoint +
// the tower widened to f32).
func TestQwenVisionImageTurn_RealCheckpoint(t *testing.T) {
	if testing.Short() {
		t.Skip("real-checkpoint load in -short")
	}
	snap := enginegate.HFModelPath(t, "mlx-community/Qwen3.6-27B-4bit")
	res := metalBackend{}.LoadModel(snap, inference.WithContextLen(4096))
	if !res.OK {
		t.Fatalf("LoadModel(%s): %v", snap, res.Err())
	}
	tm := res.Value.(inference.TextModel)
	defer func() { _ = tm.Close() }()
	vm, ok := tm.(inference.VisionModel)
	if !ok || !vm.AcceptsImages() {
		t.Fatal("the vision-towered 27B must accept images through the factory route (the #59 gap this lane closes)")
	}
	chatter, ok := tm.(qwenVisionChatter)
	if !ok {
		t.Fatal("loaded model does not expose Chat/Err")
	}
	// A 64×64 image → grid 4×4 → 4 soft tokens: the tower runs the full real geometry (27 blocks,
	// hidden 1152) on a tiny patch count, keeping the receipt minutes-cheap.
	seq := chatter.Chat(context.Background(), []inference.Message{{
		Role: "user", Content: "Reply with one short sentence: what shape is on the image?",
		Images: [][]byte{qwenVisionTestPNG(t, 64, 64)},
	}}, inference.WithMaxTokens(24), inference.WithTemperature(0))
	var reply core.Builder
	tokens := 0
	seq(func(tok inference.Token) bool {
		reply.WriteString(tok.Text)
		tokens++
		return true
	})
	if err := chatter.Err(); !err.OK {
		t.Fatalf("image turn failed: %v", err.Err())
	}
	if tokens == 0 {
		t.Fatal("image turn yielded no tokens")
	}
	t.Logf("live image-turn receipt: %d token(s): %q", tokens, reply.String())
}
