// SPDX-Licence-Identifier: EUPL-1.2

package glmocr

import (
	"os"
	"testing"
)

// live_test.go is the #37 slice's live gate against the REAL, locally-cached zai-org/GLM-OCR
// checkpoint — RUNTIME-gated, reading the checkpoint dir from GLM_OCR_DIR (skips cleanly when
// unset) and defaulting to the standard Hugging Face cache location, mirroring whisper's
// resolveWhisperTinyDir (arch/openai/whisper/live_test.go): env override, else auto-resolve the
// hash-named snapshot subdirectory. Every other _test.go in this package is hermetic (committed
// small/synthetic fixtures + testdata/block_goldens.json's toy-config activations captured from
// the real transformers modules); this file is the ONE that proves the real 1.3B-parameter
// checkpoint — real tensor names, real tokenizer, real trained weights — resolves and performs
// OCR correctly on testdata/fixture.png, against testdata/e2e_golden.json (captured from
// transformers.GlmOcrForConditionalGeneration.generate, do_sample=false, float32, CPU — matching
// this package's own host-f32 precision and generation_config.json's own greedy default).
func resolveGlmOcrDir(t *testing.T) string {
	t.Helper()
	if dir := os.Getenv("GLM_OCR_DIR"); dir != "" {
		return dir
	}
	home := os.Getenv("HOME")
	base := home + "/.cache/huggingface/hub/models--zai-org--GLM-OCR/snapshots"
	entries, err := os.ReadDir(base)
	if err != nil {
		t.Skipf("zai-org/GLM-OCR snapshot dir not found (%v) — set GLM_OCR_DIR to override, or download it: "+
			"python -c \"from huggingface_hub import snapshot_download; print(snapshot_download('zai-org/GLM-OCR'))\"", err)
	}
	for _, e := range entries {
		if e.IsDir() {
			return base + "/" + e.Name()
		}
	}
	t.Skip("zai-org/GLM-OCR snapshots dir has no snapshot subdirectory")
	return ""
}

// TestLive_RealCheckpoint_Load_Good proves Load resolves the REAL checkpoint's actual tensor
// names/shapes — the "never guessed" requirement — not just the hand-built fixtures every other
// test in this package uses. Geometry pinned to the numbers this whole lane was built and
// verified against.
func TestLive_RealCheckpoint_Load_Good(t *testing.T) {
	dir := resolveGlmOcrDir(t)
	m, err := Load(dir)
	if err != nil {
		t.Fatalf("Load(%s): %v", dir, err)
	}
	tc, vc := m.Config.TextConfig, m.Config.VisionConfig
	if tc.HiddenSize != 1536 || tc.NumHiddenLayers != 16 || tc.VocabSize != 59392 || tc.HeadDim != 128 {
		t.Fatalf("real GLM-OCR text geometry = hidden %d, layers %d, vocab %d, head_dim %d; want 1536/16/59392/128",
			tc.HiddenSize, tc.NumHiddenLayers, tc.VocabSize, tc.HeadDim)
	}
	if vc.HiddenSize != 1024 || vc.Depth != 24 || vc.NumHeads != 16 || vc.InChannels != 3 {
		t.Fatalf("real GLM-OCR vision geometry = hidden %d, depth %d, heads %d, in_channels %d; want 1024/24/16/3",
			vc.HiddenSize, vc.Depth, vc.NumHeads, vc.InChannels)
	}
	if len(m.Weights.Text.Layers) != 16 || len(m.Weights.Vision.Blocks) != 24 {
		t.Fatalf("real GLM-OCR loaded %d text layers / %d vision blocks, want 16/24", len(m.Weights.Text.Layers), len(m.Weights.Vision.Blocks))
	}
	if len(m.Weights.Text.EmbedTokens) != 59392*1536 {
		t.Fatalf("real GLM-OCR embed_tokens has %d elements, want %d", len(m.Weights.Text.EmbedTokens), 59392*1536)
	}
	if len(m.Generation.EOSTokenIDs) != 2 {
		t.Fatalf("real GLM-OCR generation_config.json eos_token_id has %d entries, want 2", len(m.Generation.EOSTokenIDs))
	}
}

// TestLive_RealCheckpoint_BuildPrompt_Good proves this package's hardcoded chat-template shape
// (BuildPrompt's file doc comment) reproduces the REAL Glm46VProcessor.apply_chat_template
// output byte-for-byte for GLM-OCR's documented single-image OCR-task usage.
func TestLive_RealCheckpoint_BuildPrompt_Good(t *testing.T) {
	dir := resolveGlmOcrDir(t)
	m, err := Load(dir)
	if err != nil {
		t.Fatalf("Load(%s): %v", dir, err)
	}
	golden := readE2EGolden(t)
	numImageTokens := 0
	for _, tt := range golden.MMTokenTypeIDs {
		if tt == 1 {
			numImageTokens++
		}
	}
	ids, mmType, err := BuildPrompt(m.Tokenizer, m.Config, golden.PromptText, numImageTokens)
	if err != nil {
		t.Fatalf("BuildPrompt: %v", err)
	}
	if len(ids) != len(golden.InputIDs) {
		t.Fatalf("BuildPrompt produced %d tokens, golden has %d", len(ids), len(golden.InputIDs))
	}
	for i := range ids {
		if ids[i] != golden.InputIDs[i] {
			t.Fatalf("BuildPrompt ids[%d] = %d, want %d (full: got %v want %v)", i, ids[i], golden.InputIDs[i], ids, golden.InputIDs)
		}
		if mmType[i] != golden.MMTokenTypeIDs[i] {
			t.Fatalf("BuildPrompt mmType[%d] = %d, want %d", i, mmType[i], golden.MMTokenTypeIDs[i])
		}
	}
}

// TestLive_RealCheckpoint_VisionForward_Good proves the REAL vision tower (all 24 blocks, real
// trained weights) reproduces the reference's pooler_output on this package's own fixture.png.
func TestLive_RealCheckpoint_VisionForward_Good(t *testing.T) {
	dir := resolveGlmOcrDir(t)
	m, err := Load(dir)
	if err != nil {
		t.Fatalf("Load(%s): %v", dir, err)
	}
	golden := readE2EGolden(t)
	imgBytes := readTestdata(t, "fixture.png")
	patches, err := DecodeAndPatchify(imgBytes, m.ImagePreprocessor, m.Config.VisionConfig)
	if err != nil {
		t.Fatalf("DecodeAndPatchify: %v", err)
	}
	if patches.GridT != golden.ImageGridTHW[0] || patches.GridH != golden.ImageGridTHW[1] || patches.GridW != golden.ImageGridTHW[2] {
		t.Fatalf("DecodeAndPatchify grid = (%d,%d,%d), want %v", patches.GridT, patches.GridH, patches.GridW, golden.ImageGridTHW)
	}
	embeds, numMerged, err := VisionForward(patches, &m.Weights.Vision, m.Config.VisionConfig)
	if err != nil {
		t.Fatalf("VisionForward: %v", err)
	}
	if numMerged*m.Config.VisionConfig.OutHiddenSize != len(golden.VisionPoolerOutput) {
		t.Fatalf("VisionForward numMerged=%d (×%d) = %d elements, golden pooler_output has %d",
			numMerged, m.Config.VisionConfig.OutHiddenSize, numMerged*m.Config.VisionConfig.OutHiddenSize, len(golden.VisionPoolerOutput))
	}
	if d := maxAbsDiff32(t, embeds, golden.VisionPoolerOutput); d > 5e-2 {
		t.Fatalf("VisionForward vs real checkpoint's pooler_output maxAbsDiff = %v, want < 5e-2 (24-layer f32 accumulation drift band)", d)
	}
}

// TestLive_RealCheckpoint_OCR_Good is the E2E gate: this package's own deterministic
// testdata/fixture.png (real rendered text "LEM OCR"), the REAL 1.3B-parameter checkpoint,
// greedy-decoded, asserting the EXACT reference transcript. Ground truth was captured by running
// the actual reference implementation (transformers.GlmOcrForConditionalGeneration.generate,
// do_sample=False — generation_config.json's own default) on this exact image — see
// testdata/e2e_golden.json's "generated_text_clean"/"generated_ids" and this package's git
// history for the capture script. Greedy decode is deterministic, so exact equality is the
// right assertion.
func TestLive_RealCheckpoint_OCR_Good(t *testing.T) {
	dir := resolveGlmOcrDir(t)
	m, err := Load(dir)
	if err != nil {
		t.Fatalf("Load(%s): %v", dir, err)
	}
	golden := readE2EGolden(t)
	imgBytes := readTestdata(t, "fixture.png")

	text, err := m.OCR(imgBytes, golden.PromptText)
	if err != nil {
		t.Fatalf("OCR: %v", err)
	}
	if text != golden.GeneratedTextClean {
		t.Fatalf("OCR text = %q, want %q (the reference's exact greedy transcript)", text, golden.GeneratedTextClean)
	}
}

// TestLive_RealCheckpoint_OCR_DefaultPrompt_Good proves an empty prompt defaults to "Text
// Recognition:" (the same task the golden was captured with), reproducing the identical result.
func TestLive_RealCheckpoint_OCR_DefaultPrompt_Good(t *testing.T) {
	dir := resolveGlmOcrDir(t)
	m, err := Load(dir)
	if err != nil {
		t.Fatalf("Load(%s): %v", dir, err)
	}
	golden := readE2EGolden(t)
	if golden.PromptText != "Text Recognition:" {
		t.Skip("golden was captured with a non-default prompt")
	}
	imgBytes := readTestdata(t, "fixture.png")

	text, err := m.OCR(imgBytes, "")
	if err != nil {
		t.Fatalf("OCR with empty prompt: %v", err)
	}
	if text != golden.GeneratedTextClean {
		t.Fatalf("OCR(empty prompt) text = %q, want %q", text, golden.GeneratedTextClean)
	}
}
