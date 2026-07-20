// SPDX-Licence-Identifier: EUPL-1.2

package dotsocr

import (
	"os"
	"testing"
)

// live_test.go is the #37 slice's live gate against the REAL, locally-cached
// rednote-hilab/dots.ocr checkpoint — RUNTIME-gated, reading the checkpoint dir from
// DOTS_OCR_DIR (skips cleanly when unset) and defaulting to the standard Hugging Face cache
// location, mirroring whisper's resolveWhisperTinyDir / gptoss's resolveGptOssDir: env override,
// else auto-resolve the hash-named snapshot subdirectory. Every other _test.go in this package is
// hermetic (committed small/synthetic fixtures or the real fixture.png with a hand-built
// VisionConfig, no external dependency); this file is the ONE that proves the real ~3B-parameter
// checkpoint — real tensor names, real trained weights, real tokenizer — resolves and OCRs
// correctly end to end.
func resolveDotsOCRDir(t *testing.T) string {
	t.Helper()
	if dir := os.Getenv("DOTS_OCR_DIR"); dir != "" {
		return dir
	}
	home := os.Getenv("HOME")
	base := home + "/.cache/huggingface/hub/models--rednote-hilab--dots.ocr/snapshots"
	entries, err := os.ReadDir(base)
	if err != nil {
		t.Skipf("rednote-hilab/dots.ocr snapshot dir not found (%v) — set DOTS_OCR_DIR to override, or download it: "+
			"python -c \"from huggingface_hub import snapshot_download; print(snapshot_download('rednote-hilab/dots.ocr'))\"", err)
	}
	for _, e := range entries {
		if e.IsDir() {
			return base + "/" + e.Name()
		}
	}
	t.Skip("rednote-hilab/dots.ocr snapshots dir has no snapshot subdirectory")
	return ""
}

// TestLive_RealCheckpoint_Load_Good proves Load resolves the REAL checkpoint's actual tensor
// names/shapes — geometry pinned to the numbers this whole lane was built and verified against.
func TestLive_RealCheckpoint_Load_Good(t *testing.T) {
	dir := resolveDotsOCRDir(t)
	m, err := Load(dir)
	if err != nil {
		t.Fatalf("Load(%s): %v", dir, err)
	}
	if m.Config.HiddenSize != 1536 || m.Config.NumHiddenLayers != 28 || m.Config.VocabSize != 151936 {
		t.Fatalf("real dots_ocr text geometry = hidden %d, layers %d, vocab %d; want 1536/28/151936",
			m.Config.HiddenSize, m.Config.NumHiddenLayers, m.Config.VocabSize)
	}
	vc := m.Config.VisionConfig
	if vc == nil || vc.EmbedDim != 1536 || vc.NumHiddenLayers != 42 || vc.PatchSize != 14 {
		t.Fatalf("real dots_ocr vision geometry = %+v, want embed_dim 1536, 42 layers, patch_size 14", vc)
	}
	if len(m.Weights.Layers) != 28 {
		t.Fatalf("loaded %d decoder layers, want 28", len(m.Weights.Layers))
	}
	if len(m.Weights.Vision.Blocks) != 42 {
		t.Fatalf("loaded %d vision blocks, want 42", len(m.Weights.Vision.Blocks))
	}
	if len(m.Weights.LMHead.Weight) != 151936*1536 {
		t.Fatalf("lm_head has %d elements, want %d (untied — tie_word_embeddings is false)", len(m.Weights.LMHead.Weight), 151936*1536)
	}
	if m.Config.ImageTokenID != 151665 {
		t.Fatalf("image_token_id = %d, want 151665 (<|imgpad|>)", m.Config.ImageTokenID)
	}
}

// visionGoldenBand is the observed max-abs-diff ceiling for the vision tower goldens below —
// float32 host maths vs the real torch reference accumulate in a different order over 42 layers
// (host: scalar f64 accumulation per dot product; torch: BLAS reduction trees), so this is a
// stated TOLERANCE band, not bit-exactness — the patch_embed/patch-merger LINEAR stages alone are
// within 1e-4 (see TestLive_PatchEmbed_Good); the full 42-block stack widens to this band.
const visionGoldenBand = 0.05

// TestLive_PatchEmbed_Good runs JUST the patch embed (linear projection + RMSNorm) with the REAL
// checkpoint's weights against vision_block_golden.json's synthetic input — the tightest,
// earliest-stage vision check (no attention/rotary involved yet).
func TestLive_PatchEmbed_Good(t *testing.T) {
	dir := resolveDotsOCRDir(t)
	m, err := Load(dir)
	if err != nil {
		t.Fatalf("Load(%s): %v", dir, err)
	}
	g := readVisionBlockGolden(t)
	n := g.GridH * g.GridW
	vc := m.Config.VisionConfig
	got := linearForward(g.PixelValues, m.Weights.Vision.PatchEmbed, n)
	got = rmsNormForward(got, m.Weights.Vision.PatchEmbedNorm, n, vc.EmbedDim, vc.RMSNormEps)
	if d := maxAbsDiff32(t, got, g.PatchEmbedOut); d > 1e-3 {
		t.Fatalf("patch_embed max abs diff = %v, want <=1e-3", d)
	}
}

// TestLive_VisionBlock0_Good runs patch_embed + ONE real DotsVisionBlock (blocks[0]) against
// vision_block_golden.json's block0_out — the first check that exercises the 2D rotary +
// full-image attention.
func TestLive_VisionBlock0_Good(t *testing.T) {
	dir := resolveDotsOCRDir(t)
	m, err := Load(dir)
	if err != nil {
		t.Fatalf("Load(%s): %v", dir, err)
	}
	g := readVisionBlockGolden(t)
	n := g.GridH * g.GridW
	vc := m.Config.VisionConfig
	hidden := linearForward(g.PixelValues, m.Weights.Vision.PatchEmbed, n)
	hidden = rmsNormForward(hidden, m.Weights.Vision.PatchEmbedNorm, n, vc.EmbedDim, vc.RMSNormEps)
	headDim := vc.EmbedDim / vc.NumAttentionHeads
	cosHalf, sinHalf := visionRotaryTable(g.GridH, g.GridW, vc.SpatialMergeSize, headDim)
	got, err := visionBlockForward(hidden, n, vc.EmbedDim, vc.NumAttentionHeads, m.Weights.Vision.Blocks[0], cosHalf, sinHalf, vc.RMSNormEps)
	if err != nil {
		t.Fatalf("visionBlockForward: %v", err)
	}
	if d := maxAbsDiff32(t, got, g.Block0Out); d > 1e-3 {
		t.Fatalf("block0 max abs diff = %v, want <=1e-3", d)
	}
}

// TestLive_FullVisionTower_Good calls the PUBLIC EncodeImage end to end (all 42 blocks +
// post-norm + PatchMerger) against vision_block_golden.json's full_vision_out — the strongest
// vision-side proof, through the exact entry point EncodeImage/OCR uses.
func TestLive_FullVisionTower_Good(t *testing.T) {
	dir := resolveDotsOCRDir(t)
	m, err := Load(dir)
	if err != nil {
		t.Fatalf("Load(%s): %v", dir, err)
	}
	g := readVisionBlockGolden(t)
	got, err := EncodeImage(g.PixelValues, g.GridT, g.GridH, g.GridW, m.Weights, m.Config)
	if err != nil {
		t.Fatalf("EncodeImage: %v", err)
	}
	if d := maxAbsDiff32(t, got, g.FullVisionOut); d > visionGoldenBand {
		t.Fatalf("full vision tower max abs diff = %v, want <=%v", d, visionGoldenBand)
	}
}

// runDecoderLayer replicates decodeLayersStep's per-layer body for exactly ONE layer against a
// fresh (empty) cache — a test-only helper so this file can check an INTERMEDIATE hidden state,
// which production decodeLayersStep never returns (only final logits after the whole stack).
func runDecoderLayer(embeds []float32, tn int, layer DecoderLayerWeights, cfg *Config) ([]float32, error) {
	d := cfg.HiddenSize
	headDim := d / cfg.NumAttentionHeads
	cache := SelfAttnCache{}
	residual := embeds
	normed := rmsNormForward(embeds, layer.InputNorm, tn, d, cfg.RMSNormEps)
	attnOut, err := decoderSelfAttention(normed, tn, d, cfg.NumAttentionHeads, cfg.NumKeyValueHeads, headDim, layer.Q, layer.K, layer.V, layer.O, &cache, 0, cfg.RopeTheta)
	if err != nil {
		return nil, err
	}
	hidden := addRows(residual, attnOut)
	residual = hidden
	normed = rmsNormForward(hidden, layer.PostAttnNorm, tn, d, cfg.RMSNormEps)
	ff := swiGLU(normed, layer.Gate, layer.Up, layer.Down, tn)
	return addRows(residual, ff), nil
}

// runDecoderStackHidden replicates decodeLayersStep's full loop but returns the post-final-norm
// hidden state at every position (not lm_head logits) — the test-only counterpart used to check
// text_layer_golden.json's final_hidden_last independently of the lm_head projection.
func runDecoderStackHidden(embeds []float32, tn int, w *Weights, cfg *Config) ([]float32, error) {
	d := cfg.HiddenSize
	headDim := d / cfg.NumAttentionHeads
	hidden := embeds
	cache := NewSelfAttnCache(len(w.Layers))
	for li := range w.Layers {
		layer := w.Layers[li]
		residual := hidden
		normed := rmsNormForward(hidden, layer.InputNorm, tn, d, cfg.RMSNormEps)
		attnOut, err := decoderSelfAttention(normed, tn, d, cfg.NumAttentionHeads, cfg.NumKeyValueHeads, headDim, layer.Q, layer.K, layer.V, layer.O, &cache[li], 0, cfg.RopeTheta)
		if err != nil {
			return nil, err
		}
		hidden = addRows(residual, attnOut)
		residual = hidden
		normed = rmsNormForward(hidden, layer.PostAttnNorm, tn, d, cfg.RMSNormEps)
		ff := swiGLU(normed, layer.Gate, layer.Up, layer.Down, tn)
		hidden = addRows(residual, ff)
	}
	return rmsNormForward(hidden, w.FinalNorm, tn, d, cfg.RMSNormEps), nil
}

// TestLive_DecoderLayer0_Good runs ONE real Qwen2DecoderLayer (layers[0]) against
// text_layer_golden.json's layer0_out, on the golden's own short real-tokenized prompt.
func TestLive_DecoderLayer0_Good(t *testing.T) {
	dir := resolveDotsOCRDir(t)
	m, err := Load(dir)
	if err != nil {
		t.Fatalf("Load(%s): %v", dir, err)
	}
	g := readTextLayerGolden(t)
	embeds, err := EmbedTokens(g.InputIDs, m.Weights, m.Config)
	if err != nil {
		t.Fatalf("EmbedTokens: %v", err)
	}
	if d := maxAbsDiff32(t, embeds, g.EmbedOut); d > 1e-4 {
		t.Fatalf("embed_tokens max abs diff = %v, want <=1e-4", d)
	}
	got, err := runDecoderLayer(embeds, len(g.InputIDs), m.Weights.Layers[0], m.Config)
	if err != nil {
		t.Fatalf("runDecoderLayer: %v", err)
	}
	if d := maxAbsDiff32(t, got, g.Layer0Out); d > 1e-3 {
		t.Fatalf("decoder layer0 max abs diff = %v, want <=1e-3", d)
	}
}

// TestLive_DecoderFullStack_Good runs all 28 real decoder layers + final norm against
// text_layer_golden.json's final_hidden_last, THEN the real lm_head against its sampled logits
// (evenly-strided across the full 151936 vocab plus the true top-10 — see golden_test.go's doc
// comment for why this sample is a strong proxy for the whole projection matrix).
func TestLive_DecoderFullStack_Good(t *testing.T) {
	dir := resolveDotsOCRDir(t)
	m, err := Load(dir)
	if err != nil {
		t.Fatalf("Load(%s): %v", dir, err)
	}
	g := readTextLayerGolden(t)
	embeds, err := EmbedTokens(g.InputIDs, m.Weights, m.Config)
	if err != nil {
		t.Fatalf("EmbedTokens: %v", err)
	}
	hiddenAll, err := runDecoderStackHidden(embeds, len(g.InputIDs), m.Weights, m.Config)
	if err != nil {
		t.Fatalf("runDecoderStackHidden: %v", err)
	}
	d := m.Config.HiddenSize
	lastHidden := hiddenAll[(len(g.InputIDs)-1)*d : len(g.InputIDs)*d]
	if diff := maxAbsDiff32(t, lastHidden, g.FinalHiddenLast); diff > 5e-3 {
		t.Fatalf("final hidden (last position) max abs diff = %v, want <=5e-3", diff)
	}

	logits := linearForward(lastHidden, m.Weights.LMHead, 1)
	sample := g.LogitsLastSampled
	for k, idx := range sample.SampleIndices {
		if diff := float32AbsDiff(logits[idx], sample.SampleValues[k]); diff > 0.25 {
			t.Fatalf("logits[%d] = %v, want %v (diff %v)", idx, logits[idx], sample.SampleValues[k], diff)
		}
	}
	if argmax32(logits) != sample.ArgmaxID {
		t.Fatalf("argmax(logits) = %d, want %d (the golden's true top-1)", argmax32(logits), sample.ArgmaxID)
	}
}

func float32AbsDiff(a, b float32) float32 {
	d := a - b
	if d < 0 {
		return -d
	}
	return d
}

// TestLive_RealCheckpoint_OCR_Good is the E2E gate: the committed testdata/fixture.png ("Lethean
// OCR 2026" rendered PIL text-on-white) through the REAL checkpoint via the PUBLIC OCR method,
// asserting the EXACT reference generation. Ground truth was captured by driving the real
// checkpoint's own modules (forward(), not the broken generate() — see e2e_golden.json's doc
// comment in golden_test.go) with greedy selection; greedy decode is deterministic, so exact
// string equality is the right assertion.
func TestLive_RealCheckpoint_OCR_Good(t *testing.T) {
	if os.Getenv("DOTS_OCR_E2E") == "" {
		t.Skip("full real-checkpoint decode runs ~10 minutes and overruns go test's default " +
			"-timeout inside wholesale ./model/... gates (#58) — set DOTS_OCR_E2E=1 (with " +
			"-timeout raised) to run it deliberately")
	}
	dir := resolveDotsOCRDir(t)
	m, err := Load(dir)
	if err != nil {
		t.Fatalf("Load(%s): %v", dir, err)
	}
	g := readE2EGolden(t)
	png := readTestdata(t, "fixture.png")

	text, err := m.OCR(png, g.Prompt)
	if err != nil {
		t.Fatalf("OCR: %v", err)
	}
	if text != g.Text {
		t.Fatalf("OCR text =\n%q\nwant\n%q", text, g.Text)
	}
}
