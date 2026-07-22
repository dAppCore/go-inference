// SPDX-Licence-Identifier: EUPL-1.2

package deepseekvl2

import (
	"os"
	"testing"
)

// live_test.go is the #37 slice's live gate against the REAL, locally-cached
// deepseek-ai/DeepSeek-OCR checkpoint — RUNTIME-gated, reading the checkpoint dir from
// DEEPSEEK_OCR_DIR (skips cleanly when unset) and defaulting to the standard Hugging Face cache
// location, mirroring whisper's resolveWhisperTinyDir (arch/openai/whisper/live_test.go): env
// override, else auto-resolve the hash-named snapshot subdirectory. Every other _test.go in this
// package is hermetic (committed small/synthetic fixtures, or the toy/spot-check goldens
// captured from the real reference implementation — see golden_test.go); this file is the ONE
// that proves the real ~6.7GB checkpoint — real tensor names, real tokenizer, real trained
// weights — resolves and runs OCR correctly end to end.

func resolveDeepSeekOCRDir(t *testing.T) string {
	t.Helper()
	if dir := os.Getenv("DEEPSEEK_OCR_DIR"); dir != "" {
		return dir
	}
	home := os.Getenv("HOME")
	base := home + "/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR/snapshots"
	entries, err := os.ReadDir(base)
	if err != nil {
		t.Skipf("deepseek-ai/DeepSeek-OCR snapshot dir not found (%v) — set DEEPSEEK_OCR_DIR to override, or download it: "+
			"python -c \"from huggingface_hub import snapshot_download; print(snapshot_download('deepseek-ai/DeepSeek-OCR'))\"", err)
	}
	for _, e := range entries {
		if e.IsDir() {
			return base + "/" + e.Name()
		}
	}
	t.Skip("deepseek-ai/DeepSeek-OCR snapshots dir has no snapshot subdirectory")
	return ""
}

// TestLive_RealCheckpoint_Load proves Load resolves the REAL checkpoint's actual tensor names/
// shapes (the "never guessed" requirement) — not just the hermetic/toy fixtures every other test
// in this package uses. Geometry pinned to the numbers this whole lane was built and verified
// against (config.go's doc comment: read from config.json's top-level fields).
func TestLive_RealCheckpoint_Load(t *testing.T) {
	dir := resolveDeepSeekOCRDir(t)
	m, err := Load(dir)
	if err != nil {
		t.Fatalf("Load(%s): %v", dir, err)
	}
	if m.Config.HiddenSize != 1280 || m.Config.NumHiddenLayers != 12 || m.Config.VocabSize != 129280 {
		t.Fatalf("real DeepSeek-OCR geometry = hidden_size %d, num_hidden_layers %d, vocab %d; want 1280/12/129280",
			m.Config.HiddenSize, m.Config.NumHiddenLayers, m.Config.VocabSize)
	}
	if m.Config.NRoutedExperts != 64 || m.Config.NSharedExperts != 2 || m.Config.NumExpertsPerTok != 6 {
		t.Fatalf("real DeepSeek-OCR MoE geometry = routed %d shared %d perTok %d; want 64/2/6",
			m.Config.NRoutedExperts, m.Config.NSharedExperts, m.Config.NumExpertsPerTok)
	}
	if m.Config.UseMLA {
		t.Fatal("real DeepSeek-OCR checkpoint has use_mla=true; want false (plain rotary MHA — decoder.go's whole design assumes this)")
	}
	if len(m.Weights.Decoder.Layers) != 12 {
		t.Fatalf("loaded %d decoder layers, want 12", len(m.Weights.Decoder.Layers))
	}
	if m.Weights.Decoder.Layers[0].IsMoE {
		t.Fatal("decoder layer 0 loaded as MoE, want dense (first_k_dense_replace=1)")
	}
	if !m.Weights.Decoder.Layers[1].IsMoE || len(m.Weights.Decoder.Layers[1].Experts) != 64 {
		t.Fatalf("decoder layer 1 IsMoE=%v with %d experts, want true/64", m.Weights.Decoder.Layers[1].IsMoE, len(m.Weights.Decoder.Layers[1].Experts))
	}
	if len(m.Weights.SAM.Blocks) != samDepth {
		t.Fatalf("loaded %d SAM blocks, want %d", len(m.Weights.SAM.Blocks), samDepth)
	}
	if len(m.Weights.CLIP.Blocks) != clipNumLayers {
		t.Fatalf("loaded %d CLIP blocks, want %d", len(m.Weights.CLIP.Blocks), clipNumLayers)
	}
	if en, ok := m.Tokenizer.TokenID("<image>"); !ok || en != 128815 {
		t.Fatalf("real checkpoint's <image> token id = %d, ok=%v; want 128815", en, ok)
	}
}

// TestLive_RealCheckpoint_OCR is the E2E gate: the committed fixture.png (a deterministic,
// generated-not-photographed image — testdata/gen_fixture.py) through the REAL checkpoint, greedy
// decode, asserting the EXACT reference transcript. Ground truth was captured by running the
// actual reference implementation (deepseek-ai/DeepSeek-OCR's own custom_code
// DeepseekOCRForCausalLM, direct construction + real safetensors weights — trust_remote_code's
// AutoModel path hits a transformers-5.x/4.46-era-custom-code config incompatibility for this
// specific checkpoint, worked around by constructing the (verified-correct) config and model
// directly instead — see the capture script referenced in testdata/e2e_golden.json's sibling
// comment) on this exact image, greedy (do_sample=False-equivalent argmax, deterministic). Greedy
// decode is deterministic, so exact string equality is the right assertion (not a similarity/
// edit-distance heuristic) — matching whisper's TestLive_RealCheckpoint_Transcribe precedent.
func TestLive_RealCheckpoint_OCR(t *testing.T) {
	dir := resolveDeepSeekOCRDir(t)
	m, err := Load(dir)
	if err != nil {
		t.Fatalf("Load(%s): %v", dir, err)
	}
	imageBytes := readTestdata(t, "fixture.png")
	golden := readE2EGolden(t)

	result, err := m.OCR(imageBytes, Options{})
	if err != nil {
		t.Fatalf("OCR: %v", err)
	}
	if result.Text != golden.Text {
		t.Fatalf("OCR text = %q, want %q (the reference's exact greedy transcript)", result.Text, golden.Text)
	}
}

// TestLive_RealCheckpoint_OCRImage_Good proves the inference.OCRModel adapter (serve/CLI's
// capability-discovery surface) reproduces OCR's own exact-golden result through its (text, error)
// reshape — mirrors whisper's TestLive_RealCheckpoint_TranscribeAudio_Good.
func TestLive_RealCheckpoint_OCRImage_Good(t *testing.T) {
	dir := resolveDeepSeekOCRDir(t)
	m, err := Load(dir)
	if err != nil {
		t.Fatalf("Load(%s): %v", dir, err)
	}
	imageBytes := readTestdata(t, "fixture.png")
	golden := readE2EGolden(t)

	text, err := m.OCRImage(imageBytes, "")
	if err != nil {
		t.Fatalf("OCRImage: %v", err)
	}
	if text != golden.Text {
		t.Fatalf("OCRImage text = %q, want %q", text, golden.Text)
	}
}
