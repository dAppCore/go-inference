// SPDX-Licence-Identifier: EUPL-1.2

package dotsocr

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// golden_test.go is the shared loader for every committed golden fixture under testdata/ —
// captured from the REAL rednote-hilab/dots.ocr checkpoint (torch 2.11 + transformers 5.5, float32
// CPU) so image_test.go/vision_test.go/decoder_test.go/prompt_test.go/live_test.go compare against
// one consistent set of structs. Test-only: stdlib os/encoding/json/path-filepath are fine here
// (the banned-import rule is for production library code — see whisper.golden_test.go's identical
// doc-comment precedent).

func readTestdata(t *testing.T, name string) []byte {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatalf("read testdata/%s: %v", name, err)
	}
	return b
}

func readJSONGolden(t *testing.T, name string, v any) {
	t.Helper()
	if err := json.Unmarshal(readTestdata(t, name), v); err != nil {
		t.Fatalf("parse testdata/%s: %v", name, err)
	}
}

// writeFile writes content to dir/name — the small-fixture counterpart to readTestdata, used by
// _Bad/_Ugly tests that need a malformed/minimal on-disk file (mirrors whisper.writeFile).
func writeFile(t *testing.T, dir, name, content string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o600); err != nil {
		t.Fatalf("write %s/%s: %v", dir, name, err)
	}
}

// --- smart_resize_golden.json — captured by calling transformers'
// image_processing_pil_qwen2_vl.smart_resize directly (pure dimension arithmetic, no checkpoint
// needed) across a spread of heights/widths (already-aligned, needs-rounding, below min_pixels,
// above max_pixels, exact boundary, extreme-but-legal aspect ratio). ---

type smartResizeCase struct {
	Height, Width               int
	ResizedHeight, ResizedWidth int
}

func (c *smartResizeCase) UnmarshalJSON(b []byte) error {
	var raw struct {
		Height        int `json:"height"`
		Width         int `json:"width"`
		ResizedHeight int `json:"resized_height"`
		ResizedWidth  int `json:"resized_width"`
	}
	if err := json.Unmarshal(b, &raw); err != nil {
		return err
	}
	c.Height, c.Width, c.ResizedHeight, c.ResizedWidth = raw.Height, raw.Width, raw.ResizedHeight, raw.ResizedWidth
	return nil
}

type smartResizeGolden struct {
	Factor    int               `json:"factor"`
	MinPixels int               `json:"min_pixels"`
	MaxPixels int               `json:"max_pixels"`
	Cases     []smartResizeCase `json:"cases"`
}

func readSmartResizeGolden(t *testing.T) smartResizeGolden {
	t.Helper()
	var g smartResizeGolden
	readJSONGolden(t, "smart_resize_golden.json", &g)
	return g
}

// --- image_preproc_golden.json — captured by running the REAL Qwen2VLImageProcessorPil (DOTS-
// OCR's own preprocessor_config.json kwargs) on the committed testdata/fixture.png. Only a
// spread-out SAMPLE of patch rows is committed (every 17th + the last), not the full 120×588
// matrix — see step3_full.py's capture notes; the E2E golden below is the whole-array backstop. ---

type imagePreprocGolden struct {
	ImageWidth         int         `json:"image_width"`
	ImageHeight        int         `json:"image_height"`
	GridTHW            [3]int      `json:"grid_thw"`
	PixelValuesShape   [2]int      `json:"pixel_values_shape"`
	SamplePatchIndices []int       `json:"sample_patch_indices"`
	SamplePatchRows    [][]float32 `json:"sample_patch_rows"`
}

func readImagePreprocGolden(t *testing.T) imagePreprocGolden {
	t.Helper()
	var g imagePreprocGolden
	readJSONGolden(t, "image_preproc_golden.json", &g)
	return g
}

// --- prompt_golden.json — captured by rendering the REAL chat_template.json through
// transformers' apply_chat_template for a list-content (image+text) user message (this
// checkpoint's own README-documented usage), then tokenizing with the real tokenizer.json. ---

type promptGolden struct {
	Prompt             string  `json:"prompt"`
	NMergedImageTokens int     `json:"n_merged_image_tokens"`
	ExpandedText       string  `json:"expanded_text"`
	InputIDs           []int32 `json:"input_ids"`
}

func readPromptGolden(t *testing.T) promptGolden {
	t.Helper()
	var g promptGolden
	readJSONGolden(t, "prompt_golden.json", &g)
	return g
}

// --- vision_block_golden.json — captured against the REAL DotsVisionTransformer (real trained
// weights) on a SMALL synthetic 2×2 patch grid (deterministic seeded random pixel values, not
// tied to fixture.png — kept small on purpose): patch_embed alone, one full DotsVisionBlock (a
// forward hook on blocks[0]), the rotary table, and the FULL tower (all 42 blocks + post-norm +
// merger) end to end. ---

type visionBlockGolden struct {
	GridT         int       `json:"grid_t"`
	GridH         int       `json:"grid_h"`
	GridW         int       `json:"grid_w"`
	EmbedDim      int       `json:"embed_dim"`
	HiddenSize    int       `json:"hidden_size"`
	PixelValues   []float32 `json:"pixel_values"`
	PatchEmbedOut []float32 `json:"patch_embed_out"`
	Block0Out     []float32 `json:"block0_out"`
	FullVisionOut []float32 `json:"full_vision_out"`
	RotPosEmb     []float32 `json:"rot_pos_emb"`
}

func readVisionBlockGolden(t *testing.T) visionBlockGolden {
	t.Helper()
	var g visionBlockGolden
	readJSONGolden(t, "vision_block_golden.json", &g)
	return g
}

// --- text_layer_golden.json — captured against the REAL Qwen2ForCausalLM decoder (real trained
// weights) on a short real tokenized prompt ("Hello, describe this image.", no vision embedding
// splice — the decoder is architecturally independent of the vision tower): the token embedding,
// one full Qwen2DecoderLayer (a forward hook on layers[0]), the final post-norm hidden state at
// the last position, and a committed-size-friendly SAMPLE of the lm_head logits at that position
// (evenly-strided across the whole 151936 vocab plus the true top-10 — a stride/transpose bug in
// lm_head would misplace EVERY position, not just a few, so this sample is a strong proxy for the
// full projection matrix without committing 151936 floats). ---

type sampledLogits struct {
	VocabSize     int       `json:"vocab_size"`
	Stride        int       `json:"stride"`
	SampleIndices []int     `json:"sample_indices"`
	SampleValues  []float32 `json:"sample_values"`
	TopIDs        []int32   `json:"top_ids"`
	TopValues     []float32 `json:"top_values"`
	ArgmaxID      int32     `json:"argmax_id"`
}

type textLayerGolden struct {
	InputIDs          []int32       `json:"input_ids"`
	EmbedOut          []float32     `json:"embed_out"`
	Layer0Out         []float32     `json:"layer0_out"`
	FinalHiddenLast   []float32     `json:"final_hidden_last"`
	LogitsLastSampled sampledLogits `json:"logits_last_sampled"`
}

func readTextLayerGolden(t *testing.T) textLayerGolden {
	t.Helper()
	var g textLayerGolden
	readJSONGolden(t, "text_layer_golden.json", &g)
	return g
}

// --- e2e_golden.json — captured by a hand-rolled Python greedy loop calling the REAL model's own
// forward() at every step (model.generate() itself throws inside this checkpoint's shipped
// prepare_inputs_for_generation override under this transformers version — a real incompatibility
// in the custom_code, not something this port controls; see step3_full.py's capture notes) on the
// real fixture.png + the checkpoint's own README-documented prompt_layout_all_en prompt. ---

type e2eGolden struct {
	Prompt       string  `json:"prompt"`
	InputIDs     []int32 `json:"input_ids"`
	GeneratedIDs []int32 `json:"generated_ids"`
	Text         string  `json:"text"`
}

func readE2EGolden(t *testing.T) e2eGolden {
	t.Helper()
	var g e2eGolden
	readJSONGolden(t, "e2e_golden.json", &g)
	return g
}

// maxAbsDiff32 reports the largest |a[i]-b[i]| over two equal-length float32 slices, failing the
// test immediately with FailNow if the lengths differ (a shape mismatch is a distinct bug from a
// value mismatch and should not be graded by tolerance) — mirrors whisper.maxAbsDiff32.
func maxAbsDiff32(t *testing.T, a, b []float32) float64 {
	t.Helper()
	if len(a) != len(b) {
		t.Fatalf("length mismatch: got %d, want %d", len(a), len(b))
	}
	var max float64
	for i := range a {
		d := float64(a[i]) - float64(b[i])
		if d < 0 {
			d = -d
		}
		if d > max {
			max = d
		}
	}
	return max
}
