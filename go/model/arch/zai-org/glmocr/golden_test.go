// SPDX-Licence-Identifier: EUPL-1.2

package glmocr

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// golden_test.go is the shared loader for every committed golden fixture under testdata/ —
// mirrors arch/openai/whisper/golden_test.go's shape. Test-only: stdlib os/encoding-json/path-
// filepath are fine here (the banned-import rule is for production library code).
//
// --- block_goldens.json — captured by a Python script (torch.manual_seed-seeded, toy
// dimensions) that runs the REAL transformers glm_ocr modules (GlmOcrVisionModel,
// GlmOcrTextModel, GlmOcrModel.get_rope_index, GlmOcrRMSNorm) directly and hooks their
// intermediate activations — never reimplemented in Python, so this pins the Go port against
// the actual reference code, not a second-hand description of it. Not committed to the repo
// (a one-off capture tool); only its JSON output is testdata. ---
//
// --- e2e_golden.json — captured against the REAL zai-org/GLM-OCR checkpoint (float32, CPU) on
// this package's own testdata/fixture.png, via processor.apply_chat_template + model.generate
// (do_sample=false, matching generation_config.json), plus the real vision tower's
// pooler_output on that same image. ---

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

type visionBlockGolden struct {
	Config struct {
		HiddenSize        int     `json:"hidden_size"`
		NumHeads          int     `json:"num_heads"`
		Depth             int     `json:"depth"`
		InChannels        int     `json:"in_channels"`
		PatchSize         int     `json:"patch_size"`
		TemporalPatchSize int     `json:"temporal_patch_size"`
		SpatialMergeSize  int     `json:"spatial_merge_size"`
		OutHiddenSize     int     `json:"out_hidden_size"`
		IntermediateSize  int     `json:"intermediate_size"`
		RMSNormEps        float32 `json:"rms_norm_eps"`
		HiddenAct         string  `json:"hidden_act"`
	} `json:"config"`
	GridTHW          [][]int              `json:"grid_thw"`
	PixelValues      []float32            `json:"pixel_values"`
	StateDict        map[string][]float32 `json:"state_dict"`
	RotaryPosEmb     []float32            `json:"rotary_pos_emb"`
	PosIDs           [][]int              `json:"pos_ids"`
	PatchEmbedOut    []float32            `json:"patch_embed_out"`
	Block0Out        []float32            `json:"block0_out"`
	Block1Out        []float32            `json:"block1_out"`
	PostLayernormOut []float32            `json:"post_layernorm_out"`
	DownsampleOut    []float32            `json:"downsample_out"`
	MergerOut        []float32            `json:"merger_out"`
	LastHiddenState  []float32            `json:"last_hidden_state"`
	PoolerOutput     []float32            `json:"pooler_output"`
}

type rmsNormGolden struct {
	Eps    float32   `json:"eps"`
	Weight []float32 `json:"weight"`
	Input  []float32 `json:"input"`
	Output []float32 `json:"output"`
	Dim    int       `json:"dim"`
}

type textBlockGolden struct {
	Config struct {
		VocabSize           int     `json:"vocab_size"`
		HiddenSize          int     `json:"hidden_size"`
		IntermediateSize    int     `json:"intermediate_size"`
		NumHiddenLayers     int     `json:"num_hidden_layers"`
		NumAttentionHeads   int     `json:"num_attention_heads"`
		NumKeyValueHeads    int     `json:"num_key_value_heads"`
		HeadDim             int     `json:"head_dim"`
		RMSNormEps          float32 `json:"rms_norm_eps"`
		HiddenAct           string  `json:"hidden_act"`
		RopeTheta           float32 `json:"rope_theta"`
		MropeSection        []int   `json:"mrope_section"`
		PartialRotaryFactor float32 `json:"partial_rotary_factor"`
	} `json:"config"`
	InputIDs        [][]int32            `json:"input_ids"`
	PositionIDs     [][][]int            `json:"position_ids"`
	StateDict       map[string][]float32 `json:"state_dict"`
	InputsEmbeds    []float32            `json:"inputs_embeds"`
	Cos             []float32            `json:"cos"`
	Sin             []float32            `json:"sin"`
	Layer0Out       []float32            `json:"layer0_out"`
	Layer1Out       []float32            `json:"layer1_out"`
	NormOut         []float32            `json:"norm_out"`
	LastHiddenState []float32            `json:"last_hidden_state"`
}

type lmHeadGolden struct {
	Weight []float32 `json:"weight"`
	Input  []float32 `json:"input"`
	Output []float32 `json:"output"`
	In     int       `json:"in"`
	Out    int       `json:"out"`
}

type ropeIndexGolden struct {
	InputIDs            [][]int32 `json:"input_ids"`
	MMTokenTypeIDs      [][]int32 `json:"mm_token_type_ids"`
	ImageGridTHW        [][]int   `json:"image_grid_thw"`
	SpatialMergeSize    int       `json:"spatial_merge_size"`
	PositionIDs         [][][]int `json:"position_ids"`
	MropePositionDeltas [][]int   `json:"mrope_position_deltas"`
}

type blockGoldens struct {
	Vision    visionBlockGolden `json:"vision"`
	RMSNorm   rmsNormGolden     `json:"rmsnorm"`
	Text      textBlockGolden   `json:"text"`
	LMHead    lmHeadGolden      `json:"lm_head"`
	RopeIndex ropeIndexGolden   `json:"rope_index"`
}

func readBlockGoldens(t *testing.T) blockGoldens {
	t.Helper()
	var g blockGoldens
	readJSONGolden(t, "block_goldens.json", &g)
	return g
}

type e2eGolden struct {
	Checkpoint              string    `json:"checkpoint"`
	PromptText              string    `json:"prompt_text"`
	InputIDs                []int32   `json:"input_ids"`
	MMTokenTypeIDs          []int32   `json:"mm_token_type_ids"`
	ImageGridTHW            []int     `json:"image_grid_thw"`
	PixelValuesShape        []int     `json:"pixel_values_shape"`
	VisionPoolerOutput      []float32 `json:"vision_pooler_output"`
	VisionPoolerOutputShape []int     `json:"vision_pooler_output_shape"`
	GeneratedIDs            []int32   `json:"generated_ids"`
	GeneratedTextRaw        string    `json:"generated_text_raw"`
	GeneratedTextClean      string    `json:"generated_text_clean"`
}

func readE2EGolden(t *testing.T) e2eGolden {
	t.Helper()
	var g e2eGolden
	readJSONGolden(t, "e2e_golden.json", &g)
	return g
}

// maxAbsDiff32 reports the largest |a[i]-b[i]| over two equal-length float32 slices, failing
// the test immediately with FailNow if the lengths differ.
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
