// SPDX-Licence-Identifier: EUPL-1.2

package deepseekvl2

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// golden_test.go is the shared loader for every committed golden fixture under testdata/ —
// captured from the REAL deepseek-ai/DeepSeek-OCR reference implementation (its own custom_code,
// run under torch 2.11/transformers 5.5 with three small, documented compatibility shims for the
// transformers-5.x/4.46-era-custom-code version gap — see the capture scripts these fixtures cite)
// so vision_sam_test.go/vision_clip_test.go/vision_test.go/decoder_test.go/live_test.go compare
// against one consistent set of structs. Test-only: stdlib os/encoding/json/path-filepath are
// fine here (the banned-import rule is for production library code — see whisper/golden_test.go's
// identical doc comment for the house precedent). Every struct field below carries an EXPLICIT
// json tag matching the capture scripts' own snake_case keys — encoding/json's default
// case-insensitive matching does not bridge snake_case to CamelCase, so an untagged field here
// would silently zero-value rather than populate.

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

// --- vision_toy_golden.json — captured by calling the REAL reference sub-module classes
// (deepencoder.PatchEmbed/Block/LayerNorm2d/nn.Conv2d/CLIPVisionEmbeddings/NoTPTransformerBlock/
// MlpProjector) directly at small, fully-controlled dims (bypassing ImageEncoderViT/build_clip_l's
// hardcoded 768/1024-wide wiring — see testdata/gen_fixture.py's sibling capture_goldens.py doc
// comment for why), seeded deterministic weights, run through the real forward. ---

type namedWeightsGolden map[string][]float32

func (g namedWeightsGolden) mustGet(t *testing.T, name string) []float32 {
	t.Helper()
	v, ok := g[name]
	if !ok {
		t.Fatalf("golden weights missing %q", name)
	}
	return v
}

type patchEmbedGolden struct {
	Kernel        int       `json:"kernel"`
	InChans       int       `json:"in_chans"`
	EmbedDim      int       `json:"embed_dim"`
	Weight        []float32 `json:"weight"`
	Bias          []float32 `json:"bias"`
	InputShape    []int     `json:"input_shape"`
	Input         []float32 `json:"input"`
	PosEmbedShape []int     `json:"pos_embed_shape"`
	PosEmbed      []float32 `json:"pos_embed"`
	OutputShape   []int     `json:"output_shape"`
	Output        []float32 `json:"output"`
}

type samBlockGolden struct {
	Dim         int                `json:"dim"`
	NumHeads    int                `json:"num_heads"`
	MLPRatio    float64            `json:"mlp_ratio"`
	WindowSize  int                `json:"window_size"`
	Grid        []int              `json:"grid"`
	Weights     namedWeightsGolden `json:"weights"`
	InputShape  []int              `json:"input_shape"`
	Input       []float32          `json:"input"`
	OutputShape []int              `json:"output_shape"`
	Output      []float32          `json:"output"`
}

type samNeckGolden struct {
	Weights      namedWeightsGolden `json:"weights"`
	InputShape   []int              `json:"input_shape"`
	Input        []float32          `json:"input"`
	NeckOutShape []int              `json:"neck_out_shape"`
	NeckOut      []float32          `json:"neck_out"`
	OutputShape  []int              `json:"output_shape"`
	Output       []float32          `json:"output"`
}

type clipEmbeddingsGolden struct {
	HiddenSize              int       `json:"hidden_size"`
	ImageSize               int       `json:"image_size"`
	PatchSize               int       `json:"patch_size"`
	ClassEmbedding          []float32 `json:"class_embedding"`
	PositionEmbeddingWeight []float32 `json:"position_embedding_weight"`
	PatchEmbedsShape        []int     `json:"patch_embeds_shape"`
	PatchEmbeds             []float32 `json:"patch_embeds"`
	OutputShape             []int     `json:"output_shape"`
	Output                  []float32 `json:"output"`
}

type clipBlockGolden struct {
	HiddenSize    int                `json:"hidden_size"`
	NumHeads      int                `json:"num_heads"`
	FFNHiddenSize int                `json:"ffn_hidden_size"`
	Weights       namedWeightsGolden `json:"weights"`
	InputShape    []int              `json:"input_shape"`
	Input         []float32          `json:"input"`
	OutputShape   []int              `json:"output_shape"`
	Output        []float32          `json:"output"`
}

type projectorGolden struct {
	InputDim    int       `json:"input_dim"`
	NEmbed      int       `json:"n_embed"`
	Weight      []float32 `json:"weight"`
	Bias        []float32 `json:"bias"`
	InputShape  []int     `json:"input_shape"`
	Input       []float32 `json:"input"`
	OutputShape []int     `json:"output_shape"`
	Output      []float32 `json:"output"`
}

type visionToyGolden struct {
	PatchEmbed        patchEmbedGolden     `json:"patch_embed"`
	SAMBlockWindowed  samBlockGolden       `json:"sam_block_windowed"`
	SAMBlockGlobal    samBlockGolden       `json:"sam_block_global"`
	SAMNeckDownsample samNeckGolden        `json:"sam_neck_downsample"`
	CLIPEmbeddings    clipEmbeddingsGolden `json:"clip_embeddings"`
	CLIPBlock         clipBlockGolden      `json:"clip_block"`
	Projector         projectorGolden      `json:"projector"`
}

func readVisionToyGolden(t *testing.T) visionToyGolden {
	t.Helper()
	var g visionToyGolden
	readJSONGolden(t, "vision_toy_golden.json", &g)
	return g
}

// --- decoder_toy_golden.json — captured by constructing the REAL DeepseekV2Model/MoEGate classes
// directly with a small, fully-controlled config (2 layers: one dense, one MoE with 4 toy
// experts), seeded deterministic weights, run through the real forward (use_cache=False — see the
// capture script's METHODOLOGY note on why: cross-transformers-version cache-plumbing shims are
// avoided entirely by never touching the cache machinery). ---

type decoderToyConfigGolden struct {
	VocabSize           int     `json:"vocab_size"`
	HiddenSize          int     `json:"hidden_size"`
	IntermediateSize    int     `json:"intermediate_size"`
	MoEIntermediateSize int     `json:"moe_intermediate_size"`
	NumHiddenLayers     int     `json:"num_hidden_layers"`
	NumAttentionHeads   int     `json:"num_attention_heads"`
	NumKeyValueHeads    int     `json:"num_key_value_heads"`
	NSharedExperts      int     `json:"n_shared_experts"`
	NRoutedExperts      int     `json:"n_routed_experts"`
	NumExpertsPerTok    int     `json:"num_experts_per_tok"`
	FirstKDenseReplace  int     `json:"first_k_dense_replace"`
	RopeTheta           float64 `json:"rope_theta"`
	RMSNormEps          float64 `json:"rms_norm_eps"`
	NormTopkProb        bool    `json:"norm_topk_prob"`
	RoutedScalingFactor float64 `json:"routed_scaling_factor"`
}

type moeGateLayerGolden struct {
	InputShape []int     `json:"input_shape"`
	Input      []float32 `json:"input"`
	TopkIdx    [][]int   `json:"topk_idx"`
	TopkWeight []float32 `json:"topk_weight"`
}

type decoderToyGolden struct {
	Config               decoderToyConfigGolden `json:"config"`
	Weights              namedWeightsGolden     `json:"weights"`
	LMHeadWeight         []float32              `json:"lm_head_weight"`
	InputIDs             []int32                `json:"input_ids"`
	HiddenStatesPerLayer [][]float32            `json:"hidden_states_per_layer"`
	FinalHiddenShape     []int                  `json:"final_hidden_shape"`
	FinalHidden          []float32              `json:"final_hidden"`
	Logits               []float32              `json:"logits"`
	ArgmaxLast           int32                  `json:"argmax_last"`
	MoEGateLayer1        moeGateLayerGolden     `json:"moe_gate_layer1"`
}

func readDecoderToyGolden(t *testing.T) decoderToyGolden {
	t.Helper()
	var g decoderToyGolden
	readJSONGolden(t, "decoder_toy_golden.json", &g)
	return g
}

// --- moe_gate_real_golden.json — captured against the REAL n_routed_experts=64/
// num_experts_per_tok=6 routing dimensions (hidden_size shrunk to 32 — only the incidental gate
// linear projection's input width, not the routing logic itself). ---

type moeGateRealGolden struct {
	HiddenSize       int       `json:"hidden_size"`
	NRoutedExperts   int       `json:"n_routed_experts"`
	NumExpertsPerTok int       `json:"num_experts_per_tok"`
	GateWeight       []float32 `json:"gate_weight"`
	InputShape       []int     `json:"input_shape"`
	Input            []float32 `json:"input"`
	TopkIdx          [][]int   `json:"topk_idx"`
	TopkWeight       []float32 `json:"topk_weight"`
}

func readMoEGateRealGolden(t *testing.T) moeGateRealGolden {
	t.Helper()
	var g moeGateRealGolden
	readJSONGolden(t, "moe_gate_real_golden.json", &g)
	return g
}

// --- e2e_vision_golden.json / e2e_golden.json — captured against the REAL, full-scale checkpoint
// (see live_test.go). ---

type e2eVisionGolden struct {
	ImageTokenCount        int       `json:"image_token_count"`
	Grid                   []int     `json:"grid"`
	VisionFeaturesShape    []int     `json:"vision_features_shape"`
	VisionFeaturesFirstRow []float32 `json:"vision_features_first_row"`
	VisionFeaturesLastRow  []float32 `json:"vision_features_last_row"`
	VisionFeaturesRow16    []float32 `json:"vision_features_row16"`
	VisionFeaturesChecksum float64   `json:"vision_features_checksum"`
}

func readE2EVisionGolden(t *testing.T) e2eVisionGolden {
	t.Helper()
	var g e2eVisionGolden
	readJSONGolden(t, "e2e_vision_golden.json", &g)
	return g
}

type e2eGolden struct {
	Prompt       string  `json:"prompt"`
	ImageTokenID int32   `json:"image_token_id"`
	InputIDs     []int32 `json:"input_ids"`
	GeneratedIDs []int32 `json:"generated_ids"`
	Text         string  `json:"text"`
	HitEOS       bool    `json:"hit_eos"`
	MaxNewTokens int     `json:"max_new_tokens"`
}

func readE2EGolden(t *testing.T) e2eGolden {
	t.Helper()
	var g e2eGolden
	readJSONGolden(t, "e2e_golden.json", &g)
	return g
}

// maxAbsDiff32 reports the largest |a[i]-b[i]| over two equal-length float32 slices, failing the
// test immediately with FailNow if the lengths differ (a shape mismatch is a distinct bug from a
// value mismatch and should not be graded by tolerance) — mirrors whisper/golden_test.go's helper
// of the same name.
func maxAbsDiff32(t *testing.T, a, b []float32) float64 {
	t.Helper()
	if len(a) != len(b) {
		t.Fatalf("length mismatch: got %d, want %d", len(a), len(b))
	}
	var maxV float64
	for i := range a {
		d := float64(a[i]) - float64(b[i])
		if d < 0 {
			d = -d
		}
		if d > maxV {
			maxV = d
		}
	}
	return maxV
}
