// SPDX-Licence-Identifier: EUPL-1.2

package whisper

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// golden_test.go is the shared loader for every committed golden fixture under testdata/ — captured
// from the REAL transformers reference implementation (see each fixture's generating comment below) so
// mel_test.go/attention_test.go/encoder_test.go/decoder_test.go/live_test.go compare against one
// consistent set of structs. Test-only: stdlib os/encoding/json/path-filepath are fine here (the banned-
// import rule is for production library code — see e.g. arch/mamba2/smoke_test.go's "os" import).

func readTestdata(t *testing.T, name string) []byte {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatalf("read testdata/%s: %v", name, err)
	}
	return b
}

// writeFile writes content to dir/name — the small-fixture counterpart to readTestdata, used by _Bad/
// _Ugly tests that need a malformed/minimal on-disk file (e.g. an empty "{}" preprocessor_config.json).
func writeFile(t *testing.T, dir, name, content string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o600); err != nil {
		t.Fatalf("write %s/%s: %v", dir, name, err)
	}
}

func readJSONGolden(t *testing.T, name string, v any) {
	t.Helper()
	if err := json.Unmarshal(readTestdata(t, name), v); err != nil {
		t.Fatalf("parse testdata/%s: %v", name, err)
	}
}

// --- e2e_golden.json — captured by a hand-rolled Python greedy loop (encoder → language-detect →
// init prompt → suppress-list-aware greedy decode, the exact algorithm this package ports) that was
// itself verified to byte-for-byte match transformers' official WhisperForConditionalGeneration.generate
// output on the same WAV before capture — see docs/superpowers/specs/2026-07-19-whisper-asr-design.md's
// implementing lane for the capture script. ---

type e2eGolden struct {
	WAV                   string  `json:"wav"`
	DetectedLanguageToken string  `json:"detected_language_token"`
	DetectedLanguageID    int32   `json:"detected_language_id"`
	InitTokens            []int32 `json:"init_tokens"`
	FullIDs               []int32 `json:"full_ids"`
	ContentIDs            []int32 `json:"content_ids"`
	Text                  string  `json:"text"`
}

func (g e2eGolden) DetectedLanguageCode() string { return fromBracketedToken(g.DetectedLanguageToken) }

func readE2EGolden(t *testing.T) e2eGolden {
	t.Helper()
	var g e2eGolden
	readJSONGolden(t, "e2e_golden.json", &g)
	return g
}

// --- mel_golden.json — captured by calling transformers' WhisperFeatureExtractor._torch_extract_fbank_
// features directly on a short deterministic two-tone waveform (bypassing the 30 s pad-to-max-length
// wrapper, so the fixture stays small), using the checkpoint's REAL shipped Slaney mel_filters. ---

type melGolden struct {
	NFFT         int       `json:"n_fft"`
	HopLength    int       `json:"hop_length"`
	NSamples     int       `json:"n_samples"`
	FeatureSize  int       `json:"feature_size"`
	Wave         []float64 `json:"wave"`
	LogSpecShape []int     `json:"log_spec_shape"` // [batch=1, mel_bins, frames]
	LogSpec      []float64 `json:"log_spec"`       // flat, batch-major (batch=1, so mel-major)
}

func readMelGolden(t *testing.T) melGolden {
	t.Helper()
	var g melGolden
	readJSONGolden(t, "mel_golden.json", &g)
	return g
}

// Row returns log-mel bin m's frames (mel-major flat layout, batch=1).
func (g melGolden) Row(m int) []float64 {
	frames := g.LogSpecShape[2]
	return g.LogSpec[m*frames : (m+1)*frames]
}

// --- toy_block_goldens.json — captured against a TOY WhisperConfig (d_model 8, 2 heads, ffn 16, 4 mel
// bins) with hand-built deterministic weights (torch.Generator seeded per-tensor), run through the REAL
// transformers WhisperEncoderLayer/WhisperDecoderLayer/Conv1d/tied-lm-head modules directly — small
// enough to commit weights+activations whole, so this gate needs no external checkpoint (unlike
// live_test.go's real-weight E2E gate). ---

type linGolden struct {
	Weight []float32 `json:"weight"`
	Bias   []float32 `json:"bias"`
}

type lnGolden struct {
	Weight []float32 `json:"weight"`
	Bias   []float32 `json:"bias"`
}

func (g linGolden) linear(in, out int) LinearWeights {
	return LinearWeights{Weight: g.Weight, Bias: g.Bias, In: in, Out: out}
}

func (g lnGolden) layerNorm() LayerNormWeights {
	return LayerNormWeights{Weight: g.Weight, Bias: g.Bias}
}

type toyGeometry struct {
	DModel  int `json:"d_model"`
	Heads   int `json:"heads"`
	HeadDim int `json:"head_dim"`
	FFN     int `json:"ffn"`
	MelBins int `json:"mel_bins"`
}

type toyConvBlock struct {
	TIn           int       `json:"T_in"`
	Conv1Weight   []float32 `json:"conv1_weight"`
	Conv1Bias     []float32 `json:"conv1_bias"`
	Conv2Weight   []float32 `json:"conv2_weight"`
	Conv2Bias     []float32 `json:"conv2_bias"`
	Input         []float32 `json:"input"`
	Conv1Out      []float32 `json:"conv1_out"`
	Gelu1Out      []float32 `json:"gelu1_out"`
	Conv2OutShape []int     `json:"conv2_out_shape"`
	Conv2Out      []float32 `json:"conv2_out"`
	Gelu2Out      []float32 `json:"gelu2_out"`
}

type toyEncoderLayerWeights struct {
	SelfAttnLayerNorm lnGolden  `json:"self_attn_layer_norm"`
	QProj             linGolden `json:"q_proj"`
	KProj             linGolden `json:"k_proj"`
	VProj             linGolden `json:"v_proj"`
	OutProj           linGolden `json:"out_proj"`
	FinalLayerNorm    lnGolden  `json:"final_layer_norm"`
	FC1               linGolden `json:"fc1"`
	FC2               linGolden `json:"fc2"`
}

type toyEncoderLayer struct {
	T       int                    `json:"T"`
	Weights toyEncoderLayerWeights `json:"weights"`
	Input   []float32              `json:"input"`
	Output  []float32              `json:"output"`
}

type toyDecoderLayerWeights struct {
	SelfAttnLayerNorm    lnGolden  `json:"self_attn_layer_norm"`
	SelfQProj            linGolden `json:"self_q_proj"`
	SelfKProj            linGolden `json:"self_k_proj"`
	SelfVProj            linGolden `json:"self_v_proj"`
	SelfOutProj          linGolden `json:"self_out_proj"`
	EncoderAttnLayerNorm lnGolden  `json:"encoder_attn_layer_norm"`
	CrossQProj           linGolden `json:"cross_q_proj"`
	CrossKProj           linGolden `json:"cross_k_proj"`
	CrossVProj           linGolden `json:"cross_v_proj"`
	CrossOutProj         linGolden `json:"cross_out_proj"`
	FinalLayerNorm       lnGolden  `json:"final_layer_norm"`
	FC1                  linGolden `json:"fc1"`
	FC2                  linGolden `json:"fc2"`
}

type toyDecoderLayer struct {
	Td       int                    `json:"Td"`
	Tenc     int                    `json:"Tenc"`
	Weights  toyDecoderLayerWeights `json:"weights"`
	DecInput []float32              `json:"dec_input"`
	EncInput []float32              `json:"enc_input"`
	Output   []float32              `json:"output"`
}

type toyLMHead struct {
	Vocab           int       `json:"vocab"`
	Embed           []float32 `json:"embed"`
	LNWeight        []float32 `json:"ln_weight"`
	LNBias          []float32 `json:"ln_bias"`
	Hidden          []float32 `json:"hidden"`
	LayerNormOutput []float32 `json:"layer_norm_output"`
	Logits          []float32 `json:"logits"`
	Argmax          int       `json:"argmax"`
}

type toyBlockGoldens struct {
	Geometry     toyGeometry     `json:"geometry"`
	ConvBlock    toyConvBlock    `json:"conv_block"`
	EncoderLayer toyEncoderLayer `json:"encoder_layer"`
	DecoderLayer toyDecoderLayer `json:"decoder_layer"`
	LMHead       toyLMHead       `json:"lm_head"`
}

func readToyBlockGoldens(t *testing.T) toyBlockGoldens {
	t.Helper()
	var g toyBlockGoldens
	readJSONGolden(t, "toy_block_goldens.json", &g)
	return g
}

// maxAbsDiff32 reports the largest |a[i]-b[i]| over two equal-length float32 slices, failing the test
// immediately with FailNow if the lengths differ (a shape mismatch is a distinct bug from a value
// mismatch and should not be graded by tolerance).
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

func maxAbsDiff64v32(t *testing.T, a []float64, b []float32) float64 {
	t.Helper()
	if len(a) != len(b) {
		t.Fatalf("length mismatch: got %d, want %d", len(b), len(a))
	}
	var max float64
	for i := range a {
		d := a[i] - float64(b[i])
		if d < 0 {
			d = -d
		}
		if d > max {
			max = d
		}
	}
	return max
}
