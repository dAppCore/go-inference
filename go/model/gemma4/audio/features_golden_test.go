// SPDX-Licence-Identifier: EUPL-1.2

package audio

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"testing"
)

// features_golden_test.go pins the host mel front-end (Extract) to HF's Gemma4AudioFeatureExtractor
// goldens (transformers 5.5.4, e2b-it-4bit processor_config). The fixture (testdata/audio_mel_golden.json,
// the engine-neutral fixture shared with the engine/metal native port) carries the feature_extractor
// config plus tiny synthetic waveforms and their HF input_features as base64 float32 little-endian, so
// the fixture is exact and small. Pure host DSP → no GPU, no checkpoint: the portable parity gate.

type melGoldenCase struct {
	Name        string `json:"name"`
	NumFrames   int    `json:"num_frames"`
	FeatureSize int    `json:"feature_size"`
	WaveformB64 string `json:"waveform_f32le_b64"`
	FeaturesB64 string `json:"features_f32le_b64"`
	Mask        []bool `json:"mask"`
}

type melGolden struct {
	FeatureExtractor FeatureConfig   `json:"feature_extractor"`
	Cases            []melGoldenCase `json:"cases"`
}

func decodeF32LE(t *testing.T, s string) []float32 {
	t.Helper()
	raw, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		t.Fatalf("base64 decode: %v", err)
	}
	if len(raw)%4 != 0 {
		t.Fatalf("f32le byte length %d not a multiple of 4", len(raw))
	}
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out
}

// TestFeatures_Extract_Golden pins Extract to the HF mel goldens at ~1-ULP parity (the go-mlx bar was
// max|Δ| 4.77e-7); frame counts and the per-frame validity mask must match exactly. Highest-ROI parity
// guard, needs no GPU.
func TestFeatures_Extract_Golden(t *testing.T) {
	raw, err := os.ReadFile("testdata/audio_mel_golden.json")
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g melGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("unmarshal golden: %v", err)
	}
	if len(g.Cases) == 0 {
		t.Fatal("golden has no cases")
	}

	cfg := g.FeatureExtractor
	extractor, err := NewFeatureExtractor(&cfg)
	if err != nil {
		t.Fatalf("NewFeatureExtractor: %v", err)
	}

	const tol = 2e-6 // HF↔host float32 mel parity; go-mlx bar was 4.77e-7
	for _, c := range g.Cases {
		wave := decodeF32LE(t, c.WaveformB64)
		want := decodeF32LE(t, c.FeaturesB64)
		feats, mask, frames, err := extractor.Extract(wave)
		if err != nil {
			t.Fatalf("%s Extract: %v", c.Name, err)
		}
		if frames != c.NumFrames {
			t.Fatalf("%s frames=%d want %d", c.Name, frames, c.NumFrames)
		}
		if len(feats) != len(want) {
			t.Fatalf("%s feats len=%d want %d", c.Name, len(feats), len(want))
		}
		if len(mask) != len(c.Mask) {
			t.Fatalf("%s mask len=%d want %d", c.Name, len(mask), len(c.Mask))
		}
		for i := range mask {
			if mask[i] != c.Mask[i] {
				t.Fatalf("%s mask[%d]=%v want %v", c.Name, i, mask[i], c.Mask[i])
			}
		}
		var maxAbs float64
		for i := range feats {
			if d := math.Abs(float64(feats[i] - want[i])); d > maxAbs {
				maxAbs = d
			}
		}
		if maxAbs > tol {
			t.Fatalf("%s max|Δ|=%.3e exceeds tol %.0e", c.Name, maxAbs, tol)
		}
		t.Logf("%s: frames=%d valid=%d max|Δ|=%.3e", c.Name, frames, countTrue(c.Mask), maxAbs)
	}
}

func countTrue(b []bool) int {
	n := 0
	for _, v := range b {
		if v {
			n++
		}
	}
	return n
}
