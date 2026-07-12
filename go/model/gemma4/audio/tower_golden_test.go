// SPDX-Licence-Identifier: EUPL-1.2

package audio

import (
	"encoding/base64"
	"encoding/json"
	"math"
	"os"
	"testing"

	"dappco.re/go/inference/model"
	_ "dappco.re/go/inference/model/gemma4" // register the gemma4 ArchSpec so model.Load assembles the audio tower
)

// tower_golden_test.go pins the host Conformer tower to HF Gemma4AudioModel goldens computed on the REAL
// e2b-4bit BF16 tower (weights loaded from the local HF cache). It skips when the checkpoint is absent
// (supervised parity, not a portable CI gate) and loads the tower engine-neutrally via model.Load, so it
// runs the same on darwin and on the AMD box. The synthetic mel is shared byte-for-byte via the fixture's
// bf16 bytes, so only tower-math rounding remains. The two defect-catchers are first: the subsample
// output collapses if the OHWI conv is double-transposed, and the tower output is off by the [1536] bias
// (max|abs| 14.875) if output_proj.bias is dropped.

type moduleGolden struct {
	Frames       int    `json:"frames"`
	MelBins      int    `json:"mel_bins"`
	SoftTokens   int    `json:"soft_tokens"`
	OutputDim    int    `json:"output_dim"`
	Hidden       int    `json:"hidden"`
	MelBF16B64   string `json:"mel_bf16le_b64"`
	SubsampleB64 string `json:"subsample_f32le_b64"`
	Layer0B64    string `json:"layer0_f32le_b64"`
	TowerB64     string `json:"tower_f32le_b64"`
}

func e2b4bitSnapshotDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	base := home + "/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots"
	entries, err := os.ReadDir(base)
	if err != nil {
		return ""
	}
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		dir := base + "/" + e.Name()
		if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
			continue
		}
		if _, err := os.Stat(dir + "/config.json"); err == nil {
			return dir
		}
	}
	return ""
}

func cosineSim(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

func maxAbsDelta(a, b []float32) float64 {
	var m float64
	for i := range a {
		if d := math.Abs(float64(a[i] - b[i])); d > m {
			m = d
		}
	}
	return m
}

// TestTower_Encode_ModuleGoldens runs the host tower on the shared synthetic mel and pins its per-module
// outputs to the HF goldens at cosine >= 0.999. Skips without the local checkpoint.
func TestTower_Encode_ModuleGoldens(t *testing.T) {
	dir := e2b4bitSnapshotDir()
	if dir == "" {
		t.Skip("e2b-it-4bit checkpoint not in HF cache — supervised parity test")
	}
	raw, err := os.ReadFile("testdata/audio_module_golden.json")
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g moduleGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("unmarshal golden: %v", err)
	}
	mel, err := base64.StdEncoding.DecodeString(g.MelBF16B64)
	if err != nil {
		t.Fatalf("mel b64: %v", err)
	}
	if len(mel) != g.Frames*g.MelBins*2 {
		t.Fatalf("mel bytes %d want %d", len(mel), g.Frames*g.MelBins*2)
	}

	lm, mapping, err := model.Load(dir)
	if err != nil {
		t.Fatalf("model.Load(%s): %v", dir, err)
	}
	// The weight byte-views reference the mapping's mmap — keep it open until the tower has run.
	if mapping != nil {
		defer func() { _ = mapping.Close() }()
	}
	if lm.Audio == nil {
		t.Fatal("model has no Conformer audio payload")
	}
	la := lm.Audio

	// Defect-catcher 1 — real conv layout: subsample output collapses on a double-transposed conv.
	sub, _, err := Subsample(mel, g.Frames, g.MelBins, la, nil)
	if err != nil {
		t.Fatalf("Subsample: %v", err)
	}
	assertCosineGE(t, "subsample", sub, decodeF32LE(t, g.SubsampleB64), 0.999)

	// Layer-0 Conformer block vs HF.
	lay0, err := Layer(sub, &la.Layers[0], la.Cfg, nil)
	if err != nil {
		t.Fatalf("Layer(0): %v", err)
	}
	assertCosineGE(t, "layer0", lay0, decodeF32LE(t, g.Layer0B64), 0.999)

	// Defect-catcher 2 — output_proj.bias: full tower last_hidden_state carries the [1536] bias.
	tower, err := Encode(mel, g.Frames, g.MelBins, la, nil)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	towerWant := decodeF32LE(t, g.TowerB64)
	assertCosineGE(t, "tower", tower, towerWant, 0.999)
	// A dropped output_proj.bias would blow max|Δ| to ~14.875 (the bias max|abs|) and cosine to ~0.9837;
	// 2.0 leaves ample headroom over the observed host-GEMM rounding.
	if d := maxAbsDelta(tower, towerWant); d > 2.0 {
		t.Fatalf("tower max|Δ|=%.4f > 2.0 — output_proj.bias likely dropped", d)
	}
}

func assertCosineGE(t *testing.T, name string, got, want []float32, minCos float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s len=%d want %d", name, len(got), len(want))
	}
	c := cosineSim(got, want)
	d := maxAbsDelta(got, want)
	t.Logf("%s: cosine=%.6f max|Δ|=%.4f (n=%d)", name, c, d, len(got))
	if c < minCos {
		t.Fatalf("%s cosine=%.6f < %.3f (max|Δ|=%.4f)", name, c, minCos, d)
	}
}
