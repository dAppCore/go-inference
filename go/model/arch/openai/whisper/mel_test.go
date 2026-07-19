// SPDX-Licence-Identifier: EUPL-1.2

package whisper

import (
	"math"
	"testing"
)

// TestLoadFeatureConfig_Good parses the REAL preprocessor_config.json shipped with openai/whisper-tiny
// (the 80×201 Slaney mel filter matrix included), pinning the stock geometry every published Whisper
// checkpoint carries.
func TestLoadFeatureConfig_Good(t *testing.T) {
	cfg, err := LoadFeatureConfig("testdata")
	if err != nil {
		t.Fatalf("LoadFeatureConfig: %v", err)
	}
	if cfg.NFFT != 400 || cfg.HopLength != 160 || cfg.NSamples != 480000 || cfg.SamplingRate != 16000 || cfg.NumMelBins != 80 {
		t.Fatalf("geometry = %+v, want n_fft 400/hop 160/n_samples 480000/rate 16000/mel_bins 80", *cfg)
	}
	if len(cfg.MelFilters) != 80 || len(cfg.MelFilters[0]) != 201 {
		t.Fatalf("mel_filters shape = %d×%d, want 80×201", len(cfg.MelFilters), len(cfg.MelFilters[0]))
	}
}

func TestLoadFeatureConfig_Bad(t *testing.T) {
	if _, err := LoadFeatureConfig(t.TempDir()); err == nil {
		t.Fatal("LoadFeatureConfig accepted a directory with no preprocessor_config.json")
	}
}

// TestLoadFeatureConfig_Ugly proves a syntactically valid but geometry-empty preprocessor_config.json is
// refused (not silently zero-valued) — distinct from _Bad's missing-file case.
func TestLoadFeatureConfig_Ugly(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "preprocessor_config.json", `{}`)
	if _, err := LoadFeatureConfig(dir); err == nil {
		t.Fatal("LoadFeatureConfig accepted an empty document")
	}
}

// TestExtractLogMel_Good replays mel_golden.json — a two-tone 2000-sample synthetic waveform run through
// the REAL transformers _torch_extract_fbank_features (bypassing its 30 s pad wrapper, so the fixture
// stays small) using the checkpoint's real shipped Slaney filters — banded at the tolerance the
// reference's own docstring claims between its torch/numpy paths (1e-5), loosened slightly for the f32
// narrowing this package's host-f32 convention applies at the end.
func TestExtractLogMel_Good(t *testing.T) {
	golden := readMelGolden(t)
	fc, err := LoadFeatureConfig("testdata")
	if err != nil {
		t.Fatalf("LoadFeatureConfig: %v", err)
	}
	samples := make([]float32, len(golden.Wave))
	for i, v := range golden.Wave {
		samples[i] = float32(v)
	}
	got, err := fc.ExtractLogMel(samples)
	if err != nil {
		t.Fatalf("ExtractLogMel: %v", err)
	}
	frames := golden.LogSpecShape[2]
	if len(got) != golden.LogSpecShape[1] || len(got[0]) != frames {
		t.Fatalf("ExtractLogMel shape = %d×%d, want %d×%d", len(got), len(got[0]), golden.LogSpecShape[1], frames)
	}
	var worst float64
	for m := range got {
		if d := maxAbsDiff64v32(t, golden.Row(m), got[m]); d > worst {
			worst = d
		}
	}
	if worst > 1e-3 {
		t.Fatalf("ExtractLogMel max abs diff vs reference = %g, want <= 1e-3", worst)
	}
}

func TestExtractLogMel_Bad(t *testing.T) {
	fc, err := LoadFeatureConfig("testdata")
	if err != nil {
		t.Fatalf("LoadFeatureConfig: %v", err)
	}
	if _, err := fc.ExtractLogMel(make([]float32, 10)); err == nil {
		t.Fatal("ExtractLogMel accepted fewer samples than one NFFT frame")
	}
}

func TestExtractLogMel_Ugly(t *testing.T) {
	if _, err := (*FeatureConfig)(nil).ExtractLogMel(make([]float32, 1000)); err == nil {
		t.Fatal("ExtractLogMel accepted a nil receiver")
	}
}

// TestHannWindow_Good pins the periodic convention (divide by n, not n-1): w[0]=0, and w is symmetric
// but never returns to exactly 1 (the sample AT index n would, but that sample is never taken).
func TestHannWindow_Good(t *testing.T) {
	w := hannWindow(8)
	if len(w) != 8 {
		t.Fatalf("len = %d, want 8", len(w))
	}
	if w[0] != 0 {
		t.Fatalf("w[0] = %g, want 0 (periodic Hann starts at zero)", w[0])
	}
	if d := w[1] - w[7]; d < -1e-12 || d > 1e-12 {
		t.Fatalf("w[1]=%g and w[7]=%g should be symmetric about the centre", w[1], w[7])
	}
}

func TestReflectPad_Good(t *testing.T) {
	x := []float64{1, 2, 3, 4, 5}
	out := reflectPad(x, 2)
	want := []float64{3, 2, 1, 2, 3, 4, 5, 4, 3}
	if len(out) != len(want) {
		t.Fatalf("len = %d, want %d", len(out), len(want))
	}
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("reflectPad(%v,2) = %v, want %v", x, out, want)
		}
	}
}

// TestDFTPower_Good proves the direct DFT (Whisper's n_fft=400 is not a power of two, so this package
// cannot reuse a radix-2 FFT — see dftTwiddles' doc comment) matches a known closed-form spectrum: a
// single-bin cosine at exactly bin k=2 of an n=8 DFT has all its energy in bin 2 (and its mirror bin 6,
// outside this function's [0,n/2] output range).
func TestDFTPower_Good(t *testing.T) {
	n, bins := 8, 5
	seg := make([]float64, n)
	for t := range seg {
		seg[t] = cosAt(2, n, t)
	}
	cosT, sinT := dftTwiddles(n, bins)
	power := make([]float64, bins)
	dftPower(seg, cosT, sinT, bins, power)
	for k, p := range power {
		if k == 2 {
			if p < 1 {
				t.Fatalf("power[2] = %g, want the dominant bin (large)", p)
			}
			continue
		}
		if p > power[2]*0.01 {
			t.Fatalf("power[%d] = %g leaked energy from the pure bin-2 tone (power[2]=%g)", k, p, power[2])
		}
	}
}

func cosAt(k, n, t int) float64 {
	return math.Cos(2 * math.Pi * float64(k) * float64(t) / float64(n))
}
