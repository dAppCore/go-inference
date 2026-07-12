// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"encoding/base64"
	"encoding/json"
	"math"
	"os"
	"testing"
	"time"

	"dappco.re/go/inference/model/gemma4/audio"
)

type countingAudioGEMM struct {
	inner    audio.GEMM
	accepted int
	rejected int
}

func (gemm *countingAudioGEMM) MatMul(a, b []float32, m, k, n int, transposeB bool) ([]float32, bool) {
	out, ok := gemm.inner.MatMul(a, b, m, k, n, transposeB)
	if ok {
		gemm.accepted++
	} else {
		gemm.rejected++
	}
	return out, ok
}

// TestHIPAudioGEMM_BadMissingROCBlas proves a driver without the optional rocBLAS extension selects
// the host tower path. fakeHIPDriver is the portable fake used by the armed HIP driver suite.
func TestHIPAudioGEMM_BadMissingROCBlas(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	if gemm := newHIPAudioGEMM(driver); gemm != nil {
		t.Fatal("fake driver without rocBLAS returned a device GEMM; want host fallback")
	}
}

func TestHIPAudioGEMM_UglyUnavailableDriver(t *testing.T) {
	driver := &fakeHIPDriver{available: false}
	if gemm := newHIPAudioGEMM(driver); gemm != nil {
		t.Fatal("unavailable fake driver returned a device GEMM; want host fallback")
	}
}

func TestHIPAudioGEMMHardwareParity_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_HIP_AUDIO_GEMM_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_HIP_AUDIO_GEMM_TESTS=1 to run rocBLAS audio tower parity")
	}
	dir := os.Getenv("GO_ROCM_AUDIO_MODEL_PATH")
	if dir == "" {
		t.Fatal("GO_ROCM_AUDIO_MODEL_PATH is required")
	}
	tower, err := LoadAudioTower(dir)
	if err != nil {
		t.Fatalf("LoadAudioTower: %v", err)
	}
	if tower == nil || tower.gemm == nil {
		t.Fatal("rocBLAS audio GEMM is unavailable")
	}
	defer tower.Close()
	counted := &countingAudioGEMM{inner: tower.gemm}
	raw, err := os.ReadFile("../../model/gemma4/audio/testdata/audio_module_golden.json")
	if err != nil {
		t.Fatalf("read module golden: %v", err)
	}
	var golden struct {
		Frames    int    `json:"frames"`
		MelBins   int    `json:"mel_bins"`
		MelBF16LE string `json:"mel_bf16le_b64"`
	}
	if err := json.Unmarshal(raw, &golden); err != nil {
		t.Fatalf("unmarshal module golden: %v", err)
	}
	mel, err := base64.StdEncoding.DecodeString(golden.MelBF16LE)
	if err != nil {
		t.Fatalf("decode module golden mel: %v", err)
	}

	host, err := audio.Encode(mel, golden.Frames, golden.MelBins, tower.loaded, nil)
	if err != nil {
		t.Fatalf("host Encode: %v", err)
	}
	device, err := audio.EncodeWithGEMM(mel, golden.Frames, golden.MelBins, tower.loaded, nil, counted)
	if err != nil {
		t.Fatalf("device EncodeWithGEMM: %v", err)
	}
	const timingRuns = 3
	var hostElapsed, deviceElapsed time.Duration
	for range timingRuns {
		started := time.Now()
		host, err = audio.Encode(mel, golden.Frames, golden.MelBins, tower.loaded, nil)
		hostElapsed += time.Since(started)
		if err != nil {
			t.Fatalf("timed host Encode: %v", err)
		}
	}
	for range timingRuns {
		started := time.Now()
		device, err = audio.EncodeWithGEMM(mel, golden.Frames, golden.MelBins, tower.loaded, nil, counted)
		deviceElapsed += time.Since(started)
		if err != nil {
			t.Fatalf("timed device EncodeWithGEMM: %v", err)
		}
	}
	if len(device) != len(host) {
		t.Fatalf("device output len=%d, want host len=%d", len(device), len(host))
	}
	var dot, hostNorm, deviceNorm float64
	var maxDelta float32
	for i := range host {
		delta := host[i] - device[i]
		if delta < 0 {
			delta = -delta
		}
		if delta > maxDelta {
			maxDelta = delta
		}
		h, d := float64(host[i]), float64(device[i])
		dot += h * d
		hostNorm += h * h
		deviceNorm += d * d
	}
	cosine := dot / math.Sqrt(hostNorm*deviceNorm)
	t.Logf("audio tower A/B (frames=%d, mean of %d warm runs)", golden.Frames, timingRuns)
	t.Log("path\tmean")
	t.Logf("host\t%s", hostElapsed/timingRuns)
	t.Logf("rocBLAS\t%s", deviceElapsed/timingRuns)
	t.Logf("rocBLAS vs host: cosine=%.9f max|delta|=%g n=%d", cosine, maxDelta, len(host))
	t.Logf("rocBLAS dispatch: accepted=%d rejected=%d", counted.accepted, counted.rejected)
	if counted.accepted == 0 || counted.rejected != 0 {
		t.Fatalf("rocBLAS dispatch accepted=%d rejected=%d, want all tower GEMMs on device", counted.accepted, counted.rejected)
	}
	if cosine < 0.999 || maxDelta > 2 {
		t.Fatalf("rocBLAS parity outside golden tolerance: cosine=%.9f max|delta|=%g", cosine, maxDelta)
	}
}
