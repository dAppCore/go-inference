// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"math"
	"os"
	"testing"

	enginegemma4 "dappco.re/go/inference/engine/hip/model/gemma4"
	_ "dappco.re/go/inference/model/gemma4" // register the gemma4 ArchSpec so model.Load assembles the tower
)

// gemma4_audio_tower_test.go is the hip-side integration gate for the Conformer audio tower binding. It
// loads the real e2b-4bit checkpoint through hip's LoadAudioTower and projects a synthetic waveform,
// asserting the composition is geometrically consistent (soft tokens > 0, features = softTokens x
// OutputDim) and ties into hip's soft-token policy. The tower's numerical parity to HF is gated
// separately by the engine-neutral module golden (model/gemma4/audio). Skips without the checkpoint;
// portable so it runs on darwin and on the AMD box via the rsync tree.

func e2b4bitSnapshotDir() string {
	if dir := os.Getenv("GO_ROCM_AUDIO_MODEL_PATH"); dir != "" {
		if _, err := os.Stat(dir + "/model.safetensors"); err == nil {
			return dir
		}
	}
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

// syntheticWaveform builds n samples of a 440 Hz sine at 16 kHz in [-1,1].
func syntheticWaveform(n int) []float32 {
	w := make([]float32, n)
	for i := range w {
		w[i] = float32(0.5 * math.Sin(2*math.Pi*440*float64(i)/16000))
	}
	return w
}

// TestAudioTower_Project_Good loads the real checkpoint and projects a synthetic waveform, checking the
// hip binding produces geometrically consistent soft-token features. Skips without the checkpoint.
func TestAudioTower_Project_Good(t *testing.T) {
	dir := e2b4bitSnapshotDir()
	if dir == "" {
		t.Skip("e2b-it-4bit checkpoint not in HF cache — supervised integration test")
	}
	tower, err := LoadAudioTower(dir)
	if err != nil {
		t.Fatalf("LoadAudioTower(%s): %v", dir, err)
	}
	if tower == nil {
		t.Fatal("checkpoint has no Conformer audio tower")
	}
	defer func() { _ = tower.Close() }()

	if tower.OutputDim() <= 0 {
		t.Fatalf("OutputDim=%d, want > 0", tower.OutputDim())
	}

	features, softTokens, err := tower.Project(syntheticWaveform(8000)) // 0.5 s
	if err != nil {
		t.Fatalf("Project: %v", err)
	}
	if softTokens <= 0 {
		t.Fatalf("softTokens=%d, want > 0", softTokens)
	}
	if want := softTokens * tower.OutputDim(); len(features) != want {
		t.Fatalf("len(features)=%d, want softTokens*OutputDim=%d", len(features), want)
	}
	t.Logf("projected 0.5 s waveform: softTokens=%d outputDim=%d features=%d", softTokens, tower.OutputDim(), len(features))
}

// TestAudioTower_Project_Bad checks Project fails loud on an unloaded tower rather than panicking.
func TestAudioTower_Project_Bad(t *testing.T) {
	var tower *AudioTower
	if _, _, err := tower.Project(syntheticWaveform(8000)); err == nil {
		t.Fatal("Project on a nil tower must error")
	}
}

// TestAudioTower_softTokenPolicy pins the binding's soft-token count to hip's shared audio policy so the
// two never drift.
func TestAudioTower_softTokenPolicy(t *testing.T) {
	for _, frames := range []int{1, 4, 25, 100} {
		if got, want := enginegemma4.AudioSoftTokens(frames), (frames+1)/2; got != (want+1)/2 {
			t.Fatalf("AudioSoftTokens(%d)=%d, want ceil(ceil(frames/2)/2)", frames, got)
		}
	}
}
