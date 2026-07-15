// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"encoding/binary"
	"math"
	"os"
	"testing"

	enginegemma4 "dappco.re/go/inference/engine/hip/model/gemma4"
	"dappco.re/go/inference/model"
	_ "dappco.re/go/inference/model/gemma4" // register the gemma4 ArchSpec so model.Load assembles the tower
	"dappco.re/go/inference/model/quant/mlxaffine"
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

func TestHipAudioProjectorF32_Good(t *testing.T) {
	projector := audioQ4GoldenProjector()
	features := []float32{1, -2, 3, -4, 5, -6, 7, -8, -1, 2, -3, 4, -5, 6, -7, 8}
	weights, err := hipLoadAudioProjectorQ4(projector)
	if err != nil {
		t.Fatalf("hipLoadAudioProjectorQ4: %v", err)
	}
	got, err := hipAudioProjectorF32(features, 2, projector, weights, 1e-6)
	if err != nil {
		t.Fatalf("hipAudioProjectorF32: %v", err)
	}
	want := []float32{-1.584236, 3.5645308, -1.5594823, 1.584236, -3.5645308, 1.5594823}
	for index := range want {
		if delta := math.Abs(float64(got[index] - want[index])); delta > 1e-6 {
			t.Fatalf("embedding[%d]=%.9f, want %.9f", index, got[index], want[index])
		}
	}
}

func TestAudioTower_ProjectEmbeddings_Bad(t *testing.T) {
	if _, _, err := (*AudioTower)(nil).ProjectEmbeddings([]float32{1}); err == nil {
		t.Fatal("ProjectEmbeddings on a nil tower must error")
	}
}

func TestAudioTower_ProjectEmbeddings_Good(t *testing.T) {
	dir := e2b4bitSnapshotDir()
	if dir == "" {
		t.Skip("e2b-it-4bit checkpoint not in HF cache — supervised integration test")
	}
	tower, err := LoadAudioTower(dir)
	if err != nil {
		t.Fatalf("LoadAudioTower(%s): %v", dir, err)
	}
	defer func() { _ = tower.Close() }()
	embeddings, softTokens, err := tower.ProjectEmbeddings(syntheticWaveform(8000))
	if err != nil {
		t.Fatalf("ProjectEmbeddings: %v", err)
	}
	if softTokens <= 0 || len(embeddings)%softTokens != 0 {
		t.Fatalf("ProjectEmbeddings geometry: embeddings=%d softTokens=%d", len(embeddings), softTokens)
	}
}

func TestAudioTower_ProjectEmbeddings_Ugly(t *testing.T) {
	if _, _, err := (&AudioTower{}).ProjectEmbeddings(nil); err == nil {
		t.Fatal("ProjectEmbeddings on an empty tower must error")
	}
}

func TestHipLoadAudioProjectorQ4_Ugly(t *testing.T) {
	projector := audioQ4GoldenProjector()
	projector.Biases = nil
	if _, err := hipLoadAudioProjectorQ4(projector); err == nil {
		t.Fatal("projector without affine biases must error")
	}
}

func audioQ4GoldenProjector() model.LoadedAudioLinear {
	const inDim, outDim, groupSize, bits = 8, 3, 8, 4
	packed := make([]byte, outDim*4)
	binary.LittleEndian.PutUint32(packed[0:], 0x76543210)
	binary.LittleEndian.PutUint32(packed[4:], 0xfedcba98)
	binary.LittleEndian.PutUint32(packed[8:], 0x13579bdf)
	return model.LoadedAudioLinear{
		Weight: packed, Scales: []byte{0x00, 0x3e, 0x80, 0xbe, 0x00, 0x3d},
		Biases: []byte{0x80, 0x3f, 0x00, 0xbf, 0x00, 0x40}, OutDim: outDim, InDim: inDim,
		GroupSize: groupSize, Bits: bits, Kind: mlxaffine.Mode,
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
