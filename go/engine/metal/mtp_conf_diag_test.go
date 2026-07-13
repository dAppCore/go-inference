// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func mtpConfTestRow(logits []float32) []byte {
	row := make([]byte, len(logits)*bf16Size)
	for i, v := range logits {
		h := f32ToBF16(v)
		row[i*bf16Size] = byte(h)
		row[i*bf16Size+1] = byte(h >> 8)
	}
	return row
}

func TestMTPConfEnabled_ReflectsCapturePath(t *testing.T) {
	saved := mtpConfCapturePath
	defer func() { mtpConfCapturePath = saved }()
	mtpConfCapturePath = ""
	if mtpConfEnabled() {
		t.Fatal("mtpConfEnabled true with an empty capture path")
	}
	mtpConfCapturePath = "capture.jsonl"
	if !mtpConfEnabled() {
		t.Fatal("mtpConfEnabled false with a configured capture path")
	}
}

// TestMTPConfProb pins the softmax probability against a hand-computed
// distribution on a 4-token row (all values exactly representable in bf16).
func TestMTPConfProb(t *testing.T) {
	row := mtpConfTestRow([]float32{2, 1, 0, -1})
	var sum float64
	for _, v := range []float64{2, 1, 0, -1} {
		sum += math.Exp(v - 2)
	}
	want := float32(math.Exp(0) / sum)
	got := mtpConfProb(row, 0, nil)
	if math.Abs(float64(got-want)) > 1e-6 {
		t.Fatalf("prob(top) = %v, want %v", got, want)
	}
	wantLow := float32(math.Exp(-3) / sum)
	if gotLow := mtpConfProb(row, 3, nil); math.Abs(float64(gotLow-wantLow)) > 1e-6 {
		t.Fatalf("prob(bottom) = %v, want %v", gotLow, wantLow)
	}
	if bad := mtpConfProb(row, 4, nil); bad != 0 {
		t.Fatalf("prob(out of range) = %v, want 0", bad)
	}
}

// TestMTPConfProbSuppressed pins that suppressed ids are excluded from the
// distribution, mirroring draftGreedyTokenWithSuppress: with the top token
// suppressed, the runner-up's probability is over the remaining support.
func TestMTPConfProbSuppressed(t *testing.T) {
	row := mtpConfTestRow([]float32{2, 1, 0, -1})
	var sum float64
	for _, v := range []float64{1, 0, -1} {
		sum += math.Exp(v - 1)
	}
	want := float32(math.Exp(0) / sum)
	got := mtpConfProb(row, 1, []int32{0})
	if math.Abs(float64(got-want)) > 1e-6 {
		t.Fatalf("prob(runner-up, top suppressed) = %v, want %v", got, want)
	}
}

// TestMTPConfRecordCycle pins the JSONL line format byte-exactly and that the
// sink appends across calls.
func TestMTPConfRecordCycle(t *testing.T) {
	path := filepath.Join(t.TempDir(), "conf.jsonl")
	savedPath := mtpConfCapturePath
	mtpConfCapturePath = path
	mtpConfOut.mu.Lock()
	savedF, savedDead := mtpConfOut.f, mtpConfOut.dead
	mtpConfOut.f, mtpConfOut.dead = nil, false
	mtpConfOut.mu.Unlock()
	defer func() {
		mtpConfOut.mu.Lock()
		if mtpConfOut.f != nil {
			mtpConfOut.f.Close()
		}
		mtpConfOut.f, mtpConfOut.dead = savedF, savedDead
		mtpConfOut.mu.Unlock()
		mtpConfCapturePath = savedPath
	}()

	mtpConfRecordCycle(100, 1, []float32{0.5, 0.25}, 2)
	mtpConfRecordCycle(104, 0, []float32{0.9990}, 0)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read capture: %v", err)
	}
	want := `{"pos":100,"carry":1,"probs":[0.5000,0.2500],"accepted":2}` + "\n" +
		`{"pos":104,"carry":0,"probs":[0.9990],"accepted":0}` + "\n"
	if string(data) != want {
		t.Fatalf("capture = %q, want %q", data, want)
	}
}
