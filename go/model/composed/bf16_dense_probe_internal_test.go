// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"math"
	"os"
	"testing"

	"dappco.re/go/inference/model"
)

// TestBF16DenseBatchPredict (#26 bisect instrument): full-stack batch forward + head over a real
// bf16-resident checkpoint; the known-good answer for Qwen3.5-2B on this prompt is id 11751
// (' Paris' — mlx-lm argmax parity, proven before the bf16 migration).
func TestBF16DenseBatchPredict(t *testing.T) {
	dir := os.Getenv("LTHN_COMPOSED_DENSE_PROBE_MODEL")
	if dir == "" {
		t.Skip("LTHN_COMPOSED_DENSE_PROBE_MODEL not set")
	}
	tm, ok, err := model.LoadComposedDir(dir)
	if err != nil || !ok {
		t.Fatalf("LoadComposedDir: ok=%v err=%v", ok, err)
	}
	ctm := tm.(*ComposedTokenModel)
	t.Logf("BF16Resident=%v EmbedB=%v OutputB=%v", ctm.m.BF16Resident, ctm.m.EmbedB != nil, ctm.m.OutputB != nil)
	er := make([]float32, ctm.m.D)
	if err := ctm.m.embedRow(er, 760); err != nil {
		t.Fatalf("embedRow: %v", err)
	}
	t.Logf("embedRow(760)[:4] = %v (want [-0.0036621, 0.0057983, -0.0324707, 0.0009766])", er[:4])
	ids := []int32{760, 6511, 314, 9338, 369}
	embs := make([][]byte, len(ids))
	for i, id := range ids {
		if embs[i], err = ctm.Embed(id); err != nil {
			t.Fatalf("Embed: %v", err)
		}
	}
	hid, derr := ctm.DecodeForward(embs)
	if derr != nil {
		t.Fatalf("DecodeForward: %v", derr)
	}
	logits, herr := ctm.Head(hid[len(hid)-1])
	if herr != nil {
		t.Fatalf("Head: %v", herr)
	}
	best, bestV := 0, float32(-1e30)
	for i := 0; i+2 <= len(logits); i += 2 {
		v := math.Float32frombits(uint32(uint16(logits[i])|uint16(logits[i+1])<<8) << 16)
		if v > bestV {
			bestV, best = v, i/2
		}
	}
	t.Logf("BATCH argmax = %d (want 11751)", best)
	if best != 11751 {
		t.Fatalf("batch argmax %d != 11751 — the bf16-resident lib path diverged", best)
	}
}

// TestBF16DenseStageFingerprint prints the layer-0 mixer output and truncated-stack hiddens for
// comparison against the banked mlx-lm references (#26 bisect).
func TestBF16DenseStageFingerprint(t *testing.T) {
	dir := os.Getenv("LTHN_COMPOSED_DENSE_PROBE_MODEL")
	if dir == "" {
		t.Skip("LTHN_COMPOSED_DENSE_PROBE_MODEL not set")
	}
	tm, ok, err := model.LoadComposedDir(dir)
	if err != nil || !ok {
		t.Fatalf("LoadComposedDir: ok=%v err=%v", ok, err)
	}
	m := tm.(*ComposedTokenModel).Model()
	ids := []int32{760, 6511, 314, 9338, 369}
	h := make([]float32, len(ids)*m.D)
	for i, id := range ids {
		if err := m.embedRow(h[i*m.D:(i+1)*m.D], int(id)); err != nil {
			t.Fatalf("embedRow: %v", err)
		}
	}
	normed := rmsNormRowsPlain(h, m.Layers[0].InputNorm, len(ids), m.D, m.Eps)
	gm := m.Layers[0].Mixer.(*gatedDeltaMixer)
	t.Logf("gd weights: B=%v Q=%v f32=%v", gm.w.InProjQKVB != nil, gm.w.InProjQKVQ != nil, gm.w.InProjQKV != nil)
	mixOut, _, merr := gm.Forward(normed, len(ids), m.D, nil)
	if merr != nil {
		t.Fatalf("mixer Forward: %v", merr)
	}
	t.Logf("L0 MIXER out[t=0][:6] = %v (mlx: [0.0304, -0.0137, 0.0057, -0.0339, -0.0240, 0.0109])", mixOut[:6])
	for _, keep := range []int{1, 4} {
		mm := *m
		mm.Layers = m.Layers[:keep]
		sess := NewSession(&mm)
		hid, ferr := sess.forwardEmb(append([]float32(nil), h...), len(ids))
		if ferr != nil {
			t.Fatalf("forwardEmb(%d layers): %v", keep, ferr)
		}
		t.Logf("layers=%d hidden[t=0][:6] = %v", keep, hid[:6])
	}
	t.Log("mlx refs: layers=1 [0.0791, 0.0127, -0.0123, -0.0693, -0.0698, 0.0052]; layers=4 [0.0117, 0.0244, 0.0564, -0.0830, 0.0205, -0.0557]")
}

// TestBF16MLPvsWidened runs layer-0's MLP in its bf16-resident form vs a widened-f32 clone on the
// same input — the branch-isolating bisect (#26).
func TestBF16MLPvsWidened(t *testing.T) {
	dir := os.Getenv("LTHN_COMPOSED_DENSE_PROBE_MODEL")
	if dir == "" {
		t.Skip("LTHN_COMPOSED_DENSE_PROBE_MODEL not set")
	}
	tm, ok, err := model.LoadComposedDir(dir)
	if err != nil || !ok {
		t.Fatalf("LoadComposedDir: ok=%v err=%v", ok, err)
	}
	m := tm.(*ComposedTokenModel).Model()
	mlp := m.Layers[0].MLP.(*MLP)
	t.Logf("MLP: B=%v FF=%d D=%d", mlp.GateB != nil, mlp.FF, m.D)
	if mlp.FF <= 0 {
		t.Fatal("MLP.FF must derive from the bf16 form (the FF=0 regression this probe caught)")
	}
	widen := func(bw *model.BF16Weight) []float32 {
		out := make([]float32, bw.OutDim*bw.InDim)
		for i := range out {
			out[i] = math.Float32frombits(uint32(uint16(bw.Data[2*i])|uint16(bw.Data[2*i+1])<<8) << 16)
		}
		return out
	}
	ref := &MLP{Gate: widen(mlp.GateB), Up: widen(mlp.UpB), Down: widen(mlp.DownB), FF: mlp.FF}
	x := make([]float32, m.D)
	for i := range x {
		x[i] = float32(i%7)*0.01 - 0.03
	}
	got := mlp.forward(append([]float32(nil), x...), 1, m.D)
	want := ref.forward(append([]float32(nil), x...), 1, m.D)
	var worst float64
	wi := 0
	for i := range want {
		if d := math.Abs(float64(got[i]) - float64(want[i])); d > worst {
			worst, wi = d, i
		}
	}
	t.Logf("MLP bf16-vs-widened max |diff| = %.3e at [%d] (got %v want %v)", worst, wi, got[wi], want[wi])
}
