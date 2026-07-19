// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"strings"
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/inference/model"
)

// train_trainer_mask_test.go gates the LossMask half of the engine.Trainer contract (#39): the metal
// trainer HONOURS batch.LossMask — masked positions contribute zero loss and zero gradient, the loss
// normaliser divides by the UNMASKED count, and an all-masked batch is refused rather than divided by
// zero. The mask-resolution and gather arithmetic are host-gated (no GPU); the trainer wiring is
// runtime-gated end to end on a tiny synthetic session.

// TestLossMaskRows_Good: an unset mask trains every row; a response-style mask resolves to exactly the
// rows whose TARGET token is unmasked (row p trains iff mask[p+1] > 0 — the hip target-token parity).
func TestLossMaskRows_Good(t *testing.T) {
	rows, masked, err := lossMaskRows(inference.LossMask{}, 0, 6)
	if err != nil || masked || rows != nil {
		t.Fatalf("unset mask must return (nil, false, nil): rows=%v masked=%v err=%v", rows, masked, err)
	}

	// tokens [t0 t1 t2 t3]; mask marks t2,t3 as response targets → rows 1 (predicts t2) and 2 (predicts t3).
	mask := inference.LossMask{Values: [][]float32{{0, 0, 1, 1}}}
	rows, masked, err = lossMaskRows(mask, 0, 4)
	if err != nil || !masked {
		t.Fatalf("set mask must resolve: masked=%v err=%v", masked, err)
	}
	if len(rows) != 2 || rows[0] != 1 || rows[1] != 2 {
		t.Fatalf("mask [0 0 1 1] must train rows [1 2] (target-token semantics), got %v", rows)
	}

	// the first token's mask entry is irrelevant (nothing predicts token 0).
	rows, _, err = lossMaskRows(inference.LossMask{Values: [][]float32{{1, 0, 1}}}, 0, 3)
	if err != nil {
		t.Fatalf("lossMaskRows: %v", err)
	}
	if len(rows) != 1 || rows[0] != 1 {
		t.Fatalf("mask [1 0 1]: only row 1 (target t2) trains, got %v", rows)
	}
}

// TestLossMaskRows_Bad: a set mask that does not cover the sequence is a malformed batch and is refused
// loudly — never silently treated as all-masked (the silent-exclusion trap this trainer refuses to copy).
func TestLossMaskRows_Bad(t *testing.T) {
	// mask row length ≠ token count.
	_, _, err := lossMaskRows(inference.LossMask{Values: [][]float32{{1, 1, 1}}}, 0, 4)
	if err == nil {
		t.Fatal("a mask row shorter than the sequence must be refused")
	}
	for _, want := range []string{"3", "4", "per token"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("shape refusal must name %q; got: %s", want, err.Error())
		}
	}

	// mask set but missing this sequence's row.
	_, _, err = lossMaskRows(inference.LossMask{Values: [][]float32{{1, 1}}}, 1, 2)
	if err == nil {
		t.Fatal("a set mask missing a sequence's row must be refused")
	}
	if !strings.Contains(err.Error(), "row per sequence") {
		t.Fatalf("missing-row refusal must say a set mask needs a row per sequence; got: %s", err.Error())
	}
}

// TestLossMaskRows_Ugly: surprising-but-valid shapes — an all-masked row resolves to zero rows WITHOUT
// error (the caller decides what an empty contribution means), fractional positives count as unmasked,
// non-positive values mask.
func TestLossMaskRows_Ugly(t *testing.T) {
	rows, masked, err := lossMaskRows(inference.LossMask{Values: [][]float32{{0, 0, 0}}}, 0, 3)
	if err != nil || !masked || len(rows) != 0 {
		t.Fatalf("all-masked row must resolve to zero rows with no error: rows=%v masked=%v err=%v", rows, masked, err)
	}
	rows, _, err = lossMaskRows(inference.LossMask{Values: [][]float32{{0, 0.5, -1}}}, 0, 3)
	if err != nil {
		t.Fatalf("lossMaskRows: %v", err)
	}
	if len(rows) != 1 || rows[0] != 0 {
		t.Fatalf("0.5 counts as unmasked, -1 masks: want rows [0], got %v", rows)
	}
}

// TestGatherRowsF32_Good proves the gathered-row loss pipeline's two contract properties on the host CE
// oracle (CrossEntropyBackwardF32, pure host — the reference CrossEntropyBackwardF32Auto falls back to):
// a masked position changes NEITHER the loss NOR the gradients (its row is simply absent from the
// gathered inputs), and the loss normaliser divides by the UNMASKED count, not the full count.
func TestGatherRowsF32_Good(t *testing.T) {
	const total, vocab = 4, 7
	logits := syntheticFloat32(total*vocab, 3)
	targets := []int32{1, 5, 2, 6}
	keep := []int{0, 2, 3} // row 1 is masked

	gl := gatherRowsF32(logits, keep, vocab)
	gt := make([]int32, len(keep))
	for i, r := range keep {
		gt[i] = targets[r]
	}
	loss, dLogits, err := CrossEntropyBackwardF32(gl, gt, len(keep), vocab)
	if err != nil {
		t.Fatalf("gathered CE: %v", err)
	}

	// normaliser: the gathered mean equals the mean of the kept rows' own single-row losses.
	var want float64
	for i := range keep {
		rl, _, e := CrossEntropyBackwardF32(gl[i*vocab:(i+1)*vocab], gt[i:i+1], 1, vocab)
		if e != nil {
			t.Fatalf("row CE: %v", e)
		}
		want += float64(rl)
	}
	want /= float64(len(keep))
	if math.Abs(float64(loss)-want) > 1e-6 {
		t.Fatalf("gathered loss must be the mean over the UNMASKED rows: got %v want %v", loss, want)
	}

	// invariance: perturbing the MASKED row's logits and target changes nothing downstream.
	perturbed := append([]float32(nil), logits...)
	for i := range vocab {
		perturbed[1*vocab+i] += 3.5
	}
	pTargets := append([]int32(nil), targets...)
	pTargets[1] = 0
	gl2 := gatherRowsF32(perturbed, keep, vocab)
	gt2 := make([]int32, len(keep))
	for i, r := range keep {
		gt2[i] = pTargets[r]
	}
	loss2, dLogits2, err := CrossEntropyBackwardF32(gl2, gt2, len(keep), vocab)
	if err != nil {
		t.Fatalf("perturbed gathered CE: %v", err)
	}
	if loss2 != loss {
		t.Fatalf("a masked position's content changed the loss: %v vs %v", loss, loss2)
	}
	for i := range dLogits {
		if dLogits2[i] != dLogits[i] {
			t.Fatalf("a masked position's content changed gradient %d: %v vs %v", i, dLogits[i], dLogits2[i])
		}
	}
}

// maskTrainerFixture builds a tiny synthetic bf16 token model (2 uniform full-attention layers, tied
// head) — the shared base for the runtime-gated LossMask trainer tests. Deterministic weights, so two
// fresh trainers over it step identically.
func maskTrainerFixture(t *testing.T) *NativeTokenModel {
	t.Helper()
	const dModel, nHeads, nKV, headDim, dFF, vocab, nL, maxLen = 64, 2, 1, 32, 128, 32, 2, 16
	layers := make([]DecodeLayerWeights, nL)
	types := make([]string, nL)
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
		types[li] = "full_attention"
	}
	specs := model.DeriveLayers(types, 0)
	for i := range specs {
		specs[i].HeadDim, specs[i].KVHeads = headDim, nKV
	}
	embed := toBF16Bytes(syntheticFloat32(vocab*dModel, 21))
	g := &BF16Model{Layers: layers, Embed: embed, FinalNorm: toBF16Bytes(syntheticFloat32(dModel, 22)), LMHead: embed, Tied: true}
	arch := model.Arch{
		Hidden: dModel, Heads: nHeads, KVHeads: nKV, HeadDim: headDim, FF: dFF, Vocab: vocab,
		GlobalHeadDim: headDim, GlobalKVHeads: nKV,
		Eps: 1e-5, AttnScale: 0.176776695, RopeBase: 10000, RopeScale: 1, RopeLocalBase: 10000,
		RotaryDim: headDim, RotaryDimLocal: headDim, Layer: specs,
	}
	tm, err := NewBF16TokenModel(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewBF16TokenModel: %v", err)
	}
	return tm
}

// maskTrainer opens a fresh LoRATrainer over tm with the deterministic test config.
func maskTrainer(t *testing.T, tm *NativeTokenModel) *LoRATrainer {
	t.Helper()
	tr, err := NewLoRATrainer(tm, inference.TrainingConfig{
		LoRA:         inference.LoRAConfig{Rank: 4, Alpha: 8},
		LearningRate: 0.02,
	})
	if err != nil {
		t.Fatalf("NewLoRATrainer: %v", err)
	}
	return tr
}

// TestLoRATrainerStepLossMask_Good: end to end on a real (synthetic-weight) session — masking the final
// target makes Step BLIND to that token: two batches identical except the masked last token produce a
// bit-identical loss AND bit-identical adapter updates (the token only ever influenced its own dropped
// prediction row). Unmasked, the same change moves the loss — proving the mask, not coincidence, is
// what removed the sensitivity. A fully-masked sequence inside a live batch is skipped, not fatal.
func TestLoRATrainerStepLossMask_Good(t *testing.T) {
	requireNativeRuntime(t)
	tm := maskTrainerFixture(t)
	ids1 := []int32{1, 2, 3, 4, 5, 6}
	ids2 := []int32{1, 2, 3, 4, 5, 29} // differs ONLY in the last token
	mask := inference.LossMask{Values: [][]float32{{1, 1, 1, 1, 1, 0}}} // last target masked

	step := func(ids []int32, m inference.LossMask) (float64, []float32, []float32) {
		tr := maskTrainer(t, tm)
		defer func() { _ = tr.Close() }()
		loss, err := tr.Step(inference.Batch{TokenIDs: [][]int32{ids}, LossMask: m})
		if err != nil {
			t.Fatalf("Step: %v", err)
		}
		return loss, append([]float32(nil), tr.a...), append([]float32(nil), tr.b...)
	}

	loss1, a1, b1 := step(ids1, mask)
	loss2, a2, b2 := step(ids2, mask)
	if loss1 != loss2 {
		t.Fatalf("a masked position's token changed the loss: %v vs %v", loss1, loss2)
	}
	for i := range a1 {
		if a1[i] != a2[i] {
			t.Fatalf("a masked position's token changed adapter gradient A[%d]", i)
		}
	}
	for i := range b1 {
		if b1[i] != b2[i] {
			t.Fatalf("a masked position's token changed adapter gradient B[%d]", i)
		}
	}

	// control: WITHOUT the mask the same token change must move the loss (the sensitivity the mask removed).
	uloss1, _, _ := step(ids1, inference.LossMask{})
	uloss2, _, _ := step(ids2, inference.LossMask{})
	if uloss1 == uloss2 {
		t.Fatalf("control failed: unmasked losses identical (%v) — the fixture cannot discriminate", uloss1)
	}

	// a fully-masked sequence inside a live batch contributes nothing and is skipped, not fatal:
	// the batch mean equals the live sequence's own mean.
	trLive := maskTrainer(t, tm)
	defer func() { _ = trLive.Close() }()
	liveLoss, err := trLive.Loss(inference.Batch{TokenIDs: [][]int32{ids1}, LossMask: mask})
	if err != nil {
		t.Fatalf("live-only Loss: %v", err)
	}
	trMixed := maskTrainer(t, tm)
	defer func() { _ = trMixed.Close() }()
	mixedLoss, err := trMixed.Loss(inference.Batch{
		TokenIDs: [][]int32{ids1, {7, 8, 9}},
		LossMask: inference.LossMask{Values: [][]float32{{1, 1, 1, 1, 1, 0}, {0, 0, 0}}},
	})
	if err != nil {
		t.Fatalf("mixed Loss: %v", err)
	}
	if mixedLoss != liveLoss {
		t.Fatalf("a fully-masked sequence must be skipped from the mean: mixed %v vs live %v", mixedLoss, liveLoss)
	}
}

// TestLoRATrainerStepLossMask_Bad: the edge shapes — an all-masked batch is refused loudly (no divide
// by zero, no NaN), and a malformed mask shape surfaces the lossMaskRows refusal through Step.
func TestLoRATrainerStepLossMask_Bad(t *testing.T) {
	requireNativeRuntime(t)
	tm := maskTrainerFixture(t)
	tr := maskTrainer(t, tm)
	defer func() { _ = tr.Close() }()

	// all-masked batch: every sequence fully masked → refused, not divided by zero.
	_, err := tr.Step(inference.Batch{
		TokenIDs: [][]int32{{1, 2, 3}, {4, 5}},
		LossMask: inference.LossMask{Values: [][]float32{{0, 0, 0}, {1, 0}}},
	})
	if err == nil {
		t.Fatal("an all-masked batch must be refused")
	}
	if !strings.Contains(err.Error(), "LossMask") {
		t.Fatalf("the all-masked refusal must name the mask; got: %s", err.Error())
	}
	_, err = tr.Loss(inference.Batch{
		TokenIDs: [][]int32{{1, 2, 3}},
		LossMask: inference.LossMask{Values: [][]float32{{0, 0, 0}}},
	})
	if err == nil {
		t.Fatal("an all-masked Loss batch must be refused")
	}

	// malformed mask shape: refused before any training.
	_, err = tr.Step(inference.Batch{
		TokenIDs: [][]int32{{1, 2, 3}},
		LossMask: inference.LossMask{Values: [][]float32{{1, 1}}},
	})
	if err == nil {
		t.Fatal("a wrong-length mask row must be refused")
	}
	if !strings.Contains(err.Error(), "per token") {
		t.Fatalf("the shape refusal must explain the mask is per token; got: %s", err.Error())
	}
}
