// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"

	"dappco.re/go/inference/model"
)

// TestGPUCaptureTrigger drives the one-shot capture trigger through a real fixture
// decode. Two legitimate outcomes, both asserted: with the Metal framework's gate up
// (MTL_CAPTURE_ENABLED=1) the .gputrace lands on disk and the trigger disarms; without
// it the start is refused, the trigger disarms CLEANLY, and — either way — decode
// output is unaffected (the trigger must never perturb the engine).
func TestGPUCaptureTrigger(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, nHeads, nKV, headDim, dFF = 256, 4, 2, 64, 512
	const vocab, nL, maxLen = 64, 2, 16
	layers := make([]DecodeLayerWeights, nL)
	types := make([]string, nL)
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
		types[li] = "full_attention"
	}
	specs := model.DeriveLayers(types, 0)
	embed := toBF16Bytes(syntheticFloat32(vocab*dModel, 21))
	g := &BF16Model{Layers: layers, Embed: embed, FinalNorm: toBF16Bytes(syntheticFloat32(dModel, 22)), LMHead: embed, Tied: true}
	arch := model.Arch{
		Hidden: dModel, Heads: nHeads, KVHeads: nKV, HeadDim: headDim, FF: dFF, Vocab: vocab,
		GlobalHeadDim: headDim, GlobalKVHeads: nKV,
		Eps: 1e-5, AttnScale: 0.125, RopeBase: 10000, RopeScale: 1, RopeLocalBase: 10000,
		RotaryDim: headDim, RotaryDimLocal: headDim, Layer: specs,
	}
	ids := []int32{1, 2, 3, 4}

	// The un-captured truth arm.
	plain, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession(plain): %v", err)
	}
	defer plain.Close()
	want, err := plain.ForwardCaptureFinalHidden(ids)
	if err != nil {
		t.Fatalf("ForwardCaptureFinalHidden(plain): %v", err)
	}

	tracePath := t.TempDir() + "/round.gputrace"
	gpuCaptureArm(tracePath, 2)
	defer gpuCaptureState.Store(gpuCaptureOff) // never leak an armed trigger into other tests

	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession(captured): %v", err)
	}
	defer sess.Close()
	got, err := sess.ForwardCaptureFinalHidden(ids)
	if err != nil {
		t.Fatalf("ForwardCaptureFinalHidden(captured): %v", err)
	}
	eqBytes(t, "decode under capture trigger", got, want)

	if gpuCaptureState.Load() != gpuCaptureOff {
		t.Fatalf("trigger state = %d after the run, want disarmed (one-shot)", gpuCaptureState.Load())
	}
	if _, statErr := os.Stat(tracePath); statErr == nil {
		t.Logf("capture WROTE %s (MTL_CAPTURE_ENABLED was up)", tracePath)
	} else {
		t.Logf("capture refused cleanly (MTL_CAPTURE_ENABLED not set) — disarm path exercised")
	}
}
