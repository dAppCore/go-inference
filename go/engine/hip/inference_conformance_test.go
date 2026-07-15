// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

// inference_conformance_test.go is the HIP-HARDWARE receipt for the reconcile —
// the tests Snider runs on his linux+AMD box to prove engine/hip's retained
// decode satisfies the shared engine contracts AND that the device<->kv.Snapshot
// round-trip is lossless. Unlike the hardware-free converter test
// (inference_kv_snapshot_test.go), these need a real ROCm/HIP device and a
// loaded Gemma4-Q4 model, so they SKIP unless:
//
//	GO_ROCM_RUN_ENGINE_CONFORMANCE=1
//	ROCM_CONFORMANCE_MODEL=<path to a Gemma4-Q4 model directory with tokenizer.json>
//	(and the ROCm runtime reports Available)
//
// There is no synthetic CPU HIP decode — hip's portable lane is metadata-only —
// so this is the deepest level that CAN be proven, and it is proven where the
// hardware lives. Run it with:
//
//	GO_ROCM_RUN_ENGINE_CONFORMANCE=1 ROCM_CONFORMANCE_MODEL=/models/gemma4-q4 \
//	  go test ./engine/hip/ -run 'TestHipEngine' -v
package hip

import (
	"bytes"
	"context"
	"os"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/decode/tokenizer"
	"dappco.re/go/inference/engine"
	"dappco.re/go/inference/engine/enginetest"
	"dappco.re/go/inference/kv"
)

const (
	hipConformanceRunEnv       = "GO_ROCM_RUN_ENGINE_CONFORMANCE"
	hipConformanceModelEnv     = "ROCM_CONFORMANCE_MODEL"
	hipConformanceModelTypeEnv = "ROCM_CONFORMANCE_MODEL_TYPE"
)

// hipRequireEngineTextModel gates the HIP-hardware conformance/parity tests and,
// when enabled, loads the real model and returns the shared engine.TextModel
// over hip's retained Gemma4-Q4 decode. It skips (never fails) when the gate
// env, the model path, the ROCm runtime, a Gemma4-Q4 linked runtime, or a
// tokenizer.json is missing — so a checkout without AMD hardware stays green.
func hipRequireEngineTextModel(t *testing.T) *engine.TextModel {
	t.Helper()
	if os.Getenv(hipConformanceRunEnv) != "1" {
		t.Skipf("set %s=1 and %s=<gemma4-q4 model dir> to run the HIP engine conformance on real AMD hardware", hipConformanceRunEnv, hipConformanceModelEnv)
	}
	modelPath := os.Getenv(hipConformanceModelEnv)
	if modelPath == "" {
		t.Skipf("set %s=<gemma4-q4 model dir> to run the HIP engine conformance", hipConformanceModelEnv)
	}
	if !ROCmAvailable() {
		t.Skip("ROCm runtime is not available on this host")
	}
	modelType := os.Getenv(hipConformanceModelTypeEnv)
	if modelType == "" {
		modelType = "gemma4"
	}

	result := (&rocmBackend{}).LoadModel(modelPath, inference.WithContextLen(4096))
	if !result.OK {
		t.Fatalf("LoadModel(%s): %v", modelPath, result.Value)
	}
	model, ok := result.Value.(*rocmModel)
	if !ok {
		t.Fatalf("LoadModel returned %T, want *rocmModel", result.Value)
	}
	loaded, ok := model.native.(*hipLoadedModel)
	if !ok {
		t.Skip("loaded model is not a native hipLoadedModel")
	}
	if !hipLoadedGemma4Q4GenerateLinked(loaded) {
		t.Skip("loaded model is not a Gemma4-Q4 linked runtime (no retained KV to exercise)")
	}
	tok, err := tokenizer.LoadTokenizer(core.PathJoin(modelPath, "tokenizer.json"))
	if err != nil {
		t.Skipf("the shared engine.TextModel needs a tokenizer.json beside the model: %v", err)
	}
	tm, err := newHipEngineTextModel(loaded, tok, modelType)
	if err != nil {
		t.Fatalf("newHipEngineTextModel: %v", err)
	}
	t.Cleanup(func() { _ = tm.Close() })
	return tm
}

// TestHipEngineConformanceSessionHandle runs the shared enginetest.SessionHandle
// suite (lifecycle / shape / error invariants) against hip's retained session.
func TestHipEngineConformanceSessionHandle(t *testing.T) {
	tm := hipRequireEngineTextModel(t)
	enginetest.SessionHandle(t, func(t *testing.T) inference.SessionHandle {
		session := tm.NewSession()
		if session == nil {
			t.Fatal("hip engine.TextModel.NewSession returned nil")
		}
		return session
	})
}

// TestHipEngineConformanceTextModel runs the shared enginetest.TextModel suite
// against hip's engine.TextModel.
func TestHipEngineConformanceTextModel(t *testing.T) {
	enginetest.TextModel(t, func(t *testing.T) inference.TextModel {
		return hipRequireEngineTextModel(t)
	})
}

// TestHipEngineKVSnapshotParity is THE reconcile receipt: a real device KV,
// captured to a kv.Snapshot, restored into a fresh session, and re-captured,
// must reproduce the KV byte-for-byte. This exercises the full chain on real
// hardware — device HostState -> hipDecodeStateToSnapshot -> hipSnapshotToDecode
// State -> hipMirrorGemma4Q4DecodeState -> device HostState -> snapshot — and is
// the proof the hardware-free converter test cannot give (it only covers the
// pure host<->snapshot leg).
func TestHipEngineKVSnapshotParity(t *testing.T) {
	tm := hipRequireEngineTextModel(t)
	ctx := context.Background()

	source := tm.NewSession()
	if source == nil {
		t.Fatal("NewSession returned nil")
	}
	defer func() { _ = source.Close() }()

	if err := source.Prefill(ctx, "The capital of France is"); err != nil {
		t.Fatalf("Prefill: %v", err)
	}
	// Generate a few tokens so the retained device KV covers prompt+generated.
	produced := 0
	for range source.Generate(ctx, inference.GenerateConfig{MaxTokens: 4}) {
		produced++
	}
	if err := source.Err(); err != nil {
		t.Fatalf("Generate error: %v", err)
	}
	if produced == 0 {
		t.Fatal("Generate produced no tokens; cannot build a device KV to capture")
	}

	snapshotA, err := source.CaptureKV(ctx)
	if err != nil {
		t.Fatalf("CaptureKV(source): %v", err)
	}
	if snapshotA == nil || snapshotA.SeqLen == 0 {
		t.Fatalf("source snapshot carries no KV: %+v", snapshotA)
	}

	restored := tm.NewSession()
	if restored == nil {
		t.Fatal("NewSession(restored) returned nil")
	}
	defer func() { _ = restored.Close() }()
	restorer, ok := restored.(inference.KVRestorer)
	if !ok {
		t.Fatal("session does not implement inference.KVRestorer")
	}
	if err := restorer.RestoreFromKV(ctx, snapshotA); err != nil {
		t.Fatalf("RestoreFromKV: %v", err)
	}
	snapshotB, err := restored.CaptureKV(ctx)
	if err != nil {
		t.Fatalf("CaptureKV(restored): %v", err)
	}

	assertHipSnapshotKVEqual(t, snapshotA, snapshotB)
}

// assertHipSnapshotKVEqual fails unless two snapshots carry byte-identical KV.
func assertHipSnapshotKVEqual(t *testing.T, want, got *kv.Snapshot) {
	t.Helper()
	if got == nil {
		t.Fatal("restored snapshot is nil")
	}
	if want.NumLayers != got.NumLayers {
		t.Fatalf("NumLayers: want %d got %d", want.NumLayers, got.NumLayers)
	}
	if want.SeqLen != got.SeqLen {
		t.Fatalf("SeqLen: want %d got %d", want.SeqLen, got.SeqLen)
	}
	if len(want.Layers) != len(got.Layers) {
		t.Fatalf("layer count: want %d got %d", len(want.Layers), len(got.Layers))
	}
	for index := range want.Layers {
		if !bytes.Equal(want.Layers[index].KeyBytes, got.Layers[index].KeyBytes) {
			t.Fatalf("layer %d KeyBytes differ after the device<->snapshot round-trip (lossy chain)", index)
		}
		if !bytes.Equal(want.Layers[index].ValueBytes, got.Layers[index].ValueBytes) {
			t.Fatalf("layer %d ValueBytes differ after the device<->snapshot round-trip", index)
		}
	}
}
