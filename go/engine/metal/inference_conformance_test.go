// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// inference_conformance_test.go is the acceptance gate for the engine-merge
// compose wave: the no-cgo Metal engine's inference.SessionHandle and
// inference.TextModel implementations run the shared go/enginetest conformance
// suites — the lifecycle / shape / error invariants any conformant engine must
// satisfy — against the hermetic synthetic gemma4 fixture (newKVContractTokenModel,
// a tiny real tokenizer + synthetic weights). requireNativeRuntime gates them on
// the metallib being present, so a checkout without weights stays green; with
// MLX_METALLIB_PATH set they run the real GPU decode path in engine/metal's new
// home. This is the finding from the prior wave (conformance kit "unwired") now
// closed: the composition exists, the suites run.
package native

import (
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/inference/enginetest"
)

// TestMetalEngineConformanceSessionHandle runs the enginetest.SessionHandle
// suite against a fresh retained nativeSession over the synthetic model.
func TestMetalEngineConformanceSessionHandle(t *testing.T) {
	requireNativeRuntime(t)
	enginetest.SessionHandle(t, func(t *testing.T) inference.SessionHandle {
		tm, _ := newKVContractTokenModel(t)
		sess := newNativeTextModel(tm, "gemma4").NewSession()
		if sess == nil {
			t.Fatal("nativeTextModel.NewSession returned nil over the synthetic fixture")
		}
		return sess
	})
}

// TestMetalEngineConformanceTextModel runs the enginetest.TextModel suite
// against a nativeTextModel over the synthetic model.
func TestMetalEngineConformanceTextModel(t *testing.T) {
	requireNativeRuntime(t)
	enginetest.TextModel(t, func(t *testing.T) inference.TextModel {
		tm, _ := newKVContractTokenModel(t)
		return newNativeTextModel(tm, "gemma4")
	})
}

// TestMetalEngineRegistersMetalBackend gates the standalone-resolution path:
// importing engine/metal self-registers "metal" so serving.NewMLXBackend and
// state/session.Session resolve it from go-inference alone (no go-mlx root).
func TestMetalEngineRegistersMetalBackend(t *testing.T) {
	b, ok := inference.Get("metal")
	if !ok {
		t.Fatal("engine/metal did not register inference backend \"metal\"")
	}
	if b.Name() != "metal" {
		t.Fatalf("registered backend Name() = %q, want \"metal\"", b.Name())
	}
}
