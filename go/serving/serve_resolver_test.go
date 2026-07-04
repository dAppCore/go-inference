// SPDX-Licence-Identifier: EUPL-1.2

package serving

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// TestHotSwapResolver_ResolveModel_LazyLoadOnce_Good proves the boot model loads
// lazily on the first ResolveModel call and is reused (not reloaded) after.
func TestHotSwapResolver_ResolveModel_LazyLoadOnce_Good(t *testing.T) {
	model := &mockTextModel{modelType: "gemma4"}
	calls := 0
	r := newHotSwapResolver("/models/x", "", 0, nil)
	r.setLoader(func(path string, _ ...inference.LoadOption) (inference.TextModel, error) {
		calls++
		if path != "/models/x" {
			t.Errorf("loader got path %q, want /models/x", path)
		}
		return model, nil
	})

	got, err := r.ResolveModel(context.Background(), "ignored-name")
	if err != nil {
		t.Fatalf("first ResolveModel: %v", err)
	}
	if got != model {
		t.Fatal("ResolveModel returned a different model than the loader produced")
	}
	if _, err := r.ResolveModel(context.Background(), "ignored-name"); err != nil {
		t.Fatalf("second ResolveModel: %v", err)
	}
	if calls != 1 {
		t.Fatalf("loader called %d times, want 1 (lazy load once, then reuse)", calls)
	}
}

// TestHotSwapResolver_ResolveModel_ModelLess_Bad proves a model-less start
// surfaces errNoModelLoaded until a model arrives via Replace.
func TestHotSwapResolver_ResolveModel_ModelLess_Bad(t *testing.T) {
	r := newHotSwapResolver("", "", 0, nil)
	if _, err := r.ResolveModel(context.Background(), ""); err == nil {
		t.Fatal("model-less ResolveModel should error until a model is loaded")
	}
}

// TestHotSwapResolver_ResolveModel_LoadError_Bad proves a boot-load failure
// surfaces to the caller rather than caching a nil model.
func TestHotSwapResolver_ResolveModel_LoadError_Bad(t *testing.T) {
	r := newHotSwapResolver("/models/x", "", 0, nil)
	r.setLoader(func(string, ...inference.LoadOption) (inference.TextModel, error) {
		return nil, core.NewError("load boom")
	})
	if _, err := r.ResolveModel(context.Background(), ""); err == nil {
		t.Fatal("expected the load error to surface")
	}
}

// TestHotSwapResolver_Replace_SwapsActive_Good proves Replace atomically swaps
// the active model, reports the previous path, and routes new resolves to the
// replacement.
func TestHotSwapResolver_Replace_SwapsActive_Good(t *testing.T) {
	first := &mockTextModel{modelType: "first"}
	second := &mockTextModel{modelType: "second"}
	r := newHotSwapResolver("/models/first", "", 0, nil)
	r.setLoader(func(path string, _ ...inference.LoadOption) (inference.TextModel, error) {
		if path == "/models/first" {
			return first, nil
		}
		return second, nil
	})

	if _, err := r.ResolveModel(context.Background(), ""); err != nil {
		t.Fatalf("initial resolve: %v", err)
	}
	if r.CurrentPath() != "/models/first" {
		t.Fatalf("CurrentPath = %q, want /models/first", r.CurrentPath())
	}

	prev, newPath, err := r.Replace("/models/second", nil)
	if err != nil {
		t.Fatalf("Replace: %v", err)
	}
	if prev == nil || prev.modelPath != "/models/first" {
		t.Fatalf("Replace prev = %+v, want modelPath /models/first", prev)
	}
	if newPath != "/models/second" {
		t.Fatalf("Replace newPath = %q, want /models/second", newPath)
	}
	got, err := r.ResolveModel(context.Background(), "")
	if err != nil {
		t.Fatalf("post-replace resolve: %v", err)
	}
	if got != second {
		t.Fatal("post-replace ResolveModel did not return the swapped-in model")
	}
	if r.CurrentPath() != "/models/second" {
		t.Fatalf("post-replace CurrentPath = %q, want /models/second", r.CurrentPath())
	}
}

// TestMetalTextModelLoader_NoBackend_Bad proves the default metal loader fails
// cleanly (no panic) when no "metal" backend is registered — the case a non-
// Apple host or a composition that forgot the engine blank-import hits.
func TestMetalTextModelLoader_NoBackend_Bad(t *testing.T) {
	if _, err := metalTextModelLoader("/models/nope"); err == nil {
		t.Skip("a metal backend is registered in this test binary — nothing to assert")
	}
}
