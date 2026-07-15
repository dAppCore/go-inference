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

// TestHotSwapResolver_ReloadModel_Good pins the string-typed reload seam the
// admin subsystem drives: it swaps in the new target and returns the previous +
// new active PATHS (never the internal loadedModel). The resolver is engine-
// neutral — it never inspects the model architecture — so a non-gemma boot model
// reloads to another non-gemma target with no special-casing.
func TestHotSwapResolver_ReloadModel_Good(t *testing.T) {
	first := &mockTextModel{modelType: "qwen3"}
	second := &mockTextModel{modelType: "llama"}
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

	prevPath, newActive, err := r.ReloadModel("/models/second", nil)
	if err != nil {
		t.Fatalf("ReloadModel: %v", err)
	}
	if prevPath != "/models/first" {
		t.Fatalf("ReloadModel prevPath = %q, want /models/first", prevPath)
	}
	if newActive != "/models/second" {
		t.Fatalf("ReloadModel newActive = %q, want /models/second", newActive)
	}
	got, err := r.ResolveModel(context.Background(), "")
	if err != nil {
		t.Fatalf("post-reload resolve: %v", err)
	}
	if got != second {
		t.Fatal("post-reload ResolveModel did not return the swapped-in model")
	}
}

// TestHotSwapResolver_ReloadModel_Bad pins the reload load-failure seam: a loader
// error surfaces as ReloadModel's error with empty paths, so the admin endpoint
// reports the real failure rather than a half-swapped state.
func TestHotSwapResolver_ReloadModel_Bad(t *testing.T) {
	r := newHotSwapResolver("/models/first", "", 0, nil)
	r.setLoader(func(path string, _ ...inference.LoadOption) (inference.TextModel, error) {
		if path == "/models/first" {
			return &mockTextModel{modelType: "gemma4"}, nil
		}
		return nil, core.NewError("reload load boom")
	})
	if _, err := r.ResolveModel(context.Background(), ""); err != nil {
		t.Fatalf("initial resolve: %v", err)
	}
	prevPath, newActive, err := r.ReloadModel("/models/broken", nil)
	if err == nil {
		t.Fatal("ReloadModel should surface the loader error")
	}
	if prevPath != "" || newActive != "" {
		t.Fatalf("failed ReloadModel returned paths (%q, %q), want empty", prevPath, newActive)
	}
}

// TestHotSwapResolver_SetOnLoad_Good pins the post-load hook: it runs after the
// lazy boot load AND after each reload swap, so per-model wiring (conversation
// continuity) re-attaches to whichever model is now active.
func TestHotSwapResolver_SetOnLoad_Good(t *testing.T) {
	boot := &mockTextModel{modelType: "gemma4"}
	next := &mockTextModel{modelType: "gemma4"}
	r := newHotSwapResolver("/models/boot", "", 0, nil)
	r.setLoader(func(path string, _ ...inference.LoadOption) (inference.TextModel, error) {
		if path == "/models/boot" {
			return boot, nil
		}
		return next, nil
	})
	var loaded []inference.TextModel
	r.setOnLoad(func(m inference.TextModel) { loaded = append(loaded, m) })

	if _, err := r.ResolveModel(context.Background(), ""); err != nil {
		t.Fatalf("resolve: %v", err)
	}
	if _, _, err := r.Replace("/models/next", nil); err != nil {
		t.Fatalf("replace: %v", err)
	}
	if len(loaded) != 2 || loaded[0] != boot || loaded[1] != next {
		t.Fatalf("onLoad fired for %v, want [boot, next] in load order", loaded)
	}
}

// TestHotSwapResolver_ReloadDetection_NonGemmaStandsDown pins arch-neutrality on
// the reload drafter ladder: with detection armed and a speculative loader
// present, a NON-gemma reload target still stands down (no MTP pair forced) —
// the reactive ladder only ever engages for a Gemma 4 family config, and every
// other architecture reloads plain autoregressive.
func TestHotSwapResolver_ReloadDetection_NonGemmaStandsDown(t *testing.T) {
	r := newHotSwapResolver("/models/boot", "", 0, nil)
	r.setDraftDetect(DraftDetectOptions{})
	r.setSpeculativeLoader(func(string, string, int, ...inference.LoadOption) (inference.TextModel, error) {
		t.Fatal("speculative loader must not engage for a non-gemma reload target")
		return nil, nil
	})
	// A directory with no gemma4 config.json — detection reads path-shape only.
	det := r.reloadDetection(t.TempDir())
	if det.Active() {
		t.Fatalf("reloadDetection over a non-gemma target = %+v, want stood down", det)
	}
}

// TestDefaultTextModelLoader_NoBackend_Bad proves the default loader fails
// cleanly (no panic) when no backend is registered — the case a composition
// that forgot the engine blank-import hits.
func TestDefaultTextModelLoader_NoBackend_Bad(t *testing.T) {
	if _, err := defaultTextModelLoader("/models/nope"); err == nil {
		t.Skip("a backend is registered in this test binary — nothing to assert")
	}
}
