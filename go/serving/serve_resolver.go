// SPDX-Licence-Identifier: EUPL-1.2

// serve_resolver.go is the hot-swap serve resolver ported out of lthn-mlx's
// cmd/mlx/serve_resolver.go so the business logic lives in a go-inference
// library rather than dying with go-mlx's cmd/. It backs /v1/admin/serve/reload
// (F-7): the active model lives in an atomic.Pointer so ResolveModel reads are
// lock-free on the hot path, and Replace serialises swaps. The engine loaders
// are injected (ModelLoader / SpeculativeLoader) so the same resolver drives the
// Apple metal engine, the native no-cgo path, or any registered backend without
// the serving library importing an engine.

package serving

import (
	"context"
	"sync"
	"sync/atomic"

	core "dappco.re/go"
	"dappco.re/go/inference"
	openai "dappco.re/go/inference/serving/provider/openai"
)

// ModelLoader loads a single model at path into an inference.TextModel. The
// default serve loader resolves the registered backend (metal on Apple, blank-
// imported by the composition root) via inference.LoadModel; a different engine
// or `serve --native` injects its own loader.
type ModelLoader func(path string, opts ...inference.LoadOption) (inference.TextModel, error)

// SpeculativeLoader loads a target + drafter pair as one speculative
// inference.TextModel running draftBlock-wide MTP verify forwards. A nil
// SpeculativeLoader means the registered engine exposes no speculative path —
// the serve orchestration then declines to arm a detected drafter and degrades
// to plain autoregressive load with an honest notice, rather than panicking.
type SpeculativeLoader func(targetPath, draftPath string, draftBlock int, opts ...inference.LoadOption) (inference.TextModel, error)

// metalTextModelLoader is the default ModelLoader: it resolves the registered
// "metal" backend (Apple no-cgo engine) through inference.LoadModel. Engines
// that aren't registered fail cleanly with "not available".
func metalTextModelLoader(path string, opts ...inference.LoadOption) (inference.TextModel, error) {
	merged := append(append([]inference.LoadOption(nil), opts...), inference.WithBackend("metal"))
	result := inference.LoadModel(path, merged...)
	if !result.OK {
		if err, ok := result.Value.(error); ok {
			return nil, err
		}
		return nil, core.E("serving.metalTextModelLoader", "metal backend failed to load model", nil)
	}
	tm, ok := result.Value.(inference.TextModel)
	if !ok || tm == nil {
		return nil, core.E("serving.metalTextModelLoader", "metal backend returned non-TextModel value", nil)
	}
	return tm, nil
}

// loadedModel is the snapshot the hotSwapResolver hands back to callers.
// modelPath stamps which weights are in use so /v1/admin/serve/status + reload
// audit lines can name the source.
type loadedModel struct {
	model     inference.TextModel
	modelPath string
}

// errNoModelLoaded is returned by ResolveModel when the driver started
// model-less (serve with no --model) and nothing has been loaded via
// /v1/admin/serve/reload yet. The compat mux surfaces it to inference callers;
// admin + health endpoints stay reachable so a model can be loaded.
var errNoModelLoaded = core.NewError("no model loaded — POST /v1/admin/serve/reload to load a model")

// hotSwapResolver is the openai.Resolver that backs /v1/admin/serve/reload. The
// active model lives in an atomic.Pointer so ResolveModel reads are lock-free on
// the hot path (every chat/completions call hits this); Replace serialises swaps
// under swapMu so two concurrent reloads can't race.
//
// First-call lazy load: the boot-time model isn't loaded eagerly — the first
// ResolveModel triggers the load via initial.Do. That keeps `serve --model X`
// from blocking on a multi-GB load before binding the listener.
//
// Drain policy: in-flight Generate/Chat calls keep their TextModel reference and
// complete on old weights; new calls hit new weights. The old model is NOT
// explicitly Closed — Go GC reclaims when the last in-flight reference drops.
type hotSwapResolver struct {
	active         atomic.Pointer[loadedModel]
	initial        sync.Once
	initErr        error
	initPath       string
	initDraftPath  string
	initDraftBlock int
	initOpts       []inference.LoadOption
	// draftDetect arms reload-time drafter detection; nil = never armed (reloads
	// stay autoregressive).
	draftDetect *DraftDetectOptions
	swapMu      sync.Mutex
	// onLoad runs after every successful load — the lazy boot load and each
	// reload swap — so per-model wiring (conversation continuity) re-attaches.
	onLoad func(inference.TextModel)
	// loader builds a TextModel from a path — the registered metal engine
	// (metalTextModelLoader, the default) or an injected native/other loader.
	loader ModelLoader
	// speculativeLoader builds a target+draft TextModel when drafter detection
	// is active. nil when the engine exposes no speculative path.
	speculativeLoader SpeculativeLoader
}

// newHotSwapResolver returns a resolver staged with the initial model path +
// options (and, when a drafter resolved, the MTP pair + draft block). The model
// is NOT loaded until the first ResolveModel call. The loader defaults to the
// registered metal backend; setLoader / setSpeculativeLoader override it.
func newHotSwapResolver(modelPath, draftPath string, draftBlock int, opts []inference.LoadOption) *hotSwapResolver {
	return &hotSwapResolver{
		initPath:       modelPath,
		initDraftPath:  draftPath,
		initDraftBlock: draftBlock,
		initOpts:       opts,
		loader:         metalTextModelLoader,
	}
}

// setLoader swaps the model loader before the first ResolveModel call.
func (r *hotSwapResolver) setLoader(loader ModelLoader) {
	if loader != nil {
		r.loader = loader
	}
}

// setSpeculativeLoader swaps the target+draft loader before the first
// ResolveModel call. A nil loader leaves the resolver without a speculative path.
func (r *hotSwapResolver) setSpeculativeLoader(loader SpeculativeLoader) {
	r.speculativeLoader = loader
}

// setOnLoad registers a hook run after every successful model load so per-model
// wiring (conversation continuity) re-attaches to the new model. Set before the
// first ResolveModel call.
func (r *hotSwapResolver) setOnLoad(hook func(inference.TextModel)) {
	r.onLoad = hook
}

// setDraftDetect arms reload-time drafter detection: each Replace re-runs the
// reactive ladder over the new target with these options, so a hot-swapped
// Gemma 4 model keeps MTP instead of silently coming up autoregressive. Set
// before the first Replace call.
func (r *hotSwapResolver) setDraftDetect(opts DraftDetectOptions) {
	r.draftDetect = &opts
}

// reloadDetection resolves the drafter for a reload target: the boot ladder,
// reactive rungs only (no explicit path). Detection disabled (or never armed),
// or the engine exposing no speculative loader, stands down.
func (r *hotSwapResolver) reloadDetection(newPath string) DraftDetection {
	if r.draftDetect == nil || r.speculativeLoader == nil {
		return DraftDetection{}
	}
	return DetectGemma4DraftPath(newPath, "", *r.draftDetect)
}

// ResolveModel returns the active model. The first call loads the initial model;
// subsequent calls return whatever's currently active (possibly swapped via
// Replace). modelName is the OpenAI-API `model` field, ignored — serve hosts one
// model at a time.
func (r *hotSwapResolver) ResolveModel(_ context.Context, _ string) (inference.TextModel, error) {
	// Already-active model wins — covers the lazy-loaded boot model and one
	// swapped in via Replace. Lock-free hot path. Checked first so a reload-
	// loaded model is never shadowed by a stale boot-load initErr.
	if cur := r.active.Load(); cur != nil {
		return cur.model, nil
	}
	// Model-less start: no boot model was staged. Inference is unavailable until
	// a model is loaded via Replace; admin + health stay reachable.
	if r.initPath == "" {
		return nil, errNoModelLoaded
	}
	// First call with a staged boot model: load it now (lazy).
	r.initial.Do(func() {
		var m inference.TextModel
		var err error
		if r.initDraftPath != "" && r.speculativeLoader != nil {
			m, err = r.speculativeLoader(r.initPath, r.initDraftPath, r.initDraftBlock, r.initOpts...)
		} else {
			m, err = r.loader(r.initPath, r.initOpts...)
		}
		if err != nil {
			r.initErr = err
			return
		}
		if r.onLoad != nil {
			r.onLoad(m)
		}
		r.active.Store(&loadedModel{model: m, modelPath: r.initPath})
	})
	if r.initErr != nil {
		return nil, r.initErr
	}
	if cur := r.active.Load(); cur != nil {
		return cur.model, nil
	}
	return nil, r.initErr
}

// Replace loads a new model at newPath with newOpts and atomically swaps it in.
// It returns the previously-active loadedModel (inspect modelPath for audit; do
// NOT Close it — see the drain policy) plus the new active path. swapMu
// serialises swaps so two concurrent reloads can't race.
//
// The auto-tuned boot options (initOpts) are preserved across reload: newOpts is
// overlaid on top of initOpts (last-wins) so a reload that only carries
// ContextLen keeps every tuned field rather than reloading with bare defaults.
func (r *hotSwapResolver) Replace(newPath string, newOpts []inference.LoadOption) (prev *loadedModel, newActive string, err error) {
	r.swapMu.Lock()
	defer r.swapMu.Unlock()
	var loaded inference.TextModel
	// Reload symmetry: the same reactive drafter ladder that ran at boot runs
	// over the swapped-in target, so a Gemma 4 model with an assistant/ pair or
	// MTP gguf beside it keeps speculative decode.
	if detection := r.reloadDetection(newPath); detection.Active() {
		loaded, err = r.speculativeLoader(newPath, detection.DraftPath, r.initDraftBlock, r.reloadLoadOpts(newOpts)...)
	} else {
		loaded, err = r.loader(newPath, r.reloadLoadOpts(newOpts)...)
	}
	if err != nil {
		return nil, "", err
	}
	if r.onLoad != nil {
		r.onLoad(loaded)
	}
	next := &loadedModel{model: loaded, modelPath: newPath}
	prev = r.active.Swap(next)
	return prev, newPath, nil
}

// reloadLoadOpts overlays the per-reload options on top of the auto-tuned boot
// options. Application is last-wins, so initOpts establishes the tuned baseline
// and newOpts overrides only the fields the reload explicitly carries.
func (r *hotSwapResolver) reloadLoadOpts(newOpts []inference.LoadOption) []inference.LoadOption {
	merged := make([]inference.LoadOption, 0, len(r.initOpts)+len(newOpts))
	merged = append(merged, r.initOpts...)
	merged = append(merged, newOpts...)
	return merged
}

// ReloadModel is the string-typed reload seam the admin subsystem drives: it
// swaps in newPath and returns the previous + new active paths (never the
// internal loadedModel, so the admin package need not import serving's private
// types). It is Replace with the audit-facing shape /v1/admin/serve/reload wants.
//
//	prevPath, newPath, err := r.ReloadModel(toPath, opts)
func (r *hotSwapResolver) ReloadModel(newPath string, newOpts []inference.LoadOption) (prevPath, newActive string, err error) {
	prev, active, err := r.Replace(newPath, newOpts)
	if err != nil {
		return "", "", err
	}
	if prev != nil {
		prevPath = prev.modelPath
	}
	return prevPath, active, nil
}

// CurrentPath returns the modelPath of the active model, or the initial path if
// no load has happened yet. Used by handlers that render the active source
// (e.g. /v1/admin/serve/status).
func (r *hotSwapResolver) CurrentPath() string {
	if cur := r.active.Load(); cur != nil {
		return cur.modelPath
	}
	return r.initPath
}

// openaiResolver returns r as an openai.Resolver for wire-up sites that want to
// keep the interface narrow without exposing the hot-swap surface.
func (r *hotSwapResolver) openaiResolver() openai.Resolver {
	return openai.ResolverFunc(r.ResolveModel)
}
