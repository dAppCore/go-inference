// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	core "dappco.re/go"
	"dappco.re/go/inference/scheme"
)

// KV data providers — the scheme-registry wiring for per-layer cache formats
// (#261). A layer's CacheMode is resolved through scheme.CacheFor instead of
// being string-matched in the codec: adding a KV data format (TurboQuant, a
// future compaction mode…) means registering a scheme value that also
// satisfies CacheProvider, never adding a switch arm here. The scheme
// package's builtin stubs already name every mode the engines implement;
// this file upgrades the modes whose WIRE semantics this package owns.

// CacheProvider is the kv-side capability a registered cache scheme
// implements when the mode carries per-layer wire semantics this codec must
// enforce. Modes without one (fp16, paged, fixed…) resolve to the scheme
// registry's info stub and need no per-layer validation.
//
//	if provider, ok := cacheScheme.(kv.CacheProvider); ok {
//	    if err := provider.ValidateLayer(&layer); err != nil { return err }
//	}
type CacheProvider interface {
	scheme.CacheScheme
	// ValidateLayer checks a layer snapshot against the mode's wire
	// invariants before encode.
	ValidateLayer(layer *LayerSnapshot) error
}

// turboQuantProvider owns the "turboquant" wire semantics: a layer captured
// under the TurboQuant cache mode MUST carry its compressed payloads (the
// float32 side slices alone cannot reconstruct the ring), and payloads are
// meaningless under any other mode. It EMBEDS the registry's builtin turboquant
// value so the per-element width (scheme.CacheWidth) the memory planner sizes
// from is FORWARDED, not stripped, when this upgrade overwrites the stub —
// turboquant's width stays single-sourced in scheme/builtin.go. Mode/Serves stay
// explicit so a zero value (used by the ValidateLayer tests) is panic-safe.
type turboQuantProvider struct {
	scheme.CacheWidth // the builtin turboquant value; promotes KVBytesPerElement
}

func (turboQuantProvider) Mode() string             { return kvSnapshotTurboQuantCacheMode }
func (turboQuantProvider) Serves() scheme.StateKind { return scheme.StateKVCache }

func (turboQuantProvider) ValidateLayer(layer *LayerSnapshot) error {
	if layer == nil {
		return errSnapshotNil
	}
	if len(layer.TurboQuantPayloads) == 0 {
		return errTurboQuantPayloadMissing
	}
	return nil
}

func init() {
	// Upgrade the scheme registry's info stub for the modes this codec owns wire
	// semantics for. RegisterCache overwrites by mode, so the richer value
	// replaces the stub while every other mode keeps its stub — but it embeds the
	// stub's CacheWidth so the planner's per-element width survives the overwrite.
	base, ok := scheme.CacheFor(kvSnapshotTurboQuantCacheMode)
	if !ok {
		return
	}
	width, _ := base.(scheme.CacheWidth)
	scheme.RegisterCache(turboQuantProvider{width})
}

// errUnknownCacheMode is raised when a layer names a cache mode the scheme
// registry has never heard of — a loud failure instead of silently encoding
// a snapshot no engine can restore. Register the scheme (scheme.RegisterCache)
// before capturing under it.
var errUnknownCacheMode = core.NewError("mlx: KV layer cache mode is not a registered scheme")

// validateKVSnapshotLayerSchemes resolves every layer's CacheMode through the
// scheme registry and applies each resolved CacheProvider's wire invariants.
// An empty CacheMode is the legacy/default lane and skips resolution. The
// turboquant payload⇄mode invariant this replaces is preserved exactly:
// payloads under any other mode are rejected here (the TurboQuantPayloads
// field is turboquant's alone), and the turboquant provider rejects the
// missing-payload side.
func validateKVSnapshotLayerSchemes(snapshot *Snapshot) error {
	if snapshot == nil {
		return errSnapshotNil
	}
	for i := range snapshot.Layers {
		layer := &snapshot.Layers[i]
		if len(layer.TurboQuantPayloads) > 0 && layer.CacheMode != kvSnapshotTurboQuantCacheMode {
			return errTurboQuantPayloadMode
		}
		if layer.CacheMode == "" {
			continue
		}
		cacheScheme, ok := scheme.CacheFor(layer.CacheMode)
		if !ok {
			return errUnknownCacheMode
		}
		provider, ok := cacheScheme.(CacheProvider)
		if !ok {
			continue
		}
		if err := provider.ValidateLayer(layer); err != nil {
			return err
		}
	}
	return nil
}
