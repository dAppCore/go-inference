// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"testing"

	"dappco.re/go/inference/scheme"
)

func TestCacheProvider_TurboQuantRegistered_Good(t *testing.T) {
	cacheScheme, ok := scheme.CacheFor("turboquant")
	if !ok {
		t.Fatal("turboquant not resolvable via scheme.CacheFor")
	}
	provider, ok := cacheScheme.(CacheProvider)
	if !ok {
		t.Fatalf("turboquant scheme %T does not satisfy kv.CacheProvider — the init upgrade did not land", cacheScheme)
	}
	if provider.Mode() != "turboquant" || provider.Serves() != scheme.StateKVCache {
		t.Fatalf("provider identity = %q/%v, want turboquant/StateKVCache", provider.Mode(), provider.Serves())
	}
}

func TestTurboQuantProvider_ValidateLayer_Good(t *testing.T) {
	layer := &LayerSnapshot{CacheMode: "turboquant", TurboQuantPayloads: [][]byte{{1, 2, 3}}}

	if err := (turboQuantProvider{}).ValidateLayer(layer); err != nil {
		t.Fatalf("ValidateLayer(payload-carrying layer) = %v, want nil", err)
	}
}

func TestTurboQuantProvider_ValidateLayer_BadMissingPayloads(t *testing.T) {
	layer := &LayerSnapshot{CacheMode: "turboquant"}

	if err := (turboQuantProvider{}).ValidateLayer(layer); err != errTurboQuantPayloadMissing {
		t.Fatalf("ValidateLayer(no payloads) = %v, want errTurboQuantPayloadMissing", err)
	}
}

func TestTurboQuantProvider_ValidateLayer_UglyNilLayer(t *testing.T) {
	if err := (turboQuantProvider{}).ValidateLayer(nil); err == nil {
		t.Fatal("ValidateLayer(nil) = nil, want error")
	}
}

func TestValidateKVSnapshotLayerSchemes_Good(t *testing.T) {
	snapshot := &Snapshot{Layers: []LayerSnapshot{
		{},                 // empty mode: legacy/default lane, skips resolution
		{CacheMode: "q8"},  // registered stub, no provider semantics
		{CacheMode: "turboquant", TurboQuantPayloads: [][]byte{{9}}},
	}}

	if err := validateKVSnapshotLayerSchemes(snapshot); err != nil {
		t.Fatalf("validateKVSnapshotLayerSchemes = %v, want nil", err)
	}
}

func TestValidateKVSnapshotLayerSchemes_BadUnknownMode(t *testing.T) {
	snapshot := &Snapshot{Layers: []LayerSnapshot{{CacheMode: "not-a-registered-scheme"}}}

	if err := validateKVSnapshotLayerSchemes(snapshot); err != errUnknownCacheMode {
		t.Fatalf("unknown mode error = %v, want errUnknownCacheMode", err)
	}
}

func TestValidateKVSnapshotLayerSchemes_BadPayloadsWrongMode(t *testing.T) {
	snapshot := &Snapshot{Layers: []LayerSnapshot{{CacheMode: "q8", TurboQuantPayloads: [][]byte{{1}}}}}

	if err := validateKVSnapshotLayerSchemes(snapshot); err != errTurboQuantPayloadMode {
		t.Fatalf("payloads-under-q8 error = %v, want errTurboQuantPayloadMode", err)
	}
}

func TestValidateKVSnapshotLayerSchemes_UglyNilSnapshot(t *testing.T) {
	if err := validateKVSnapshotLayerSchemes(nil); err != errSnapshotNil {
		t.Fatalf("nil snapshot error = %v, want errSnapshotNil", err)
	}
}
