// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/inference/scheme"
)

// TestNativeSchemeConsumption gates R4/R5: native registers the gemma4 sequence-mixer + KV-cache
// identities and resolveSequenceSchemes resolves them from the shared registries + enforces the
// mixer-owns-state contract. The negative case proves the gate actually refuses a mismatched
// pairing (a recurrent-state mixer cannot use a KV cache) rather than rubber-stamping. No model
// load — pure scheme logic.
func TestNativeSchemeConsumption(t *testing.T) {
	if err := resolveSequenceSchemes(); err != nil {
		t.Fatalf("resolveSequenceSchemes: %v", err)
	}
	m, ok := scheme.MixerFor(mixerSoftmaxHybrid)
	if !ok {
		t.Fatalf("mixer %q not registered", mixerSoftmaxHybrid)
	}
	if m.State() != scheme.StateKVCache {
		t.Fatalf("mixer state = %v, want kv-cache", m.State())
	}
	c, ok := scheme.CacheFor(cacheModeDefault)
	if !ok {
		t.Fatal("default KV-cache scheme not registered")
	}
	if c.Serves() != scheme.StateKVCache {
		t.Fatalf("cache serves = %v, want kv-cache", c.Serves())
	}
	if !scheme.Compatible(m, c) {
		t.Fatal("softmax-hybrid mixer + default KV cache must be compatible")
	}
	if scheme.Compatible(recurrentProbe{}, c) {
		t.Fatal("a recurrent-state mixer against a KV cache must be incompatible (the gate is a no-op)")
	}
}

// recurrentProbe is a throwaway mixer that needs recurrent state — used only to prove the KV cache
// is correctly judged incompatible with it.
type recurrentProbe struct{}

func (recurrentProbe) Kind() string            { return "native-test-recurrent" }
func (recurrentProbe) State() scheme.StateKind { return scheme.StateRecurrent }
