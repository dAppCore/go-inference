// SPDX-Licence-Identifier: EUPL-1.2

package scheme

import "slices"

import "testing"

// StateKind.String renders every state for logs and error messages: the two
// named kinds plus the stateless default. This is the human-readable face the
// engine prints when it reports a mixer/cache pairing.
func TestStateKindString_Good(t *testing.T) {
	cases := []struct {
		kind StateKind
		want string
	}{
		{StateNone, "none"},
		{StateKVCache, "kv-cache"},
		{StateRecurrent, "recurrent"},
		{StateKind(99), "none"}, // any unnamed value falls through to the default
	}
	for _, tc := range cases {
		if got := tc.kind.String(); got != tc.want {
			t.Errorf("StateKind(%d).String() = %q, want %q", tc.kind, got, tc.want)
		}
	}
}

// A cache mode the engine does not know resolves to (nil,false) — the miss
// branch of CacheFor — so the engine reports a clean "unsupported KV cache"
// rather than dereferencing a nil scheme.
func TestCacheForMiss_Bad(t *testing.T) {
	c, ok := CacheFor("no-such-cache-mode")
	if ok {
		t.Error("unknown cache mode should not resolve")
	}
	if c != nil {
		t.Errorf("missed CacheFor should return nil scheme, got %v", c)
	}
}

// A quant kind the engine does not know resolves to (nil,false) — the miss
// branch of QuantFor — mirroring the cache miss path.
func TestQuantForMiss_Bad(t *testing.T) {
	q, ok := QuantFor("no-such-quant-kind")
	if ok {
		t.Error("unknown quant kind should not resolve")
	}
	if q != nil {
		t.Errorf("missed QuantFor should return nil scheme, got %v", q)
	}
}

// The catalogue listers are the engine's "what can I load" view — each returns
// the registered names in registration order. The registries are init-seeded
// package globals other tests also write to, so assert the builtins are PRESENT
// rather than equal to a fixed slice (registration order/length is shared state).
func TestCatalogueListers_Good(t *testing.T) {
	contains := func(names []string, want string) bool {
		return slices.Contains(names, want)
	}

	mixers := MixerKinds()
	if !contains(mixers, "softmax-hybrid") {
		t.Errorf("MixerKinds() = %v, missing softmax-hybrid", mixers)
	}

	modes := CacheModes()
	for _, want := range []string{"default", "q8", "recurrent"} {
		if !contains(modes, want) {
			t.Errorf("CacheModes() = %v, missing %q", modes, want)
		}
	}

	quants := QuantKinds()
	if !contains(quants, "affine") {
		t.Errorf("QuantKinds() = %v, missing affine", quants)
	}
}
