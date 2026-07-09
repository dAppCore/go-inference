// SPDX-Licence-Identifier: EUPL-1.2

package scheme

import "testing"

// The existing engine pieces are present as catalogue entry-one.
func TestBuiltinsRegistered_Good(t *testing.T) {
	if _, ok := MixerFor("softmax-hybrid"); !ok {
		t.Error("softmax-hybrid mixer not registered")
	}
	for _, mode := range []string{"default", "fp16", "q8", "k-q8-v-q4", "paged", "fixed", "turboquant", "recurrent"} {
		if _, ok := CacheFor(mode); !ok {
			t.Errorf("cache scheme %q not registered", mode)
		}
	}
	if _, ok := QuantFor("affine"); !ok {
		t.Error("affine quant scheme not registered")
	}
	for _, name := range []string{"bfloat16", "float32"} {
		if _, ok := DTypeFor(name); !ok {
			t.Errorf("activation dtype %q not registered", name)
		}
	}
}

// The mixer-owns-state contract: a cache may serve a mixer only when their
// state kinds agree. A softmax mixer pairs with a KV cache, never a recurrent
// holder; an SSM mixer is the reverse.
func TestMixerOwnsStateContract_Good(t *testing.T) {
	softmax, _ := MixerFor("softmax-hybrid") // StateKVCache
	kv, _ := CacheFor("q8")                  // serves KVCache
	recur, _ := CacheFor("recurrent")        // serves Recurrent

	if !Compatible(softmax, kv) {
		t.Error("softmax mixer + KV cache should be compatible")
	}
	if Compatible(softmax, recur) {
		t.Error("softmax mixer + recurrent cache must NOT be compatible (mixer owns state)")
	}

	// A hypothetical SSM mixer declaring recurrent state pairs the other way.
	ssm := mixerInfo{"mamba2", StateRecurrent}
	if Compatible(ssm, kv) {
		t.Error("recurrent mixer + KV cache must NOT be compatible")
	}
	if !Compatible(ssm, recur) {
		t.Error("recurrent mixer + recurrent cache should be compatible")
	}
}

// Registering a new scheme is one call, then it resolves — the population path.
func TestRegisterAndResolve_Good(t *testing.T) {
	RegisterQuant(quantInfo{"q4_0", 4})
	q, ok := QuantFor("q4_0")
	if !ok {
		t.Fatal("q4_0 did not resolve after registration")
	}
	if q.Bits() != 4 {
		t.Errorf("q4_0 bits = %d, want 4", q.Bits())
	}
}

// The activation dtype scheme: bf16 is a registered scheme (not a hardcoded op
// suffix), resolving with its element size, and the exported instance matches the
// registry — the same registry shape weights use, now for the compute dtype.
func TestDType_Good(t *testing.T) {
	bf16, ok := DTypeFor("bfloat16")
	if !ok {
		t.Fatal("bfloat16 dtype not registered")
	}
	if bf16.Name() != BFloat16.Name() || bf16.Bytes() != BFloat16.Bytes() {
		t.Errorf("DTypeFor(bfloat16) = %q/%d, exported BFloat16 = %q/%d",
			bf16.Name(), bf16.Bytes(), BFloat16.Name(), BFloat16.Bytes())
	}
	if BFloat16.Bytes() != 2 || Float32.Bytes() != 4 {
		t.Errorf("element sizes: bf16=%d f32=%d, want 2/4", BFloat16.Bytes(), Float32.Bytes())
	}
	if _, ok := DTypeFor("float64"); ok {
		t.Error("unregistered dtype float64 must not resolve")
	}
	var seen bool
	for _, n := range DTypeNames() {
		if n == "bfloat16" {
			seen = true
		}
	}
	if !seen {
		t.Errorf("DTypeNames() = %v, missing bfloat16", DTypeNames())
	}
}

// Unknown kinds resolve to (nil,false), never a panic — the engine reports a
// clean "unsupported scheme" rather than miscomputing.
func TestUnknownKind_Bad(t *testing.T) {
	if _, ok := MixerFor("does-not-exist"); ok {
		t.Error("unknown mixer kind should not resolve")
	}
	if Compatible(nil, nil) {
		t.Error("nil mixer/cache must be incompatible")
	}
}
