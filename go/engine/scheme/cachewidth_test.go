// SPDX-Licence-Identifier: EUPL-1.2

package scheme_test

import (
	"fmt"
	"testing"

	"dappco.re/go/inference/engine/scheme"
)

// TestCacheWidth_KVModesCarryWidth_Good asserts every builtin KV-cache mode
// resolves to a scheme that satisfies CacheWidth and reports the exact
// per-element byte rational a memory planner sizes a cache from. These are the
// neutral wire-format facts the planner's old hard-coded byte table duplicated;
// the registry is now their single home.
func TestCacheWidth_KVModesCarryWidth_Good(t *testing.T) {
	cases := []struct {
		mode     string
		num, den uint64
		roundUp  bool
	}{
		{"default", 2, 1, false},
		{"fp16", 2, 1, false},
		{"q8", 1, 1, false},
		{"k-q8-v-q4", 3, 4, false},
		{"paged", 2, 1, false},
		{"fixed", 2, 1, false},
		{"turboquant", 7, 16, true},
	}
	for _, c := range cases {
		cacheScheme, ok := scheme.CacheFor(c.mode)
		if !ok {
			t.Fatalf("CacheFor(%q) did not resolve", c.mode)
		}
		width, ok := cacheScheme.(scheme.CacheWidth)
		if !ok {
			t.Fatalf("scheme %q (%T) does not satisfy CacheWidth — its builtin width did not register", c.mode, cacheScheme)
		}
		num, den, roundUp := width.KVBytesPerElement()
		if num != c.num || den != c.den || roundUp != c.roundUp {
			t.Fatalf("KVBytesPerElement(%q) = %d/%d roundUp=%v, want %d/%d roundUp=%v", c.mode, num, den, roundUp, c.num, c.den, c.roundUp)
		}
	}
}

// TestCacheWidth_RecurrentMissesWidth_Bad pins the recurrent path: a
// recurrent-state holder is a registered cache scheme but holds no growing KV,
// so it carries no width and the CacheWidth probe simply misses (type assertion
// false) — the same miss fp16/paged give kv.CacheProvider in #261. A planner
// reading widths therefore never mistakes a recurrent cache for a KV cache.
func TestCacheWidth_RecurrentMissesWidth_Bad(t *testing.T) {
	cacheScheme, ok := scheme.CacheFor("recurrent")
	if !ok {
		t.Fatal("CacheFor(recurrent) did not resolve — the builtin is missing")
	}
	if cacheScheme.Serves() != scheme.StateRecurrent {
		t.Fatalf("recurrent Serves() = %v, want StateRecurrent", cacheScheme.Serves())
	}
	if _, ok := cacheScheme.(scheme.CacheWidth); ok {
		t.Fatal("recurrent scheme satisfies CacheWidth, want the probe to miss (no KV width)")
	}
}

// TestCacheWidth_UnknownModeHasNoScheme_Ugly is the near-miss: a mode the
// registry never registered does not resolve at all, so there is no scheme to
// probe for a width — distinct from recurrent, which resolves but carries none.
func TestCacheWidth_UnknownModeHasNoScheme_Ugly(t *testing.T) {
	if _, ok := scheme.CacheFor("k-q8-v-q2"); ok {
		t.Fatal("CacheFor(k-q8-v-q2) resolved, want no scheme for an unregistered mode")
	}
}

// A KV format's storage cost plugs in through the scheme registry: resolve the
// mode with scheme.CacheFor, then probe for scheme.CacheWidth to read its
// per-element byte ratio. Sizing a KV cache is a registry lookup — never a
// per-mode byte table in the planner.
func ExampleCacheWidth() {
	cacheScheme, _ := scheme.CacheFor("turboquant")
	if width, ok := cacheScheme.(scheme.CacheWidth); ok {
		num, den, roundUp := width.KVBytesPerElement()
		fmt.Printf("turboquant KV width: %d/%d rounded-up=%v\n", num, den, roundUp)
	}
	// Output: turboquant KV width: 7/16 rounded-up=true
}
