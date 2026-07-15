// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"os"
	"testing"
)

// TestICBSessionDefersLBKVCaches pins the #367 cache-dedupe contract: a
// recorded-ICB session decodes, prefills, snapshots and restores exclusively
// over the replay's own caches, so the linear lb KV set must stay UNALLOCATED
// (kvCacheBytes recorded, buffers nil) through the whole session lifecycle —
// eager allocation here doubled the entire KV footprint (1.9GB idle on
// e2b@128K, 17.2GB on 31B@256K) for buffers no lane ever read.
func TestICBSessionDefersLBKVCaches(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	sess := newKVQ8ICBFixture(t)
	if sess.state.icb == nil {
		t.Fatal("fixture must record an ICB")
	}
	owners := 0
	assertDeferred := func(stage string) {
		t.Helper()
		for li := range sess.state.lb {
			lb := &sess.state.lb[li]
			if !sess.state.specs[li].OwnsCache() {
				continue
			}
			if lb.kvCacheBytes == 0 {
				t.Fatalf("%s: layer %d owner recorded no kvCacheBytes", stage, li)
			}
			if lb.kCache != nil || lb.vCache != nil {
				t.Fatalf("%s: layer %d materialised the lb KV cache on an ICB session", stage, li)
			}
		}
	}
	for li := range sess.state.specs {
		if sess.state.specs[li].OwnsCache() {
			owners++
		}
	}
	if owners == 0 {
		t.Fatal("fixture has no cache-owning layers")
	}
	assertDeferred("post-build")

	// prompt-scale prefill (the batched fold / GEMM lane) + incremental decode.
	ids := make([]int32, 300)
	for i := range ids {
		ids[i] = int32(1 + i%50)
	}
	if err := sess.PrefillTokens(ids); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	assertDeferred("post-prefill")
	for _, id := range []int32{7, 3, 9} {
		if _, err := sess.stepID(id); err != nil {
			t.Fatalf("stepID: %v", err)
		}
	}
	assertDeferred("post-decode")

	// -state sleep/wake: serialize reads the ICB caches (q8 layers via the
	// snapshot mirrors), restore writes them back — lb stays out of the loop.
	blob, err := sess.SerializeState()
	if err != nil {
		t.Fatalf("SerializeState: %v", err)
	}
	assertDeferred("post-serialize")
	if err := sess.RestoreState(blob); err != nil {
		t.Fatalf("RestoreState: %v", err)
	}
	assertDeferred("post-restore")
	if _, err := sess.stepID(5); err != nil {
		t.Fatalf("stepID after restore: %v", err)
	}
	assertDeferred("post-restore-decode")
}
