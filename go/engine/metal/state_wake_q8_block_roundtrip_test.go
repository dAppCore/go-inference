// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"bytes"
	"testing"

	"dappco.re/go/inference/kv"
)

// TestStateWakeQ8BlockRoundTrip_StoreBytes pins the bit-exactness of the q8 KV
// BLOCK round trip AT THE STORE LEVEL (#1846 block lane): a q8 session's
// RangeKVBlocks blocks carry the store's raw int8 codes + f32 scales under
// kv.KVNativeDTypeQ8, and RestoreKVBlocks lands them into a fresh q8 session
// reproducing the codes + scales byte-for-byte — no dequantise→requantise pass.
// The block lane double-quantised (q8 store → bf16 block → q8 mirror flush) until
// this fix, perturbing every restored prefix row and flipping downstream tokens.
// Comparing the raw store bytes (not a dequantised bf16 view) is the whole point:
// a lossy round trip can match dequantised views while the underlying codes
// differ. A second block covers [4,8) so the per-block window offset is exercised.
func TestStateWakeQ8BlockRoundTrip_StoreBytes(t *testing.T) {
	requireNativeRuntime(t)
	kvQ8ICBForTest = true
	t.Cleanup(func() { kvQ8ICBForTest = false })

	prompt := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	const blockSize, maxNew = 4, 8 // two full blocks — the second lands at token 4

	src := newKVQ8ICBFixtureLen(t, 256)
	defer src.Close()
	if !src.state.icb.hasKVQ8() {
		t.Fatal("fixture did not arm the q8 KV store — the raw block round trip was never exercised")
	}
	if err := src.PrefillTokens(prompt); err != nil {
		t.Fatalf("src PrefillTokens: %v", err)
	}

	// Capture blocks on the NATIVE (RawKVOnly) lane — the sleep lane that carries
	// the raw q8 block (SaveKVBlocksToState sets RawKVOnly for EncodingNative).
	var blocks []kv.Block
	if err := src.RangeKVBlocks(blockSize, kv.CaptureOptions{RawKVOnly: true}, func(b kv.Block) (bool, error) {
		blocks = append(blocks, b)
		return true, nil
	}); err != nil {
		t.Fatalf("RangeKVBlocks: %v", err)
	}
	if len(blocks) != 2 {
		t.Fatalf("RangeKVBlocks yielded %d blocks, want 2", len(blocks))
	}

	// Every armed layer must have emitted the raw q8 block on every block, not the
	// bf16 slab — otherwise the round trip below is testing the wrong path.
	q8Layers := 0
	for li := range src.state.icb.kvQ8.enabled {
		if !src.state.icb.kvQ8.on(li) {
			continue
		}
		q8Layers++
		for _, b := range blocks {
			layer, ok := nativeKVSnapshotLayer(b.Snapshot, li)
			if !ok {
				t.Fatalf("block %d layer %d: missing from snapshot", b.Index, li)
			}
			if !nativeKVLayerIsQ8Native(layer) {
				t.Fatalf("block %d layer %d: dtype key=%q value=%q, want %q (raw q8 block not emitted)",
					b.Index, li, layer.KeyDType, layer.ValueDType, kv.KVNativeDTypeQ8)
			}
		}
	}
	if q8Layers == 0 {
		t.Fatal("no q8 layers armed — the raw block round trip was never exercised")
	}

	// Restore through the serialisable block path: nativeStateSource stays nil, so
	// RestoreKVBlocks reassembles the yielded blocks (the disk/reassembled lane)
	// rather than borrowing the source's live state (the in-process fast path).
	source := KVBlockSource{
		TokenCount:   src.pos,
		PrefixTokens: src.pos,
		BlockCount:   len(blocks),
		CachedIDs:    append([]int32(nil), prompt...),
		Load:         func(i int) (kv.Block, error) { return blocks[i], nil },
	}
	dst := newKVQ8ICBFixtureLen(t, 256)
	defer dst.Close()
	if err := dst.RestoreKVBlocks(source); err != nil {
		t.Fatalf("RestoreKVBlocks: %v", err)
	}
	if dst.pos != src.pos {
		t.Fatalf("restored position %d, want %d", dst.pos, src.pos)
	}
	// The block wake path must auto-arm canonical landing so a woken append does
	// not reintroduce the q8 wobble (#1846).
	if !dst.reuseCanonicalLanding {
		t.Fatal("RestoreKVBlocks into a q8 session must auto-arm canonical wake landing")
	}

	// The store-level proof: re-read the woken store's raw int8 codes + f32 scales
	// and require them byte-equal to the source's live store (codes AND scales).
	for li := range dst.state.icb.kvQ8.enabled {
		if !dst.state.icb.kvQ8.on(li) {
			continue
		}
		srcK, srcV, err := src.state.icb.captureQ8LayerRaw(li, 0, src.pos)
		if err != nil {
			t.Fatalf("layer %d: re-read source store: %v", li, err)
		}
		dstK, dstV, err := dst.state.icb.captureQ8LayerRaw(li, 0, dst.pos)
		if err != nil {
			t.Fatalf("layer %d: re-read woken store: %v", li, err)
		}
		if !bytes.Equal(dstK, srcK) {
			t.Fatalf("layer %d: woken K store bytes diverge from source (codes+scales)", li)
		}
		if !bytes.Equal(dstV, srcV) {
			t.Fatalf("layer %d: woken V store bytes diverge from source (codes+scales)", li)
		}
	}

	// A bit-exact restore leaves the cache identical to a fresh prefill, so a
	// generate from the woken cache matches an unbroken session token-for-token.
	got, err := dst.GenerateFromCache(maxNew, -1)
	if err != nil {
		t.Fatalf("woken GenerateFromCache: %v", err)
	}
	cold := newKVQ8ICBFixtureLen(t, 256)
	defer cold.Close()
	want, err := cold.Generate(prompt, maxNew, -1)
	if err != nil {
		t.Fatalf("cold Generate: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("woken generated %d tokens, want %d (%v vs %v)", len(got), len(want), got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("woken diverges from unbroken at token %d: got %v want %v", i, got, want)
		}
	}
}
