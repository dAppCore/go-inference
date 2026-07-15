// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"bytes"
	"context"
	"testing"

	"dappco.re/go/inference/kv"
)

// TestStateWakeQ8SnapshotRoundTrip_StoreBytes pins the bit-exactness of the q8
// KV snapshot round trip AT THE STORE LEVEL (#1846 state lane): a q8 session's
// captured snapshot carries the store's raw int8 codes + f32 scales under
// kv.KVNativeDTypeQ8, and restoring it into a fresh q8 session reproduces the
// int8 codes + scales byte-for-byte — no dequantise→requantise pass, which is
// what perturbed every prefix row and flipped downstream tokens. Comparing the
// raw store bytes (not a dequantised bf16 view) is the whole point: a lossy
// round trip can match dequantised views while the underlying codes differ.
func TestStateWakeQ8SnapshotRoundTrip_StoreBytes(t *testing.T) {
	requireNativeRuntime(t)
	kvQ8ICBForTest = true
	t.Cleanup(func() { kvQ8ICBForTest = false })

	prompt := []int32{1, 2, 3, 4, 5, 6, 7, 8}

	src := newKVQ8ICBFixtureLen(t, 256)
	defer src.Close()
	if !src.state.icb.hasKVQ8() {
		t.Fatal("fixture did not arm the q8 KV store — the raw round trip was never exercised")
	}
	if err := src.PrefillTokens(prompt); err != nil {
		t.Fatalf("src PrefillTokens: %v", err)
	}

	snap, err := src.CaptureKVWithOptions(kv.CaptureOptions{})
	if err != nil {
		t.Fatalf("CaptureKVWithOptions: %v", err)
	}

	// Capture must have taken the raw q8 path for every armed layer.
	q8Layers := 0
	for li := range src.state.icb.kvQ8.enabled {
		if !src.state.icb.kvQ8.on(li) {
			continue
		}
		q8Layers++
		layer, ok := nativeKVSnapshotLayer(snap, li)
		if !ok {
			t.Fatalf("layer %d: missing from snapshot", li)
		}
		if !nativeKVLayerIsQ8Native(layer) {
			t.Fatalf("layer %d: captured dtype key=%q value=%q, want %q (raw q8 path not taken)",
				li, layer.KeyDType, layer.ValueDType, kv.KVNativeDTypeQ8)
		}
	}
	if q8Layers == 0 {
		t.Fatal("no q8 layers armed — the raw round trip was never exercised")
	}

	dst := newKVQ8ICBFixtureLen(t, 256)
	defer dst.Close()
	if err := dst.RestoreFromKV(context.Background(), snap); err != nil {
		t.Fatalf("RestoreFromKV: %v", err)
	}
	if dst.pos != src.pos {
		t.Fatalf("restored position %d, want %d", dst.pos, src.pos)
	}
	// Deliverable 3: restore auto-arms the woken session's canonical landing.
	if !dst.reuseCanonicalLanding {
		t.Fatal("restore into a q8 session must auto-arm canonical wake landing")
	}

	// The store-level proof: re-read the woken store's raw int8 codes + scales
	// and require them byte-equal to what the source captured.
	for li := range dst.state.icb.kvQ8.enabled {
		if !dst.state.icb.kvQ8.on(li) {
			continue
		}
		layer, ok := nativeKVSnapshotLayer(snap, li)
		if !ok {
			t.Fatalf("layer %d: missing from snapshot", li)
		}
		gotK, gotV, err := dst.state.icb.captureQ8LayerRaw(li, 0, dst.pos)
		if err != nil {
			t.Fatalf("layer %d: re-read woken store: %v", li, err)
		}
		if !bytes.Equal(gotK, layer.KeyBytes) {
			t.Fatalf("layer %d: woken K store bytes diverge from source (%d vs %d bytes)",
				li, len(gotK), len(layer.KeyBytes))
		}
		if !bytes.Equal(gotV, layer.ValueBytes) {
			t.Fatalf("layer %d: woken V store bytes diverge from source (%d vs %d bytes)",
				li, len(gotV), len(layer.ValueBytes))
		}
	}
}
