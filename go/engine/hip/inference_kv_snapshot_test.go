// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

// inference_kv_snapshot_test.go is the HARDWARE-FREE receipt for the structural
// reconcile (hip host decode state <-> kv.Snapshot). It exercises the pure
// float32 converter with synthetic per-layer K/V — no HIP device, no GPU — so
// it runs on any linux/amd64 host (Snider's box or a linux CI), not just on the
// AMD hardware the full engine.Session parity test needs. It proves the one
// property the converter must have: capture -> snapshot -> restore reproduces
// hip's per-layer float32 K/V exactly.
package hip

import (
	"testing"

	"dappco.re/go/inference/kv"
)

// hipKVSnapshotTestConfig builds a forward config with the given per-layer
// HeadDim (QueryHeads/KeyHeads are informational for the converter; the
// retained KV is single-head).
func hipKVSnapshotTestConfig(headDims ...int) hipGemma4Q4ForwardConfig {
	layers := make([]hipGemma4Q4Layer0Config, len(headDims))
	for index, headDim := range headDims {
		layers[index] = hipGemma4Q4Layer0Config{HeadDim: headDim, QueryHeads: 2, KeyHeads: 1}
	}
	return hipGemma4Q4ForwardConfig{Layers: layers}
}

// hipKVSnapshotTestState builds a host decode state whose per-layer Keys/Values
// are distinct ramps, so an axis/stride swap in the converter would show up.
func hipKVSnapshotTestState(headDim, tokens, layers int) hipGemma4Q4DecodeState {
	state := hipGemma4Q4DecodeState{Layers: make([]hipGemma4Q4LayerKVState, layers)}
	for layer := range state.Layers {
		keys := make([]float32, tokens*headDim)
		values := make([]float32, tokens*headDim)
		for i := range keys {
			keys[i] = float32(layer*1000 + i)
			values[i] = float32(layer*1000+i) + 0.5
		}
		state.Layers[layer] = hipGemma4Q4LayerKVState{Keys: keys, Values: values}
	}
	return state
}

func hipKVStatesEqual(a, b hipGemma4Q4DecodeState) bool {
	if len(a.Layers) != len(b.Layers) {
		return false
	}
	for layer := range a.Layers {
		if !hipKVFloat32SlicesEqual(a.Layers[layer].Keys, b.Layers[layer].Keys) ||
			!hipKVFloat32SlicesEqual(a.Layers[layer].Values, b.Layers[layer].Values) {
			return false
		}
	}
	return true
}

func hipKVFloat32SlicesEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestInferenceKVSnapshot_Roundtrip_Good is the receipt: a multi-layer host KV
// state survives capture -> snapshot -> restore byte-for-byte (exact float32).
func TestInferenceKVSnapshot_Roundtrip_Good(t *testing.T) {
	const headDim, tokens, layers = 4, 3, 2
	cfg := hipKVSnapshotTestConfig(headDim, headDim)
	host := hipKVSnapshotTestState(headDim, tokens, layers)
	sequence := []int32{5, 9, 2}

	snapshot, err := hipDecodeStateToSnapshot(host, cfg, sequence, sequence[2:], kv.CaptureOptions{})
	if err != nil {
		t.Fatalf("hipDecodeStateToSnapshot: %v", err)
	}
	if snapshot.NumLayers != layers || snapshot.HeadDim != headDim || snapshot.SeqLen != tokens {
		t.Fatalf("snapshot geometry = layers %d headDim %d seqLen %d, want %d %d %d",
			snapshot.NumLayers, snapshot.HeadDim, snapshot.SeqLen, layers, headDim, tokens)
	}
	if snapshot.NumHeads != 1 {
		t.Fatalf("snapshot NumHeads = %d, want 1 (single KV row per token)", snapshot.NumHeads)
	}
	if len(snapshot.Tokens) != len(sequence) {
		t.Fatalf("snapshot Tokens len = %d, want %d", len(snapshot.Tokens), len(sequence))
	}

	restored, err := hipSnapshotToDecodeState(snapshot, cfg)
	if err != nil {
		t.Fatalf("hipSnapshotToDecodeState: %v", err)
	}
	if !hipKVStatesEqual(host, restored) {
		t.Fatal("roundtrip host decode state differs from the original (lossy converter)")
	}
}

// TestInferenceKVSnapshot_Roundtrip_RawKVOnly proves the KeyBytes image path:
// with RawKVOnly the per-head float32 slices are dropped, and restore must
// still reproduce the state from the little-endian bytes alone.
func TestInferenceKVSnapshot_Roundtrip_RawKVOnly(t *testing.T) {
	const headDim, tokens, layers = 6, 4, 3
	cfg := hipKVSnapshotTestConfig(headDim, headDim, headDim)
	host := hipKVSnapshotTestState(headDim, tokens, layers)

	snapshot, err := hipDecodeStateToSnapshot(host, cfg, nil, nil, kv.CaptureOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("hipDecodeStateToSnapshot(RawKVOnly): %v", err)
	}
	for _, layer := range snapshot.Layers {
		if len(layer.Heads) != 0 {
			t.Fatal("RawKVOnly snapshot must not carry per-head float32 slices")
		}
	}
	restored, err := hipSnapshotToDecodeState(snapshot, cfg)
	if err != nil {
		t.Fatalf("hipSnapshotToDecodeState(RawKVOnly): %v", err)
	}
	if !hipKVStatesEqual(host, restored) {
		t.Fatal("RawKVOnly roundtrip host decode state differs from the original")
	}
}

// TestInferenceKVSnapshot_Capture_Bad rejects a host state whose layer count
// does not match the forward config.
func TestInferenceKVSnapshot_Capture_Bad(t *testing.T) {
	cfg := hipKVSnapshotTestConfig(4, 4)
	host := hipKVSnapshotTestState(4, 2, 1) // one layer, cfg has two
	if _, err := hipDecodeStateToSnapshot(host, cfg, nil, nil, kv.CaptureOptions{}); err == nil {
		t.Fatal("expected an error for a layer-count mismatch, got nil")
	}
}

// TestInferenceKVSnapshot_Restore_Ugly rejects a nil snapshot, a foreign
// architecture, and a K/V length that does not align with HeadDim.
func TestInferenceKVSnapshot_Restore_Ugly(t *testing.T) {
	cfg := hipKVSnapshotTestConfig(4)
	if _, err := hipSnapshotToDecodeState(nil, cfg); err == nil {
		t.Fatal("expected an error for a nil snapshot, got nil")
	}
	foreign := &kv.Snapshot{Architecture: "metal", Layers: []kv.LayerSnapshot{{}}}
	if _, err := hipSnapshotToDecodeState(foreign, cfg); err == nil {
		t.Fatal("expected an error for a foreign architecture, got nil")
	}
	misaligned := &kv.Snapshot{
		Architecture: hipKVSnapshotArchitecture,
		Layers: []kv.LayerSnapshot{{
			Layer:      0,
			KeyDType:   hipKVSnapshotFloat32DType,
			KeyBytes:   hipFloat32SliceToLEBytes([]float32{1, 2, 3}), // 3 not a multiple of HeadDim 4
			ValueBytes: hipFloat32SliceToLEBytes([]float32{1, 2, 3}),
		}},
	}
	if _, err := hipSnapshotToDecodeState(misaligned, cfg); err == nil {
		t.Fatal("expected an error for a HeadDim-misaligned layer, got nil")
	}
}
