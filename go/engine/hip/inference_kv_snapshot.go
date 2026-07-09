// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

// inference_kv_snapshot.go is the structural reconcile between hip's retained
// Gemma4-Q4 host decode state and go-inference's portable inference/kv.Snapshot
// contract — the piece that lets engine/hip satisfy engine.Session's
// CaptureKVWithOptions / RestoreFromKV.
//
// # The two layouts (why this is a reconcile, not a rename)
//
// hip stores its retained KV per LAYER as a flat float32 vector, HeadDim-wide
// per token — one KV row per token per layer (keyWidth == valueWidth ==
// HeadDim, KeyHeads == 1 / MQA). This is exactly what the native decode driver
// appends (hip_gemma4_q4_kv.go: len(next.Keys) == len(prev.Keys) + HeadDim per
// token) and validates (hip_gemma4_q4_layer.go hipGemma4Q4ValidateKVState:
// len(keys) % HeadDim == 0). The host boundary is float32 — hipGemma4Q4Device
// DecodeState.HostState restores the device cache to float32 via
// rocmKVCache.Restore, so no dequant happens here.
//
// kv.Snapshot is organised the transformer way — per [layer][head] tensors with
// KeyBytes/KeyShape/KeyDType (and optional per-head float32 slices). engine/
// metal's ArchSession produces exactly that from its multi-head native cache.
//
// # The mapping (layout assumed — Snider's parity test proves it end-to-end)
//
// Because hip's retained cache is one HeadDim-wide row per token per layer, the
// [layer][head] mapping is a single KV head: NumHeads = 1, HeadDim =
// cfg.Layers[i].HeadDim, and layer i's flat float32 Keys/Values are the token
// rows in order (token t = Keys[t*HeadDim : (t+1)*HeadDim]). For a single KV
// head the token-row order and the layer-slab order coincide, so no reshuffle
// is needed — KeyBytes is the little-endian float32 image of Keys directly.
// The roundtrip is therefore lossless by construction (float32 in, float32
// out, no head reinterpretation); the HIP-gated parity test in
// inference_conformance_test.go is the receipt that proves it against a real
// device-produced cache.
package hip

import (
	"encoding/binary"
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/kv"
)

// hipKVSnapshotArchitecture tags snapshots captured from the retained Gemma4-Q4
// engine so a restore can reject a snapshot from a different engine family.
const hipKVSnapshotArchitecture = "gemma4-q4"

// hipKVSnapshotFloat32DType is the K/V element dtype at hip's host boundary —
// HostState always returns float32 (the device FP16 cache is widened on copy).
const hipKVSnapshotFloat32DType = "float32"

// hipDecodeStateToSnapshot converts hip's retained Gemma4-Q4 host decode state
// into a portable kv.Snapshot. host is the per-layer float32 K/V read back from
// the device (deviceState.HostState); cfg supplies each layer's HeadDim (the KV
// row width); tokens is the full prompt+generated sequence held by the session
// (Snapshot.Tokens) and generated the generated-only suffix (Snapshot.Generated).
// opts.RawKVOnly skips the per-head float32 side slices, keeping only the
// KeyBytes/ValueBytes image (the restore reads either).
func hipDecodeStateToSnapshot(host hipGemma4Q4DecodeState, cfg hipGemma4Q4ForwardConfig, tokens, generated []int32, opts kv.CaptureOptions) (*kv.Snapshot, error) {
	if len(host.Layers) != len(cfg.Layers) {
		return nil, core.E("rocm.hip.KVSnapshot.Capture", "host decode state layer count must match forward config", nil)
	}
	if err := host.validate(cfg); err != nil {
		return nil, core.E("rocm.hip.KVSnapshot.Capture", "host decode state is invalid", err)
	}
	headDim := 0
	if len(cfg.Layers) > 0 {
		headDim = cfg.Layers[0].HeadDim
	}
	layers := make([]kv.LayerSnapshot, len(host.Layers))
	seqLen := 0
	for index, layerState := range host.Layers {
		layerHeadDim := cfg.Layers[index].HeadDim
		if layerHeadDim <= 0 {
			return nil, core.E("rocm.hip.KVSnapshot.Capture", "layer HeadDim must be positive", nil)
		}
		if len(layerState.Keys)%layerHeadDim != 0 || len(layerState.Values) != len(layerState.Keys) {
			return nil, core.E("rocm.hip.KVSnapshot.Capture", "layer K/V lengths must align with HeadDim", nil)
		}
		layerTokens := len(layerState.Keys) / layerHeadDim
		if layerTokens > seqLen {
			seqLen = layerTokens
		}
		// Shape [batch=1, kvHeads=1, tokens, headDim] — the single-KV-head form
		// (engine/metal uses [1, kvHeads, tokens, headDim]; hip's kvHeads is 1).
		shape := []int32{1, 1, int32(layerTokens), int32(layerHeadDim)}
		layer := kv.LayerSnapshot{
			Layer:      index,
			KeyDType:   hipKVSnapshotFloat32DType,
			KeyBytes:   hipFloat32SliceToLEBytes(layerState.Keys),
			KeyShape:   shape,
			ValueDType: hipKVSnapshotFloat32DType,
			ValueBytes: hipFloat32SliceToLEBytes(layerState.Values),
			ValueShape: append([]int32(nil), shape...),
		}
		if !opts.RawKVOnly {
			layer.Heads = []kv.HeadSnapshot{{
				Key:        append([]float32(nil), layerState.Keys...),
				KeyDType:   hipKVSnapshotFloat32DType,
				Value:      append([]float32(nil), layerState.Values...),
				ValueDType: hipKVSnapshotFloat32DType,
			}}
		}
		layers[index] = layer
	}
	return &kv.Snapshot{
		Version:       kv.SnapshotVersion,
		Architecture:  hipKVSnapshotArchitecture,
		Tokens:        append([]int32(nil), tokens...),
		Generated:     append([]int32(nil), generated...),
		NumLayers:     len(host.Layers),
		NumHeads:      1,
		SeqLen:        seqLen,
		HeadDim:       headDim,
		NumQueryHeads: hipForwardConfigQueryHeads(cfg),
		Layers:        layers,
	}, nil
}

// hipSnapshotToDecodeState is the inverse: it rebuilds hip's per-layer float32
// host decode state from a kv.Snapshot so hipMirrorGemma4Q4DecodeState can push
// it back onto the device (engine.Session.RestoreFromKV). It reads the per-head
// float32 slices when present (exact) and falls back to the KeyBytes/ValueBytes
// float32 image otherwise, then validates the reconstructed state against cfg.
func hipSnapshotToDecodeState(snapshot *kv.Snapshot, cfg hipGemma4Q4ForwardConfig) (hipGemma4Q4DecodeState, error) {
	if snapshot == nil {
		return hipGemma4Q4DecodeState{}, core.E("rocm.hip.KVSnapshot.Restore", "snapshot is nil", nil)
	}
	if snapshot.Architecture != "" && snapshot.Architecture != hipKVSnapshotArchitecture {
		return hipGemma4Q4DecodeState{}, core.E("rocm.hip.KVSnapshot.Restore", "snapshot architecture is not gemma4-q4", nil)
	}
	if len(snapshot.Layers) != len(cfg.Layers) {
		return hipGemma4Q4DecodeState{}, core.E("rocm.hip.KVSnapshot.Restore", "snapshot layer count must match forward config", nil)
	}
	layers := make([]hipGemma4Q4LayerKVState, len(snapshot.Layers))
	for index, layerSnapshot := range snapshot.Layers {
		layerHeadDim := cfg.Layers[index].HeadDim
		if layerHeadDim <= 0 {
			return hipGemma4Q4DecodeState{}, core.E("rocm.hip.KVSnapshot.Restore", "layer HeadDim must be positive", nil)
		}
		keys, values := hipLayerSnapshotKV(layerSnapshot)
		if len(keys)%layerHeadDim != 0 || len(values) != len(keys) {
			return hipGemma4Q4DecodeState{}, core.E("rocm.hip.KVSnapshot.Restore", "snapshot layer K/V lengths must align with HeadDim", nil)
		}
		layers[index] = hipGemma4Q4LayerKVState{Keys: keys, Values: values}
	}
	host := hipGemma4Q4DecodeState{Layers: layers}
	if err := host.validate(cfg); err != nil {
		return hipGemma4Q4DecodeState{}, core.E("rocm.hip.KVSnapshot.Restore", "reconstructed host decode state is invalid", err)
	}
	return host, nil
}

// hipLayerSnapshotKV reads one layer's float32 K/V, preferring the exact per-head
// slices and falling back to the little-endian KeyBytes/ValueBytes image.
func hipLayerSnapshotKV(layer kv.LayerSnapshot) (keys, values []float32) {
	if len(layer.Heads) == 1 && len(layer.Heads[0].Key) > 0 {
		keys = append([]float32(nil), layer.Heads[0].Key...)
		values = append([]float32(nil), layer.Heads[0].Value...)
		return keys, values
	}
	return hipLEBytesToFloat32Slice(layer.KeyBytes), hipLEBytesToFloat32Slice(layer.ValueBytes)
}

// hipForwardConfigQueryHeads reports the layer-0 query-head count (informational
// NumQueryHeads on the snapshot — the retained KV itself is single-head).
func hipForwardConfigQueryHeads(cfg hipGemma4Q4ForwardConfig) int {
	if len(cfg.Layers) == 0 {
		return 0
	}
	return cfg.Layers[0].QueryHeads
}

// hipFloat32SliceToLEBytes packs a float32 slice as little-endian IEEE-754
// bytes (the KeyBytes/ValueBytes image; mirrors kv_cache_raw.go's encoding).
func hipFloat32SliceToLEBytes(values []float32) []byte {
	out := make([]byte, len(values)*4)
	for index, value := range values {
		binary.LittleEndian.PutUint32(out[index*4:], math.Float32bits(value))
	}
	return out
}

// hipLEBytesToFloat32Slice unpacks a little-endian IEEE-754 float32 image; a
// trailing partial word (len not a multiple of 4) is ignored.
func hipLEBytesToFloat32Slice(data []byte) []float32 {
	count := len(data) / 4
	out := make([]float32, count)
	for index := range out {
		out[index] = math.Float32frombits(binary.LittleEndian.Uint32(data[index*4:]))
	}
	return out
}
