// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

// inference_kv_snapshot.go is the structural reconcile between hip's retained
// Gemma4-Q4 host decode state and go-inference's portable inference/kv.Snapshot
// contract — the piece that lets engine/hip satisfy engine.Session's
// CaptureKVWithOptions / RestoreFromKV.
//
// # The two layouts (why this is a reconcile, not a rename)
//
// hip stores its retained KV per LAYER as a flat token-major float32 vector,
// KeyHeads*HeadDim wide per token. E2B uses one KV head; larger Gemma 4
// variants use multiple heads. The host boundary is float32 — hipGemma4Q4Device
// DecodeState.HostState restores the device cache to float32 via
// rocmKVCache.Restore, so no dequant happens here.
//
// kv.Snapshot is organised the transformer way — per [layer][head] tensors with
// KeyBytes/KeyShape/KeyDType (and optional per-head float32 slices). engine/
// metal's ArchSession produces exactly that from its multi-head native cache.
//
// # The mapping (layout assumed — Snider's parity test proves it end-to-end)
//
// kv.Snapshot stores [head][token][dim], while HIP's host state is
// [token][head][dim]. Capture de-interleaves each layer into head-major bytes
// and per-head slices; restore interleaves it back. For one KV head the layouts
// coincide, preserving the original byte representation.
package hip

import (
	"encoding/binary"
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/kv"
)

// hipKVSnapshotArchitecture tags the Gemma4 text cache independently of weight
// dtype. The retained runtime serves both MLX Q4 and dense BF16 models.
const hipKVSnapshotArchitecture = "gemma4_text"

// hipKVSnapshotLegacyQ4Architecture remains readable for snapshots emitted
// before the dense BF16 retained lane was linked.
const hipKVSnapshotLegacyQ4Architecture = "gemma4-q4"

// hipKVSnapshotFloat32DType is the K/V element dtype at hip's host boundary —
// HostState always returns float32 (the device FP16 cache is widened on copy).
const hipKVSnapshotFloat32DType = "float32"

// hipDecodeStateToSnapshot converts hip's retained Gemma4-Q4 host decode state
// into a portable kv.Snapshot. host is the per-layer float32 K/V read back from
// the device (deviceState.HostState); cfg supplies each layer's HeadDim (the KV
// head geometry); tokens is the full prompt+generated sequence held by the session
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
	numHeads := 0
	for index, layerState := range host.Layers {
		layerHeadDim := cfg.Layers[index].HeadDim
		layerHeads := firstPositiveInt(cfg.Layers[index].KeyHeads, 1)
		layerWidth := layerHeads * layerHeadDim
		if layerHeadDim <= 0 || layerWidth <= 0 {
			return nil, core.E("rocm.hip.KVSnapshot.Capture", "layer KV geometry must be positive", nil)
		}
		if len(layerState.Keys)%layerWidth != 0 || len(layerState.Values) != len(layerState.Keys) {
			return nil, core.E("rocm.hip.KVSnapshot.Capture", "layer K/V lengths must align with KV row width", nil)
		}
		layerTokens := len(layerState.Keys) / layerWidth
		if layerTokens > seqLen {
			seqLen = layerTokens
		}
		if layerHeads > numHeads {
			numHeads = layerHeads
		}
		shape := []int32{1, int32(layerHeads), int32(layerTokens), int32(layerHeadDim)}
		keysByHead := hipKVTokenMajorToHeadMajor(layerState.Keys, layerTokens, layerHeads, layerHeadDim)
		valuesByHead := hipKVTokenMajorToHeadMajor(layerState.Values, layerTokens, layerHeads, layerHeadDim)
		layer := kv.LayerSnapshot{
			Layer:      index,
			KeyDType:   hipKVSnapshotFloat32DType,
			KeyBytes:   hipFloat32SliceToLEBytes(keysByHead),
			KeyShape:   shape,
			ValueDType: hipKVSnapshotFloat32DType,
			ValueBytes: hipFloat32SliceToLEBytes(valuesByHead),
			ValueShape: append([]int32(nil), shape...),
		}
		if !opts.RawKVOnly {
			layer.Heads = make([]kv.HeadSnapshot, layerHeads)
			headValues := layerTokens * layerHeadDim
			for head := 0; head < layerHeads; head++ {
				start := head * headValues
				end := start + headValues
				layer.Heads[head] = kv.HeadSnapshot{
					Key:        append([]float32(nil), keysByHead[start:end]...),
					KeyDType:   hipKVSnapshotFloat32DType,
					Value:      append([]float32(nil), valuesByHead[start:end]...),
					ValueDType: hipKVSnapshotFloat32DType,
				}
			}
		}
		layers[index] = layer
	}
	return &kv.Snapshot{
		Version:       kv.SnapshotVersion,
		Architecture:  hipKVSnapshotArchitecture,
		Tokens:        append([]int32(nil), tokens...),
		Generated:     append([]int32(nil), generated...),
		NumLayers:     len(host.Layers),
		NumHeads:      numHeads,
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
	if snapshot.Architecture != "" && snapshot.Architecture != hipKVSnapshotArchitecture && snapshot.Architecture != hipKVSnapshotLegacyQ4Architecture {
		return hipGemma4Q4DecodeState{}, core.E("rocm.hip.KVSnapshot.Restore", "snapshot architecture is not Gemma4 text", nil)
	}
	if len(snapshot.Layers) != len(cfg.Layers) {
		return hipGemma4Q4DecodeState{}, core.E("rocm.hip.KVSnapshot.Restore", "snapshot layer count must match forward config", nil)
	}
	layers := make([]hipGemma4Q4LayerKVState, len(snapshot.Layers))
	for index, layerSnapshot := range snapshot.Layers {
		layerHeadDim := cfg.Layers[index].HeadDim
		layerHeads := firstPositiveInt(cfg.Layers[index].KeyHeads, 1)
		layerWidth := layerHeads * layerHeadDim
		if layerHeadDim <= 0 || layerWidth <= 0 {
			return hipGemma4Q4DecodeState{}, core.E("rocm.hip.KVSnapshot.Restore", "layer KV geometry must be positive", nil)
		}
		keys, values, err := hipLayerSnapshotKV(layerSnapshot, layerHeads, layerHeadDim)
		if err != nil {
			return hipGemma4Q4DecodeState{}, core.E("rocm.hip.KVSnapshot.Restore", core.Sprintf("decode layer %d", index), err)
		}
		if len(keys)%layerWidth != 0 || len(values) != len(keys) {
			return hipGemma4Q4DecodeState{}, core.E("rocm.hip.KVSnapshot.Restore", "snapshot layer K/V lengths must align with KV row width", nil)
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
func hipLayerSnapshotKV(layer kv.LayerSnapshot, headCount, headDim int) (keys, values []float32, err error) {
	rowWidth := headCount * headDim
	if rowWidth <= 0 {
		return nil, nil, core.NewError("snapshot KV geometry must be positive")
	}
	if len(layer.Heads) == headCount && len(layer.Heads) > 0 && len(layer.Heads[0].Key) > 0 {
		headValues := len(layer.Heads[0].Key)
		if headValues%headDim != 0 {
			return nil, nil, core.NewError("snapshot head length must align with HeadDim")
		}
		keysByHead := make([]float32, 0, headCount*headValues)
		valuesByHead := make([]float32, 0, headCount*headValues)
		for _, head := range layer.Heads {
			if len(head.Key) != headValues || len(head.Value) != headValues {
				return nil, nil, core.NewError("snapshot heads must have matching K/V geometry")
			}
			keysByHead = append(keysByHead, head.Key...)
			valuesByHead = append(valuesByHead, head.Value...)
		}
		tokens := headValues / headDim
		return hipKVHeadMajorToTokenMajor(keysByHead, tokens, headCount, headDim), hipKVHeadMajorToTokenMajor(valuesByHead, tokens, headCount, headDim), nil
	}
	if len(layer.Heads) == 1 && headCount > 1 && len(layer.Heads[0].Key) > 0 {
		keys = append([]float32(nil), layer.Heads[0].Key...)
		values = append([]float32(nil), layer.Heads[0].Value...)
		if len(keys)%rowWidth != 0 || len(values) != len(keys) {
			return nil, nil, core.NewError("legacy snapshot K/V geometry is invalid")
		}
		return keys, values, nil
	}
	keys = hipLEBytesToFloat32Slice(layer.KeyBytes)
	values = hipLEBytesToFloat32Slice(layer.ValueBytes)
	if len(keys) != len(values) {
		return nil, nil, core.NewError("snapshot K/V byte lengths must match")
	}
	if hipLayerSnapshotUsesHeadMajorLayout(layer, headCount, headDim, len(keys)) {
		tokens := len(keys) / rowWidth
		return hipKVHeadMajorToTokenMajor(keys, tokens, headCount, headDim), hipKVHeadMajorToTokenMajor(values, tokens, headCount, headDim), nil
	}
	if len(keys)%rowWidth != 0 {
		return nil, nil, core.NewError("legacy snapshot K/V lengths must align with KV row width")
	}
	return keys, values, nil
}

func hipLayerSnapshotUsesHeadMajorLayout(layer kv.LayerSnapshot, headCount, headDim, valueCount int) bool {
	if len(layer.KeyShape) != 4 || len(layer.ValueShape) != 4 || layer.KeyShape[0] != 1 || layer.ValueShape[0] != 1 {
		return false
	}
	if int(layer.KeyShape[1]) != headCount || int(layer.ValueShape[1]) != headCount ||
		int(layer.KeyShape[3]) != headDim || int(layer.ValueShape[3]) != headDim ||
		layer.KeyShape[2] < 0 || layer.ValueShape[2] != layer.KeyShape[2] {
		return false
	}
	return int(layer.KeyShape[2])*headCount*headDim == valueCount
}

func hipKVTokenMajorToHeadMajor(values []float32, tokens, heads, headDim int) []float32 {
	if len(values) == 0 {
		return nil
	}
	out := make([]float32, len(values))
	for token := 0; token < tokens; token++ {
		for head := 0; head < heads; head++ {
			source := (token*heads + head) * headDim
			target := (head*tokens + token) * headDim
			copy(out[target:target+headDim], values[source:source+headDim])
		}
	}
	return out
}

func hipKVHeadMajorToTokenMajor(values []float32, tokens, heads, headDim int) []float32 {
	if len(values) == 0 {
		return nil
	}
	out := make([]float32, len(values))
	for head := 0; head < heads; head++ {
		for token := 0; token < tokens; token++ {
			source := (head*tokens + token) * headDim
			target := (token*heads + head) * headDim
			copy(out[target:target+headDim], values[source:source+headDim])
		}
	}
	return out
}

// hipForwardConfigQueryHeads reports the layer-0 query-head count (informational
// NumQueryHeads on the snapshot).
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
