// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"encoding/json"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/state"
)

const (
	rocmGemma4Q4StateBundleKind     = "rocm-gemma4-q4-device-kv-state-bundle"
	rocmGemma4Q4StateBundleEncoding = "rocm/gemma4-q4-device-kv-state-bundle+json"
)

type rocmGemma4Q4StateBundleSnapshot struct {
	Version     int                                  `json:"version"`
	Kind        string                               `json:"kind"`
	Mode        string                               `json:"mode,omitempty"`
	LayerCount  int                                  `json:"layer_count,omitempty"`
	TokenCount  int                                  `json:"token_count,omitempty"`
	MemoryBytes uint64                               `json:"memory_bytes,omitempty"`
	Labels      map[string]string                    `json:"labels,omitempty"`
	Layers      []rocmGemma4Q4StateBundleLayerRecord `json:"layers,omitempty"`
}

type rocmGemma4Q4StateBundleLayerRecord struct {
	Index      int               `json:"index"`
	URI        string            `json:"uri"`
	State      state.ChunkRef    `json:"state,omitempty"`
	TokenCount int               `json:"token_count,omitempty"`
	BlockSize  int               `json:"block_size,omitempty"`
	Blocks     int               `json:"blocks,omitempty"`
	SizeBytes  uint64            `json:"size_bytes,omitempty"`
	Encoding   string            `json:"encoding,omitempty"`
	Labels     map[string]string `json:"labels,omitempty"`
}

type hipGemma4Q4HostDecodeStateRuntime struct {
	state      hipGemma4Q4DecodeState
	mode       string
	tokenCount int
	labels     map[string]string
}

func (runtime *hipGemma4Q4HostDecodeStateRuntime) Close() error {
	return nil
}

func sleepGemma4Q4DeviceDecodeStateBundle(ctx context.Context, req inference.AgentMemorySleepRequest, writer state.BinaryWriter, entryURI string, labels map[string]string, runtime *hipGemma4Q4DeviceDecodeState) (state.ChunkRef, []inference.StateRef, string, uint64, int, int, error) {
	if runtime == nil || runtime.LayerCount() == 0 {
		return state.ChunkRef{}, nil, "", 0, 0, 0, core.E("rocm.SleepState", "Gemma4 q4 device state is empty", nil)
	}
	labels["kv_serialize"] = "gemma4_q4_device_layer_blocks"
	labels["kv_block_bundle"] = "gemma4_q4_layers"
	labels["kv_restore_path"] = "gemma4_q4_layer_block_stream"
	labels["gemma4_q4_state_bundle"] = "layer_block_bundles"
	labels["gemma4_q4_device_kv_layers"] = core.Sprintf("%d", runtime.LayerCount())
	labels["gemma4_q4_device_kv_tokens"] = core.Sprintf("%d", runtime.maxLayerTokenCount())
	for key, value := range runtime.Labels() {
		labels[key] = value
	}

	layerRecords := make([]rocmGemma4Q4StateBundleLayerRecord, 0, runtime.LayerCount())
	stateRefs := make([]inference.StateRef, 0, runtime.LayerCount())
	var totalBytes uint64
	var totalBlocks int
	for index, layer := range runtime.layers {
		if layer.cache == nil || layer.cache.PageCount() == 0 {
			return state.ChunkRef{}, nil, "", 0, 0, 0, core.E("rocm.SleepState", core.Sprintf("Gemma4 q4 device layer %d KV cache is empty", index), nil)
		}
		host, err := layer.cache.hostCache()
		if err != nil {
			return state.ChunkRef{}, nil, "", 0, 0, 0, core.E("rocm.SleepState", core.Sprintf("copy Gemma4 q4 device layer %d KV", index), err)
		}
		layerURI := core.Sprintf("%s/layer/%04d", entryURI, index)
		layerLabels := mergeStringMaps(labels, map[string]string{
			"gemma4_q4_layer":        core.Sprintf("%d", index),
			"gemma4_q4_layer_tokens": core.Sprintf("%d", host.TokenCount()),
		})
		ref, refs, encoding, sizeBytes, tokens, blocks, err := sleepKVCacheBlockBundle(ctx, req, writer, layerURI, layerLabels, host, "gemma4_q4_device_layer_blocks")
		if err != nil {
			return state.ChunkRef{}, nil, "", 0, 0, 0, err
		}
		totalBytes += sizeBytes
		totalBlocks += blocks
		stateRefs = append(stateRefs, inference.StateRef{
			Kind:      "gemma4-q4-layer-kv-bundle",
			URI:       layerURI,
			SizeBytes: sizeBytes,
			Encoding:  encoding,
			Labels:    cloneStringMap(layerLabels),
		})
		stateRefs = append(stateRefs, refs...)
		layerRecords = append(layerRecords, rocmGemma4Q4StateBundleLayerRecord{
			Index:      index,
			URI:        layerURI,
			State:      ref,
			TokenCount: tokens,
			BlockSize:  host.blockSize,
			Blocks:     blocks,
			SizeBytes:  sizeBytes,
			Encoding:   encoding,
			Labels:     cloneStringMap(layerLabels),
		})
	}
	bundle := rocmGemma4Q4StateBundleSnapshot{
		Version:     1,
		Kind:        rocmGemma4Q4StateBundleKind,
		Mode:        runtime.mode,
		LayerCount:  runtime.LayerCount(),
		TokenCount:  runtime.maxLayerTokenCount(),
		MemoryBytes: runtime.MemoryBytes(),
		Labels:      cloneStringMap(labels),
		Layers:      layerRecords,
	}
	payload, err := json.Marshal(bundle)
	if err != nil {
		return state.ChunkRef{}, nil, "", 0, 0, 0, core.E("rocm.SleepState", "encode Gemma4 q4 state bundle", err)
	}
	ref, err := writer.PutBytes(ctx, payload, state.PutOptions{
		URI:   entryURI,
		Title: req.Title,
		Kind:  rocmGemma4Q4StateBundleKind,
		Track: rocmGemma4Q4StateBundleEncoding,
		Tags:  mergeStringMaps(req.Metadata, labels),
	})
	if err != nil {
		return state.ChunkRef{}, nil, "", 0, 0, 0, core.E("rocm.SleepState", "write Gemma4 q4 state bundle", err)
	}
	totalBytes += uint64(len(payload))
	labels["gemma4_q4_state_bundle_bytes"] = core.Sprintf("%d", totalBytes)
	labels["gemma4_q4_state_bundle_layers"] = core.Sprintf("%d", len(layerRecords))
	stateRefs = append([]inference.StateRef{{
		Kind:      "gemma4-q4-device-state",
		URI:       entryURI,
		SizeBytes: uint64(len(payload)),
		Encoding:  rocmGemma4Q4StateBundleEncoding,
		Labels:    cloneStringMap(labels),
	}}, stateRefs...)
	return ref, stateRefs, rocmGemma4Q4StateBundleEncoding, uint64(len(payload)), bundle.TokenCount, totalBlocks, nil
}

func wakeGemma4Q4HostDecodeStateFromChunk(ctx context.Context, store state.Store, chunk state.Chunk) (*hipGemma4Q4HostDecodeStateRuntime, bool, map[string]string, error) {
	data := chunk.Data
	if len(data) == 0 && chunk.Text != "" {
		data = []byte(chunk.Text)
	}
	if len(data) == 0 {
		return nil, false, nil, nil
	}
	var bundle rocmGemma4Q4StateBundleSnapshot
	if err := json.Unmarshal(data, &bundle); err != nil || bundle.Kind != rocmGemma4Q4StateBundleKind {
		return nil, false, nil, nil
	}
	if bundle.LayerCount <= 0 || len(bundle.Layers) != bundle.LayerCount {
		return nil, true, nil, core.E("rocm.WakeState", "Gemma4 q4 state bundle layer count mismatch", nil)
	}
	runtime := &hipGemma4Q4HostDecodeStateRuntime{
		state:      hipGemma4Q4DecodeState{Layers: make([]hipGemma4Q4LayerKVState, bundle.LayerCount)},
		mode:       bundle.Mode,
		tokenCount: bundle.TokenCount,
		labels:     cloneStringMap(bundle.Labels),
	}
	for _, layer := range bundle.Layers {
		if layer.Index < 0 || layer.Index >= bundle.LayerCount {
			return nil, true, nil, core.E("rocm.WakeState", "Gemma4 q4 state bundle layer index is invalid", nil)
		}
		layerChunk, err := resolveGemma4Q4LayerBundleChunk(ctx, store, layer)
		if err != nil {
			return nil, true, nil, err
		}
		cache, ok, _, err := wakeKVCacheFromChunk(ctx, store, layerChunk)
		if err != nil {
			return nil, true, nil, err
		}
		if !ok || cache == nil {
			return nil, true, nil, core.E("rocm.WakeState", "Gemma4 q4 layer KV bundle is required", nil)
		}
		keys, values, err := cache.Restore(0, cache.TokenCount())
		if err != nil {
			return nil, true, nil, err
		}
		runtime.state.Layers[layer.Index] = hipGemma4Q4LayerKVState{Keys: keys, Values: values}
	}
	labels := mergeStringMaps(bundle.Labels, map[string]string{
		"kv_restore":                    "runtime_owned",
		"kv_restore_path":               "gemma4_q4_layer_block_stream",
		"gemma4_q4_state_bundle":        "layer_block_bundles",
		"gemma4_q4_state_bundle_layers": core.Sprintf("%d", bundle.LayerCount),
		"gemma4_q4_state_bundle_tokens": core.Sprintf("%d", bundle.TokenCount),
		"gemma4_q4_device_kv_mode":      bundle.Mode,
		"gemma4_q4_device_kv_backing":   "host_restored_pending_device_mirror",
		"production_kv_cache_backing":   hipKernelStatusNotLinked,
	})
	return runtime, true, labels, nil
}

func resolveGemma4Q4LayerBundleChunk(ctx context.Context, store state.Store, layer rocmGemma4Q4StateBundleLayerRecord) (state.Chunk, error) {
	if layer.State.ChunkID != 0 || layer.State.HasFrameOffset || layer.State.Segment != "" || layer.State.Codec != "" {
		chunk, err := state.ResolveRefBytes(ctx, store, layer.State)
		if err != nil {
			return state.Chunk{}, core.E("rocm.WakeState", "resolve Gemma4 q4 layer bundle ref", err)
		}
		return chunk, nil
	}
	if layer.URI == "" {
		return state.Chunk{}, core.E("rocm.WakeState", "Gemma4 q4 layer bundle URI is required", nil)
	}
	chunk, err := state.ResolveURI(ctx, store, layer.URI)
	if err != nil {
		return state.Chunk{}, core.E("rocm.WakeState", "resolve Gemma4 q4 layer bundle URI", err)
	}
	return chunk, nil
}
