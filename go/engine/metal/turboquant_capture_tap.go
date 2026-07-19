// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
)

// turboquant_capture_tap.go is the RFC #41 slice S3 read-only tap: it reads a
// resident layer's cached K/V rows back to host float32 for the TurboQuant
// real-KV distortion instrument (go/kv/turboquant.MeasureReal). It reuses
// CaptureKV's own accessors (stateLayerViews, nativeKVLayerCaptureWindow,
// stateBlockLayerBytes, nativeKVTokenRowsToLayerSlab, nativeKVLayerSlabHeads
// — see session_kv_snapshot.go) rather than re-deriving the cache geometry,
// so a resident q8 layer's bf16 mirror is refreshed exactly as CaptureKV
// refreshes it. It issues no GPU work of its own and writes nothing back —
// the decode it observes is unaffected.

// DumpKVRows reads layer's appended K and V cache rows back to host float32,
// POST-RoPE — exactly the vectors the live cache holds, at the granularity
// TurboQuant's codec operates over: one row per (attention head, cached
// token), headDim wide (turboquant_kv_payload.go's own TurboQuant KV
// integration already keys off the same [heads, seq_len, head_dim] shape).
// Rows are ordered head-major then token-major ascending; keys[i] and
// values[i] share index i between the two returned slices.
//
// layer must be a resident cache-OWNING layer index (LayerSpec.OwnsCache());
// a layer that shares another's cache (gemma4's KV-shared tail layers, a
// gated-delta recurrence layer) has no view of its own and returns an error
// — dump the owning layer's index instead (LayerSpec.KVShareFrom).
//
//	keys, values, err := session.DumpKVRows(4)
//	if err != nil { ... }
//	d := len(keys[0]) // headDim
func (s *ArchSession) DumpKVRows(layer int) (keys, values [][]float32, err error) {
	if s == nil {
		return nil, nil, core.NewError("native.DumpKVRows: nil session")
	}
	if layer < 0 {
		return nil, nil, core.NewError("native.DumpKVRows: negative layer")
	}
	if s.pos <= 0 {
		return nil, nil, core.NewError("native.DumpKVRows: empty cache")
	}
	views, err := s.stateLayerViews()
	if err != nil {
		return nil, nil, err
	}
	view, ok := dumpKVRowsFindView(views, layer)
	if !ok {
		return nil, nil, core.NewError("native.DumpKVRows: layer has no resident KV cache (shared or gated-delta layer — dump its owner instead)")
	}
	start, tokenCount, err := nativeKVLayerCaptureWindow(view, s.pos)
	if err != nil {
		return nil, nil, err
	}
	keyRows, valueRows, err := stateBlockLayerBytes(view, start, tokenCount, s.pos)
	if err != nil {
		return nil, nil, err
	}
	if len(keyRows) != tokenCount*view.rowBytes || len(valueRows) != tokenCount*view.rowBytes {
		return nil, nil, core.NewError("native.DumpKVRows: layer payload size mismatch")
	}
	keySlab := make([]byte, len(keyRows))
	valueSlab := make([]byte, len(valueRows))
	nativeKVTokenRowsToLayerSlab(keySlab, keyRows, tokenCount, view.kvHeads, view.headDim)
	nativeKVTokenRowsToLayerSlab(valueSlab, valueRows, tokenCount, view.kvHeads, view.headDim)
	heads := nativeKVLayerSlabHeads(keySlab, valueSlab, tokenCount, view.kvHeads, view.headDim)

	keys = make([][]float32, 0, view.kvHeads*tokenCount)
	values = make([][]float32, 0, view.kvHeads*tokenCount)
	for _, head := range heads {
		for t := 0; t < tokenCount; t++ {
			keys = append(keys, head.Key[t*view.headDim:(t+1)*view.headDim])
			values = append(values, head.Value[t*view.headDim:(t+1)*view.headDim])
		}
	}
	return keys, values, nil
}

// dumpKVRowsFindView returns layer's resident view — the same per-layer
// lookup CaptureKV performs inline over its views slice (session_kv_snapshot.go),
// pulled out here since DumpKVRows is this file's only caller.
func dumpKVRowsFindView(views []sessionStateLayerView, layer int) (sessionStateLayerView, bool) {
	for _, v := range views {
		if v.layer == layer {
			return v, true
		}
	}
	return sessionStateLayerView{}, false
}
