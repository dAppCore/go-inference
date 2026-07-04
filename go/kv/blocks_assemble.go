// SPDX-Licence-Identifier: EUPL-1.2

package kv

import core "dappco.re/go"

// AssembleBlocks reassembles contiguous blocks produced by SplitBlocks.
func AssembleBlocks(blocks []Block) (*Snapshot, error) {
	if len(blocks) == 0 {
		return nil, errBlocksEmpty
	}
	totalTokens, err := validateKVSnapshotBlockOrder(blocks)
	if err != nil {
		return nil, err
	}
	first := blocks[0].Snapshot
	if first == nil {
		return nil, errBlockNil
	}
	assembled := &Snapshot{
		Version:       first.Version,
		Architecture:  first.Architecture,
		NumLayers:     first.NumLayers,
		NumHeads:      first.NumHeads,
		HeadDim:       first.HeadDim,
		NumQueryHeads: first.NumQueryHeads,
		Layers:        emptyKVSnapshotLayers(first.Layers),
		// Pre-size Tokens against the validated total — append-block
		// accumulates a known count, so geometric grow is pure waste.
		Tokens: make([]int32, 0, totalTokens),
	}
	// Pre-size the per-head KeyBytes/ValueBytes buffers against the summed
	// raw payload across all blocks. appendKVSnapshotRawBlock otherwise
	// rides through Go's geometric grow on every block — once on first
	// arrival, plus one or two grows by block 3. The pre-sum pass walks
	// blocks × layers × heads but does no allocs.
	preSizeAssembledRawBytes(assembled, blocks)
	for _, block := range blocks {
		if block.Snapshot == nil {
			return nil, errBlockNil
		}
		if err := appendKVSnapshotBlock(assembled, block.Snapshot); err != nil {
			return nil, err
		}
	}
	last := blocks[len(blocks)-1].Snapshot
	assembled.Generated = core.SliceClone(last.Generated)
	assembled.TokenOffset = last.TokenOffset
	assembled.LogitShape = core.SliceClone(last.LogitShape)
	assembled.Logits = core.SliceClone(last.Logits)
	if assembled.TokenOffset == 0 {
		assembled.TokenOffset = len(assembled.Tokens)
	}
	return assembled, nil
}

// preSizeAssembledRawBytes pre-allocates per-head raw byte buffers in the
// assembled snapshot against the total payload across all blocks. Saves
// the appendKVSnapshotRawBlock geometric-grow path during AssembleBlocks.
func preSizeAssembledRawBytes(assembled *Snapshot, blocks []Block) {
	if assembled == nil || len(assembled.Layers) == 0 || len(blocks) == 0 {
		return
	}
	for layerIndex := range assembled.Layers {
		var layerKeyTotal, layerValueTotal int
		for _, block := range blocks {
			if block.Snapshot == nil || layerIndex >= len(block.Snapshot.Layers) {
				continue
			}
			srcLayer := block.Snapshot.Layers[layerIndex]
			layerKeyTotal += len(srcLayer.KeyBytes)
			layerValueTotal += len(srcLayer.ValueBytes)
		}
		dstLayer := &assembled.Layers[layerIndex]
		if layerKeyTotal > 0 {
			dstLayer.KeyBytes = make([]byte, 0, layerKeyTotal)
		}
		if layerValueTotal > 0 {
			dstLayer.ValueBytes = make([]byte, 0, layerValueTotal)
		}
		for headIndex := range assembled.Layers[layerIndex].Heads {
			var keyTotal, valueTotal int
			for _, block := range blocks {
				if block.Snapshot == nil || layerIndex >= len(block.Snapshot.Layers) {
					continue
				}
				srcLayer := block.Snapshot.Layers[layerIndex]
				if headIndex >= len(srcLayer.Heads) {
					continue
				}
				srcHead := srcLayer.Heads[headIndex]
				keyTotal += len(srcHead.KeyBytes)
				valueTotal += len(srcHead.ValueBytes)
			}
			var keyValueTotal, valueValueTotal int
			for _, block := range blocks {
				if block.Snapshot == nil || layerIndex >= len(block.Snapshot.Layers) {
					continue
				}
				srcLayer := block.Snapshot.Layers[layerIndex]
				if headIndex >= len(srcLayer.Heads) {
					continue
				}
				srcHead := srcLayer.Heads[headIndex]
				keyValueTotal += len(srcHead.Key)
				valueValueTotal += len(srcHead.Value)
			}
			dstHead := &assembled.Layers[layerIndex].Heads[headIndex]
			if keyTotal > 0 {
				dstHead.KeyBytes = make([]byte, 0, keyTotal)
			}
			if valueTotal > 0 {
				dstHead.ValueBytes = make([]byte, 0, valueTotal)
			}
			// Pre-size the float32 Key/Value slices too — appendKVSnapshotBlock
			// grows these per block on the float32-encoded path, otherwise
			// riding Go's geometric grow. The KeyBytes/ValueBytes pre-size
			// above only covers the native raw path.
			if keyValueTotal > 0 {
				dstHead.Key = make([]float32, 0, keyValueTotal)
			}
			if valueValueTotal > 0 {
				dstHead.Value = make([]float32, 0, valueValueTotal)
			}
		}
	}
}

func validateKVSnapshotBlockOrder(blocks []Block) (int, error) {
	nextStart := 0
	for index, block := range blocks {
		if block.Index != index {
			return 0, errBlocksOutOfOrder
		}
		if block.TokenStart != nextStart || block.TokenCount <= 0 {
			return 0, errBlocksNotContiguous
		}
		if block.Snapshot == nil || len(block.Snapshot.Tokens) != block.TokenCount {
			return 0, errBlockTokenCountMismatch
		}
		nextStart += block.TokenCount
	}
	return nextStart, nil
}

func emptyKVSnapshotLayers(layers []LayerSnapshot) []LayerSnapshot {
	out := make([]LayerSnapshot, len(layers))
	// Heads-slab: one backing slice across all layers — typical assembled
	// snapshots carry uniform NumHeads per layer (the first block sets
	// shape so we use it as the slab size). Layers with a divergent head
	// count fall back to per-layer make.
	var slabHeadsPerLayer int
	for _, layer := range layers {
		if len(layer.Heads) > slabHeadsPerLayer {
			slabHeadsPerLayer = len(layer.Heads)
		}
	}
	var headSlab []HeadSnapshot
	var slabCursor int
	if slabHeadsPerLayer > 0 {
		headSlab = make([]HeadSnapshot, len(layers)*slabHeadsPerLayer)
	}
	for i, layer := range layers {
		out[i] = LayerSnapshot{
			Layer:      layer.Layer,
			CacheIndex: layer.CacheIndex,
			CacheMode:  layer.CacheMode,
			MaxSize:    layer.MaxSize,
			KeyDType:   layer.KeyDType,
			KeyShape:   core.SliceClone(layer.KeyShape),
			ValueDType: layer.ValueDType,
			ValueShape: core.SliceClone(layer.ValueShape),
		}
		headCount := len(layer.Heads)
		if headCount > 0 {
			if headSlab != nil && slabCursor+headCount <= len(headSlab) {
				out[i].Heads = headSlab[slabCursor : slabCursor+headCount : slabCursor+headCount]
				slabCursor += headCount
			} else {
				out[i].Heads = make([]HeadSnapshot, headCount)
			}
		}
	}
	return out
}

func appendKVSnapshotBlock(dst *Snapshot, block *Snapshot) error {
	if block.Architecture != "" && dst.Architecture != "" && block.Architecture != dst.Architecture {
		return errBlockArchMismatch
	}
	if block.HeadDim != dst.HeadDim || block.NumHeads != dst.NumHeads || block.NumLayers != dst.NumLayers {
		return errBlockShapeMismatch
	}
	if len(block.Layers) != len(dst.Layers) {
		return errBlockLayerCountMismatch
	}
	dst.Tokens = append(dst.Tokens, block.Tokens...)
	dst.SeqLen += block.SeqLen
	for layerIndex, layer := range block.Layers {
		dstLayer := &dst.Layers[layerIndex]
		if layer.CacheMode != "" {
			if dstLayer.CacheMode != "" && dstLayer.CacheMode != layer.CacheMode {
				return errBlockMetadataMismatch
			}
			dstLayer.CacheMode = layer.CacheMode
		}
		if layer.MaxSize > 0 {
			if dstLayer.MaxSize > 0 && dstLayer.MaxSize != layer.MaxSize {
				return errBlockMetadataMismatch
			}
			dstLayer.MaxSize = layer.MaxSize
		}
		if len(layer.TurboQuantPayloads) > 0 {
			dstLayer.TurboQuantPayloads = append(dstLayer.TurboQuantPayloads, cloneKVByteSlices(layer.TurboQuantPayloads)...)
		}
		if len(layer.KeyBytes) > 0 {
			if err := appendKVSnapshotLayerRawBlock(&dstLayer.KeyDType, &dstLayer.KeyBytes, &dstLayer.KeyShape, layer.KeyDType, layer.KeyBytes, layer.KeyShape); err != nil {
				return core.E("AssembleBlocks", "append native layer key tensor", err)
			}
		}
		if len(layer.ValueBytes) > 0 {
			if err := appendKVSnapshotLayerRawBlock(&dstLayer.ValueDType, &dstLayer.ValueBytes, &dstLayer.ValueShape, layer.ValueDType, layer.ValueBytes, layer.ValueShape); err != nil {
				return core.E("AssembleBlocks", "append native layer value tensor", err)
			}
		}
		if len(layer.Heads) == 0 {
			continue
		}
		if len(dst.Layers[layerIndex].Heads) == 0 {
			dst.Layers[layerIndex].Heads = make([]HeadSnapshot, len(layer.Heads))
		}
		if len(layer.Heads) != len(dst.Layers[layerIndex].Heads) {
			return errBlockHeadCountMismatch
		}
		for headIndex, head := range layer.Heads {
			dstHead := &dst.Layers[layerIndex].Heads[headIndex]
			dstHead.Key = append(dstHead.Key, head.Key...)
			dstHead.Value = append(dstHead.Value, head.Value...)
			if err := appendKVSnapshotRawBlock(&dstHead.KeyDType, &dstHead.KeyBytes, head.KeyDType, head.KeyBytes); err != nil {
				return core.E("AssembleBlocks", "append native key tensor", err)
			}
			if err := appendKVSnapshotRawBlock(&dstHead.ValueDType, &dstHead.ValueBytes, head.ValueDType, head.ValueBytes); err != nil {
				return core.E("AssembleBlocks", "append native value tensor", err)
			}
		}
	}
	return nil
}

func appendKVSnapshotLayerRawBlock(dstDType *string, dstBytes *[]byte, dstShape *[]int32, dtype string, raw []byte, shape []int32) error {
	if len(raw) == 0 {
		return nil
	}
	dtype, bytesPerValue := normalizeKVSnapshotTensorDType(dtype)
	if dtype == "" || bytesPerValue <= 0 {
		return errUnsupportedLayerRawTensor
	}
	if len(shape) == 3 {
		L, H, D := int(shape[0]), int(shape[1]), int(shape[2])
		if L <= 0 || H <= 0 || D <= 0 || len(raw) != L*H*D*bytesPerValue {
			return errLayerRawTensorShape
		}
		if *dstDType == "" {
			*dstDType = dtype
		} else if *dstDType != dtype {
			return errLayerRawDtypeMismatch
		}
		if len(*dstBytes) == 0 {
			*dstBytes = append((*dstBytes)[:0], raw...)
			*dstShape = core.SliceClone(shape)
			return nil
		}
		if len(*dstShape) != 3 || int((*dstShape)[1]) != H || int((*dstShape)[2]) != D {
			return errLayerRawTensorShape
		}
		oldLen := int((*dstShape)[0])
		if oldLen <= 0 || len(*dstBytes) != oldLen*H*D*bytesPerValue {
			return errLayerRawByteLenMismatch
		}
		*dstBytes = append(*dstBytes, raw...)
		(*dstShape)[0] = int32(oldLen + L)
		return nil
	}
	if len(shape) != 4 {
		return errUnsupportedLayerRawTensor
	}
	B, H, L, D := int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3])
	if B <= 0 || H <= 0 || L <= 0 || D <= 0 || len(raw) != B*H*L*D*bytesPerValue {
		return errLayerRawTensorShape
	}
	if *dstDType == "" {
		*dstDType = dtype
	} else if *dstDType != dtype {
		return errLayerRawDtypeMismatch
	}
	if len(*dstBytes) == 0 {
		// First-arrival path is the only owner of the new shape — clone
		// happens here, not unconditionally on every call. Subsequent
		// calls rewrite dstShape[2] in-place after validating B/H/D.
		*dstBytes = append((*dstBytes)[:0], raw...)
		*dstShape = core.SliceClone(shape)
		return nil
	}
	if len(*dstShape) != 4 || int((*dstShape)[0]) != B || int((*dstShape)[1]) != H || int((*dstShape)[3]) != D {
		return errLayerRawTensorShape
	}
	// oldShape was previously cloned + read for oldLen — direct read
	// from dstShape eliminates the clone alloc; we only need shape[2]
	// (the sequence-length dim) and shape is rewritten in-place below.
	oldLen := int((*dstShape)[2])
	if oldLen <= 0 || len(*dstBytes) != B*H*oldLen*D*bytesPerValue {
		return errLayerRawByteLenMismatch
	}
	totalLen := oldLen + L
	if B*H == 1 {
		*dstBytes = append(*dstBytes, raw...)
		(*dstShape)[2] = int32(totalLen)
		return nil
	}
	merged := make([]byte, B*H*totalLen*D*bytesPerValue)
	oldRowBytes := oldLen * D * bytesPerValue
	newRowBytes := L * D * bytesPerValue
	totalRowBytes := totalLen * D * bytesPerValue
	for b := range B {
		for h := range H {
			row := b*H + h
			dstStart := row * totalRowBytes
			oldStart := row * oldRowBytes
			newStart := row * newRowBytes
			copy(merged[dstStart:dstStart+oldRowBytes], (*dstBytes)[oldStart:oldStart+oldRowBytes])
			copy(merged[dstStart+oldRowBytes:dstStart+oldRowBytes+newRowBytes], raw[newStart:newStart+newRowBytes])
		}
	}
	*dstBytes = merged
	(*dstShape)[2] = int32(totalLen)
	return nil
}

func appendKVSnapshotRawBlock(dstDType *string, dstBytes *[]byte, dtype string, raw []byte) error {
	if len(raw) == 0 {
		return nil
	}
	dtype, bytesPerValue := normalizeKVSnapshotTensorDType(dtype)
	if dtype == "" || bytesPerValue <= 0 {
		return errUnsupportedRawTensorDtype
	}
	if *dstDType == "" {
		*dstDType = dtype
	} else if *dstDType != dtype {
		return errRawTensorDtypeMismatch
	}
	*dstBytes = append(*dstBytes, raw...)
	return nil
}
