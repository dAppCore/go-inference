// SPDX-Licence-Identifier: EUPL-1.2

package kv

import core "dappco.re/go"

func kvSnapshotLayerWindowLen(layer LayerSnapshot, seqLen, headDim int) (int, error) {
	// Inline the per-length collect+iterate to skip a [2]int + [4]int
	// slice literal alloc per layer + per head (SaveStateBlocks fires
	// once per checkpointed block, with O(layers × heads) alloc count).
	windowLen := 0
	for _, length := range [2]int{
		kvSnapshotLayerRawWindowLen(layer.KeyBytes, layer.KeyDType, layer.KeyShape, seqLen),
		kvSnapshotLayerRawWindowLen(layer.ValueBytes, layer.ValueDType, layer.ValueShape, seqLen),
	} {
		if length < 0 {
			return 0, errLayerRawShapeMismatch
		}
		if length <= 0 {
			continue
		}
		if windowLen == 0 {
			windowLen = length
			continue
		}
		if windowLen != length {
			return 0, errLayerMixesWindowLens
		}
	}
	for _, head := range layer.Heads {
		for _, length := range [4]int{
			kvSnapshotTensorWindowLen(len(head.Key), seqLen, headDim),
			kvSnapshotTensorWindowLen(len(head.Value), seqLen, headDim),
			kvSnapshotRawTensorWindowLen(head.KeyBytes, head.KeyDType, seqLen, headDim),
			kvSnapshotRawTensorWindowLen(head.ValueBytes, head.ValueDType, seqLen, headDim),
		} {
			if length < 0 {
				return 0, errTensorShapeSeqHead
			}
			if length <= 0 {
				continue
			}
			if windowLen == 0 {
				windowLen = length
				continue
			}
			if windowLen != length {
				return 0, errLayerMixesWindowLens
			}
		}
	}
	return windowLen, nil
}

func kvSnapshotTensorWindowLen(valueCount, seqLen, headDim int) int {
	if valueCount <= 0 {
		return 0
	}
	if seqLen > 0 && valueCount%seqLen == 0 {
		return seqLen
	}
	if headDim > 0 && valueCount%headDim == 0 {
		return valueCount / headDim
	}
	return -1
}

func kvSnapshotRawTensorWindowLen(raw []byte, dtype string, seqLen, headDim int) int {
	if len(raw) == 0 {
		return 0
	}
	_, bytesPerValue := normalizeKVSnapshotTensorDType(dtype)
	if bytesPerValue <= 0 || len(raw)%bytesPerValue != 0 {
		return -1
	}
	return kvSnapshotTensorWindowLen(len(raw)/bytesPerValue, seqLen, headDim)
}

func kvSnapshotLayerRawWindowLen(raw []byte, dtype string, shape []int32, seqLen int) int {
	if len(raw) == 0 {
		return 0
	}
	_, bytesPerValue := normalizeKVSnapshotTensorDType(dtype)
	if bytesPerValue <= 0 {
		return -1
	}
	if len(shape) == 3 {
		L, H, D := int(shape[0]), int(shape[1]), int(shape[2])
		if L <= 0 || H <= 0 || D <= 0 {
			return -1
		}
		if len(raw) != L*H*D*bytesPerValue {
			return -1
		}
		if seqLen > 0 && L > seqLen {
			return -1
		}
		return L
	}
	if len(shape) != 4 {
		return -1
	}
	elements := 1
	for _, dim := range shape {
		if dim <= 0 {
			return -1
		}
		elements *= int(dim)
	}
	if len(raw) != elements*bytesPerValue {
		return -1
	}
	if seqLen > 0 && int(shape[2]) > seqLen {
		return -1
	}
	return int(shape[2])
}

func sliceKVSnapshotTensor(values []float32, start, end, headDim, seqLen int) ([]float32, error) {
	return sliceKVSnapshotTensorOpt(values, start, end, headDim, seqLen, true)
}

// sliceKVSnapshotTensorOpt slices a head Key/Value tensor. clone=false
// returns a sub-view of values (zero-alloc) — only the internal
// SaveStateBlocks walkBlocks path uses this, because the block snapshot
// is encoded + discarded within the yield call.
func sliceKVSnapshotTensorOpt(values []float32, start, end, headDim, seqLen int, clone bool) ([]float32, error) {
	if len(values) == 0 {
		return nil, nil
	}
	if seqLen <= 0 {
		return nil, errTensorShapeSeqHead
	}
	if headDim <= 0 || len(values) != seqLen*headDim {
		if len(values)%seqLen != 0 {
			return nil, errTensorShapeSeqHead
		}
		headDim = len(values) / seqLen
	}
	begin := start * headDim
	finish := end * headDim
	if begin < 0 || finish > len(values) || begin >= finish {
		return nil, errTensorBlockRangeInvalid
	}
	if clone {
		return core.SliceClone(values[begin:finish]), nil
	}
	return values[begin:finish:finish], nil
}

func sliceKVSnapshotRawTensor(raw []byte, dtype string, start, end, seqLen, valueCount int) ([]byte, error) {
	return sliceKVSnapshotRawTensorOpt(raw, dtype, start, end, seqLen, valueCount, true)
}

// sliceKVSnapshotRawTensorOpt slices a head's raw-byte tensor. clone=false
// returns a sub-view — see sliceKVSnapshotTensorOpt for the safe-use rule.
func sliceKVSnapshotRawTensorOpt(raw []byte, dtype string, start, end, seqLen, valueCount int, clone bool) ([]byte, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	_, bytesPerValue := normalizeKVSnapshotTensorDType(dtype)
	if bytesPerValue <= 0 {
		return nil, errUnsupportedRawTensorDtype
	}
	if valueCount <= 0 {
		if len(raw)%bytesPerValue != 0 {
			return nil, errRawTensorByteLenInvalid
		}
		valueCount = len(raw) / bytesPerValue
	}
	if seqLen <= 0 || valueCount%seqLen != 0 || len(raw) != valueCount*bytesPerValue {
		return nil, errRawTensorShapeSeq
	}
	headDim := valueCount / seqLen
	begin := start * headDim * bytesPerValue
	finish := end * headDim * bytesPerValue
	if begin < 0 || finish > len(raw) || begin >= finish {
		return nil, errRawTensorBlockRangeInvalid
	}
	if clone {
		return core.SliceClone(raw[begin:finish]), nil
	}
	return raw[begin:finish:finish], nil
}

func sliceKVSnapshotLayerRawTensor(raw []byte, dtype string, shape []int32, start, end int) ([]byte, []int32, error) {
	return sliceKVSnapshotLayerRawTensorOpt(raw, dtype, shape, start, end, true)
}

// sliceKVSnapshotLayerRawTensorOpt slices a native layer slab. clone=false can
// return a borrowed sub-view only when the requested sequence range is
// physically contiguous in the [B,H,L,D] row-major storage; for Gemma-style
// single K/V head slabs this keeps SaveStateBlocks from copying every block
// before the State writer immediately serialises it.
func sliceKVSnapshotLayerRawTensorOpt(raw []byte, dtype string, shape []int32, start, end int, clone bool) ([]byte, []int32, error) {
	if len(raw) == 0 {
		return nil, nil, nil
	}
	_, bytesPerValue := normalizeKVSnapshotTensorDType(dtype)
	if bytesPerValue <= 0 {
		return nil, nil, errUnsupportedLayerRawTensor
	}
	if len(shape) == 3 {
		L, H, D := int(shape[0]), int(shape[1]), int(shape[2])
		if L <= 0 || H <= 0 || D <= 0 || start < 0 || end <= start || end > L {
			return nil, nil, errLayerRawTensorRangeInvalid
		}
		rowBytes := H * D * bytesPerValue
		if len(raw) != L*rowBytes {
			return nil, nil, errLayerRawByteLenMismatch
		}
		begin := start * rowBytes
		finish := end * rowBytes
		outShape := core.SliceClone(shape)
		outShape[0] = int32(end - start)
		if clone {
			return core.SliceClone(raw[begin:finish]), outShape, nil
		}
		return raw[begin:finish:finish], outShape, nil
	}
	if len(shape) != 4 {
		return nil, nil, errUnsupportedLayerRawTensor
	}
	B, H, L, D := int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3])
	if B <= 0 || H <= 0 || L <= 0 || D <= 0 || start < 0 || end <= start || end > L {
		return nil, nil, errLayerRawTensorRangeInvalid
	}
	if len(raw) != B*H*L*D*bytesPerValue {
		return nil, nil, errLayerRawByteLenMismatch
	}
	take := end - start
	rowBytes := take * D * bytesPerValue
	if !clone && B*H == 1 {
		begin := start * D * bytesPerValue
		finish := begin + rowBytes
		outShape := core.SliceClone(shape)
		outShape[2] = int32(take)
		return raw[begin:finish:finish], outShape, nil
	}
	out := make([]byte, B*H*take*D*bytesPerValue)
	dst := 0
	for b := range B {
		for h := range H {
			src := (((b*H+h)*L + start) * D) * bytesPerValue
			copy(out[dst:dst+rowBytes], raw[src:src+rowBytes])
			dst += rowBytes
		}
	}
	outShape := core.SliceClone(shape)
	outShape[2] = int32(take)
	return out, outShape, nil
}
