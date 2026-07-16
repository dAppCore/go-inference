// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"encoding/binary"
	"math"

	core "dappco.re/go"
)

const (
	hipGGUFQ4_0TensorType        uint32 = 2
	hipGGUFQ4_1TensorType        uint32 = 3
	hipGGUFQ5_1TensorType        uint32 = 7
	hipGGUFQ4KTensorType         uint32 = 12
	hipGGUFQ5KTensorType         uint32 = 13
	hipGGUFQ6KTensorType         uint32 = 14
	hipGGUFQ8_0TensorType        uint32 = 8
	hipGGUFQ4_0BlockSize                = 32
	hipGGUFQ4_0BlockBytes               = 18
	hipGGUFQ4_0GroupSize                = 32
	hipGGUFQ4_0PackedWeightBytes        = 16
	hipGGUFQ4_1BlockSize                = 32
	hipGGUFQ4_1BlockBytes               = 20
	hipGGUFQ4_1GroupSize                = 32
	hipGGUFQ4_1PackedWeightBytes        = 16
	hipGGUFQ5_1BlockSize                = 32
	hipGGUFQ5_1BlockBytes               = 24
	hipGGUFQ4KBlockSize                 = 256
	hipGGUFQ4KBlockBytes                = 144
	hipGGUFQ4KGroupSize                 = 32
	hipGGUFQ4KGroupsPerBlock            = 8
	hipGGUFQ4KPackedWeightBytes         = 128
	hipGGUFQ5KBlockSize                 = 256
	hipGGUFQ5KBlockBytes                = 176
	hipGGUFQ5KGroupSize                 = 32
	hipGGUFQ5KGroupsPerBlock            = 8
	hipGGUFQ5KPackedWeightBytes         = 192
	hipGGUFQ6KBlockSize                 = 256
	hipGGUFQ6KBlockBytes                = 210
	hipGGUFQ6KGroupSize                 = 16
	hipGGUFQ6KGroupsPerBlock            = 16
	hipGGUFQ6KPackedWeightBytes         = 192
	hipGGUFQ8_0BlockSize                = 32
	hipGGUFQ8_0BlockBytes               = 34
	hipGGUFQ8_0GroupSize                = 32
	hipNativeTensorTypeF32       uint32 = 0
	hipNativeTensorTypeF16       uint32 = 1
	hipNativeTensorTypeU32       uint32 = 26
	hipNativeTensorTypeBF16      uint32 = 30
)

type hipGGUFQ4KAffinePayload struct {
	Weights    []byte
	Scales     []byte
	Biases     []byte
	Rows       int
	Cols       int
	GroupSize  int
	Groups     int
	PackedCols int
	Bits       int
}

func (model *hipLoadedModel) synthesizeGemma4GGUFAffineTensors() error {
	if model == nil || model.driver == nil ||
		(!isROCmGemma4Architecture(model.modelInfo.Architecture) && !isROCmGemma4AssistantArchitecture(model.modelInfo.Architecture)) ||
		!rocmGemma4SourceFormatGGUF(model.modelLabels) {
		return nil
	}
	type sourceTensor struct {
		base   string
		tensor hipTensor
	}
	var sources []sourceTensor
	for name, tensor := range model.tensors {
		base, ok := hipGemma4CanonicalAffineBaseForGGUFWeightName(name)
		if !ok || !hipNativeTensorInfoCanRepackAsAffine(tensor.info) {
			continue
		}
		sources = append(sources, sourceTensor{base: base, tensor: tensor})
	}
	for _, source := range sources {
		if err := model.synthesizeGemma4GGUFAffineTensor(source.base, source.tensor); err != nil {
			return err
		}
	}
	return model.synthesizeGemma4GGUFBF16AliasTensors()
}

func (model *hipLoadedModel) synthesizeGemma4GGUFAffineTensor(baseName string, tensor hipTensor) error {
	data := make([]byte, int(tensor.info.ByteSize))
	if err := model.driver.CopyDeviceToHost(tensor.pointer, data); err != nil {
		return core.E("rocm.hip.GGUFAffine", "copy GGUF tensor "+tensor.info.Name, err)
	}
	payload, err := hipRepackGGUFKQuantToAffine(tensor.info, data)
	if err != nil {
		return core.E("rocm.hip.GGUFAffine", "repack GGUF tensor "+tensor.info.Name, err)
	}
	if err := model.uploadSyntheticHIPTensor(baseName+".weight", hipNativeTensorTypeU32, "U32",
		[]uint64{uint64(payload.Rows), uint64(payload.PackedCols)}, payload.Weights); err != nil {
		return err
	}
	if err := model.uploadSyntheticHIPTensor(baseName+".scales", hipNativeTensorTypeBF16, "BF16",
		[]uint64{uint64(payload.Rows), uint64(payload.Groups)}, payload.Scales); err != nil {
		return err
	}
	return model.uploadSyntheticHIPTensor(baseName+".biases", hipNativeTensorTypeBF16, "BF16",
		[]uint64{uint64(payload.Rows), uint64(payload.Groups)}, payload.Biases)
}

func (model *hipLoadedModel) uploadSyntheticHIPTensor(name string, tensorType uint32, typeName string, dimensions []uint64, payload []byte) error {
	if _, ok := model.tensors[name]; ok {
		return nil
	}
	pointer, err := model.driver.Malloc(uint64(len(payload)))
	if err != nil {
		return core.E("rocm.hip.GGUFAffine", "allocate synthetic tensor "+name, err)
	}
	if err := hipCopyHostToDeviceLabeled(model.driver, pointer, payload, "rocm.hip.GGUFAffine", name); err != nil {
		_ = model.driver.Free(pointer)
		return core.E("rocm.hip.GGUFAffine", "copy synthetic tensor "+name, err)
	}
	model.tensors[name] = hipTensor{
		info: nativeTensorInfo{
			Name:       name,
			Dimensions: append([]uint64(nil), dimensions...),
			Type:       tensorType,
			TypeName:   typeName,
			ByteSize:   uint64(len(payload)),
		},
		pointer: pointer,
	}
	return nil
}

func (model *hipLoadedModel) synthesizeGemma4GGUFBF16AliasTensors() error {
	type sourceTensor struct {
		name   string
		tensor hipTensor
	}
	var sources []sourceTensor
	for name, tensor := range model.tensors {
		canonical, ok := hipGemma4CanonicalBF16NameForGGUFName(name)
		if !ok || !hipNativeTensorInfoCanAliasAsBF16(tensor.info) {
			continue
		}
		sources = append(sources, sourceTensor{name: canonical, tensor: tensor})
	}
	for _, source := range sources {
		data := make([]byte, int(source.tensor.info.ByteSize))
		if err := model.driver.CopyDeviceToHost(source.tensor.pointer, data); err != nil {
			return core.E("rocm.hip.GGUFBF16Alias", "copy GGUF tensor "+source.tensor.info.Name, err)
		}
		payload, dimensions, tensorType, typeName, err := hipGGUFAliasPayload(source.tensor.info, data)
		if err != nil {
			return err
		}
		if err := model.uploadSyntheticHIPTensor(source.name, tensorType, typeName, dimensions, payload); err != nil {
			return err
		}
	}
	return nil
}

func hipGGUFAliasPayload(info nativeTensorInfo, data []byte) ([]byte, []uint64, uint32, string, error) {
	if uint64(len(data)) != info.ByteSize {
		return nil, nil, 0, "", core.E("rocm.hip.GGUFAlias", "payload byte count mismatch", nil)
	}
	dimensions := hipGGUFAliasDimensions(info.Dimensions)
	if hipNativeTensorInfoIsBF16(info) {
		return append([]byte(nil), data...), dimensions, hipNativeTensorTypeBF16, "BF16", nil
	}
	if hipNativeTensorInfoIsF16(info) {
		if len(data)%2 != 0 {
			return nil, nil, 0, "", core.E("rocm.hip.GGUFAlias", "F16 tensor byte count must align to float16", nil)
		}
		payload := make([]byte, len(data))
		for index := 0; index < len(data)/2; index++ {
			value := hipFloat16ToFloat32(binary.LittleEndian.Uint16(data[index*2:]))
			binary.LittleEndian.PutUint16(payload[index*2:], hipFloat32ToBFloat16(value))
		}
		return payload, dimensions, hipNativeTensorTypeBF16, "BF16", nil
	}
	if !hipNativeTensorInfoIsF32(info) {
		return nil, nil, 0, "", core.E("rocm.hip.GGUFAlias", "tensor must be F16, F32, or BF16", nil)
	}
	if len(data)%4 != 0 {
		return nil, nil, 0, "", core.E("rocm.hip.GGUFAlias", "F32 tensor byte count must align to float32", nil)
	}
	if len(info.Dimensions) == 1 {
		return append([]byte(nil), data...), dimensions, hipNativeTensorTypeF32, "F32", nil
	}
	payload := make([]byte, len(data)/2)
	for index := 0; index < len(data)/4; index++ {
		value := math.Float32frombits(binary.LittleEndian.Uint32(data[index*4:]))
		binary.LittleEndian.PutUint16(payload[index*2:], hipFloat32ToBFloat16(value))
	}
	return payload, dimensions, hipNativeTensorTypeBF16, "BF16", nil
}

func hipGGUFAliasDimensions(dimensions []uint64) []uint64 {
	if len(dimensions) == 2 {
		return []uint64{dimensions[1], dimensions[0]}
	}
	return append([]uint64(nil), dimensions...)
}

func hipRepackGGUFKQuantToAffine(info nativeTensorInfo, data []byte) (hipGGUFQ4KAffinePayload, error) {
	switch {
	case hipNativeTensorInfoIsGGUFQ4_0(info):
		return hipRepackGGUFQ4_0ToAffine(info, data)
	case hipNativeTensorInfoIsGGUFQ4_1(info):
		return hipRepackGGUFQ4_1ToAffine(info, data)
	case hipNativeTensorInfoIsGGUFQ4K(info):
		return hipRepackGGUFQ4KToAffine(info, data)
	case hipNativeTensorInfoIsGGUFQ5K(info):
		return hipRepackGGUFQ5KToAffine(info, data)
	case hipNativeTensorInfoIsGGUFQ6K(info):
		return hipRepackGGUFQ6KToAffine(info, data)
	case hipNativeTensorInfoIsGGUFQ8_0(info):
		return hipRepackGGUFQ8_0ToAffine(info, data)
	case hipNativeTensorInfoIsF32(info):
		return hipQuantizeGGUFF32ToAffine8(info, data)
	default:
		return hipGGUFQ4KAffinePayload{}, core.E("rocm.hip.GGUFAffine", "tensor must be GGUF affine-compatible", nil)
	}
}

func hipRepackGGUFQ4_0ToAffine(info nativeTensorInfo, data []byte) (hipGGUFQ4KAffinePayload, error) {
	if !hipNativeTensorInfoIsGGUFQ4_0(info) {
		return hipGGUFQ4KAffinePayload{}, core.E("rocm.hip.GGUFQ4_0Affine", "tensor must be GGUF Q4_0", nil)
	}
	if len(info.Dimensions) != 2 {
		return hipGGUFQ4KAffinePayload{}, core.E("rocm.hip.GGUFQ4_0Affine", "Q4_0 tensor must be rank-2", nil)
	}
	if uint64(len(data)) != info.ByteSize {
		return hipGGUFQ4KAffinePayload{}, core.E("rocm.hip.GGUFQ4_0Affine", "Q4_0 payload byte count mismatch", nil)
	}
	cols := int(info.Dimensions[0])
	rows := int(info.Dimensions[1])
	if rows <= 0 || cols <= 0 || cols%hipGGUFQ4_0BlockSize != 0 {
		return hipGGUFQ4KAffinePayload{}, core.E("rocm.hip.GGUFQ4_0Affine", "Q4_0 tensor shape must align to 32-column blocks", nil)
	}
	blocksPerRow := cols / hipGGUFQ4_0BlockSize
	wantBytes := rows * blocksPerRow * hipGGUFQ4_0BlockBytes
	if len(data) != wantBytes {
		return hipGGUFQ4KAffinePayload{}, core.E("rocm.hip.GGUFQ4_0Affine", "Q4_0 tensor byte count does not match dimensions", nil)
	}
	groups := cols / hipGGUFQ4_0GroupSize
	packedCols := cols / 8
	weights := make([]byte, rows*packedCols*4)
	scales := make([]byte, rows*groups*2)
	biases := make([]byte, rows*groups*2)
	for row := 0; row < rows; row++ {
		for blockIndex := 0; blockIndex < blocksPerRow; blockIndex++ {
			sourceOffset := (row*blocksPerRow + blockIndex) * hipGGUFQ4_0BlockBytes
			block := data[sourceOffset : sourceOffset+hipGGUFQ4_0BlockBytes]
			scale := hipFloat16ToFloat32(binary.LittleEndian.Uint16(block[0:]))
			groupIndex := row*groups + blockIndex
			binary.LittleEndian.PutUint16(scales[groupIndex*2:], hipFloat32ToBFloat16(scale))
			binary.LittleEndian.PutUint16(biases[groupIndex*2:], hipFloat32ToBFloat16(-8*scale))
			for lane := 0; lane < hipGGUFQ4_0BlockSize; lane++ {
				packed := block[2+lane%(hipGGUFQ4_0BlockSize/2)]
				value := packed & 0x0f
				if lane >= hipGGUFQ4_0BlockSize/2 {
					value = packed >> 4
				}
				col := blockIndex*hipGGUFQ4_0BlockSize + lane
				hipPutAffinePackedValue(weights, row*packedCols, col, 4, uint32(value))
			}
		}
	}
	return hipGGUFQ4KAffinePayload{
		Weights:    weights,
		Scales:     scales,
		Biases:     biases,
		Rows:       rows,
		Cols:       cols,
		GroupSize:  hipGGUFQ4_0GroupSize,
		Groups:     groups,
		PackedCols: packedCols,
		Bits:       4,
	}, nil
}

func hipRepackGGUFQ4_1ToAffine(info nativeTensorInfo, data []byte) (hipGGUFQ4KAffinePayload, error) {
	if !hipNativeTensorInfoIsGGUFQ4_1(info) {
		return hipGGUFQ4KAffinePayload{}, core.E("rocm.hip.GGUFQ4_1Affine", "tensor must be GGUF Q4_1", nil)
	}
	if len(info.Dimensions) != 2 {
		return hipGGUFQ4KAffinePayload{}, core.E("rocm.hip.GGUFQ4_1Affine", "Q4_1 tensor must be rank-2", nil)
	}
	if uint64(len(data)) != info.ByteSize {
		return hipGGUFQ4KAffinePayload{}, core.E("rocm.hip.GGUFQ4_1Affine", "Q4_1 payload byte count mismatch", nil)
	}
	cols := int(info.Dimensions[0])
	rows := int(info.Dimensions[1])
	if rows <= 0 || cols <= 0 || cols%hipGGUFQ4_1BlockSize != 0 {
		return hipGGUFQ4KAffinePayload{}, core.E("rocm.hip.GGUFQ4_1Affine", "Q4_1 tensor shape must align to 32-column blocks", nil)
	}
	blocksPerRow := cols / hipGGUFQ4_1BlockSize
	wantBytes := rows * blocksPerRow * hipGGUFQ4_1BlockBytes
	if len(data) != wantBytes {
		return hipGGUFQ4KAffinePayload{}, core.E("rocm.hip.GGUFQ4_1Affine", "Q4_1 tensor byte count does not match dimensions", nil)
	}
	groups := cols / hipGGUFQ4_1GroupSize
	packedCols := cols / 8
	weights := make([]byte, rows*packedCols*4)
	scales := make([]byte, rows*groups*2)
	biases := make([]byte, rows*groups*2)
	for row := 0; row < rows; row++ {
		for blockIndex := 0; blockIndex < blocksPerRow; blockIndex++ {
			sourceOffset := (row*blocksPerRow + blockIndex) * hipGGUFQ4_1BlockBytes
			block := data[sourceOffset : sourceOffset+hipGGUFQ4_1BlockBytes]
			scale := hipFloat16ToFloat32(binary.LittleEndian.Uint16(block[0:]))
			bias := hipFloat16ToFloat32(binary.LittleEndian.Uint16(block[2:]))
			groupIndex := row*groups + blockIndex
			binary.LittleEndian.PutUint16(scales[groupIndex*2:], hipFloat32ToBFloat16(scale))
			binary.LittleEndian.PutUint16(biases[groupIndex*2:], hipFloat32ToBFloat16(bias))
			for lane := 0; lane < hipGGUFQ4_1BlockSize; lane++ {
				packed := block[4+lane%(hipGGUFQ4_1BlockSize/2)]
				value := packed & 0x0f
				if lane >= hipGGUFQ4_1BlockSize/2 {
					value = packed >> 4
				}
				col := blockIndex*hipGGUFQ4_1BlockSize + lane
				hipPutAffinePackedValue(weights, row*packedCols, col, 4, uint32(value))
			}
		}
	}
	return hipGGUFQ4KAffinePayload{
		Weights:    weights,
		Scales:     scales,
		Biases:     biases,
		Rows:       rows,
		Cols:       cols,
		GroupSize:  hipGGUFQ4_1GroupSize,
		Groups:     groups,
		PackedCols: packedCols,
		Bits:       4,
	}, nil
}

func hipRepackGGUFQ4KToAffine(info nativeTensorInfo, data []byte) (hipGGUFQ4KAffinePayload, error) {
	if !hipNativeTensorInfoIsGGUFQ4K(info) {
		return hipGGUFQ4KAffinePayload{}, core.E("rocm.hip.GGUFQ4KAffine", "tensor must be GGUF Q4_K", nil)
	}
	if len(info.Dimensions) != 2 {
		return hipGGUFQ4KAffinePayload{}, core.E("rocm.hip.GGUFQ4KAffine", "Q4_K tensor must be rank-2", nil)
	}
	if uint64(len(data)) != info.ByteSize {
		return hipGGUFQ4KAffinePayload{}, core.E("rocm.hip.GGUFQ4KAffine", "Q4_K payload byte count mismatch", nil)
	}
	cols := int(info.Dimensions[0])
	rows := int(info.Dimensions[1])
	if rows <= 0 || cols <= 0 || cols%hipGGUFQ4KBlockSize != 0 {
		return hipGGUFQ4KAffinePayload{}, core.E("rocm.hip.GGUFQ4KAffine", "Q4_K tensor shape must align to 256-column blocks", nil)
	}
	blocksPerRow := cols / hipGGUFQ4KBlockSize
	wantBytes := rows * blocksPerRow * hipGGUFQ4KBlockBytes
	if len(data) != wantBytes {
		return hipGGUFQ4KAffinePayload{}, core.E("rocm.hip.GGUFQ4KAffine", "Q4_K tensor byte count does not match dimensions", nil)
	}
	groups := cols / hipGGUFQ4KGroupSize
	packedCols := cols / 8
	weights := make([]byte, rows*packedCols*4)
	scales := make([]byte, rows*groups*2)
	biases := make([]byte, rows*groups*2)
	for row := 0; row < rows; row++ {
		for blockIndex := 0; blockIndex < blocksPerRow; blockIndex++ {
			sourceOffset := (row*blocksPerRow + blockIndex) * hipGGUFQ4KBlockBytes
			block := data[sourceOffset : sourceOffset+hipGGUFQ4KBlockBytes]
			if err := hipRepackGGUFQ4KBlockToAffine(row, blockIndex, packedCols, groups, block, weights, scales, biases); err != nil {
				return hipGGUFQ4KAffinePayload{}, err
			}
		}
	}
	return hipGGUFQ4KAffinePayload{
		Weights:    weights,
		Scales:     scales,
		Biases:     biases,
		Rows:       rows,
		Cols:       cols,
		GroupSize:  hipGGUFQ4KGroupSize,
		Groups:     groups,
		PackedCols: packedCols,
		Bits:       4,
	}, nil
}

func hipRepackGGUFQ5KToAffine(info nativeTensorInfo, data []byte) (hipGGUFQ4KAffinePayload, error) {
	if !hipNativeTensorInfoIsGGUFQ5K(info) {
		return hipGGUFQ4KAffinePayload{}, core.E("rocm.hip.GGUFQ5KAffine", "tensor must be GGUF Q5_K", nil)
	}
	if len(info.Dimensions) != 2 {
		return hipGGUFQ4KAffinePayload{}, core.E("rocm.hip.GGUFQ5KAffine", "Q5_K tensor must be rank-2", nil)
	}
	if uint64(len(data)) != info.ByteSize {
		return hipGGUFQ4KAffinePayload{}, core.E("rocm.hip.GGUFQ5KAffine", "Q5_K payload byte count mismatch", nil)
	}
	cols := int(info.Dimensions[0])
	rows := int(info.Dimensions[1])
	if rows <= 0 || cols <= 0 || cols%hipGGUFQ5KBlockSize != 0 {
		return hipGGUFQ4KAffinePayload{}, core.E("rocm.hip.GGUFQ5KAffine", "Q5_K tensor shape must align to 256-column blocks", nil)
	}
	blocksPerRow := cols / hipGGUFQ5KBlockSize
	wantBytes := rows * blocksPerRow * hipGGUFQ5KBlockBytes
	if len(data) != wantBytes {
		return hipGGUFQ4KAffinePayload{}, core.E("rocm.hip.GGUFQ5KAffine", "Q5_K tensor byte count does not match dimensions", nil)
	}
	groups := cols / hipGGUFQ5KGroupSize
	packedCols := cols * 6 / 32
	weights := make([]byte, rows*packedCols*4)
	scales := make([]byte, rows*groups*2)
	biases := make([]byte, rows*groups*2)
	for row := 0; row < rows; row++ {
		for blockIndex := 0; blockIndex < blocksPerRow; blockIndex++ {
			sourceOffset := (row*blocksPerRow + blockIndex) * hipGGUFQ5KBlockBytes
			block := data[sourceOffset : sourceOffset+hipGGUFQ5KBlockBytes]
			if err := hipRepackGGUFQ5KBlockToAffine(row, blockIndex, packedCols, groups, block, weights, scales, biases); err != nil {
				return hipGGUFQ4KAffinePayload{}, err
			}
		}
	}
	return hipGGUFQ4KAffinePayload{
		Weights:    weights,
		Scales:     scales,
		Biases:     biases,
		Rows:       rows,
		Cols:       cols,
		GroupSize:  hipGGUFQ5KGroupSize,
		Groups:     groups,
		PackedCols: packedCols,
		Bits:       6,
	}, nil
}

func hipRepackGGUFQ6KToAffine(info nativeTensorInfo, data []byte) (hipGGUFQ4KAffinePayload, error) {
	if !hipNativeTensorInfoIsGGUFQ6K(info) {
		return hipGGUFQ4KAffinePayload{}, core.E("rocm.hip.GGUFQ6KAffine", "tensor must be GGUF Q6_K", nil)
	}
	if len(info.Dimensions) != 2 {
		return hipGGUFQ4KAffinePayload{}, core.E("rocm.hip.GGUFQ6KAffine", "Q6_K tensor must be rank-2", nil)
	}
	if uint64(len(data)) != info.ByteSize {
		return hipGGUFQ4KAffinePayload{}, core.E("rocm.hip.GGUFQ6KAffine", "Q6_K payload byte count mismatch", nil)
	}
	cols := int(info.Dimensions[0])
	rows := int(info.Dimensions[1])
	if rows <= 0 || cols <= 0 || cols%hipGGUFQ6KBlockSize != 0 {
		return hipGGUFQ4KAffinePayload{}, core.E("rocm.hip.GGUFQ6KAffine", "Q6_K tensor shape must align to 256-column blocks", nil)
	}
	blocksPerRow := cols / hipGGUFQ6KBlockSize
	wantBytes := rows * blocksPerRow * hipGGUFQ6KBlockBytes
	if len(data) != wantBytes {
		return hipGGUFQ4KAffinePayload{}, core.E("rocm.hip.GGUFQ6KAffine", "Q6_K tensor byte count does not match dimensions", nil)
	}
	groups := cols / hipGGUFQ6KGroupSize
	packedCols := cols * 6 / 32
	weights := make([]byte, rows*packedCols*4)
	scales := make([]byte, rows*groups*2)
	biases := make([]byte, rows*groups*2)
	for row := 0; row < rows; row++ {
		for blockIndex := 0; blockIndex < blocksPerRow; blockIndex++ {
			sourceOffset := (row*blocksPerRow + blockIndex) * hipGGUFQ6KBlockBytes
			block := data[sourceOffset : sourceOffset+hipGGUFQ6KBlockBytes]
			if err := hipRepackGGUFQ6KBlockToAffine(row, blockIndex, packedCols, groups, block, weights, scales, biases); err != nil {
				return hipGGUFQ4KAffinePayload{}, err
			}
		}
	}
	return hipGGUFQ4KAffinePayload{
		Weights:    weights,
		Scales:     scales,
		Biases:     biases,
		Rows:       rows,
		Cols:       cols,
		GroupSize:  hipGGUFQ6KGroupSize,
		Groups:     groups,
		PackedCols: packedCols,
		Bits:       6,
	}, nil
}

func hipRepackGGUFQ8_0ToAffine(info nativeTensorInfo, data []byte) (hipGGUFQ4KAffinePayload, error) {
	if !hipNativeTensorInfoIsGGUFQ8_0(info) {
		return hipGGUFQ4KAffinePayload{}, core.E("rocm.hip.GGUFQ8_0Affine", "tensor must be GGUF Q8_0", nil)
	}
	if len(info.Dimensions) != 2 {
		return hipGGUFQ4KAffinePayload{}, core.E("rocm.hip.GGUFQ8_0Affine", "Q8_0 tensor must be rank-2", nil)
	}
	if uint64(len(data)) != info.ByteSize {
		return hipGGUFQ4KAffinePayload{}, core.E("rocm.hip.GGUFQ8_0Affine", "Q8_0 payload byte count mismatch", nil)
	}
	cols := int(info.Dimensions[0])
	rows := int(info.Dimensions[1])
	if rows <= 0 || cols <= 0 || cols%hipGGUFQ8_0BlockSize != 0 {
		return hipGGUFQ4KAffinePayload{}, core.E("rocm.hip.GGUFQ8_0Affine", "Q8_0 tensor shape must align to 32-column blocks", nil)
	}
	blocksPerRow := cols / hipGGUFQ8_0BlockSize
	wantBytes := rows * blocksPerRow * hipGGUFQ8_0BlockBytes
	if len(data) != wantBytes {
		return hipGGUFQ4KAffinePayload{}, core.E("rocm.hip.GGUFQ8_0Affine", "Q8_0 tensor byte count does not match dimensions", nil)
	}
	groups := cols / hipGGUFQ8_0GroupSize
	packedCols := cols / 4
	weights := make([]byte, rows*packedCols*4)
	scales := make([]byte, rows*groups*2)
	biases := make([]byte, rows*groups*2)
	for row := 0; row < rows; row++ {
		for blockIndex := 0; blockIndex < blocksPerRow; blockIndex++ {
			sourceOffset := (row*blocksPerRow + blockIndex) * hipGGUFQ8_0BlockBytes
			block := data[sourceOffset : sourceOffset+hipGGUFQ8_0BlockBytes]
			scale := hipFloat16ToFloat32(binary.LittleEndian.Uint16(block[0:]))
			groupIndex := row*groups + blockIndex
			binary.LittleEndian.PutUint16(scales[groupIndex*2:], hipFloat32ToBFloat16(scale))
			binary.LittleEndian.PutUint16(biases[groupIndex*2:], hipFloat32ToBFloat16(-128*scale))
			for lane := 0; lane < hipGGUFQ8_0BlockSize; lane++ {
				col := blockIndex*hipGGUFQ8_0BlockSize + lane
				hipPutAffinePackedValue(weights, row*packedCols, col, 8, uint32(int(int8(block[2+lane]))+128))
			}
		}
	}
	return hipGGUFQ4KAffinePayload{
		Weights:    weights,
		Scales:     scales,
		Biases:     biases,
		Rows:       rows,
		Cols:       cols,
		GroupSize:  hipGGUFQ8_0GroupSize,
		Groups:     groups,
		PackedCols: packedCols,
		Bits:       8,
	}, nil
}

func hipQuantizeGGUFF32ToAffine8(info nativeTensorInfo, data []byte) (hipGGUFQ4KAffinePayload, error) {
	if !hipNativeTensorInfoIsF32(info) {
		return hipGGUFQ4KAffinePayload{}, core.E("rocm.hip.GGUFF32Affine", "tensor must be GGUF F32", nil)
	}
	if len(info.Dimensions) != 2 {
		return hipGGUFQ4KAffinePayload{}, core.E("rocm.hip.GGUFF32Affine", "F32 affine tensor must be rank-2", nil)
	}
	if uint64(len(data)) != info.ByteSize || len(data)%4 != 0 {
		return hipGGUFQ4KAffinePayload{}, core.E("rocm.hip.GGUFF32Affine", "F32 affine payload byte count mismatch", nil)
	}
	cols := int(info.Dimensions[0])
	rows := int(info.Dimensions[1])
	if rows <= 0 || cols <= 0 || cols%32 != 0 {
		return hipGGUFQ4KAffinePayload{}, core.E("rocm.hip.GGUFF32Affine", "F32 affine tensor shape must align to 32-column groups", nil)
	}
	groups := cols / 32
	packedCols := cols / 4
	weights := make([]byte, rows*packedCols*4)
	scales := make([]byte, rows*groups*2)
	biases := make([]byte, rows*groups*2)
	values := make([]float32, 32)
	for row := 0; row < rows; row++ {
		for group := 0; group < groups; group++ {
			minValue := float32(0)
			maxValue := float32(0)
			for lane := 0; lane < 32; lane++ {
				col := group*32 + lane
				offset := (row*cols + col) * 4
				value := math.Float32frombits(binary.LittleEndian.Uint32(data[offset:]))
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
					value = 0
				}
				values[lane] = value
				if lane == 0 || value < minValue {
					minValue = value
				}
				if lane == 0 || value > maxValue {
					maxValue = value
				}
			}
			scale := (maxValue - minValue) / 255
			if scale == 0 {
				scale = 1
			}
			groupIndex := row*groups + group
			binary.LittleEndian.PutUint16(scales[groupIndex*2:], hipFloat32ToBFloat16(scale))
			binary.LittleEndian.PutUint16(biases[groupIndex*2:], hipFloat32ToBFloat16(minValue))
			for lane, value := range values {
				quantized := uint32(0)
				if maxValue != minValue {
					quantized = uint32(math.Round(float64((value - minValue) / scale)))
					if quantized > 255 {
						quantized = 255
					}
				}
				hipPutAffinePackedValue(weights, row*packedCols, group*32+lane, 8, quantized)
			}
		}
	}
	return hipGGUFQ4KAffinePayload{
		Weights:    weights,
		Scales:     scales,
		Biases:     biases,
		Rows:       rows,
		Cols:       cols,
		GroupSize:  32,
		Groups:     groups,
		PackedCols: packedCols,
		Bits:       8,
	}, nil
}

func hipRepackGGUFQ4KBlockToAffine(row, blockIndex, packedCols, groups int, block, weights, scales, biases []byte) error {
	if len(block) != hipGGUFQ4KBlockBytes {
		return core.E("rocm.hip.GGUFQ4KAffine", "Q4_K block byte count mismatch", nil)
	}
	d := hipFloat16ToFloat32(binary.LittleEndian.Uint16(block[0:]))
	dmin := hipFloat16ToFloat32(binary.LittleEndian.Uint16(block[2:]))
	scaleMin := block[4:16]
	qs := block[16:]
	groupBase := row*groups + blockIndex*hipGGUFQ4KGroupsPerBlock
	packedBase := row*packedCols + blockIndex*(hipGGUFQ4KPackedWeightBytes/4)
	for group := 0; group < hipGGUFQ4KGroupsPerBlock; group++ {
		scale, minValue := hipGGUFQ4KScaleMin(scaleMin, group)
		binary.LittleEndian.PutUint16(scales[(groupBase+group)*2:], hipFloat32ToBFloat16(d*float32(scale)))
		binary.LittleEndian.PutUint16(biases[(groupBase+group)*2:], hipFloat32ToBFloat16(-dmin*float32(minValue)))
		groupPair := group / 2
		highNibble := group%2 == 1
		for wordIndex := 0; wordIndex < 4; wordIndex++ {
			word := uint32(0)
			for lane := 0; lane < 8; lane++ {
				value := qs[groupPair*32+wordIndex*8+lane]
				if highNibble {
					value >>= 4
				}
				word |= uint32(value&0x0f) << (lane * 4)
			}
			binary.LittleEndian.PutUint32(weights[(packedBase+group*4+wordIndex)*4:], word)
		}
	}
	return nil
}

func hipRepackGGUFQ5KBlockToAffine(row, blockIndex, packedCols, groups int, block, weights, scales, biases []byte) error {
	if len(block) != hipGGUFQ5KBlockBytes {
		return core.E("rocm.hip.GGUFQ5KAffine", "Q5_K block byte count mismatch", nil)
	}
	d := hipFloat16ToFloat32(binary.LittleEndian.Uint16(block[0:]))
	dmin := hipFloat16ToFloat32(binary.LittleEndian.Uint16(block[2:]))
	scaleMin := block[4:16]
	groupBase := row*groups + blockIndex*hipGGUFQ5KGroupsPerBlock
	packedBase := row * packedCols
	for group := 0; group < hipGGUFQ5KGroupsPerBlock; group++ {
		scale, minValue := hipGGUFQ4KScaleMin(scaleMin, group)
		binary.LittleEndian.PutUint16(scales[(groupBase+group)*2:], hipFloat32ToBFloat16(d*float32(scale)))
		binary.LittleEndian.PutUint16(biases[(groupBase+group)*2:], hipFloat32ToBFloat16(-dmin*float32(minValue)))
		for lane := 0; lane < hipGGUFQ5KGroupSize; lane++ {
			col := blockIndex*hipGGUFQ5KBlockSize + group*hipGGUFQ5KGroupSize + lane
			hipPutAffinePackedValue(weights, packedBase, col, 6, uint32(hipGGUFQ5KQuant(block, group, lane)))
		}
	}
	return nil
}

func hipRepackGGUFQ6KBlockToAffine(row, blockIndex, packedCols, groups int, block, weights, scales, biases []byte) error {
	if len(block) != hipGGUFQ6KBlockBytes {
		return core.E("rocm.hip.GGUFQ6KAffine", "Q6_K block byte count mismatch", nil)
	}
	d := hipFloat16ToFloat32(binary.LittleEndian.Uint16(block[208:]))
	groupBase := row*groups + blockIndex*hipGGUFQ6KGroupsPerBlock
	packedBase := row * packedCols
	for group := 0; group < hipGGUFQ6KGroupsPerBlock; group++ {
		scale := d * float32(int8(block[192+group]))
		binary.LittleEndian.PutUint16(scales[(groupBase+group)*2:], hipFloat32ToBFloat16(scale))
		binary.LittleEndian.PutUint16(biases[(groupBase+group)*2:], hipFloat32ToBFloat16(-32*scale))
		for lane := 0; lane < hipGGUFQ6KGroupSize; lane++ {
			col := blockIndex*hipGGUFQ6KBlockSize + group*hipGGUFQ6KGroupSize + lane
			hipPutAffinePackedValue(weights, packedBase, col, 6, uint32(hipGGUFQ6KQuant(block, group, lane)))
		}
	}
	return nil
}

func hipGGUFQ4KScaleMin(scales []byte, group int) (uint8, uint8) {
	if group < 4 {
		return scales[group] & 63, scales[group+4] & 63
	}
	return (scales[group+4] & 0x0f) | ((scales[group-4] >> 6) << 4),
		(scales[group+4] >> 4) | ((scales[group] >> 6) << 4)
}

func hipGGUFQ5KQuant(block []byte, group, lane int) uint8 {
	il := group / 2
	highGroup := group%2 == 1
	ir := lane / 2
	sub := lane % 2
	qsIndex := 48 + 32*il + 2*ir + sub
	value := block[qsIndex]
	if highGroup {
		value >>= 4
	}
	value &= 0x0f
	mask := byte(1 << uint(2*il))
	if highGroup {
		mask <<= 1
	}
	if block[16+2*ir+sub]&mask != 0 {
		value += 16
	}
	return value & 0x1f
}

func hipGGUFQ6KQuant(block []byte, group, lane int) uint8 {
	ip := group / 8
	localGroup := group % 8
	laneOffset := 0
	if localGroup%2 == 1 {
		laneOffset = 16
	}
	quartile := localGroup / 2
	qlIndex := ip*64 + laneOffset + lane
	if quartile == 1 || quartile == 3 {
		qlIndex += 32
	}
	ql := block[qlIndex]
	if quartile >= 2 {
		ql >>= 4
	}
	qhIndex := 128 + ip*32 + laneOffset + lane
	qhShift := uint(quartile * 2)
	return (ql & 0x0f) | (((block[qhIndex] >> qhShift) & 0x03) << 4)
}

func hipPutAffinePackedValue(weights []byte, rowPackedBase, col, bits int, value uint32) {
	bitOffset := col * bits
	wordIndex := rowPackedBase + bitOffset/32
	shift := uint(bitOffset % 32)
	wordOffset := wordIndex * 4
	word := binary.LittleEndian.Uint32(weights[wordOffset:])
	word |= (value & ((1 << bits) - 1)) << shift
	binary.LittleEndian.PutUint32(weights[wordOffset:], word)
	if shift+uint(bits) > 32 {
		nextOffset := wordOffset + 4
		next := binary.LittleEndian.Uint32(weights[nextOffset:])
		next |= (value & ((1 << bits) - 1)) >> (32 - shift)
		binary.LittleEndian.PutUint32(weights[nextOffset:], next)
	}
}

func hipNativeTensorInfoIsGGUFQ4_0(info nativeTensorInfo) bool {
	return info.Type == hipGGUFQ4_0TensorType || core.Upper(info.TypeName) == "Q4_0"
}

func hipNativeTensorInfoIsGGUFQ4_1(info nativeTensorInfo) bool {
	return info.Type == hipGGUFQ4_1TensorType || core.Upper(info.TypeName) == "Q4_1"
}

func hipNativeTensorInfoIsGGUFQ5_1(info nativeTensorInfo) bool {
	return info.Type == hipGGUFQ5_1TensorType || core.Upper(info.TypeName) == "Q5_1"
}

func hipNativeTensorInfoIsGGUFQ4K(info nativeTensorInfo) bool {
	return info.Type == hipGGUFQ4KTensorType || core.Upper(info.TypeName) == "Q4_K"
}

func hipNativeTensorInfoIsGGUFQ5K(info nativeTensorInfo) bool {
	return info.Type == hipGGUFQ5KTensorType || core.Upper(info.TypeName) == "Q5_K"
}

func hipNativeTensorInfoIsGGUFQ6K(info nativeTensorInfo) bool {
	return info.Type == hipGGUFQ6KTensorType || core.Upper(info.TypeName) == "Q6_K"
}

func hipNativeTensorInfoIsGGUFQ8_0(info nativeTensorInfo) bool {
	return info.Type == hipGGUFQ8_0TensorType || core.Upper(info.TypeName) == "Q8_0"
}

func hipNativeTensorInfoIsF32(info nativeTensorInfo) bool {
	return info.Type == 0 || core.Upper(info.TypeName) == "F32"
}

func hipNativeTensorInfoIsF16(info nativeTensorInfo) bool {
	return info.Type == hipNativeTensorTypeF16 || core.Upper(info.TypeName) == "F16"
}

func hipNativeTensorInfoIsBF16(info nativeTensorInfo) bool {
	return info.Type == hipNativeTensorTypeBF16 || core.Upper(info.TypeName) == "BF16"
}

func hipNativeTensorInfoCanAliasAsBF16(info nativeTensorInfo) bool {
	return hipNativeTensorInfoIsF16(info) || hipNativeTensorInfoIsF32(info) || hipNativeTensorInfoIsBF16(info)
}

func hipNativeTensorInfoCanRepackAsAffine(info nativeTensorInfo) bool {
	return hipNativeTensorInfoIsGGUFQ4_0(info) ||
		hipNativeTensorInfoIsGGUFQ4_1(info) ||
		hipNativeTensorInfoIsGGUFQ4K(info) ||
		hipNativeTensorInfoIsGGUFQ5K(info) ||
		hipNativeTensorInfoIsGGUFQ6K(info) ||
		hipNativeTensorInfoIsGGUFQ8_0(info) ||
		hipNativeTensorInfoIsF32(info)
}

func hipGemma4CanonicalAffineBaseForGGUFWeightName(name string) (string, bool) {
	switch name {
	case "token_embd.weight":
		return "language_model.model.embed_tokens", true
	case "per_layer_token_embd.weight":
		return "language_model.model.embed_tokens_per_layer", true
	case "nextn.pre_projection.weight":
		return "pre_projection", true
	case "nextn.post_projection.weight":
		return "post_projection", true
	}
	const prefix = "blk."
	if !core.HasPrefix(name, prefix) {
		return "", false
	}
	parts := core.Split(core.TrimPrefix(name, prefix), ".")
	if len(parts) < 2 || parts[0] == "" {
		return "", false
	}
	layerPrefix := "language_model.model.layers." + parts[0]
	suffix := core.Join(".", parts[1:]...)
	switch suffix {
	case "attn_q.weight":
		return layerPrefix + ".self_attn.q_proj", true
	case "attn_k.weight":
		return layerPrefix + ".self_attn.k_proj", true
	case "attn_v.weight":
		return layerPrefix + ".self_attn.v_proj", true
	case "attn_output.weight":
		return layerPrefix + ".self_attn.o_proj", true
	case "ffn_gate.weight":
		return layerPrefix + ".mlp.gate_proj", true
	case "ffn_up.weight":
		return layerPrefix + ".mlp.up_proj", true
	case "ffn_down.weight":
		return layerPrefix + ".mlp.down_proj", true
	case "inp_gate.weight":
		return layerPrefix + ".per_layer_input_gate", true
	case "proj.weight":
		return layerPrefix + ".per_layer_projection", true
	default:
		return "", false
	}
}

func hipGemma4CanonicalBF16NameForGGUFName(name string) (string, bool) {
	switch name {
	case "output_norm.weight":
		return "language_model.model.norm.weight", true
	case "per_layer_model_proj.weight":
		return "language_model.model.per_layer_model_projection.weight", true
	case "per_layer_proj_norm.weight":
		return "language_model.model.per_layer_projection_norm.weight", true
	}
	const prefix = "blk."
	if !core.HasPrefix(name, prefix) {
		return "", false
	}
	parts := core.Split(core.TrimPrefix(name, prefix), ".")
	if len(parts) < 2 || parts[0] == "" {
		return "", false
	}
	layerPrefix := "language_model.model.layers." + parts[0]
	suffix := core.Join(".", parts[1:]...)
	switch suffix {
	case "attn_norm.weight":
		return layerPrefix + ".input_layernorm.weight", true
	case "layer_output_scale.weight":
		return layerPrefix + ".layer_scalar", true
	case "attn_q_norm.weight":
		return layerPrefix + ".self_attn.q_norm.weight", true
	case "attn_k_norm.weight":
		return layerPrefix + ".self_attn.k_norm.weight", true
	case "post_attention_norm.weight":
		return layerPrefix + ".post_attention_layernorm.weight", true
	case "ffn_norm.weight":
		return layerPrefix + ".pre_feedforward_layernorm.weight", true
	case "post_ffw_norm.weight":
		return layerPrefix + ".post_feedforward_layernorm.weight", true
	case "pre_ffw_norm_2.weight":
		return layerPrefix + ".pre_feedforward_layernorm_2.weight", true
	case "post_ffw_norm_1.weight":
		return layerPrefix + ".post_feedforward_layernorm_1.weight", true
	case "post_ffw_norm_2.weight":
		return layerPrefix + ".post_feedforward_layernorm_2.weight", true
	case "ffn_gate_inp.scale":
		return layerPrefix + ".router.scale", true
	case "ffn_down_exps.scale":
		return layerPrefix + ".router.per_expert_scale", true
	case "post_norm.weight":
		return layerPrefix + ".post_per_layer_input_norm.weight", true
	default:
		return "", false
	}
}
