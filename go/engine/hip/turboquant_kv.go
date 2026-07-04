// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"math"

	core "dappco.re/go"
)

const (
	rocmTurboQuantKVMode              = "turboquant-kv"
	rocmTurboQuantKVDefaultSeed       = uint64(0x9e3779b97f4a7c15)
	rocmTurboQuantKVDefaultGroupSize  = 64
	rocmTurboQuantKVDefaultGroupLabel = "64"
	rocmTurboQuantKVDefaultBitsNum    = 7
	rocmTurboQuantKVDefaultBitsDenom  = 2
	rocmTurboQuantKVResidualPrecision = "fp32-group-mean"
)

type rocmTurboQuantKVDescriptor struct {
	BitsNumerator      int
	BitsDenominator    int
	GroupSize          int
	Seed               uint64
	ResidualCorrection bool
}

type rocmTurboQuantKVTensor struct {
	Descriptor rocmTurboQuantKVDescriptor
	Length     int
	Packed     []byte
	Scales     []float32
	Residuals  []float32
	SizeBytes  uint64
}

type rocmTurboQuantKVWorkspace struct {
	quantized []int8
	packed    []byte
	scales    []float32
	residuals []float32
	decoded   []float32
}

func defaultROCmTurboQuantKVDescriptor() rocmTurboQuantKVDescriptor {
	return rocmTurboQuantKVDescriptor{
		BitsNumerator:      rocmTurboQuantKVDefaultBitsNum,
		BitsDenominator:    rocmTurboQuantKVDefaultBitsDenom,
		GroupSize:          rocmTurboQuantKVDefaultGroupSize,
		Seed:               rocmTurboQuantKVDefaultSeed,
		ResidualCorrection: true,
	}
}

func encodeROCmTurboQuantKV(values []float32, desc rocmTurboQuantKVDescriptor) (rocmTurboQuantKVTensor, error) {
	var workspace rocmTurboQuantKVWorkspace
	return encodeROCmTurboQuantKVInto(values, desc, &workspace)
}

func encodeROCmTurboQuantKVInto(values []float32, desc rocmTurboQuantKVDescriptor, workspace *rocmTurboQuantKVWorkspace) (rocmTurboQuantKVTensor, error) {
	if len(values) == 0 {
		return rocmTurboQuantKVTensor{}, core.E("rocm.TurboQuantKV.Encode", "values are required", nil)
	}
	if !rocmFloat32SliceFinite(values) {
		return rocmTurboQuantKVTensor{}, core.E("rocm.TurboQuantKV.Encode", "values must be finite", nil)
	}
	if err := validateROCmTurboQuantKVDescriptor(desc); err != nil {
		return rocmTurboQuantKVTensor{}, err
	}
	groupCount := (len(values) + desc.GroupSize - 1) / desc.GroupSize
	if workspace == nil {
		workspace = &rocmTurboQuantKVWorkspace{}
	}
	scales := workspace.float32s(&workspace.scales, groupCount)
	residuals := workspace.float32s(&workspace.residuals, groupCount)
	quantized := workspace.int8s(&workspace.quantized, len(values))
	for group := 0; group < groupCount; group++ {
		start := group * desc.GroupSize
		end := start + desc.GroupSize
		if end > len(values) {
			end = len(values)
		}
		maxAbs := float32(0)
		for i := start; i < end; i++ {
			rotated := values[i] * rocmTurboQuantKVSign(desc.Seed, i)
			if abs := float32(math.Abs(float64(rotated))); abs > maxAbs {
				maxAbs = abs
			}
		}
		scale := float32(1)
		if maxAbs > 0 {
			scale = maxAbs / float32(rocmTurboQuantKVGroupPositiveRange(desc, start, end))
		}
		scales[group] = scale
		residualSum := float32(0)
		for i := start; i < end; i++ {
			bits := rocmTurboQuantKVBitWidth(desc, i)
			rotated := values[i] * rocmTurboQuantKVSign(desc.Seed, i)
			quantized[i] = int8(clampInt(int(math.Round(float64(rotated/scale))), rocmTurboQuantKVMin(bits), rocmTurboQuantKVMax(bits)))
			decoded := float32(quantized[i]) * scale * rocmTurboQuantKVSign(desc.Seed, i)
			residualSum += values[i] - decoded
		}
		if desc.ResidualCorrection {
			residuals[group] = residualSum / float32(end-start)
		}
	}
	packed, err := packROCmTurboQuantKVSignedBitsInto(quantized, desc, workspace)
	if err != nil {
		return rocmTurboQuantKVTensor{}, err
	}
	if !desc.ResidualCorrection {
		residuals = nil
	}
	return rocmTurboQuantKVTensor{
		Descriptor: desc,
		Length:     len(values),
		Packed:     packed,
		Scales:     scales,
		Residuals:  residuals,
		SizeBytes:  uint64(len(packed) + len(scales)*4 + len(residuals)*4),
	}, nil
}

func (tensor rocmTurboQuantKVTensor) decode() ([]float32, error) {
	var workspace rocmTurboQuantKVWorkspace
	return tensor.decodeInto(&workspace)
}

func (tensor rocmTurboQuantKVTensor) decodeInto(workspace *rocmTurboQuantKVWorkspace) ([]float32, error) {
	if tensor.Length <= 0 {
		return nil, core.E("rocm.TurboQuantKV.Decode", "tensor length is required", nil)
	}
	if err := validateROCmTurboQuantKVDescriptor(tensor.Descriptor); err != nil {
		return nil, err
	}
	groupCount := (tensor.Length + tensor.Descriptor.GroupSize - 1) / tensor.Descriptor.GroupSize
	if len(tensor.Scales) != groupCount {
		return nil, core.E("rocm.TurboQuantKV.Decode", "scale count must match group count", nil)
	}
	if tensor.Descriptor.ResidualCorrection && len(tensor.Residuals) != groupCount {
		return nil, core.E("rocm.TurboQuantKV.Decode", "residual count must match group count", nil)
	}
	if workspace == nil {
		workspace = &rocmTurboQuantKVWorkspace{}
	}
	quantized, err := unpackROCmTurboQuantKVSignedBitsInto(tensor.Packed, tensor.Descriptor, tensor.Length, workspace)
	if err != nil {
		return nil, err
	}
	out := workspace.float32s(&workspace.decoded, tensor.Length)
	for i, value := range quantized {
		group := i / tensor.Descriptor.GroupSize
		correction := float32(0)
		if tensor.Descriptor.ResidualCorrection {
			correction = tensor.Residuals[group]
		}
		out[i] = float32(value)*tensor.Scales[group]*rocmTurboQuantKVSign(tensor.Descriptor.Seed, i) + correction
	}
	return out, nil
}

func validateROCmTurboQuantKVDescriptor(desc rocmTurboQuantKVDescriptor) error {
	if desc.BitsDenominator <= 0 {
		return core.E("rocm.TurboQuantKV.Descriptor", "bits denominator must be positive", nil)
	}
	if desc.BitsNumerator < 2*desc.BitsDenominator || desc.BitsNumerator > 8*desc.BitsDenominator {
		return core.E("rocm.TurboQuantKV.Descriptor", "average bits must be between 2 and 8", nil)
	}
	if desc.GroupSize <= 0 || desc.GroupSize&(desc.GroupSize-1) != 0 {
		return core.E("rocm.TurboQuantKV.Descriptor", "group size must be a positive power of two", nil)
	}
	for i := 0; i < desc.BitsDenominator; i++ {
		bits := rocmTurboQuantKVBitWidth(desc, i)
		if bits < 2 || bits > 8 {
			return core.E("rocm.TurboQuantKV.Descriptor", "per-channel bit width must be between 2 and 8", nil)
		}
	}
	return nil
}

func packROCmTurboQuantKVSignedBits(values []int8, desc rocmTurboQuantKVDescriptor) ([]byte, error) {
	var workspace rocmTurboQuantKVWorkspace
	return packROCmTurboQuantKVSignedBitsInto(values, desc, &workspace)
}

func packROCmTurboQuantKVSignedBitsInto(values []int8, desc rocmTurboQuantKVDescriptor, workspace *rocmTurboQuantKVWorkspace) ([]byte, error) {
	if err := validateROCmTurboQuantKVDescriptor(desc); err != nil {
		return nil, err
	}
	totalBits := rocmTurboQuantKVTotalBits(desc, len(values))
	if workspace == nil {
		workspace = &rocmTurboQuantKVWorkspace{}
	}
	packed := workspace.bytes(&workspace.packed, (totalBits+7)/8)
	for i := range packed {
		packed[i] = 0
	}
	bitOffset := 0
	for i, value := range values {
		bits := rocmTurboQuantKVBitWidth(desc, i)
		if int(value) < rocmTurboQuantKVMin(bits) || int(value) > rocmTurboQuantKVMax(bits) {
			return nil, core.E("rocm.TurboQuantKV.Pack", "quantized value is outside bit width", nil)
		}
		raw := int(value)
		if raw < 0 {
			raw += 1 << bits
		}
		for bit := 0; bit < bits; bit++ {
			if raw&(1<<bit) != 0 {
				packed[(bitOffset+bit)/8] |= byte(1 << ((bitOffset + bit) % 8))
			}
		}
		bitOffset += bits
	}
	return packed, nil
}

func unpackROCmTurboQuantKVSignedBits(packed []byte, desc rocmTurboQuantKVDescriptor, count int) ([]int8, error) {
	var workspace rocmTurboQuantKVWorkspace
	return unpackROCmTurboQuantKVSignedBitsInto(packed, desc, count, &workspace)
}

func unpackROCmTurboQuantKVSignedBitsInto(packed []byte, desc rocmTurboQuantKVDescriptor, count int, workspace *rocmTurboQuantKVWorkspace) ([]int8, error) {
	if count < 0 {
		return nil, core.E("rocm.TurboQuantKV.Unpack", "count must be non-negative", nil)
	}
	if err := validateROCmTurboQuantKVDescriptor(desc); err != nil {
		return nil, err
	}
	requiredBytes := (rocmTurboQuantKVTotalBits(desc, count) + 7) / 8
	if len(packed) < requiredBytes {
		return nil, core.E("rocm.TurboQuantKV.Unpack", core.Sprintf("packed values need %d bytes, got %d", requiredBytes, len(packed)), nil)
	}
	if workspace == nil {
		workspace = &rocmTurboQuantKVWorkspace{}
	}
	out := workspace.int8s(&workspace.quantized, count)
	bitOffset := 0
	for i := 0; i < count; i++ {
		bits := rocmTurboQuantKVBitWidth(desc, i)
		raw := 0
		for bit := 0; bit < bits; bit++ {
			if packed[(bitOffset+bit)/8]&(1<<((bitOffset+bit)%8)) != 0 {
				raw |= 1 << bit
			}
		}
		if raw&(1<<(bits-1)) != 0 {
			raw -= 1 << bits
		}
		out[i] = int8(raw)
		bitOffset += bits
	}
	return out, nil
}

func (workspace *rocmTurboQuantKVWorkspace) bytes(values *[]byte, length int) []byte {
	if cap(*values) < length {
		*values = make([]byte, length)
	}
	return (*values)[:length]
}

func (workspace *rocmTurboQuantKVWorkspace) int8s(values *[]int8, length int) []int8 {
	if cap(*values) < length {
		*values = make([]int8, length)
	}
	return (*values)[:length]
}

func (workspace *rocmTurboQuantKVWorkspace) float32s(values *[]float32, length int) []float32 {
	if cap(*values) < length {
		*values = make([]float32, length)
	}
	return (*values)[:length]
}

func rocmTurboQuantKVTotalBits(desc rocmTurboQuantKVDescriptor, count int) int {
	if count <= 0 {
		return 0
	}
	return count * desc.BitsNumerator / desc.BitsDenominator
}

func rocmTurboQuantKVBitWidth(desc rocmTurboQuantKVDescriptor, index int) int {
	if index < 0 {
		index = 0
	}
	return ((index+1)*desc.BitsNumerator)/desc.BitsDenominator - (index*desc.BitsNumerator)/desc.BitsDenominator
}

func rocmTurboQuantKVGroupPositiveRange(desc rocmTurboQuantKVDescriptor, start, end int) int {
	positiveRange := rocmTurboQuantKVMax(8)
	for i := start; i < end; i++ {
		if positive := rocmTurboQuantKVMax(rocmTurboQuantKVBitWidth(desc, i)); positive < positiveRange {
			positiveRange = positive
		}
	}
	if positiveRange <= 0 {
		return 1
	}
	return positiveRange
}

func rocmTurboQuantKVMin(bits int) int { return -(1 << (bits - 1)) }

func rocmTurboQuantKVMax(bits int) int { return (1 << (bits - 1)) - 1 }

func rocmTurboQuantKVSign(seed uint64, index int) float32 {
	value := uint64(index) + seed + 0x9e3779b97f4a7c15
	value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9
	value = (value ^ (value >> 27)) * 0x94d049bb133111eb
	value ^= value >> 31
	if value&1 == 0 {
		return 1
	}
	return -1
}
