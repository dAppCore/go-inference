// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"encoding/binary"
	"math"

	core "dappco.re/go"
	pkgsafetensors "dappco.re/go/inference/model/safetensors"
)

// TensorMapping is a loaded GGUF tensor payload. Dense F32/F16/BF16 tensors
// view Data directly; quantized tensors that need dequantisation own their
// materialised tensor byte slices.
type TensorMapping struct {
	Data    []byte
	Tensors map[string]pkgsafetensors.Tensor
	close   func() error
}

// LoadTensors reads GGUF tensor payloads into the byte-native tensor shape used
// by pkg/native. Dense F32/F16/BF16 tensors stay as views into the file buffer;
// Q8_0 tensors are dequantised to F16 to mirror MLX's GGUF load behaviour.
func LoadTensors(path string) (*TensorMapping, error) {
	_, infos, dataStart, err := parseGGUFWithDataStart(path)
	if err != nil {
		return nil, err
	}
	data, closeMapping, err := mmapGGUFFile(path)
	if err != nil {
		return nil, err
	}
	if dataStart > uint64(len(data)) {
		_ = closeMapping()
		return nil, core.NewError("mlx: gguf tensor data section starts past EOF")
	}
	tensors := make(map[string]pkgsafetensors.Tensor, len(infos))
	for i := range infos {
		tensor, err := ggufLoadTensorData(data, dataStart, infos[i])
		if err != nil {
			_ = closeMapping()
			return nil, err
		}
		tensors[infos[i].Name] = tensor
	}
	return &TensorMapping{Data: data, Tensors: tensors, close: closeMapping}, nil
}

// Close releases references held by the mapping. It is intentionally a no-op
// for the backing bytes today; keeping the method lets callers pair GGUF and
// safetensors lifetimes the same way.
func (m *TensorMapping) Close() error {
	if m == nil {
		return nil
	}
	closeMapping := m.close
	m.Data = nil
	m.Tensors = nil
	m.close = nil
	if closeMapping != nil {
		return closeMapping()
	}
	return nil
}

func ggufLoadTensorData(data []byte, dataStart uint64, info TensorInfo) (pkgsafetensors.Tensor, error) {
	shape, elements, err := ggufTensorShapeElements(info)
	if err != nil {
		return pkgsafetensors.Tensor{}, err
	}
	dtype, size, err := ggufTensorNativeStorage(info, elements)
	if err != nil {
		return pkgsafetensors.Tensor{}, err
	}
	if info.Offset > ^uint64(0)-dataStart {
		return pkgsafetensors.Tensor{}, core.NewError("mlx: gguf tensor " + info.Name + " payload offset overflows")
	}
	start := dataStart + info.Offset
	if size > ^uint64(0)-start {
		return pkgsafetensors.Tensor{}, core.NewError("mlx: gguf tensor " + info.Name + " payload end overflows")
	}
	end := start + size
	if start > uint64(len(data)) || end > uint64(len(data)) {
		return pkgsafetensors.Tensor{}, core.NewError("mlx: gguf tensor " + info.Name + " payload is out of range")
	}
	payload := data[start:end]
	switch info.Type {
	case TensorTypeQ4_0:
		decoded, err := ggufDequantizeQ4_0ToF16(payload, elements)
		if err != nil {
			return pkgsafetensors.Tensor{}, err
		}
		payload = decoded
	case TensorTypeQ8_0:
		decoded, err := ggufDequantizeQ8_0ToF16(payload, elements)
		if err != nil {
			return pkgsafetensors.Tensor{}, err
		}
		payload = decoded
	}
	return pkgsafetensors.Tensor{Dtype: dtype, Shape: shape, Data: payload}, nil
}

func ggufTensorShapeElements(info TensorInfo) ([]int, uint64, error) {
	shape := make([]int, len(info.Shape))
	elements := uint64(1)
	maxInt := uint64(^uint(0) >> 1)
	for i, dim := range info.Shape {
		if dim > maxInt {
			return nil, 0, core.NewError("mlx: gguf tensor " + info.Name + " dimension overflows int")
		}
		if dim != 0 && elements > math.MaxUint64/dim {
			return nil, 0, core.NewError("mlx: gguf tensor " + info.Name + " element count overflows")
		}
		shape[i] = int(dim)
		elements *= dim
	}
	return shape, elements, nil
}

func ggufTensorNativeStorage(info TensorInfo, elements uint64) (string, uint64, error) {
	switch info.Type {
	case ggufTensorTypeF32:
		return ggufTensorNativeDenseStorage(info, "F32", elements, 4)
	case ggufTensorTypeF16:
		return ggufTensorNativeDenseStorage(info, "F16", elements, 2)
	case ggufTensorTypeBF16:
		return ggufTensorNativeDenseStorage(info, "BF16", elements, 2)
	case TensorTypeQ4_0:
		return ggufTensorNativeBlockStorage(info, elements, 32, 18, "Q4_0")
	case TensorTypeQ8_0:
		return ggufTensorNativeBlockStorage(info, elements, 32, 34, "Q8_0")
	default:
		return "", 0, core.NewError(core.Sprintf("mlx: gguf tensor %s has unsupported native load type %d", info.Name, info.Type))
	}
}

func ggufTensorNativeDenseStorage(info TensorInfo, dtype string, elements, elemBytes uint64) (string, uint64, error) {
	size, ok := ggufCheckedMul(elements, elemBytes)
	if !ok {
		return "", 0, core.NewError("mlx: gguf tensor " + info.Name + " byte size overflows")
	}
	return dtype, size, nil
}

func ggufTensorNativeBlockStorage(info TensorInfo, elements, blockElements, blockBytes uint64, typeName string) (string, uint64, error) {
	if elements%blockElements != 0 {
		return "", 0, core.NewError("mlx: gguf tensor " + info.Name + " " + typeName + " element count is not block-aligned")
	}
	size, ok := ggufCheckedMul(elements/blockElements, blockBytes)
	if !ok {
		return "", 0, core.NewError("mlx: gguf tensor " + info.Name + " byte size overflows")
	}
	return "F16", size, nil
}

func ggufCheckedMul(a, b uint64) (uint64, bool) {
	if a != 0 && b > ^uint64(0)/a {
		return 0, false
	}
	return a * b, true
}

func ggufDataAlignment(metadata map[string]any) uint64 {
	if alignment := metadataInt(metadata["general.alignment"]); alignment > 0 {
		return uint64(alignment)
	}
	return 32
}

func ggufDequantizeQ8_0ToF16(raw []byte, elements uint64) ([]byte, error) {
	if elements%32 != 0 || uint64(len(raw)) != (elements/32)*34 {
		return nil, core.NewError("mlx: gguf Q8_0 payload length does not match element count")
	}
	if elements > uint64((^uint(0)>>1)/2) {
		return nil, core.NewError("mlx: gguf Q8_0 output is too large")
	}
	out := make([]byte, int(elements)*2)
	blocks := elements / 32
	for b := range blocks {
		block := raw[b*34 : b*34+34]
		scale := ggufFloat16ToFloat32(binary.LittleEndian.Uint16(block[:2]))
		for i := range 32 {
			value := float32(int8(block[2+i])) * scale
			off := int((b*32 + uint64(i)) * 2)
			binary.LittleEndian.PutUint16(out[off:off+2], ggufFloat32ToFloat16(value))
		}
	}
	return out, nil
}

func ggufDequantizeQ4_0ToF16(raw []byte, elements uint64) ([]byte, error) {
	if elements%32 != 0 || uint64(len(raw)) != (elements/32)*18 {
		return nil, core.NewError("mlx: gguf Q4_0 payload length does not match element count")
	}
	if elements > uint64((^uint(0)>>1)/2) {
		return nil, core.NewError("mlx: gguf Q4_0 output is too large")
	}
	out := make([]byte, int(elements)*2)
	blocks := elements / 32
	for b := range blocks {
		block := raw[b*18 : b*18+18]
		scale := ggufFloat16ToFloat32(binary.LittleEndian.Uint16(block[:2]))
		qs := block[2:]
		for i := range 16 {
			packed := qs[i]
			lo := (int(packed&0x0f) - 8)
			hi := (int(packed>>4) - 8)
			loOff := int((b*32 + uint64(i)) * 2)
			hiOff := int((b*32 + uint64(i+16)) * 2)
			binary.LittleEndian.PutUint16(out[loOff:loOff+2], ggufFloat32ToFloat16(float32(lo)*scale))
			binary.LittleEndian.PutUint16(out[hiOff:hiOff+2], ggufFloat32ToFloat16(float32(hi)*scale))
		}
	}
	return out, nil
}

// ggufFloat32ToFloat16 encodes value as an IEEE-754 binary16 bit pattern. Used
// by the dequantise-on-load path above to re-encode a scaled Q4_0/Q8_0
// element back to f16 (mirroring MLX's native load format). Relocated here
// from quantize_kernels.go when the nine quantise kernels moved to the
// shared dappco.re/go/inference/gguf package — this is the one conversion
// helper the dequantise path still needs locally.
func ggufFloat32ToFloat16(value float32) uint16 {
	bits := math.Float32bits(value)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits >> 23) & 0xff)
	frac := bits & 0x7fffff
	if exp == 255 {
		if frac == 0 {
			return sign | 0x7c00
		}
		return sign | 0x7e00
	}
	exp = exp - 127 + 15
	if exp >= 31 {
		return sign | 0x7c00
	}
	if exp <= 0 {
		if exp < -10 {
			return sign
		}
		frac |= 0x800000
		shift := uint32(14 - exp)
		half := uint16(frac >> shift)
		if (frac>>(shift-1))&1 != 0 {
			half++
		}
		return sign | half
	}
	half := sign | uint16(exp<<10) | uint16(frac>>13)
	if frac&0x00001000 != 0 {
		half++
	}
	return half
}

func ggufFloat16ToFloat32(value uint16) float32 {
	sign := uint32(value>>15) & 0x1
	exp := int((value >> 10) & 0x1f)
	frac := uint32(value & 0x03ff)
	if exp == 0 {
		if frac == 0 {
			return math.Float32frombits(sign << 31)
		}
		for frac&0x0400 == 0 {
			frac <<= 1
			exp--
		}
		exp++
		frac &= 0x03ff
	} else if exp == 31 {
		return math.Float32frombits((sign << 31) | 0x7f800000 | (frac << 13))
	}
	exp = exp + (127 - 15)
	return math.Float32frombits((sign << 31) | (uint32(exp) << 23) | (frac << 13))
}
