// SPDX-Licence-Identifier: EUPL-1.2

package autoround

import core "dappco.re/go"

type PackedWeights struct {
	Scheme     Scheme       `json:"scheme,omitempty"`
	Format     ExportFormat `json:"format,omitempty"`
	Bits       int          `json:"bits"`
	GroupSize  int          `json:"group_size"`
	Symmetric  bool         `json:"sym"`
	Shape      []int32      `json:"shape,omitempty"`
	Packed     []byte       `json:"packed,omitempty"`
	Scales     []float32    `json:"scales,omitempty"`
	ZeroPoints []float32    `json:"zero_points,omitempty"`
	QMin       int          `json:"qmin"`
	QMax       int          `json:"qmax"`
}

func PackQuantizedWeights(weights QuantizedWeights, shape []int32) (PackedWeights, error) {
	if weights.Bits != 2 && weights.Bits != 3 && weights.Bits != 4 && weights.Bits != 8 {
		return PackedWeights{}, core.NewError("autoround: packed bits must be one of 2, 3, 4, or 8")
	}
	if len(weights.QValues) == 0 {
		return PackedWeights{}, core.NewError("autoround: qvalues are required")
	}
	if err := validatePackedShape(shape, len(weights.QValues)); err != nil {
		return PackedWeights{}, err
	}
	qmin, qmax := quantRange(QuantizeConfig{Bits: weights.Bits, Symmetric: weights.Symmetric})
	packed := PackedWeights{
		Scheme:     weights.Scheme,
		Format:     FormatAutoRound,
		Bits:       weights.Bits,
		GroupSize:  weights.GroupSize,
		Symmetric:  weights.Symmetric,
		Shape:      core.SliceClone(shape),
		Packed:     make([]byte, (len(weights.QValues)*weights.Bits+7)/8),
		Scales:     core.SliceClone(weights.Scales),
		ZeroPoints: core.SliceClone(weights.ZeroPoints),
		QMin:       qmin,
		QMax:       qmax,
	}
	for i, value := range weights.QValues {
		q := int(value)
		if q < qmin || q > qmax {
			return PackedWeights{}, core.Errorf("autoround: qvalue %d outside range [%d,%d]", q, qmin, qmax)
		}
		packUnsignedBits(packed.Packed, i, weights.Bits, uint32(q-qmin))
	}
	return packed, nil
}

func DequantizePackedWeights(weights PackedWeights) ([]float32, error) {
	elements, err := validatePackedWeights(weights)
	if err != nil {
		return nil, err
	}
	out := make([]float32, elements)
	for i := range out {
		group := i / weights.GroupSize
		q := int(unpackUnsignedBits(weights.Packed, i, weights.Bits)) + weights.QMin
		out[i] = (float32(q) - weights.ZeroPoints[group]) * weights.Scales[group]
	}
	return out, nil
}

func validatePackedWeights(weights PackedWeights) (int, error) {
	if weights.Bits != 2 && weights.Bits != 3 && weights.Bits != 4 && weights.Bits != 8 {
		return 0, core.NewError("autoround: packed bits must be one of 2, 3, 4, or 8")
	}
	if weights.GroupSize <= 0 {
		return 0, core.NewError("autoround: packed group size must be positive")
	}
	elements, err := packedShapeElements(weights.Shape)
	if err != nil {
		return 0, err
	}
	expectedPacked := (elements*weights.Bits + 7) / 8
	if len(weights.Packed) != expectedPacked {
		return 0, core.Errorf("autoround: packed length %d, expected %d", len(weights.Packed), expectedPacked)
	}
	expectedGroups := (elements + weights.GroupSize - 1) / weights.GroupSize
	if len(weights.Scales) != expectedGroups {
		return 0, core.Errorf("autoround: scale count %d, expected %d", len(weights.Scales), expectedGroups)
	}
	if len(weights.ZeroPoints) != expectedGroups {
		return 0, core.Errorf("autoround: zero-point count %d, expected %d", len(weights.ZeroPoints), expectedGroups)
	}
	return elements, nil
}

func validatePackedShape(shape []int32, values int) error {
	elements, err := packedShapeElements(shape)
	if err != nil {
		return err
	}
	if elements != values {
		return core.Errorf("autoround: shape elements %d, qvalues %d", elements, values)
	}
	return nil
}

func packedShapeElements(shape []int32) (int, error) {
	if len(shape) == 0 {
		return 0, core.NewError("autoround: packed shape is required")
	}
	elements := 1
	maxIntValue := int(^uint(0) >> 1)
	for _, dim := range shape {
		if dim <= 0 {
			return 0, core.NewError("autoround: packed shape dimensions must be positive")
		}
		if elements > maxIntValue/int(dim) {
			return 0, core.NewError("autoround: packed shape is too large")
		}
		elements *= int(dim)
	}
	return elements, nil
}

func packUnsignedBits(out []byte, index, bits int, value uint32) {
	bitOffset := index * bits
	byteIndex := bitOffset >> 3
	shift := bitOffset & 7
	out[byteIndex] |= byte(value << shift)
	if shift+bits > 8 {
		out[byteIndex+1] |= byte(value >> (8 - shift))
	}
}

func unpackUnsignedBits(in []byte, index, bits int) uint32 {
	bitOffset := index * bits
	byteIndex := bitOffset >> 3
	shift := bitOffset & 7
	word := uint32(in[byteIndex])
	if shift+bits > 8 {
		word |= uint32(in[byteIndex+1]) << 8
	}
	return (word >> shift) & uint32((1<<bits)-1)
}
