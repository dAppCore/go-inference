// SPDX-Licence-Identifier: EUPL-1.2

package autoround

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

type PackedProjection struct {
	Tensor  PackTensor    `json:"tensor"`
	Weights PackedWeights `json:"weights"`
	Bias    []float32     `json:"bias,omitempty"`
}

func LoadPackedProjectionFromSafetensors(info PackInfo, weightFiles []string, tensorName string) (PackedProjection, error) {
	if !info.NativeTensorMap() {
		return PackedProjection{}, core.NewError("autoround: native tensor map is required")
	}
	tensor, ok := info.LookupTensor(tensorName)
	if !ok {
		return PackedProjection{}, core.NewError("autoround: tensor map does not contain: " + tensorName)
	}
	if err := tensor.Validate(); err != nil {
		return PackedProjection{}, err
	}
	index, err := safetensors.IndexFiles(weightFiles)
	if err != nil {
		return PackedProjection{}, core.E("autoround.load_projection", "index safetensors", err)
	}
	packedRef, err := lookupAutoRoundTensorRef(index, tensor.Packed, "U8", tensor.PackedBytes)
	if err != nil {
		return PackedProjection{}, err
	}
	scaleRef, err := lookupAutoRoundTensorRef(index, tensor.Scales, "F32", tensor.Groups)
	if err != nil {
		return PackedProjection{}, err
	}
	zeroRef, err := lookupAutoRoundTensorRef(index, tensor.ZeroPoints, "F32", tensor.Groups)
	if err != nil {
		return PackedProjection{}, err
	}
	packed, err := safetensors.ReadRefRaw(packedRef)
	if err != nil {
		return PackedProjection{}, core.E("autoround.load_projection", "read packed tensor", err)
	}
	scales, err := safetensors.ReadRefValues(scaleRef)
	if err != nil {
		return PackedProjection{}, core.E("autoround.load_projection", "read scale tensor", err)
	}
	zeroPoints, err := safetensors.ReadRefValues(zeroRef)
	if err != nil {
		return PackedProjection{}, core.E("autoround.load_projection", "read zero-point tensor", err)
	}
	projection := PackedProjection{
		Tensor: tensor,
		Weights: PackedWeights{
			Scheme:     info.Scheme,
			Format:     info.ExportFormat,
			Bits:       tensor.Bits,
			GroupSize:  tensor.GroupSize,
			Symmetric:  tensor.Symmetric,
			Shape:      core.SliceClone(tensor.Shape),
			Packed:     packed,
			Scales:     scales,
			ZeroPoints: zeroPoints,
			QMin:       tensor.QMin,
			QMax:       tensor.QMax,
		},
	}
	if tensor.Bias != "" {
		biasRef, err := lookupAutoRoundTensorRef(index, tensor.Bias, "F32", int(tensor.Shape[0]))
		if err != nil {
			return PackedProjection{}, err
		}
		projection.Bias, err = safetensors.ReadRefValues(biasRef)
		if err != nil {
			return PackedProjection{}, core.E("autoround.load_projection", "read bias tensor", err)
		}
	}
	if _, err := validatePackedWeights(projection.Weights); err != nil {
		return PackedProjection{}, err
	}
	return projection, nil
}

func (info PackInfo) LookupTensor(name string) (PackTensor, bool) {
	name = core.Trim(name)
	for _, tensor := range info.Tensors {
		if tensor.Name == name {
			return tensor, true
		}
	}
	return PackTensor{}, false
}

func lookupAutoRoundTensorRef(index safetensors.Index, name, dtype string, elements int) (safetensors.TensorRef, error) {
	ref, ok := index.Tensors[name]
	if !ok {
		return safetensors.TensorRef{}, core.NewError("autoround: tensor map missing safetensors tensor: " + name)
	}
	if ref.DType != dtype {
		return safetensors.TensorRef{}, core.Errorf("autoround: tensor %s dtype %s, expected %s", name, ref.DType, dtype)
	}
	if ref.Elements != elements {
		return safetensors.TensorRef{}, core.Errorf("autoround: tensor %s elements %d, expected %d", name, ref.Elements, elements)
	}
	return ref, nil
}
