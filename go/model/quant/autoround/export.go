// SPDX-Licence-Identifier: EUPL-1.2

package autoround

import (
	"context"
	"encoding/binary"
	"math"

	core "dappco.re/go"
)

type packedProjectionTensor struct {
	name  string
	dtype string
	shape []int
	raw   []byte
}

type NativePackExportResult struct {
	ConfigPath  string `json:"config_path"`
	WeightPath  string `json:"weight_path"`
	TensorCount int    `json:"tensor_count"`
}

// WritePackedProjectionSafetensors writes one native AutoRound packed
// projection to safetensors. Full model-pack export orchestration can layer
// over this primitive without re-encoding individual tensor payloads.
func WritePackedProjectionSafetensors(ctx context.Context, path string, projection PackedProjection) error {
	return WritePackedProjectionsSafetensors(ctx, path, []PackedProjection{projection})
}

// WriteNativePack writes a directory-level AutoRound native pack sidecar plus a
// model.safetensors payload. It intentionally does not emit GGUF or model config
// files; model-loader wiring can consume the resulting tensor map separately.
func WriteNativePack(ctx context.Context, root string, info PackInfo, projections []PackedProjection) (NativePackExportResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return NativePackExportResult{}, err
	}
	root = core.Trim(root)
	if root == "" {
		return NativePackExportResult{}, core.NewError("autoround: native pack root is empty")
	}
	if len(projections) == 0 {
		return NativePackExportResult{}, core.NewError("autoround: native pack requires at least one projection")
	}
	info = nativePackInfoForExport(info, projections)
	if err := info.Validate(); err != nil {
		return NativePackExportResult{}, err
	}

	if result := core.MkdirAll(root, 0o755); !result.OK {
		return NativePackExportResult{}, result.Value.(error)
	}
	weightPath := core.PathJoin(root, "model.safetensors")
	if err := WritePackedProjectionsSafetensors(ctx, weightPath, projections); err != nil {
		return NativePackExportResult{}, err
	}
	configPath := core.PathJoin(root, PackConfigFileAutoRound)
	encoded := core.JSONMarshalIndent(info, "", "  ")
	if !encoded.OK {
		return NativePackExportResult{}, encoded.Value.(error)
	}
	if result := core.WriteFile(configPath, encoded.Value.([]byte), 0o644); !result.OK {
		return NativePackExportResult{}, result.Value.(error)
	}
	return NativePackExportResult{
		ConfigPath:  configPath,
		WeightPath:  weightPath,
		TensorCount: info.TensorCount,
	}, nil
}

// WritePackedProjectionsSafetensors writes multiple native AutoRound packed
// projections to one safetensors file. This is the native pack-export primitive
// used before higher-level model config and sharding orchestration.
func WritePackedProjectionsSafetensors(ctx context.Context, path string, projections []PackedProjection) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if core.Trim(path) == "" {
		return core.NewError("autoround: safetensors export path is empty")
	}
	if len(projections) == 0 {
		return core.NewError("autoround: safetensors export requires at least one projection")
	}
	tensors := make([]packedProjectionTensor, 0, len(projections)*3)
	for _, projection := range projections {
		projectionTensors, err := packedProjectionSafetensorsTensors(projection)
		if err != nil {
			return err
		}
		tensors = append(tensors, projectionTensors...)
	}
	return writeAutoRoundRawSafetensors(ctx, path, tensors)
}

func nativePackInfoForExport(info PackInfo, projections []PackedProjection) PackInfo {
	info.QuantMethod = QuantMethodAutoRound
	info.PackingFormat = string(FormatAutoRound)
	info.ExportFormat = FormatAutoRound
	if info.Bits == 0 {
		info.Bits = projections[0].Weights.Bits
	}
	if info.GroupSize == 0 {
		info.GroupSize = projections[0].Weights.GroupSize
	}
	if !info.Symmetric {
		info.Symmetric = projections[0].Weights.Symmetric
	}
	if info.Scheme == "" {
		info.Scheme = projections[0].Weights.Scheme
	}
	info.Tensors = make([]PackTensor, 0, len(projections))
	for _, projection := range projections {
		tensor := projection.Tensor
		tensor.normalise(info)
		info.Tensors = append(info.Tensors, tensor)
	}
	info.normalise()
	return info
}

func packedProjectionSafetensorsTensors(projection PackedProjection) ([]packedProjectionTensor, error) {
	tensor := projection.Tensor
	tensor.normalise(PackInfo{
		Bits:          projection.Weights.Bits,
		GroupSize:     projection.Weights.GroupSize,
		Symmetric:     projection.Weights.Symmetric,
		QuantMethod:   QuantMethodAutoRound,
		PackingFormat: string(FormatAutoRound),
	})
	if err := tensor.Validate(); err != nil {
		return nil, err
	}
	if err := validateProjectionWeightsMatch(tensor, projection.Weights); err != nil {
		return nil, err
	}
	if tensor.Bias != "" && len(projection.Bias) != int(tensor.Shape[0]) {
		return nil, core.Errorf("autoround: bias length %d, expected %d", len(projection.Bias), tensor.Shape[0])
	}

	// The packed payload is read-only here: writeAutoRoundRawSafetensors only
	// copies it via append, and the caller never mutates the source projection.
	// Alias it instead of cloning a full packed buffer per projection.
	tensors := []packedProjectionTensor{
		{name: tensor.Packed, dtype: "U8", shape: []int{tensor.PackedBytes}, raw: projection.Weights.Packed},
		{name: tensor.Scales, dtype: "F32", shape: []int{tensor.Groups}, raw: encodeAutoRoundF32(projection.Weights.Scales)},
		{name: tensor.ZeroPoints, dtype: "F32", shape: []int{tensor.Groups}, raw: encodeAutoRoundF32(projection.Weights.ZeroPoints)},
	}
	if tensor.Bias != "" {
		tensors = append(tensors, packedProjectionTensor{
			name:  tensor.Bias,
			dtype: "F32",
			shape: []int{int(tensor.Shape[0])},
			raw:   encodeAutoRoundF32(projection.Bias),
		})
	}
	return tensors, nil
}

func validateProjectionWeightsMatch(tensor PackTensor, weights PackedWeights) error {
	if _, err := validatePackedWeights(weights); err != nil {
		return err
	}
	if weights.Bits != tensor.Bits {
		return core.Errorf("autoround: packed bits %d, expected %d", weights.Bits, tensor.Bits)
	}
	if weights.GroupSize != tensor.GroupSize {
		return core.Errorf("autoround: packed group size %d, expected %d", weights.GroupSize, tensor.GroupSize)
	}
	if weights.Symmetric != tensor.Symmetric {
		return core.Errorf("autoround: packed symmetry %v, expected %v", weights.Symmetric, tensor.Symmetric)
	}
	if len(weights.Shape) != len(tensor.Shape) {
		return core.Errorf("autoround: packed shape rank %d, expected %d", len(weights.Shape), len(tensor.Shape))
	}
	for i := range weights.Shape {
		if weights.Shape[i] != tensor.Shape[i] {
			return core.Errorf("autoround: packed shape[%d] %d, expected %d", i, weights.Shape[i], tensor.Shape[i])
		}
	}
	return nil
}

func encodeAutoRoundF32(values []float32) []byte {
	raw := make([]byte, len(values)*4)
	for i, value := range values {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(value))
	}
	return raw
}

func writeAutoRoundRawSafetensors(ctx context.Context, path string, tensors []packedProjectionTensor) error {
	type entry struct {
		DType       string `json:"dtype"`
		Shape       []int  `json:"shape"`
		DataOffsets []int  `json:"data_offsets"`
	}
	header := make(map[string]entry, len(tensors))
	var data []byte
	for _, tensor := range tensors {
		if err := ctx.Err(); err != nil {
			return err
		}
		if core.Trim(tensor.name) == "" {
			return core.NewError("autoround: safetensors tensor name is empty")
		}
		if _, ok := header[tensor.name]; ok {
			return core.NewError("autoround: duplicate safetensors tensor: " + tensor.name)
		}
		start := len(data)
		data = append(data, tensor.raw...)
		header[tensor.name] = entry{
			DType:       tensor.dtype,
			Shape:       core.SliceClone(tensor.shape),
			DataOffsets: []int{start, len(data)},
		}
	}
	encoded := core.JSONMarshal(header)
	if !encoded.OK {
		return encoded.Value.(error)
	}
	headerBytes := encoded.Value.([]byte)

	parent := core.PathDir(path)
	if result := core.MkdirAll(parent, 0o755); !result.OK {
		return result.Value.(error)
	}
	created := core.OpenFile(path, core.O_CREATE|core.O_WRONLY|core.O_TRUNC, 0o644)
	if !created.OK {
		return created.Value.(error)
	}
	file := created.Value.(*core.OSFile)
	defer file.Close()

	var headerLen [8]byte
	binary.LittleEndian.PutUint64(headerLen[:], uint64(len(headerBytes)))
	if _, err := file.Write(headerLen[:]); err != nil {
		return err
	}
	if _, err := file.Write(headerBytes); err != nil {
		return err
	}
	_, err := file.Write(data)
	return err
}
