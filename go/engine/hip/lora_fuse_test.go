// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"encoding/binary"
	"math"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestFuseLoRAIntoModelPack_Good(t *testing.T) {
	root := t.TempDir()
	baseRoot := core.PathJoin(root, "base")
	adapterRoot := core.PathJoin(root, "adapter")
	outputRoot := core.PathJoin(root, "fused")
	basePath := core.PathJoin(baseRoot, "model.safetensors")
	adapterPath := core.PathJoin(adapterRoot, "adapter.safetensors")

	core.RequireNoError(t, rocmWriteFuseSafetensors(basePath, []rocmFuseWriteTensor{
		{Name: "layer.weight", DType: "F32", Shape: []uint64{2, 3}, Data: loraFuseF32Bytes([]float32{1, 2, 3, 4, 5, 6})},
		{Name: "untouched.weight", DType: "F32", Shape: []uint64{1}, Data: loraFuseF32Bytes([]float32{9})},
	}))
	core.RequireNoError(t, rocmWriteFuseSafetensors(adapterPath, []rocmFuseWriteTensor{
		{Name: "layer.lora_a.weight", DType: "F32", Shape: []uint64{1, 3}, Data: loraFuseF32Bytes([]float32{1, 2, 3})},
		{Name: "layer.lora_b.weight", DType: "F32", Shape: []uint64{2, 1}, Data: loraFuseF32Bytes([]float32{4, 5})},
	}))

	result, err := FuseLoRAIntoModelPack(context.Background(), LoRAFuseOptions{
		BasePath:    baseRoot,
		AdapterPath: adapterRoot,
		OutputPath:  outputRoot,
		Adapter: inference.AdapterIdentity{
			Path:  "seeded-fixture",
			Rank:  1,
			Alpha: 2,
		},
		Labels: map[string]string{"fixture": "seeded-varied-fills"},
	})

	core.RequireNoError(t, err)
	core.AssertNotNil(t, result)
	if result == nil {
		t.Fatal("FuseLoRAIntoModelPack returned nil result")
	}
	core.AssertEqual(t, 1, result.FusedWeights)
	core.AssertEqual(t, []string{"layer.weight"}, result.FusedWeightKeys)
	core.AssertEqual(t, []string{"layer"}, result.FusedLayers)
	core.AssertEqual(t, "seeded-varied-fills", result.Labels["fixture"])
	core.AssertEqual(t, "dense_f32_cpu", result.Labels["fuse_runtime"])
	core.AssertEqual(t, 1, len(result.WeightFiles))

	index, err := rocmReadFuseSafetensorsIndex(result.WeightFiles[0])
	core.RequireNoError(t, err)
	fused, err := rocmReadFuseTensorF32(index["layer.weight"])
	core.RequireNoError(t, err)
	core.AssertEqual(t, []float32{9, 18, 27, 14, 25, 36}, fused)
	untouched, err := rocmReadFuseTensorF32(index["untouched.weight"])
	core.RequireNoError(t, err)
	core.AssertEqual(t, []float32{9}, untouched)
}

func TestFuseLoRAIntoModelPack_Bad(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	result, err := FuseLoRAIntoModelPack(ctx, LoRAFuseOptions{})

	core.AssertNil(t, result)
	core.AssertError(t, err)
}

func TestFuseLoRAIntoModelPack_Ugly(t *testing.T) {
	result, err := FuseLoRAIntoModelPack(nil, LoRAFuseOptions{})

	core.AssertNil(t, result)
	core.AssertError(t, err)
}

func loraFuseF32Bytes(values []float32) []byte {
	data := make([]byte, len(values)*4)
	for i, value := range values {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(value))
	}
	return data
}
