// SPDX-Licence-Identifier: EUPL-1.2
package fp8

import (
	"context"
	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
	"testing"
)

func TestConvertSnapshot_Good(t *testing.T) {
	src, out := fixture(t), core.PathJoin(t.TempDir(), "out")
	result, err := ConvertSnapshot(context.Background(), src, out, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.QuantizedWeights != 1 {
		t.Fatalf("result = %+v", result)
	}
	idx, err := safetensors.IndexFiles([]string{result.WeightFile})
	if err != nil {
		t.Fatal(err)
	}
	if idx.Tensors["layer.weight"].DType != "F8_E4M3" || idx.Tensors["layer.weight_scale"].DType != "F32" {
		t.Fatalf("FP8 schema = %+v", idx.Tensors)
	}
	config := core.ReadFile(result.ConfigFile)
	if !config.OK || !core.Contains(string(config.Value.([]byte)), `"quant_method": "compressed-tensors"`) {
		t.Fatal("compressed-tensors config missing")
	}
}
func TestConvertSnapshot_Bad(t *testing.T) {
	if _, err := ConvertSnapshot(context.Background(), "", t.TempDir(), nil); err == nil {
		t.Fatal("empty source error = nil")
	}
}
func TestConvertSnapshot_Ugly(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := ConvertSnapshot(ctx, t.TempDir(), t.TempDir(), nil); err == nil {
		t.Fatal("cancelled error = nil")
	}
}
func fixture(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	values := make([]float32, 128)
	for i := range values {
		values[i] = float32(i-64) / 16
	}
	blob, err := safetensors.Encode(map[string]safetensors.Tensor{"layer.weight": {Dtype: "F32", Shape: []int{2, 64}, Data: safetensors.EncodeFloat32(values)}})
	if err != nil {
		t.Fatal(err)
	}
	if r := core.WriteFile(core.PathJoin(dir, "model.safetensors"), blob, 0o644); !r.OK {
		t.Fatal(r.Err())
	}
	return dir
}
