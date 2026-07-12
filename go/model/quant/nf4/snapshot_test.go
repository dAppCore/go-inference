// SPDX-Licence-Identifier: EUPL-1.2
package nf4

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
	idx, err := safetensors.IndexFiles([]string{result.WeightFile})
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"layer.weight", "layer.weight.absmax", "layer.weight.quant_map", "layer.weight.quant_state.bitsandbytes__nf4"} {
		if _, ok := idx.Tensors[name]; !ok {
			t.Fatalf("missing %s", name)
		}
	}
	ref := idx.Tensors["layer.weight"]
	if ref.DType != "U8" || ref.Shape[0] != 64 || ref.Shape[1] != 1 {
		t.Fatalf("packed weight = %s %v", ref.DType, ref.Shape)
	}
	config := core.ReadFile(result.ConfigFile)
	if !config.OK || !core.Contains(string(config.Value.([]byte)), `"bnb_4bit_use_double_quant": false`) {
		t.Fatal("bnb config missing")
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
