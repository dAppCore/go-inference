// SPDX-Licence-Identifier: EUPL-1.2

package gptq

import (
	"context"
	"encoding/binary"
	"math"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

func TestConvertSnapshot_Good(t *testing.T) {
	src := core.PathJoin(t.TempDir(), "toy-bf16")
	out := core.PathJoin(t.TempDir(), "toy-gptq")
	if r := core.MkdirAll(src, 0o755); !r.OK {
		t.Fatal(r.Err())
	}
	values := make([]float32, 32*64)
	for i := range values {
		values[i] = 0.02*float32(i%19-9) + 0.0001*float32(i)
	}
	blob, err := safetensors.Encode(map[string]safetensors.Tensor{
		"model.layers.0.self_attn.q_proj.weight": {Dtype: "F32", Shape: []int{32, 64}, Data: safetensors.EncodeFloat32(values)},
		"model.norm.weight":                      {Dtype: "F32", Shape: []int{32}, Data: safetensors.EncodeFloat32(make([]float32, 32))},
	})
	if err != nil {
		t.Fatal(err)
	}
	for name, data := range map[string][]byte{
		"model.safetensors": blob,
		"config.json":       []byte(`{"model_type":"llama"}`),
		"tokenizer.json":    []byte(`{}`),
	} {
		if r := core.WriteFile(core.PathJoin(src, name), data, 0o644); !r.OK {
			t.Fatal(r.Err())
		}
	}
	result, err := ConvertSnapshot(context.Background(), src, out, Options{Bits: 4, GroupSize: 32, Symmetric: true}, nil)
	if err != nil {
		t.Fatalf("ConvertSnapshot() error = %v", err)
	}
	if result.QuantizedWeights != 1 || result.PassthroughCount != 1 {
		t.Fatalf("ConvertSnapshot() result = %+v", result)
	}
	idx, err := safetensors.IndexFiles([]string{core.PathJoin(out, "model.safetensors")})
	if err != nil {
		t.Fatal(err)
	}
	base := "model.layers.0.self_attn.q_proj"
	assertRef(t, idx, base+".qweight", "I32", []uint64{8, 32})
	assertRef(t, idx, base+".qzeros", "I32", []uint64{2, 4})
	assertRef(t, idx, base+".scales", "F16", []uint64{2, 32})
	assertRef(t, idx, base+".g_idx", "I32", []uint64{64})
	if _, ok := idx.Tensors[base+".weight"]; ok {
		t.Fatal("dense weight remained in GPTQ output")
	}
	config := core.ReadFile(core.PathJoin(out, "quantize_config.json"))
	if !config.OK || !core.Contains(string(config.Value.([]byte)), `"quant_method": "gptq"`) {
		t.Fatalf("quantize_config.json = %v", config.Value)
	}
	if len(core.PathGlob(core.PathJoin(out, "tokenizer.json"))) != 1 {
		t.Fatal("tokenizer sidecar was not copied")
	}
	cache := safetensors.NewShardCache()
	defer cache.Close()
	read := func(name string) []byte {
		raw, readErr := cache.ReadRefRaw(idx.Tensors[name])
		if readErr != nil {
			t.Fatal(readErr)
		}
		return raw
	}
	qweight := decodeU32(read(base + ".qweight"))
	qzeros := decodeU32(read(base + ".qzeros"))
	scales, err := safetensors.DecodeFloat32("F16", read(base+".scales"), 2*32)
	if err != nil {
		t.Fatal(err)
	}
	graw := read(base + ".g_idx")
	gidx := make([]int32, len(graw)/4)
	for i := range gidx {
		gidx[i] = int32(binary.LittleEndian.Uint32(graw[i*4:]))
	}
	reconstructed, err := Dequantize(Tensor{Shape: [2]int{32, 64}, Bits: 4, GroupSize: 32, Symmetric: true, QWeight: qweight, QZeros: qzeros, Scales: scales, GIdx: gidx})
	if err != nil {
		t.Fatal(err)
	}
	var maxError float64
	for i := range values {
		maxError = math.Max(maxError, math.Abs(float64(values[i]-reconstructed[i])))
	}
	if maxError > 0.1 {
		t.Fatalf("written GPTQ dequantisation maximum error %g exceeds scheme bound", maxError)
	}
}

func decodeU32(raw []byte) []uint32 {
	out := make([]uint32, len(raw)/4)
	for i := range out {
		out[i] = binary.LittleEndian.Uint32(raw[i*4:])
	}
	return out
}

func TestConvertSnapshot_Bad(t *testing.T) {
	if _, err := ConvertSnapshot(context.Background(), "", t.TempDir(), Options{}, nil); err == nil {
		t.Fatal("ConvertSnapshot(empty source) error = nil")
	}
}

func TestConvertSnapshot_Ugly(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := ConvertSnapshot(ctx, t.TempDir(), t.TempDir(), Options{}, nil); err == nil {
		t.Fatal("ConvertSnapshot(cancelled) error = nil")
	}
}

func assertRef(t *testing.T, idx safetensors.Index, name, dtype string, shape []uint64) {
	t.Helper()
	ref, ok := idx.Tensors[name]
	if !ok {
		t.Fatalf("missing tensor %q", name)
	}
	if ref.DType != dtype || len(ref.Shape) != len(shape) {
		t.Fatalf("%s = %s %v, want %s %v", name, ref.DType, ref.Shape, dtype, shape)
	}
	for i := range shape {
		if ref.Shape[i] != shape[i] {
			t.Fatalf("%s shape = %v, want %v", name, ref.Shape, shape)
		}
	}
}
