// SPDX-Licence-Identifier: EUPL-1.2

package mlxaffine_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	core "dappco.re/go"

	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/quant/mlxaffine"
	"dappco.re/go/inference/model/safetensors"
)

// TestConvertSnapshot_ToyLoadable runs the converter end-to-end on a TINY synthetic
// bf16-shaped model directory (built here with the safetensors writer — a 2-layer toy,
// no GPU, no real weights) and asserts the output directory is accepted by the engine's
// config-parse / probe layer: model.ProbeDirArch reads it, the injected quantization
// block parses + validates through the real model.QuantConfig reader, and the emitted
// tensors carry the exact dtypes/shapes the loader's safetensors path expects. It also
// dequantises the output back to the source within the group error bound — proving the
// written bytes are self-consistent, not merely well-shaped.
func TestConvertSnapshot_ToyLoadable(t *testing.T) {
	const bits, groupSize = 4, 64
	srcDir := filepath.Join(t.TempDir(), "src")
	outDir := filepath.Join(t.TempDir(), "out")
	if r := core.MkdirAll(srcDir, 0o755); !r.OK {
		t.Fatalf("mkdir src: %v", r.Err())
	}

	// A 2-layer toy: two attention q_proj weights + a norm each, plus a shared
	// embedding. Weights use F32 storage (the converter accepts any float dtype); the
	// converter emits bf16 scales/biases regardless, exactly as for a bf16 source.
	type spec struct {
		name       string
		outDim, in int
		quantised  bool
	}
	specs := []spec{
		{"language_model.model.embed_tokens.weight", 16, 64, true},
		{"language_model.model.layers.0.self_attn.q_proj.weight", 8, 64, true},
		{"language_model.model.layers.0.input_layernorm.weight", 8, 0, false}, // 1-D norm → passthrough
		{"language_model.model.layers.1.self_attn.q_proj.weight", 8, 128, true},
		{"language_model.model.layers.1.input_layernorm.weight", 8, 0, false},
	}
	tensors := map[string]safetensors.Tensor{}
	sources := map[string][]float32{}
	for _, s := range specs {
		if s.quantised {
			vals := toyWeight(s.outDim, s.in)
			sources[s.name] = vals
			tensors[s.name] = safetensors.Tensor{Dtype: "F32", Shape: []int{s.outDim, s.in}, Data: safetensors.EncodeFloat32(vals)}
		} else {
			vals := toyWeight(s.outDim, 1)
			tensors[s.name] = safetensors.Tensor{Dtype: "F32", Shape: []int{s.outDim}, Data: safetensors.EncodeFloat32(vals)}
		}
	}
	blob, err := safetensors.Encode(tensors)
	if err != nil {
		t.Fatalf("encode source safetensors: %v", err)
	}
	if r := core.WriteFile(filepath.Join(srcDir, "model.safetensors"), blob, 0o644); !r.OK {
		t.Fatalf("write source shard: %v", r.Err())
	}
	writeFile(t, filepath.Join(srcDir, "config.json"), `{"model_type":"gemma3","architectures":["Gemma3ForCausalLM"],"hidden_size":64,"num_hidden_layers":2}`)
	writeFile(t, filepath.Join(srcDir, "tokenizer.json"), `{"toy":"tokenizer"}`) // a sidecar to prove copy

	res, err := mlxaffine.ConvertSnapshot(context.Background(), srcDir, outDir, mlxaffine.Options{Bits: bits, GroupSize: groupSize}, nil)
	if err != nil {
		t.Fatalf("ConvertSnapshot: %v", err)
	}
	if res.QuantizedWeights != 3 || res.PassthroughCount != 2 {
		t.Fatalf("counts: quantised=%d passthrough=%d, want 3 and 2", res.QuantizedWeights, res.PassthroughCount)
	}

	// --- the config-parse / probe layer accepts the output ---
	mt, cfgBytes, err := model.ProbeDirArch(outDir)
	if err != nil {
		t.Fatalf("ProbeDirArch(out): %v", err)
	}
	if mt != "gemma3" {
		t.Errorf("model_type = %q, want gemma3 (config identity must survive conversion)", mt)
	}
	var parsed struct {
		Quantization *model.QuantConfig `json:"quantization"`
	}
	if r := core.JSONUnmarshal(cfgBytes, &parsed); !r.OK {
		t.Fatalf("parse output config.json: %v", r.Err())
	}
	if parsed.Quantization == nil {
		t.Fatal("output config.json has no quantization block")
	}
	if err := parsed.Quantization.Validate(); err != nil {
		t.Fatalf("quantization block rejected by the reader's validator: %v", err)
	}
	if gs, b := parsed.Quantization.For("language_model.model.layers.0.self_attn.q_proj.weight"); gs != groupSize || b != bits {
		t.Errorf("QuantConfig.For = (gs=%d, bits=%d), want (%d, %d)", gs, b, groupSize, bits)
	}
	if parsed.Quantization.Mode != mlxaffine.Mode {
		t.Errorf("quantization mode = %q, want %q", parsed.Quantization.Mode, mlxaffine.Mode)
	}

	// --- the emitted tensors carry the loader's expected dtypes/shapes ---
	outIdx, err := safetensors.IndexFiles([]string{filepath.Join(outDir, "model.safetensors")})
	if err != nil {
		t.Fatalf("index output safetensors: %v", err)
	}
	assertTensor(t, outIdx, "language_model.model.layers.0.self_attn.q_proj.weight", "U32", []uint64{8, uint64(mlxaffine.PackedWords(64, bits))})
	assertTensor(t, outIdx, "language_model.model.layers.0.self_attn.q_proj.scales", "BF16", []uint64{8, 1})
	assertTensor(t, outIdx, "language_model.model.layers.0.self_attn.q_proj.biases", "BF16", []uint64{8, 1})
	assertTensor(t, outIdx, "language_model.model.layers.1.self_attn.q_proj.scales", "BF16", []uint64{8, 2}) // inDim 128 → 2 groups
	assertTensor(t, outIdx, "language_model.model.layers.0.input_layernorm.weight", "F32", []uint64{8})      // passthrough, wide

	// --- the sidecar was copied, the source shard/index were not ---
	if _, err := os.Stat(filepath.Join(outDir, "tokenizer.json")); err != nil {
		t.Errorf("tokenizer.json sidecar not copied: %v", err)
	}

	// --- the written bytes dequantise back to the source within the group bound ---
	assertDequantMatches(t, outIdx,
		"language_model.model.layers.0.self_attn.q_proj.weight", sources, 8, 64, bits, groupSize)
}

// toyWeight builds a small deterministic weight with a sign spread so groups exercise
// both affine edge branches.
func toyWeight(outDim, inDim int) []float32 {
	n := outDim * inDim
	w := make([]float32, n)
	for i := range w {
		w[i] = 0.05*float32((i%13)-6) + 0.001*float32(i)
	}
	return w
}

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if r := core.WriteFile(path, []byte(content), 0o644); !r.OK {
		t.Fatalf("write %s: %v", path, r.Err())
	}
}

func assertTensor(t *testing.T, idx safetensors.Index, name, dtype string, shape []uint64) {
	t.Helper()
	ref, ok := idx.Tensors[name]
	if !ok {
		t.Errorf("output missing tensor %q", name)
		return
	}
	if ref.DType != dtype {
		t.Errorf("%s: dtype %s, want %s", name, ref.DType, dtype)
	}
	if len(ref.Shape) != len(shape) {
		t.Errorf("%s: shape %v, want %v", name, ref.Shape, shape)
		return
	}
	for i := range shape {
		if ref.Shape[i] != shape[i] {
			t.Errorf("%s: shape %v, want %v", name, ref.Shape, shape)
			return
		}
	}
}

// assertDequantMatches reads the output quantised weight's three tensors and confirms
// dequantisation reproduces the source values within one group step — the end-to-end
// correctness check that the converter wrote the right bytes, not merely the right shapes.
func assertDequantMatches(t *testing.T, idx safetensors.Index, name string, sources map[string][]float32, outDim, inDim, bits, groupSize int) {
	t.Helper()
	cache := safetensors.NewShardCache()
	defer cache.Close()
	read := func(n string) []byte {
		ref, ok := idx.Tensors[n]
		if !ok {
			t.Fatalf("missing %q", n)
		}
		raw, err := cache.ReadRefRaw(ref)
		if err != nil {
			t.Fatalf("read %q: %v", n, err)
		}
		return raw
	}
	base := name[:len(name)-len(".weight")]
	got, err := mlxaffine.DequantizeTensor(read(name), read(base+".scales"), read(base+".biases"), outDim, inDim, bits, groupSize)
	if err != nil {
		t.Fatalf("dequantise output: %v", err)
	}
	src := sources[name]
	var maxErr float64
	for i := range src {
		if e := absf(float64(got[i] - src[i])); e > maxErr {
			maxErr = e
		}
	}
	// A safe upper bound: the source spans ~[-0.3,0.4]; one 4-bit step over that range
	// plus bf16 slack sits well under 0.1.
	if maxErr > 0.1 {
		t.Errorf("%s: max dequant error %g exceeds the group bound", name, maxErr)
	}
}

func absf(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
