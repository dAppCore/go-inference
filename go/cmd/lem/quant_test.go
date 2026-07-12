// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

// writeToyModel writes a minimal dense model directory (one eligible 2-D weight + a 1-D
// norm, F32 storage) so the quant verb has something real to convert without a GPU or a
// downloaded checkpoint.
func writeToyModel(t *testing.T) string {
	t.Helper()
	dir := filepath.Join(t.TempDir(), "toy-bf16")
	if r := core.MkdirAll(dir, 0o755); !r.OK {
		t.Fatalf("mkdir: %v", r.Err())
	}
	weight := make([]float32, 8*64)
	for i := range weight {
		weight[i] = 0.03*float32((i%11)-5) + 0.0005*float32(i)
	}
	norm := []float32{1, 1.01, 0.99, 1.02, 0.98, 1.0, 1.03, 0.97}
	blob, err := safetensors.Encode(map[string]safetensors.Tensor{
		"language_model.model.layers.0.self_attn.q_proj.weight": {Dtype: "F32", Shape: []int{8, 64}, Data: safetensors.EncodeFloat32(weight)},
		"language_model.model.layers.0.input_layernorm.weight":  {Dtype: "F32", Shape: []int{8}, Data: safetensors.EncodeFloat32(norm)},
	})
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if r := core.WriteFile(filepath.Join(dir, "model.safetensors"), blob, 0o644); !r.OK {
		t.Fatalf("write shard: %v", r.Err())
	}
	if r := core.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"model_type":"gemma3","hidden_size":64}`), 0o644); !r.OK {
		t.Fatalf("write config: %v", r.Err())
	}
	return dir
}

// TestRunQuantCommand_PositionalOrder proves the verb accepts the <src-model-dir>
// positional both before and after the flags (the two-phase parse) and writes a loadable
// MLX-affine directory either way.
func TestRunQuantCommand_PositionalOrder(t *testing.T) {
	for _, tc := range []struct {
		name string
		args func(src, out string) []string
	}{
		{"dir then flags", func(src, out string) []string { return []string{src, "-bits", "4", "-o", out} }},
		{"flags then dir", func(src, out string) []string { return []string{"-bits", "4", "-o", out, src} }},
		{"flags around dir", func(src, out string) []string { return []string{"-bits", "4", src, "-group-size", "64", "-o", out} }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			src := writeToyModel(t)
			out := filepath.Join(t.TempDir(), "out")
			var stdout, stderr bytes.Buffer
			if code := runQuantCommand(context.Background(), tc.args(src, out), &stdout, &stderr); code != 0 {
				t.Fatalf("exit %d; stderr=%s", code, stderr.String())
			}
			// A U32 packed weight + BF16 scales/biases must exist in the output.
			idx, err := safetensors.IndexFiles([]string{filepath.Join(out, "model.safetensors")})
			if err != nil {
				t.Fatalf("index output: %v", err)
			}
			w := idx.Tensors["language_model.model.layers.0.self_attn.q_proj.weight"]
			if w.DType != "U32" {
				t.Errorf("packed weight dtype = %s, want U32", w.DType)
			}
			if _, ok := idx.Tensors["language_model.model.layers.0.self_attn.q_proj.scales"]; !ok {
				t.Error("output missing .scales sibling")
			}
			if _, err := os.Stat(filepath.Join(out, "config.json")); err != nil {
				t.Errorf("output config.json missing: %v", err)
			}
		})
	}
}

// writeGGUFToy writes a toy dense model whose tensor dims are multiples of the
// GGUF block size (32), so the GGUF quant lane has valid input to convert — the
// standard writeToyModel's 8-wide norm is not divisible by 32 and is used to
// drive the rejection path instead.
func writeGGUFToy(t *testing.T) string {
	t.Helper()
	dir := filepath.Join(t.TempDir(), "toy-gguf")
	if r := core.MkdirAll(dir, 0o755); !r.OK {
		t.Fatalf("mkdir: %v", r.Err())
	}
	weight := make([]float32, 64*64)
	for i := range weight {
		weight[i] = 0.01 * float32((i%7)-3)
	}
	norm := make([]float32, 64)
	for i := range norm {
		norm[i] = 1.0 + 0.001*float32(i)
	}
	blob, err := safetensors.Encode(map[string]safetensors.Tensor{
		"language_model.model.layers.0.self_attn.q_proj.weight": {Dtype: "F32", Shape: []int{64, 64}, Data: safetensors.EncodeFloat32(weight)},
		"language_model.model.layers.0.input_layernorm.weight":  {Dtype: "F32", Shape: []int{64}, Data: safetensors.EncodeFloat32(norm)},
	})
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if r := core.WriteFile(filepath.Join(dir, "model.safetensors"), blob, 0o644); !r.OK {
		t.Fatalf("write shard: %v", r.Err())
	}
	if r := core.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"model_type":"gemma3","hidden_size":64}`), 0o644); !r.OK {
		t.Fatalf("write config: %v", r.Err())
	}
	return dir
}

// TestRunQuantGGUF covers the -gguf lane: a block-size-valid model quantises to
// a model.gguf (Good), the default output dir carries the gguf-<format> suffix
// when -o is omitted (Good), and a tensor whose width is not divisible by the
// GGUF block size falls back to raw F32 storage rather than failing (Good).
func TestRunQuantGGUF(t *testing.T) {
	t.Run("Good/explicit out", func(t *testing.T) {
		src := writeGGUFToy(t)
		out := filepath.Join(t.TempDir(), "gguf-out")
		var stdout, stderr bytes.Buffer
		if code := runQuantCommand(context.Background(), []string{src, "-gguf", "q8_0", "-o", out}, &stdout, &stderr); code != 0 {
			t.Fatalf("exit %d; stderr=%s", code, stderr.String())
		}
		if _, err := os.Stat(filepath.Join(out, "model.gguf")); err != nil {
			t.Errorf("model.gguf not written: %v", err)
		}
	})
	t.Run("Good/default out dir", func(t *testing.T) {
		src := writeGGUFToy(t)
		var stdout, stderr bytes.Buffer
		if code := runQuantCommand(context.Background(), []string{src, "-gguf", "q8_0"}, &stdout, &stderr); code != 0 {
			t.Fatalf("exit %d; stderr=%s", code, stderr.String())
		}
		// defaultOutDir tags the source basename with -gguf-<format> in its parent.
		wantDir := filepath.Join(filepath.Dir(src), "toy-gguf-gguf-q8_0")
		if _, err := os.Stat(filepath.Join(wantDir, "model.gguf")); err != nil {
			t.Errorf("default-named output missing model.gguf at %s: %v", wantDir, err)
		}
	})
	t.Run("Good/block-incompatible falls back to F32", func(t *testing.T) {
		// Block-incompatible tensors (the 8-wide norm) no longer fail the model —
		// they store as raw F32 (llama.cpp's own quantizer convention), so the
		// conversion succeeds end-to-end.
		src := writeToyModel(t) // 8-wide norm, not divisible by 32
		out := filepath.Join(t.TempDir(), "gguf-out")
		var stdout, stderr bytes.Buffer
		if code := runQuantCommand(context.Background(), []string{src, "-gguf", "q8_0", "-o", out}, &stdout, &stderr); code != 0 {
			t.Fatalf("exit %d, want 0 (F32 fallback); stderr=%s", code, stderr.String())
		}
		if _, err := os.Stat(filepath.Join(out, "model.gguf")); err != nil {
			t.Fatalf("model.gguf not written: %v", err)
		}
	})
}

func TestRunQuantGPTQ(t *testing.T) {
	t.Run("Good/explicit out", func(t *testing.T) {
		src := writeToyModel(t)
		out := filepath.Join(t.TempDir(), "gptq-out")
		var stdout, stderr bytes.Buffer
		if code := runQuantCommand(context.Background(), []string{src, "-gptq", "-group-size", "32", "-o", out}, &stdout, &stderr); code != 0 {
			t.Fatalf("exit %d; stderr=%s", code, stderr.String())
		}
		idx, err := safetensors.IndexFiles([]string{filepath.Join(out, "model.safetensors")})
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := idx.Tensors["language_model.model.layers.0.self_attn.q_proj.qweight"]; !ok {
			t.Fatal("GPTQ qweight was not written")
		}
	})
	t.Run("Bad/mutually exclusive formats", func(t *testing.T) {
		src := writeToyModel(t)
		var stdout, stderr bytes.Buffer
		if code := runQuantCommand(context.Background(), []string{src, "-gptq", "-gguf", "q8_0"}, &stdout, &stderr); code != 2 {
			t.Fatalf("exit %d, want 2; stderr=%s", code, stderr.String())
		}
	})
}

func TestRunQuantAWQ(t *testing.T) {
	t.Run("Good/explicit out", func(t *testing.T) {
		src := writeToyModel(t)
		out := filepath.Join(t.TempDir(), "awq-out")
		var stdout, stderr bytes.Buffer
		if code := runQuantCommand(context.Background(), []string{src, "-awq", "-group-size", "32", "-o", out}, &stdout, &stderr); code != 0 {
			t.Fatalf("exit %d; stderr=%s", code, stderr.String())
		}
		idx, err := safetensors.IndexFiles([]string{filepath.Join(out, "model.safetensors")})
		if err != nil {
			t.Fatal(err)
		}
		ref, ok := idx.Tensors["language_model.model.layers.0.self_attn.q_proj.qweight"]
		if !ok || ref.Shape[0] != 64 || ref.Shape[1] != 1 {
			t.Fatalf("AWQ qweight = %v, present %v", ref.Shape, ok)
		}
		config, err := os.ReadFile(filepath.Join(out, "quantize_config.json"))
		if err != nil || !bytes.Contains(config, []byte(`"data_free": true`)) {
			t.Fatalf("AWQ disclosure config = %s, %v", config, err)
		}
	})
	t.Run("Bad/mutually exclusive formats", func(t *testing.T) {
		src := writeToyModel(t)
		var stdout, stderr bytes.Buffer
		if code := runQuantCommand(context.Background(), []string{src, "-awq", "-gptq"}, &stdout, &stderr); code != 2 {
			t.Fatalf("exit %d, want 2; stderr=%s", code, stderr.String())
		}
	})
}

// TestDefaultOutDir covers the mlx_lm.convert naming convention: strip a trailing
// slash, drop a -bf16/-f32 dense tag, and append the quant suffix in the source's
// parent directory. A name with no dense tag keeps its basename intact.
func TestDefaultOutDir(t *testing.T) {
	for _, tc := range []struct {
		name, src, suffix, want string
	}{
		{"bf16 tag stripped", "/models/gemma-4-12B-it-bf16", "4bit", "/models/gemma-4-12B-it-4bit"},
		{"f32 tag stripped", "/models/gemma-f32", "8bit", "/models/gemma-8bit"},
		{"trailing slash", "/models/gemma-bf16/", "4bit", "/models/gemma-4bit"},
		{"no dense tag kept", "/models/gemma-4bit", "gguf-q4_k_m", "/models/gemma-4bit-gguf-q4_k_m"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := defaultOutDir(tc.src, tc.suffix); got != tc.want {
				t.Errorf("defaultOutDir(%q, %q) = %q, want %q", tc.src, tc.suffix, got, tc.want)
			}
		})
	}
}

// TestHumanBytes covers all three magnitude branches of the summary formatter:
// plain bytes below a MiB, one-decimal MiB, and two-decimal GiB — with the
// boundary values that select each branch.
func TestHumanBytes(t *testing.T) {
	const (
		mib = 1 << 20
		gib = 1 << 30
	)
	for _, tc := range []struct {
		name string
		n    int64
		want string
	}{
		{"zero bytes", 0, "0 B"},
		{"sub-MiB", mib - 1, "1048575 B"},
		{"exactly one MiB", mib, "1.0 MiB"},
		{"mid MiB", 3*mib + mib/2, "3.5 MiB"},
		{"exactly one GiB", gib, "1.00 GiB"},
		{"mid GiB", 2*gib + gib/4, "2.25 GiB"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := humanBytes(tc.n); got != tc.want {
				t.Errorf("humanBytes(%d) = %q, want %q", tc.n, got, tc.want)
			}
		})
	}
}

// TestRunQuantCommand_Rejects covers the argument/validation failures: no positional,
// an unsupported bit-width, and a source directory with no shards.
func TestRunQuantCommand_Rejects(t *testing.T) {
	src := writeToyModel(t)
	for _, tc := range []struct {
		name     string
		args     []string
		wantCode int
	}{
		{"no positional", []string{"-bits", "4"}, 2},
		{"unsupported bits", []string{src, "-bits", "3"}, 1},
		{"no shards", []string{filepath.Join(t.TempDir(), "empty")}, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			if code := runQuantCommand(context.Background(), tc.args, &stdout, &stderr); code != tc.wantCode {
				t.Fatalf("exit %d, want %d; stderr=%s", code, tc.wantCode, stderr.String())
			}
		})
	}
}
