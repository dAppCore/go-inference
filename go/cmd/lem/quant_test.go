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
