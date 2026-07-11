// SPDX-Licence-Identifier: EUPL-1.2

package mlxaffine_test

import (
	"os"
	"path/filepath"
	"testing"

	"dappco.re/go/inference/model/quant/mlxaffine"
	"dappco.re/go/inference/model/safetensors"
)

// The byte-oracle. mlx-community publishes gemma-4-12B-it in both bf16 and a 4-bit
// affine snapshot produced from that bf16 by mlx_lm.convert. This test quantises a
// representative sample of tensors from the bf16 with QuantizeTensor at the 4-bit
// snapshot's declared bits/group (4, 64) and asserts the resulting packed / scales /
// biases bytes are byte-for-byte identical to the cached 4-bit snapshot's. If they
// match, QuantizeTensor reproduces mlx_lm.convert exactly — the format-authoring loop
// is closed against the reader's own producer.
//
// It reads only a SAMPLE OF ROWS of each sampled tensor via the safetensors reader's
// partial ReadAt path (never the whole 12B model), so it is fast and memory-light.
// Gated on the two snapshot directories existing (set the env overrides, or have the
// HF cache populated); it skips cleanly otherwise, like the repo's other real-model
// tests.

const (
	envBF16Dir = "LEM_MLX_BF16_DIR"
	env4BitDir = "LEM_MLX_4BIT_DIR"

	// oracleSampleRows caps how many leading rows of each sampled tensor are
	// quantised + compared. Rows are quantised independently, so a slice of rows
	// is a faithful byte-for-byte probe without materialising a multi-GB tensor.
	oracleSampleRows = 96
)

// oracleSamples names weight tensors spanning the eligible kinds and a range of inner
// dimensions: attention projections (q/v, GQA-asymmetric), both MLP directions (gate
// up-projects, down projects from the wider intermediate), and the tied embedding.
var oracleSamples = []string{
	"language_model.model.layers.0.self_attn.q_proj.weight",
	"language_model.model.layers.0.self_attn.v_proj.weight",
	"language_model.model.layers.0.mlp.gate_proj.weight",
	"language_model.model.layers.0.mlp.down_proj.weight",
	"language_model.model.embed_tokens.weight",
}

func TestByteOracle_MLXCommunityGemma4_12B(t *testing.T) {
	bf16Dir := resolveSnapshotDir(t, envBF16Dir, "models--mlx-community--gemma-4-12B-it-bf16")
	fourDir := resolveSnapshotDir(t, env4BitDir, "models--mlx-community--gemma-4-12B-it-4bit")
	if bf16Dir == "" || fourDir == "" {
		t.Skipf("byte-oracle needs both snapshots; set %s and %s (or populate the HF cache)", envBF16Dir, env4BitDir)
	}

	const bits, groupSize = 4, 64 // the 4-bit snapshot's declared quantization block

	bf16Idx, err := safetensors.IndexFiles(shardPaths(bf16Dir))
	if err != nil {
		t.Fatalf("index bf16 shards: %v", err)
	}
	fourIdx, err := safetensors.IndexFiles(shardPaths(fourDir))
	if err != nil {
		t.Fatalf("index 4bit shards: %v", err)
	}

	src := safetensors.NewShardCache()
	defer src.Close()
	dst := safetensors.NewShardCache()
	defer dst.Close()

	var comparedBytes int64
	for _, name := range oracleSamples {
		bf16Ref, ok := bf16Idx.Tensors[name]
		if !ok {
			t.Fatalf("bf16 snapshot missing sample tensor %q", name)
		}
		if len(bf16Ref.Shape) != 2 {
			t.Fatalf("%s: expected a 2-D weight, got shape %v", name, bf16Ref.Shape)
		}
		outDim, inDim := int(bf16Ref.Shape[0]), int(bf16Ref.Shape[1])
		rows := oracleSampleRows
		if rows > outDim {
			rows = outDim
		}

		// Read the leading `rows` of the bf16 weight (row-major → a leading byte
		// prefix) and decode to float32.
		wf32 := readRowsF32(t, src, bf16Ref, rows, inDim)
		packed, scales, biases, err := mlxaffine.QuantizeTensor(wf32, rows, inDim, bits, groupSize)
		if err != nil {
			t.Fatalf("%s: QuantizeTensor: %v", name, err)
		}

		// The reference's three tensors for the same leading rows.
		wantPacked := readRowsRaw(t, dst, fourIdx, name, rows*mlxaffine.PackedWords(inDim, bits)*4)
		wantScales := readRowsRaw(t, dst, fourIdx, siblingName(name, ".scales"), rows*(inDim/groupSize)*2)
		wantBiases := readRowsRaw(t, dst, fourIdx, siblingName(name, ".biases"), rows*(inDim/groupSize)*2)

		assertByteEqual(t, name+" scales", scales, wantScales, groupSize, inDim, bits)
		assertByteEqual(t, name+" biases", biases, wantBiases, groupSize, inDim, bits)
		assertByteEqual(t, name+" packed", packed, wantPacked, groupSize, inDim, bits)

		comparedBytes += int64(len(packed) + len(scales) + len(biases))
		t.Logf("%-58s rows=%d inDim=%d  BYTE-IDENTICAL (packed %dB scales %dB biases %dB)",
			name, rows, inDim, len(packed), len(scales), len(biases))
	}
	t.Logf("byte-oracle PASSED: %d sample tensors, %d bytes compared byte-for-byte against mlx_lm.convert output", len(oracleSamples), comparedBytes)
}

// resolveSnapshotDir returns the model snapshot directory: the env override when set,
// else the newest snapshot under the HF hub cache for repo. "" when neither resolves.
func resolveSnapshotDir(t *testing.T, envKey, repo string) string {
	t.Helper()
	if v := os.Getenv(envKey); v != "" {
		if dirHasShards(v) {
			return v
		}
		t.Fatalf("%s=%q has no *.safetensors shards", envKey, v)
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	base := filepath.Join(home, ".cache", "huggingface", "hub", repo, "snapshots")
	entries, err := os.ReadDir(base)
	if err != nil {
		return ""
	}
	for _, e := range entries {
		cand := filepath.Join(base, e.Name())
		if dirHasShards(cand) {
			return cand
		}
	}
	return ""
}

func dirHasShards(dir string) bool { return len(shardPaths(dir)) > 0 }

func shardPaths(dir string) []string {
	m, _ := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	return m
}

// readRowsF32 reads the leading `rows` of a 2-D bf16/f32 weight and decodes to float32.
// It clones the ref with a shrunk ByteLen so the reader's ReadAt path fetches only the
// leading row-prefix — never the whole tensor.
func readRowsF32(t *testing.T, c *safetensors.ShardCache, ref safetensors.TensorRef, rows, inDim int) []float32 {
	t.Helper()
	bytesPerElem, err := safetensors.DTypeByteSize(ref.DType)
	if err != nil {
		t.Fatalf("%s: dtype size: %v", ref.Name, err)
	}
	prefix := ref
	prefix.ByteLen = int64(rows * inDim * bytesPerElem)
	raw, err := c.ReadRefRaw(prefix)
	if err != nil {
		t.Fatalf("%s: read leading rows: %v", ref.Name, err)
	}
	values, err := safetensors.DecodeFloat32(ref.DType, raw, rows*inDim)
	if err != nil {
		t.Fatalf("%s: decode float32: %v", ref.Name, err)
	}
	return values
}

// readRowsRaw reads the leading `byteLen` bytes of a named tensor's payload — the
// leading rows of a row-major tensor — via the reader's ReadAt path.
func readRowsRaw(t *testing.T, c *safetensors.ShardCache, idx safetensors.Index, name string, byteLen int) []byte {
	t.Helper()
	ref, ok := idx.Tensors[name]
	if !ok {
		t.Fatalf("4bit snapshot missing tensor %q", name)
	}
	if int64(byteLen) > ref.ByteLen {
		t.Fatalf("%s: want %d bytes but tensor is only %d", name, byteLen, ref.ByteLen)
	}
	prefix := ref
	prefix.ByteLen = int64(byteLen)
	raw, err := c.ReadRefRaw(prefix)
	if err != nil {
		t.Fatalf("%s: read leading rows: %v", name, err)
	}
	return raw
}

// siblingName replaces a weight tensor's ".weight" suffix with suffix (".scales" /
// ".biases") — the MLX sibling naming for the group parameters.
func siblingName(weight, suffix string) string {
	return weight[:len(weight)-len(".weight")] + suffix
}

// assertByteEqual fails with a precise characterisation of the FIRST divergence: which
// byte, the two values, and the row/group the byte belongs to — so a genuine algorithm
// divergence (scale derivation, rounding, bias) is reported with numeric evidence
// rather than a bare "not equal".
func assertByteEqual(t *testing.T, label string, got, want []byte, groupSize, inDim, bits int) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d != reference %d", label, len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("%s: FIRST divergence at byte %d: got 0x%02x want 0x%02x (groups/row=%d, inDim=%d, %d-bit) — characterise before accepting drift",
				label, i, got[i], want[i], inDim/groupSize, inDim, bits)
		}
	}
}
