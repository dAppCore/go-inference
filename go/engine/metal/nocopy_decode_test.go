// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"os"
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/inference/model"
	g4 "dappco.re/go/inference/model/gemma4"
	"dappco.re/go/inference/safetensors"
)

// writeShardedCheckpoint writes a tensor set to dir as a 2-shard checkpoint (index.json + two
// shards), so a directory load exercises the MULTI-shard zero-copy resolver (bufFor across more
// than one mmap) — the single-file case is the degenerate one-shard path. Returns nothing; fails
// the test on any I/O error.
func writeShardedCheckpoint(t *testing.T, dir string, tensors map[string]safetensors.Tensor) {
	t.Helper()
	half1, half2 := map[string]safetensors.Tensor{}, map[string]safetensors.Tensor{}
	wm := map[string]string{}
	i := 0
	for name, tns := range tensors {
		if i%2 == 0 {
			half1[name], wm[name] = tns, "model-00001-of-00002.safetensors"
		} else {
			half2[name], wm[name] = tns, "model-00002-of-00002.safetensors"
		}
		i++
	}
	if err := coreio.Local.Write(core.PathJoin(dir, "model-00001-of-00002.safetensors"), string(mustEncode(t, half1))); err != nil {
		t.Fatalf("write shard1: %v", err)
	}
	if err := coreio.Local.Write(core.PathJoin(dir, "model-00002-of-00002.safetensors"), string(mustEncode(t, half2))); err != nil {
		t.Fatalf("write shard2: %v", err)
	}
	idx := core.JSONMarshal(map[string]any{"weight_map": wm})
	if !idx.OK {
		t.Fatalf("marshal index")
	}
	if err := coreio.Local.Write(core.PathJoin(dir, "model.safetensors.index.json"), string(idx.Value.([]byte))); err != nil {
		t.Fatalf("write index: %v", err)
	}
}

// stepHiddens drives a session over a fixed id sequence and returns each step's output hidden
// state (dModel bf16 bytes) PLUS the head logits for the final hidden — the full per-step decode
// + head output, captured as raw bytes for an exact (not token-id) comparison.
func stepHiddens(t *testing.T, s *ArchSession, head func([]byte, bool) ([]byte, error), ids []int32) [][]byte {
	t.Helper()
	out := make([][]byte, 0, len(ids)+1)
	var last []byte
	for _, id := range ids {
		emb, err := s.embed(id)
		if err != nil {
			t.Fatalf("embed: %v", err)
		}
		h, err := s.StepWithID(id, emb)
		if err != nil {
			t.Fatalf("StepWithID: %v", err)
		}
		out = append(out, h)
		last = h
	}
	logits, err := head(last, false) // both compared paths apply the softcap → parity holds
	if err != nil {
		t.Fatalf("head: %v", err)
	}
	out = append(out, logits)
	return out
}

// TestNoCopyByteIdentity_BF16 is the byte-identity gate for the bf16 zero-copy weight path: the
// SAME synthetic gemma4 checkpoint is loaded BOTH ways — the in-memory copy path (assemble the
// parsed tensors + NewArchSession, which uploads each weight into an owned Metal buffer) and the
// on-disk zero-copy path (LoadGemma4BF16Dir → LoadDirMmap + per-shard no-copy buffers + offset
// binding) — and a fixed decode + head must produce BYTE-FOR-BYTE identical output. The refactor
// changes only WHERE the weight bytes are bound from (a fresh owned copy vs a no-copy view into
// the shared shard mmap at an offset); the math is untouched, so the outputs must be bit-identical.
// This is the gate every zero-copy split (loader, enc offsets, projector views, head) must pass.
// Uses a 2-shard checkpoint to exercise the multi-shard bufFor resolver.
func TestNoCopyByteIdentity_BF16(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const maxLen = 16
	cfg := g4.Config{
		HiddenSize: 128, NumHiddenLayers: 2, IntermediateSize: 256,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 64, VocabSize: 32, RMSNormEps: 1e-6,
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	tensors, _ := gemma4Tensors(arch, false)
	ids := []int32{1, 5, 3, 7}

	// copy path: assemble the parsed tensors (heap bytes) → session (sharedBytes-copied weights).
	lmCopy, err := model.Assemble(tensors, arch, model.StandardWeightNames())
	if err != nil {
		t.Fatalf("model.Assemble: %v", err)
	}
	gCopy := loadedToBF16(lmCopy)
	sCopy, err := NewArchSession(gCopy, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	wantHead := func(h []byte, _ bool) ([]byte, error) {
		return LMHeadBF16(h, gCopy.FinalNorm, gCopy.LMHead, arch.Hidden, arch.Vocab, arch.Eps, arch.SoftCap)
	}
	want := stepHiddens(t, sCopy, wantHead, ids)

	// zero-copy path: write a 2-shard checkpoint, load it mmap'd.
	dir := t.TempDir()
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), string(gemma4ConfigJSON(t, cfg))); err != nil {
		t.Fatalf("write config: %v", err)
	}
	writeShardedCheckpoint(t, dir, tensors)
	sMmap, err := LoadDir(dir, maxLen)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	defer func() { _ = sMmap.Close() }()
	got := stepHiddens(t, sMmap, sMmap.head, ids)

	if len(got) != len(want) {
		t.Fatalf("step count %d != %d", len(got), len(want))
	}
	for i := range want {
		if !bytes.Equal(got[i], want[i]) {
			t.Fatalf("step %d output differs: zero-copy mmap path is NOT byte-identical to the copy path (len got %d want %d)", i, len(got[i]), len(want[i]))
		}
	}
	t.Logf("bf16 zero-copy: %d-step decode + head BYTE-IDENTICAL across copy vs 2-shard mmap load", len(ids))
}

// TestNoCopyByteIdentity_Quant is the byte-identity gate for the 4-bit zero-copy path — the
// sibling of TestNoCopyByteIdentity_BF16 for the quantised decode + head (the path the per-token
// LM-head balloon lived on). REAL affine-packed weights (quantGemma4Tensors) loaded the copy way
// (NewArchQuantSession over heap bytes) vs the zero-copy way (LoadGemma4Quant4Dir → mmap + no-
// copy shard buffers) must give byte-for-byte identical decode + head output. 2-shard checkpoint.
func TestNoCopyByteIdentity_Quant(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const gs, bits = 32, 4
	const maxLen = 16
	cfg := g4.Config{
		HiddenSize: 128, NumHiddenLayers: 2, IntermediateSize: 256,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 64, VocabSize: 32, RMSNormEps: 1e-6,
		Quantization: &model.QuantConfig{GroupSize: gs, Bits: bits},
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	tensors := quantGemma4Tensors(t, arch, gs, bits)
	ids := []int32{1, 5, 3, 7}

	lmCopy, err := model.Assemble(tensors, arch, model.StandardWeightNames())
	if err != nil {
		t.Fatalf("model.Assemble: %v", err)
	}
	gCopy, err := loadedToQuant(lmCopy, gs, bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	sCopy, err := NewArchQuantSession(gCopy, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession: %v", err)
	}
	wantHead := func(h []byte, _ bool) ([]byte, error) {
		return LMHeadQuant(h, gCopy.FinalNorm, gCopy.LMHead, gCopy.LMHeadScales, gCopy.LMHeadBiases, arch.Hidden, arch.Vocab, gs, bits, arch.Eps, arch.SoftCap)
	}
	want := stepHiddens(t, sCopy, wantHead, ids)

	dir := t.TempDir()
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), string(gemma4ConfigJSON(t, cfg))); err != nil {
		t.Fatalf("write config: %v", err)
	}
	writeShardedCheckpoint(t, dir, tensors)
	sMmap, err := LoadDir(dir, maxLen)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	defer func() { _ = sMmap.Close() }()
	got := stepHiddens(t, sMmap, sMmap.head, ids)

	if len(got) != len(want) {
		t.Fatalf("step count %d != %d", len(got), len(want))
	}
	for i := range want {
		if !bytes.Equal(got[i], want[i]) {
			t.Fatalf("step %d output differs: 4-bit zero-copy mmap path is NOT byte-identical to the copy path", i)
		}
	}
	t.Logf("4-bit zero-copy: %d-step decode + head BYTE-IDENTICAL across copy vs 2-shard mmap load", len(ids))
}

// TestNoCopyHead_TokenModelServePath gates the per-token SERVE head specifically: model.Generate's
// generateStepwise calls NativeTokenModel.Head every token (NOT the session's head), so that is the
// path the LM-head balloon lived on and the resident headEncoder fixes. It builds the SAME 4-bit
// checkpoint as a directory token model (LoadGemma4TokenModelDir, whose m.Head is the resident
// upload-once head) and as an in-memory token model (NewQuantTokenModel, whose m.Head re-uploads
// via LMHeadQuant), and asserts m.Head is BYTE-FOR-BYTE identical for a fixed hidden — the resident
// head must not change the logits. (The balloon-gone metric itself is BenchmarkHeadEncoderQuant.)
func TestNoCopyHead_TokenModelServePath(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const gs, bits = 32, 4
	const maxLen = 16
	cfg := g4.Config{
		HiddenSize: 128, NumHiddenLayers: 2, IntermediateSize: 256,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 64, VocabSize: 32, RMSNormEps: 1e-6,
		Quantization: &model.QuantConfig{GroupSize: gs, Bits: bits},
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	tensors := quantGemma4Tensors(t, arch, gs, bits)

	lmCopy, err := model.Assemble(tensors, arch, model.StandardWeightNames())
	if err != nil {
		t.Fatalf("model.Assemble: %v", err)
	}
	gCopy, err := loadedToQuant(lmCopy, gs, bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	tmCopy, err := NewQuantTokenModel(gCopy, arch, maxLen) // m.Head = the per-token upload head
	if err != nil {
		t.Fatalf("NewQuantTokenModel: %v", err)
	}

	dir := t.TempDir()
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), string(gemma4ConfigJSON(t, cfg))); err != nil {
		t.Fatalf("write config: %v", err)
	}
	writeShardedCheckpoint(t, dir, tensors)
	tm, err := LoadTokenModelDir(dir, maxLen) // m.Head = the resident head (the balloon fix)
	if err != nil {
		t.Fatalf("LoadTokenModelDir: %v", err)
	}
	if c, ok := tm.(interface{ Close() error }); ok {
		defer func() { _ = c.Close() }()
	}

	hidden := bf16ConstBytes(arch.Hidden, 0.02)
	want, err := tmCopy.Head(hidden)
	if err != nil {
		t.Fatalf("copy Head: %v", err)
	}
	got, err := tm.Head(hidden)
	if err != nil {
		t.Fatalf("resident Head: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("resident token-model Head is NOT byte-identical to the upload head (the serve-path balloon fix changed the logits)")
	}
	t.Logf("serve-path head: resident NativeTokenModel.Head ≡ upload Head, byte-for-byte (balloon gone, logits unchanged)")
}
