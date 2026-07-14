// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"

	core "dappco.re/go"
	g3 "dappco.re/go/inference/model/gemma3" // importing registers the gemma3 ArchSpec (NormBiasOne folding) for LoadDir
	coreio "dappco.re/go/io"
)

// TestLoadDir_Gemma3TiedFoldedNorms is the #1851 regression: a gemma3 checkpoint ties its lm_head to
// the embedding AND folds the gemma "(1 + weight)" RMSNorm convention into EVERY norm at load
// (ArchSpec.NormBiasOne → foldNormBiasOne), producing FRESH heap norm buffers that are not views into
// any mapped shard. The zero-copy layer builder used to bind every norm through the strict projection
// resolver (mustBufFor), which errors on a non-shard-view weight — so LoadDir failed at the build seam
// with "native.shardBuffers.bufForAligned: weight is not a view into any mapped shard" (surfaced as the
// generate/warm error). shardBuffers.bufForNorm now binds a synthesised norm resident (a tiny per-
// session vector, no per-token balloon) while the projections stay zero-copy, so the load completes.
// Before the bufForNorm fix this LoadDir returns that error; after, it loads. Uses the shared synthetic-
// checkpoint machinery (gemma4Tensors builds the standard tied gemma layout; a gemma3_text config drives
// the folding). No real checkpoint dependency.
func TestLoadDir_Gemma3TiedFoldedNorms(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const maxLen = 16
	cfg := g3.Config{
		HiddenSize: 128, NumHiddenLayers: 2, IntermediateSize: 256,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 64, VocabSize: 32, RMSNormEps: 1e-6,
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("gemma3 Config.Arch: %v", err)
	}
	tensors, _ := gemma4Tensors(arch, false) // withLMHead=false ⇒ tied (no lm_head), the gemma3 shape

	dir := t.TempDir()
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), string(configJSONWithModelType(t, cfg, "gemma3_text"))); err != nil {
		t.Fatalf("write config: %v", err)
	}
	writeShardedCheckpoint(t, dir, tensors)

	sess, err := LoadDir(dir, maxLen)
	if err != nil {
		t.Fatalf("LoadDir gemma3 tied+folded-norm checkpoint: %v", err)
	}
	if sess == nil {
		t.Fatal("LoadDir returned a nil session")
	}
	_ = sess.Close()
}

// TestShardBuffersBufForNormSynthesisedBindsResident pins bufForNorm's seam directly: a synthesised
// norm (a folded gemma norm — fresh heap, outside every mapped shard range) must bind RESIDENT rather
// than error, while the strict projection resolver (bufFor) still rejects it (a projection that is not
// a shard view is a wrong-mapping bug). The empty-weight case yields the zero bufView ("skip").
func TestShardBuffersBufForNormSynthesisedBindsResident(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set (bufForNorm's resident path uploads to the device)")
	}
	// A heap slice at an address covered by no shard range — the shape of a folded norm.
	folded := []byte{0x80, 0x3f, 0x00, 0x40} // 2 bf16 elements
	sb := &shardBuffers{bases: []uintptr{1}, ends: []uintptr{2}}

	if _, err := sb.bufFor(folded); err == nil {
		t.Fatal("bufFor must still reject a non-shard weight (the projection guard is intact)")
	}
	v := sb.bufForNorm(folded)
	if v.buf == nil {
		t.Fatal("bufForNorm must bind a synthesised norm resident, not error (#1851)")
	}
	if v.off != 0 {
		t.Fatalf("bufForNorm resident copy must sit at offset 0, got %d", v.off)
	}
	if got := sb.bufForNorm(nil); got.buf != nil || got.off != 0 {
		t.Fatalf("bufForNorm(nil) = %+v, want the zero bufView", got)
	}
}
