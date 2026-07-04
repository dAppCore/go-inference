// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	g4 "dappco.re/go/inference/model/gemma4"
)

// zz_cover_load_test.go closes the buildShardBuffers failure legs in the
// registry directory loaders (LoadDir, LoadTokenModelDir). Those loaders read
// config.json, mmap the shards and assemble the weights with NO Metal work, then
// call buildShardBuffers — the
// FIRST step that needs the device (newShardBuffers calls ensureInit). Breaking
// the runtime AFTER a valid checkpoint is on disk (unset the metallib env + reset
// the init once) makes ensureInit fail exactly at buildShardBuffers, exercising
// the `_ = dm.Close(); return nil, err` cleanup leg in each loader. The runtime is
// restored before the test returns so later files are unaffected.

// withBrokenRuntime runs fn with the native runtime deliberately un-initialisable
// (metallib env unset, init globals reset) and restores it afterwards.
func withBrokenRuntime(t *testing.T, fn func()) {
	t.Helper()
	good, had := os.LookupEnv(MetallibPathEnv)
	if !had || good == "" {
		t.Skip("no metallib path to restore")
	}
	t.Cleanup(func() {
		_ = os.Setenv(MetallibPathEnv, good)
		resetNativeInitGlobalsForCoverage()
		if err := ensureInit(); err != nil {
			t.Fatalf("restore native runtime: %v", err)
		}
	})
	_ = os.Unsetenv(MetallibPathEnv)
	resetNativeInitGlobalsForCoverage()
	fn()
}

// TestCoverLoaderBuildShardBuffersFailure covers the buildShardBuffers cleanup
// legs in the three directory loaders by breaking the runtime after a valid
// checkpoint is written.
func TestCoverLoaderBuildShardBuffersFailure(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, vocab = 64, 2, 1, 64, 256, 32
	const maxLen = 8
	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: 1, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim, VocabSize: vocab, RMSNormEps: 1e-6,
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}

	// (a) dense bf16 directory.
	bf16Dir := t.TempDir()
	writeLocal(t, core.PathJoin(bf16Dir, "config.json"), gemma4ConfigJSON(t, cfg))
	writeLocal(t, core.PathJoin(bf16Dir, "model.safetensors"), encodedTensors(t, gemma4TensorsMust(t, arch)))

	// (b) quant directory.
	const gs, bits = 64, 4
	quantCfg := cfg
	quantCfg.Quantization = &model.QuantConfig{GroupSize: gs, Bits: bits}
	quantDir := t.TempDir()
	writeLocal(t, core.PathJoin(quantDir, "config.json"), gemma4ConfigJSON(t, quantCfg))
	writeLocal(t, core.PathJoin(quantDir, "model.safetensors"), encodedTensors(t, quantGemma4TensorsGuard(t, arch, gs, bits)))

	withBrokenRuntime(t, func() {
		// LoadDir (bf16 dir): config + mmap + assemble succeed, buildShardBuffers fails.
		if _, e := LoadDir(bf16Dir, maxLen); e == nil {
			t.Fatal("LoadDir bf16: expected buildShardBuffers failure")
		}
		// LoadTokenModelDir (bf16 path): same cleanup leg.
		if _, e := LoadTokenModelDir(bf16Dir, maxLen); e == nil {
			t.Fatal("LoadTokenModelDir bf16: expected buildShardBuffers failure")
		}
		// LoadDir (quant dir): the quant sibling.
		if _, e := LoadDir(quantDir, maxLen); e == nil {
			t.Fatal("LoadDir quant: expected buildShardBuffers failure")
		}
		// LoadTokenModelDir (quant path): the quant token-model cleanup leg.
		if _, e := LoadTokenModelDir(quantDir, maxLen); e == nil {
			t.Fatal("LoadTokenModelDir quant: expected buildShardBuffers failure")
		}
	})
}
