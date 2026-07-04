// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"
	"time"

	"dappco.re/go/inference/model"
	"github.com/tmc/apple/metal"
)

// TestPipelinedBatchMatchesSerial — the production pipelined batch path (DecodeForwardArchICBQuant,
// double-buffered for ≥4 tokens) must be byte-identical to the serial path (same ICB ops, only the
// submission overlaps; the shared-cache hazard serialises the GPU side).
func TestPipelinedBatchMatchesSerial(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, nHeads, nKV, headDim, dFF, maxLen = 1536, 8, 1, 256, 6144, 128
	inputs, layers, arch := spikeE2BFixture(t)
	pipelinedBatchDisabled = true
	serial, err := DecodeForwardArchICBQuant(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm)
	pipelinedBatchDisabled = false
	if err != nil {
		t.Fatalf("serial: %v", err)
	}
	pipe, err := DecodeForwardArchICBQuant(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm)
	if err != nil {
		t.Fatalf("pipelined: %v", err)
	}
	for tok := range serial {
		if cos := cosineBF16(pipe[tok], serial[tok]); cos < 0.99999 {
			t.Fatalf("tok%d: pipelined batch cosine=%.7f vs serial — overlap broke a dependency", tok, cos)
		}
	}
	t.Logf("pipelined batch matches serial across %d tokens", len(serial))
}

// BenchmarkSpikeE2BDecodeSerial / -Pipelined — serial runBatch vs the production double-buffered path.
func BenchmarkSpikeE2BDecodeSerial(b *testing.B) {
	pipelinedBatchDisabled = true
	defer func() { pipelinedBatchDisabled = false }()
	spikeE2BDecode(b)
}

func BenchmarkSpikeE2BDecodePipelined(b *testing.B) {
	pipelinedBatchDisabled = false
	spikeE2BDecode(b)
}

// TestSpikeGPUvsWall splits the decode wall into GPU-busy span vs host overhead (per-token
// WaitUntilCompleted turnaround + submit + read). A large host-overhead fraction is the idle that a
// pipelined / submit-ahead decode loop (pkg/metal's PipelinedDecode) reclaims — no kernel change.
func TestSpikeGPUvsWall(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, nHeads, nKV, headDim, dFF, maxLen = 1536, 8, 1, 256, 6144, 128
	inputs, layers, arch := spikeE2BFixture(t)
	// warm
	_, _ = DecodeForwardArchICBQuant(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm)
	pieceTimingOn = true
	icbGPUNs = 0
	start := time.Now()
	if _, err := DecodeForwardArchICBQuant(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm); err != nil {
		pieceTimingOn = false
		t.Fatalf("decode: %v", err)
	}
	wall := time.Since(start)
	gpu := time.Duration(icbGPUNs)
	pieceTimingOn = false
	t.Logf("64-token decode: wall=%v  GPU-busy=%v  host-overhead=%v (%.0f%%)", wall, gpu, wall-gpu, 100*float64(wall-gpu)/float64(wall))
}

// THROWAWAY spike instrument. e2b-scale ICB quant decode (dModel=1536, gs64/b4, 6 layers, 64 tokens so
// the one-off ICB build amortises and the per-token REPLAY dominates). Real dims (input-rms fusion
// engages). Synthetic weights — the perf delta is the dispatch/barrier structure, not the values.
// Measures pkg/native (NOT lthn-mlx serve, which is the pkg/metal cgo engine).
func spikeE2BFixture(tb testing.TB) (inputs [][]byte, layers []QuantizedLayerWeights, arch model.Arch) {
	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers = 1536, 8, 1, 256, 6144, 32, 32 // real e2b layer count
	const groupSize, bits = 64, 4
	const nTokens = 64
	arch = archFixture(tb, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	inputs = decodeInputsFixture(nTokens, dModel)
	layers = make([]QuantizedLayerWeights, nLayers)
	for li := range layers {
		layers[li] = quantizedLayerFixture(tb, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, li+3)
	}
	if b, ok := tb.(*testing.B); ok {
		b.SetBytes(int64(nTokens))
	}
	return inputs, layers, arch
}

// TestSpikeFineGrainedReplayMatchesCoarse documents an invalid R&D path: splitting ICB replay into
// fine-grained ExecuteCommandsInBufferWithRange calls plus encoder memory barriers does not preserve the
// dependency ordering provided by per-command ICB barriers. Keep production on coarse ICB barriers or the
// pipelined batch path; keep the benchmark below timing-only while this experiment remains archived.
func TestSpikeFineGrainedReplayMatchesCoarse(t *testing.T) {
	requireNativeRuntime(t)
	t.Skip("fine-grained ICB replay is an invalid R&D spike: encoder memory barriers between ICB ranges do not enforce command dependencies")
	const dModel, nHeads, nKV, headDim, dFF, maxLen = 1536, 8, 1, 256, 6144, 128
	inputs, layers, arch := spikeE2BFixture(t)

	fineGrainedReplay = false
	coarse, err := DecodeForwardArchICBQuant(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm)
	if err != nil {
		t.Fatalf("coarse: %v", err)
	}
	fineGrainedReplay = true
	fine, err := DecodeForwardArchICBQuant(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm)
	fineGrainedReplay = false
	if err != nil {
		t.Fatalf("fine: %v", err)
	}
	for tok := range coarse {
		if cos := cosineBF16(fine[tok], coarse[tok]); cos < 0.9999 {
			t.Fatalf("tok%d: fine-grained replay cosine=%.6f vs coarse — memory barriers don't enforce the deps", tok, cos)
		}
	}
	t.Logf("fine-grained replay matches coarse across %d tokens", len(coarse))
}

// BenchmarkSpikeE2BDecodeFineGrained records the archived fine-grained replay timing only. It is not a
// correctness-backed production candidate unless the skipped diagnostic above starts matching coarse replay.
func BenchmarkSpikeE2BDecodeFineGrained(b *testing.B) {
	fineGrainedReplay = true
	defer func() { fineGrainedReplay = false }()
	spikeE2BDecode(b)
}

func spikeE2BDecode(b *testing.B) {
	requireNativeRuntime(b)
	const dModel, nHeads, nKV, headDim, dFF, maxLen = 1536, 8, 1, 256, 6144, 128
	inputs, layers, arch := spikeE2BFixture(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := DecodeForwardArchICBQuant(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkSpikeE2BReplayOnly isolates the STEADY-STATE per-token cost: record the ICB ONCE, then replay
// a single token per b.N iteration — vs BenchmarkSpikeE2BDecode, which re-records AND replays the whole
// 64-token sequence every iteration (so its allocs/op bury the per-token figure under one-time recording).
// This is the number production decode actually pays per token, and the one that surfaces a per-token
// replay LEAK if one exists (a replay that's truly at the floor allocs only its inherent output copy).
func BenchmarkSpikeE2BReplayOnly(b *testing.B) {
	requireNativeRuntime(b)
	const dModel, nHeads, nKV, headDim, dFF, maxLen = 1536, 8, 1, 256, 6144, 128
	inputs, layers, arch := spikeE2BFixture(b)
	specs := arch.Layer
	var r *archICBReplay
	withAutoreleasePool(func() {
		kCaches := make([]metal.MTLBuffer, len(layers))
		vCaches := make([]metal.MTLBuffer, len(layers))
		for li := range specs {
			if specs[li].OwnsCache() {
				cb := uint(maxLen * nKV * headDimOf(specs[li], headDim) * bf16Size)
				kCaches[li] = device.NewBufferWithLengthOptions(cb, metal.MTLResourceStorageModeShared)
				vCaches[li] = device.NewBufferWithLengthOptions(cb, metal.MTLResourceStorageModeShared)
			}
		}
		var err error
		r, err = recordArchICBQuant(layers, specs, kCaches, vCaches, nil, 0, 0, 0, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, simpleICBRope(arch.RopeBase, headDim), arch.AttnScale, arch.Eps, arch.ValueNorm)
		if err != nil {
			b.Fatal(err)
		}
	})
	emb := inputs[0]
	b.SetBytes(1)
	b.ResetTimer()
	withAutoreleasePool(func() { // one pool for the whole replay loop, mirroring runBatch
		for i := 0; i < b.N; i++ {
			_ = r.stepBody(emb, 0, nil)
		}
	})
}

// BenchmarkSpikeE2BDecode — current barrier structure (input-rms fused per the recorder gate).
func BenchmarkSpikeE2BDecode(b *testing.B) { spikeE2BDecode(b) }

// BenchmarkSpikeE2BDecodeNoBarrier — ALL barriers off: the absolute no-barrier ceiling (the "311"
// floor; output is racy garbage, timing only). The gap to BenchmarkSpikeE2BDecode is the TOTAL barrier
// cost — how much is actually on the table, and whether element-wise (~4%) or the matmul tier owns it.
func BenchmarkSpikeE2BDecodeNoBarrier(b *testing.B) {
	allBarriersOffForTest = true
	defer func() { allBarriersOffForTest = false }()
	spikeE2BDecode(b)
}

// BenchmarkSpikeE2BDecodeReencode — the RE-ENCODE path (regular Metal encoder, fine-grained hazard
// tracking) instead of the ICB (COARSE wait-all-prior barriers). If this beats BenchmarkSpikeE2BDecode,
// the COARSE barrier is the cost and the lever is finer sync; if it's slower/equal, the barrier cost is
// inherent and only fusion removes it. This decides which matmul-tier fix to build.
func BenchmarkSpikeE2BDecodeReencode(b *testing.B) {
	requireNativeRuntime(b)
	const dModel, nHeads, nKV, headDim, dFF, maxLen = 1536, 8, 1, 256, 6144, 128
	inputs, layers, arch := spikeE2BFixture(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := DecodeForwardArchQuant(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, arch.RopeBase, arch.AttnScale, arch.Eps, arch.ValueNorm); err != nil {
			b.Fatal(err)
		}
	}
}
