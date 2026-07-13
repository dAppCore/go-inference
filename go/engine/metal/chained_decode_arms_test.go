// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"os"
	"slices"
	"testing"

	"github.com/tmc/apple/metal"
)

// TestChainedDecodeArmsAgree pins the three greedy decode arms to one another
// on a no-ICB (MoE) quant fixture: the chained live tail WITH submit-ahead
// (the production serve shape), the chain with the speculative link disabled,
// and the plain host loop. The original fork receipt: the submit-ahead link's
// encode wrote the next position into the ONE shared offBuf while the
// previous link was committed-not-waited — the in-flight link's kernels read
// the clobbered position at execution (RoPE/KV row), so its KV landed wrong
// and every later token inherited the corruption (token 3 right, token 4 on
// wrong, deterministically). rotateOffBuf gives each step encode its own
// position buffer; this gate holds the three arms together so the class
// cannot return silently.
func TestChainedDecodeArmsAgree(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set — GPU decode fixture")
	}
	m := laneSetQuantMoEFixtureModel(t)
	defer m.Close()
	prompt := []int32{2, 9, 4, 6, 8, 3}
	const n = 8

	run := func(disableSpec, disableChain bool) []int32 {
		t.Helper()
		liveSubmitAheadDisabled = disableSpec
		stepGreedyChainDisabled = disableChain
		defer func() { liveSubmitAheadDisabled = false; stepGreedyChainDisabled = false }()
		s, err := m.OpenSession()
		if err != nil {
			t.Fatalf("OpenSession(spec=%v chain=%v): %v", !disableSpec, !disableChain, err)
		}
		as := s.(*ArchSession)
		defer as.Close()
		got, err := as.Generate(prompt, n, -1)
		if err != nil {
			t.Fatalf("Generate(spec=%v chain=%v): %v", !disableSpec, !disableChain, err)
		}
		return got
	}
	chainSpec := run(false, false)
	chainNoSpec := run(true, false)
	host := run(false, true)
	if !slices.Equal(chainSpec, chainNoSpec) {
		t.Fatalf("chain+spec %v != chain-nospec %v — the submit-ahead link diverges again", chainSpec, chainNoSpec)
	}
	if !slices.Equal(chainSpec, host) {
		t.Fatalf("chain %v != host loop %v — the decode arms fork again", chainSpec, host)
	}
	if len(chainSpec) != n {
		t.Fatalf("generated %d tokens, want %d", len(chainSpec), n)
	}
}

// TestChainedDecodeEmbedTwins pins the two next-input embed producers to each
// other on the quant fixture: the HOST embed (embedID → CPU dequant) and the
// GPU next-inputs gather (encNextInputsGPU — the chained tail's producer)
// must return the same bytes for the same id, or the chained and host arms
// drift apart one ulp at a time.
func TestChainedDecodeEmbedTwins(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set — GPU decode fixture")
	}
	m := laneSetQuantMoEFixtureModel(t)
	defer m.Close()
	sess, err := m.OpenSession()
	if err != nil {
		t.Fatalf("OpenSession: %v", err)
	}
	as := sess.(*ArchSession)
	defer as.Close()
	if as.encNextInputsGPU == nil {
		t.Fatal("fixture session has no GPU next-inputs seam — the twin proof is vacuous")
	}
	dModel := as.arch.Hidden
	tokenBuf := device.NewBufferWithLengthOptions(4, metal.MTLResourceStorageModeShared)
	embOut := device.NewBufferWithLengthOptions(uint(dModel*bf16Size), metal.MTLResourceStorageModeShared)
	sc := as.plScratchNew()

	for _, id := range []int32{1, 5, 7, 15, 26, 34, 42, 47} {
		host, err := as.embedID(id)
		if err != nil {
			t.Fatalf("embedID(%d): %v", id, err)
		}
		*(*int32)(tokenBuf.Contents()) = id
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		if err := as.encNextInputsGPU(enc, tokenBuf, embOut, sc); err != nil {
			t.Fatalf("encNextInputsGPU(%d): %v", id, err)
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if gpu := as.state.bufferBytes(embOut, dModel*bf16Size); !bytes.Equal(host, gpu) {
			t.Fatalf("id %d: HOST embed != GPU embed — the chained tail's input producer drifted", id)
		}
	}
}
