// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"os"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"github.com/tmc/apple/metal"
)

// Opt-in #283 instrument (env-gated real-model load): per-param-shape sampled decode cost on the real
// e2b checkpoint. Greedy is the floor; each sampled variant shows what its
// param axis adds per token. Kept as the sampled-lane measurement harness — its receipts chose the host top-k lane.
func TestSampledParamAxisCost(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" || os.Getenv("LEM_REAL_E2B") == "" {
		t.Skip("set MLX_METALLIB_PATH + LEM_REAL_E2B=1")
	}
	dir := os.Getenv("E2B_Q4_DIR")
	if dir == "" {
		t.Skip("set E2B_Q4_DIR")
	}
	const maxLen, n = 512, 48
	prompt := []int32{2, 1841, 689, 573, 6182, 576}

	run := func(name string, params *model.SampleParams) {
		sess, err := LoadDir(dir, maxLen)
		if err != nil {
			t.Fatalf("%s LoadDir: %v", name, err)
		}
		defer func() { _ = sess.Close() }()
		// warmup pass (shader JIT + first-touch), untimed
		if params == nil {
			if _, err := sess.Generate(prompt, 8, -1); err != nil {
				t.Fatalf("%s warmup: %v", name, err)
			}
		} else {
			if _, err := sess.GenerateSampledEach(prompt, 8, nil, model.NewSampler(7), *params, nil, nil); err != nil {
				t.Fatalf("%s warmup: %v", name, err)
			}
		}
		t0 := time.Now()
		if params == nil {
			if _, err := sess.Generate(prompt, n, -1); err != nil {
				t.Fatalf("%s generate: %v", name, err)
			}
		} else {
			if _, err := sess.GenerateSampledEach(prompt, n, nil, model.NewSampler(7), *params, nil, nil); err != nil {
				t.Fatalf("%s generate: %v", name, err)
			}
		}
		wall := time.Since(t0)
		t.Logf("%-28s %6.2f ms/token  (%5.1f tok/s)", name, wall.Seconds()*1000/n, n/wall.Seconds())
	}

	run("greedy (floor)", nil)
	run("temp-only 0.8", &model.SampleParams{Temperature: 0.8})
	run("temp+topk40", &model.SampleParams{Temperature: 0.8, TopK: 40})
	run("temp+topk40+topp0.95", &model.SampleParams{Temperature: 0.8, TopK: 40, TopP: 0.95})
	run("temp+topp0.95 (no topk)", &model.SampleParams{Temperature: 0.8, TopP: 0.95})
}

// TestSampledTopKLaneMicroAB times the three head-sampling lanes in isolation on
// the real e2b head (262k vocab): fused Q4 topk tiles, qmv+logits-topk tiles,
// and the logits-sample kernel (which also applies topK). One dispatch per
// iteration, own command buffer, wait each — per-dispatch GPU+sync cost.
func TestSampledTopKLaneMicroAB(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" || os.Getenv("LEM_REAL_E2B") == "" {
		t.Skip("set MLX_METALLIB_PATH + LEM_REAL_E2B=1")
	}
	dir := os.Getenv("E2B_Q4_DIR")
	if dir == "" {
		t.Skip("set E2B_Q4_DIR")
	}
	sess, err := LoadDir(dir, 64)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	defer func() { _ = sess.Close() }()
	if _, err := sess.Generate([]int32{2, 1841, 689}, 4, -1); err != nil {
		t.Fatalf("warmup: %v", err)
	}
	h := sess.headEnc
	if h == nil {
		t.Fatal("no headEncoder")
	}
	// REAL boundary hidden from the warmup decode — synthetic byte patterns
	// make selection kernels take degenerate fast paths and lie by 10x.
	if len(sess.retainedHidden) != sess.arch.Hidden*bf16Size {
		t.Fatalf("no retained hidden after warmup (len %d)", len(sess.retainedHidden))
	}
	hidden := append([]byte(nil), sess.retainedHidden...)
	params := model.SampleParams{Temperature: 0.8, TopK: 40, TopP: 0.95}
	const iters = 20

	lane := func(name string, encode func(enc metal.MTLComputeCommandEncoderObject, hb metal.MTLBuffer) error) {
		var wall time.Duration
		withAutoreleasePool(func() {
			hs, hb, herr := h.hiddenBuffer(hidden)
			if herr != nil {
				t.Fatalf("%s hiddenBuffer: %v", name, herr)
			}
			defer h.putHiddenScratch(hs)
			// warm one dispatch untimed
			cb := commandBufferFast(queue)
			enc := computeCommandEncoderFast(cb)
			if err := encode(enc, hb); err != nil {
				endEncodingFast(enc)
				t.Fatalf("%s warm encode: %v", name, err)
			}
			endEncodingFast(enc)
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
			t0 := time.Now()
			for i := 0; i < iters; i++ {
				cb := commandBufferFast(queue)
				enc := computeCommandEncoderFast(cb)
				if err := encode(enc, hb); err != nil {
					endEncodingFast(enc)
					t.Fatalf("%s encode: %v", name, err)
				}
				endEncodingFast(enc)
				commitCommandBufferFast(cb)
				waitUntilCompletedFast(cb)
			}
			wall = time.Since(t0)
		})
		t.Logf("%-30s %7.2f ms/dispatch", name, wall.Seconds()*1000/iters)
	}

	lane("fusedQ4 topk tiles", func(enc metal.MTLComputeCommandEncoderObject, hb metal.MTLBuffer) error {
		scratch, ok, err := h.encodeTopKSampleObjectAt(enc, hb, 0, params, 0.5, nil, true)
		if scratch != nil {
			defer h.putTopKScratch(scratch)
		}
		if err == nil && !ok {
			return core.NewError("fusedQ4 lane declined")
		}
		return err
	})
	lane("qmv+logits topk tiles", func(enc metal.MTLComputeCommandEncoderObject, hb metal.MTLBuffer) error {
		scratch, ok, err := h.encodeTopKSampleObjectAt(enc, hb, 0, params, 0.5, nil, false)
		if scratch != nil {
			defer h.putTopKScratch(scratch)
		}
		if err == nil && !ok {
			return core.NewError("qmv lane declined")
		}
		return err
	})
	lane("logits-sample kernel", func(enc metal.MTLComputeCommandEncoderObject, hb metal.MTLBuffer) error {
		scratch, ok, err := h.encodeLogitsSampleObjectAt(enc, hb, 0, params, 0.5, nil)
		if scratch != nil {
			defer h.putGreedyScratch(scratch)
		}
		if err == nil && !ok {
			return core.NewError("logits-sample lane declined")
		}
		return err
	})
	lane("greedy argmax (floor)", func(enc metal.MTLComputeCommandEncoderObject, hb metal.MTLBuffer) error {
		scratch, ok, err := h.encodeGreedy(enc, hb, nil)
		if scratch != nil {
			defer h.putGreedyScratch(scratch)
		}
		if err == nil && !ok {
			return core.NewError("greedy lane declined")
		}
		return err
	})
}

// TestSampledTopKGateBooleans prints every gate on the topK routing ladder for the
// bench params — which rung actually fires.
func TestSampledTopKGateBooleans(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" || os.Getenv("LEM_REAL_E2B") == "" {
		t.Skip("set MLX_METALLIB_PATH + LEM_REAL_E2B=1")
	}
	dir := os.Getenv("E2B_Q4_DIR")
	if dir == "" {
		t.Skip("set E2B_Q4_DIR")
	}
	sess, err := LoadDir(dir, 64)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	defer func() { _ = sess.Close() }()
	h := sess.headEnc
	params := model.SampleParams{Temperature: 0.8, TopK: 40, TopP: 0.95}
	t.Logf("headSampleTopKMaxK=%d vocab=%d dModel=%d gs=%d bits=%d quant=%v", headSampleTopKMaxK, h.vocab, h.dModel, h.groupSize, h.bits, h.quant)
	t.Logf("topKSampleUsable(free fn)          = %v", topKSampleUsable(params.TopK))
	t.Logf("h.topKSampleUsable(40)             = %v", h.topKSampleUsable(params.TopK))
	t.Logf("q4LMHeadTopKUsable                 = %v", q4LMHeadTopKUsable(h.dModel, h.vocab, h.groupSize, h.bits, params.TopK))
	t.Logf("qmvLogitsTopKUsable                = %v", qmvLogitsTopKUsable(h.dModel, h.vocab, h.groupSize, h.bits, params.TopK))
	t.Logf("preferFusedQ4TopK(40)              = %v", h.preferFusedQ4TopK(params.TopK))
	t.Logf("sampleTopKParamsEligible           = %v", sess.sampleTopKParamsEligible(params))
	t.Logf("sampleTopKTokenParamsEligible      = %v", sess.sampleTopKTokenParamsEligible(params))
	t.Logf("sampleLogitsTokenParamsEligible    = %v", sess.sampleLogitsTokenParamsEligible(params))
	t.Logf("sampledChainedGPUTailCanContinue   = %v", sess.sampledChainedGPUTailCanContinue(params, nil, nil))
	t.Logf("sampledPipelinedGPUTailCanContinue = %v", sess.sampledPipelinedGPUTailCanContinue(params, nil, nil))
}

// TestSampledPipelinedVsChained isolates the sampled tail variant: same
// params (temp+topk40), pipelined on vs off. The greedy pipelined tail is a
// win; if the SAMPLED pipelined tail is the 22ms/token monster, off beats on.
func TestSampledPipelinedVsChained(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" || os.Getenv("LEM_REAL_E2B") == "" {
		t.Skip("set MLX_METALLIB_PATH + LEM_REAL_E2B=1")
	}
	dir := os.Getenv("E2B_Q4_DIR")
	if dir == "" {
		t.Skip("set E2B_Q4_DIR")
	}
	const maxLen, n = 512, 48
	prompt := []int32{2, 1841, 689, 573, 6182, 576}
	params := model.SampleParams{Temperature: 0.8, TopK: 40}

	run := func(name string, pipe bool) {
		old := pipelinedGPUDecodeEnabled
		pipelinedGPUDecodeEnabled = pipe
		defer func() { pipelinedGPUDecodeEnabled = old }()
		sess, err := LoadDir(dir, maxLen)
		if err != nil {
			t.Fatalf("%s LoadDir: %v", name, err)
		}
		defer func() { _ = sess.Close() }()
		if _, err := sess.GenerateSampledEach(prompt, 8, nil, model.NewSampler(7), params, nil, nil); err != nil {
			t.Fatalf("%s warmup: %v", name, err)
		}
		t0 := time.Now()
		if _, err := sess.GenerateSampledEach(prompt, n, nil, model.NewSampler(7), params, nil, nil); err != nil {
			t.Fatalf("%s generate: %v", name, err)
		}
		wall := time.Since(t0)
		t.Logf("%-24s %6.2f ms/token  (%5.1f tok/s)", name, wall.Seconds()*1000/n, n/wall.Seconds())
	}

	run("sampled pipelined ON", true)
	run("sampled pipelined OFF", false)
}

// TestSampledHostSamplerCost times the pure-host sampling lane on real logits:
// headLogitsScratch (GPU matvec + readback) once, then sampleVocabBF16 in a
// loop — the candidate routing target if it beats the 17-33ms GPU selection
// kernels.
func TestSampledHostSamplerCost(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" || os.Getenv("LEM_REAL_E2B") == "" {
		t.Skip("set MLX_METALLIB_PATH + LEM_REAL_E2B=1")
	}
	dir := os.Getenv("E2B_Q4_DIR")
	if dir == "" {
		t.Skip("set E2B_Q4_DIR")
	}
	sess, err := LoadDir(dir, 64)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	defer func() { _ = sess.Close() }()
	if _, err := sess.Generate([]int32{2, 1841, 689}, 4, -1); err != nil {
		t.Fatalf("warmup: %v", err)
	}
	hidden := append([]byte(nil), sess.retainedHidden...)

	// logits production cost (GPU matvec + host readback), one dispatch each
	const iters = 20
	t0 := time.Now()
	var logits []byte
	for i := 0; i < iters; i++ {
		var lerr error
		logits, lerr = sess.headLogitsScratch(hidden, false)
		if lerr != nil {
			t.Fatalf("headLogitsScratch: %v", lerr)
		}
	}
	t.Logf("headLogitsScratch (matvec+readback)  %6.2f ms/call", time.Since(t0).Seconds()*1000/iters)

	sampler := model.NewSampler(7)
	for _, tc := range []struct {
		name   string
		params model.SampleParams
	}{
		{"host topk40", model.SampleParams{Temperature: 0.8, TopK: 40}},
		{"host topk40+topp0.95", model.SampleParams{Temperature: 0.8, TopK: 40, TopP: 0.95}},
		{"host topp0.95 only", model.SampleParams{Temperature: 0.8, TopP: 0.95}},
	} {
		t1 := time.Now()
		for i := 0; i < iters; i++ {
			if _, serr := sess.sampleVocabBF16(logits, sess.arch.Vocab, sampler, tc.params); serr != nil {
				t.Fatalf("%s: %v", tc.name, serr)
			}
		}
		t.Logf("sampleVocabBF16 %-22s %6.2f ms/call", tc.name, time.Since(t1).Seconds()*1000/iters)
	}
}
