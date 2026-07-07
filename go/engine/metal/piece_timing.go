// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "time"

// piece_timing.go is a decode-piece GPU-time diagnostic: where does the per-token wall go across the three
// GPU pieces — the PLE projection, the ICB layer stack, and the head (final norm + lm_head)? Each piece does
// its own Commit+WaitUntilCompleted, so the wall-clock of the call is ~its GPU time. Off in production
// (pieceTimingOn=false → ptStart returns the zero Time and ptEnd is a no-op; the compiler inlines both to a
// bool check, no allocation). A test flips it on, resets pieceNs, decodes, and reads the split.
var (
	pieceTimingOn bool
	pieceNs       [3]int64 // [0]=PLE  [1]=ICB layer stack  [2]=head
	icbGPUNs      int64    // ICB replay GPU execution span (GPUEndTime-GPUStartTime), to split GPU vs host in the ICB wall
	// chainedGPUSpanNs accumulates the per-token GPU execution span across a chained-GPU decode (gated by
	// pieceTimingOn). Σ span vs wall is the remaining per-token host/sync gap — the headroom a submit-ahead
	// pipeline could overlap. Reset before a measured run.
	chainedGPUSpanNs int64
	// allBarriersOffForTest records the ICB with NO barriers — a TIMING-ONLY ceiling probe (output is racy
	// garbage). The no-barrier GPU span is the floor fusion chases: span_with_barriers − span_no_barriers
	// is the barrier-serialisation cost. Never true in production.
	allBarriersOffForTest bool
	// ffnBarriersOffForTest drops the FFN's barriers (gate / gelu·up / down) only — a TIMING-ONLY ceiling
	// probe (output races) for how much GPU-span a fused FFN megakernel could reclaim. Never true in prod.
	ffnBarriersOffForTest bool
	// fineGrainedReplay records the ICB barrier-free and enforces each true dependency with an encoder
	// memory barrier (MemoryBarrierWithScope, resource-scoped) during replay, split into ranges at the
	// barrier points — testing whether a memory-coherency barrier lets dependent ops pipeline vs the
	// coarse all-prior ICB SetBarrier's full drain. Experiment flag.
	fineGrainedReplay bool
	// pipelinedBatchDisabled forces the serial runBatch in DecodeForwardArchICBQuant (default: pipelined
	// double-buffer for batches ≥4 tokens). Test hook to byte-compare serial vs pipelined.
	pipelinedBatchDisabled bool
	// liveSubmitAheadDisabled forces the chained live decode to wait each link before encoding
	// the next (no speculative command buffer). Test/bench hook for A/B token compares.
	liveSubmitAheadDisabled bool
	// moeConcurrentDisabled forces the serial (single-encoder) MoE block encode even when the
	// concurrent-pass lane's gates hold. Test/bench hook for serial-vs-concurrent byte compares.
	moeConcurrentDisabled bool
	// stepGreedyChainDisabled forces the serial greedy+stepID decode loop (default: chain the prior
	// token's stepBody with the next token's head+argmax in one command buffer). Test/bench hook.
	stepGreedyChainDisabled bool
	// chainedGPUInputsDisabled forces the host embed/PLE chained path even when the GPU next-inputs seam
	// is available (default: produce each step's next emb+pli on-GPU, one command buffer/token for e2b).
	chainedGPUInputsDisabled bool
	// pipelinedGPUDecodeEnabled opts the chained-GPU decode into submit-ahead: two ICBs over shared KV, the
	// host submits token t+1 before reading t (1-ahead, discard-safe for greedy: the chain feeds each
	// step's input on-GPU from the prior argmax, so the pre-submitted step is always correct and a stop
	// merely discards one speculative step). On by default — worth ~0.6ms/token on e2b-4bit (162→180
	// tok/s); the parity tests byte-compare it against the serial loop, and tests that need the
	// unpipelined lane clear it explicitly.
	pipelinedGPUDecodeEnabled = true
)

// SetPipelinedGPUDecode flips the submit-ahead decode default for the process — the
// engine-level knob behind `lem generate -pipeline=false` (A/B: chained serial loop vs
// 1-ahead pipelined). Sessions read the toggle per Generate call, so it applies to
// sessions already open. SetPipelinedGPUDecode(false); defer SetPipelinedGPUDecode(true)
// brackets a serial-lane measurement.
func SetPipelinedGPUDecode(on bool) { pipelinedGPUDecodeEnabled = on }

func ptStart() time.Time {
	if pieceTimingOn {
		return time.Now()
	}
	return time.Time{}
}

func ptEnd(idx int, t time.Time) {
	if pieceTimingOn {
		pieceNs[idx] += int64(time.Since(t))
	}
}

// layerSpanProbeForTest, when non-nil (test-only), makes stepToken commit per layer and
// accumulate each layer's GPU span — the decode-piece diagnostic at layer grain.
var layerSpanProbeForTest []int64
