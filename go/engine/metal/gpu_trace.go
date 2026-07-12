// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"time"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// batchedGPUTrace — per-stage GPU-time attribution for the batched dense pass. When
// LTHN_GPU_TRACE is set, the pass's single command buffer is SPLIT at named stage boundaries:
// each segment commits, waits, and charges its GPUStartTime→GPUEndTime span to the stage that
// just ran, accumulated across the layer loop. Splitting serialises the stages and pays a CB
// round-trip per checkpoint (~6 per layer), so the traced total runs SLOWER than production —
// the report prints both the per-bucket shares AND the traced total so the overhead is visible.
// Attribution is the product: where the GPU time actually goes at real model shapes, measured —
// not inferred from FLOP counts (the query-tiling episode showed how that road ends).
//
// Zero cost when the env is unset: one nil check per checkpoint.
type batchedGPUTrace struct {
	cb      metal.MTLCommandBufferObject
	stage   string
	seconds map[string]float64
	order   []string
	calls   int
}

// gpuTraceEnabled reports whether the batched pass should trace (env-gated, read per pass —
// a prefill runs a handful of chunks, the Getenv is noise).
func gpuTraceEnabled() bool { return os.Getenv("LTHN_GPU_TRACE") != "" }

// newBatchedGPUTrace adopts the pass's opening command buffer under the first stage name.
// Returns nil when tracing is off — every method is nil-safe. LTHN_GPU_TRACE=host arms
// the HOST spans only (hostSpan lines) without the per-segment command-buffer splitting,
// so wall-vs-seam decomposition reads at production GPU fidelity (#381 seams).
func newBatchedGPUTrace(cb metal.MTLCommandBufferObject, first string) *batchedGPUTrace {
	if !gpuTraceEnabled() || os.Getenv("LTHN_GPU_TRACE") == "host" {
		return nil
	}
	return &batchedGPUTrace{cb: cb, stage: first, seconds: map[string]float64{}}
}

func (t *batchedGPUTrace) charge() {
	span := float64(t.cb.GPUEndTime() - t.cb.GPUStartTime())
	if _, seen := t.seconds[t.stage]; !seen {
		t.order = append(t.order, t.stage)
	}
	t.seconds[t.stage] += span
	t.calls++
}

// checkpoint closes the current segment (end encoder → commit → wait), charges its GPU span to
// the stage that just ran, and opens a fresh command buffer + encoder under the next stage name.
// Off (nil receiver): returns enc unchanged.
func (t *batchedGPUTrace) checkpoint(enc metal.MTLComputeCommandEncoderObject, next string) metal.MTLComputeCommandEncoderObject {
	if t == nil {
		return enc
	}
	endEncodingFast(enc)
	commitCommandBufferFast(t.cb)
	waitUntilCompletedFast(t.cb)
	t.charge()
	t.cb = commandBufferFast(queue)
	t.stage = next
	return computeCommandEncoderFast(t.cb)
}

// commandBuffer returns the live command buffer — the pass's final end/commit/wait must target
// the trace's current CB once checkpoints have rotated it.
func (t *batchedGPUTrace) commandBuffer(fallback metal.MTLCommandBufferObject) metal.MTLCommandBufferObject {
	if t == nil {
		return fallback
	}
	return t.cb
}

// hostSpan logs a host-side phase duration under the same trace gate — the wall-vs-GPU gap's
// decomposition (embedding gathers, the PLE slab scatter, the paged<->linear KV syncs).
func hostSpan(name string, since time.Time, rows int) {
	if !gpuTraceEnabled() {
		return
	}
	nativeTraceLog(core.Sprintf("gpu-trace: host  %-16s %7.1fms  rows=%d\n", name, float64(time.Since(since))/1e6, rows))
}

// finish charges the final segment (the caller has already committed+waited the last CB) and
// prints the per-bucket table to stderr.
func (t *batchedGPUTrace) finish(rows, basePos int) {
	if t == nil {
		return
	}
	t.charge()
	total := 0.0
	for _, s := range t.seconds {
		total += s
	}
	nativeTraceLog(core.Sprintf("gpu-trace: batched pass rows=%d basePos=%d segments=%d gpuTotal=%.1fms\n", rows, basePos, t.calls, total*1e3))
	for _, name := range t.order {
		s := t.seconds[name]
		pct := 0.0
		if total > 0 {
			pct = 100 * s / total
		}
		nativeTraceLog(core.Sprintf("gpu-trace:   %-18s %7.1fms  %5.1f%%\n", name, s*1e3, pct))
	}
}
