// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"sync/atomic"
	"time"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// gpu_busy_probe.go — the GPU duty-cycle instrument for the host-orchestrated lanes (the composed
// Qwen hybrid especially). Every synchronous seam commits ONE command buffer and blocks on
// waitUntilCompletedFast; this probe charges each waited CB's GPUStartTime→GPUEndTime span to a
// process-global accumulator, and a background ticker prints, once a second, how much of that wall
// second the GPU was actually BUSY versus idle in the host submit/wait seam, plus the CB count.
//
// It answers the one fork-deciding question for the 14x composed-vs-mlx-lm gap (#18): is per-token
// time spent in GPU work (chase the qmv/kernels) or idle between thousands of tiny synchronous CB
// round-trips (chase a device-resident decode loop). Δbusy per wall-second IS the duty cycle:
//   Δbusy ≈ 1000ms  → GPU saturated, compute/bandwidth-bound → kernel is the lever.
//   Δbusy ≈  200ms  → GPU idle 80% of the time → the host seam is the lever (fewer, bigger CBs / ICB).
// Measuring Δ per wall-second (not cumulative from process start) sidesteps the load/idle windows —
// only the seconds where decode is actually running report a meaningful duty cycle.
//
// Armed by LTHN_GPU_BUSY (any non-empty value). Zero cost when unset: one relaxed bool load per CB.
var (
	gpuBusyArmed  bool
	gpuBusyNs     int64 // Σ GPUEndTime-GPUStartTime over every waited CB, nanoseconds
	gpuBusyCBs    int64 // count of charged command buffers
)

func init() {
	if os.Getenv("LTHN_GPU_BUSY") == "" {
		return
	}
	gpuBusyArmed = true
	go func() {
		start := time.Now()
		var lastBusy, lastCBs int64
		for range time.Tick(time.Second) {
			busy := atomic.LoadInt64(&gpuBusyNs)
			cbs := atomic.LoadInt64(&gpuBusyCBs)
			dBusy, dCBs := busy-lastBusy, cbs-lastCBs
			if dCBs == 0 {
				lastBusy, lastCBs = busy, cbs
				continue // no decode this second — don't print idle-window noise
			}
			nativeTraceLog(core.Sprintf(
				"gpu-busy: wall=%4.0fs  Δbusy=%6.1fms/s (%3.0f%% duty)  ΔcbS=%5d  µs/cb=%4.1f\n",
				time.Since(start).Seconds(),
				float64(dBusy)/1e6, 100*float64(dBusy)/1e9,
				dCBs, float64(dBusy)/1e3/float64(dCBs),
			))
			lastBusy, lastCBs = busy, cbs
		}
	}()
}

// chargeGPUBusy adds a completed command buffer's GPU-execution span to the duty-cycle accumulator.
// Called from waitUntilCompletedFast after the wait returns (GPU timestamps are then valid). Nil-cheap
// when the probe is unarmed — a single relaxed bool load, no objc round-trip. The parameter must stay
// the concrete struct type: an interface parameter boxes the struct on the heap at every call site
// BEFORE the armed check, taxing all encode paths +1 alloc per wait (#56).
func chargeGPUBusy(cb metal.MTLCommandBufferObject) {
	if !gpuBusyArmed {
		return
	}
	atomic.AddInt64(&gpuBusyNs, int64((cb.GPUEndTime()-cb.GPUStartTime())*1e9))
	atomic.AddInt64(&gpuBusyCBs, 1)
}
