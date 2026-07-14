// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"sync/atomic"
	"time"

	core "dappco.re/go"
	"github.com/tmc/apple/foundation"
	"github.com/tmc/apple/metal"
)

// gpu_capture.go is the engine's one-shot programmatic Metal capture trigger — the
// kernel-level instrument (#393). Every receipt to date is a Go wall-clock or a whole
// command buffer's GPU span; what happens INSIDE a dispatch (occupancy, the
// ALU/memory/synchronisation limiter, register pressure, per-line shader cost) is
// invisible from the host. This trigger writes a .gputrace that Xcode's Metal Debugger
// opens with exactly that per-dispatch view.
//
// Arming: LTHN_GPU_CAPTURE=/path/round.gputrace. The capture starts immediately before
// the next command-buffer submission through the engine's commit funnel and stops after
// N of them (LTHN_GPU_CAPTURE_CBS, default 2 — one chained decode round is forward+head;
// a batched prefill chunk is 1–2). The capture object is the engine's shared queue, so
// every submission between start and stop lands in the trace regardless of which commit
// path issued it; the counter counts funnel commits only, which merely times the stop.
//
// Programmatic capture OUTSIDE Xcode also needs MTL_CAPTURE_ENABLED=1 in the process
// environment (the Metal framework's own gate; Xcode sets it for child processes). A
// refused start logs that hint once and disarms. One-shot per process; when off, the
// funnel pays a single atomic load.

const (
	gpuCaptureOff     int32 = iota // not armed, finished, or disarmed after a refusal
	gpuCaptureArmed                // armed — the next funnel commit starts the capture
	gpuCaptureRunning              // capturing — remaining counts down to the stop
)

var (
	gpuCaptureState     atomic.Int32
	gpuCaptureRemaining atomic.Int64
	gpuCapturePath      atomic.Pointer[string]
)

func init() {
	p := os.Getenv("LTHN_GPU_CAPTURE")
	if p == "" {
		return
	}
	n := int64(2)
	if v := os.Getenv("LTHN_GPU_CAPTURE_CBS"); v != "" {
		if r := core.Atoi(v); r.OK {
			if c, ok := r.Value.(int); ok && c > 0 {
				n = int64(c)
			}
		}
	}
	// LTHN_GPU_CAPTURE_TRIGGER defers arming until the named file appears, so a serve
	// can boot and warm first and the trace opens on a REAL decode round instead of the
	// boot command buffers. Without it, arm immediately (right for one-shot binaries).
	if trig := os.Getenv("LTHN_GPU_CAPTURE_TRIGGER"); trig != "" {
		go func() {
			for {
				if _, err := os.Stat(trig); err == nil {
					nativeTraceLog("gpu-capture: trigger file seen — arming for the next " + core.Itoa(int(n)) + " submissions\n")
					gpuCaptureArm(p, n)
					return
				}
				time.Sleep(250 * time.Millisecond)
			}
		}()
		return
	}
	gpuCaptureArm(p, n)
}

// gpuCaptureArm arms the one-shot capture for the next cbs funnel submissions. The env
// init above is the production caller; tests arm directly.
func gpuCaptureArm(path string, cbs int64) {
	gpuCapturePath.Store(&path)
	gpuCaptureRemaining.Store(cbs)
	gpuCaptureState.Store(gpuCaptureArmed)
}

// gpuCaptureBeforeCommit starts the capture immediately before the first armed
// submission, so the trace opens on a real engine round rather than a warmup straggler.
func gpuCaptureBeforeCommit() {
	if gpuCaptureState.Load() != gpuCaptureArmed {
		return
	}
	if !gpuCaptureState.CompareAndSwap(gpuCaptureArmed, gpuCaptureRunning) {
		return // another committer won the arm race
	}
	path := ""
	if p := gpuCapturePath.Load(); p != nil {
		path = *p
	}
	mgr := metal.GetMTLCaptureManagerClass().SharedCaptureManager()
	desc := metal.NewMTLCaptureDescriptor()
	desc.SetCaptureObject(queue) // the shared queue: every submission between start/stop is traced
	desc.SetDestination(metal.MTLCaptureDestinationGPUTraceDocument)
	desc.SetOutputURL(foundation.NewURLFileURLWithPath(path))
	if ok, err := mgr.StartCaptureWithDescriptorError(desc); !ok {
		msg := "capture refused"
		if err != nil {
			msg = err.Error()
		}
		nativeTraceLog("gpu-capture: START FAILED (" + msg + ") — programmatic capture needs MTL_CAPTURE_ENABLED=1 in the environment\n")
		gpuCaptureState.Store(gpuCaptureOff)
		return
	}
	nativeTraceLog("gpu-capture: recording " + path + "\n")
}

// gpuCaptureAfterCommit counts a funnel submission and, after the last armed one, stops
// the capture and finalises the .gputrace.
func gpuCaptureAfterCommit() {
	if gpuCaptureState.Load() != gpuCaptureRunning {
		return
	}
	if gpuCaptureRemaining.Add(-1) != 0 {
		return
	}
	metal.GetMTLCaptureManagerClass().SharedCaptureManager().StopCapture()
	gpuCaptureState.Store(gpuCaptureOff)
	path := ""
	if p := gpuCapturePath.Load(); p != nil {
		path = *p
	}
	nativeTraceLog("gpu-capture: wrote " + path + " — open in Xcode (Metal Debugger) for per-dispatch occupancy/limiter\n")
}
