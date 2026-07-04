// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// Package native is go-inference's Apple-GPU compute engine, living at path
// engine/metal — "metal" names the Apple Metal API this engine drives, NOT
// go-mlx's cgo pkg/metal (which is DELETED, never ported: it remains only as
// the byte-for-byte parity oracle in go-mlx beside its cross-engine tests).
// There is NO cgo and no mlx-c here: it dispatches the MLX Metal compute
// kernels directly from Go through tmc/apple's objc bridge (purego
// objc_msgSend), gated by darwin && arm64 build tags. The package clause stays
// `native` (historical continuity — the source and its ~900 diagnostic strings
// say native.*); the path engine/metal is the contract.
//
// It loads the SAME compiled mlx.metallib the cgo oracle uses and dispatches
// its kernels itself, replacing only MLX's host-side command-encode layer (the
// per-step re-encode that dominates decode). The kernels are shared; the encode
// path is what differs. Because decode and diffusion are fixed per-step command
// sequences, the payoff is recording the sequence once into an Indirect Command
// Buffer and replaying it — bypassing the re-encode the cgo path pays on every
// step. Every op was parity-tested byte-for-byte against the cgo oracle before
// it could be trusted; those parity tests stay in go-mlx beside pkg/metal.
//
// Usage:
//
//	out, err := native.Square([]float32{1, 2, 3}) // out = [1 4 9], on the GPU
package native

import (
	"os"
	"sync"

	core "dappco.re/go"
	"github.com/tmc/apple/foundation"
	"github.com/tmc/apple/metal"
)

// MetallibPathEnv names the environment variable that locates the compiled
// kernel library; it mirrors what the cgo MLX backend itself reads, so a single
// export drives both paths.
const MetallibPathEnv = "MLX_METALLIB_PATH"

var (
	initOnce            sync.Once
	device              metal.MTLDeviceObject
	queue               metal.MTLCommandQueue
	library             metal.MTLLibrary
	customLibrary       metal.MTLLibrary // go-mlx's own kernels (the fused gelu, future fused/novel ops)
	customLibraryLoaded bool             // true once the sibling kernels metallib loaded
	initErr             error
)

// ensureInit lazily creates the shared device + command queue and loads the
// metallib named by MLX_METALLIB_PATH. It is idempotent; the first failure is
// cached and returned to every caller (a device or metallib problem is fatal
// for the whole package, not per-op).
func ensureInit() error {
	initOnce.Do(func() {
		device = metal.MTLCreateSystemDefaultDevice()
		if device.ID == 0 {
			initErr = core.NewError("native: no system default Metal device")
			return
		}
		path := os.Getenv(MetallibPathEnv)
		if path == "" {
			initErr = core.NewError("native: " + MetallibPathEnv + " not set")
			return
		}
		url := foundation.GetNSURLClass().FileURLWithPath(path)
		lib, err := device.NewLibraryWithURLError(url)
		if err != nil {
			initErr = core.E("native.ensureInit", "load metallib", err)
			return
		}
		library = lib
		// Optional sibling metallib of go-mlx's own kernels (the fused gelu, etc.). Absent ⇒ those ops
		// fall back to the composed primitives. Lives beside the resolved main metallib.
		if kp := siblingMetallib(path, "lthn_kernels.metallib"); kp != "" {
			if klib, kerr := device.NewLibraryWithURLError(foundation.GetNSURLClass().FileURLWithPath(kp)); kerr == nil {
				customLibrary = klib
				customLibraryLoaded = true
			}
		}
		queue = device.NewCommandQueue()
	})
	return initErr
}

// siblingMetallib returns name placed in the same directory as metallibPath.
func siblingMetallib(metallibPath, name string) string {
	for i := len(metallibPath) - 1; i >= 0; i-- {
		if metallibPath[i] == '/' {
			return metallibPath[:i+1] + name
		}
	}
	return name
}

// nativeTraceEnabled reports whether the per-layer decode diagnostic is on
// (LTHN_NATIVE_TRACE set non-empty). DEBUG instrument: stepToken then flushes +
// reads back each layer's output hidden to log its max-abs + NaN/Inf count,
// localising where a decode degrades (e.g. the 12B hybrid layers). Off by
// default — the readback serialises the token, so it is never on a measured path.
func nativeTraceEnabled() bool { return os.Getenv("LTHN_NATIVE_TRACE") != "" }

// nativeTraceLog writes one diagnostic line to stderr (keeps os confined to this
// file; callers format with core.Sprintf).
func nativeTraceLog(line string) { _, _ = os.Stderr.WriteString(line) }
