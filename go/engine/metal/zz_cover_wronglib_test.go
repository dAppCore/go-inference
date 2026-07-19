// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	core "dappco.re/go"
)

// zz_cover_wronglib_test.go closes the kernel-lookup failure legs in the
// low-level pipeline builders by pointing the library at a REAL-but-wrong
// metallib that lacks the requested kernel. The existing guard suite nulls the
// main library, which trips the EARLIER `library == nil` guard, leaving these
// downstream legs unreachable. Two flavours of builder respond differently to a
// wrong library, and BOTH legs are valuable:
//   - plain lookups (pipelineFor / pipelineForICB use NewFunctionWithName):
//     the wrong library returns nil cleanly ⇒ the `fn == nil` not-found leg.
//   - function-constant lookups (ropePipeline*, sdpaVectorPipeline*,
//     ropeFreqsPipelineBF16 use NewFunctionWithNameConstantValuesError): the
//     wrong library surfaces an ERROR ⇒ the `if err != nil` leg (the fn==nil
//     sibling stays shadowed behind it — that is the inherent floor).
// The swaps: library = customLibrary (the tiny lthn_kernels.metallib, only the
// fused gelu) for the main builders; customLibrary = library (the big
// mlx.metallib, no lthn_gelu) for the fused-gelu builders; or a straight null for
// the library-unavailable legs. resetNativePipelineCachesForCoverage (defined in
// coverage_guard_test.go) clears every PSO cache + the gelu sync.Once, so each
// builder rebuilds against the swapped library and restores cleanly afterwards.

// withWrongMainLibrary runs fn with the main library pointed at customLibrary (a
// valid metallib lacking the main kernels) and every PSO cache cleared, then
// restores the library + caches. fn is expected to surface a "kernel not found".
func withWrongMainLibrary(t *testing.T, fn func()) {
	t.Helper()
	if customLibrary == nil {
		t.Skip("customLibrary (lthn_kernels.metallib) not loaded")
	}
	oldLib := library
	t.Cleanup(func() {
		library = oldLib
		resetNativePipelineCachesForCoverage()
	})
	resetNativePipelineCachesForCoverage()
	library = customLibrary
	fn()
}

// withNulledMainLibrary runs fn with the main library nulled and caches cleared —
// for the ICB builders' `library == nil` legs that the per-op guard test doesn't
// reach directly (it goes through the op wrappers, which fail earlier).
func withNulledMainLibrary(t *testing.T, fn func()) {
	t.Helper()
	oldLib := library
	t.Cleanup(func() {
		library = oldLib
		resetNativePipelineCachesForCoverage()
	})
	resetNativePipelineCachesForCoverage()
	library = nil
	fn()
}

// withWrongCustomLibrary runs fn with customLibrary pointed at the main library
// (which lacks the lthn_gelu kernel) and caches cleared — for the fused-gelu
// builders' fn==nil legs.
func withWrongCustomLibrary(t *testing.T, fn func()) {
	t.Helper()
	oldCustom := customLibrary
	t.Cleanup(func() {
		customLibrary = oldCustom
		resetNativePipelineCachesForCoverage()
	})
	resetNativePipelineCachesForCoverage()
	customLibrary = library
	fn()
}

// withNulledCustomLibrary runs fn with customLibrary nulled and caches cleared —
// for the fused-gelu builders' `customLibrary == nil` legs.
func withNulledCustomLibrary(t *testing.T, fn func()) {
	t.Helper()
	oldCustom := customLibrary
	t.Cleanup(func() {
		customLibrary = oldCustom
		resetNativePipelineCachesForCoverage()
	})
	resetNativePipelineCachesForCoverage()
	customLibrary = nil
	fn()
}

// TestCoverMainBuilderKernelNotFound covers the fn==nil legs in the main-library
// pipeline builders (pipelineFor, ropePipeline, ropePipelineBF16,
// ropeFreqsPipelineBF16, sdpaVectorPipeline) by driving the public op against the
// wrong library so the kernel lookup returns nil.
func TestCoverMainBuilderKernelNotFound(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, kvLen = 64, 1, 1, 64, 1
	x32 := syntheticFloat32(dModel, 3)
	xb := toBF16Bytes(x32)
	kb := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 5))
	vb := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 7))
	invFreqs := plainRopeInvFreqsGuard(10000, headDim)

	withWrongMainLibrary(t, func() {
		if _, err := RunUnary("v_Squarefloat32float32", x32); err == nil {
			t.Fatal("pipelineFor: expected kernel-not-found")
		}
		// encCopyCast pipelineFor error leg (the bf16↔f32 cast kernels are absent).
		withAutoreleasePool(func() {
			bfBuf := sharedBytes(toBF16Bytes(x32))
			dst := scratch(len(x32))
			cb := queue.CommandBuffer()
			enc := cb.ComputeCommandEncoder()
			if err := encWidenBF16ToF32(enc, bfBuf, dst, len(x32)); err == nil {
				t.Fatal("encCopyCast: expected kernel-not-found")
			}
			enc.EndEncoding()
		})
		if _, err := RoPE(x32, 1, nHeads, headDim, 10000, 1, 0, false); err == nil {
			t.Fatal("ropePipeline: expected kernel-not-found")
		}
		if _, err := RoPEBF16(xb, 1, nHeads, headDim, 10000, 1, 0, false); err == nil {
			t.Fatal("ropePipelineBF16: expected kernel-not-found")
		}
		if _, err := RoPEFreqsBF16(xb, 1, nHeads, headDim, headDim, invFreqs, 1, 0, false); err == nil {
			t.Fatal("ropeFreqsPipelineBF16: expected kernel-not-found")
		}
		// #28 changed this contract: with the MAIN library wrong but the lthn
		// customLibrary live, an absent fixed width no longer errors — the
		// runtime-dim fallback serves it. The error leg now needs BOTH gone.
		// Explicit save/restore (not withNulledCustomLibrary): t.Cleanup fires
		// at test end, and the success half below needs the library back NOW.
		oldCustom := customLibrary
		resetNativePipelineCachesForCoverage()
		customLibrary = nil
		if _, err := SDPA(xb, kb, vb, 1, nHeads, nKV, headDim, kvLen, 0.125); err == nil {
			t.Fatal("sdpaVectorPipeline: expected kernel-not-found with custom library nulled")
		}
		customLibrary = oldCustom
		resetNativePipelineCachesForCoverage()
		if _, err := SDPA(xb, kb, vb, 1, nHeads, nKV, headDim, kvLen, 0.125); err != nil {
			t.Fatalf("sdpaVector runtime-dim fallback: expected success with live custom library, got %v", err)
		}
	})
}

// TestCoverICBBuilderKernelNotFound covers the fn==nil legs in the ICB-capable
// builders (pipelineForICB, ropePipelineICB, sdpaVectorPipelineICB) by calling
// them directly against the wrong library.
func TestCoverICBBuilderKernelNotFound(t *testing.T) {
	requireNativeRuntime(t)

	const headDim = 64
	withWrongMainLibrary(t, func() {
		if _, err := pipelineForICB("rmsbfloat16"); err == nil {
			t.Fatal("pipelineForICB: expected kernel-not-found")
		}
		if _, err := ropePipelineICB(false); err == nil {
			t.Fatal("ropePipelineICB: expected kernel-not-found")
		}
		if _, err := sdpaVectorPipelineICB(core.Sprintf("sdpa_vector_bfloat16_t_%d_%d", headDim, headDim)); err == nil {
			t.Fatal("sdpaVectorPipelineICB: expected kernel-not-found")
		}
	})
}

// TestCoverICBBuilderLibraryUnavailable covers the `library == nil` legs in
// ropePipelineICB and sdpaVectorPipelineICB by calling them directly with the
// main library nulled (the guard suite reaches these only through the recorders,
// which fail at the first pipeline; here the builders are hit head-on).
func TestCoverICBBuilderLibraryUnavailable(t *testing.T) {
	requireNativeRuntime(t)

	const headDim = 64
	withNulledMainLibrary(t, func() {
		if _, err := ropePipelineICB(false); err == nil {
			t.Fatal("ropePipelineICB: expected library-unavailable")
		}
		if _, err := sdpaVectorPipelineICB(core.Sprintf("sdpa_vector_bfloat16_t_%d_%d", headDim, headDim)); err == nil {
			t.Fatal("sdpaVectorPipelineICB: expected library-unavailable")
		}
	})
}

// TestCoverFusedGeluBuilderKernelNotFound covers the lthn_gelu fn==nil legs in
// geluPipeline (lthn_kernels.go) and geluPipelineICB (icb.go) by pointing
// customLibrary at the main mlx metallib, which lacks the lthn_gelu kernel.
func TestCoverFusedGeluBuilderKernelNotFound(t *testing.T) {
	requireNativeRuntime(t)

	const n = 8
	gate := toBF16Bytes(syntheticFloat32(n, 3))
	up := toBF16Bytes(syntheticFloat32(n, 5))
	withWrongCustomLibrary(t, func() {
		if _, err := geluGateMulFused(gate, up, n); err == nil {
			t.Fatal("geluPipeline: expected lthn_gelu kernel-not-found")
		}
		if _, err := geluPipelineICB(); err == nil {
			t.Fatal("geluPipelineICB: expected lthn_gelu kernel-not-found")
		}
	})
}

// TestCoverFusedGeluBuilderLibraryUnavailable covers the `customLibrary == nil`
// legs in geluPipeline and geluPipelineICB by nulling customLibrary.
func TestCoverFusedGeluBuilderLibraryUnavailable(t *testing.T) {
	requireNativeRuntime(t)

	const n = 8
	gate := toBF16Bytes(syntheticFloat32(n, 3))
	up := toBF16Bytes(syntheticFloat32(n, 5))
	withNulledCustomLibrary(t, func() {
		if _, err := geluGateMulFused(gate, up, n); err == nil {
			t.Fatal("geluPipeline: expected custom-library-unavailable")
		}
		if _, err := geluPipelineICB(); err == nil {
			t.Fatal("geluPipelineICB: expected custom-library-unavailable")
		}
	})
}
