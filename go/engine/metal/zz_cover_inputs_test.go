// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"
	"unsafe"
)

// zz_cover_inputs_test.go closes the input-shape and branch-selection legs the
// existing suites leave uncovered: short-circuit returns (n==0), the
// out-length mismatch validators on the *Into helpers, the small-n
// threadgroup branch in the cast encoder, the partial-rotary branch in the
// freqs rope, and the dimension validators on the public block ops. These need
// no global mutation — they are pure input-driven branches, so they are
// unconditional wins independent of the device-poking batches.

// TestCoverConstAndEmptyShortCircuits drives the n==0 short-circuits in the
// const-vector builders and the composed-op entry points: bf16ConstBytes(0) and
// fillConst(0) return nil, and GeluBF16/Gelu on an empty input return an empty
// (init-checked) result without dispatching.
func TestCoverConstAndEmptyShortCircuits(t *testing.T) {
	requireNativeRuntime(t)

	if got := bf16ConstBytes(0, 1.5); got != nil {
		t.Fatalf("bf16ConstBytes(0) = %v, want nil", got)
	}
	if got := fillConst(0, 1.5); got != nil {
		t.Fatalf("fillConst(0) = %v, want nil", got)
	}
	// non-zero, fresh keys to exercise the fill + cache-store paths too.
	if got := bf16ConstBytes(3, 0.123); len(got) != 3*bf16Size {
		t.Fatalf("bf16ConstBytes(3) len = %d, want %d", len(got), 3*bf16Size)
	}
	if got := fillConst(3, 0.123); len(got) != 3 {
		t.Fatalf("fillConst(3) len = %d, want 3", len(got))
	}

	if got, err := GeluBF16(nil); err != nil || len(got) != 0 {
		t.Fatalf("GeluBF16(nil) = (%v, %v), want (empty, nil)", got, err)
	}
	if got, err := Gelu(nil); err != nil || len(got) != 0 {
		t.Fatalf("Gelu(nil) = (%v, %v), want (empty, nil)", got, err)
	}
}

// TestCoverIntoOutLengthValidators hits the "out must be the same length"
// validators on the *Into helpers (runBinaryBF16Into, tanhBF16Into,
// RunUnaryInto, RunBinaryInto) by supplying a correctly-paired (a,b)/(in) but a
// wrong-length out. These legs guard a caller-supplied destination, so they are
// only reachable through the lower-level Into entry points.
func TestCoverIntoOutLengthValidators(t *testing.T) {
	requireNativeRuntime(t)

	four := toBF16Bytes(syntheticFloat32(4, 1)) // 8 bytes, well-formed bf16
	bad := make([]byte, len(four)+bf16Size)

	if err := runBinaryBF16Into("vv_Addbfloat16", four, four, bad); err == nil {
		t.Fatal("runBinaryBF16Into: expected out-length error")
	}
	if err := tanhBF16Into(four, bad); err == nil {
		t.Fatal("tanhBF16Into: expected out-length error")
	}

	inF := syntheticFloat32(4, 3)
	badF := make([]float32, len(inF)+1)
	if err := RunUnaryInto("v_Tanhfloat32float32", inF, badF); err == nil {
		t.Fatal("RunUnaryInto: expected out-length error")
	}
	if err := RunBinaryInto("vv_Addfloat32", inF, inF, badF); err == nil {
		t.Fatal("RunBinaryInto: expected out-length error")
	}

	// empty inputs pass the matched-length checks then hit the n==0 short-circuit
	// (no dispatch), covering the early-return legs in the Into helpers.
	if err := RunUnaryInto("v_Tanhfloat32float32", nil, nil); err != nil {
		t.Fatalf("RunUnaryInto(empty): %v", err)
	}
	if err := RunBinaryInto("vv_Addfloat32", nil, nil, nil); err != nil {
		t.Fatalf("RunBinaryInto(empty): %v", err)
	}
	if err := runBinaryBF16Into("vv_Addbfloat16", nil, nil, nil); err != nil {
		t.Fatalf("runBinaryBF16Into(empty): %v", err)
	}
	if err := tanhBF16Into(nil, nil); err != nil {
		t.Fatalf("tanhBF16Into(empty): %v", err)
	}
}

// TestCoverCastSmallNThreadgroup hits the small-n branch in encCopyCast
// (uint(n) < group ⇒ group = n) by encoding a widen of fewer than 256 elements.
// The existing roundtrip uses n=1024 so the branch never fired.
func TestCoverCastSmallNThreadgroup(t *testing.T) {
	requireNativeRuntime(t)

	const n = 4
	f := syntheticFloat32(n, 5)
	bf := toBF16Bytes(f)
	var back []float32
	withAutoreleasePool(func() {
		bfBuf := sharedBytes(bf)
		f32 := scratch(n)
		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		if err := encWidenBF16ToF32(enc, bfBuf, f32, n); err != nil {
			t.Fatalf("encWidenBF16ToF32 small-n: %v", err)
		}
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		back = make([]float32, n)
		copy(back, unsafe.Slice((*float32)(f32.Contents()), n))
	})
	for i := range f {
		if back[i] != f[i] {
			t.Fatalf("widen small-n[%d] = %v, want %v", i, back[i], f[i])
		}
	}
}

// TestCoverCastNarrowSmallNThreadgroup hits the same small-n branch through the
// narrow kernel, so encNarrowF32ToBF16's encCopyCast call covers the branch too.
func TestCoverCastNarrowSmallNThreadgroup(t *testing.T) {
	requireNativeRuntime(t)

	const n = 8
	f := syntheticFloat32(n, 9)
	var back []byte
	withAutoreleasePool(func() {
		fBuf := scratch(n)
		copy(unsafe.Slice((*float32)(fBuf.Contents()), n), f)
		bf2 := scratchBF16(n)
		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		if err := encNarrowF32ToBF16(enc, fBuf, bf2, n); err != nil {
			t.Fatalf("encNarrowF32ToBF16 small-n: %v", err)
		}
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		back = make([]byte, n*bf16Size)
		copy(back, unsafe.Slice((*byte)(bf2.Contents()), n*bf16Size))
	})
	want := toBF16Bytes(f)
	for i := range want {
		if back[i] != want[i] {
			t.Fatalf("narrow small-n byte %d = %#x, want %#x", i, back[i], want[i])
		}
	}
}

// TestCoverRoPEFreqsPartialRotary hits the partial-rotary branch in the public
// RoPEFreqsBF16 wrapper (rotaryDim < headDim ⇒ seed out with x for the
// pass-through tail) by roping with rotaryDim strictly below headDim. The guard
// suite only roped with rotaryDim == headDim, so the partial leg never fired.
// (The decode-executor encRoPEFreqsBF16To partial leg is covered by the
// architecture-session freqs path, not this standalone wrapper.)
func TestCoverRoPEFreqsPartialRotary(t *testing.T) {
	requireNativeRuntime(t)

	const nHeads, headDim, rotaryDim = 1, 64, 32
	x := toBF16Bytes(syntheticFloat32(nHeads*headDim, 7))
	invFreqs := plainRopeInvFreqsGuard(10000, rotaryDim) // len rotaryDim/2
	out, err := RoPEFreqsBF16(x, 1, nHeads, headDim, rotaryDim, invFreqs, 1, 0, false)
	if err != nil {
		t.Fatalf("RoPEFreqsBF16 partial rotary: %v", err)
	}
	if len(out) != len(x) {
		t.Fatalf("RoPEFreqsBF16 partial rotary out len = %d, want %d", len(out), len(x))
	}
	// the untouched tail (rotaryDim..headDim) must pass through byte-identical.
	for i := rotaryDim * bf16Size; i < headDim*bf16Size; i++ {
		if out[i] != x[i] {
			t.Fatalf("partial rotary modified pass-through byte %d", i)
		}
	}
}

// TestCoverRoPEFreqsTraditionalPipeline builds the traditional-rope freqs
// pipeline (traditional=true) so ropeFreqsPipelineBF16(true) is exercised in
// addition to the interleaved variant the suite already covered.
func TestCoverRoPEFreqsTraditionalPipeline(t *testing.T) {
	requireNativeRuntime(t)

	const nHeads, headDim = 1, 64
	x := toBF16Bytes(syntheticFloat32(nHeads*headDim, 11))
	invFreqs := plainRopeInvFreqsGuard(10000, headDim)
	if _, err := RoPEFreqsBF16(x, 1, nHeads, headDim, headDim, invFreqs, 1, 0, true); err != nil {
		t.Fatalf("RoPEFreqsBF16 traditional: %v", err)
	}
}

// TestCoverBlockDimensionValidators hits the leading dimension validators on the
// public block ops (AttentionBlock x/normWeight, MoERouterQuant x) by supplying
// a wrong-length leading buffer that passes ensureInit but trips the size guard.
func TestCoverBlockDimensionValidators(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, kvLen, dFF = 64, 2, 1, 32, 4, 128
	const gs, bits = 64, 4
	const eps = float32(1e-6)
	shortX := toBF16Bytes(syntheticFloat32(dModel-1, 1)) // wrong length

	normB := toBF16Bytes(syntheticFloat32(dModel, 3))
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 5)
	kb := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 7))
	vb := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 9))
	if _, err := AttentionBlock(shortX, normB, layer.WQ, layer.WO, kb, vb, dModel, nHeads, nKV, headDim, kvLen, 10000, 0.125, 0, eps); err == nil {
		t.Fatal("AttentionBlock: expected x/normWeight size error")
	}

	qRouter := quantWeightFixture(t, 4, dModel, gs, bits, 13)
	perExpertScale := toBF16Bytes([]float32{1, 0.75, 0.5, 0.25})
	if _, _, err := MoERouterQuant(shortX, normB, qRouter, perExpertScale, 4, 2, dModel, gs, bits, eps, true); err == nil {
		t.Fatal("MoERouterQuant: expected x size error")
	}
}

// TestCoverNormProjectICBReplaysFloor hits the replays<1 floor in NormProjectICB
// (replays = 1) by passing replays = 0; the suite always passed replays >= 1.
func TestCoverNormProjectICBReplaysFloor(t *testing.T) {
	requireNativeRuntime(t)

	const eps = float32(1e-6)
	out, err := NormProjectICB([]float32{1, 2}, []float32{1, 1}, []float32{1, 2, 3, 4}, 2, 2, eps, 0)
	if err != nil {
		t.Fatalf("NormProjectICB replays=0: %v", err)
	}
	if len(out) != 2 {
		t.Fatalf("NormProjectICB replays=0 out len = %d, want 2", len(out))
	}
}

// ropeICBTraditionalProbe builds the traditional-variant ICB rope pipeline so the
// `if traditional { trad = 1 }` branch in ropePipelineICB is covered. The
// production callers only ever request the non-traditional variant.
func TestCoverRopePipelineICBTraditional(t *testing.T) {
	requireNativeRuntime(t)

	pso, err := ropePipelineICB(true)
	if err != nil {
		t.Fatalf("ropePipelineICB(true): %v", err)
	}
	if pso == nil {
		t.Fatal("ropePipelineICB(true) returned a nil pipeline")
	}
}
