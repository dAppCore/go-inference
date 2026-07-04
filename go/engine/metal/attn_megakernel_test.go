// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"os"
	"sync"
	"testing"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

var (
	attnMegaPSOOnce sync.Once
	attnMegaPSO     metal.MTLComputePipelineState
	attnMegaErr     error
)

func attnMegaPipeline() (metal.MTLComputePipelineState, error) {
	attnMegaPSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			attnMegaErr = core.NewError("attnmega: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_attn_megakernel")
		if fn == nil || fn.GetID() == 0 {
			attnMegaErr = core.NewError("attnmega: kernel not found")
			return
		}
		attnMegaPSO, attnMegaErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return attnMegaPSO, attnMegaErr
}

// TestAttnMegakernel validates the attention half in ONE dispatch (RMSNorm → QKV → RoPE → cache → SDPA → O
// → residual, four stages separated by device-scope grid barriers, every cross-TG handoff through atomics)
// against a host reference computing the identical math. A pass proves the staged megakernel structure +
// the atomic cross-TG handoffs are correct on a real 4-stage attention — the second half of the full-layer
// megakernel (the FFN half is lthn_ffn_megakernel, proven token-identical).
func TestAttnMegakernel(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("device init: %v", err)
	}
	pso, err := attnMegaPipeline()
	if err != nil {
		t.Skipf("attnmega pipeline: %v", err)
	}
	const dModel, nHeads, nKVHeads, headDim, maxLen, pos = 128, 2, 1, 64, 8, 3
	const numTG, threadsPerTG = 8, 64
	const maxSpin = int32(1_000_000)
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	qDim, kvDim, kvLen, hd2 := nHeads*headDim, nKVHeads*headDim, pos+1, headDim/2
	gqa := nHeads / nKVHeads

	xf := syntheticFloat32(dModel, 1)
	nwf := syntheticFloat32(dModel, 2)
	wQf := syntheticFloat32(qDim*dModel, 3)
	wKf := syntheticFloat32(kvDim*dModel, 4)
	wVf := syntheticFloat32(kvDim*dModel, 5)
	wOf := syntheticFloat32(dModel*qDim, 6)
	x, nw := toBF16Bytes(xf), toBF16Bytes(nwf)
	wQ, wK, wV, wO := toBF16Bytes(wQf), toBF16Bytes(wKf), toBF16Bytes(wVf), toBF16Bytes(wOf)
	// caches: rows 0..pos-1 pre-filled (synthetic), row pos written by the kernel.
	kCacheF := syntheticFloat32(maxLen*kvDim, 7)
	vCacheF := syntheticFloat32(maxLen*kvDim, 8)
	kCache, vCache := toBF16Bytes(kCacheF), toBF16Bytes(vCacheF)
	invFreqs := make([]float32, hd2)
	for d := 0; d < hd2; d++ {
		invFreqs[d] = float32(1.0 / math.Pow(float64(base), float64(2*d)/float64(headDim)))
	}

	// --- host reference: identical math + bf16 rounding points to the kernel ---
	rb := func(b []byte, i int) float32 { return bf16ToF32(b[i*2], b[i*2+1]) }
	matvec := func(w []byte, xv []float32, o, inDim int) float32 { // fp32 accum, bf16 weights, fp32 x
		acc := float32(0)
		for k := 0; k < inDim; k++ {
			acc += rb(w, o*inDim+k) * xv[k]
		}
		return acc
	}
	bf := func(v float32) float32 { b := f32ToBF16(v); return bf16ToF32(byte(b), byte(b>>8)) } // round to bf16
	// RMSNorm
	var ss float32
	for k := 0; k < dModel; k++ {
		ss += rb(x, k) * rb(x, k)
	}
	rms := float32(1.0 / math.Sqrt(float64(ss/float32(dModel)+eps)))
	normed := make([]float32, dModel)
	for i := 0; i < dModel; i++ {
		normed[i] = bf(rb(x, i) * rms * rb(nw, i))
	}
	// QKV + RoPE
	qr := make([]float32, qDim)
	kRow := make([]float32, kvDim)
	vRow := make([]float32, kvDim)
	rope := func(a0, a1 float32, d int) (float32, float32) {
		ang := float64(pos) * float64(invFreqs[d])
		c, s := float32(math.Cos(ang)), float32(math.Sin(ang))
		return a0*c - a1*s, a0*s + a1*c
	}
	for h := 0; h < nHeads; h++ {
		for d := 0; d < hd2; d++ {
			q0 := matvec(wQ, normed, h*headDim+d, dModel)
			q1 := matvec(wQ, normed, h*headDim+d+hd2, dModel)
			r0, r1 := rope(q0, q1, d)
			qr[h*headDim+d], qr[h*headDim+d+hd2] = bf(r0), bf(r1)
		}
	}
	for hk := 0; hk < nKVHeads; hk++ {
		for d := 0; d < hd2; d++ {
			k0 := matvec(wK, normed, hk*headDim+d, dModel)
			k1 := matvec(wK, normed, hk*headDim+d+hd2, dModel)
			r0, r1 := rope(k0, k1, d)
			kRow[hk*headDim+d], kRow[hk*headDim+d+hd2] = bf(r0), bf(r1)
			vRow[hk*headDim+d] = bf(matvec(wV, normed, hk*headDim+d, dModel))
			vRow[hk*headDim+d+hd2] = bf(matvec(wV, normed, hk*headDim+d+hd2, dModel))
		}
	}
	// write current row into the reference cache
	kc := make([]float32, maxLen*kvDim)
	vc := make([]float32, maxLen*kvDim)
	for i := range kc {
		kc[i] = rb(kCache, i)
		vc[i] = rb(vCache, i)
	}
	for i := 0; i < kvDim; i++ {
		kc[pos*kvDim+i] = kRow[i]
		vc[pos*kvDim+i] = vRow[i]
	}
	// SDPA
	attn := make([]float32, qDim)
	for h := 0; h < nHeads; h++ {
		kvh := (h / gqa) * headDim
		m := float32(-3e38)
		for j := 0; j < kvLen; j++ {
			var dot float32
			for d := 0; d < headDim; d++ {
				dot += qr[h*headDim+d] * kc[j*kvDim+kvh+d]
			}
			if dot*scale > m {
				m = dot * scale
			}
		}
		var denom float32
		acc := make([]float32, headDim)
		for j := 0; j < kvLen; j++ {
			var dot float32
			for d := 0; d < headDim; d++ {
				dot += qr[h*headDim+d] * kc[j*kvDim+kvh+d]
			}
			p := float32(math.Exp(float64(dot*scale - m)))
			denom += p
			for d := 0; d < headDim; d++ {
				acc[d] += p * vc[j*kvDim+kvh+d]
			}
		}
		for d := 0; d < headDim; d++ {
			attn[h*headDim+d] = bf(acc[d] / denom)
		}
	}
	// O + residual
	refOut := make([]byte, dModel*bf16Size)
	for i := 0; i < dModel; i++ {
		h := f32ToBF16(rb(x, i) + matvec(wO, attn, i, qDim))
		refOut[i*2], refOut[i*2+1] = byte(h), byte(h>>8)
	}

	// --- run the megakernel ---
	got := make([]byte, dModel*bf16Size)
	withAutoreleasePool(func() {
		kBuf := sharedBytes(append([]byte(nil), kCache...))
		vBuf := sharedBytes(append([]byte(nil), vCache...))
		normedB := device.NewBufferWithLengthOptions(uint(dModel*4), metal.MTLResourceStorageModeShared)
		qrB := device.NewBufferWithLengthOptions(uint(qDim*4), metal.MTLResourceStorageModeShared)
		attnB := device.NewBufferWithLengthOptions(uint(qDim*4), metal.MTLResourceStorageModeShared)
		outB := device.NewBufferWithLengthOptions(uint(dModel*bf16Size), metal.MTLResourceStorageModeShared)
		arrive := device.NewBufferWithLengthOptions(4, metal.MTLResourceStorageModeShared)
		*(*uint32)(arrive.Contents()) = 0
		invB := sharedBytes(unsafe.Slice((*byte)(unsafe.Pointer(&invFreqs[0])), len(invFreqs)*4))
		bufs := []metal.MTLBuffer{sharedBytes(x), sharedBytes(nw), sharedBytes(wQ), sharedBytes(wK), sharedBytes(wV), sharedBytes(wO), kBuf, vBuf, normedB, qrB, attnB, outB, arrive, invB}
		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		enc.SetComputePipelineState(pso)
		for i, b := range bufs {
			enc.SetBufferWithOffsetAtIndex(b, 0, uint(i))
		}
		setEncInt32(enc, dModel, 14)
		setEncInt32(enc, nHeads, 15)
		setEncInt32(enc, nKVHeads, 16)
		setEncInt32(enc, headDim, 17)
		setEncInt32(enc, pos, 18)
		setEncFloat32(enc, scale, 19)
		setEncFloat32(enc, eps, 20)
		setEncInt32(enc, numTG, 21)
		setEncInt32(enc, maxSpin, 22)
		enc.DispatchThreadgroupsThreadsPerThreadgroup(metal.MTLSize{Width: numTG, Height: 1, Depth: 1}, metal.MTLSize{Width: threadsPerTG, Height: 1, Depth: 1})
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(got, unsafe.Slice((*byte)(outB.Contents()), dModel*bf16Size))
	})

	cos := cosineBF16(got, refOut)
	if cos < 0.999 {
		t.Fatalf("attention megakernel cosine=%.6f vs host reference — staged structure / atomic handoff broken", cos)
	}
	t.Logf("attention megakernel (one dispatch, 4 stages, device-scope grid barriers, atomic handoffs): cosine=%.6f vs host reference — the attention half is structurally correct", cos)
}
