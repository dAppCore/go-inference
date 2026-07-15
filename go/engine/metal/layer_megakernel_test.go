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
	layerMegaPSOOnce sync.Once
	layerMegaPSO     metal.MTLComputePipelineState
	layerMegaErr     error
)

func layerMegaPipeline() (metal.MTLComputePipelineState, error) {
	layerMegaPSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			layerMegaErr = core.NewError("layermega: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_layer_megakernel")
		if fn == nil || fn.GetID() == 0 {
			layerMegaErr = core.NewError("layermega: kernel not found")
			return
		}
		layerMegaPSO, layerMegaErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return layerMegaPSO, layerMegaErr
}

type layerDims struct {
	dModel, nHeads, nKVHeads, headDim, dFF, pos, numTG, maxSpin uint32
	scale, eps                                                  float32
}

// TestLayerMegakernel validates a WHOLE gemma decode layer in ONE dispatch — attention half (RMSNorm → QKV
// → RoPE → cache → SDPA → O → residual) chained into the FFN half (RMSNorm → gate/up → gelu·mul → down →
// residual), six stages, device-scope grid barriers, h handed off attention→FFN through atomics — against a
// host reference computing the identical math. This is the full-layer megakernel: ~15 barriered ICB ops
// collapsed into one persistent dispatch. The FFN half is byte-aligned with lthn_ffn_megakernel's gelu.
func TestLayerMegakernel(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("device init: %v", err)
	}
	pso, err := layerMegaPipeline()
	if err != nil {
		t.Skipf("layermega pipeline: %v", err)
	}
	const dModel, nHeads, nKVHeads, headDim, dFF, maxLen, pos = 128, 2, 1, 64, 256, 8, 3
	const numTG, threadsPerTG = 8, 64
	const maxSpin = int32(1_000_000)
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	qDim, kvDim, kvLen, hd2 := nHeads*headDim, nKVHeads*headDim, pos+1, headDim/2
	gqa := nHeads / nKVHeads

	x := toBF16Bytes(syntheticFloat32(dModel, 1))
	nw := toBF16Bytes(syntheticFloat32(dModel, 2))
	mnw := toBF16Bytes(syntheticFloat32(dModel, 9))
	wQ := toBF16Bytes(syntheticFloat32(qDim*dModel, 3))
	wK := toBF16Bytes(syntheticFloat32(kvDim*dModel, 4))
	wV := toBF16Bytes(syntheticFloat32(kvDim*dModel, 5))
	wO := toBF16Bytes(syntheticFloat32(dModel*qDim, 6))
	wG := toBF16Bytes(syntheticFloat32(dFF*dModel, 10))
	wU := toBF16Bytes(syntheticFloat32(dFF*dModel, 11))
	wD := toBF16Bytes(syntheticFloat32(dModel*dFF, 12))
	kCache := toBF16Bytes(syntheticFloat32(maxLen*kvDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(maxLen*kvDim, 8))
	invFreqs := make([]float32, hd2)
	for d := range hd2 {
		invFreqs[d] = float32(1.0 / math.Pow(float64(base), float64(2*d)/float64(headDim)))
	}

	// --- host reference (identical math + bf16 rounding points) ---
	rb := func(b []byte, i int) float32 { return bf16ToF32(b[i*2], b[i*2+1]) }
	bf := func(v float32) float32 { h := f32ToBF16(v); return bf16ToF32(byte(h), byte(h>>8)) }
	mv := func(w []byte, xv []float32, o, inDim int) float32 {
		acc := float32(0)
		for k := range inDim {
			acc += rb(w, o*inDim+k) * xv[k]
		}
		return acc
	}
	// attention: normed → QKV+RoPE → cache → SDPA → h = x + Wo·attn
	var ss float32
	for k := range dModel {
		ss += rb(x, k) * rb(x, k)
	}
	rms := float32(1.0 / math.Sqrt(float64(ss/float32(dModel)+eps)))
	normed := make([]float32, dModel)
	for i := range dModel {
		normed[i] = bf(rb(x, i) * rms * rb(nw, i))
	}
	rope := func(a0, a1 float32, d int) (float32, float32) {
		ang := float64(pos) * float64(invFreqs[d])
		c, s := float32(math.Cos(ang)), float32(math.Sin(ang))
		return a0*c - a1*s, a0*s + a1*c
	}
	qr := make([]float32, qDim)
	for hh := range nHeads {
		for d := range hd2 {
			r0, r1 := rope(mv(wQ, normed, hh*headDim+d, dModel), mv(wQ, normed, hh*headDim+d+hd2, dModel), d)
			qr[hh*headDim+d], qr[hh*headDim+d+hd2] = bf(r0), bf(r1)
		}
	}
	kc := make([]float32, maxLen*kvDim)
	vc := make([]float32, maxLen*kvDim)
	for i := range kc {
		kc[i], vc[i] = rb(kCache, i), rb(vCache, i)
	}
	for hk := range nKVHeads {
		for d := range hd2 {
			r0, r1 := rope(mv(wK, normed, hk*headDim+d, dModel), mv(wK, normed, hk*headDim+d+hd2, dModel), d)
			kc[pos*kvDim+hk*headDim+d], kc[pos*kvDim+hk*headDim+d+hd2] = bf(r0), bf(r1)
			vc[pos*kvDim+hk*headDim+d] = bf(mv(wV, normed, hk*headDim+d, dModel))
			vc[pos*kvDim+hk*headDim+d+hd2] = bf(mv(wV, normed, hk*headDim+d+hd2, dModel))
		}
	}
	attn := make([]float32, qDim)
	for hh := range nHeads {
		kvh := (hh / gqa) * headDim
		m := float32(-3e38)
		for j := range kvLen {
			var dot float32
			for d := range headDim {
				dot += qr[hh*headDim+d] * kc[j*kvDim+kvh+d]
			}
			if dot*scale > m {
				m = dot * scale
			}
		}
		var denom float32
		acc := make([]float32, headDim)
		for j := range kvLen {
			var dot float32
			for d := range headDim {
				dot += qr[hh*headDim+d] * kc[j*kvDim+kvh+d]
			}
			p := float32(math.Exp(float64(dot*scale - m)))
			denom += p
			for d := range headDim {
				acc[d] += p * vc[j*kvDim+kvh+d]
			}
		}
		for d := range headDim {
			attn[hh*headDim+d] = bf(acc[d] / denom)
		}
	}
	h := make([]float32, dModel)
	for i := range dModel {
		h[i] = bf(rb(x, i) + mv(wO, attn, i, qDim))
	}
	// FFN: mlpNormed = RMSNorm(h) → gate/up → gelu·mul → down → out = h + down
	var ssh float32
	for k := range dModel {
		ssh += h[k] * h[k]
	}
	rmsh := float32(1.0 / math.Sqrt(float64(ssh/float32(dModel)+eps)))
	mlpNormed := make([]float32, dModel)
	for i := range dModel {
		mlpNormed[i] = bf(h[i] * rmsh * rb(mnw, i))
	}
	gated := make([]float32, dFF)
	for i := range dFF {
		g := bf(mv(wG, mlpNormed, i, dModel))
		u := bf(mv(wU, mlpNormed, i, dModel))
		inner := g + 0.044715*(g*g*g)
		tnh := float32(math.Tanh(float64(0.7978845608028654 * inner)))
		gated[i] = bf(0.5 * g * (1.0 + tnh) * u)
	}
	refOut := make([]byte, dModel*bf16Size)
	for i := range dModel {
		o := f32ToBF16(h[i] + mv(wD, gated, i, dFF))
		refOut[i*2], refOut[i*2+1] = byte(o), byte(o>>8)
	}

	// --- run the full-layer megakernel ---
	got := make([]byte, dModel*bf16Size)
	withAutoreleasePool(func() {
		mk := func(n int) metal.MTLBuffer {
			return device.NewBufferWithLengthOptions(uint(n*4), metal.MTLResourceStorageModeShared)
		}
		outB := device.NewBufferWithLengthOptions(uint(dModel*bf16Size), metal.MTLResourceStorageModeShared)
		arrive := device.NewBufferWithLengthOptions(4, metal.MTLResourceStorageModeShared)
		*(*uint32)(arrive.Contents()) = 0
		dims := layerDims{dModel, nHeads, nKVHeads, headDim, dFF, pos, numTG, uint32(maxSpin), scale, eps}
		dimsB := sharedBytes(unsafe.Slice((*byte)(unsafe.Pointer(&dims)), int(unsafe.Sizeof(dims))))
		invB := sharedBytes(unsafe.Slice((*byte)(unsafe.Pointer(&invFreqs[0])), len(invFreqs)*4))
		bufs := []metal.MTLBuffer{
			sharedBytes(x), sharedBytes(nw), sharedBytes(wQ), sharedBytes(wK), sharedBytes(wV), sharedBytes(wO),
			sharedBytes(append([]byte(nil), kCache...)), sharedBytes(append([]byte(nil), vCache...)),
			sharedBytes(mnw), sharedBytes(wG), sharedBytes(wU), sharedBytes(wD),
			mk(dModel), mk(qDim), mk(qDim), mk(dModel), mk(dModel), mk(dFF), outB, arrive, invB, dimsB,
		}
		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		enc.SetComputePipelineState(pso)
		for i, b := range bufs {
			enc.SetBufferWithOffsetAtIndex(b, 0, uint(i))
		}
		enc.DispatchThreadgroupsThreadsPerThreadgroup(metal.MTLSize{Width: numTG, Height: 1, Depth: 1}, metal.MTLSize{Width: threadsPerTG, Height: 1, Depth: 1})
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(got, unsafe.Slice((*byte)(outB.Contents()), dModel*bf16Size))
	})

	cos := cosineBF16(got, refOut)
	if cos < 0.999 {
		t.Fatalf("full-layer megakernel cosine=%.6f vs host reference — chained attention+FFN structure broken", cos)
	}
	t.Logf("full-layer megakernel (ONE dispatch: attention 4 stages + FFN 2 stages, 6 device-scope grid barriers, "+
		"atomic handoffs incl. attn→FFN h): cosine=%.6f vs host reference — the whole decode layer in one kernel", cos)
}
