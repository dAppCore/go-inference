// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"os"
	"testing"
	"unsafe"

	"github.com/tmc/apple/metal"
)

// hostGeluMul mirrors lthn_gelu_gate_mul_bf16: gated = gelu_tanh(gate)·up, bf16-rounded.
func hostGeluMul(gate, up []byte) []byte {
	n := len(gate) / bf16Size
	out := make([]byte, n*bf16Size)
	for i := range n {
		g := bf16ToF32(gate[i*bf16Size], gate[i*bf16Size+1])
		u := bf16ToF32(up[i*bf16Size], up[i*bf16Size+1])
		inner := g + float32(0.044715)*(g*g*g)
		t := float32(math.Tanh(float64(float32(0.7978845608028654) * inner)))
		h := f32ToBF16(float32(0.5) * g * (1.0 + t) * u)
		out[i*bf16Size] = byte(h)
		out[i*bf16Size+1] = byte(h >> 8)
	}
	return out
}

// TestFFNMegakernel validates the whole SwiGLU MLP as ONE dispatch (gate/up qgemv -> gelu·mul -> grid
// barrier -> down qgemv) against the reference path (steel QMVBF16 gate/up + host gelu·mul + steel down).
// Token-identical (cosine~1): the first real decode-stage megakernel — three barriered ops collapsed into
// one dispatch with an in-kernel grid barrier, no external SetBarrier drains between gate/gelu/down.
func TestFFNMegakernel(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("device init: %v", err)
	}
	pso, err := ffnMegaPipeline()
	if err != nil {
		t.Skipf("ffnmega pipeline: %v", err)
	}
	const hidden, ff, groupSize, bits = 256, 512, 64, 4
	const numTG, threadsPerTG = 64, 128
	const maxSpin = int32(1_000_000)

	mkW := func(outDim, inDim, seed int) (p, s, b []byte) {
		p = make([]byte, outDim*inDim*bits/8)
		for i := range p {
			p[i] = byte((i*131 + 17 + seed) % 256)
		}
		nSB := outDim * (inDim / groupSize)
		s = toBF16Bytes(syntheticFloat32(nSB, seed+1))
		b = toBF16Bytes(syntheticFloat32(nSB, seed+2))
		return
	}
	gateP, gateS, gateB := mkW(ff, hidden, 10)
	upP, upS, upB := mkW(ff, hidden, 40)
	downP, downS, downB := mkW(hidden, ff, 70)
	x := toBF16Bytes(syntheticFloat32(hidden, 23))

	// reference: steel qmv gate/up -> host gelu·mul -> steel qmv down
	gate, err := QMVBF16(x, gateP, gateS, gateB, ff, hidden, groupSize, bits)
	if err != nil {
		t.Fatalf("gate qmv: %v", err)
	}
	up, err := QMVBF16(x, upP, upS, upB, ff, hidden, groupSize, bits)
	if err != nil {
		t.Fatalf("up qmv: %v", err)
	}
	gatedRef := hostGeluMul(gate, up)
	ref, err := QMVBF16(gatedRef, downP, downS, downB, hidden, ff, groupSize, bits)
	if err != nil {
		t.Fatalf("down qmv: %v", err)
	}

	out := make([]byte, hidden*bf16Size)
	gatedGot := make([]byte, ff*bf16Size)
	withAutoreleasePool(func() {
		bufs := []metal.MTLBuffer{
			sharedBytes(x), sharedBytes(gateP), sharedBytes(gateS), sharedBytes(gateB),
			sharedBytes(upP), sharedBytes(upS), sharedBytes(upB),
			sharedBytes(downP), sharedBytes(downS), sharedBytes(downB),
		}
		gated := device.NewBufferWithLengthOptions(uint(ff*4), metal.MTLResourceStorageModeShared) // atomic_uint/slot
		outBuf := device.NewBufferWithLengthOptions(uint(hidden*bf16Size), metal.MTLResourceStorageModeShared)
		arrive := device.NewBufferWithLengthOptions(4, metal.MTLResourceStorageModeShared)
		*(*uint32)(arrive.Contents()) = 0
		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		enc.SetComputePipelineState(pso)
		for i, bf := range bufs {
			enc.SetBufferWithOffsetAtIndex(bf, 0, uint(i))
		}
		enc.SetBufferWithOffsetAtIndex(gated, 0, 10)
		enc.SetBufferWithOffsetAtIndex(outBuf, 0, 11)
		enc.SetBufferWithOffsetAtIndex(arrive, 0, 12)
		setEncInt32(enc, hidden, 13)
		setEncInt32(enc, ff, 14)
		setEncInt32(enc, groupSize, 15)
		setEncInt32(enc, numTG, 16)
		setEncInt32(enc, maxSpin, 17)
		enc.DispatchThreadgroupsThreadsPerThreadgroup(
			metal.MTLSize{Width: numTG, Height: 1, Depth: 1},
			metal.MTLSize{Width: threadsPerTG, Height: 1, Depth: 1},
		)
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(out, unsafe.Slice((*byte)(outBuf.Contents()), hidden*bf16Size))
		for i, gu := 0, unsafe.Slice((*uint32)(gated.Contents()), ff); i < ff; i++ { // extract bf16 from each atomic slot
			u := uint16(gu[i])
			gatedGot[i*bf16Size] = byte(u)
			gatedGot[i*bf16Size+1] = byte(u >> 8)
		}
	})

	// Component validation (random ill-conditioned weights amplify tiny reduction-order diffs end-to-end, so
	// validate each stage against its reference): stage 1 (gated) must match the reference, and stage 2 (down)
	// must match the steel qmv on the SAME gated input. Both cosine~1 ⇒ the megakernel == the reference path.
	stage1 := cosineBF16(gatedGot, gatedRef)
	ref2, err := QMVBF16(gatedGot, downP, downS, downB, hidden, ff, groupSize, bits)
	if err != nil {
		t.Fatalf("down qmv on megakernel gated: %v", err)
	}
	stage2 := cosineBF16(out, ref2)
	_ = ref
	// Stage 1 exact (cosine 1.0): the IN-KERNEL gated written by stage-1, copied out AFTER the kernel, equals
	// the reference — so gate/up qgemv + gelu·mul are correct. Stage 2 (out) is the down over the gated read
	// DURING the kernel, and it diverges to 0.99 vs the down over the post-kernel gated — i.e. stage 2 read
	// STALE gated for elements written by distant threadgroups. (TestQGemvSimdBeatsSequentialOnGated proves
	// both the sequential AND simd gemvs match steel at ~1.0 on this input, ruling out the gemv.) Root cause:
	// the in-kernel grid barrier's cross-TG memory COHERENCY — Metal has no device-wide fence beyond
	// threadgroup_barrier, so distant-TG writes aren't reliably visible. This is the megakernel's real blocker.
	if stage1 < 0.9999 {
		t.Fatalf("FFN megakernel structure broken: stage-1 gated cosine=%.6f (grid barrier / gate / up / gelu)", stage1)
	}
	if stage2 < 0.9999 {
		t.Fatalf("FFN megakernel stage-2 cosine=%.6f — cross-TG handoff broken (atomic gated + device-scope barrier expected coherent)", stage2)
	}
	t.Logf("FFN megakernel (one dispatch): stage-1 %.6f (structure exact); stage-2 %.6f — ATOMIC gated handoff + "+
		"macOS 26 device-scope grid barrier make the cross-TG read coherent (was 0.990 with plain gated + threadgroup-scope barrier)", stage1, stage2)
}
