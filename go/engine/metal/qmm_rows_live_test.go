// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"testing"
	"unsafe"

	"dappco.re/go/inference/internal/enginegate"
)

// TestRealQMMTProjectRowsMatchesQMV pins the quant batched projection (MLX qmm_t — ONE weight
// pass for all rows, the prompt-prefill fold) against the per-row qmv decode kernel on REAL
// e2b-4bit layer weights, with an fp64 host dequant-dot as a loose sanity ceiling.
//
// The INVARIANT is qmm_t == qmv: the batched fold and the per-row kernel compute the same
// projection, so a layout/offset bug in projectRows makes them diverge (and lands orders
// beyond bf16 rounding). Measured, they agree to 0.00000 across 512 sampled dots — prefill and
// decode fold identically. Both accumulate in GPU precision, so BOTH sit up to ~0.065 from an
// fp64 host dot on the cancellation-heavy 2560-term rows: that gap is GPU-accum-vs-fp64, NOT a
// kernel error, so the oracle is a sanity ceiling only. (An earlier 2-bf16-step-vs-fp64 bound
// over-constrained it — no GPU matvec matches fp64 that tightly on cancellation-heavy dots; the
// qmm==qmv cross-check is the real correctness signal, and it is exact.)
func TestRealQMMTProjectRowsMatchesQMV(t *testing.T) {
	requireNativeRuntime(t)
	targetDir := enginegate.HFModelPath(t, "mlx-community/gemma-4-e2b-it-4bit")
	target, err := LoadDir(targetDir, 4096)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	defer func() { _ = target.Close() }()
	proj, ok := target.state.lb[0].proj.(qmvProjector)
	if !ok {
		t.Fatalf("layer 0 projector is %T, want qmvProjector", target.state.lb[0].proj)
	}
	const rows = 4
	dModel := target.arch.Hidden
	rowBytes := dModel * bf16Size
	inBuf := device.NewBufferWithLengthOptions(uint(rows*rowBytes), 0)
	in := unsafe.Slice((*byte)(inBuf.Contents()), rows*rowBytes)
	ids := []int32{506, 8134, 529, 236776}
	for i, id := range ids {
		emb, eerr := target.embed(id)
		if eerr != nil {
			t.Fatalf("embed: %v", eerr)
		}
		copy(in[i*rowBytes:(i+1)*rowBytes], emb)
	}
	_, qDim, _, _ := proj.weightDims(projQ)
	outRowBytes := qDim * bf16Size
	qmmOut := device.NewBufferWithLengthOptions(uint(rows*outRowBytes), 0)
	qmvOut := device.NewBufferWithLengthOptions(uint(rows*outRowBytes), 0)

	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	handled, perr := proj.projectRows(enc, inBuf, qmmOut, 0, 0, rows, projQ)
	if perr != nil || !handled {
		endEncodingFast(enc)
		t.Fatalf("projectRows: handled=%v err=%v", handled, perr)
	}
	for i := range rows {
		if perr = encQMVBF16At(enc, proj.q.wq.buf, proj.q.scales.buf, proj.q.biases.buf, inBuf, qmvOut,
			proj.q.wq.off, proj.q.scales.off, proj.q.biases.off, uint(i*rowBytes), uint(i*outRowBytes),
			qDim, dModel, proj.groupSize, proj.bits); perr != nil {
			endEncodingFast(enc)
			t.Fatalf("qmv row %d: %v", i, perr)
		}
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)

	// host fp32 dequant-dot oracle (affine 4-bit): w packed row-major [N, K/8 uint32], nibbles
	// LE within each byte; scales/biases [N, K/gs] bf16; w[n,k] = scale·q + bias.
	gs := proj.groupSize
	wq := unsafe.Slice((*byte)(proj.q.wq.buf.Contents()), int(proj.q.wq.off)+qDim*dModel/2)[proj.q.wq.off:]
	sc := unsafe.Slice((*byte)(proj.q.scales.buf.Contents()), int(proj.q.scales.off)+qDim*(dModel/gs)*bf16Size)[proj.q.scales.off:]
	bi := unsafe.Slice((*byte)(proj.q.biases.buf.Contents()), int(proj.q.biases.off)+qDim*(dModel/gs)*bf16Size)[proj.q.biases.off:]
	oracle := func(r, n int) float64 {
		var acc float64
		for k := range dModel {
			nib := wq[n*(dModel/2)+k/2]
			var q byte
			if k%2 == 0 {
				q = nib & 0xF
			} else {
				q = nib >> 4
			}
			g := n*(dModel/gs) + k/gs
			scale := float64(bf16ToF32(sc[g*2], sc[g*2+1]))
			bias := float64(bf16ToF32(bi[g*2], bi[g*2+1]))
			x := float64(bf16ToF32(in[r*rowBytes+k*2], in[r*rowBytes+k*2+1]))
			acc += x * (scale*float64(q) + bias)
		}
		return acc
	}
	a := unsafe.Slice((*byte)(qmmOut.Contents()), rows*outRowBytes)
	b := unsafe.Slice((*byte)(qmvOut.Contents()), rows*outRowBytes)
	rel := func(v, ref float64) float64 {
		d := v - ref
		if d < 0 {
			d = -d
		}
		s := ref
		if s < 0 {
			s = -s
		}
		if s < 1 {
			s = 1
		}
		return d / s
	}
	// every 16th output element per row: 512 oracle dots — dense enough to catch any
	// layout/offset bug (those corrupt whole stripes), cheap enough to run per-push.
	var worstQMM, worstQMV, worstPair float64
	var overCount int
	for r := range rows {
		for n := 0; n < qDim; n += 16 {
			ref := oracle(r, n)
			e := r*outRowBytes + n*bf16Size
			qmm := float64(bf16ToF32(a[e], a[e+1]))
			qmv := float64(bf16ToF32(b[e], b[e+1]))
			if v := rel(qmm, ref); v > worstQMM {
				worstQMM = v
			}
			if v := rel(qmv, ref); v > worstQMV {
				worstQMV = v
			}
			if v := rel(qmm, qmv); v > worstPair { // the real correctness invariant
				worstPair = v
			}
			if rel(qmm, ref) > 0.016 {
				overCount++
			}
		}
	}
	// INVARIANT: batched qmm_t must equal per-row qmv (same projection) — a projectRows
	// layout/offset bug makes them diverge and lands orders beyond bf16 rounding.
	if worstPair > 0.016 {
		t.Errorf("qmm_t vs qmv worst rel %.5f (>2 bf16 steps) — projectRows layout/offset bug", worstPair)
	}
	// SANITY only: both GPU kernels sit ~0.065 from the fp64 dot on cancellation-heavy dots
	// (bf16 accumulation, not error); a gross layout error would land near 1.0, not 0.1.
	if worstQMM > 0.15 {
		t.Errorf("qmm_t vs fp64 oracle worst %.5f (>0.15 sanity ceiling) — gross error", worstQMM)
	}
	t.Logf("over %d dots: worst |qmm-qmv| %.5f (invariant, exact=ok) · qmm-vs-fp64 %.5f · qmv-vs-fp64 %.5f (GPU-accum gap, sanity only) · elems >0.016-vs-fp64: %d",
		rows*qDim/16, worstPair, worstQMM, worstQMV, overCount)
}
