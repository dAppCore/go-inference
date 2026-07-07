// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"testing"
	"unsafe"

	"dappco.re/go/inference/internal/enginegate"
)

// TestRealQMMTProjectRowsMatchesOracle pins the quant batched projection (MLX qmm_t — ONE
// weight pass for all rows, the prompt-prefill fold) against a HOST fp32 dequant-dot oracle
// on REAL e2b-4bit layer weights. The oracle — not the per-row qmv kernel — is the reference:
// the two GPU kernels accumulate in different orders and qmv sits FURTHER from fp32 truth on
// the spread elements (measured: worst element fp32 0.95688, qmm 0.95703, qmv 1.01562), so
// qmm-vs-qmv comparison mis-flags qmm for qmv's own rounding. Every qmm element must land
// within two bf16 mantissa steps of the fp32 dot; qmv's distance is logged for the record.
func TestRealQMMTProjectRowsMatchesOracle(t *testing.T) {
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
	var worstQMM, worstQMV float64
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
			// two bf16 mantissa steps ≈ 2/128; a layout bug lands orders beyond this
			if rel(qmm, ref) > 0.016 {
				t.Fatalf("qmm_t row %d out %d = %.5f, fp32 oracle %.5f (rel %.4f > 2 bf16 steps)", r, n, qmm, ref, rel(qmm, ref))
			}
		}
	}
	t.Logf("vs fp32 oracle over %d sampled dots: qmm worst rel %.5f · qmv worst rel %.5f", rows*qDim/16, worstQMM, worstQMV)
}
