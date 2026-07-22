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

// TestVerifyStackICBReplay_RMSAddTwoLayerParity_Good is #71's minimal live
// repro, transplanted in-suite: on E4B the adds+rms2560 skip stream cut to TWO
// interior layers (12 fully-barriered rms/add commands) NaNs on its first
// replay, while ONE layer is clean and E2B replays five layers clean. The
// stream per layer is entry-rms, the attn-residual rms+add pair (scratch
// aliases the entry-rms output), mlp-rms, and the mlp-residual rms+add pair —
// layer 1 crosses rowsA→rowsB, layer 2 runs IN PLACE on rowsB, exactly the
// fold's ping-pong. Recorded and replayed against the live twin at both
// models' dModel.
func TestVerifyStackICBReplay_RMSAddTwoLayerParity_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("MLX_METALLIB_PATH not set")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("metal init: %v", err)
	}
	const rows = 5
	const eps = 1e-6
	cases := []struct {
		name      string
		dModel    int
		layers    int
		shardW    bool // weights as offset views into ONE big buffer (the mmap shard shape)
		shardBase uint // first weight offset — >2^32 probes the ICB offset width
	}{
		{"e4b-2560-2layer", 2560, 2, false, 0},
		{"e2b-1536-5layer", 1536, 5, false, 0},
		{"e4b-2560-2layer-shardw", 2560, 2, true, 0},
		{"e4b-2560-40layer-shardw", 2560, 40, true, 0},
		{"e4b-2560-2layer-shard4g", 2560, 2, true, 1<<32 + 1<<20},
	}
	rng := uint32(0xfeedbeef)
	next := func() uint32 { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng }

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			n := rows * tc.dModel
			fill := func(scale float32) metal.MTLBuffer {
				f := make([]float32, n)
				for i := range f {
					f[i] = (float32(next()%2000) - 1000) / 1000 * scale
				}
				b := toBF16Bytes(f)
				return device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&b[0]), uint(len(b)), metal.MTLResourceStorageModeShared)
			}
			type wView struct {
				buf metal.MTLBuffer
				off uint
			}
			var shard metal.MTLBuffer
			shardOff := tc.shardBase
			if tc.shardW {
				page := uint(os.Getpagesize())
				need := tc.shardBase + uint(tc.layers*4*tc.dModel*2)
				need = (need + page - 1) &^ (page - 1)
				shard = device.NewBufferWithLengthOptions(need, metal.MTLResourceStorageModeShared)
				if shard == nil {
					t.Skipf("shard allocation of %d bytes unavailable", need)
				}
			}
			weight := func() wView {
				f := make([]float32, tc.dModel)
				for i := range f {
					f[i] = -0.3 + float32(i%61)*0.01
				}
				b := toBF16Bytes(f)
				if tc.shardW {
					dst := unsafe.Slice((*byte)(shard.Contents()), int(shard.Length()))
					copy(dst[shardOff:], b)
					v := wView{shard, shardOff}
					shardOff += uint(len(b))
					return v
				}
				return wView{residentBytes(b), 0}
			}
			type layerW struct{ wE, wPA, wM, wPF wView }
			ws := make([]layerW, tc.layers)
			for i := range ws {
				ws[i] = layerW{weight(), weight(), weight(), weight()}
			}

			rmsPSO, err := pipelineForICB(rmsKernelBF16(tc.dModel))
			if err != nil {
				t.Fatalf("rms pso: %v", err)
			}
			addPSO, err := pipelineForICB("vv_Addbfloat16")
			if err != nil {
				t.Fatalf("add pso: %v", err)
			}
			tg := rmsThreadgroup(tc.dModel, rmsPSO)

			mkLane := func() (rowsA, rowsB, attnOut, down, attnNorm, hSlab, mlpNorm metal.MTLBuffer) {
				return fill(1), fill(1), fill(1), fill(1),
					device.NewBufferWithLengthOptions(uint(n*2), metal.MTLResourceStorageModeShared),
					device.NewBufferWithLengthOptions(uint(n*2), metal.MTLResourceStorageModeShared),
					device.NewBufferWithLengthOptions(uint(n*2), metal.MTLResourceStorageModeShared)
			}
			// seed both lanes identically: fill uses the shared rng, so build
			// lane A's buffers, snapshot bytes, and clone for lane B.
			la0, lb0, lao0, ld0, lan, lh, lm := mkLane()
			clone := func(src metal.MTLBuffer) metal.MTLBuffer {
				sb := unsafe.Slice((*byte)(src.Contents()), n*2)
				return device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&sb[0]), uint(n*2), metal.MTLResourceStorageModeShared)
			}
			ra0, rb0, rao0, rd0 := clone(la0), clone(lb0), clone(lao0), clone(ld0)
			ran := device.NewBufferWithLengthOptions(uint(n*2), metal.MTLResourceStorageModeShared)
			rh := device.NewBufferWithLengthOptions(uint(n*2), metal.MTLResourceStorageModeShared)
			rm := device.NewBufferWithLengthOptions(uint(n*2), metal.MTLResourceStorageModeShared)

			// one layer's six ops against a generic sink pair.
			liveLayer := func(enc metal.MTLComputeCommandEncoderObject, w layerW, readRows, writeRows, attnOut, down, attnNorm, hSlab, mlpNorm metal.MTLBuffer) {
				emitRMSNormRows(encObjectSink{enc}, rmsPSO, readRows, w.wE.buf, attnNorm, 0, w.wE.off, 0, tc.dModel, eps, rows, tg)
				emitRMSNormRows(encObjectSink{enc}, rmsPSO, attnOut, w.wPA.buf, attnNorm, 0, w.wPA.off, 0, tc.dModel, eps, rows, tg)
				emitBinary(encObjectSink{enc}, addPSO, readRows, 0, attnNorm, 0, hSlab, 0, rows*tc.dModel)
				emitRMSNormRows(encObjectSink{enc}, rmsPSO, hSlab, w.wM.buf, mlpNorm, 0, w.wM.off, 0, tc.dModel, eps, rows, tg)
				emitRMSNormRows(encObjectSink{enc}, rmsPSO, down, w.wPF.buf, mlpNorm, 0, w.wPF.off, 0, tc.dModel, eps, rows, tg)
				emitBinary(encObjectSink{enc}, addPSO, hSlab, 0, mlpNorm, 0, writeRows, 0, rows*tc.dModel)
			}
			withAutoreleasePool(func() {
				cb := commandBufferFast(queue)
				enc := computeCommandEncoderFast(cb)
				read := la0
				for li := 0; li < tc.layers; li++ {
					liveLayer(enc, ws[li], read, lb0, lao0, ld0, lan, lh, lm)
					read = lb0 // in-place from layer 2 on
				}
				endEncodingFast(enc)
				commitCommandBufferFast(cb)
				waitUntilCompletedFast(cb)
			})

			// recorded lane: the recorder's own mirrors, whole-stream barriers.
			rec := newVerifyStackRecorder(tc.layers+2, rows, verifyStackKey{}, nil)
			if rec == nil {
				t.Fatal("newVerifyStackRecorder returned nil")
			}
			read := ra0
			for li := 0; li < tc.layers; li++ {
				rec.setLayer(li + 1)
				rec.layerEntry()
				rec.recRMSRows(read, ws[li].wE.buf, ran, 0, ws[li].wE.off, 0, rows, tc.dModel, eps)
				rec.recRMSRows(rao0, ws[li].wPA.buf, ran, 0, ws[li].wPA.off, 0, rows, tc.dModel, eps)
				rec.recAdd(read, 0, ran, 0, rh, 0, rows*tc.dModel)
				rec.recRMSRows(rh, ws[li].wM.buf, rm, 0, ws[li].wM.off, 0, rows, tc.dModel, eps)
				rec.recRMSRows(rd0, ws[li].wPF.buf, rm, 0, ws[li].wPF.off, 0, rows, tc.dModel, eps)
				rec.recAdd(rh, 0, rm, 0, rb0, 0, rows*tc.dModel)
				read = rb0
			}
			// >=2^32 binds REBASE onto no-copy windows over the same memory
			// (the #71 re-enable): recording succeeds and replays byte-exact.
			if rec.failed {
				t.Fatal("recorder failed")
			}
			vs := rec.finish()
			if vs == nil {
				t.Fatal("finish returned nil")
			}
			withAutoreleasePool(func() {
				cb := commandBufferFast(queue)
				enc := concurrentComputeEncoderFast(cb)
				vs.executeInto(enc, 0, nil)
				endEncodingFast(enc)
				commitCommandBufferFast(cb)
				waitUntilCompletedFast(cb)
			})

			for lane, b := range map[string]metal.MTLBuffer{"live-rowsB": lb0, "replay-rowsB": rb0, "replay-h": rh, "replay-mlpNorm": rm, "replay-attnNorm": ran} {
				nan, inf, _, _, first := bf16BufStats(b, 0, n)
				if nan > 0 || inf > 0 {
					t.Errorf("%s: NaN=%d Inf=%d first=%d", lane, nan, inf, first)
				}
			}
			live := unsafe.Slice((*byte)(lb0.Contents()), n*2)
			icb := unsafe.Slice((*byte)(rb0.Contents()), n*2)
			var dot, ng, nw float64
			for i := 0; i < n; i++ {
				o := i * 2
				g := float64(bf16ToF32(icb[o], icb[o+1]))
				w := float64(bf16ToF32(live[o], live[o+1]))
				dot += g * w
				ng += g * g
				nw += w * w
			}
			cos := dot / (math.Sqrt(ng)*math.Sqrt(nw) + 1e-30)
			t.Logf("%s live↔replay rows cos=%.6f", tc.name, cos)
			if cos < 0.9999 {
				t.Errorf("%s replay rows diverge from live: cos %.6f", tc.name, cos)
			}
		})
	}
}

// TestVsRebaseHighBind_Ugly pins the rebase edges directly: offsets under
// 2^31 within the window rebase to (window, off-base); a bind starting inside
// the page-floored tail fragment declines; base past the buffer declines.
func TestVsRebaseHighBind_Ugly(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("MLX_METALLIB_PATH not set")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("metal init: %v", err)
	}
	page := uint(os.Getpagesize())
	ln := uint(1)<<32 + 3*page + 123 // odd tail: floored window excludes the final 123 bytes
	buf := device.NewBufferWithLengthOptions(ln, metal.MTLResourceStorageModeShared)
	if buf == nil {
		t.Skip("4GiB allocation unavailable")
	}
	if _, _, ok := vsRebaseHighBind(buf, uint(1)<<33); ok {
		t.Error("base beyond the buffer must decline")
	}
	w, off, ok := vsRebaseHighBind(buf, uint(1)<<32+page)
	if !ok || w == nil || off != page {
		t.Errorf("in-window bind must rebase to the segment window: ok=%v off=%d", ok, off)
	}
	if w2, off2, ok2 := vsRebaseHighBind(buf, uint(1)<<32+2*page); !ok2 || bufID(w2) != bufID(w) || off2 != 2*page {
		t.Errorf("same-segment bind must reuse the memoised window: ok=%v same=%v off=%d", ok2, ok2 && bufID(w2) == bufID(w), off2)
	}
	if _, _, ok := vsRebaseHighBind(buf, uint(1)<<32+3*page+16); ok {
		t.Error("a bind starting inside the floored tail fragment must decline")
	}
}
