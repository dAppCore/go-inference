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

// TestMoEGroupedSortPipeline pins the bucket-sort kernel's dispatch legality (#348's class:
// an over-max threadgroup is DROPPED silently) and its output: a permutation of the pairs,
// non-decreasing expert order, expert ids matching the routing.
func TestMoEGroupedSortPipeline(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("MLX_METALLIB_PATH not set")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("metal init: %v", err)
	}
	pso, err := moePairSortPipeline()
	if err != nil {
		t.Skipf("sort kernel unavailable: %v", err)
	}
	if maxT := pso.MaxTotalThreadsPerThreadgroup(); maxT < moePairSortThreads {
		t.Fatalf("sort threadgroup %d exceeds pipeline max %d — the dispatch would be dropped silently", moePairSortThreads, maxT)
	}
	const pairs, experts = 777, 128
	idx := make([]int32, pairs)
	rng := uint32(7)
	for i := range idx {
		rng ^= rng << 13
		rng ^= rng >> 17
		rng ^= rng << 5
		idx[i] = int32(rng % experts)
	}
	idxBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&idx[0]), uint(pairs*4), metal.MTLResourceStorageModeShared)
	spBuf := device.NewBufferWithLengthOptions(uint(pairs*4), metal.MTLResourceStorageModeShared)
	seBuf := device.NewBufferWithLengthOptions(uint(pairs*4), metal.MTLResourceStorageModeShared)
	withAutoreleasePool(func() {
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		sink := encSink{enc}
		sink.setPSO(pso)
		sink.setBuf(idxBuf, 0, 0)
		sink.setBuf(spBuf, 0, 1)
		sink.setBuf(seBuf, 0, 2)
		sink.setI32(int32(pairs), 3)
		sink.setI32(int32(experts), 4)
		sink.dispatchThreadgroups(
			metal.MTLSize{Width: 1, Height: 1, Depth: 1},
			metal.MTLSize{Width: moePairSortThreads, Height: 1, Depth: 1},
		)
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
	})
	sp := unsafe.Slice((*int32)(unsafe.Pointer(spBuf.Contents())), pairs)
	se := unsafe.Slice((*uint32)(unsafe.Pointer(seBuf.Contents())), pairs)
	seen := make([]bool, pairs)
	prev := uint32(0)
	for i := range pairs {
		p := sp[i]
		if p < 0 || int(p) >= pairs || seen[p] {
			t.Fatalf("sortedPair[%d]=%d is not a permutation", i, p)
		}
		seen[p] = true
		if se[i] != uint32(idx[p]) {
			t.Fatalf("sortedExpert[%d]=%d but pair %d routes to %d", i, se[i], p, idx[p])
		}
		if se[i] < prev {
			t.Fatalf("sortedExpert not non-decreasing at %d: %d after %d", i, se[i], prev)
		}
		prev = se[i]
	}
}

// TestMoEGroupedExpertsParity drives the grouped-GEMM expert lane and the all-routes gather
// lane over the SAME synthetic experts (per-group VARYING scales — uniform scales are blind
// to group/expert indexing defects) and the same routing, comparing the pair-ordered down
// outputs. Token-identity tier: cosine per pair row against the all-routes reference.
func TestMoEGroupedExpertsParity(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("MLX_METALLIB_PATH not set")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("metal init: %v", err)
	}
	const (
		K          = 96
		topK       = 8
		numExperts = 32
		dModel     = 256
		expertDFF  = 128
		gs, bits   = 64, 4
	)
	pairs := K * topK
	rng := uint32(0x1234567)
	next := func() uint32 { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng }
	quant := func(outDim, inDim int) (packed, scales, biases []byte) {
		groups := inDim / gs
		packed = make([]byte, outDim*inDim/2)
		for i := range packed {
			packed[i] = byte(next())
		}
		sf := make([]float32, outDim*groups)
		bf := make([]float32, outDim*groups)
		for i := range sf {
			sf[i] = 0.005 + float32(i%89)*0.0007
			bf[i] = -0.3 + float32(i%97)*0.006
		}
		return packed, toBF16Bytes(sf), toBF16Bytes(bf)
	}
	// fused gate_up [E, 2·dff, dModel] + down [E, dModel, dff]
	guPacked, guScales, guBiases := quant(numExperts*2*expertDFF, dModel)
	dnPacked, dnScales, dnBiases := quant(numExperts*dModel, expertDFF)

	xf := make([]float32, K*dModel)
	for i := range xf {
		xf[i] = (float32(next()%2000) - 1000) / 1000
	}
	idx := make([]int32, pairs)
	for i := range idx {
		idx[i] = int32(next() % numExperts)
	}

	s := &archDecodeState{dModel: dModel, dFF: expertDFF, eps: 1e-6}
	s.denseBatch.moeBatch = &moeBatchScratch{}
	mb := s.denseBatch.moeBatch
	if err := mb.ensure(K, dModel, expertDFF, expertDFF, topK, numExperts); err != nil {
		t.Fatalf("moeBatch ensure: %v", err)
	}
	copy(unsafe.Slice((*byte)(mb.expertIn.Contents()), K*dModel*2), toBF16Bytes(xf))
	copy(unsafe.Slice((*byte)(mb.idx.Contents()), pairs*4), unsafe.Slice((*byte)(unsafe.Pointer(&idx[0])), pairs*4))

	guP, guS, guB := bufView{buf: residentBytes(guPacked)}, bufView{buf: residentBytes(guScales)}, bufView{buf: residentBytes(guBiases)}
	dnP, dnS, dnB := bufView{buf: residentBytes(dnPacked)}, bufView{buf: residentBytes(dnScales)}, bufView{buf: residentBytes(dnBiases)}

	runAllRoutes := func() []float32 {
		gatherInPSO, err := gatherQMVBF16SteelPipeline(expertDFF, dModel, gs, bits)
		if err != nil {
			t.Skipf("all-routes gather unavailable: %v", err)
		}
		gatherDownPSO, err := gatherQMVBF16SteelPipeline(dModel, expertDFF, gs, bits)
		if err != nil {
			t.Skipf("all-routes down gather unavailable: %v", err)
		}
		inKey := gatherQMVAllRoutesMetaKey{numExperts: numExperts, outDim: expertDFF, inDim: dModel, groupSize: gs, bits: bits, expertRows: 2 * expertDFF, routes: pairs, xRows: K, batchedX: true}
		inMeta, err := gatherQMVAllRoutesMetadata(numExperts, expertDFF, dModel, gs, bits, 2*expertDFF, pairs, K, true)
		if err != nil {
			t.Fatalf("in meta: %v", err)
		}
		downKey := gatherQMVAllRoutesMetaKey{numExperts: numExperts, outDim: dModel, inDim: expertDFF, groupSize: gs, bits: bits, expertRows: dModel, routes: pairs, xRows: pairs, batchedX: true}
		downMeta, err := gatherQMVAllRoutesMetadata(numExperts, dModel, expertDFF, gs, bits, dModel, pairs, pairs, true)
		if err != nil {
			t.Fatalf("down meta: %v", err)
		}
		var encErr error
		withAutoreleasePool(func() {
			cb := commandBufferFast(queue)
			enc := computeCommandEncoderFast(cb)
			sink := encSink{enc}
			emitGatherQMVAllRoutes(sink, gatherInPSO, inMeta, inKey, mb.expertIn, 0, guP.buf, guP.off, guS.buf, guS.off, guB.buf, guB.off, mb.pairToToken, mb.idx, 0, mb.gateAll, 0, expertDFF, dModel, gs, bits, 0, pairs)
			emitGatherQMVAllRoutes(sink, gatherInPSO, inMeta, inKey, mb.expertIn, 0, guP.buf, guP.off, guS.buf, guS.off, guB.buf, guB.off, mb.pairToToken, mb.idx, 0, mb.upAll, 0, expertDFF, dModel, gs, bits, expertDFF, pairs)
			encErr = encGeluGateMulFused(enc, mb.gateAll, mb.upAll, mb.gatedAll, pairs*expertDFF)
			if encErr == nil {
				emitGatherQMVAllRoutes(sink, gatherDownPSO, downMeta, downKey, mb.gatedAll, 0, dnP.buf, dnP.off, dnS.buf, dnS.off, dnB.buf, dnB.off, mb.pairIota, mb.idx, 0, mb.downAll, 0, dModel, expertDFF, gs, bits, 0, pairs)
			}
			endEncodingFast(enc)
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
		})
		if encErr != nil {
			t.Fatalf("all-routes encode: %v", encErr)
		}
		bb := unsafe.Slice((*byte)(mb.downAll.Contents()), pairs*dModel*2)
		out := make([]float32, pairs*dModel)
		for i := range out {
			out[i] = bf16ToF32(bb[i*2], bb[i*2+1])
		}
		return out
	}

	runGrouped := func() []float32 {
		if !moeGroupedUsable(MoEQuantLayerWeights{NumExperts: numExperts}, bits, gs) {
			t.Skip("grouped lane unavailable")
		}
		// scrub downAll so stale all-routes bytes cannot fake parity
		zero := unsafe.Slice((*byte)(mb.downAll.Contents()), pairs*dModel*2)
		for i := range zero {
			zero[i] = 0
		}
		var encErr error
		withAutoreleasePool(func() {
			cb := commandBufferFast(queue)
			e := computeCommandEncoderFast(cb)
			encErr = s.encMoEExpertsGrouped(e, mb, true,
				guP, guS, guB,
				bufView{}, bufView{}, bufView{},
				bufView{}, bufView{}, bufView{},
				dnP, dnS, dnB,
				gs, bits, gs, bits,
				numExperts, topK, expertDFF, dModel, pairs)
			endEncodingFast(e)
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
		})
		if encErr != nil {
			t.Fatalf("grouped encode: %v", encErr)
		}
		bb := unsafe.Slice((*byte)(mb.downAll.Contents()), pairs*dModel*2)
		out := make([]float32, pairs*dModel)
		for i := range out {
			out[i] = bf16ToF32(bb[i*2], bb[i*2+1])
		}
		return out
	}

	want := runAllRoutes()
	got := runGrouped()
	worst, worstRow := 2.0, -1
	for r := range pairs {
		var dot, ng, nw float64
		for i := range dModel {
			g, w := float64(got[r*dModel+i]), float64(want[r*dModel+i])
			dot += g * w
			ng += g * g
			nw += w * w
		}
		cos := dot / (math.Sqrt(ng)*math.Sqrt(nw) + 1e-30)
		if cos < worst {
			worst, worstRow = cos, r
		}
	}
	t.Logf("grouped vs all-routes: worst pair cos=%.6f@row%d", worst, worstRow)
	// token-identity tier: simdgroup-MMA accumulation order vs the per-pair GEMV — the same
	// boundary the prompt fold's qmm_t trades at. Ulp noise compounds through gate→gelu→down.
	if worst < 0.995 {
		// stage bisection for the worst pair: which grouped stage first parts from the
		// all-routes slabs (which still hold the reference run's intermediates).
		gsc := s.denseBatch.moeGrouped
		sp := unsafe.Slice((*int32)(unsafe.Pointer(gsc.sortedPair.Contents())), pairs)
		srow := -1
		for i := range pairs {
			if int(sp[i]) == worstRow {
				srow = i
				break
			}
		}
		readRow := func(buf metal.MTLBuffer, row, width int) []float64 {
			bb := unsafe.Slice((*byte)(buf.Contents()), (row+1)*width*2)
			out := make([]float64, width)
			for i := range width {
				o := (row*width + i) * 2
				out[i] = float64(bf16ToF32(bb[o], bb[o+1]))
			}
			return out
		}
		cosOf := func(a, b []float64) float64 {
			var dot, na, nb float64
			for i := range a {
				dot += a[i] * b[i]
				na += a[i] * a[i]
				nb += b[i] * b[i]
			}
			return dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-30)
		}
		tok := worstRow / topK
		xs := readRow(gsc.xSorted, srow, dModel)
		xr := readRow(mb.expertIn, tok, dModel)
		gu := readRow(gsc.guSorted, srow, 2*expertDFF)
		gRef := readRow(mb.gateAll, worstRow, expertDFF)
		uRef := readRow(mb.upAll, worstRow, expertDFF)
		gated := readRow(gsc.gatedSorted, srow, expertDFF)
		gatedRef := readRow(mb.gatedAll, worstRow, expertDFF)
		down := readRow(gsc.downSorted, srow, dModel)
		wantRow := make([]float64, dModel)
		for i := range dModel {
			wantRow[i] = float64(want[worstRow*dModel+i])
		}
		t.Logf("stage bisect pair %d (srow %d, expert %d): x=%.6f gate=%.6f up=%.6f gated=%.6f down=%.6f",
			worstRow, srow, idx[worstRow], cosOf(xs, xr), cosOf(gu[:expertDFF], gRef), cosOf(gu[expertDFF:], uRef), cosOf(gated, gatedRef), cosOf(down, wantRow))
		t.Errorf("grouped expert lane diverges from the all-routes reference: worst cos %.6f at pair %d (expert %d)", worst, worstRow, idx[worstRow])
	}
}
