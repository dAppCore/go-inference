// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"slices"
	"testing"
)

// routerFusedParityCase drives ONE router through both encode paths on the same
// plan — the fused single-dispatch kernel and the rms|qmv|topk chain — and
// returns (idx, weights) from each. Engagement is asserted in both directions
// so the compare cannot go vacuous.
func routerFusedParityCase(t *testing.T, numExperts, topK, dModel, groupSize, bits int, withScale bool) ([]int32, []byte, []int32, []byte) {
	t.Helper()
	x := toBF16Bytes(syntheticFloat32(dModel, 7))
	normW := toBF16Bytes(syntheticFloat32(dModel, 11))
	packed := make([]byte, numExperts*dModel*bits/8)
	for i := range packed {
		packed[i] = byte((i*131 + 17) % 256)
	}
	nSB := numExperts * (dModel / groupSize)
	proj := QuantWeight{
		Packed: packed,
		Scales: toBF16Bytes(syntheticFloat32(nSB, 13)),
		Biases: toBF16Bytes(syntheticFloat32(nSB, 17)),
	}
	var scale []byte
	if withScale {
		scale = toBF16Bytes(syntheticFloat32(numExperts, 19))
	}

	run := func(fused bool) ([]int32, []byte) {
		wasEnabled := routerFusedEnabled
		routerFusedEnabled = fused
		defer func() { routerFusedEnabled = wasEnabled }()

		scratch, err := getRouterDeviceScratch(dModel, numExperts, topK)
		if err != nil {
			t.Fatalf("getRouterDeviceScratch: %v", err)
		}
		defer putRouterDeviceScratch(scratch)
		inputBuf, ok := scratch.inputView(x)
		if !ok {
			if inputBuf, err = scratch.x.copyPrefixBuffer(x); err != nil {
				t.Fatalf("copyPrefixBuffer: %v", err)
			}
		}
		plan, err := buildRouterEncodePlan(scratch, inputBuf, normW, bufView{}, proj, scale, bufView{}, numExperts, topK, dModel, groupSize, bits, 1e-6)
		if err != nil {
			t.Fatalf("buildRouterEncodePlan: %v", err)
		}
		if fused && plan.fusedPSO == nil {
			t.Skipf("fused router pipeline unavailable (gs=%d b=%d topK=%d)", groupSize, bits, topK)
		}
		before := routerFusedDispatches.Load()

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		sink := encSink{enc}
		if !plan.emitFused(sink) {
			plan.emitRMS(sink)
			plan.emitQMV(sink)
			plan.emitTopK(sink)
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)

		engaged := routerFusedDispatches.Load() > before
		if fused && !engaged {
			t.Fatal("fused router lane did not engage")
		}
		if !fused && engaged {
			t.Fatal("fused router lane engaged while disabled — the A/B is vacuous")
		}
		return copyRouterTopKOutput(scratch, topK)
	}

	chainIdx, chainW := run(false)
	fusedIdx, fusedW := run(true)
	return chainIdx, chainW, fusedIdx, fusedW
}

// TestRouterFusedMatchesChain gates the single-dispatch fused router (#340):
// the fused kernel must reproduce the rms|qmv|topk chain's routing decision
// BYTE-FOR-BYTE — same expert indices in the same order, same softmax weight
// bytes — because a ~1 ULP score drift can flip a top-k near-tie and change
// expert selection. The kernel earns this by structural replication (the rms
// tree at its own threadgroup shape, qmv_impl called verbatim on virtual
// threadgroups, the shared top-k body), and this test is the receipt: the 26B
// shape (128 experts, top-8, dModel 2816), a non-multiple-of-8 expert count
// (qmv_impl's guarded tail tile), a tiny single-simdgroup shape (the
// sequential virtual-pair fallback), scaled and unscaled, across quant widths.
func TestRouterFusedMatchesChain(t *testing.T) {
	requireNativeRuntime(t)
	if !gpuHasGeluKernel() {
		t.Skip("custom kernel library (lthn_kernels.metallib) not loaded — run `task metallib:kernels`")
	}

	cases := []struct {
		name                               string
		numExperts, topK, dModel, gs, bits int
		withScale                          bool
	}{
		{"26B shape", 128, 8, 2816, 64, 4, true},
		{"26B shape unscaled", 128, 8, 2816, 64, 4, false},
		{"tail tile", 12, 4, 512, 32, 4, true},
		{"single simdgroup", 16, 2, 128, 32, 4, false},
		{"wide k topk32", 64, 32, 1024, 64, 4, true},
		{"8-bit", 32, 4, 512, 64, 8, true},
		{"3-bit", 32, 4, 768, 32, 3, false},
	}
	for _, c := range cases {
		chainIdx, chainW, fusedIdx, fusedW := routerFusedParityCase(t, c.numExperts, c.topK, c.dModel, c.gs, c.bits, c.withScale)
		if !slices.Equal(chainIdx, fusedIdx) {
			t.Fatalf("%s: expert selection differs: chain %v fused %v", c.name, chainIdx, fusedIdx)
		}
		if !bytes.Equal(chainW, fusedW) {
			t.Fatalf("%s: routing weights differ (idx %v): chain % x fused % x", c.name, chainIdx, chainW, fusedW)
		}
	}
	t.Logf("fused router matches the 3-dispatch chain byte-for-byte across %d geometries", len(cases))
}
