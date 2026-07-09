// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"os"
	"testing"
	"unsafe"
)

var topKByScoreSink []int32

func TestTopKByScoreDirectSelectionAllocationBudget(t *testing.T) {
	const numExperts, topK = 4096, 2
	scores := make([]float32, numExperts)
	for i := range scores {
		scores[i] = float32((i*37)%1000) * 0.001
	}
	scores[17] = 9
	scores[4095] = 8

	allocs := testing.AllocsPerRun(100, func() {
		got := topKByScore(scores, topK)
		if len(got) != topK || got[0] != 17 || got[1] != 4095 {
			t.Fatalf("topKByScore = %v, want [17 4095]", got)
		}
		topKByScoreSink = got
	})
	if allocs > 1 {
		t.Fatalf("topKByScore allocs/run = %.1f, want <= 1 for direct top-k selection", allocs)
	}
}

func TestMoERouterTopKKernelMatchesHostSelection(t *testing.T) {
	requireNativeRuntime(t)

	scores := toBF16Bytes([]float32{0.5, 2.0, -1.0, 2.0, 0.25, 1.5, 3.0, 0.75})
	scale := toBF16Bytes([]float32{1.0, 0.5, 2.0, 0.25, 1.5, 0.75, 3.0, 0.1})
	const numExperts, topK = 8, 3

	gotIdx, gotWeights, err := routerTopKBF16(scores, scale, numExperts, topK)
	if err != nil {
		t.Fatalf("routerTopKBF16: %v", err)
	}
	wantIdx, wantWeights := routerSelect(scores, scale, numExperts, topK)
	if len(gotIdx) != len(wantIdx) || len(gotWeights) != len(wantWeights) {
		t.Fatalf("routerTopKBF16 returned %d idx/%d weight bytes, want %d/%d", len(gotIdx), len(gotWeights), len(wantIdx), len(wantWeights))
	}
	for i := range wantIdx {
		if gotIdx[i] != wantIdx[i] {
			t.Fatalf("routerTopKBF16 idx[%d] = %d, want %d (idx=%v want=%v)", i, gotIdx[i], wantIdx[i], gotIdx, wantIdx)
		}
		got := bf16ToF32(gotWeights[i*bf16Size], gotWeights[i*bf16Size+1])
		want := bf16ToF32(wantWeights[i*bf16Size], wantWeights[i*bf16Size+1])
		if d := got - want; d > 0.005 || d < -0.005 {
			t.Fatalf("routerTopKBF16 weight[%d] = %.6f, want %.6f (delta %.6f)", i, got, want, d)
		}
	}
}

func TestRouterTopKBF16AllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	scores := toBF16Bytes([]float32{0.5, 2.0, -1.0, 2.0, 0.25, 1.5, 3.0, 0.75})
	scale := toBF16Bytes([]float32{1.0, 0.5, 2.0, 0.25, 1.5, 0.75, 3.0, 0.1})
	const numExperts, topK = 8, 3
	if _, _, err := routerTopKBF16(scores, scale, numExperts, topK); err != nil {
		t.Fatalf("routerTopKBF16 warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(20, func() {
		idx, weights, err := routerTopKBF16(scores, scale, numExperts, topK)
		if err != nil {
			t.Fatalf("routerTopKBF16: %v", err)
		}
		if len(idx) != topK || len(weights) != topK*bf16Size {
			t.Fatalf("routerTopKBF16 returned %d idx/%d weight bytes", len(idx), len(weights))
		}
	})
	if allocs > 25 {
		t.Fatalf("routerTopKBF16 allocations = %.0f, want <= 25", allocs)
	}
}

func TestMoERouterDeviceTopKAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const numExperts, topK, dModel = 8, 2, 64
	x := toBF16Bytes(syntheticFloat32(dModel, 31))
	normW := toBF16Bytes(syntheticFloat32(dModel, 17))
	routerW := toBF16Bytes(syntheticFloat32(numExperts*dModel, 43))
	scale := toBF16Bytes([]float32{1.0, 0.5, 2.0, 0.25, 1.5, 0.75, 3.0, 0.1})
	for i := range 2 {
		if _, _, err := MoERouter(x, normW, routerW, scale, numExperts, topK, dModel, 1e-5); err != nil {
			t.Fatalf("MoERouter warm %d: %v", i, err)
		}
	}

	allocs := testing.AllocsPerRun(50, func() {
		idx, weights, err := MoERouter(x, normW, routerW, scale, numExperts, topK, dModel, 1e-5)
		if err != nil {
			t.Fatalf("MoERouter: %v", err)
		}
		if len(idx) != topK || len(weights) != topK*bf16Size {
			t.Fatalf("MoERouter returned %d idx/%d weight bytes", len(idx), len(weights))
		}
	})
	if allocs > 405 {
		t.Fatalf("MoERouter warmed device top-k allocations = %.0f, want <= 405", allocs)
	}
}

func TestMoERouterQuantDeviceTopKAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const numExperts, topK, dModel, groupSize, bits = 8, 2, 64, 32, 4
	x := toBF16Bytes(syntheticFloat32(dModel, 31))
	normW := toBF16Bytes(syntheticFloat32(dModel, 17))
	routerW := quantWeightFixture(t, numExperts, dModel, groupSize, bits, 43)
	scale := toBF16Bytes([]float32{1.0, 0.5, 2.0, 0.25, 1.5, 0.75, 3.0, 0.1})
	for i := range 2 {
		if _, _, err := MoERouterQuant(x, normW, routerW, scale, numExperts, topK, dModel, groupSize, bits, 1e-5); err != nil {
			t.Fatalf("MoERouterQuant warm %d: %v", i, err)
		}
	}

	allocs := testing.AllocsPerRun(50, func() {
		idx, weights, err := MoERouterQuant(x, normW, routerW, scale, numExperts, topK, dModel, groupSize, bits, 1e-5)
		if err != nil {
			t.Fatalf("MoERouterQuant: %v", err)
		}
		if len(idx) != topK || len(weights) != topK*bf16Size {
			t.Fatalf("MoERouterQuant returned %d idx/%d weight bytes", len(idx), len(weights))
		}
	})
	if allocs > 364 {
		t.Fatalf("MoERouterQuant warmed device top-k allocations = %.0f, want <= 364", allocs)
	}
}

func TestRouterDeviceScratchInputViewUsesCallerBacking(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, numExperts, topK = 64, 8, 2
	x := toBF16Bytes(syntheticFloat32(dModel, 31))
	scratch, err := getRouterDeviceScratch(dModel, numExperts, topK)
	if err != nil {
		t.Fatalf("getRouterDeviceScratch: %v", err)
	}
	defer scratch.Close()

	buf, ok := scratch.inputView(x)
	if !ok {
		t.Fatal("inputView ok = false")
	}
	if got, want := uintptr(buf.Contents()), uintptr(unsafe.Pointer(&x[0])); got != want {
		t.Fatalf("inputView buffer pointer = %#x, want caller backing %#x", got, want)
	}
	reused, ok := scratch.inputView(x)
	if !ok {
		t.Fatal("reused inputView ok = false")
	}
	if reused.GetID() != buf.GetID() {
		t.Fatal("inputView did not reuse the cached no-copy buffer")
	}
}

func TestRouterDeviceScratchInputViewReusesPinnedOwnerBuffer(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, numExperts, topK = 64, 8, 2
	pinned, err := newPinnedNoCopyBytes(dModel * bf16Size)
	if err != nil {
		t.Fatalf("newPinnedNoCopyBytes: %v", err)
	}
	defer pinned.Close()

	scratch, err := getRouterDeviceScratch(dModel, numExperts, topK)
	if err != nil {
		t.Fatalf("getRouterDeviceScratch: %v", err)
	}
	defer scratch.Close()

	buf, ok := scratch.inputView(pinned.bytes)
	if !ok {
		t.Fatal("inputView ok = false")
	}
	requirePinnedOwnerBuffer(t, "router input view", buf, pinned)
}

func TestRouterDeviceScratchPoolKeepsShapesResident(t *testing.T) {
	requireNativeRuntime(t)

	small, err := getRouterDeviceScratch(65, 9, 3)
	if err != nil {
		t.Fatalf("get small router device scratch: %v", err)
	}
	putRouterDeviceScratch(small)
	large, err := getRouterDeviceScratch(97, 17, 4)
	if err != nil {
		t.Fatalf("get large router device scratch: %v", err)
	}
	putRouterDeviceScratch(large)

	gotSmall, err := getRouterDeviceScratch(65, 9, 3)
	if err != nil {
		t.Fatalf("get small router device scratch again: %v", err)
	}
	defer putRouterDeviceScratch(gotSmall)
	if gotSmall != small {
		t.Fatal("router device scratch pool evicted the small shape after using a larger shape")
	}
	gotLarge, err := getRouterDeviceScratch(97, 17, 4)
	if err != nil {
		t.Fatalf("get large router device scratch again: %v", err)
	}
	defer putRouterDeviceScratch(gotLarge)
	if gotLarge != large {
		t.Fatal("router device scratch pool evicted the large shape after reusing the small shape")
	}
}

// routerRef independently computes the ideal routing decision from the parity-proven
// ops: scores = MatVecBF16 on the RMS-normed input, the genuine top-k SET found by a
// repeated max-scan with separate bookkeeping, then a float64 softmax over those
// scores and the optional per-expert scale. Returns an expert→weight map —
// order-invariant, so the gate never depends on the order idx is returned in.
func routerRef(t *testing.T, x, normWScaled, routerW, perExpertScale []byte, numExperts, topK, dModel int, eps float32) map[int32]float32 {
	t.Helper()
	normed, err := RMSNormBF16(x, normWScaled, 1, dModel, eps)
	if err != nil {
		t.Fatalf("routerRef rms: %v", err)
	}
	scoresB, err := MatVecBF16(routerW, normed, numExperts, dModel)
	if err != nil {
		t.Fatalf("routerRef gemv: %v", err)
	}
	scores := make([]float32, numExperts)
	for e := range scores {
		scores[e] = bf16ToF32(scoresB[e*bf16Size], scoresB[e*bf16Size+1])
	}
	// genuine top-k by repeated max-scan; strict > resolves ties to the lower index.
	used := make([]bool, numExperts)
	sel := make([]int, 0, topK)
	for range topK {
		best := -1
		for e := range numExperts {
			if used[e] {
				continue
			}
			if best == -1 || scores[e] > scores[best] {
				best = e
			}
		}
		used[best] = true
		sel = append(sel, best)
	}
	// float64 softmax over the selected scores (the ideal; MoERouter does it in f32
	// then rounds to bf16, hence the tolerance in the gate).
	maxS := math.Inf(-1)
	for _, e := range sel {
		if float64(scores[e]) > maxS {
			maxS = float64(scores[e])
		}
	}
	ex := make([]float64, topK)
	var sum float64
	for i, e := range sel {
		ex[i] = math.Exp(float64(scores[e]) - maxS)
		sum += ex[i]
	}
	m := make(map[int32]float32, topK)
	for i, e := range sel {
		w := ex[i] / sum
		if perExpertScale != nil {
			w *= float64(bf16ToF32(perExpertScale[e*bf16Size], perExpertScale[e*bf16Size+1]))
		}
		m[int32(e)] = float32(w)
	}
	return m
}

// gotMap decodes MoERouter's (idx, weights) into an expert→weight map.
func gotMap(t *testing.T, idx []int32, weights []byte, topK int) map[int32]float32 {
	t.Helper()
	if len(idx) != topK || len(weights) != topK*bf16Size {
		t.Fatalf("MoERouter returned %d idx / %d weight bytes, want topK=%d", len(idx), len(weights), topK)
	}
	m := make(map[int32]float32, topK)
	for i, e := range idx {
		if _, dup := m[e]; dup {
			t.Fatalf("MoERouter returned duplicate expert %d", e)
		}
		m[e] = bf16ToF32(weights[i*bf16Size], weights[i*bf16Size+1])
	}
	return m
}

// TestMoERouter gates the MoE router sub-slice. MoERouter (RMS-norm → expert-score
// gemv → host top-k + softmax + optional per-expert scale) is checked against an
// independent reference: the selected expert SET must match exactly (the routing
// decision — the load-bearing correctness property), and each expert's weight must
// match the ideal softmax within bf16 tolerance. Order-invariant throughout.
func TestMoERouter(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const numExperts, dModel = 8, 256
	const eps = float32(1e-6)
	mk := func(n, salt int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32((i*salt+11)%97-48) * 0.02
		}
		return s
	}
	x := toBF16Bytes(mk(dModel, 31))
	normWScaled := toBF16Bytes(mk(dModel, 17)) // already Scale·RootSize (folded at load)
	routerW := toBF16Bytes(mk(numExperts*dModel, 43))
	perExpertScale := toBF16Bytes([]float32{1.0, 0.5, 2.0, 0.25, 1.5, 0.75, 3.0, 0.1})

	// the expert→weight maps must agree to within bf16 precision (the router rounds
	// each weight to bf16; the reference is the ideal f64 value).
	const tol = float32(0.02)
	check := func(name string, topK int, scale []byte, wantSum bool) {
		idx, weights, err := MoERouter(x, normWScaled, routerW, scale, numExperts, topK, dModel, eps)
		if err != nil {
			t.Fatalf("%s: MoERouter: %v", name, err)
		}
		got := gotMap(t, idx, weights, topK)
		want := routerRef(t, x, normWScaled, routerW, scale, numExperts, topK, dModel, eps)
		if len(got) != len(want) {
			t.Fatalf("%s: got %d experts, want %d", name, len(got), len(want))
		}
		var sum float32
		for e, gw := range got {
			ww, ok := want[e]
			if !ok {
				t.Fatalf("%s: router selected expert %d not in the reference top-k set", name, e)
			}
			if d := gw - ww; d > tol || d < -tol {
				t.Fatalf("%s: expert %d weight %.5f, want %.5f (Δ%.5f > %.5f)", name, e, gw, ww, d, tol)
			}
			sum += gw
		}
		if wantSum && (sum < 1-tol || sum > 1+tol) {
			t.Fatalf("%s: softmax weights sum to %.5f, want ~1", name, sum)
		}
		t.Logf("%s (top-%d): expert set ✓ exact, weights ✓ within %.3f", name, topK, tol)
	}

	check("plain top-2", 2, nil, true)                        // softmax weights sum to 1
	check("per-expert-scale top-2", 2, perExpertScale, false) // scaled → no unit sum
	check("top-3", 3, nil, true)
	check("all experts (topK==numExperts)", numExperts, nil, true)
}

func TestMoERouterCachesProjectionWeight(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const numExperts, topK, dModel = 8, 2, 256
	const eps = float32(1e-6)
	x := toBF16Bytes(syntheticFloat32(dModel, 31))
	normWScaled := toBF16Bytes(syntheticFloat32(dModel, 17))
	routerW := toBF16Bytes(syntheticFloat32(numExperts*dModel, 43))

	if _, _, err := MoERouter(x, normWScaled, routerW, nil, numExperts, topK, dModel, eps); err != nil {
		t.Fatalf("MoERouter: %v", err)
	}

	residentBufMu.Lock()
	got := len(residentBufs)
	_, ok := residentBufs[uintptr(unsafe.Pointer(&routerW[0]))]
	residentBufMu.Unlock()
	if !ok {
		t.Fatalf("MoERouter did not keep router projection resident (resident=%d want>=1)", got)
	}
}

func TestMoERouterDeviceTopKCachesNormAndScale(t *testing.T) {
	requireNativeRuntime(t)
	if !routerTopKUsable(8, 2) {
		t.Fatal("native router top-k kernel is unavailable")
	}
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const numExperts, topK, dModel = 8, 2, 256
	const eps = float32(1e-6)
	x := toBF16Bytes(syntheticFloat32(dModel, 31))
	normWScaled := toBF16Bytes(syntheticFloat32(dModel, 17))
	routerW := toBF16Bytes(syntheticFloat32(numExperts*dModel, 43))
	scale := toBF16Bytes([]float32{1.0, 0.5, 2.0, 0.25, 1.5, 0.75, 3.0, 0.1})

	if _, _, err := MoERouter(x, normWScaled, routerW, scale, numExperts, topK, dModel, eps); err != nil {
		t.Fatalf("MoERouter: %v", err)
	}

	has := func(b []byte) bool {
		t.Helper()
		residentBufMu.Lock()
		defer residentBufMu.Unlock()
		_, ok := residentBufs[uintptr(unsafe.Pointer(&b[0]))]
		return ok
	}
	if !has(normWScaled) || !has(routerW) || !has(scale) {
		t.Fatalf("MoERouter did not keep device top-k inputs resident (norm=%v router=%v scale=%v)", has(normWScaled), has(routerW), has(scale))
	}
}

func TestMoERouterHostSelectScratchReusesNormAndScoreBuffers(t *testing.T) {
	requireNativeRuntime(t)

	const numExperts, topK, dModel = 8, 2, 64
	const eps = float32(1e-5)
	x := toBF16Bytes(syntheticFloat32(dModel, 31))
	normWScaled := toBF16Bytes(syntheticFloat32(dModel, 17))
	routerW := toBF16Bytes(syntheticFloat32(numExperts*dModel, 43))
	scale := toBF16Bytes([]float32{1.0, 0.5, 2.0, 0.25, 1.5, 0.75, 3.0, 0.1})
	normed, err := RMSNormBF16(x, normWScaled, 1, dModel, eps)
	if err != nil {
		t.Fatalf("RMSNormBF16 reference: %v", err)
	}
	scores, err := matVecBF16Resident(routerW, normed, numExperts, dModel)
	if err != nil {
		t.Fatalf("matVecBF16Resident reference: %v", err)
	}
	wantIdx, wantWeights := routerSelect(scores, scale, numExperts, topK)

	scratch, err := newRouterHostScratch(dModel, numExperts)
	if err != nil {
		t.Fatalf("newRouterHostScratch: %v", err)
	}
	defer scratch.Close()
	normPtr := unsafe.Pointer(&scratch.normed.bytes[0])
	scorePtr := unsafe.Pointer(&scratch.scores.bytes[0])
	var idxPtr unsafe.Pointer
	var weightPtr unsafe.Pointer

	for i := range 2 {
		gotIdx, gotWeights, err := moeRouterBF16HostSelectWithScratch(x, normWScaled, routerW, scale, numExperts, topK, dModel, eps, scratch)
		if err != nil {
			t.Fatalf("moeRouterBF16HostSelectWithScratch %d: %v", i, err)
		}
		if unsafe.Pointer(&scratch.normed.bytes[0]) != normPtr || unsafe.Pointer(&scratch.scores.bytes[0]) != scorePtr {
			t.Fatal("router host scratch did not keep stable norm/score backing")
		}
		if len(scratch.selectScores) != numExperts || len(scratch.selectIdx) != topK || len(scratch.selectWeights) != topK*bf16Size {
			t.Fatalf("router host scratch selector lengths = scores:%d idx:%d weights:%d", len(scratch.selectScores), len(scratch.selectIdx), len(scratch.selectWeights))
		}
		if i == 0 {
			idxPtr = unsafe.Pointer(&scratch.selectIdx[0])
			weightPtr = unsafe.Pointer(&scratch.selectWeights[0])
		} else if unsafe.Pointer(&scratch.selectIdx[0]) != idxPtr || unsafe.Pointer(&scratch.selectWeights[0]) != weightPtr {
			t.Fatal("router host scratch did not keep stable selector backing")
		}
		if len(gotIdx) != len(wantIdx) || len(gotWeights) != len(wantWeights) {
			t.Fatalf("router host fallback lengths = %d/%d, want %d/%d", len(gotIdx), len(gotWeights), len(wantIdx), len(wantWeights))
		}
		for j := range wantIdx {
			if gotIdx[j] != wantIdx[j] {
				t.Fatalf("router host fallback idx[%d] = %d, want %d", j, gotIdx[j], wantIdx[j])
			}
			if gotWeights[j*bf16Size] != wantWeights[j*bf16Size] || gotWeights[j*bf16Size+1] != wantWeights[j*bf16Size+1] {
				t.Fatalf("router host fallback weight[%d] = %v, want %v", j, gotWeights, wantWeights)
			}
		}
	}
}

func TestRouterHostScratchPoolKeepsDimensionsResident(t *testing.T) {
	requireNativeRuntime(t)

	small, err := getRouterHostScratch(64, 8)
	if err != nil {
		t.Fatalf("get small router host scratch: %v", err)
	}
	putRouterHostScratch(small)

	large, err := getRouterHostScratch(128, 16)
	if err != nil {
		t.Fatalf("get large router host scratch: %v", err)
	}
	putRouterHostScratch(large)

	gotSmall, err := getRouterHostScratch(64, 8)
	if err != nil {
		t.Fatalf("get small router host scratch again: %v", err)
	}
	defer putRouterHostScratch(gotSmall)
	if gotSmall != small {
		t.Fatal("router host scratch pool evicted the small scratch after using a larger scratch")
	}

	gotLarge, err := getRouterHostScratch(128, 16)
	if err != nil {
		t.Fatalf("get large router host scratch again: %v", err)
	}
	defer putRouterHostScratch(gotLarge)
	if gotLarge != large {
		t.Fatal("router host scratch pool evicted the large scratch after reusing the small scratch")
	}
}

func TestMoERouterQuantCachesProjectionWeight(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const numExperts, topK, dModel, groupSize, bits = 8, 2, 64, 32, 4
	const eps = float32(1e-6)
	x := toBF16Bytes(syntheticFloat32(dModel, 31))
	normWScaled := toBF16Bytes(syntheticFloat32(dModel, 17))
	routerW := quantWeightFixture(t, numExperts, dModel, groupSize, bits, 43)

	if _, _, err := MoERouterQuant(x, normWScaled, routerW, nil, numExperts, topK, dModel, groupSize, bits, eps); err != nil {
		t.Fatalf("MoERouterQuant: %v", err)
	}

	key := func(b []byte) uintptr { return uintptr(unsafe.Pointer(&b[0])) }
	residentBufMu.Lock()
	got := len(residentBufs)
	_, hasPacked := residentBufs[key(routerW.Packed)]
	_, hasScales := residentBufs[key(routerW.Scales)]
	_, hasBiases := residentBufs[key(routerW.Biases)]
	residentBufMu.Unlock()
	if !hasPacked || !hasScales || !hasBiases {
		t.Fatalf("MoERouterQuant did not keep router quant projection resident (packed=%v scales=%v biases=%v resident=%d want>=3)", hasPacked, hasScales, hasBiases, got)
	}
}

func TestMoERouterQuantHonoursWeightGeometry(t *testing.T) {
	requireNativeRuntime(t)

	const numExperts, topK, dModel = 8, 2, 64
	const routerGroupSize, fallbackGroupSize, bits = 64, 32, 4
	const eps = float32(1e-6)
	x := toBF16Bytes(syntheticFloat32(dModel, 31))
	normWScaled := toBF16Bytes(syntheticFloat32(dModel, 17))
	scale := toBF16Bytes([]float32{1.0, 0.5, 2.0, 0.25, 1.5, 0.75, 3.0, 0.1})
	routerW := quantWeightFixture(t, numExperts, dModel, routerGroupSize, bits, 43)

	gotIdx, gotWeights, err := MoERouterQuant(x, normWScaled, routerW, scale, numExperts, topK, dModel, fallbackGroupSize, bits, eps)
	if err != nil {
		t.Fatalf("MoERouterQuant with per-weight geometry: %v", err)
	}
	normed, err := RMSNormBF16(x, normWScaled, 1, dModel, eps)
	if err != nil {
		t.Fatalf("RMSNormBF16: %v", err)
	}
	scores, err := QMVBF16(normed, routerW.Packed, routerW.Scales, routerW.Biases, numExperts, dModel, routerGroupSize, bits)
	if err != nil {
		t.Fatalf("QMVBF16 reference: %v", err)
	}
	wantIdx, wantWeights := routerSelect(scores, scale, numExperts, topK)
	if len(gotIdx) != len(wantIdx) || len(gotWeights) != len(wantWeights) {
		t.Fatalf("MoERouterQuant lengths = %d/%d, want %d/%d", len(gotIdx), len(gotWeights), len(wantIdx), len(wantWeights))
	}
	for i := range wantIdx {
		if gotIdx[i] != wantIdx[i] {
			t.Fatalf("MoERouterQuant idx[%d] = %d, want %d (got=%v want=%v)", i, gotIdx[i], wantIdx[i], gotIdx, wantIdx)
		}
		if gotWeights[i*bf16Size] != wantWeights[i*bf16Size] || gotWeights[i*bf16Size+1] != wantWeights[i*bf16Size+1] {
			t.Fatalf("MoERouterQuant weight[%d] = %v, want %v", i, gotWeights, wantWeights)
		}
	}
}

func TestMoERouterQuantHostSelectScratchReusesNormAndScoreBuffers(t *testing.T) {
	requireNativeRuntime(t)

	const numExperts, topK, dModel, groupSize, bits = 8, 2, 64, 32, 4
	const eps = float32(1e-5)
	x := toBF16Bytes(syntheticFloat32(dModel, 31))
	normWScaled := toBF16Bytes(syntheticFloat32(dModel, 17))
	scale := toBF16Bytes([]float32{1.0, 0.5, 2.0, 0.25, 1.5, 0.75, 3.0, 0.1})
	routerW := quantWeightFixture(t, numExperts, dModel, groupSize, bits, 43)
	normed, err := RMSNormBF16(x, normWScaled, 1, dModel, eps)
	if err != nil {
		t.Fatalf("RMSNormBF16 reference: %v", err)
	}
	scores, err := qmvBF16Resident(normed, routerW, numExperts, dModel, groupSize, bits)
	if err != nil {
		t.Fatalf("qmvBF16Resident reference: %v", err)
	}
	wantIdx, wantWeights := routerSelect(scores, scale, numExperts, topK)

	scratch, err := newRouterQuantHostScratch(dModel, numExperts)
	if err != nil {
		t.Fatalf("newRouterQuantHostScratch: %v", err)
	}
	defer scratch.Close()
	normPtr := unsafe.Pointer(&scratch.normed.bytes[0])
	scorePtr := unsafe.Pointer(&scratch.scores.bytes[0])
	var idxPtr unsafe.Pointer
	var weightPtr unsafe.Pointer

	for i := range 2 {
		gotIdx, gotWeights, err := moeRouterQuantHostSelectWithScratch(x, normWScaled, bufView{}, routerW, scale, numExperts, topK, dModel, groupSize, bits, eps, scratch)
		if err != nil {
			t.Fatalf("moeRouterQuantHostSelectWithScratch %d: %v", i, err)
		}
		if unsafe.Pointer(&scratch.normed.bytes[0]) != normPtr || unsafe.Pointer(&scratch.scores.bytes[0]) != scorePtr {
			t.Fatal("router quant host scratch did not keep stable norm/score backing")
		}
		if len(scratch.selectScores) != numExperts || len(scratch.selectIdx) != topK || len(scratch.selectWeights) != topK*bf16Size {
			t.Fatalf("router quant host scratch selector lengths = scores:%d idx:%d weights:%d", len(scratch.selectScores), len(scratch.selectIdx), len(scratch.selectWeights))
		}
		if i == 0 {
			idxPtr = unsafe.Pointer(&scratch.selectIdx[0])
			weightPtr = unsafe.Pointer(&scratch.selectWeights[0])
		} else if unsafe.Pointer(&scratch.selectIdx[0]) != idxPtr || unsafe.Pointer(&scratch.selectWeights[0]) != weightPtr {
			t.Fatal("router quant host scratch did not keep stable selector backing")
		}
		if len(gotIdx) != len(wantIdx) || len(gotWeights) != len(wantWeights) {
			t.Fatalf("router quant host fallback lengths = %d/%d, want %d/%d", len(gotIdx), len(gotWeights), len(wantIdx), len(wantWeights))
		}
		for j := range wantIdx {
			if gotIdx[j] != wantIdx[j] {
				t.Fatalf("router quant host fallback idx[%d] = %d, want %d", j, gotIdx[j], wantIdx[j])
			}
			if gotWeights[j*bf16Size] != wantWeights[j*bf16Size] || gotWeights[j*bf16Size+1] != wantWeights[j*bf16Size+1] {
				t.Fatalf("router quant host fallback weight[%d] = %v, want %v", j, gotWeights, wantWeights)
			}
		}
	}
}

func TestMoERouterQuantDeviceTopKCachesNormAndScale(t *testing.T) {
	requireNativeRuntime(t)
	if !routerTopKUsable(8, 2) {
		t.Fatal("native router top-k kernel is unavailable")
	}
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const numExperts, topK, dModel, groupSize, bits = 8, 2, 64, 32, 4
	const eps = float32(1e-6)
	x := toBF16Bytes(syntheticFloat32(dModel, 31))
	normWScaled := toBF16Bytes(syntheticFloat32(dModel, 17))
	routerW := quantWeightFixture(t, numExperts, dModel, groupSize, bits, 43)
	scale := toBF16Bytes([]float32{1.0, 0.5, 2.0, 0.25, 1.5, 0.75, 3.0, 0.1})

	if _, _, err := MoERouterQuant(x, normWScaled, routerW, scale, numExperts, topK, dModel, groupSize, bits, eps); err != nil {
		t.Fatalf("MoERouterQuant: %v", err)
	}

	has := func(b []byte) bool {
		t.Helper()
		residentBufMu.Lock()
		defer residentBufMu.Unlock()
		_, ok := residentBufs[uintptr(unsafe.Pointer(&b[0]))]
		return ok
	}
	if !has(normWScaled) || !has(routerW.Packed) || !has(routerW.Scales) || !has(routerW.Biases) || !has(scale) {
		t.Fatalf("MoERouterQuant did not keep device top-k inputs resident (norm=%v packed=%v scales=%v biases=%v scale=%v)",
			has(normWScaled), has(routerW.Packed), has(routerW.Scales), has(routerW.Biases), has(scale))
	}
}
