// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"
	"unsafe"
)

// bf16UlpDist returns the worst per-element ULP distance between two bf16 byte slices and the index
// it occurs at. Same-sign values a ULP apart differ by 1 in the sign-magnitude-ordered key; 0 means
// byte-identical.
func bf16UlpDist(a, b []byte) (worst, at int) {
	key := func(lo, hi byte) int32 {
		u := uint16(lo) | uint16(hi)<<8
		if u&0x8000 != 0 {
			return -int32(u & 0x7FFF)
		}
		return int32(u & 0x7FFF)
	}
	n := len(a) / 2
	for i := 0; i < n; i++ {
		ka := key(a[2*i], a[2*i+1])
		kb := key(b[2*i], b[2*i+1])
		d := int(ka - kb)
		if d < 0 {
			d = -d
		}
		if d > worst {
			worst, at = d, i
		}
	}
	return worst, at
}

// TestChainResidualNormMoEQuantTail gates the recordable MoE FFN tail: the chain emitter's MoE BODY
// (router → gather experts → SiLU SwiGLU → weighted combine → shared expert), on a given normed bf16
// input, must equal the CURRENT device MoE — MoEExpertsQuantSiLU over the routed idx/weights from the
// SAME router, plus the shared expert (SiLU, ×σ(gate)) — to ≤1 bf16 ULP, for BOTH an ungated and a
// σ-gated shared expert. And it must DIFFER from a GELU-activated body, the one fact a greedy model
// A/B cannot show: this proves SiLU is engaged and the routing/combine are wired to the device idx.
func TestChainResidualNormMoEQuantTail(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Fatalf("ensureInit: %v", err)
	}
	const nE, topK, D, expertDFF, sharedFF, gs, bits = 8, 2, 64, 32, 32, 32, 4
	// the whole tail rides the lean lthn_gather_qmv; without the custom library there is nothing to test.
	gateKey := lthnGatherQMVKey{groupSize: gs, bits: bits, expertRows: expertDFF, fast: expertDFF%8 == 0 && D%512 == 0, batchedX: false}
	if _, ok := lthnGatherQMVPipeline(gateKey); !ok {
		t.Skip("lean gather kernel unavailable (custom library not loaded)")
	}

	mk := func(n, salt int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32((i*salt+11)%89-44) * 0.02
		}
		return s
	}
	// batched routed experts + a packed shared expert (all mlx-affine 4-bit).
	gate, up, down := quantMoEExpertsFixture(t, nE, D, expertDFF, gs, bits)
	sGate := QuantWeight{GroupSize: gs, Bits: bits}
	sGate.Packed, sGate.Scales, sGate.Biases = quantizeProj(t, sharedFF, D, gs, bits, 17)
	sUp := QuantWeight{GroupSize: gs, Bits: bits}
	sUp.Packed, sUp.Scales, sUp.Biases = quantizeProj(t, sharedFF, D, gs, bits, 29)
	sDown := QuantWeight{GroupSize: gs, Bits: bits}
	sDown.Packed, sDown.Scales, sDown.Biases = quantizeProj(t, D, sharedFF, gs, bits, 41)

	routerF32 := mk(nE*D, 13)
	routerBFbytes := f32sToBF16Bytes(routerF32)
	nBFbytes := toBF16Bytes(mk(D, 5)) // the post-attn-normed hidden row (already bf16)

	// device router shared by chain and reference — both select the SAME experts + softmax-over-topK
	// weights, so the gate proves the EXPERT/COMBINE/SHARED wiring, not the routing.
	scores, err := matVecBF16ResidentInto(nil, routerBFbytes, nBFbytes, nE, D)
	if err != nil {
		t.Fatalf("router scores: %v", err)
	}
	idx, weights, err := routerTopKBF16(scores, nil, nE, topK)
	if err != nil {
		t.Fatalf("router topk: %v", err)
	}

	must := func(b []byte, e error) []byte {
		t.Helper()
		if e != nil {
			t.Fatalf("ref op: %v", e)
		}
		return b
	}
	// routed reference: the current device MoE over the routed experts (SiLU SwiGLU + weighted combine).
	routedRef := must(MoEExpertsQuantSiLU(nBFbytes, idx, weights, gate, up, down, nE, topK, D, expertDFF, gs, bits))
	// shared reference: the shared expert as a 1-expert MoE at weight 1.0 — the SAME SiLU/qmv path,
	// so it equals the chain's raw shared output byte-for-byte.
	sharedRaw := must(MoEExpertsQuantSiLU(nBFbytes, []int32{0}, toBF16Bytes([]float32{1}), sGate, sUp, sDown, 1, 1, D, sharedFF, gs, bits))
	// the GELU sibling of the routed experts — a body built on it must differ from the SiLU body.
	geluRouted := must(MoEExpertsQuant(nBFbytes, idx, weights, gate, up, down, nE, topK, D, expertDFF, gs, bits))

	run := func(t *testing.T, sharedGateF32 []float32) {
		w := &moeChainWeights{
			routerBF:     residentBytes(routerBFbytes),
			routerBacker: routerBFbytes,
			numExperts:   nE, topK: topK, expertDFF: expertDFF, groupSize: gs, bits: bits,
			gate: gate, up: up, down: down,
			hasShared: true,
			sGate:     sGate, sUp: sUp, sDown: sDown, sharedFF: sharedFF,
		}
		// reference shared scale: σ(sharedGate·normed) (or ungated).
		sharedScaled := sharedRaw
		if sharedGateF32 != nil {
			sgBytes := f32sToBF16Bytes(sharedGateF32)
			w.sharedGateBF = residentBytes(sgBytes)
			w.sharedGateBacker = sgBytes
			gScore := must(matVecBF16ResidentInto(nil, sgBytes, nBFbytes, 1, D))
			g := must(SigmoidBF16(gScore))
			sharedScaled = must(MulBF16(sharedRaw, scalarFillBF16(g, D)))
		}
		bodyRef := must(AddBF16(routedRef, sharedScaled))
		geluBody := must(AddBF16(geluRouted, sharedScaled))

		sc, err := newMoEChainScratch(D, nE, topK, expertDFF, sharedFF)
		if err != nil {
			t.Fatalf("newMoEChainScratch: %v", err)
		}
		var bodyBytes []byte
		var bodyErr error
		withAutoreleasePool(func() {
			nBFbuf := sharedBytes(nBFbytes)
			cb := commandBufferFast(queue)
			enc := computeCommandEncoderFast(cb)
			tt := &chainTarget{enc: enc}
			bodyErr = chainMoEBody(tt, sc, nBFbuf, w, D)
			endEncodingFast(enc)
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
			bodyBytes = make([]byte, D*bf16Size)
			copy(bodyBytes, unsafe.Slice((*byte)(sc.bodyBF.Contents()), D*bf16Size))
		})
		if bodyErr != nil {
			t.Fatalf("chainMoEBody: %v", bodyErr)
		}
		if worst, at := bf16UlpDist(bodyBytes, bodyRef); worst > 1 {
			t.Fatalf("chain MoE body != device reference: worst %d ULP at elem %d", worst, at)
		}
		// SiLU is genuinely engaged: the GELU-activated body must differ.
		if worst, _ := bf16UlpDist(bodyBytes, geluBody); worst == 0 {
			t.Fatal("chain MoE body matches the GELU body — SiLU activation not engaged")
		}
	}

	t.Run("SharedGate_nil", func(t *testing.T) { run(t, nil) })
	t.Run("SharedGate_present", func(t *testing.T) { run(t, mk(D, 23)) })
}

// TestChainMoEBodyRoutingConsumed proves the routing is genuinely read from the device idx buffer:
// a body over one normed input must differ from a body over a different normed input (different
// experts + gate values), i.e. the gather is not frozen to a fixed selection.
func TestChainMoEBodyRoutingConsumed(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Fatalf("ensureInit: %v", err)
	}
	const nE, topK, D, expertDFF, sharedFF, gs, bits = 8, 2, 64, 32, 32, 32, 4
	gateKey := lthnGatherQMVKey{groupSize: gs, bits: bits, expertRows: expertDFF, fast: expertDFF%8 == 0 && D%512 == 0, batchedX: false}
	if _, ok := lthnGatherQMVPipeline(gateKey); !ok {
		t.Skip("lean gather kernel unavailable (custom library not loaded)")
	}
	mk := func(n, salt int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32((i*salt+11)%89-44) * 0.02
		}
		return s
	}
	gate, up, down := quantMoEExpertsFixture(t, nE, D, expertDFF, gs, bits)
	sGate := QuantWeight{GroupSize: gs, Bits: bits}
	sGate.Packed, sGate.Scales, sGate.Biases = quantizeProj(t, sharedFF, D, gs, bits, 17)
	sUp := QuantWeight{GroupSize: gs, Bits: bits}
	sUp.Packed, sUp.Scales, sUp.Biases = quantizeProj(t, sharedFF, D, gs, bits, 29)
	sDown := QuantWeight{GroupSize: gs, Bits: bits}
	sDown.Packed, sDown.Scales, sDown.Biases = quantizeProj(t, D, sharedFF, gs, bits, 41)
	routerBFbytes := f32sToBF16Bytes(mk(nE*D, 13))
	w := &moeChainWeights{
		routerBF: residentBytes(routerBFbytes), routerBacker: routerBFbytes,
		numExperts: nE, topK: topK, expertDFF: expertDFF, groupSize: gs, bits: bits,
		gate: gate, up: up, down: down, hasShared: true,
		sGate: sGate, sUp: sUp, sDown: sDown, sharedFF: sharedFF,
	}
	body := func(nBFbytes []byte) []byte {
		sc, err := newMoEChainScratch(D, nE, topK, expertDFF, sharedFF)
		if err != nil {
			t.Fatalf("newMoEChainScratch: %v", err)
		}
		var out []byte
		var berr error
		withAutoreleasePool(func() {
			cb := commandBufferFast(queue)
			enc := computeCommandEncoderFast(cb)
			tt := &chainTarget{enc: enc}
			berr = chainMoEBody(tt, sc, sharedBytes(nBFbytes), w, D)
			endEncodingFast(enc)
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
			out = make([]byte, D*bf16Size)
			copy(out, unsafe.Slice((*byte)(sc.bodyBF.Contents()), D*bf16Size))
		})
		if berr != nil {
			t.Fatalf("chainMoEBody: %v", berr)
		}
		return out
	}
	a := body(toBF16Bytes(mk(D, 5)))
	b := body(toBF16Bytes(mk(D, 61)))
	if worst, _ := bf16UlpDist(a, b); worst == 0 {
		t.Fatal("different normed inputs produced an identical MoE body — routing/gather not consumed")
	}
}
