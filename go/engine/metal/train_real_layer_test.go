// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"strings"
	"testing"
)

// train_real_layer_test.go gates the REAL-ARCH layer reference (#40 stage 2) by central finite
// differences — pure host, NO runtime gate (the whole point of the host-pure re-derivation: these
// gates cannot silently skip). Each gemma4 feature rung lands with its own FD sub-gate here before
// the next feature is written; the trainer wiring (stage 3) is allowed to accept only shapes whose
// every feature gate below is green.

// tinyRealLayer builds the tiny synthetic REAL layer the FD checks run on — base feature set
// (rung 1): pre-norm GQA attention, full standard rotary, global window, pre-norm gated-GELU MLP.
// Small weights keep the deep f32 chain numerically clean (the tinyTrainLayer precedent).
func tinyRealLayer() *RealTrainLayerF32 {
	const T, dModel, dFF, H, Hkv, d = 3, 8, 12, 2, 1, 4
	s := func(n, salt int) []float32 { return scaleSlice(syntheticFloat32(n, salt), 0.3) }
	return &RealTrainLayerF32{
		AttnNormW: syntheticFloat32(dModel, 11), WQ: s(H*d*dModel, 12), WK: s(Hkv*d*dModel, 13),
		WV: s(Hkv*d*dModel, 14), WO: s(dModel*H*d, 15),
		MLPNormW: syntheticFloat32(dModel, 16), WGate: s(dFF*dModel, 17), WUp: s(dFF*dModel, 18), WDown: s(dModel*dFF, 19),
		T: T, DModel: dModel, DFF: dFF, Heads: H, KVHeads: Hkv, HeadDim: d,
		RopeInvFreq: realRopeInvFreqs(d, 10000), RopePairHalf: d / 2, RopeScale: 1,
		AttnScale: 0.5, Window: 0, Eps: 1e-5,
	}
}

// realLayerFDCheck runs the shared FD gate for one layer/target: loss = Σ RealLayerForwardF32(h)·cot
// with the LoRA's effective weight substituted at target, and the analytic dA/dB (strided full
// coverage) and dH (sample) from RealLayerProjLoRABackwardF32 must match central differences —
// the same eps/tolerance bar as the #31 simplified-layer gate (train_lora_layer_test.go).
func realLayerFDCheck(t *testing.T, L *RealTrainLayerF32, target string) {
	t.Helper()
	const rank = 2
	scaling := float32(16.0 / rank)
	out, in, err := L.projDims(target)
	if err != nil {
		t.Fatalf("projDims: %v", err)
	}
	h := scaleSlice(syntheticFloat32(L.T*L.DModel, 21), 0.5)
	cot := syntheticFloat32(L.T*L.DModel, 22)
	a := scaleSlice(syntheticFloat32(rank*in, 23), 0.2)
	b := scaleSlice(syntheticFloat32(out*rank, 24), 0.2)

	frozen := L.projWeight(target)
	loss := func() float64 {
		eff, err := LoRAEffectiveWeightF32(frozen, a, b, out, in, rank, scaling)
		if err != nil {
			t.Fatalf("eff: %v", err)
		}
		saved := L.setProjWeight(target, eff)
		y, err := RealLayerForwardF32(h, L)
		L.setProjWeight(target, saved)
		if err != nil {
			t.Fatalf("forward: %v", err)
		}
		var s float64
		for i := range y {
			s += float64(y[i]) * float64(cot[i])
		}
		return s
	}

	dA, dB, dH, err := RealLayerProjLoRABackwardF32(cot, h, L, target, a, b, rank, scaling)
	if err != nil {
		t.Fatalf("RealLayerProjLoRABackwardF32: %v", err)
	}

	const eps = 1.0 / 512
	check := func(name string, params, grad []float32) {
		step := 1
		if len(params) > 12 {
			step = len(params) / 12
		}
		for i := 0; i < len(params); i += step {
			orig := params[i]
			params[i] = orig + eps
			lp := loss()
			params[i] = orig - eps
			lm := loss()
			params[i] = orig
			fd := (lp - lm) / (2 * eps)
			if math.Abs(fd-float64(grad[i])) > 2e-2*(1+math.Abs(fd)) {
				t.Errorf("%s[%d]: analytic %.5f vs finite-diff %.5f", name, i, grad[i], fd)
			}
		}
	}
	check("dA", a, dA)
	check("dB", b, dB)
	check("dH", h, dH)
	if !t.Failed() {
		t.Logf("%s: dA[%d] dB[%d] dH[%d] match finite differences through the real layer", target, len(dA), len(dB), len(dH))
	}
}

// realLayerTargets are the projection targets a dense layer carries — every FD gate runs all of them.
var realLayerTargets = []string{ProjQ, ProjK, ProjV, ProjO, ProjGate, ProjUp, ProjDown}

// TestRealLayerProjLoRABackwardF32 finite-difference-checks the BASE real-layer backward (rung 1:
// pre-norm GQA + standard rope + causal + gated-GELU MLP) for every canonical projection target —
// the host-pure foundation every later feature rung extends.
func TestRealLayerProjLoRABackwardF32(t *testing.T) {
	for _, target := range realLayerTargets {
		t.Run(target, func(t *testing.T) {
			realLayerFDCheck(t, tinyRealLayer(), target)
		})
	}
}

// TestRealLayerProjLoRABackwardF32_QKNorm is the rung-2 FD gate: the same seven-target check with
// gemma4's per-head QK-norm present (shared [HeadDim] weights on Q and K, between projection and
// rope — the station the simplified reference lacked). dH and every dA/dB must still match central
// differences with the extra norm Jacobian in the chain.
func TestRealLayerProjLoRABackwardF32_QKNorm(t *testing.T) {
	mk := func() *RealTrainLayerF32 {
		L := tinyRealLayer()
		// weights near 1 (the real checkpoints' shape) so the norm actually scales.
		qn := syntheticFloat32(L.HeadDim, 41)
		kn := syntheticFloat32(L.HeadDim, 43)
		for i := range qn {
			qn[i] = 1 + 0.3*qn[i]
			kn[i] = 1 + 0.3*kn[i]
		}
		L.QNormW, L.KNormW = qn, kn
		return L
	}
	for _, target := range realLayerTargets {
		t.Run(target, func(t *testing.T) {
			realLayerFDCheck(t, mk(), target)
		})
	}
	// a Q-only norm (K bare) exercises the asymmetric wiring: the two stations are independent.
	t.Run("q-norm-only/"+ProjK, func(t *testing.T) {
		L := mk()
		L.KNormW = nil
		realLayerFDCheck(t, L, ProjK)
	})
}

// TestRealLayerProjLoRABackwardF32_SandwichNorms is the rung-3 FD gate: the gemma4 post-attention
// and post-feed-forward norms sit on the BRANCH outputs before their residual adds (the
// encResidualMaybeNorm order — norm the branch, then add), stacked ON TOP of the QK-norm rung.
// Every target's dA/dB and dH must ride the two extra norm Jacobians.
func TestRealLayerProjLoRABackwardF32_SandwichNorms(t *testing.T) {
	mk := func() *RealTrainLayerF32 {
		L := tinyRealLayer()
		near1 := func(salt int, n int) []float32 {
			w := syntheticFloat32(n, salt)
			for i := range w {
				w[i] = 1 + 0.3*w[i]
			}
			return w
		}
		L.QNormW, L.KNormW = near1(41, L.HeadDim), near1(43, L.HeadDim)
		L.PostAttnNormW, L.PostFFNormW = near1(47, L.DModel), near1(51, L.DModel)
		return L
	}
	for _, target := range realLayerTargets {
		t.Run(target, func(t *testing.T) {
			realLayerFDCheck(t, mk(), target)
		})
	}
	// a post-attn-only sandwich (no post-FF) exercises the two stations independently.
	t.Run("post-attn-only/"+ProjDown, func(t *testing.T) {
		L := mk()
		L.PostFFNormW = nil
		realLayerFDCheck(t, L, ProjDown)
	})
}

// TestRealLayerProjLoRABackwardF32_ValueNorm is the rung-4a FD gate: gemma4's NO-SCALE per-head
// RMSNorm on V (Arch.ValueNorm — applied on every gemma4 attention layer) enters the chain between
// the value projection and the SDPA; all seven targets must ride its Jacobian.
func TestRealLayerProjLoRABackwardF32_ValueNorm(t *testing.T) {
	for _, target := range realLayerTargets {
		t.Run(target, func(t *testing.T) {
			L := tinyRealLayer()
			L.ValueNorm = true
			realLayerFDCheck(t, L, target)
		})
	}
}

// TestRealLayerProjLoRABackwardF32_KEqV is the rung-4b FD gate: the gemma4 K==V layer (nil WV —
// V rides the KEY projection's RAW output, copied before the k norms/rope, value-normed after).
// The k_proj target's dW must accumulate BOTH paths (through K's norm+rope+scores AND through V);
// the v_proj target is refused (the layer has no v_proj). QK-norms ride along so the copy point
// (pre-norm) is what the FD differentiates, not an approximation of it.
func TestRealLayerProjLoRABackwardF32_KEqV(t *testing.T) {
	mk := func() *RealTrainLayerF32 {
		L := tinyRealLayer()
		L.WV = nil
		L.ValueNorm = true
		near1 := func(salt, n int) []float32 {
			w := syntheticFloat32(n, salt)
			for i := range w {
				w[i] = 1 + 0.3*w[i]
			}
			return w
		}
		L.QNormW, L.KNormW = near1(41, L.HeadDim), near1(43, L.HeadDim)
		return L
	}
	for _, target := range []string{ProjQ, ProjK, ProjO, ProjGate, ProjUp, ProjDown} {
		t.Run(target, func(t *testing.T) {
			realLayerFDCheck(t, mk(), target)
		})
	}
	t.Run("v_proj-refused", func(t *testing.T) {
		L := mk()
		dout := make([]float32, L.T*L.DModel)
		h := make([]float32, L.T*L.DModel)
		_, _, _, err := RealLayerProjLoRABackwardF32(dout, h, L, ProjV, make([]float32, L.DModel), make([]float32, L.KVHeads*L.HeadDim), 1, 1)
		if err == nil {
			t.Fatal("a v_proj target on a K==V layer must be refused")
		}
		if !strings.Contains(err.Error(), "K==V") || !strings.Contains(err.Error(), ProjK) {
			t.Fatalf("the K==V refusal must explain the layer shape and point at %s; got: %s", ProjK, err.Error())
		}
	})
}

// TestRealLayerProjLoRABackwardF32_PLE is the rung-5 FD gate: the gemma4 per-layer-input gate
// (E2B/E4B) is the layer's LAST station — out = h + rms(WProj·(gelu(WGate·h)·pli), postNorm) —
// with FROZEN tower weights and a CONSTANT per-layer input. Every target's gradient now rides the
// gate branch's extra path into the layer output; the FD gate differentiates through the full
// tower. Run on the full E2B-shaped stack (QK-norms + sandwich norms + value-norm + PLE together).
func TestRealLayerProjLoRABackwardF32_PLE(t *testing.T) {
	mk := func() *RealTrainLayerF32 {
		L := tinyRealLayer()
		const pliDim = 4
		near1 := func(salt, n int) []float32 {
			w := syntheticFloat32(n, salt)
			for i := range w {
				w[i] = 1 + 0.3*w[i]
			}
			return w
		}
		L.QNormW, L.KNormW = near1(41, L.HeadDim), near1(43, L.HeadDim)
		L.PostAttnNormW, L.PostFFNormW = near1(47, L.DModel), near1(51, L.DModel)
		L.ValueNorm = true
		L.PLIDim = pliDim
		L.PLEGateW = scaleSlice(syntheticFloat32(pliDim*L.DModel, 61), 0.3)
		L.PLEProjW = scaleSlice(syntheticFloat32(L.DModel*pliDim, 62), 0.3)
		L.PLEPostNormW = near1(63, L.DModel)
		L.PLEInput = scaleSlice(syntheticFloat32(L.T*pliDim, 64), 0.5)
		return L
	}
	for _, target := range realLayerTargets {
		t.Run(target, func(t *testing.T) {
			realLayerFDCheck(t, mk(), target)
		})
	}
	// the per-layer input is a CONSTANT of the layer: the FD gate above differentiates every
	// TRAINABLE path; this pin proves the forward actually consumes pli (a zeroed pli changes
	// the output), so the constant is genuinely in the chain rather than dead.
	t.Run("pli-is-consumed", func(t *testing.T) {
		L := mk()
		h := scaleSlice(syntheticFloat32(L.T*L.DModel, 21), 0.5)
		y1, err := RealLayerForwardF32(h, L)
		if err != nil {
			t.Fatalf("forward: %v", err)
		}
		L.PLEInput = make([]float32, L.T*L.PLIDim)
		y2, err := RealLayerForwardF32(h, L)
		if err != nil {
			t.Fatalf("zero-pli forward: %v", err)
		}
		same := true
		for i := range y1 {
			if y1[i] != y2[i] {
				same = false
				break
			}
		}
		if same {
			t.Fatal("zeroing the per-layer input did not change the forward — the PLE tower is not wired")
		}
	})
	// a partial tower is a caller bug, refused loudly.
	t.Run("partial-tower-refused", func(t *testing.T) {
		L := mk()
		L.PLEProjW = nil
		dout := make([]float32, L.T*L.DModel)
		h := make([]float32, L.T*L.DModel)
		out, in, _ := tinyRealLayer().projDims(ProjDown)
		if _, _, _, err := RealLayerProjLoRABackwardF32(dout, h, L, ProjDown, make([]float32, in), make([]float32, out), 1, 1); err == nil {
			t.Fatal("a partial PLE tower must be refused")
		}
	})
}

// TestRealRopeForwardF32_Good pins the reference rope against the PROVEN host rope
// (ropeForwardF32, the train_backward.go convention the engine's rope kernel was gated against) on
// the configurations where the two must coincide — full rotary and standard partial rotary (pairs
// (j, j+rotDim/2), base-derived spectrum) — and pins the proportional GLOBAL pairing (pairs
// (j, j+headDim/2), spectrum zero-padded past rotDim/2 — rope_freqs.go proportionalRopePeriods)
// against a hand-derived rotation. Host-pure; the rotation the FD gates then differentiate.
func TestRealRopeForwardF32_Good(t *testing.T) {
	const heads, d = 2, 8
	x := scaleSlice(syntheticFloat32(heads*d, 5), 0.7)
	for _, pos := range []int{0, 1, 5} {
		// full rotary: pairHalf = d/2, full base-derived spectrum.
		got := realRopeForwardF32(x, pos, heads, d, d/2, realRopeInvFreqs(d, 10000), 1)
		want := ropeForwardF32(x, pos, heads, d, d, 10000)
		assertFloat32Near(t, "full-rotary rope", got, want, 1e-6)

		// standard partial rotary (the sliding-layer form): rotate the first rotDim dims,
		// pairs (j, j+rotDim/2) — pairHalf = rotDim/2, spectrum over rotDim.
		const rotDim = 4
		got = realRopeForwardF32(x, pos, heads, d, rotDim/2, realRopeInvFreqs(rotDim, 10000), 1)
		want = ropeForwardF32(x, pos, heads, d, rotDim, 10000)
		assertFloat32Near(t, "partial-rotary rope", got, want, 1e-6)
	}

	// proportional pairing (the gemma4 global form): pairs (j, j+d/2) over the WHOLE head with
	// only the first rotDim/2 pairs rotated — inv-freq base^(−2j/headDim), the exponent over the
	// FULL head dim (proportionalRopePeriods). Hand-derive pair (0, d/2) at pos 1.
	const rotDim = 4
	inv := make([]float32, rotDim/2)
	for j := range inv {
		inv[j] = float32(math.Pow(32, -2*float64(j)/float64(d)))
	}
	got := realRopeForwardF32(x, 1, heads, d, d/2, inv, 1)
	for h := range heads {
		off := h * d
		// rotated pairs: (0, 4) and (1, 5); dims 2, 3, 6, 7 pass through untouched.
		for _, j := range []int{2, 3, 6, 7} {
			if got[off+j] != x[off+j] {
				t.Fatalf("proportional rope: unrotated dim %d must pass through (head %d)", j, h)
			}
		}
		ang := 1.0 * math.Pow(32, 0) // pair 0: inv-freq 32^0 = 1
		c, s := float32(math.Cos(ang)), float32(math.Sin(ang))
		wantA := x[off]*c - x[off+d/2]*s
		wantB := x[off]*s + x[off+d/2]*c
		if math.Abs(float64(got[off]-wantA)) > 1e-6 || math.Abs(float64(got[off+d/2]-wantB)) > 1e-6 {
			t.Fatalf("proportional rope pair (0,%d) head %d: got (%v,%v) want (%v,%v)", d/2, h, got[off], got[off+d/2], wantA, wantB)
		}
	}

	// rope position scale: angle = scale·pos·invFreq — scale 0.5 at pos 2 equals scale 1 at pos 1.
	gotScaled := realRopeForwardF32(x, 2, heads, d, d/2, realRopeInvFreqs(d, 10000), 0.5)
	want := realRopeForwardF32(x, 1, heads, d, d/2, realRopeInvFreqs(d, 10000), 1)
	assertFloat32Near(t, "rope position scale", gotScaled, want, 1e-6)
}

// TestRealLayerProjLoRABackwardF32_RopeVariants is the rung-6a FD gate: the rotation is orthogonal
// and its backward is the inverse rotation — PROVEN per variant rather than assumed. Each rope
// shape the engine serves (standard partial — the sliding-layer form; proportional full-head
// pairing — the gemma4 global form; a non-unit position scale) gets its own FD run over the
// attention-side targets.
func TestRealLayerProjLoRABackwardF32_RopeVariants(t *testing.T) {
	variants := []struct {
		name string
		mut  func(L *RealTrainLayerF32)
	}{
		{"partial-standard", func(L *RealTrainLayerF32) {
			// rotate the first 2 of 4 dims: pairHalf 1, spectrum over rotDim 2.
			L.RopePairHalf = 1
			L.RopeInvFreq = realRopeInvFreqs(2, 10000)
		}},
		{"proportional-global", func(L *RealTrainLayerF32) {
			// pairs (j, j+d/2) over the whole head, only pair 0 rotated (rotDim/2 = 1 of 2),
			// exponent over the FULL head dim — the proportionalRopePeriods shape.
			L.RopePairHalf = L.HeadDim / 2
			inv := make([]float32, 1)
			inv[0] = 1 // base^0
			L.RopeInvFreq = inv
		}},
		{"rope-scale", func(L *RealTrainLayerF32) {
			L.RopeScale = 0.25
		}},
	}
	for _, v := range variants {
		for _, target := range []string{ProjQ, ProjK, ProjV, ProjO} {
			t.Run(v.name+"/"+target, func(t *testing.T) {
				L := tinyRealLayer()
				v.mut(L)
				realLayerFDCheck(t, L, target)
			})
		}
	}
}

// TestRealLayerProjLoRABackwardF32_SlidingWindow is the rung-6b FD gate: a sliding layer attends
// only the last Window positions (row i sees [max(0, i−Window+1), i] — the ring cache's live
// window, hostArchQuantReference's first = len−window). The mask changes which probabilities exist
// at all, so every target's gradient is FD-checked under a window that genuinely bites (T=4,
// Window=2), and a forward pin proves the mask semantics against a hand-built global/window pair.
func TestRealLayerProjLoRABackwardF32_SlidingWindow(t *testing.T) {
	for _, target := range realLayerTargets {
		t.Run(target, func(t *testing.T) {
			L := tinyRealLayer()
			L.T = 4
			L.Window = 2
			realLayerFDCheck(t, L, target)
		})
	}
	// mask-semantics pin: with Window=2, row 2 must IGNORE position 0 — its probabilities over
	// {1,2} renormalise, so the row differs from the global layer's; row 1 (window not yet full)
	// must MATCH the global layer's row exactly.
	t.Run("window-semantics", func(t *testing.T) {
		L := tinyRealLayer()
		L.T = 3
		g := hostSDPAProbsF32(scaleSlice(syntheticFloat32(L.T*L.HeadDim, 5), 0.5), scaleSlice(syntheticFloat32(L.T*L.HeadDim, 6), 0.5), L)
		L.Window = 2
		w := hostSDPAProbsF32(scaleSlice(syntheticFloat32(L.T*L.HeadDim, 5), 0.5), scaleSlice(syntheticFloat32(L.T*L.HeadDim, 6), 0.5), L)
		T := L.T
		if w[2*T+0] != 0 {
			t.Fatalf("window 2: row 2 must not attend position 0 (got P=%v)", w[2*T+0])
		}
		for j := range T {
			if g[1*T+j] != w[1*T+j] {
				t.Fatalf("window 2: row 1 (window not yet full) must equal the global row: j=%d %v vs %v", j, g[1*T+j], w[1*T+j])
			}
		}
		if g[2*T+1] == w[2*T+1] {
			t.Fatal("window 2: row 2 must renormalise over {1,2} and differ from the global row")
		}
	})
}

// TestRealLayerProjLoRABackwardF32_LayerScalar is the rung-6c FD gate: gemma4's per-layer output
// scalar multiplies the layer's final hidden AFTER the PLE gate (the arch executor's op order);
// the backward scales the incoming gradient before everything else. FD-checked with a non-trivial
// scalar; a forward pin proves the placement.
func TestRealLayerProjLoRABackwardF32_LayerScalar(t *testing.T) {
	for _, target := range realLayerTargets {
		t.Run(target, func(t *testing.T) {
			L := tinyRealLayer()
			L.LayerScalar = 0.75
			realLayerFDCheck(t, L, target)
		})
	}
	t.Run("scalar-placement", func(t *testing.T) {
		L := tinyRealLayer()
		h := scaleSlice(syntheticFloat32(L.T*L.DModel, 21), 0.5)
		y1, err := RealLayerForwardF32(h, L)
		if err != nil {
			t.Fatalf("forward: %v", err)
		}
		L.LayerScalar = 0.75
		y2, err := RealLayerForwardF32(h, L)
		if err != nil {
			t.Fatalf("scaled forward: %v", err)
		}
		for i := range y1 {
			if math.Abs(float64(y2[i]-0.75*y1[i])) > 1e-6 {
				t.Fatalf("layer scalar must multiply the FINAL hidden: y2[%d]=%v want %v", i, y2[i], 0.75*y1[i])
			}
		}
	})
}

// TestRealLayerProjLoRABackwardF32_Bad: an unknown target, a nil layer, a wrong-shaped upstream
// gradient, and a broken geometry are refused before any work.
func TestRealLayerProjLoRABackwardF32_Bad(t *testing.T) {
	L := tinyRealLayer()
	const rank = 1
	out, in, _ := L.projDims(ProjDown)
	a := make([]float32, rank*in)
	b := make([]float32, out*rank)
	dout := make([]float32, L.T*L.DModel)
	h := make([]float32, L.T*L.DModel)

	_, _, _, err := RealLayerProjLoRABackwardF32(dout, h, L, "lm_head", a, b, rank, 1)
	if err == nil {
		t.Fatal("a non-layer target must be refused")
	}
	if !strings.Contains(err.Error(), "lm_head") || !strings.Contains(err.Error(), ProjDown) {
		t.Fatalf("unknown-target error must name the request and the supported set; got: %s", err.Error())
	}
	if _, _, _, err := RealLayerProjLoRABackwardF32(dout, h, nil, ProjDown, a, b, rank, 1); err == nil {
		t.Fatal("a nil layer must be refused")
	}
	if _, _, _, err := RealLayerProjLoRABackwardF32(dout[:1], h, L, ProjDown, a, b, rank, 1); err == nil {
		t.Fatal("a wrong-shaped dout must be refused")
	}
	bad := tinyRealLayer()
	bad.KVHeads = 0
	if _, _, _, err := RealLayerProjLoRABackwardF32(dout, h, bad, ProjDown, a, b, rank, 1); err == nil {
		t.Fatal("a zero KVHeads geometry must be refused")
	}
	bad2 := tinyRealLayer()
	bad2.RopePairHalf = bad2.HeadDim // > HeadDim/2
	if _, _, _, err := RealLayerProjLoRABackwardF32(dout, h, bad2, ProjDown, a, b, rank, 1); err == nil {
		t.Fatal("a rope pairing wider than the head must be refused")
	}
	bad3 := tinyRealLayer()
	bad3.QNormW = make([]float32, bad3.HeadDim+1)
	if _, _, _, err := RealLayerProjLoRABackwardF32(dout, h, bad3, ProjDown, a, b, rank, 1); err == nil {
		t.Fatal("a wrong-length QNormW must be refused")
	}
	bad4 := tinyRealLayer()
	bad4.PostAttnNormW = make([]float32, bad4.DModel-1)
	if _, _, _, err := RealLayerProjLoRABackwardF32(dout, h, bad4, ProjDown, a, b, rank, 1); err == nil {
		t.Fatal("a wrong-length PostAttnNormW must be refused")
	}
}

// TestRealLayerForwardF32_Good pins the base forward against the PROVEN simplified-layer block
// forwards on the exact configuration where the two must coincide (full rotary, no window, no
// gemma4 extras): the host-pure re-derivation may differ from the steel-GEMM chain only by
// accumulation noise. Runtime-gated — the comparison target is the GPU-backed reference.
func TestRealLayerForwardF32_Good(t *testing.T) {
	requireNativeRuntime(t)
	L := tinyRealLayer()
	h := scaleSlice(syntheticFloat32(L.T*L.DModel, 21), 0.5)
	got, err := RealLayerForwardF32(h, L)
	if err != nil {
		t.Fatalf("RealLayerForwardF32: %v", err)
	}
	attnOut, err := MultiHeadAttnBlockForwardF32(h, L.AttnNormW, L.WQ, L.WK, L.WV, L.WO, L.T, L.DModel, L.Heads, L.KVHeads, L.HeadDim, L.HeadDim, 10000, L.AttnScale, L.Eps, true)
	if err != nil {
		t.Fatalf("attn block fwd: %v", err)
	}
	want, err := MLPBlockForwardF32(attnOut, L.MLPNormW, L.WGate, L.WUp, L.WDown, L.T, L.DModel, L.DFF, L.Eps)
	if err != nil {
		t.Fatalf("mlp block fwd: %v", err)
	}
	assertFloat32Near(t, "real-vs-simplified forward", got, want, 1e-4)
}

// TestRealLayerForwardF32_Bad: shape refusals at the forward entry.
func TestRealLayerForwardF32_Bad(t *testing.T) {
	L := tinyRealLayer()
	if _, err := RealLayerForwardF32(make([]float32, 1), L); err == nil {
		t.Fatal("a wrong-shaped h must be refused")
	}
	if _, err := RealLayerForwardF32(nil, nil); err == nil {
		t.Fatal("a nil layer must be refused")
	}
}
