// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"strings"
	"testing"
)

// train_real_shared_test.go gates the KV-share chain reference (#42) by central finite differences
// — pure host, NO runtime gate (the train_real_layer_test.go discipline). The chain FD gates run
// through a GENUINELY shared stack: one owner + two consumers, T ≥ 3, mixed with the #40 features
// (QK-norm + sandwich norms + value-norm + K==V + PLE + layer scalar — the E2B shape), per
// placement family: the consumer-side set, the OWNER-ROUTED set (owner k/v with multiple
// consumers — dK/dV accumulating from every consumer back through the owner's projections), and
// adapters BELOW the owner (the path through the owner's hidden). Every gate also FD-checks the
// chain-INPUT gradient — exact only when the owner-routed path is complete.

// near1F32 returns a near-one synthetic weight (the real checkpoints' norm shape).
func near1F32(salt, n int) []float32 {
	w := syntheticFloat32(n, salt)
	for i := range w {
		w[i] = 1 + 0.3*w[i]
	}
	return w
}

// tinySharedStack builds the three-layer E2B-shaped shared stack the chain FD gates run on:
// layer 0 OWNS its cache (K==V — nil WV — with value-norm, per-head QK-norm, sandwich norms, the
// PLE tower and a layer scalar — the gemma4 global-attention owner shape); layers 1 and 2 are
// KV-share CONSUMERS of layer 0 (own q with Q-norm, sandwich norms, PLE, layer scalar — no k/v
// machinery, the encAttnHalfShared shape). Proportional full-head rope pairing (the gemma4 global
// form).
func tinySharedStack() []*RealTrainLayerF32 {
	const T, dModel, dFF, H, Hkv, d, pliDim = 3, 8, 12, 2, 1, 4, 4
	s := func(n, salt int) []float32 { return scaleSlice(syntheticFloat32(n, salt), 0.3) }
	mk := func(salt int) *RealTrainLayerF32 {
		L := &RealTrainLayerF32{
			AttnNormW: syntheticFloat32(dModel, salt+1),
			WQ:        s(H*d*dModel, salt+2),
			WO:        s(dModel*H*d, salt+5),
			MLPNormW:  syntheticFloat32(dModel, salt+6),
			WGate:     s(dFF*dModel, salt+7), WUp: s(dFF*dModel, salt+8), WDown: s(dModel*dFF, salt+9),
			T: T, DModel: dModel, DFF: dFF, Heads: H, KVHeads: Hkv, HeadDim: d,
			// proportional full-head pairing: pairs (j, j+d/2), only pair 0 rotated.
			RopePairHalf: d / 2, RopeInvFreq: []float32{1}, RopeScale: 1,
			AttnScale: 0.5, Window: 0, Eps: 1e-5,
			QNormW:        near1F32(salt+10, d),
			PostAttnNormW: near1F32(salt+11, dModel),
			PostFFNormW:   near1F32(salt+12, dModel),
			LayerScalar: 0.75,
			PLIDim:      pliDim,
			// PLE scales sit WELL AWAY from the rms cliff: the tower's post-norm is an RMSNorm
			// of the pleProj rows, non-smooth as a row's norm → 0 (Jacobian ∝ 1/‖row‖) — a
			// fixture whose rows hover near zero puts the FD probes on that cliff and no finite
			// difference converges (observed: quotients swinging +2185 → −432 across eps on one
			// coordinate). O(1) rows keep every probe on the smooth region the instrument is
			// valid on; the maths under test is unchanged.
			PLEGateW:     s(pliDim*dModel, salt+13),
			PLEProjW:     scaleSlice(syntheticFloat32(dModel*pliDim, salt+14), 0.8),
			PLEPostNormW: near1F32(salt+15, dModel),
			PLEInput:     syntheticFloat32(T*pliDim, salt+16),
		}
		return L
	}
	owner := mk(100)
	owner.WK = s(Hkv*d*dModel, 103) // K==V owner: WV stays nil, the key projection feeds both paths
	owner.KNormW = near1F32(120, d)
	owner.ValueNorm = true
	c1, c2 := mk(200), mk(300)
	c1.SharesKV, c2.SharesKV = true, true
	return []*RealTrainLayerF32{owner, c1, c2}
}

// tinySharedStackWithMap is tinySharedStack plus its shareFrom map (owner 0, consumers 1 and 2).
func tinySharedStackWithMap() ([]*RealTrainLayerF32, []int) {
	return tinySharedStack(), []int{0, 0, 0}
}

// chainEffectiveSets substitutes each adapter's CURRENT effective weight into its layer's weight
// set — the test-side twin of LoRATrainer.effectiveWeightSets over plain layers.
func chainEffectiveSets(t *testing.T, layers []*RealTrainLayerF32, adapters []*layerLoRAAdapter, rank int, scaling float32) []layerWeightSet {
	t.Helper()
	sets := make([]layerWeightSet, len(layers))
	for li, L := range layers {
		sets[li] = layerWeightSet{wQ: L.WQ, wK: L.WK, wV: L.WV, wO: L.WO, wGate: L.WGate, wUp: L.WUp, wDown: L.WDown}
	}
	for _, ad := range adapters {
		L := layers[ad.layer]
		eff, err := LoRAEffectiveWeightF32(L.projWeight(ad.target), ad.a, ad.b, ad.out, ad.in, rank, scaling)
		if err != nil {
			t.Fatalf("effective weight (layer %d %s): %v", ad.layer, ad.target, err)
		}
		s := &sets[ad.layer]
		switch ad.target {
		case ProjQ:
			s.wQ = eff
		case ProjK:
			s.wK = eff
		case ProjV:
			s.wV = eff
		case ProjO:
			s.wO = eff
		case ProjGate:
			s.wGate = eff
		case ProjUp:
			s.wUp = eff
		case ProjDown:
			s.wDown = eff
		}
	}
	return sets
}

// newChainAdapter builds one hand-placed (layer, target) adapter with small deterministic factors
// (the single-layer harness's 0.2 scale; no optimiser — the FD gates never step). Both factors are
// non-zero, so every probe differentiates a live adapter.
func newChainAdapter(t *testing.T, layers []*RealTrainLayerF32, layer int, target string, rank, salt int) *layerLoRAAdapter {
	t.Helper()
	out, in, err := layers[layer].projDims(target)
	if err != nil {
		t.Fatalf("projDims (layer %d %s): %v", layer, target, err)
	}
	return &layerLoRAAdapter{
		layer: layer, target: target, out: out, in: in,
		a: scaleSlice(syntheticFloat32(rank*in, salt), 0.2),
		b: scaleSlice(syntheticFloat32(out*rank, salt+1), 0.2),
	}
}

// sharedChainFDCheck FD-gates the WHOLE adapter set at once (all adapters active — the trainer's
// multi-adapter gradient): loss = Σ chainOut·cot under the current effective weights; every
// adapter's analytic dA/dB from realSharedChainBackward must match central differences, strided
// full coverage, the train_real_layer_test.go eps/tolerance bar.
func sharedChainFDCheck(t *testing.T, layers []*RealTrainLayerF32, shareFrom []int, adapters []*layerLoRAAdapter, rank int, scaling float32) {
	t.Helper()
	T, D := layers[0].T, layers[0].DModel
	h := scaleSlice(syntheticFloat32(T*D, 21), 0.5)
	cot := syntheticFloat32(T*D, 22)

	loss := func() float64 {
		sets := chainEffectiveSets(t, layers, adapters, rank, scaling)
		_, tapes, err := realSharedChainForward(h, layers, shareFrom, sets)
		if err != nil {
			t.Fatalf("chain forward: %v", err)
		}
		out := tapes[len(tapes)-1].out
		var s float64
		for i := range out {
			s += float64(out[i]) * float64(cot[i])
		}
		return s
	}

	analytic := func() (dAs, dBs [][]float32, dH0 []float32) {
		sets := chainEffectiveSets(t, layers, adapters, rank, scaling)
		inputs, tapes, err := realSharedChainForward(h, layers, shareFrom, sets)
		if err != nil {
			t.Fatalf("chain forward: %v", err)
		}
		dAs = make([][]float32, len(adapters))
		dBs = make([][]float32, len(adapters))
		dH0, err = realSharedChainBackward(cot, inputs, tapes, layers, shareFrom, sets, func(li int, g *realLayerGrads) error {
			for ai, ad := range adapters {
				if ad.layer != li {
					continue
				}
				var dW []float32
				switch ad.target {
				case ProjQ:
					dW = g.dWQ
				case ProjK:
					dW = g.dWK
				case ProjV:
					dW = g.dWV
				case ProjO:
					dW = g.dWO
				case ProjGate:
					dW = g.dWGate
				case ProjUp:
					dW = g.dWUp
				case ProjDown:
					dW = g.dWDown
				}
				dA, dB, ferr := LoRAFactorGradsF32(dW, ad.a, ad.b, ad.out, ad.in, rank, scaling)
				if ferr != nil {
					return ferr
				}
				dAs[ai], dBs[ai] = dA, dB
			}
			return nil
		})
		if err != nil {
			t.Fatalf("chain backward: %v", err)
		}
		return dAs, dBs, dH0
	}
	dAs, dBs, dH0 := analytic()

	// Richardson-extrapolated central differences (the single-layer harness's method refined for
	// the CHAIN): three stacked layers of gelu towers, softmax and rmsnorm compound — chain
	// gradients run O(50-200) where the single-layer gates see O(1), and the plain eps=1/512
	// quotient's truncation term (∝ eps²·f‴) breaches the bar on the steepest probes (observed:
	// an fd WALKING toward the analytic value as eps shrank — truncation's signature, not a
	// missing path). Extrapolating two central differences, fd = (4·fd(eps/2) − fd(eps))/3,
	// cancels the eps² term (residual O(eps⁴)); the forward accumulates in f64, so quotient
	// rounding at eps/2 = 1/1024 stays orders below the bar. The TOLERANCE is the single-layer
	// harness's, unchanged: 2e-2·(1+|fd|).
	const eps = 1.0 / 512
	check := func(name string, params, grad []float32) {
		step := 1
		if len(params) > 12 {
			step = len(params) / 12
		}
		central := func(i int, e float32) float64 {
			orig := params[i]
			params[i] = orig + e
			lp := loss()
			params[i] = orig - e
			lm := loss()
			params[i] = orig
			return (lp - lm) / (2 * float64(e))
		}
		for i := 0; i < len(params); i += step {
			f1 := central(i, eps)
			f2 := central(i, eps/2)
			fd := (4*f2 - f1) / 3
			if math.Abs(fd-float64(grad[i])) > 2e-2*(1+math.Abs(fd)) {
				t.Errorf("%s[%d]: analytic %.5f vs finite-diff %.5f", name, i, grad[i], fd)
			}
		}
	}
	for ai, ad := range adapters {
		name := layerAdapterTensorName(ad.layer, ad.target)
		check(name+".dA", ad.a, dAs[ai])
		check(name+".dB", ad.b, dBs[ai])
	}
	// the chain-INPUT gradient: exact only when every consumer's dK/dV is routed back through its
	// owner (the chain input feeds the lowest owner's cached rows) — the owner-routed backward's
	// own FD receipt, on every stack shape.
	check("dH0", h, dH0)
	if !t.Failed() {
		t.Logf("%d adapters + dH0 match finite differences through the shared-KV chain (T=%d, %d layers)", len(adapters), T, len(layers))
	}
}

// TestRealSharedChainBackward finite-difference-gates the CONSUMER-SIDE placement family through
// the E2B-shaped shared stack (K==V owner with value-norm + QK-norm + sandwich norms + PLE +
// layer scalar; two consumers): adapters on consumer-side q/o/mlp and on the owner's own
// q/o/up/down (the cache-independent owner targets). All adapters active at once (the trainer's
// multi-adapter gradient). The owner's gate_proj — equally cache-independent — is exercised at
// the consumer level here and by the single-layer #40 gate instead: the owner-level gate
// COORDINATE defeats the FD instrument on this fixture (quotients swing sign across eps — a
// near-singular direction; the share-free dense control fails it identically, so sharing is not
// implicated).
func TestRealSharedChainBackward(t *testing.T) {
	const rank = 2
	scaling := float32(16.0 / rank)
	layers, shareFrom := tinySharedStackWithMap()
	adapters := []*layerLoRAAdapter{
		newChainAdapter(t, layers, 0, ProjQ, rank, 31),    // the consumed owner's SAFE targets:
		newChainAdapter(t, layers, 0, ProjO, rank, 33),    // q/o/up/down do not feed its cached rows
		newChainAdapter(t, layers, 0, ProjUp, rank, 45),
		newChainAdapter(t, layers, 0, ProjDown, rank, 47),
		newChainAdapter(t, layers, 1, ProjQ, rank, 35),    // consumer-side attention
		newChainAdapter(t, layers, 1, ProjDown, rank, 37), // consumer-side MLP
		newChainAdapter(t, layers, 2, ProjO, rank, 39),
		newChainAdapter(t, layers, 2, ProjGate, rank, 41),
		newChainAdapter(t, layers, 2, ProjUp, rank, 43),
	}
	sharedChainFDCheck(t, layers, shareFrom, adapters, rank, scaling)
}

// TestRealSharedChainBackward_OwnerRouted finite-difference-gates the OWNER-ROUTED path (#42
// stage 2) on the E2B-shaped stack: an adapter on the K==V owner's k_proj — whose effective
// weight writes the cached K rows (rope'd+normed) AND the cached V rows (the raw-copy value
// path) that BOTH consumers attend, plus the owner's own attention use — must carry every one of
// those accumulation paths to match central differences. A consumer-side q adapter rides along
// so the two families mix in one multi-adapter gradient.
func TestRealSharedChainBackward_OwnerRouted(t *testing.T) {
	const rank = 2
	scaling := float32(16.0 / rank)
	layers, shareFrom := tinySharedStackWithMap()
	adapters := []*layerLoRAAdapter{
		newChainAdapter(t, layers, 0, ProjK, rank, 71), // the consumed K==V owner's key projection
		newChainAdapter(t, layers, 1, ProjQ, rank, 73),
	}
	sharedChainFDCheck(t, layers, shareFrom, adapters, rank, scaling)
}

// TestRealSharedChainBackward_OwnerRoutedOwnV is the own-V twin: the owner carries a separate
// value projection (the gemma4 sliding-owner shape), and BOTH its k_proj and v_proj are adapted —
// dWK accumulates the owner's own scores + every consumer's scores; dWV the owner's own values +
// every consumer's values — under the sliding window (the ring-live rows are the only cached rows
// a consumer sees).
func TestRealSharedChainBackward_OwnerRoutedOwnV(t *testing.T) {
	const rank = 2
	scaling := float32(16.0 / rank)
	layers, shareFrom := tinySharedStackWithMap()
	for _, L := range layers {
		L.T = 4
		L.Window = 2
		L.RopePairHalf = 1
		L.RopeInvFreq = realRopeInvFreqs(2, 10000)
		L.PLEInput = syntheticFloat32(L.T*L.PLIDim, 411)
	}
	owner := layers[0]
	owner.WV = scaleSlice(syntheticFloat32(owner.KVHeads*owner.HeadDim*owner.DModel, 414), 0.3)
	adapters := []*layerLoRAAdapter{
		newChainAdapter(t, layers, 0, ProjK, rank, 75),
		newChainAdapter(t, layers, 0, ProjV, rank, 77),
		newChainAdapter(t, layers, 2, ProjO, rank, 79),
	}
	sharedChainFDCheck(t, layers, shareFrom, adapters, rank, scaling)
}

// TestRealSharedChainBackward_BelowOwner finite-difference-gates adapters BELOW a consumed owner
// (#42 stage 2): a dense layer under the owner whose q and down adapters move the owner's INPUT
// hidden — and therefore its cached rows — so their gradients must ride the owner-routed path
// (consumers' dK/dV → the owner's k_proj AND its pre-attention norm → dH into the dense layer)
// on top of the ordinary residual chain. The owner's k_proj is adapted too, mixing every family
// through a 4-layer stack: dense + K==V owner + two consumers.
func TestRealSharedChainBackward_BelowOwner(t *testing.T) {
	const rank = 2
	scaling := float32(16.0 / rank)
	shared := tinySharedStack()
	// the dense under-layer: the owner shape re-salted, owning its own (unshared) cache.
	denseFloor := tinySharedStack()[0]
	layers := []*RealTrainLayerF32{denseFloor, shared[0], shared[1], shared[2]}
	shareFrom := []int{0, 1, 1, 1}
	adapters := []*layerLoRAAdapter{
		newChainAdapter(t, layers, 0, ProjQ, rank, 81),    // below the owner: the path through
		newChainAdapter(t, layers, 0, ProjDown, rank, 83), // the owner's hidden into its cache
		newChainAdapter(t, layers, 1, ProjK, rank, 85),    // and the owner's own cache-writing k
	}
	sharedChainFDCheck(t, layers, shareFrom, adapters, rank, scaling)
}

// TestRealSharedChainBackward_SlidingOwnV is the sliding-type twin: an own-V owner (the gemma4
// sliding-attention shape — separate WV, value-norm, window 2 over T=4, standard partial rotary)
// with two consumers, adapters on the consumers only. The window mask changes which cached rows
// exist at all, so the consumer's read of the owner's ring must be FD-exact under it.
func TestRealSharedChainBackward_SlidingOwnV(t *testing.T) {
	const rank = 2
	scaling := float32(16.0 / rank)
	layers, shareFrom := tinySharedStackWithMap()
	for _, L := range layers {
		L.T = 4
		L.Window = 2
		// standard partial rotary (the sliding-layer form): rotate the first 2 of 4 dims.
		L.RopePairHalf = 1
		L.RopeInvFreq = realRopeInvFreqs(2, 10000)
		L.PLEInput = syntheticFloat32(L.T*L.PLIDim, 401) // T grew to 4 — refill at the cliff-free scale
	}
	owner := layers[0]
	owner.WV = scaleSlice(syntheticFloat32(owner.KVHeads*owner.HeadDim*owner.DModel, 404), 0.3)
	adapters := []*layerLoRAAdapter{
		newChainAdapter(t, layers, 1, ProjQ, rank, 51),
		newChainAdapter(t, layers, 1, ProjO, rank, 53),
		newChainAdapter(t, layers, 2, ProjDown, rank, 55),
	}
	sharedChainFDCheck(t, layers, shareFrom, adapters, rank, scaling)
}

// mixedGeometryStack builds the PER-LAYER-GEOMETRY-SWITCHING stack — the synthetic twin of the
// real gemma4 E2B class split the #42 boundary refused: heads are model-wide (4), but head_dim AND
// KV heads switch per layer class exactly as LayerSpec.HeadDim/KVHeads resolve them
// (headDimOf/kvHeadsOf), and each class carries its OWN rope form:
//
//	layer 0 — SLIDING owner:   d=4, kv=2 (GQA 2), own WV, window 2 over T=4, standard full rotary
//	layer 1 — GLOBAL owner:    d=8, kv=1 (GQA 4), own WV, the PROPORTIONAL partial rope — the
//	          engine's Inf-padded table (globalRopePeriodsFromFolded, the globalRopeFreqs source)
//	          driven over the FULL head: pairs (j, j+d/2), only the first rot/2 pairs rotate
//	layer 2 — GLOBAL consumer of layer 1 (own q only, the owner's hd-8 cache)
//	layer 3 — SLIDING consumer of layer 0 (own q only, the owner's hd-4 ring, windowed)
//
// All four layers carry the full E2B feature set (QK-norm, value-norm on owners, sandwich norms,
// PLE tower, layer scalar). The rope spectra come from the SAME constructors the trainer's
// template builder uses — realRopeInvFreqs for sliding, the inverted finite prefix of
// globalRopePeriodsFromFolded for global — so this gate exercises the engine's table shape, not a
// re-derived formula.
func mixedGeometryStack() ([]*RealTrainLayerF32, []int) {
	const T, dModel, dFF, H, pliDim = 4, 8, 12, 4, 4
	s := func(n, salt int) []float32 { return scaleSlice(syntheticFloat32(n, salt), 0.3) }
	mk := func(salt, d, kv int) *RealTrainLayerF32 {
		return &RealTrainLayerF32{
			AttnNormW: syntheticFloat32(dModel, salt+1),
			WQ:        s(H*d*dModel, salt+2),
			WO:        s(dModel*H*d, salt+5),
			MLPNormW:  syntheticFloat32(dModel, salt+6),
			WGate:     s(dFF*dModel, salt+7), WUp: s(dFF*dModel, salt+8), WDown: s(dModel*dFF, salt+9),
			T: T, DModel: dModel, DFF: dFF, Heads: H, KVHeads: kv, HeadDim: d,
			RopeScale: 1, AttnScale: 0.5, Eps: 1e-5,
			QNormW:        near1F32(salt+10, d),
			PostAttnNormW: near1F32(salt+11, dModel),
			PostFFNormW:   near1F32(salt+12, dModel),
			LayerScalar:   0.75,
			PLIDim:        pliDim,
			PLEGateW:      s(pliDim*dModel, salt+13),
			PLEProjW:      scaleSlice(syntheticFloat32(dModel*pliDim, salt+14), 0.8),
			PLEPostNormW:  near1F32(salt+15, dModel),
			PLEInput:      syntheticFloat32(T*pliDim, salt+16),
		}
	}
	slidingRope := func(L *RealTrainLayerF32) { // standard full rotary over the small head
		L.RopePairHalf = L.HeadDim / 2
		L.RopeInvFreq = realRopeInvFreqs(L.HeadDim, 10000)
		L.Window = 2
	}
	globalRope := func(L *RealTrainLayerF32) { // the engine's Inf-padded proportional table, rot=2 of d=8
		const rot = 2
		folded := float32(math.Pow(1e6, float64(rot)/float64(L.HeadDim))) // the arch's pre-folded base
		periods := globalRopePeriodsFromFolded(L.HeadDim, rot, folded)
		inv := make([]float32, 0, rot/2)
		for _, p := range periods[:rot/2] {
			inv = append(inv, 1/p)
		}
		L.RopePairHalf = L.HeadDim / 2
		L.RopeInvFreq = inv
	}
	slideOwner := mk(600, 4, 2)
	slideOwner.WK = s(2*4*dModel, 620)
	slideOwner.WV = s(2*4*dModel, 621)
	slideOwner.KNormW = near1F32(622, 4)
	slideOwner.ValueNorm = true
	slidingRope(slideOwner)
	globalOwner := mk(700, 8, 1)
	globalOwner.WK = s(1*8*dModel, 720)
	globalOwner.WV = s(1*8*dModel, 721) // E2B globals carry their own v_proj (K≠V)
	globalOwner.KNormW = near1F32(722, 8)
	globalOwner.ValueNorm = true
	globalRope(globalOwner)
	globalConsumer := mk(800, 8, 1)
	globalConsumer.SharesKV = true
	globalRope(globalConsumer)
	globalConsumer.Window = 0
	slideConsumer := mk(900, 4, 2)
	slideConsumer.SharesKV = true
	slidingRope(slideConsumer)
	return []*RealTrainLayerF32{slideOwner, globalOwner, globalConsumer, slideConsumer}, []int{0, 1, 1, 0}
}

// TestRealSharedChainBackward_MixedHeadDim finite-difference-gates the per-layer GEOMETRY
// SWITCHING shape (the #42 last rung): sliding hd-4/kv-2 beside global hd-8/kv-1 in ONE chain,
// each class under its own rope form (standard windowed vs proportional Inf-padded full-head),
// with shared-KV consumers of BOTH owner classes. Every one of the seven projection targets is
// adapted somewhere in the mixed stack — k/v/o on the GLOBAL owner (the hd-8 cache the global
// consumer attends, gradients owner-routed across the geometry switch), q/down on the global
// consumer, up on the sliding owner, gate on the sliding consumer — and the harness FD-checks
// dH0 through all four layers. Green here means the chain maths is exact ACROSS a head-dim
// switch, the property the live E2B anchor asserts at B=0 on the real checkpoint.
func TestRealSharedChainBackward_MixedHeadDim(t *testing.T) {
	const rank = 2
	scaling := float32(16.0 / rank)
	layers, shareFrom := mixedGeometryStack()
	adapters := []*layerLoRAAdapter{
		newChainAdapter(t, layers, 1, ProjK, rank, 61),    // global owner: the consumed hd-8 cache paths
		newChainAdapter(t, layers, 1, ProjV, rank, 63),
		newChainAdapter(t, layers, 1, ProjO, rank, 65),
		newChainAdapter(t, layers, 2, ProjQ, rank, 67),    // global consumer attention + MLP
		newChainAdapter(t, layers, 2, ProjDown, rank, 69),
		newChainAdapter(t, layers, 0, ProjUp, rank, 71),   // sliding owner MLP
		newChainAdapter(t, layers, 3, ProjGate, rank, 73), // sliding consumer gate (consumer level — the owner-level gate coordinate defeats the FD instrument, see TestRealSharedChainBackward)
	}
	sharedChainFDCheck(t, layers, shareFrom, adapters, rank, scaling)
}

// TestRealSharedChainBackward_DenseChain gates the chain helpers on a SHARE-FREE stack — the
// trainer now routes every per-layer walk through them, so the dense 3-layer composition must be
// FD-exact for adapters at every depth (the deepest layer's parameters ride two more full layers
// of curvature — the hardest numerical case the single-layer #40 gates never composed).
func TestRealSharedChainBackward_DenseChain(t *testing.T) {
	const rank = 2
	scaling := float32(16.0 / rank)
	layers, _ := tinySharedStackWithMap()
	shareFrom := []int{0, 1, 2}
	s := func(n, salt int) []float32 { return scaleSlice(syntheticFloat32(n, salt), 0.3) }
	for li, L := range layers[1:] {
		// densify the consumers: own K==V key projection + K-norm + value-norm, cache owned.
		L.SharesKV = false
		L.WK = s(L.KVHeads*L.HeadDim*L.DModel, 500+li*7)
		L.KNormW = near1F32(520+li*7, L.HeadDim)
		L.ValueNorm = true
	}
	adapters := []*layerLoRAAdapter{
		newChainAdapter(t, layers, 0, ProjQ, rank, 31),
		newChainAdapter(t, layers, 0, ProjO, rank, 33),
		newChainAdapter(t, layers, 1, ProjQ, rank, 35),
		newChainAdapter(t, layers, 2, ProjDown, rank, 37),
	}
	sharedChainFDCheck(t, layers, shareFrom, adapters, rank, scaling)
}

// TestRealConsumerForwardTape_Good pins the consumer forward's defining property: it ATTENDS THE
// OWNER'S CACHED ROWS, so the owner's key projection changes the consumer layer's output even
// though the consumer has no k/v of its own; and the tape's kr/v alias the owner's rows as-is (no
// consumer-side re-norm/re-rope — the encAttnHalfShared read semantics).
func TestRealConsumerForwardTape_Good(t *testing.T) {
	layers, shareFrom := tinySharedStackWithMap()
	owner, consumer := layers[0], layers[1]
	h := scaleSlice(syntheticFloat32(owner.T*owner.DModel, 21), 0.5)

	run := func() ([]float32, *realLayerTape, *realLayerTape) {
		sets := make([]layerWeightSet, len(layers))
		for li, L := range layers {
			sets[li] = layerWeightSet{wQ: L.WQ, wK: L.WK, wV: L.WV, wO: L.WO, wGate: L.WGate, wUp: L.WUp, wDown: L.WDown}
		}
		_, tapes, err := realSharedChainForward(h, layers, shareFrom, sets)
		if err != nil {
			t.Fatalf("chain forward: %v", err)
		}
		return tapes[1].out, tapes[0], tapes[1]
	}
	base, ownTape, conTape := run()

	// the consumer reads the owner's rows VERBATIM: same backing arrays, no copy, no extra op.
	if &conTape.kr[0] != &ownTape.kr[0] || &conTape.v[0] != &ownTape.v[0] {
		t.Fatal("the consumer tape must alias the owner's cached kr/v rows as-is")
	}

	// perturbing the OWNER's key projection must change the CONSUMER's output (the cross-layer
	// read is live) — the consumer itself has no key path to carry it.
	owner.WK[0] += 0.25
	changed, _, _ := run()
	owner.WK[0] -= 0.25
	same := true
	for i := range base {
		if base[i] != changed[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("perturbing the owner's WK did not move the consumer's output — the shared-KV read is not wired")
	}

	// a consumer refuses the wrong entry points and the k/v projection vocabulary.
	if _, err := realLayerForwardTape(h, consumer, consumer.WQ, nil, nil, consumer.WO, consumer.WGate, consumer.WUp, consumer.WDown); err == nil {
		t.Fatal("realLayerForwardTape must refuse a SharesKV consumer")
	}
	if _, _, err := consumer.projDims(ProjK); err == nil || !strings.Contains(err.Error(), "OWNER") {
		t.Fatalf("a consumer's k_proj must refuse pointing at the owner; got: %v", err)
	}
	if _, _, err := consumer.projDims(ProjV); err == nil || !strings.Contains(err.Error(), "OWNER") {
		t.Fatalf("a consumer's v_proj must refuse pointing at the owner; got: %v", err)
	}
}

// TestRealConsumerForwardTape_Bad: the ext-row shape contract and the SharesKV declaration are
// refused before any work.
func TestRealConsumerForwardTape_Bad(t *testing.T) {
	layers, _ := tinySharedStackWithMap()
	owner, consumer := layers[0], layers[1]
	h := make([]float32, consumer.T*consumer.DModel)
	kv := make([]float32, consumer.T*consumer.KVHeads*consumer.HeadDim)

	if _, err := realConsumerForwardTape(h, owner, kv, kv, owner.WQ, owner.WO, owner.WGate, owner.WUp, owner.WDown); err == nil {
		t.Fatal("a non-SharesKV layer must be refused by the consumer forward")
	}
	if _, err := realConsumerForwardTape(h, consumer, kv[:1], kv, consumer.WQ, consumer.WO, consumer.WGate, consumer.WUp, consumer.WDown); err == nil {
		t.Fatal("a wrong-shaped extK must be refused")
	}
	if _, err := realConsumerForwardTape(h, consumer, kv, kv[:1], consumer.WQ, consumer.WO, consumer.WGate, consumer.WUp, consumer.WDown); err == nil {
		t.Fatal("a wrong-shaped extV must be refused")
	}
	if _, err := realConsumerForwardTape(h[:1], consumer, kv, kv, consumer.WQ, consumer.WO, consumer.WGate, consumer.WUp, consumer.WDown); err == nil {
		t.Fatal("a wrong-shaped h must be refused")
	}
	// a consumer carrying its own key machinery is a broken template.
	bad := *consumer
	bad.KNormW = near1F32(9, bad.HeadDim)
	if _, err := realConsumerForwardTape(h, &bad, kv, kv, bad.WQ, bad.WO, bad.WGate, bad.WUp, bad.WDown); err == nil {
		t.Fatal("a SharesKV layer with its own KNormW must be refused")
	}
}

// TestRealConsumerBackward_Bad: the upstream-gradient shape contract.
func TestRealConsumerBackward_Bad(t *testing.T) {
	layers, shareFrom := tinySharedStackWithMap()
	consumer := layers[1]
	h := scaleSlice(syntheticFloat32(consumer.T*consumer.DModel, 21), 0.5)
	sets := make([]layerWeightSet, len(layers))
	for li, L := range layers {
		sets[li] = layerWeightSet{wQ: L.WQ, wK: L.WK, wV: L.WV, wO: L.WO, wGate: L.WGate, wUp: L.WUp, wDown: L.WDown}
	}
	inputs, tapes, err := realSharedChainForward(h, layers, shareFrom, sets)
	if err != nil {
		t.Fatalf("chain forward: %v", err)
	}
	if _, err := realConsumerBackward(make([]float32, 1), inputs[1], consumer, tapes[1], consumer.WQ, consumer.WO, consumer.WGate, consumer.WUp, consumer.WDown); err == nil {
		t.Fatal("a wrong-shaped dout must be refused")
	}
}

// TestValidateShareTopology_Good: the E2B-shaped map and the share-free map both pass.
func TestValidateShareTopology_Good(t *testing.T) {
	layers, shareFrom := tinySharedStackWithMap()
	if err := validateShareTopology(layers, shareFrom); err != nil {
		t.Fatalf("the owner+2-consumers topology must pass: %v", err)
	}
	dense := []*RealTrainLayerF32{tinyRealLayer(), tinyRealLayer()}
	if err := validateShareTopology(dense, []int{0, 1}); err != nil {
		t.Fatalf("a share-free chain must pass: %v", err)
	}
}

// TestValidateShareTopology_Bad: every malformed map refuses — a later/self owner, an owner that
// is itself a consumer, an inconsistent SharesKV flag (both directions), a geometry mismatch, and
// a wrong-length map.
func TestValidateShareTopology_Bad(t *testing.T) {
	mk := func() ([]*RealTrainLayerF32, []int) { return tinySharedStackWithMap() }

	layers, shareFrom := mk()
	if err := validateShareTopology(layers, shareFrom[:2]); err == nil {
		t.Fatal("a wrong-length shareFrom must be refused")
	}
	layers, shareFrom = mk()
	shareFrom[1] = 2 // owner LATER than the consumer
	if err := validateShareTopology(layers, shareFrom); err == nil {
		t.Fatal("a later owner must be refused")
	}
	layers, shareFrom = mk()
	shareFrom[2] = 1 // layer 1 is itself a consumer
	if err := validateShareTopology(layers, shareFrom); err == nil {
		t.Fatal("an owner that is itself a consumer must be refused")
	}
	layers, shareFrom = mk()
	layers[1].SharesKV = false // map says consumer, layer says owner
	if err := validateShareTopology(layers, shareFrom); err == nil {
		t.Fatal("a consumer without the SharesKV declaration must be refused")
	}
	layers, shareFrom = mk()
	shareFrom[1] = 1 // map says owner, layer says consumer
	if err := validateShareTopology(layers, shareFrom); err == nil {
		t.Fatal("a SharesKV layer mapped as an owner must be refused")
	}
	layers, shareFrom = mk()
	layers[2].HeadDim *= 2 // cache geometry mismatch against the owner
	if err := validateShareTopology(layers, shareFrom); err == nil {
		t.Fatal("a consumer/owner cache-geometry mismatch must be refused")
	}
}

// TestBuildLayerAdaptersSharedKV_Good: on a shared stack, k_proj/v_proj requests skip the
// consumers (they carry no k/v tensors) and resolve on the owner only; q_proj resolves everywhere.
func TestBuildLayerAdaptersSharedKV_Good(t *testing.T) {
	layers, _ := tinySharedStackWithMap()
	adapters, err := buildLayerAdapters(layers, []string{ProjQ, ProjK}, 2, 0.02)
	if err != nil {
		t.Fatalf("buildLayerAdapters: %v", err)
	}
	var names []string
	for _, ad := range adapters {
		names = append(names, layerAdapterTensorName(ad.layer, ad.target))
	}
	want := map[string]bool{
		"model.layers.0.self_attn.q_proj": true,
		"model.layers.1.self_attn.q_proj": true,
		"model.layers.2.self_attn.q_proj": true,
		"model.layers.0.self_attn.k_proj": true, // the owner's k_proj; consumers carry none
	}
	if len(names) != len(want) {
		t.Fatalf("adapter set: got %v want %d entries", names, len(want))
	}
	for _, n := range names {
		if !want[n] {
			t.Fatalf("unexpected adapter %s in %v", n, names)
		}
	}
}
