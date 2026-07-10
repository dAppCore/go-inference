// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"github.com/tmc/apple/metal"
)

type archICBQuantPSOKey struct {
	outDim, inDim, groupSize, bits int
	dense                          bool
}

type archICBQuantProjCheck struct {
	w           QuantWeight
	outDim, inD int
}

type archICBQuantLayerProjBuffers struct {
	q, k, v, o, g, u, d qmvWeight
}

type archICBQuantPLEProjBuffers struct {
	gate, proj qmvWeight
}

type archICBQuantSetupScratch struct {
	lFF                  []int
	anwBufs, mnwBufs     []metal.MTLBuffer
	qNormBufs, kNormBufs []metal.MTLBuffer
	postAttnBufs         []metal.MTLBuffer
	postFFBufs           []metal.MTLBuffer
	layerScalarBufs      []metal.MTLBuffer
	lb                   []archICBQuantLayerProjBuffers
	pleLB                []archICBQuantPLEProjBuffers
	plePostNorms         []metal.MTLBuffer
	projResident         []metal.MTLBuffer
	pleResident          []metal.MTLBuffer
	projChecks           []archICBQuantProjCheck
	projNames            []string
	psoByKey             map[archICBQuantPSOKey]metal.MTLComputePipelineState
	nQDimByHd, kQDimByHd map[int]metal.MTLBuffer
	nKvDimByKvd          map[int]metal.MTLBuffer
	kDFFByW, nDFFByW     map[int]metal.MTLBuffer
}

type archICBQuantCacheSlices struct {
	kCaches, vCaches []metal.MTLBuffer
}

var archICBQuantSetupScratchPool sync.Pool
var archICBQuantCacheSlicesPool sync.Pool

func newArchICBQuantSetupScratch(nLayers int) *archICBQuantSetupScratch {
	return &archICBQuantSetupScratch{
		lFF:             make([]int, nLayers),
		anwBufs:         make([]metal.MTLBuffer, nLayers),
		mnwBufs:         make([]metal.MTLBuffer, nLayers),
		qNormBufs:       make([]metal.MTLBuffer, nLayers),
		kNormBufs:       make([]metal.MTLBuffer, nLayers),
		postAttnBufs:    make([]metal.MTLBuffer, nLayers),
		postFFBufs:      make([]metal.MTLBuffer, nLayers),
		layerScalarBufs: make([]metal.MTLBuffer, nLayers),
		lb:              make([]archICBQuantLayerProjBuffers, nLayers),
		pleLB:           make([]archICBQuantPLEProjBuffers, nLayers),
		plePostNorms:    make([]metal.MTLBuffer, nLayers),
		projResident:    make([]metal.MTLBuffer, 0, nLayers*24+16),
		pleResident:     make([]metal.MTLBuffer, 0, nLayers*6+2),
		projChecks:      make([]archICBQuantProjCheck, 0, 7),
		projNames:       make([]string, 0, 7),
		psoByKey:        make(map[archICBQuantPSOKey]metal.MTLComputePipelineState, nLayers*7),
		nQDimByHd:       make(map[int]metal.MTLBuffer, nLayers),
		kQDimByHd:       make(map[int]metal.MTLBuffer, nLayers),
		nKvDimByKvd:     make(map[int]metal.MTLBuffer, nLayers),
		kDFFByW:         make(map[int]metal.MTLBuffer, nLayers),
		nDFFByW:         make(map[int]metal.MTLBuffer, nLayers),
	}
}

func (s *archICBQuantSetupScratch) fits(nLayers int) bool {
	return s != nil &&
		cap(s.lFF) >= nLayers &&
		cap(s.anwBufs) >= nLayers &&
		cap(s.mnwBufs) >= nLayers &&
		cap(s.qNormBufs) >= nLayers &&
		cap(s.kNormBufs) >= nLayers &&
		cap(s.postAttnBufs) >= nLayers &&
		cap(s.postFFBufs) >= nLayers &&
		cap(s.layerScalarBufs) >= nLayers &&
		cap(s.lb) >= nLayers &&
		cap(s.pleLB) >= nLayers &&
		cap(s.plePostNorms) >= nLayers &&
		cap(s.projResident) >= nLayers*24+16 &&
		cap(s.pleResident) >= nLayers*6+2 &&
		cap(s.projChecks) >= 7 &&
		cap(s.projNames) >= 7 &&
		s.psoByKey != nil &&
		s.nQDimByHd != nil &&
		s.kQDimByHd != nil &&
		s.nKvDimByKvd != nil &&
		s.kDFFByW != nil &&
		s.nDFFByW != nil
}

func (s *archICBQuantSetupScratch) reset(nLayers int) *archICBQuantSetupScratch {
	clear(s.lFF)
	clear(s.anwBufs)
	clear(s.mnwBufs)
	clear(s.qNormBufs)
	clear(s.kNormBufs)
	clear(s.postAttnBufs)
	clear(s.postFFBufs)
	clear(s.layerScalarBufs)
	clear(s.lb)
	clear(s.pleLB)
	clear(s.plePostNorms)
	clear(s.projResident)
	clear(s.pleResident)
	clear(s.projChecks)
	clear(s.projNames)
	clear(s.psoByKey)
	clear(s.nQDimByHd)
	clear(s.kQDimByHd)
	clear(s.nKvDimByKvd)
	clear(s.kDFFByW)
	clear(s.nDFFByW)
	s.lFF = s.lFF[:nLayers]
	s.anwBufs = s.anwBufs[:nLayers]
	s.mnwBufs = s.mnwBufs[:nLayers]
	s.qNormBufs = s.qNormBufs[:nLayers]
	s.kNormBufs = s.kNormBufs[:nLayers]
	s.postAttnBufs = s.postAttnBufs[:nLayers]
	s.postFFBufs = s.postFFBufs[:nLayers]
	s.layerScalarBufs = s.layerScalarBufs[:nLayers]
	s.lb = s.lb[:nLayers]
	s.pleLB = s.pleLB[:nLayers]
	s.plePostNorms = s.plePostNorms[:nLayers]
	s.projResident = s.projResident[:0]
	s.pleResident = s.pleResident[:0]
	s.projChecks = s.projChecks[:0]
	s.projNames = s.projNames[:0]
	return s
}

func getArchICBQuantSetupScratch(nLayers int) *archICBQuantSetupScratch {
	if v := archICBQuantSetupScratchPool.Get(); v != nil {
		s := v.(*archICBQuantSetupScratch)
		if s.fits(nLayers) {
			return s.reset(nLayers)
		}
	}
	return newArchICBQuantSetupScratch(nLayers)
}

func putArchICBQuantSetupScratch(s *archICBQuantSetupScratch) {
	if s != nil {
		archICBQuantSetupScratchPool.Put(s.reset(0))
	}
}

func newArchICBQuantCacheSlices(nLayers int) *archICBQuantCacheSlices {
	return &archICBQuantCacheSlices{
		kCaches: make([]metal.MTLBuffer, nLayers),
		vCaches: make([]metal.MTLBuffer, nLayers),
	}
}

func (s *archICBQuantCacheSlices) reset(nLayers int) *archICBQuantCacheSlices {
	clear(s.kCaches)
	clear(s.vCaches)
	s.kCaches = s.kCaches[:nLayers]
	s.vCaches = s.vCaches[:nLayers]
	return s
}

func getArchICBQuantCacheSlices(nLayers int) *archICBQuantCacheSlices {
	if v := archICBQuantCacheSlicesPool.Get(); v != nil {
		s := v.(*archICBQuantCacheSlices)
		if cap(s.kCaches) >= nLayers && cap(s.vCaches) >= nLayers {
			return s.reset(nLayers)
		}
	}
	return newArchICBQuantCacheSlices(nLayers)
}

func putArchICBQuantCacheSlices(s *archICBQuantCacheSlices) {
	if s != nil {
		archICBQuantCacheSlicesPool.Put(s.reset(0))
	}
}

// DecodeForwardArchICBQuant is the arch-driven decode with BOTH fast-path levers
// stacked: quant qmv weights (cut the GPU read) AND the ICB encode-bypass replay (cut
// the per-token host re-encode), DRIVEN by the declared arch (KV-share + sliding). It
// is DecodeForwardArchICB with a qmv `recordProj` (affine_qmv_bfloat16_t) instead of
// gemv, running the same arch-aware decodeForwardArchICBCore — the V projection binds at
// index 4 (qmv) not 3 (gemv), so vOutBind=4. Byte-for-byte equal to DecodeForwardArchQuant
// on the same arch. Public MoE calls route through the native re-encode MoE decoder before
// recording, because the router's host top-k cannot sit in a recorded/replayed command
// buffer. All raw bf16 activations.
// recordArchICBQuant records the 4-bit arch ICB and returns the held *archICBReplay — the
// recorder shared by the batch DecodeForwardArchICBQuant (record + runBatch) and the
// ArchSession (record once at open, stepBody per token). Caches + the PLE runtime are
// parameters: the batch passes fresh caches + a batch-token-id runtime; the session passes its
// own lb caches (so prefill's KV is visible) + {nil, s.perLayerInput}. pleRuntime nil ⇒ no PLE;
// pleGS/pleBits are the PLE gate/proj quant geometry for quantPLELayers.
func recordArchICBQuant(
	qlayers []QuantizedLayerWeights, specs []model.LayerSpec,
	kCaches, vCaches []metal.MTLBuffer,
	pleRuntime *archDecodePLEInputs, pliDim, pleGS, pleBits int,
	dModel, nHeads, nKVHeads, headDim, maxLen, dFF, slidingWindow int,
	rope icbRope, scale, eps float32, valueNorm bool,
) (*archICBReplay, error) {
	nLayers := len(qlayers)
	setup := getArchICBQuantSetupScratch(nLayers)
	defer putArchICBQuantSetupScratch(setup)

	for li := range specs {
		o := specs[li].KVShareFrom
		if o < 0 || o > li || (o != li && !specs[o].OwnsCache()) {
			return nil, core.NewError("native.DecodeForwardArchICBQuant: KVShareFrom must reference an earlier owner layer")
		}
		if specs[li].MoE {
			return nil, core.NewError("native.DecodeForwardArchICBQuant: MoE layers are not supported on the ICB path")
		}
	}
	// per-layer FFN width (gemma4 E2B/E4B MatFormer): lFF[li] (from ql.DFF, fallback dFF) —
	// drives the Gate/Up/Down size validation, the per-width PSO/scalar keying, and the core.
	lFF := setup.lFF
	for li := range qlayers {
		lFF[li] = dFF
		if qlayers[li].DFF > 0 {
			lFF[li] = qlayers[li].DFF
		}
	}
	for li := range qlayers {
		ql := qlayers[li]
		// Affine geometry is required only when a projection actually carries sidecars: an
		// all-dense layer (recordArchICBBF16 wraps bf16 weights as sidecar-less QuantWeights)
		// has no groupSize/bits to demand — every weight dispatches through the dense gemv.
		if (ql.GroupSize == 0 || ql.Bits == 0) && quantizedLayerHasAffine(ql) {
			return nil, core.NewError("native.recordArchICBQuant: GroupSize/Bits unset")
		}
		if len(ql.AttnNormW) != dModel*bf16Size || len(ql.MLPNormW) != dModel*bf16Size {
			return nil, core.NewError("native.DecodeForwardArchICBQuant: norm weight size mismatch")
		}
		lff := lFF[li]
		lhd := headDimOf(specs[li], headDim) // per-layer head dim (gemma4 full_attention > sliding)
		lqDim, lkvDim := nHeads*lhd, kvHeadsOf(specs[li], nKVHeads)*lhd
		projChecks := setup.projChecks[:0]
		projNames := setup.projNames[:0]
		projChecks = append(projChecks,
			archICBQuantProjCheck{ql.Q, lqDim, dModel}, archICBQuantProjCheck{ql.O, dModel, lqDim},
			archICBQuantProjCheck{ql.Gate, lff, dModel}, archICBQuantProjCheck{ql.Up, lff, dModel}, archICBQuantProjCheck{ql.Down, dModel, lff},
		)
		projNames = append(projNames, "Q", "O", "Gate", "Up", "Down")
		if specs[li].OwnsCache() { // KV-shared layers carry no own K/V (they read the owner's) — only owners have K/V to size-check
			projChecks = append(projChecks, archICBQuantProjCheck{ql.K, lkvDim, dModel})
			projNames = append(projNames, "K")
			if len(ql.V.Packed) > 0 { // K==V layers carry no v_proj — V rides the k-proj output
				projChecks = append(projChecks, archICBQuantProjCheck{ql.V, lkvDim, dModel})
				projNames = append(projNames, "V")
			}
		}
		for pi, p := range projChecks {
			effGS, effBits := quantWeightGeometry(p.w, ql.GroupSize, ql.Bits)
			wantPacked, wantSB := 0, 0
			if effGS > 0 && effBits > 0 {
				wantPacked = p.outDim * p.inD * effBits / 8
				wantSB = p.outDim * (p.inD / effGS) * bf16Size
			}
			if !quantWeightProjectionShapeOK(p.w, p.outDim, p.inD, ql.GroupSize, ql.Bits) {
				return nil, core.NewError(core.Sprintf("native.DecodeForwardArchICBQuant: %s quant size mismatch — outDim=%d inD=%d bits=%d gs=%d; Packed=%d want %d; Scales=%d want %d; Biases=%d want %d",
					projNames[pi], p.outDim, p.inD, effBits, effGS, len(p.w.Packed), wantPacked, len(p.w.Scales), wantSB, len(p.w.Biases), wantSB))
			}
		}
	}
	var pleLayers []pleLayer
	var err error
	if pleRuntime != nil {
		pleLayers, err = quantPLELayers("native.recordArchICBQuant", qlayers, dModel, pliDim, pleGS, pleBits)
		if err != nil {
			return nil, err
		}
	}
	// V-projection output bind index: prepareStepRebind re-points THIS binding at the KV cache
	// row each token, so it must match the recorded op's output slot — the qmv binds out at 4
	// (wq/scales/biases=0/1/2, x=3), the dense gemv at 3 (mat=0, vec=1). Derived from the owner
	// layers' actual V weight (K when V rides the k-proj, gemma4 K==V); one global index serves
	// every layer, so a dense/quant mix across owners cannot be recorded.
	vOutBind := uint(0)
	for li := range qlayers {
		if !specs[li].OwnsCache() {
			continue
		}
		w := qlayers[li].V
		if len(w.Packed) == 0 {
			w = qlayers[li].K
		}
		b := uint(4)
		if len(w.Scales) == 0 && len(w.Biases) == 0 {
			b = 3
		}
		if vOutBind == 0 {
			vOutBind = b
		} else if vOutBind != b {
			return nil, core.NewError("native.recordArchICBQuant: dense and quantised V projections mixed across owner layers — one vOutBind serves the whole replay")
		}
	}
	if vOutBind == 0 {
		vOutBind = 4
	}

	// qmv ICB pipelines, one per distinct (outDim,inDim,groupSize,bits) shape
	// (built before the pool). Mixed-precision packs need distinct recorded PSOs.
	psoByKey := setup.psoByKey
	qmvPSO := func(outDim, inDim, groupSize, bits int) (metal.MTLComputePipelineState, error) {
		key := archICBQuantPSOKey{outDim: outDim, inDim: inDim, groupSize: groupSize, bits: bits}
		if pso, ok := psoByKey[key]; ok {
			return pso, nil
		}
		pso, err := pipelineForICB(qmvBF16KernelName(outDim, inDim, groupSize, bits))
		if err != nil {
			return nil, err
		}
		psoByKey[key] = pso
		return pso, nil
	}
	denseGemvPSO := func(outDim, inDim int) (metal.MTLComputePipelineState, error) {
		key := archICBQuantPSOKey{outDim: outDim, inDim: inDim, dense: true}
		if pso, ok := psoByKey[key]; ok {
			return pso, nil
		}
		bm, bn, sm, sn, tm, tn := gemvTiles(inDim, outDim)
		pso, err := pipelineForICB(gemvKernelName("bfloat16", bm, bn, sm, sn, tm, tn))
		if err != nil {
			return nil, err
		}
		psoByKey[key] = pso
		return pso, nil
	}
	ensureQMVPSO := func(w QuantWeight, outDim, inDim, groupSize, bits int) error {
		if quantWeightDenseShapeOK(w, outDim, inDim) {
			_, err := denseGemvPSO(outDim, inDim)
			return err
		}
		groupSize, bits = quantWeightGeometry(w, groupSize, bits)
		_, err := qmvPSO(outDim, inDim, groupSize, bits)
		return err
	}
	for li := range qlayers {
		ql := qlayers[li]
		lff := lFF[li]
		lhd := headDimOf(specs[li], headDim)
		lqDim, lkvDim := nHeads*lhd, kvHeadsOf(specs[li], nKVHeads)*lhd
		projChecks := setup.projChecks[:0]
		projChecks = append(projChecks,
			archICBQuantProjCheck{ql.Q, lqDim, dModel}, archICBQuantProjCheck{ql.O, dModel, lqDim},
			archICBQuantProjCheck{ql.Gate, lff, dModel}, archICBQuantProjCheck{ql.Up, lff, dModel}, archICBQuantProjCheck{ql.Down, dModel, lff},
		)
		if specs[li].OwnsCache() {
			projChecks = append(projChecks, archICBQuantProjCheck{ql.K, lkvDim, dModel})
			if len(ql.V.Packed) > 0 {
				projChecks = append(projChecks, archICBQuantProjCheck{ql.V, lkvDim, dModel})
			}
		}
		for _, p := range projChecks {
			if err := ensureQMVPSO(p.w, p.outDim, p.inD, ql.GroupSize, ql.Bits); err != nil {
				return nil, err
			}
		}
	}
	if pleRuntime != nil {
		for li := range pleLayers {
			if err := ensureQMVPSO(pleLayers[li].gate, pliDim, dModel, pleGS, pleBits); err != nil {
				return nil, err
			}
			if err := ensureQMVPSO(pleLayers[li].proj, dModel, pliDim, pleGS, pleBits); err != nil {
				return nil, err
			}
		}
	}

	var r *archICBReplay
	var coreErr error
	withAutoreleasePool(func() {
		anwBufs := setup.anwBufs
		mnwBufs := setup.mnwBufs
		qNormBufs := setup.qNormBufs
		kNormBufs := setup.kNormBufs
		postAttnBufs := setup.postAttnBufs
		postFFBufs := setup.postFFBufs
		layerScalarBufs := setup.layerScalarBufs
		lb := setup.lb
		pleLB := setup.pleLB
		plePostNorms := setup.plePostNorms
		residentView := func(b []byte) bufView { return bufView{buf: residentBytes(b)} }
		residentOrNil := func(b []byte) metal.MTLBuffer {
			if len(b) == 0 {
				return nil
			}
			return residentBytes(b)
		}
		mkW := func(w QuantWeight, groupSize, bits int) qmvWeight {
			if len(w.Packed) == 0 { // absent projection (gemma4 K==V: no v_proj) ⇒ nil weight, hasV()==false
				return qmvWeight{}
			}
			if len(w.Scales) == 0 && len(w.Biases) == 0 {
				return qmvWeight{wq: residentView(w.Packed)}
			}
			groupSize, bits = quantWeightGeometry(w, groupSize, bits)
			return qmvWeight{wq: residentView(w.Packed), scales: residentView(w.Scales), biases: residentView(w.Biases), gs: groupSize, bits: bits}
		}
		// psoFor returns the qmv pipeline for this geometry, BUILDING IT ON A MISS rather than
		// trusting the pre-pool enumeration to be exhaustive. The precompute (ensureQMVPSO above)
		// is a cache-warming optimisation, not a correctness contract: the recorder emits a projK
		// for EVERY layer to keep the ICB op layout uniform (decode_forward_arch_icb.go ~L657),
		// including KV-sharer layers the precompute's OwnsCache() guard skips. A bare map miss there
		// returned a nil pipeline state, which SetComputePipelineState msgSend'd into → SIGSEGV.
		// Build-on-miss makes the recorder self-sufficient so the two paths cannot diverge; a
		// genuinely unbuildable geometry sets coreErr (caught after the pool) instead of crashing.
		psoFor := func(w qmvWeight, outDim, inDim int) metal.MTLComputePipelineState {
			if w.dense() {
				pso, err := denseGemvPSO(outDim, inDim)
				if err != nil {
					if coreErr == nil {
						coreErr = core.E("native.recordArchICBQuant", core.Sprintf("gemv pipeline outDim=%d inDim=%d", outDim, inDim), err)
					}
					return nil
				}
				return pso
			}
			pso, err := qmvPSO(outDim, inDim, w.gs, w.bits)
			if err != nil {
				if coreErr == nil {
					coreErr = core.E("native.recordArchICBQuant", core.Sprintf("qmv pipeline outDim=%d inDim=%d gs=%d bits=%d", outDim, inDim, w.gs, w.bits), err)
				}
				return nil
			}
			return pso
		}
		// presized to the upper bound (every layer's 7 projections × wq/scales/biases, the 5 shared
		// trailing scalar buffers, plus ≤2 FFN dim scalars per distinct dFF width) so the per-forward
		// build never geometrically regrows its backing array — K==V layers simply leave the v-proj
		// slot unused. Byte-identical.
		projResident := setup.projResident
		appendResidentWeight := func(w qmvWeight) {
			if w.wq.buf != nil { // K==V / KV-shared: no separate weight to make resident
				projResident = append(projResident, w.wq.buf)
				if w.scales.buf != nil {
					projResident = append(projResident, w.scales.buf)
				}
				if w.biases.buf != nil {
					projResident = append(projResident, w.biases.buf)
				}
			}
		}
		for li := range qlayers {
			ql := qlayers[li]
			anwBufs[li] = residentBytes(ql.AttnNormW)
			mnwBufs[li] = residentBytes(ql.MLPNormW)
			qNormBufs[li] = residentOrNil(ql.QNormW)
			kNormBufs[li] = residentOrNil(ql.KNormW)
			postAttnBufs[li] = residentOrNil(ql.PostAttnNormW)
			postFFBufs[li] = residentOrNil(ql.PostFFNormW)
			layerScalarBufs[li] = layerScalarBuf(ql.LayerScalarW, dModel)
			lb[li] = archICBQuantLayerProjBuffers{
				mkW(ql.Q, ql.GroupSize, ql.Bits), mkW(ql.K, ql.GroupSize, ql.Bits),
				mkW(ql.V, ql.GroupSize, ql.Bits), mkW(ql.O, ql.GroupSize, ql.Bits),
				mkW(ql.Gate, ql.GroupSize, ql.Bits), mkW(ql.Up, ql.GroupSize, ql.Bits),
				mkW(ql.Down, ql.GroupSize, ql.Bits),
			}
			appendResidentWeight(lb[li].q)
			appendResidentWeight(lb[li].k)
			appendResidentWeight(lb[li].v)
			appendResidentWeight(lb[li].o)
			appendResidentWeight(lb[li].g)
			appendResidentWeight(lb[li].u)
			appendResidentWeight(lb[li].d)
			// KV-shared layers carry no own K/V weights, yet the recorder still emits a discarded
			// projK/projV per layer for ICB op-layout uniformity (output -> kThrow/vThrow). Point that
			// placeholder at the OWNER's K/V (same head dim ⇒ a valid PRECOMPUTED PSO + already-resident
			// buffers) rather than a degenerate empty (gs=0/bits=0) qmv with nil weight buffers —
			// correctness-neutral (the result is thrown away) and it removes the driver-dependent
			// nil-buffer dispatch the psoFor crash-guard previously had to absorb.
			if !specs[li].OwnsCache() {
				own := specs[li].KVShareFrom
				if lb[li].k.wq.buf == nil {
					lb[li].k = lb[own].k
				}
				if lb[li].v.wq.buf == nil {
					lb[li].v = lb[own].v
				}
			}
			if pleRuntime != nil {
				pleLB[li] = archICBQuantPLEProjBuffers{mkW(pleLayers[li].gate, pleGS, pleBits), mkW(pleLayers[li].proj, pleGS, pleBits)}
				plePostNorms[li] = residentBytes(pleLayers[li].postNorm)
			}
		}
		kDModel, nDModel := scalarI32(int32(dModel)), scalarI32(int32(dModel))
		kvOf := func(li int) int { return kvHeadsOf(specs[li], nKVHeads) } // per-layer KV heads (12B/31B MQA globals)
		// per-hd qmv dim scalars: nQDim = qDim out (projQ), kQDim = qDim in (projO) — both hd-only. The K/V
		// projection out dim (nKvDim = kvHeads·hd) varies with PER-LAYER kvHeads, so it's keyed by kvDim.
		nQDimByHd := setup.nQDimByHd
		kQDimByHd := setup.kQDimByHd
		nKvDimByKvd := setup.nKvDimByKvd
		for li := range specs {
			hd := headDimOf(specs[li], headDim)
			if _, ok := nQDimByHd[hd]; !ok {
				nQDimByHd[hd] = scalarI32(int32(nHeads * hd))
				kQDimByHd[hd] = scalarI32(int32(nHeads * hd))
			}
			if kvd := kvOf(li) * hd; nil == nKvDimByKvd[kvd] {
				nKvDimByKvd[kvd] = scalarI32(int32(kvd))
			}
		}
		// per-distinct-dFF qmv dim scalars: kDFF (down's K=inDim=lff) and nDFF (gate/up's N=outDim=lff).
		kDFFByW := setup.kDFFByW
		nDFFByW := setup.nDFFByW
		for li := range lFF {
			lff := lFF[li]
			if _, ok := kDFFByW[lff]; !ok {
				kDFFByW[lff] = scalarI32(int32(lff))
				nDFFByW[lff] = scalarI32(int32(lff))
			}
		}
		projResident = append(projResident, kDModel, nDModel)
		for hd, b := range nQDimByHd {
			projResident = append(projResident, b, kQDimByHd[hd])
		}
		for _, b := range nKvDimByKvd {
			projResident = append(projResident, b)
		}
		for lff, b := range kDFFByW {
			projResident = append(projResident, b, nDFFByW[lff])
		}

		// 4-bit qmv through the SHARED emitQMV body (with encQMVBF16). K/N bind the same memoised scalars
		// the kDModel/nQDimByHd/… count buffers hold, so they're dropped from the call in favour of the values.
		setQMV := func(c metal.MTLIndirectComputeCommand, pso metal.MTLComputePipelineState, w qmvWeight, vec, out metal.MTLBuffer, outOff uint, inDim, outDim int) {
			if pso == nil { // psoFor failed (coreErr already set) — never msgSend into a nil pipeline state
				return
			}
			if w.dense() {
				bm, bn, sm, _, tm, _ := gemvTiles(inDim, outDim)
				emitGemv(fastICBSink{c}, pso, w.wq.buf, w.wq.off, vec, out, outOff, inDim, outDim, bm, bn, sm, tm)
				return
			}
			emitQMV(fastICBSink{c}, pso, w.wq.buf, w.wq.off, w.scales.buf, w.scales.off, w.biases.buf, w.biases.off, vec, out, outOff, inDim, outDim)
		}
		var plePlan *archICBPLEPlan
		if pleRuntime != nil {
			kPLIDim, nPLIDim := scalarI32(int32(pliDim)), scalarI32(int32(pliDim))
			pleResident := append(setup.pleResident, kPLIDim, nPLIDim)
			appendPLEResident := func(w qmvWeight) { // dense PLE weights (bf16 recorder) carry no scales/biases
				pleResident = append(pleResident, w.wq.buf)
				if w.scales.buf != nil {
					pleResident = append(pleResident, w.scales.buf)
				}
				if w.biases.buf != nil {
					pleResident = append(pleResident, w.biases.buf)
				}
			}
			for li := range pleLB {
				appendPLEResident(pleLB[li].gate)
				appendPLEResident(pleLB[li].proj)
			}
			setup.pleResident = pleResident
			plePlan = &archICBPLEPlan{
				runtime: pleRuntime, pliDim: pliDim, postNormBufs: plePostNorms, resident: pleResident,
			}
			plePlan.recordGate = func(li int, c metal.MTLIndirectComputeCommand, vec, out metal.MTLBuffer) {
				setQMV(c, psoFor(pleLB[li].gate, pliDim, dModel), pleLB[li].gate, vec, out, 0, dModel, pliDim)
			}
			// The fused gate+gelu·pli op (lthn_ple_gate_gelu_qmv, #373) is NOT
			// recorded here: it is byte-identical to the composed pair in every
			// standalone context (both PSO builds, every real e2b layer's weights
			// — the kernel parity test), but IN THE RECORDED REPLAY it drifted
			// the session off the PerLayerInputGate interleave from the second
			// sequential step onward (#371, TestRealQuantVerifyBatchedHiddensParity;
			// first bad d00c526, in-situ cause unresolved). Its receipt was ~3µs
			// (thin-stage), so the composed two-stage path stays — the parity
			// spine outranks a free-but-unproven op.
			plePlan.recordProj = func(li int, c metal.MTLIndirectComputeCommand, vec, out metal.MTLBuffer) {
				setQMV(c, psoFor(pleLB[li].proj, dModel, pliDim), pleLB[li].proj, vec, out, 0, pliDim, dModel)
			}
		}
		recordProj := func(li int, c metal.MTLIndirectComputeCommand, vec, out metal.MTLBuffer, outOff uint, p projIndex) {
			l := lb[li]
			hd := headDimOf(specs[li], headDim)
			switch p {
			case projQ:
				setQMV(c, psoFor(l.q, nHeads*hd, dModel), l.q, vec, out, outOff, dModel, nHeads*hd)
			case projK:
				kvd := kvOf(li) * hd
				setQMV(c, psoFor(l.k, kvd, dModel), l.k, vec, out, outOff, dModel, kvd)
			case projV:
				kvd := kvOf(li) * hd
				setQMV(c, psoFor(l.v, kvd, dModel), l.v, vec, out, outOff, dModel, kvd)
			case projO:
				setQMV(c, psoFor(l.o, dModel, nHeads*hd), l.o, vec, out, outOff, nHeads*hd, dModel)
			case projGate:
				lff := lFF[li]
				setQMV(c, psoFor(l.g, lff, dModel), l.g, vec, out, outOff, dModel, lff)
			case projUp:
				lff := lFF[li]
				setQMV(c, psoFor(l.u, lff, dModel), l.u, vec, out, outOff, dModel, lff)
			case projDown:
				lff := lFF[li]
				setQMV(c, psoFor(l.d, dModel, lff), l.d, vec, out, outOff, lff, dModel)
			}
		}
		// --- fused input-RMSNorm + qmv (matmul-fusion spike): fold the input-rms INTO the Q/K/V/gate/up
		// projections so there's no separate barriered setRMS before the matmul. Fast-variant only
		// (outDim%8==0 && inDim%512==0 — all e2b input projections qualify); gated on the custom lib.
		setRMSQMV := func(c metal.MTLIndirectComputeCommand, pso metal.MTLComputePipelineState, w qmvWeight, vec, normW, out, kB, nB, epsB metal.MTLBuffer, outOff uint, outDim int) {
			if pso == nil { // rmsQMVPSOFor failed (coreErr already set)
				return
			}
			c.SetComputePipelineState(pso)
			c.SetKernelBufferOffsetAtIndex(w.wq.buf, w.wq.off, 0)
			c.SetKernelBufferOffsetAtIndex(w.scales.buf, w.scales.off, 1)
			c.SetKernelBufferOffsetAtIndex(w.biases.buf, w.biases.off, 2)
			c.SetKernelBufferOffsetAtIndex(vec, 0, 3)
			c.SetKernelBufferOffsetAtIndex(out, outOff, 4)
			c.SetKernelBufferOffsetAtIndex(kB, 0, 5)
			c.SetKernelBufferOffsetAtIndex(nB, 0, 6)
			c.SetKernelBufferOffsetAtIndex(normW, 0, 7)
			c.SetKernelBufferOffsetAtIndex(epsB, 0, 8)
			const bn, bk = 8, 32
			nTgp := (outDim + bn - 1) / bn
			c.ConcurrentDispatchThreadgroupsThreadsPerThreadgroup(metal.MTLSize{Width: 1, Height: uint(nTgp), Depth: 1}, metal.MTLSize{Width: bk, Height: 2, Depth: 1})
		}
		rmsQMVPSOFor := func(w qmvWeight, outDim, inDim int) metal.MTLComputePipelineState {
			if outDim%8 != 0 || inDim%512 != 0 { // the fused kernel is the FAST qmv variant only
				if coreErr == nil {
					coreErr = core.NewError(core.Sprintf("native.recordArchICBQuant: fused rms+qmv needs outDim%%8==0 && inDim%%512==0, got %d/%d", outDim, inDim))
				}
				return nil
			}
			pso, err := rmsQMVPipelineICB(w.gs, w.bits)
			if err != nil {
				if coreErr == nil {
					coreErr = core.E("native.recordArchICBQuant", core.Sprintf("fused rms+qmv pso gs=%d bits=%d", w.gs, w.bits), err)
				}
				return nil
			}
			return pso
		}
		// Enable the fusion only when the custom lib is loaded AND every input-rms-fed projection on every
		// layer satisfies the fast-variant geometry (inDim=dModel %512==0, outDim %8==0). Otherwise fall
		// back to the plain setRMS+qmv path (recordFusedRMSProj==nil) rather than hard-failing — a small
		// synthetic dModel (e.g. 256) simply doesn't fuse.
		// enableInputRMSFusion: the fused input-rms→qmv (lthn_rms_affine_qmv_fast) is correct but measured
		// NET-ZERO (the 3× redundant rms recompute cancels the 2 barriers it removes), and it makes the ICB
		// byte-differ from the re-encode path. Disabled — kept as dormant capability (the kernel + the
		// closure below) for the matmul-fusion-tier batch, which needs the value-norm sibling to pay.
		const enableInputRMSFusion = false
		fusedGeomOK := dModel%512 == 0
		for li := range qlayers {
			hd := headDimOf(specs[li], headDim)
			for _, od := range []int{nHeads * hd, kvOf(li) * hd, lFF[li]} {
				if od%8 != 0 {
					fusedGeomOK = false
				}
			}
		}
		var recordFusedRMSProj func(li int, c metal.MTLIndirectComputeCommand, rawIn, normW, epsB, out metal.MTLBuffer, outOff uint, p projIndex)
		if enableInputRMSFusion && gpuHasGeluKernel() && fusedGeomOK { // disabled: net-zero + ICB byte-diff
			recordFusedRMSProj = func(li int, c metal.MTLIndirectComputeCommand, rawIn, normW, epsB, out metal.MTLBuffer, outOff uint, p projIndex) {
				l := lb[li]
				hd := headDimOf(specs[li], headDim)
				switch p {
				case projQ:
					setRMSQMV(c, rmsQMVPSOFor(l.q, nHeads*hd, dModel), l.q, rawIn, normW, out, kDModel, nQDimByHd[hd], epsB, outOff, nHeads*hd)
				case projK:
					kvd := kvOf(li) * hd
					setRMSQMV(c, rmsQMVPSOFor(l.k, kvd, dModel), l.k, rawIn, normW, out, kDModel, nKvDimByKvd[kvd], epsB, outOff, kvd)
				case projV:
					kvd := kvOf(li) * hd
					setRMSQMV(c, rmsQMVPSOFor(l.v, kvd, dModel), l.v, rawIn, normW, out, kDModel, nKvDimByKvd[kvd], epsB, outOff, kvd)
				case projGate:
					lff := lFF[li]
					setRMSQMV(c, rmsQMVPSOFor(l.g, lff, dModel), l.g, rawIn, normW, out, kDModel, nDFFByW[lff], epsB, outOff, lff)
				case projUp:
					lff := lFF[li]
					setRMSQMV(c, rmsQMVPSOFor(l.u, lff, dModel), l.u, rawIn, normW, out, kDModel, nDFFByW[lff], epsB, outOff, lff)
				}
			}
		}
		valueNormOnes := valueNormOnesBuf(valueNorm, maxHeadDimOf(specs, headDim))
		vProjIdxOf := func(li int) projIndex { // gemma4 K==V is PER-LAYER (12B: sliding layers carry V, global layers don't)
			if len(qlayers[li].V.Packed) == 0 {
				return projK // V rides the k-proj
			}
			return projV
		}
		r, coreErr = recordArchICB(specs, anwBufs, mnwBufs, kCaches, vCaches, projResident, qNormBufs, kNormBufs, postAttnBufs, postFFBufs, layerScalarBufs, plePlan, recordProj, recordFusedRMSProj, vOutBind, valueNormOnes, vProjIdxOf, dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slidingWindow, lFF, rope, scale, eps)
	})
	if coreErr != nil {
		return nil, coreErr
	}
	return r, nil
}

// quantizedLayerHasAffine reports whether any projection in the layer carries affine sidecars
// (scales) — i.e. actually needs a groupSize/bits geometry to decode. An all-dense layer (the
// bf16 arch ICB recorder wraps bf16 weights as sidecar-less QuantWeights) returns false.
func quantizedLayerHasAffine(ql QuantizedLayerWeights) bool {
	return len(ql.Q.Scales) > 0 || len(ql.K.Scales) > 0 || len(ql.V.Scales) > 0 || len(ql.O.Scales) > 0 ||
		len(ql.Gate.Scales) > 0 || len(ql.Up.Scales) > 0 || len(ql.Down.Scales) > 0 ||
		len(ql.PerLayerGate.Scales) > 0 || len(ql.PerLayerProjection.Scales) > 0
}

// DecodeForwardArchICBQuant is the batch quant arch ICB: record the stack once + replay it
// across the whole input sequence (the encode-bypass). It is recordArchICBQuant + runBatch,
// byte-identical to the pre-split entry. MoE layers use the native re-encode MoE decoder.
// All bf16 activations.
func DecodeForwardArchICBQuant(
	inputs [][]byte, qlayers []QuantizedLayerWeights, specs []model.LayerSpec,
	dModel, nHeads, nKVHeads, headDim, maxLen, dFF, slidingWindow int,
	base, scale, eps float32, valueNorm bool,
	pleArgs ...ArchPLEQuant,
) ([][]byte, error) {
	return decodeForwardArchICBQuantInto(nil, inputs, qlayers, specs, dModel, nHeads, nKVHeads, headDim, maxLen, dFF, slidingWindow, base, scale, eps, valueNorm, false, pleArgs...)
}

// DecodeForwardArchICBQuantInto is DecodeForwardArchICBQuant with caller-owned
// per-token output storage. Output slices with enough capacity are reused for
// the final hidden readback from each ICB replay.
func DecodeForwardArchICBQuantInto(
	outputs [][]byte, inputs [][]byte, qlayers []QuantizedLayerWeights, specs []model.LayerSpec,
	dModel, nHeads, nKVHeads, headDim, maxLen, dFF, slidingWindow int,
	base, scale, eps float32, valueNorm bool,
	pleArgs ...ArchPLEQuant,
) ([][]byte, error) {
	return decodeForwardArchICBQuantInto(outputs, inputs, qlayers, specs, dModel, nHeads, nKVHeads, headDim, maxLen, dFF, slidingWindow, base, scale, eps, valueNorm, true, pleArgs...)
}

func decodeForwardArchICBQuantInto(
	outputs [][]byte, inputs [][]byte, qlayers []QuantizedLayerWeights, specs []model.LayerSpec,
	dModel, nHeads, nKVHeads, headDim, maxLen, dFF, slidingWindow int,
	base, scale, eps float32, valueNorm bool,
	useCallerOut bool,
	pleArgs ...ArchPLEQuant,
) ([][]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	nLayers, T := len(qlayers), len(inputs)
	if nLayers == 0 || T == 0 {
		return nil, core.NewError("native.DecodeForwardArchICBQuant: need layers and inputs")
	}
	if len(specs) != nLayers {
		return nil, core.NewError("native.DecodeForwardArchICBQuant: specs length must equal layers")
	}
	if T > maxLen {
		return nil, core.NewError("native.DecodeForwardArchICBQuant: more tokens than maxLen cache rows")
	}
	for i := range inputs {
		if len(inputs[i]) != dModel*bf16Size {
			return nil, core.NewError("native.DecodeForwardArchICBQuant: each input must be dModel bf16 bytes")
		}
	}
	hasMoE, mixedHeadDim := false, false
	for li := range specs {
		o := specs[li].KVShareFrom
		if o < 0 || o > li || (o != li && !specs[o].OwnsCache()) {
			return nil, core.NewError("native.DecodeForwardArchICBQuant: KVShareFrom must reference an earlier owner layer")
		}
		if specs[li].MoE {
			hasMoE = true
		}
		if headDimOf(specs[li], headDim) != headDim {
			mixedHeadDim = true // gemma4 global layers are WIDER (e.g. 512 vs sliding 256)
		}
	}
	// This whole-sequence recorder records simpleICBRope (one base spectrum) for every layer and takes
	// no proportional/partial rope params, so on gemma4's wider global head dim it would rope the global
	// layers wrong past pos 0 (the per-hd projections/caches it DOES handle). For MoE or a mixed head
	// dim, fall back to the per-layer-correct re-encode forward — DecodeForwardArchQuant now validates +
	// decodes per head dim. Byte-identical, just not the ICB fast path for this cold batch call; the
	// SESSION path keeps the fast ICB (it records the full per-layer rope spectrum).
	if hasMoE || mixedHeadDim {
		if useCallerOut {
			return DecodeForwardArchQuantInto(outputs, inputs, qlayers, specs, dModel, nHeads, nKVHeads, headDim, maxLen, dFF, slidingWindow, base, scale, eps, valueNorm, pleArgs...)
		}
		return DecodeForwardArchQuant(inputs, qlayers, specs, dModel, nHeads, nKVHeads, headDim, maxLen, dFF, slidingWindow, base, scale, eps, valueNorm, pleArgs...)
	}
	plePayload, err := singleArchPLEQuant("native.DecodeForwardArchICBQuant", pleArgs)
	if err != nil {
		return nil, err
	}
	pleRuntime, pliDim, err := archPLEQuantRuntime("native.DecodeForwardArchICBQuant", plePayload, nLayers, T, dModel, eps)
	if err != nil {
		return nil, err
	}
	pleGS, pleBits := 0, 0
	if plePayload != nil {
		pleGS, pleBits = plePayload.GroupSize, plePayload.Bits
	}
	cacheSlices := getArchICBQuantCacheSlices(nLayers)
	defer putArchICBQuantCacheSlices(cacheSlices)
	kCaches, vCaches := cacheSlices.kCaches, cacheSlices.vCaches
	// Pipeline the replay once the batch is long enough to amortise a SECOND ICB recording: the
	// double-buffered loop overlaps each token's host turn with the prior token's GPU compute,
	// reclaiming the ~40% per-token WaitUntilCompleted idle (≈1.6× on e2b prefill). Short batches stay
	// serial (the 2nd recording isn't worth it).
	pipeline := len(inputs) >= 4 && !pipelinedBatchDisabled
	var r, r2 *archICBReplay
	var coreErr error
	withAutoreleasePool(func() {
		for li := range specs {
			if specs[li].OwnsCache() { // per-layer cache — global layers' rows are wider (larger head_dim)
				cacheLen := maxLen
				if slidingWindow > 0 && slidingWindow < maxLen && specs[li].Attention != model.GlobalAttention {
					// Bounded ring — the sliding-window KV memory fix: a sliding owner only ever
					// attends its own window, so it only ever needs slidingWindow rows of storage.
					// prepareStepRebind detects the smaller allocation (via the actual buffer
					// length) and rebinds pos%cacheRows instead of the absolute position.
					cacheLen = slidingWindow
				}
				cacheBytes := uint(cacheLen * kvHeadsOf(specs[li], nKVHeads) * headDimOf(specs[li], headDim) * bf16Size)
				kCaches[li] = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
				vCaches[li] = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
			}
		}
		r, coreErr = recordArchICBQuant(qlayers, specs, kCaches, vCaches, pleRuntime, pliDim, pleGS, pleBits, dModel, nHeads, nKVHeads, headDim, maxLen, dFF, slidingWindow, simpleICBRope(base, headDim), scale, eps, valueNorm)
		if coreErr == nil && pipeline {
			r2, coreErr = recordArchICBQuant(qlayers, specs, kCaches, vCaches, pleRuntime, pliDim, pleGS, pleBits, dModel, nHeads, nKVHeads, headDim, maxLen, dFF, slidingWindow, simpleICBRope(base, headDim), scale, eps, valueNorm)
		}
	})
	if coreErr != nil {
		return nil, coreErr
	}
	defer r.releaseScratch()
	if r2 != nil {
		defer r2.releaseScratch()
		return r.runBatchPipelinedInto(r2, outputs, inputs, useCallerOut)
	}
	return r.runBatchInto(outputs, inputs, useCallerOut)
}
