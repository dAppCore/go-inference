// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"github.com/tmc/apple/metal"
)

// DecodeForwardArchQuant is the 4-bit arch-driven decode forward: DecodeForwardArch
// with quantised projections. It runs the SAME arch-driven loop (runArchDecode) over
// the SAME cache-topology + sliding-window the bf16 path does — the projector seam is
// the only difference (qmvProjector / affine_qmv_bfloat16_t instead of bf16Projector),
// so KV-sharing and sliding layers get 4-bit weights for free. With an all-owner,
// all-global arch it equals DecodeForwardQuant byte-for-byte (gated). The norms stay
// bf16 (not quantised). MoE layers run the same host-orchestrated MoEBlockQuant path
// as ArchQuantSession. All raw bf16 activations.
func DecodeForwardArchQuant(
	inputs [][]byte, qlayers []QuantizedLayerWeights, specs []model.LayerSpec,
	dModel, nHeads, nKVHeads, headDim, maxLen, dFF, slidingWindow int,
	base, scale, eps float32, valueNorm bool,
	pleArgs ...ArchPLEQuant,
) ([][]byte, error) {
	return decodeForwardArchQuantInto(nil, inputs, qlayers, specs, dModel, nHeads, nKVHeads, headDim, maxLen, dFF, slidingWindow, base, scale, eps, valueNorm, false, pleArgs...)
}

// DecodeForwardArchQuantInto is DecodeForwardArchQuant with caller-owned
// per-token output storage. Output slices with enough capacity are reused for
// the final hidden readback from each token.
func DecodeForwardArchQuantInto(
	outputs [][]byte, inputs [][]byte, qlayers []QuantizedLayerWeights, specs []model.LayerSpec,
	dModel, nHeads, nKVHeads, headDim, maxLen, dFF, slidingWindow int,
	base, scale, eps float32, valueNorm bool,
	pleArgs ...ArchPLEQuant,
) ([][]byte, error) {
	return decodeForwardArchQuantInto(outputs, inputs, qlayers, specs, dModel, nHeads, nKVHeads, headDim, maxLen, dFF, slidingWindow, base, scale, eps, valueNorm, true, pleArgs...)
}

func decodeForwardArchQuantInto(
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
		return nil, core.NewError("native.DecodeForwardArchQuant: need layers and inputs")
	}
	if len(specs) != nLayers {
		return nil, core.NewError("native.DecodeForwardArchQuant: specs length must equal layers")
	}
	if T > maxLen {
		return nil, core.NewError("native.DecodeForwardArchQuant: more tokens than maxLen cache rows")
	}
	for i := range inputs {
		if len(inputs[i]) != dModel*bf16Size {
			return nil, core.NewError("native.DecodeForwardArchQuant: each input must be dModel bf16 bytes")
		}
	}
	for li := range specs {
		o := specs[li].KVShareFrom
		if o < 0 || o > li || (o != li && !specs[o].OwnsCache()) {
			return nil, core.NewError("native.DecodeForwardArchQuant: KVShareFrom must reference an earlier owner layer")
		}
		if specs[li].MoE != (qlayers[li].MoE != nil) {
			return nil, core.NewError("native.DecodeForwardArchQuant: spec.MoE must match the presence of layer MoE weights")
		}
	}
	// validate each layer's quant weight shapes (norms bf16; the seven projections).
	type pj struct {
		w           QuantWeight
		outDim, inD int
	}
	for li := range qlayers {
		ql := qlayers[li]
		if ql.GroupSize == 0 || ql.Bits == 0 {
			return nil, core.NewError("native.DecodeForwardArchQuant: GroupSize/Bits unset")
		}
		if len(ql.AttnNormW) != dModel*bf16Size {
			return nil, core.NewError("native.DecodeForwardArchQuant: attention norm weight size mismatch")
		}
		if ql.MoE == nil && len(ql.MLPNormW) != dModel*bf16Size {
			return nil, core.NewError("native.DecodeForwardArchQuant: MLP norm weight size mismatch")
		}
		// per-layer FFN width (gemma4 E2B/E4B MatFormer varies it): validate Gate/Up/Down against
		// THIS layer's lff, not the uniform dFF — buildQuantArchLayerBufs already runs the decode at
		// ql.DFF, so a uniform-dFF check would reject the heterogeneous layer it can correctly execute.
		// lff==dFF for uniform callers ⇒ byte-identical validation.
		lff := dFF
		if ql.DFF > 0 {
			lff = ql.DFF
		}
		// per-layer attention geometry: gemma4 global layers use a WIDER head_dim (e.g. 512 vs sliding
		// 256), so size Q/K/V/O against THIS layer's head dim, not the uniform base — buildQuantArchLayerBufs
		// already runs the decode at headDimOf(spec) per layer, so a uniform check would reject the
		// heterogeneous arch it can correctly execute. lhd==headDim for uniform callers ⇒ byte-identical.
		lhd := headDimOf(specs[li], headDim)
		lqDim, lkvDim := nHeads*lhd, kvHeadsOf(specs[li], nKVHeads)*lhd
		if ql.MoE != nil {
			if err := validateMoEQuantLayerWeights("native.DecodeForwardArchQuant", ql.MoE, dModel, lff); err != nil {
				return nil, err
			}
		}
		projChecks := []pj{
			{ql.Q, lqDim, dModel}, {ql.O, dModel, lqDim},
		}
		if ql.MoE == nil {
			projChecks = append(projChecks, pj{ql.Gate, lff, dModel}, pj{ql.Up, lff, dModel}, pj{ql.Down, dModel, lff})
		}
		if specs[li].OwnsCache() { // KV-shared layers carry no own K/V (they read the owner's) — only owners have K/V to size-check
			projChecks = append(projChecks, pj{ql.K, lkvDim, dModel})
			if len(ql.V.Packed) > 0 { // K==V layers carry no v_proj — V rides the k-proj output
				projChecks = append(projChecks, pj{ql.V, lkvDim, dModel})
			}
		}
		for _, p := range projChecks {
			if !quantWeightProjectionShapeOK(p.w, p.outDim, p.inD, ql.GroupSize, ql.Bits) {
				return nil, core.NewError("native.DecodeForwardArchQuant: quantised weight size mismatch")
			}
		}
	}
	plePayload, err := singleArchPLEQuant("native.DecodeForwardArchQuant", pleArgs)
	if err != nil {
		return nil, err
	}
	pleRuntime, pliDim, err := archPLEQuantRuntime("native.DecodeForwardArchQuant", plePayload, nLayers, T, dModel, eps)
	if err != nil {
		return nil, err
	}
	if pleRuntime != nil {
		defer pleRuntime.Close()
	}
	var pleLayers []pleLayer
	if pleRuntime != nil {
		pleLayers, err = quantPLELayers("native.DecodeForwardArchQuant", qlayers, dModel, pliDim, plePayload.GroupSize, plePayload.Bits)
		if err != nil {
			return nil, err
		}
	}

	withAutoreleasePool(func() {
		setup := getArchQuantLayerBufScratch(nLayers)
		defer putArchQuantLayerBufScratch(setup)
		lb, moeQuant, berr := buildQuantArchLayerBufsIntoScratch(setup, qlayers, specs, dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slidingWindow, nil)
		if berr != nil {
			err = berr
			return
		}
		moeWeights := make([]*MoELayerWeights, nLayers) // bf16 MoE unused on the quant path
		state := newArchDecodeState(specs, lb, moeWeights, dModel, nHeads, nKVHeads, headDim, dFF, slidingWindow, headDim, headDim, base, base, scale, eps, valueNorm, maxLen)
		defer state.Close()
		state.moeQuant = moeQuant
		if pleRuntime != nil {
			state.ple, state.pliDim = pleLayers, pliDim
			outputs, err = runArchDecodeStateInto(outputs, inputs, &state, pleRuntime, useCallerOut)
			return
		}
		outputs, err = runArchDecodeStateInto(outputs, inputs, &state, nil, useCallerOut)
	})
	return outputs, err
}

func quantWeightShapeOK(w QuantWeight, outDim, inDim, groupSize, bits int) bool {
	groupSize, bits = quantWeightGeometry(w, groupSize, bits)
	return groupSize > 0 && bits > 0 && inDim%groupSize == 0 &&
		len(w.Packed) == outDim*inDim*bits/8 &&
		len(w.Scales) == outDim*(inDim/groupSize)*bf16Size &&
		len(w.Biases) == outDim*(inDim/groupSize)*bf16Size
}

func quantWeightDenseShapeOK(w QuantWeight, outDim, inDim int) bool {
	return len(w.Packed) == outDim*inDim*bf16Size && len(w.Scales) == 0 && len(w.Biases) == 0
}

func quantWeightProjectionShapeOK(w QuantWeight, outDim, inDim, groupSize, bits int) bool {
	return quantWeightDenseShapeOK(w, outDim, inDim) || quantWeightShapeOK(w, outDim, inDim, groupSize, bits)
}

func quantWeightGeometry(w QuantWeight, groupSize, bits int) (int, int) {
	if w.GroupSize > 0 {
		groupSize = w.GroupSize
	}
	if w.Bits > 0 {
		bits = w.Bits
	}
	return groupSize, bits
}

func quantWeightGeometryForShape(w QuantWeight, outDim, inDim, groupSize, bits int) (int, int) {
	if w.GroupSize > 0 || w.Bits > 0 {
		wgs, wbits := quantWeightGeometry(w, groupSize, bits)
		if quantWeightBytesFit(w, outDim, inDim, wgs, wbits) {
			return wgs, wbits
		}
	}
	if quantWeightBytesFit(w, outDim, inDim, groupSize, bits) {
		return groupSize, bits
	}
	return quantWeightGeometry(w, groupSize, bits)
}

func quantWeightBytesFit(w QuantWeight, outDim, inDim, groupSize, bits int) bool {
	return groupSize > 0 && bits > 0 && inDim%groupSize == 0 &&
		len(w.Packed) == outDim*inDim*bits/8 &&
		len(w.Scales) == outDim*(inDim/groupSize)*bf16Size &&
		len(w.Biases) == outDim*(inDim/groupSize)*bf16Size
}

func validateMoEQuantLayerWeights(fn string, w *MoEQuantLayerWeights, dModel, dFF int) error {
	if w == nil {
		return core.NewError(fn + ": missing MoE quant weights")
	}
	if w.NumExperts <= 0 || w.TopK <= 0 || w.TopK > w.NumExperts || w.ExpertDFF <= 0 {
		return core.NewError(fn + ": invalid MoE quant geometry")
	}
	for _, norm := range [][]byte{w.PreFFNormW, w.PreFFNorm2W, w.PostFFNorm1W, w.PostFFNorm2W, w.PostFFNormW, w.RouterNormWScaled} {
		if len(norm) != dModel*bf16Size {
			return core.NewError(fn + ": MoE norm weight size mismatch")
		}
	}
	if w.PerExpertScale != nil && len(w.PerExpertScale) != w.NumExperts*bf16Size {
		return core.NewError(fn + ": MoE per-expert scale size mismatch")
	}
	if !quantWeightShapeOK(w.LocalGate, dFF, dModel, w.LocalGroupSize, w.LocalBits) ||
		!quantWeightShapeOK(w.LocalUp, dFF, dModel, w.LocalGroupSize, w.LocalBits) ||
		!quantWeightShapeOK(w.LocalDown, dModel, dFF, w.LocalGroupSize, w.LocalBits) {
		return core.NewError(fn + ": MoE local MLP quant size mismatch")
	}
	if !quantWeightShapeOK(w.Router, w.NumExperts, dModel, w.RouterGroupSize, w.RouterBits) {
		return core.NewError(fn + ": MoE router quant size mismatch")
	}
	splitExpertsOK := quantWeightShapeOK(w.ExpGate, w.NumExperts*w.ExpertDFF, dModel, w.ExpertGroupSize, w.ExpertBits) &&
		quantWeightShapeOK(w.ExpUp, w.NumExperts*w.ExpertDFF, dModel, w.ExpertGroupSize, w.ExpertBits)
	fusedExpertsOK := quantWeightShapeOK(w.ExpGateUp, w.NumExperts*2*w.ExpertDFF, dModel, w.ExpertGroupSize, w.ExpertBits)
	if (!splitExpertsOK && !fusedExpertsOK) ||
		!quantWeightShapeOK(w.ExpDown, w.NumExperts*dModel, w.ExpertDFF, w.ExpertGroupSize, w.ExpertBits) {
		return core.NewError(fn + ": MoE expert quant size mismatch")
	}
	return nil
}

// buildQuantArchLayerBufs builds the per-layer archLayerBufs for the 4-bit path: bf16 norm
// buffers (the norms aren't quantised), owner-layer KV caches, and a qmvProjector per layer —
// the only difference from buildBF16ArchLayerBufs. Shared by DecodeForwardArchQuant and
// NewArchQuantSession. sb is the zero-copy weight source (see buildBF16ArchLayerBufs): non-nil
// binds every weight (norms + the quant triples) as no-copy shard views; nil uploads owned copies.
// MUST be called inside a withAutoreleasePool.
func buildQuantArchLayerBufs(qlayers []QuantizedLayerWeights, specs []model.LayerSpec, dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slidingWindow int, sb *shardBuffers) ([]archLayerBufs, []*MoEQuantLayerWeights, error) {
	return buildQuantArchLayerBufsInternal(make([]archLayerBufs, len(qlayers)), make([]*MoEQuantLayerWeights, len(qlayers)), nil, qlayers, specs, dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slidingWindow, sb)
}

func buildQuantArchLayerBufsIntoScratch(setup *archQuantLayerBufScratch, qlayers []QuantizedLayerWeights, specs []model.LayerSpec, dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slidingWindow int, sb *shardBuffers) ([]archLayerBufs, []*MoEQuantLayerWeights, error) {
	if setup == nil || !setup.fits(len(qlayers)) {
		return buildQuantArchLayerBufs(qlayers, specs, dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slidingWindow, sb)
	}
	setup.reset(len(qlayers))
	return buildQuantArchLayerBufsInternal(setup.lb, setup.moe, setup, qlayers, specs, dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slidingWindow, sb)
}

func buildQuantArchLayerBufsInternal(lb []archLayerBufs, moeQuant []*MoEQuantLayerWeights, setup *archQuantLayerBufScratch, qlayers []QuantizedLayerWeights, specs []model.LayerSpec, dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slidingWindow int, sb *shardBuffers) ([]archLayerBufs, []*MoEQuantLayerWeights, error) {
	var ferr error
	view := func(b []byte) bufView {
		if sb != nil {
			return sb.mustBufFor(b, &ferr)
		}
		return bufView{buf: residentBytes(b)}
	}
	view4 := func(b []byte) bufView { // 4-bit packed uint32 weights need 4-byte alignment (affine_qmv reads uint32)
		if sb != nil {
			return sb.mustBufFor4(b, &ferr)
		}
		return bufView{buf: residentBytes(b)}
	}
	viewOrNil := func(b []byte) bufView {
		if len(b) == 0 {
			return bufView{}
		}
		return view(b)
	}
	// mkW resolves one 4-bit triple to bufViews (no-copy shard views or copies); an absent
	// projection (gemma4 K==V: no v_proj) ⇒ the zero qmvWeight, hasV()==false.
	mkW := func(qw QuantWeight) qmvWeight {
		if len(qw.Packed) == 0 {
			return qmvWeight{}
		}
		if len(qw.Scales) == 0 && len(qw.Biases) == 0 {
			return qmvWeight{wq: view(qw.Packed)}
		}
		return qmvWeight{wq: view4(qw.Packed), scales: view(qw.Scales), biases: view(qw.Biases), gs: qw.GroupSize, bits: qw.Bits}
	}
	viewQuantWeight := func(qw QuantWeight) QuantWeight {
		if len(qw.Packed) == 0 {
			return qw
		}
		qw.packedView = view4(qw.Packed)
		qw.scalesView = view(qw.Scales)
		qw.biasesView = view(qw.Biases)
		return qw
	}
	viewOptional := func(b []byte) bufView {
		if len(b) == 0 {
			return bufView{}
		}
		return view(b)
	}
	residentOptional := func(b []byte) bufView {
		if len(b) == 0 {
			return bufView{}
		}
		return bufView{buf: residentBytes(b)}
	}
	for li := range qlayers {
		ql := qlayers[li]
		// per-attention-type geometry: full layers use the larger global head_dim.
		lhd, lkv := headDimOf(specs[li], headDim), kvHeadsOf(specs[li], nKVHeads)
		qDim, kvDim := nHeads*lhd, lkv*lhd
		// sliding layers RING at slidingWindow rows (the full-context KV memory fix) — see the bf16
		// build for the rationale; global (full_attention) layers keep maxLen.
		cacheLen := maxLen
		if slidingWindow > 0 && slidingWindow < maxLen && specs[li].Attention != model.GlobalAttention {
			cacheLen = slidingWindow
		}
		cacheBytes := uint(cacheLen * kvDim * bf16Size)
		lb[li].anw = view(ql.AttnNormW)
		lb[li].postAttnNorm = viewOrNil(ql.PostAttnNormW)
		lb[li].postFFNorm = viewOrNil(ql.PostFFNormW)
		lb[li].qNorm = viewOrNil(ql.QNormW)
		lb[li].kNorm = viewOrNil(ql.KNormW)
		lb[li].layerScalar = layerScalarBuf(ql.LayerScalarW, dModel) // synthesised broadcast (not a shard view)
		if specs[li].OwnsCache() {
			if setup != nil {
				lb[li].kCache, lb[li].vCache, lb[li].kCachePtr, lb[li].vCachePtr = setup.kvCache(li, cacheBytes)
			} else {
				lb[li].kCache = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
				lb[li].vCache = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
			}
		}
		lFF := dFF // per-layer FFN width (gemma4 E2B/E4B vary it); 0 ⇒ arch default
		if ql.DFF > 0 {
			lFF = ql.DFF
		}
		lb[li].dFF = lFF
		proj := qmvProjector{
			q: mkW(ql.Q), k: mkW(ql.K), v: mkW(ql.V), o: mkW(ql.O),
			dModel: dModel, qDim: qDim, kvDim: kvDim, dFF: lFF,
			groupSize: ql.GroupSize, bits: ql.Bits,
		}
		// MoE layers run MoEBlockQuant (host-orchestrated) instead of the dense MLP, so the
		// projector binds only attention; the dense MLP weights/norm are unused (and nil).
		if ql.MoE != nil {
			var mw *MoEQuantLayerWeights
			if setup != nil {
				setup.moeVals[li] = *ql.MoE
				mw = &setup.moeVals[li]
			} else {
				mwv := *ql.MoE
				mw = &mwv
			}
			mw.LocalGate = viewQuantWeight(mw.LocalGate)
			mw.LocalUp = viewQuantWeight(mw.LocalUp)
			mw.LocalDown = viewQuantWeight(mw.LocalDown)
			mw.Router = viewQuantWeight(mw.Router)
			mw.ExpGate = viewQuantWeight(mw.ExpGate)
			mw.ExpUp = viewQuantWeight(mw.ExpUp)
			mw.ExpGateUp = viewQuantWeight(mw.ExpGateUp)
			mw.ExpDown = viewQuantWeight(mw.ExpDown)
			mw.preFFNormView = viewOptional(mw.PreFFNormW)
			mw.preFFNorm2View = viewOptional(mw.PreFFNorm2W)
			mw.postFFNorm1View = viewOptional(mw.PostFFNorm1W)
			mw.postFFNorm2View = viewOptional(mw.PostFFNorm2W)
			mw.postFFNormView = viewOptional(mw.PostFFNormW)
			mw.routerNormView = residentOptional(mw.RouterNormWScaled)
			mw.perExpertScaleView = viewOptional(mw.PerExpertScale)
			moeQuant[li] = mw
		} else {
			lb[li].mnw = view(ql.MLPNormW)
			proj.gate, proj.up, proj.down = mkW(ql.Gate), mkW(ql.Up), mkW(ql.Down)
		}
		lb[li].proj = proj
	}
	return lb, moeQuant, ferr
}
