// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

type DiffusionLayerKV struct {
	K           []byte // [prefixLen, kvHeads, headDim] bf16, native session row layout
	V           []byte // [prefixLen, kvHeads, headDim] bf16, native session row layout
	PrefixStart int    // absolute token position for K/V row 0; zero keeps the full-prefix default
	Position    int    // absolute token position where the canvas starts; zero defaults to PrefixStart+prefixLen
}

func DiffusionDenoiseForwardBF16(g *BF16Model, diffusion *model.LoadedDiffusion, arch model.Arch, canvas []int32, scEmb []byte, layerKV []DiffusionLayerKV, globalMask, localMask []float32) ([]byte, error) {
	const op = "native.DiffusionDenoiseForwardBF16"
	if g == nil {
		return nil, core.NewError(op + ": model is nil")
	}
	if len(g.Layers) == 0 || len(g.Layers) != len(arch.Layer) {
		return nil, core.NewError(op + ": layer count mismatch")
	}
	if len(layerKV) != len(g.Layers) {
		return nil, core.NewError(op + ": layerKV count mismatch")
	}
	L := len(canvas)
	dModel, vocab := arch.Hidden, arch.Vocab
	if L == 0 {
		return []byte{}, nil
	}
	if dModel <= 0 || vocab <= 0 || arch.Heads <= 0 || arch.KVHeads <= 0 {
		return nil, core.NewError(op + ": invalid arch dimensions")
	}
	embRows, err := EmbedTokensBF16(g.Embed, canvas, vocab, dModel, embedScaleOf(arch))
	if err != nil {
		return nil, err
	}
	inputEmb := diffusionJoinRowsBF16(embRows, dModel)
	ple, pliDim, err := diffusionBF16PLEInputs(g, arch, canvas, inputEmb)
	if err != nil {
		return nil, err
	}
	h, err := diffusionApplySelfConditionLinear(inputEmb, scEmb, diffusion, L, dModel, arch.FF, arch.Eps, op)
	if err != nil {
		return nil, err
	}

	scale := attnScaleOf(arch)
	canvasKRows := make([][]byte, len(g.Layers))
	canvasVRows := make([][]byte, len(g.Layers))
	for li, w := range g.Layers {
		spec := arch.Layer[li]
		if spec.MoE != (w.MoE != nil) {
			return nil, core.NewError(op + ": spec.MoE must match the presence of layer MoE weights")
		}
		owner := spec.KVShareFrom
		if owner < 0 || owner >= len(g.Layers) {
			return nil, core.NewError(op + ": invalid KV-sharing owner")
		}
		ownerSpec := arch.Layer[owner]
		if !ownerSpec.OwnsCache() || ownerSpec.KVShareFrom != owner {
			return nil, core.NewError(op + ": invalid KV-sharing owner")
		}
		lhd, ownerHeadDim := headDimOf(spec, arch.HeadDim), headDimOf(ownerSpec, arch.HeadDim)
		if ownerHeadDim != lhd {
			return nil, core.NewError(op + ": shared K/V head_dim mismatch")
		}
		lkv := kvHeadsOf(ownerSpec, arch.KVHeads)
		qDim, kvDim := arch.Heads*lhd, lkv*lhd
		prefixLen, _, position, err := diffusionLayerKVGeometry(layerKV[owner], kvDim)
		if err != nil {
			return nil, err
		}
		keyLen := prefixLen + L
		mask := globalMask
		if spec.Attention == model.SlidingAttention {
			mask = localMask
		}
		if len(mask) != L*keyLen {
			return nil, core.NewError(op + ": mask must be canvasLen*keyLen")
		}

		var ownerCanvasKRows, ownerCanvasVRows []byte
		ownsKV := owner == li
		if !ownsKV {
			ownerCanvasKRows, ownerCanvasVRows = canvasKRows[owner], canvasVRows[owner]
		}
		var kRows, vRows []byte
		h, kRows, vRows, err = diffusionBF16LayerForward(h, w, spec, layerKV[owner], mask, L, prefixLen, position, keyLen, dModel, arch.Heads, lkv, lhd, qDim, kvDim, arch, scale, ownsKV, ownerCanvasKRows, ownerCanvasVRows, ple, pliDim, li, len(g.Layers))
		if err != nil {
			return nil, err
		}
		if ownsKV {
			canvasKRows[li], canvasVRows[li] = kRows, vRows
		}
	}

	head := g.LMHead
	if len(head) == 0 {
		head = g.Embed
	}
	logits := make([]byte, L*vocab*bf16Size)
	for i := range L {
		row := h[i*dModel*bf16Size : (i+1)*dModel*bf16Size]
		out, err := LMHeadBF16(row, g.FinalNorm, head, dModel, vocab, arch.Eps, arch.SoftCap)
		if err != nil {
			return nil, err
		}
		copy(logits[i*vocab*bf16Size:(i+1)*vocab*bf16Size], out)
	}
	return logits, nil
}

func DiffusionDenoiseForwardQuant(g *QuantModel, diffusion *model.LoadedDiffusion, arch model.Arch, canvas []int32, scEmb []byte, layerKV []DiffusionLayerKV, globalMask, localMask []float32) ([]byte, error) {
	const op = "native.DiffusionDenoiseForwardQuant"
	if g == nil {
		return nil, core.NewError(op + ": model is nil")
	}
	if len(g.Layers) == 0 || len(g.Layers) != len(arch.Layer) {
		return nil, core.NewError(op + ": layer count mismatch")
	}
	if len(layerKV) != len(g.Layers) {
		return nil, core.NewError(op + ": layerKV count mismatch")
	}
	L := len(canvas)
	dModel, vocab := arch.Hidden, arch.Vocab
	if L == 0 {
		return []byte{}, nil
	}
	if dModel <= 0 || vocab <= 0 || arch.Heads <= 0 || arch.KVHeads <= 0 {
		return nil, core.NewError(op + ": invalid arch dimensions")
	}
	embRows, err := EmbedTokensQuant(g.Embed, g.EmbedScales, g.EmbedBiases, canvas, vocab, dModel, g.GroupSize, g.Bits, embedScaleOf(arch))
	if err != nil {
		return nil, err
	}
	inputEmb := diffusionJoinRowsBF16(embRows, dModel)
	ple, pliDim, err := diffusionQuantPLEInputs(g, arch, canvas, inputEmb)
	if err != nil {
		return nil, err
	}
	h, err := diffusionApplySelfConditionLinear(inputEmb, scEmb, diffusion, L, dModel, arch.FF, arch.Eps, op)
	if err != nil {
		return nil, err
	}

	scale := attnScaleOf(arch)
	canvasKRows := make([][]byte, len(g.Layers))
	canvasVRows := make([][]byte, len(g.Layers))
	for li, w := range g.Layers {
		spec := arch.Layer[li]
		if spec.MoE != (w.MoE != nil) {
			return nil, core.NewError(op + ": spec.MoE must match the presence of layer MoE weights")
		}
		owner := spec.KVShareFrom
		if owner < 0 || owner >= len(g.Layers) {
			return nil, core.NewError(op + ": invalid KV-sharing owner")
		}
		ownerSpec := arch.Layer[owner]
		if !ownerSpec.OwnsCache() || ownerSpec.KVShareFrom != owner {
			return nil, core.NewError(op + ": invalid KV-sharing owner")
		}
		lhd, ownerHeadDim := headDimOf(spec, arch.HeadDim), headDimOf(ownerSpec, arch.HeadDim)
		if ownerHeadDim != lhd {
			return nil, core.NewError(op + ": shared K/V head_dim mismatch")
		}
		lkv := kvHeadsOf(ownerSpec, arch.KVHeads)
		qDim, kvDim := arch.Heads*lhd, lkv*lhd
		prefixLen, _, position, err := diffusionLayerKVGeometry(layerKV[owner], kvDim)
		if err != nil {
			return nil, err
		}
		keyLen := prefixLen + L
		mask := globalMask
		if spec.Attention == model.SlidingAttention {
			mask = localMask
		}
		if len(mask) != L*keyLen {
			return nil, core.NewError(op + ": mask must be canvasLen*keyLen")
		}

		var ownerCanvasKRows, ownerCanvasVRows []byte
		ownsKV := owner == li
		if !ownsKV {
			ownerCanvasKRows, ownerCanvasVRows = canvasKRows[owner], canvasVRows[owner]
		}
		var kRows, vRows []byte
		h, kRows, vRows, err = diffusionQuantLayerForward(h, w, spec, layerKV[owner], mask, L, prefixLen, position, keyLen, dModel, arch.Heads, lkv, lhd, qDim, kvDim, arch, scale, ownsKV, ownerCanvasKRows, ownerCanvasVRows, ple, pliDim, li, len(g.Layers), g.GroupSize, g.Bits)
		if err != nil {
			return nil, err
		}
		if ownsKV {
			canvasKRows[li], canvasVRows[li] = kRows, vRows
		}
	}

	head, scales, biases := g.LMHead, g.LMHeadScales, g.LMHeadBiases
	if len(head) == 0 {
		head, scales, biases = g.Embed, g.EmbedScales, g.EmbedBiases
	}
	headWeight := QuantWeight{Packed: head, Scales: scales, Biases: biases, GroupSize: g.GroupSize, Bits: g.Bits}
	headGS, headBits := quantWeightGeometryForShape(headWeight, vocab, dModel, g.GroupSize, g.Bits)
	logits := make([]byte, L*vocab*bf16Size)
	for i := range L {
		row := h[i*dModel*bf16Size : (i+1)*dModel*bf16Size]
		out, err := LMHeadQuant(row, g.FinalNorm, head, scales, biases, dModel, vocab, headGS, headBits, arch.Eps, arch.SoftCap)
		if err != nil {
			return nil, err
		}
		copy(logits[i*vocab*bf16Size:(i+1)*vocab*bf16Size], out)
	}
	return logits, nil
}

func diffusionApplySelfConditionLinear(h, scEmb []byte, diffusion *model.LoadedDiffusion, rows, dModel, fallbackDFF int, eps float32, op string) ([]byte, error) {
	if len(scEmb) == 0 {
		return DiffusionSelfConditionBF16(h, nil, nil, nil, nil, nil, rows, dModel, fallbackDFF, eps)
	}
	if diffusion == nil || diffusion.SelfCondGate == nil || diffusion.SelfCondUp == nil || diffusion.SelfCondDown == nil {
		return nil, core.NewError(op + ": self-conditioning weights are missing")
	}
	dFF := diffusion.SelfCondGate.OutDim
	if dFF <= 0 {
		dFF = fallbackDFF
	}
	if len(h) != rows*dModel*bf16Size || len(scEmb) != len(h) {
		return nil, core.NewError(op + ": self-conditioning embedding size mismatch")
	}
	if len(diffusion.SelfCondPreNorm) != dModel*bf16Size {
		return nil, core.NewError(op + ": self-conditioning prenorm must be dModel bf16 bytes")
	}
	normed, err := RMSNormBF16(scEmb, diffusion.SelfCondPreNorm, rows, dModel, eps)
	if err != nil {
		return nil, err
	}
	gate, err := diffusionLinearRowsBF16(diffusion.SelfCondGate, normed, rows, dFF, dModel, op)
	if err != nil {
		return nil, err
	}
	up, err := diffusionLinearRowsBF16(diffusion.SelfCondUp, normed, rows, dFF, dModel, op)
	if err != nil {
		return nil, err
	}
	gated, err := GeluGateMulBF16(gate, up)
	if err != nil {
		return nil, err
	}
	ffw, err := diffusionLinearRowsBF16(diffusion.SelfCondDown, gated, rows, dModel, dFF, op)
	if err != nil {
		return nil, err
	}
	combined, err := AddBF16(h, ffw)
	if err != nil {
		return nil, err
	}
	return RMSNormBF16(combined, diffusionOnesBF16(dModel), rows, dModel, eps)
}

func diffusionLinearRowsBF16(w *model.Linear, x []byte, rows, outDim, inDim int, op string) ([]byte, error) {
	if w == nil {
		return nil, core.NewError(op + ": linear weight is nil")
	}
	if w.OutDim > 0 {
		outDim = w.OutDim
	}
	if w.InDim > 0 {
		inDim = w.InDim
	}
	if w.Quantised() {
		q := QuantWeight{Packed: w.Weight, Scales: w.Scales, Biases: w.Biases, GroupSize: w.GroupSize, Bits: w.Bits}
		return diffusionMatRowsQuant(q, x, rows, outDim, inDim, w.GroupSize, w.Bits)
	}
	return MatRowsBF16(w.Weight, x, rows, outDim, inDim)
}

func diffusionBF16LayerForward(h []byte, w DecodeLayerWeights, spec model.LayerSpec, kv DiffusionLayerKV, mask []float32, L, prefixLen, position, keyLen, dModel, nHeads, nKVHeads, headDim, qDim, kvDim int, arch model.Arch, scale float32, ownsKV bool, ownerCanvasKRows, ownerCanvasVRows []byte, ple []byte, pliDim, layer, numLayers int) ([]byte, []byte, []byte, error) {
	normed, err := RMSNormBF16(h, w.AttnNormW, L, dModel, arch.Eps)
	if err != nil {
		return nil, nil, nil, err
	}
	qRows, err := MatRowsBF16(w.WQ, normed, L, qDim, dModel)
	if err != nil {
		return nil, nil, nil, err
	}
	if len(w.QNormW) > 0 {
		qRows, err = RMSNormBF16(qRows, w.QNormW, L*nHeads, headDim, arch.Eps)
		if err != nil {
			return nil, nil, nil, err
		}
	}
	var kRows, vRows []byte
	if ownsKV {
		kRows, err = MatRowsBF16(w.WK, normed, L, kvDim, dModel)
		if err != nil {
			return nil, nil, nil, err
		}
		vRows, err = MatRowsBF16(w.WV, normed, L, kvDim, dModel)
		if err != nil {
			return nil, nil, nil, err
		}
		if len(w.KNormW) > 0 {
			kRows, err = RMSNormBF16(kRows, w.KNormW, L*nKVHeads, headDim, arch.Eps)
			if err != nil {
				return nil, nil, nil, err
			}
		}
		if arch.ValueNorm {
			vRows, err = RMSNormBF16(vRows, diffusionOnesBF16(headDim), L*nKVHeads, headDim, arch.Eps)
			if err != nil {
				return nil, nil, nil, err
			}
		}
	} else {
		want := L * kvDim * bf16Size
		if len(ownerCanvasKRows) != want || len(ownerCanvasVRows) != want {
			return nil, nil, nil, core.NewError("native.DiffusionDenoiseForwardBF16: shared owner canvas K/V missing")
		}
		kRows, vRows = ownerCanvasKRows, ownerCanvasVRows
	}
	ropeBase, rotaryDim := diffusionLayerRope(spec, arch, headDim)
	ropeScale := arch.RopeScale
	if ropeScale == 0 {
		ropeScale = 1
	}
	qRows, err = diffusionRopeRowsBF16(qRows, L, nHeads, headDim, rotaryDim, ropeBase, ropeScale, position)
	if err != nil {
		return nil, nil, nil, err
	}
	if ownsKV {
		kRows, err = diffusionRopeRowsBF16(kRows, L, nKVHeads, headDim, rotaryDim, ropeBase, ropeScale, position)
		if err != nil {
			return nil, nil, nil, err
		}
	}
	qHM := diffusionRowsToHeadMajorBF16(qRows, L, nHeads, headDim)
	kHM := diffusionConcatPrefixCanvasHeadMajor(kv.K, kRows, prefixLen, L, nKVHeads, headDim)
	vHM := diffusionConcatPrefixCanvasHeadMajor(kv.V, vRows, prefixLen, L, nKVHeads, headDim)
	attnHM, err := DiffusionSDPA(qHM, kHM, vHM, L, keyLen, nHeads, nKVHeads, headDim, scale, mask)
	if err != nil {
		return nil, nil, nil, err
	}
	attnRows := diffusionHeadMajorToRowsBF16(attnHM, L, nHeads, headDim)
	proj, err := MatRowsBF16(w.WO, attnRows, L, dModel, qDim)
	if err != nil {
		return nil, nil, nil, err
	}
	if len(w.PostAttnNormW) > 0 {
		proj, err = RMSNormBF16(proj, w.PostAttnNormW, L, dModel, arch.Eps)
		if err != nil {
			return nil, nil, nil, err
		}
	}
	h, err = AddBF16(h, proj)
	if err != nil {
		return nil, nil, nil, err
	}
	var mlp []byte
	if w.MoE != nil {
		mlp, err = diffusionMoERowsBF16(h, *w.MoE, L, dModel, diffusionLayerDFF(w, arch.FF), arch.Eps)
	} else {
		mlp, err = diffusionDenseMLPBF16(h, w, L, dModel, diffusionLayerDFF(w, arch.FF), arch.Eps)
	}
	if err != nil {
		return nil, nil, nil, err
	}
	mlp, err = diffusionApplyBF16PLE(mlp, w, ple, L, dModel, numLayers, pliDim, layer, arch.Eps)
	if err != nil {
		return nil, nil, nil, err
	}
	h, err = diffusionMulLayerScalarBF16(mlp, w.LayerScalarW, L, dModel)
	if err != nil {
		return nil, nil, nil, err
	}
	return h, kRows, vRows, nil
}

func diffusionQuantLayerForward(h []byte, w QuantizedLayerWeights, spec model.LayerSpec, kv DiffusionLayerKV, mask []float32, L, prefixLen, position, keyLen, dModel, nHeads, nKVHeads, headDim, qDim, kvDim int, arch model.Arch, scale float32, ownsKV bool, ownerCanvasKRows, ownerCanvasVRows []byte, ple []byte, pliDim, layer, numLayers, groupSize, bits int) ([]byte, []byte, []byte, error) {
	normed, err := RMSNormBF16(h, w.AttnNormW, L, dModel, arch.Eps)
	if err != nil {
		return nil, nil, nil, err
	}
	qRows, err := diffusionMatRowsQuant(w.Q, normed, L, qDim, dModel, groupSize, bits)
	if err != nil {
		return nil, nil, nil, err
	}
	if len(w.QNormW) > 0 {
		qRows, err = RMSNormBF16(qRows, w.QNormW, L*nHeads, headDim, arch.Eps)
		if err != nil {
			return nil, nil, nil, err
		}
	}
	var kRows, vRows []byte
	if ownsKV {
		kRows, err = diffusionMatRowsQuant(w.K, normed, L, kvDim, dModel, groupSize, bits)
		if err != nil {
			return nil, nil, nil, err
		}
		vRows, err = diffusionMatRowsQuant(w.V, normed, L, kvDim, dModel, groupSize, bits)
		if err != nil {
			return nil, nil, nil, err
		}
		if len(w.KNormW) > 0 {
			kRows, err = RMSNormBF16(kRows, w.KNormW, L*nKVHeads, headDim, arch.Eps)
			if err != nil {
				return nil, nil, nil, err
			}
		}
		if arch.ValueNorm {
			vRows, err = RMSNormBF16(vRows, diffusionOnesBF16(headDim), L*nKVHeads, headDim, arch.Eps)
			if err != nil {
				return nil, nil, nil, err
			}
		}
	} else {
		want := L * kvDim * bf16Size
		if len(ownerCanvasKRows) != want || len(ownerCanvasVRows) != want {
			return nil, nil, nil, core.NewError("native.DiffusionDenoiseForwardQuant: shared owner canvas K/V missing")
		}
		kRows, vRows = ownerCanvasKRows, ownerCanvasVRows
	}
	ropeBase, rotaryDim := diffusionLayerRope(spec, arch, headDim)
	ropeScale := arch.RopeScale
	if ropeScale == 0 {
		ropeScale = 1
	}
	qRows, err = diffusionRopeRowsBF16(qRows, L, nHeads, headDim, rotaryDim, ropeBase, ropeScale, position)
	if err != nil {
		return nil, nil, nil, err
	}
	if ownsKV {
		kRows, err = diffusionRopeRowsBF16(kRows, L, nKVHeads, headDim, rotaryDim, ropeBase, ropeScale, position)
		if err != nil {
			return nil, nil, nil, err
		}
	}
	qHM := diffusionRowsToHeadMajorBF16(qRows, L, nHeads, headDim)
	kHM := diffusionConcatPrefixCanvasHeadMajor(kv.K, kRows, prefixLen, L, nKVHeads, headDim)
	vHM := diffusionConcatPrefixCanvasHeadMajor(kv.V, vRows, prefixLen, L, nKVHeads, headDim)
	attnHM, err := DiffusionSDPA(qHM, kHM, vHM, L, keyLen, nHeads, nKVHeads, headDim, scale, mask)
	if err != nil {
		return nil, nil, nil, err
	}
	attnRows := diffusionHeadMajorToRowsBF16(attnHM, L, nHeads, headDim)
	proj, err := diffusionMatRowsQuant(w.O, attnRows, L, dModel, qDim, groupSize, bits)
	if err != nil {
		return nil, nil, nil, err
	}
	if len(w.PostAttnNormW) > 0 {
		proj, err = RMSNormBF16(proj, w.PostAttnNormW, L, dModel, arch.Eps)
		if err != nil {
			return nil, nil, nil, err
		}
	}
	h, err = AddBF16(h, proj)
	if err != nil {
		return nil, nil, nil, err
	}
	var mlp []byte
	if w.MoE != nil {
		mlp, err = diffusionMoERowsQuant(h, *w.MoE, L, dModel, diffusionQuantLayerDFF(w, arch.FF), arch.Eps)
	} else {
		mlp, err = diffusionDenseMLPQuant(h, w, L, dModel, diffusionQuantLayerDFF(w, arch.FF), arch.Eps)
	}
	if err != nil {
		return nil, nil, nil, err
	}
	mlp, err = diffusionApplyQuantPLE(mlp, w, ple, L, dModel, numLayers, pliDim, layer, groupSize, bits, arch.Eps)
	if err != nil {
		return nil, nil, nil, err
	}
	h, err = diffusionMulLayerScalarBF16(mlp, w.LayerScalarW, L, dModel)
	if err != nil {
		return nil, nil, nil, err
	}
	return h, kRows, vRows, nil
}

func diffusionDenseMLPBF16(h []byte, w DecodeLayerWeights, rows, dModel, dFF int, eps float32) ([]byte, error) {
	normed, err := RMSNormBF16(h, w.MLPNormW, rows, dModel, eps)
	if err != nil {
		return nil, err
	}
	gate, err := MatRowsBF16(w.WGate, normed, rows, dFF, dModel)
	if err != nil {
		return nil, err
	}
	up, err := MatRowsBF16(w.WUp, normed, rows, dFF, dModel)
	if err != nil {
		return nil, err
	}
	gated, err := GeluGateMulBF16(gate, up)
	if err != nil {
		return nil, err
	}
	down, err := MatRowsBF16(w.WDown, gated, rows, dModel, dFF)
	if err != nil {
		return nil, err
	}
	if len(w.PostFFNormW) > 0 {
		down, err = RMSNormBF16(down, w.PostFFNormW, rows, dModel, eps)
		if err != nil {
			return nil, err
		}
	}
	return AddBF16(h, down)
}

func diffusionMoERowsBF16(h []byte, w MoELayerWeights, rows, dModel, dFF int, eps float32) ([]byte, error) {
	if rows < 0 || dModel < 0 {
		return nil, core.NewError("native.DiffusionDenoiseForwardBF16: MoE dimensions must be non-negative")
	}
	if len(h) != rows*dModel*bf16Size {
		return nil, core.NewError("native.DiffusionDenoiseForwardBF16: MoE input size mismatch")
	}
	out := make([]byte, len(h))
	for r := range rows {
		row := h[r*dModel*bf16Size : (r+1)*dModel*bf16Size]
		got, err := MoEBlockBF16(row, w, dModel, dFF, eps)
		if err != nil {
			return nil, err
		}
		copy(out[r*dModel*bf16Size:(r+1)*dModel*bf16Size], got)
	}
	return out, nil
}

func diffusionDenseMLPQuant(h []byte, w QuantizedLayerWeights, rows, dModel, dFF int, eps float32) ([]byte, error) {
	normed, err := RMSNormBF16(h, w.MLPNormW, rows, dModel, eps)
	if err != nil {
		return nil, err
	}
	gate, err := diffusionMatRowsQuant(w.Gate, normed, rows, dFF, dModel, w.GroupSize, w.Bits)
	if err != nil {
		return nil, err
	}
	up, err := diffusionMatRowsQuant(w.Up, normed, rows, dFF, dModel, w.GroupSize, w.Bits)
	if err != nil {
		return nil, err
	}
	gated, err := GeluGateMulBF16(gate, up)
	if err != nil {
		return nil, err
	}
	down, err := diffusionMatRowsQuant(w.Down, gated, rows, dModel, dFF, w.GroupSize, w.Bits)
	if err != nil {
		return nil, err
	}
	if len(w.PostFFNormW) > 0 {
		down, err = RMSNormBF16(down, w.PostFFNormW, rows, dModel, eps)
		if err != nil {
			return nil, err
		}
	}
	return AddBF16(h, down)
}

func diffusionMoERowsQuant(h []byte, w MoEQuantLayerWeights, rows, dModel, dFF int, eps float32) ([]byte, error) {
	if rows < 0 || dModel < 0 {
		return nil, core.NewError("native.DiffusionDenoiseForwardQuant: MoE dimensions must be non-negative")
	}
	if len(h) != rows*dModel*bf16Size {
		return nil, core.NewError("native.DiffusionDenoiseForwardQuant: MoE input size mismatch")
	}
	out := make([]byte, len(h))
	for r := range rows {
		row := h[r*dModel*bf16Size : (r+1)*dModel*bf16Size]
		got, err := MoEBlockQuant(row, w, dModel, dFF, eps)
		if err != nil {
			return nil, err
		}
		copy(out[r*dModel*bf16Size:(r+1)*dModel*bf16Size], got)
	}
	return out, nil
}

func diffusionLayerDFF(w DecodeLayerWeights, fallback int) int {
	if w.DFF > 0 {
		return w.DFF
	}
	return fallback
}

func diffusionQuantLayerDFF(w QuantizedLayerWeights, fallback int) int {
	if w.DFF > 0 {
		return w.DFF
	}
	return fallback
}

func diffusionMatRowsQuant(w QuantWeight, x []byte, rows, outDim, inDim, groupSize, bits int) ([]byte, error) {
	if rows < 0 || outDim < 0 || inDim < 0 {
		return nil, core.NewError("native.diffusionMatRowsQuant: dimensions must be non-negative")
	}
	if len(x) != rows*inDim*bf16Size {
		return nil, core.NewError("native.diffusionMatRowsQuant: input size mismatch")
	}
	out := make([]byte, rows*outDim*bf16Size)
	if rows == 0 || outDim == 0 || inDim == 0 {
		return out, nil
	}
	if len(w.Scales) == 0 && len(w.Biases) == 0 {
		if len(w.Packed) != outDim*inDim*bf16Size {
			return nil, core.NewError("native.diffusionMatRowsQuant: dense weight size mismatch")
		}
		return MatRowsBF16(w.Packed, x, rows, outDim, inDim)
	}
	groupSize, bits = quantWeightGeometryForShape(w, outDim, inDim, groupSize, bits)
	for r := range rows {
		src := x[r*inDim*bf16Size : (r+1)*inDim*bf16Size]
		dst := out[r*outDim*bf16Size : (r+1)*outDim*bf16Size]
		row, err := QMVBF16Into(dst, src, w.Packed, w.Scales, w.Biases, outDim, inDim, groupSize, bits)
		if err != nil {
			return nil, err
		}
		if len(row) != len(dst) {
			return nil, core.NewError("native.diffusionMatRowsQuant: qmv output size mismatch")
		}
	}
	return out, nil
}

func diffusionQuantPLEInputs(g *QuantModel, arch model.Arch, canvas []int32, inputEmb []byte) ([]byte, int, error) {
	if g == nil || !g.HasPLE() || arch.PerLayerInputHidden <= 0 {
		return nil, 0, nil
	}
	pliDim, dModel, nLayers := arch.PerLayerInputHidden, arch.Hidden, len(arch.Layer)
	if len(inputEmb) != len(canvas)*dModel*bf16Size {
		return nil, 0, core.NewError("native.DiffusionDenoiseForwardQuant: PLE embedding size mismatch")
	}
	plDim := nLayers * pliDim
	out := make([]byte, len(canvas)*plDim*bf16Size)
	var projView bufView
	for i, id := range canvas {
		emb := inputEmb[i*dModel*bf16Size : (i+1)*dModel*bf16Size]
		pli, err := PerLayerInputs(g.EmbedPerLayer, g.EmbedPerLayerScales, g.EmbedPerLayerBiases, g.PerLayerModelProjW, g.PerLayerModelProjScales, g.PerLayerModelProjBiases, g.PerLayerProjNormW, id, emb, arch.PerLayerInputVocab, nLayers, pliDim, dModel, g.GroupSize, g.Bits, g.PerLayerModelProjGS, g.PerLayerModelProjBits, arch.Eps, projView)
		if err != nil {
			return nil, 0, err
		}
		if len(pli) != plDim*bf16Size {
			return nil, 0, core.NewError("native.DiffusionDenoiseForwardQuant: PLE tensor size mismatch")
		}
		copy(out[i*plDim*bf16Size:(i+1)*plDim*bf16Size], pli)
	}
	return out, pliDim, nil
}

func diffusionBF16PLEInputs(g *BF16Model, arch model.Arch, canvas []int32, inputEmb []byte) ([]byte, int, error) {
	if g == nil || !g.HasPLE() || arch.PerLayerInputHidden <= 0 {
		return nil, 0, nil
	}
	pliDim, dModel, nLayers := arch.PerLayerInputHidden, arch.Hidden, len(arch.Layer)
	if len(inputEmb) != len(canvas)*dModel*bf16Size {
		return nil, 0, core.NewError("native.DiffusionDenoiseForwardBF16: PLE embedding size mismatch")
	}
	plDim := nLayers * pliDim
	out := make([]byte, len(canvas)*plDim*bf16Size)
	for i, id := range canvas {
		emb := inputEmb[i*dModel*bf16Size : (i+1)*dModel*bf16Size]
		pli, err := PerLayerInputs(g.EmbedPerLayer, nil, nil, g.PerLayerModelProjW, nil, nil, g.PerLayerProjNormW, id, emb, arch.PerLayerInputVocab, nLayers, pliDim, dModel, 0, 0, 0, 0, arch.Eps, bufView{})
		if err != nil {
			return nil, 0, err
		}
		if len(pli) != plDim*bf16Size {
			return nil, 0, core.NewError("native.DiffusionDenoiseForwardBF16: PLE tensor size mismatch")
		}
		copy(out[i*plDim*bf16Size:(i+1)*plDim*bf16Size], pli)
	}
	return out, pliDim, nil
}

func diffusionApplyQuantPLE(h []byte, w QuantizedLayerWeights, ple []byte, rows, dModel, numLayers, pliDim, layer, groupSize, bits int, eps float32) ([]byte, error) {
	if len(ple) == 0 {
		return h, nil
	}
	if pliDim <= 0 || numLayers <= 0 || layer < 0 || layer >= numLayers {
		return nil, core.NewError("native.DiffusionDenoiseForwardQuant: invalid PLE geometry")
	}
	if len(h) != rows*dModel*bf16Size || len(ple) != rows*numLayers*pliDim*bf16Size {
		return nil, core.NewError("native.DiffusionDenoiseForwardQuant: PLE tensor size mismatch")
	}
	out := make([]byte, len(h))
	plDimBytes := numLayers * pliDim * bf16Size
	pliBytes := pliDim * bf16Size
	for r := range rows {
		hRow := h[r*dModel*bf16Size : (r+1)*dModel*bf16Size]
		pliOff := r*plDimBytes + layer*pliBytes
		pli := ple[pliOff : pliOff+pliBytes]
		row, err := PerLayerInputGateQuant(hRow, w.PerLayerGate, pli, w.PerLayerProjection, w.PostPerLayerInputNormW, dModel, pliDim, groupSize, bits, eps)
		if err != nil {
			return nil, err
		}
		copy(out[r*dModel*bf16Size:(r+1)*dModel*bf16Size], row)
	}
	return out, nil
}

func diffusionApplyBF16PLE(h []byte, w DecodeLayerWeights, ple []byte, rows, dModel, numLayers, pliDim, layer int, eps float32) ([]byte, error) {
	if len(ple) == 0 {
		return h, nil
	}
	if pliDim <= 0 || numLayers <= 0 || layer < 0 || layer >= numLayers {
		return nil, core.NewError("native.DiffusionDenoiseForwardBF16: invalid PLE geometry")
	}
	if len(h) != rows*dModel*bf16Size || len(ple) != rows*numLayers*pliDim*bf16Size {
		return nil, core.NewError("native.DiffusionDenoiseForwardBF16: PLE tensor size mismatch")
	}
	out := make([]byte, len(h))
	plDimBytes := numLayers * pliDim * bf16Size
	pliBytes := pliDim * bf16Size
	for r := range rows {
		hRow := h[r*dModel*bf16Size : (r+1)*dModel*bf16Size]
		pliOff := r*plDimBytes + layer*pliBytes
		pli := ple[pliOff : pliOff+pliBytes]
		row, err := PerLayerInputGateBF16(hRow, w.PerLayerGate, pli, w.PerLayerProjection, w.PostPerLayerInputNormW, dModel, pliDim, eps)
		if err != nil {
			return nil, err
		}
		copy(out[r*dModel*bf16Size:(r+1)*dModel*bf16Size], row)
	}
	return out, nil
}

func diffusionLayerRope(spec model.LayerSpec, arch model.Arch, headDim int) (float32, int) {
	base, rotaryDim := arch.RopeBase, arch.RotaryDim
	if spec.Attention == model.SlidingAttention {
		if arch.RopeLocalBase != 0 {
			base = arch.RopeLocalBase
		}
		if arch.RotaryDimLocal != 0 {
			rotaryDim = arch.RotaryDimLocal
		}
	}
	if base == 0 {
		base = 10000
	}
	if rotaryDim <= 0 {
		rotaryDim = headDim
	}
	return base, rotaryDim
}

func diffusionPrefixLen(kv DiffusionLayerKV, kvDim int) (int, error) {
	rowBytes := kvDim * bf16Size
	if kvDim <= 0 || rowBytes == 0 {
		return 0, core.NewError("native.DiffusionDenoiseForwardBF16: invalid KV dimensions")
	}
	if len(kv.K)%rowBytes != 0 || len(kv.V) != len(kv.K) {
		return 0, core.NewError("native.DiffusionDenoiseForwardBF16: prefix K/V size mismatch")
	}
	return len(kv.K) / rowBytes, nil
}

func diffusionLayerKVGeometry(kv DiffusionLayerKV, kvDim int) (int, int, int, error) {
	prefixLen, err := diffusionPrefixLen(kv, kvDim)
	if err != nil {
		return 0, 0, 0, err
	}
	start, position := kv.PrefixStart, kv.Position
	if start < 0 || position < 0 {
		return 0, 0, 0, core.NewError("native.DiffusionDenoiseForwardBF16: negative K/V prefix geometry")
	}
	if position == 0 {
		position = start + prefixLen
	} else if start == 0 && position > prefixLen {
		start = position - prefixLen
	}
	if position < start || position-start != prefixLen {
		return 0, 0, 0, core.NewError("native.DiffusionDenoiseForwardBF16: K/V prefix geometry mismatch")
	}
	return prefixLen, start, position, nil
}

func diffusionJoinRowsBF16(rows [][]byte, dModel int) []byte {
	out := make([]byte, len(rows)*dModel*bf16Size)
	for i, row := range rows {
		copy(out[i*dModel*bf16Size:(i+1)*dModel*bf16Size], row)
	}
	return out
}

func diffusionRopeRowsBF16(rows []byte, seq, heads, headDim, rotaryDim int, base, scale float32, offset int) ([]byte, error) {
	rowBytes := heads * headDim * bf16Size
	if len(rows) != seq*rowBytes {
		return nil, core.NewError("native.DiffusionDenoiseForwardBF16: rope row size mismatch")
	}
	out := make([]byte, len(rows))
	for i := range seq {
		row := rows[i*rowBytes : (i+1)*rowBytes]
		roped, err := RoPEDimsBF16(row, 1, heads, headDim, rotaryDim, base, scale, offset+i, false)
		if err != nil {
			return nil, err
		}
		copy(out[i*rowBytes:(i+1)*rowBytes], roped)
	}
	return out, nil
}

func diffusionRowsToHeadMajorBF16(rows []byte, seq, heads, headDim int) []byte {
	out := make([]byte, len(rows))
	for t := range seq {
		for h := range heads {
			src := (t*heads + h) * headDim * bf16Size
			dst := (h*seq + t) * headDim * bf16Size
			copy(out[dst:dst+headDim*bf16Size], rows[src:src+headDim*bf16Size])
		}
	}
	return out
}

func diffusionHeadMajorToRowsBF16(headMajor []byte, seq, heads, headDim int) []byte {
	out := make([]byte, len(headMajor))
	for h := range heads {
		for t := range seq {
			src := (h*seq + t) * headDim * bf16Size
			dst := (t*heads + h) * headDim * bf16Size
			copy(out[dst:dst+headDim*bf16Size], headMajor[src:src+headDim*bf16Size])
		}
	}
	return out
}

func diffusionConcatPrefixCanvasHeadMajor(prefixRows, canvasRows []byte, prefixLen, canvasLen, heads, headDim int) []byte {
	keyLen := prefixLen + canvasLen
	out := make([]byte, heads*keyLen*headDim*bf16Size)
	prefixHM := diffusionRowsToHeadMajorBF16(prefixRows, prefixLen, heads, headDim)
	canvasHM := diffusionRowsToHeadMajorBF16(canvasRows, canvasLen, heads, headDim)
	for h := range heads {
		dst := h * keyLen * headDim * bf16Size
		pb := prefixLen * headDim * bf16Size
		cb := canvasLen * headDim * bf16Size
		copy(out[dst:dst+pb], prefixHM[h*pb:(h+1)*pb])
		copy(out[dst+pb:dst+pb+cb], canvasHM[h*cb:(h+1)*cb])
	}
	return out
}

func diffusionMulLayerScalarBF16(h, scalar []byte, rows, dModel int) ([]byte, error) {
	if len(scalar) == 0 {
		return h, nil
	}
	vals := bf16ToF32Slice(h)
	switch len(scalar) {
	case bf16Size:
		s := bf16ToF32(scalar[0], scalar[1])
		for i := range vals {
			vals[i] *= s
		}
	case dModel * bf16Size:
		s := bf16ToF32Slice(scalar)
		for r := range rows {
			for d := range dModel {
				vals[r*dModel+d] *= s[d]
			}
		}
	default:
		return nil, core.NewError("native.DiffusionDenoiseForwardBF16: layer scalar size mismatch")
	}
	return f32ToBf16Slice(vals), nil
}
