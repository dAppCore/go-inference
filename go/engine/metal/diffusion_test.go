// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"context"
	"math"
	"strings"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

func TestDiffusionGlobalCanvasMaskData_Geometry_Good(t *testing.T) {
	const B, L, keyLen = 2, 3, 5
	values, shape := diffusionGlobalCanvasMaskData(B, L, keyLen)
	if len(shape) != 4 || shape[0] != B || shape[1] != 1 || shape[2] != L || shape[3] != keyLen {
		t.Fatalf("shape = %v, want [%d 1 %d %d]", shape, B, L, keyLen)
	}
	if len(values) != B*L*keyLen {
		t.Fatalf("values length = %d, want %d", len(values), B*L*keyLen)
	}
	for i, v := range values {
		if v != 0 {
			t.Fatalf("mask[%d] = %f, want 0", i, v)
		}
	}
}

func TestDiffusionBlockLocalCanvasMaskData_Geometry_Good(t *testing.T) {
	const (
		B      = 2
		L      = 3
		offset = 6
		window = 4
		keyLen = offset + L
	)
	values, shape := diffusionBlockLocalCanvasMaskData(B, L, keyLen, offset, window)
	if len(shape) != 4 || shape[0] != B || shape[1] != 1 || shape[2] != L || shape[3] != keyLen {
		t.Fatalf("shape = %v, want [%d 1 %d %d]", shape, B, L, keyLen)
	}
	negInf := float32(math.Inf(-1))
	for b := range B {
		for i := range L {
			for j := range keyLen {
				got := values[b*L*keyLen+i*keyLen+j]
				inContext := j >= offset-window && j < offset
				inCanvas := j >= offset && j < offset+L
				want := negInf
				if inContext || inCanvas {
					want = 0
				}
				if got != want {
					t.Fatalf("mask[%d][%d][%d] = %f, want %f", b, i, j, got, want)
				}
			}
		}
	}
}

func TestDiffusionBlockLocalCanvasMaskData_ContextClampsAtZero_Ugly(t *testing.T) {
	const B, L, offset, window, keyLen = 1, 2, 2, 8, 4
	values, _ := diffusionBlockLocalCanvasMaskData(B, L, keyLen, offset, window)
	for i, v := range values {
		if v != 0 {
			t.Fatalf("mask[%d] = %f, want all-attend when clamped context covers the prefix", i, v)
		}
	}
}

func TestDiffusionSDPAWithMaskMatchesReference_Good(t *testing.T) {
	requireNativeRuntime(t)
	const (
		qLen     = 3
		keyLen   = 5
		nHeads   = 4
		nKVHeads = 2
		headDim  = 8
	)
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	q := toBF16Bytes(bf16Round(syntheticFloat32(nHeads*qLen*headDim, 31)))
	k := toBF16Bytes(bf16Round(syntheticFloat32(nKVHeads*keyLen*headDim, 37)))
	v := toBF16Bytes(bf16Round(syntheticFloat32(nKVHeads*keyLen*headDim, 41)))
	mask := make([]float32, qLen*keyLen)
	negInf := float32(math.Inf(-1))
	mask[0*keyLen+0] = negInf
	mask[1*keyLen+1] = negInf
	mask[2*keyLen+0] = negInf
	mask[2*keyLen+1] = negInf

	got, err := DiffusionSDPA(q, k, v, qLen, keyLen, nHeads, nKVHeads, headDim, scale, mask)
	if err != nil {
		t.Fatalf("DiffusionSDPA: %v", err)
	}
	want := diffusionSDPAReference(q, k, v, qLen, keyLen, nHeads, nKVHeads, headDim, scale, mask)
	relL2, cos := relL2Cos(bf16Floats(got), bf16Floats(want))
	if relL2 > 1e-2 || cos < 0.999 {
		t.Fatalf("DiffusionSDPA rel-L2/cos = %.3e/%.6f, want masked attention reference", relL2, cos)
	}
}

func TestDiffusionDenoiseForwardBF16_UsesPrefixMask_Good(t *testing.T) {
	requireNativeRuntime(t)
	const (
		dModel  = 4
		vocab   = 4
		qLen    = 2
		keyLen  = 3
		nHeads  = 1
		nKV     = 1
		headDim = 4
		dFF     = 4
	)
	embed := toBF16Bytes([]float32{
		0, 0, 0, 0,
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
	})
	layer := DecodeLayerWeights{
		AttnNormW: toBF16Bytes(fillConst(dModel, 1)),
		WQ:        diffusionIdentityBF16(dModel, dModel),
		WK:        diffusionIdentityBF16(dModel, dModel),
		WV:        diffusionIdentityBF16(dModel, dModel),
		WO:        diffusionIdentityBF16(dModel, dModel),
		MLPNormW:  toBF16Bytes(fillConst(dModel, 1)),
		WGate:     toBF16Bytes(make([]float32, dFF*dModel)),
		WUp:       toBF16Bytes(make([]float32, dFF*dModel)),
		WDown:     toBF16Bytes(make([]float32, dModel*dFF)),
	}
	g := &BF16Model{
		Layers:    []DecodeLayerWeights{layer},
		Embed:     embed,
		FinalNorm: toBF16Bytes(fillConst(dModel, 1)),
		LMHead:    embed,
		Tied:      true,
	}
	arch := model.Arch{
		Hidden: dModel, Heads: nHeads, KVHeads: nKV, HeadDim: headDim, FF: dFF, Vocab: vocab,
		Eps: 1e-6, AttnScale: 1, RopeBase: 10000, RopeScale: 1, RotaryDim: headDim, RotaryDimLocal: headDim,
		Layer: []model.LayerSpec{{Attention: model.SlidingAttention, KVShareFrom: 0, CacheIndex: 0}},
	}
	prefix := DiffusionLayerKV{
		K: toBF16Bytes([]float32{5, 0, 0, 0}),
		V: toBF16Bytes([]float32{0, 0, 4, 0}),
	}
	globalMask := make([]float32, qLen*keyLen)
	localAll := make([]float32, qLen*keyLen)
	localBlocked := append([]float32(nil), localAll...)
	for i := range qLen {
		localBlocked[i*keyLen] = float32(math.Inf(-1))
	}

	gotAll, err := DiffusionDenoiseForwardBF16(g, nil, arch, []int32{1, 2}, nil, []DiffusionLayerKV{prefix}, globalMask, localAll)
	if err != nil {
		t.Fatalf("DiffusionDenoiseForwardBF16 all-prefix: %v", err)
	}
	gotBlocked, err := DiffusionDenoiseForwardBF16(g, nil, arch, []int32{1, 2}, nil, []DiffusionLayerKV{prefix}, globalMask, localBlocked)
	if err != nil {
		t.Fatalf("DiffusionDenoiseForwardBF16 blocked-prefix: %v", err)
	}
	if len(gotAll) != qLen*vocab*bf16Size || len(gotBlocked) != len(gotAll) {
		t.Fatalf("logits lengths = %d/%d, want %d", len(gotAll), len(gotBlocked), qLen*vocab*bf16Size)
	}
	if bytes.Equal(gotAll, gotBlocked) {
		t.Fatal("DiffusionDenoiseForwardBF16 ignored the additive prefix mask")
	}
}

func TestDiffusionDenoiseForwardBF16_UsesSelfConditioning_Good(t *testing.T) {
	requireNativeRuntime(t)
	const (
		dModel  = 4
		vocab   = 4
		qLen    = 2
		keyLen  = 3
		nHeads  = 1
		nKV     = 1
		headDim = 4
		dFF     = 4
	)
	embed := toBF16Bytes([]float32{
		0, 0, 0, 0,
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
	})
	layer := DecodeLayerWeights{
		AttnNormW: toBF16Bytes(fillConst(dModel, 1)),
		WQ:        diffusionIdentityBF16(dModel, dModel),
		WK:        diffusionIdentityBF16(dModel, dModel),
		WV:        diffusionIdentityBF16(dModel, dModel),
		WO:        diffusionIdentityBF16(dModel, dModel),
		MLPNormW:  toBF16Bytes(fillConst(dModel, 1)),
		WGate:     toBF16Bytes(make([]float32, dFF*dModel)),
		WUp:       toBF16Bytes(make([]float32, dFF*dModel)),
		WDown:     toBF16Bytes(make([]float32, dModel*dFF)),
	}
	g := &BF16Model{
		Layers:    []DecodeLayerWeights{layer},
		Embed:     embed,
		FinalNorm: toBF16Bytes(fillConst(dModel, 1)),
		LMHead:    embed,
		Tied:      true,
	}
	arch := model.Arch{
		Hidden: dModel, Heads: nHeads, KVHeads: nKV, HeadDim: headDim, FF: dFF, Vocab: vocab,
		Eps: 1e-6, AttnScale: 1, RopeBase: 10000, RopeScale: 1, RotaryDim: headDim, RotaryDimLocal: headDim,
		Layer: []model.LayerSpec{{Attention: model.SlidingAttention, KVShareFrom: 0, CacheIndex: 0}},
	}
	diffusion := &model.LoadedDiffusion{
		SelfCondPreNorm: toBF16Bytes(fillConst(dModel, 1)),
		SelfCondGate:    &model.Linear{Weight: diffusionIdentityBF16(dFF, dModel), OutDim: dFF, InDim: dModel},
		SelfCondUp:      &model.Linear{Weight: diffusionIdentityBF16(dFF, dModel), OutDim: dFF, InDim: dModel},
		SelfCondDown:    &model.Linear{Weight: diffusionIdentityBF16(dModel, dFF), OutDim: dModel, InDim: dFF},
	}
	prefix := DiffusionLayerKV{
		K: toBF16Bytes([]float32{5, 0, 0, 0}),
		V: toBF16Bytes([]float32{0, 0, 4, 0}),
	}
	mask := make([]float32, qLen*keyLen)
	without, err := DiffusionDenoiseForwardBF16(g, diffusion, arch, []int32{1, 2}, nil, []DiffusionLayerKV{prefix}, mask, mask)
	if err != nil {
		t.Fatalf("DiffusionDenoiseForwardBF16 without SCEmb: %v", err)
	}
	with, err := DiffusionDenoiseForwardBF16(g, diffusion, arch, []int32{1, 2}, toBF16Bytes(syntheticFloat32(qLen*dModel, 53)), []DiffusionLayerKV{prefix}, mask, mask)
	if err != nil {
		t.Fatalf("DiffusionDenoiseForwardBF16 with SCEmb: %v", err)
	}
	if bytes.Equal(without, with) {
		t.Fatal("DiffusionDenoiseForwardBF16 ignored the self-conditioning embedding")
	}
}

func TestDiffusionDenoiseForwardBF16_QuantSelfConditioning_Good(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers = 64, 2, 1, 32, 64, 32, 1
	const groupSize, bits = 32, 4
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	lin := func(outDim, inDim, salt int) *model.Linear {
		w := quantWeightFixture(t, outDim, inDim, groupSize, bits, salt)
		return &model.Linear{
			Weight: w.Packed, Scales: w.Scales, Biases: w.Biases,
			OutDim: outDim, InDim: inDim, GroupSize: groupSize, Bits: bits, Kind: "affine",
		}
	}
	diffusion := &model.LoadedDiffusion{
		SelfCondPreNorm: toBF16Bytes(fillConst(dModel, 1)),
		SelfCondGate:    lin(dFF, dModel, 503),
		SelfCondUp:      lin(dFF, dModel, 509),
		SelfCondDown:    lin(dModel, dFF, 521),
	}
	mask := []float32{0}
	_, err := DiffusionDenoiseForwardBF16(g, diffusion, arch, []int32{1}, toBF16Bytes(syntheticFloat32(dModel, 541)), []DiffusionLayerKV{{}}, mask, mask)
	if err != nil {
		t.Fatalf("DiffusionDenoiseForwardBF16 quant self-conditioning: %v", err)
	}
}

func TestDiffusionDenoiseForwardBF16_UsesPLE_Good(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers = 64, 2, 1, 32, 128, 32, 1
	const pliDim = 32
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	arch.PerLayerInputVocab = vocab
	arch.PerLayerInputHidden = pliDim
	g.EmbedPerLayer = toBF16Bytes(syntheticFloat32(vocab*nLayers*pliDim, 401))
	g.PerLayerModelProjW = toBF16Bytes(syntheticFloat32(nLayers*pliDim*dModel, 403))
	g.PerLayerProjNormW = toBF16Bytes(fillConst(pliDim, 1))
	g.Layers[0].PerLayerGate = toBF16Bytes(syntheticFloat32(pliDim*dModel, 409))
	g.Layers[0].PerLayerProjection = toBF16Bytes(syntheticFloat32(dModel*pliDim, 419))
	g.Layers[0].PostPerLayerInputNormW = toBF16Bytes(fillConst(dModel, 1))
	mask := []float32{0}

	withPLE, err := DiffusionDenoiseForwardBF16(g, nil, arch, []int32{1}, nil, []DiffusionLayerKV{{}}, mask, mask)
	if err != nil {
		t.Fatalf("DiffusionDenoiseForwardBF16 PLE: %v", err)
	}
	noPLE := *g
	noPLE.EmbedPerLayer = nil
	noPLE.PerLayerModelProjW = nil
	noPLE.PerLayerProjNormW = nil
	withoutPLE, err := DiffusionDenoiseForwardBF16(&noPLE, nil, arch, []int32{1}, nil, []DiffusionLayerKV{{}}, mask, mask)
	if err != nil {
		t.Fatalf("DiffusionDenoiseForwardBF16 no PLE: %v", err)
	}
	if bytes.Equal(withPLE, withoutPLE) {
		t.Fatal("DiffusionDenoiseForwardBF16 ignored the BF16 per-layer-input gate")
	}
}

func TestDiffusionDenoiseForwardBF16_ReusesOwnerKVForSharedLayer_Good(t *testing.T) {
	requireNativeRuntime(t)
	const (
		dModel  = 4
		vocab   = 4
		qLen    = 2
		keyLen  = 3
		nHeads  = 1
		nKV     = 1
		headDim = 4
		dFF     = 4
	)
	embed := toBF16Bytes([]float32{
		0, 0, 0, 0,
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
	})
	zeroWO := toBF16Bytes(make([]float32, dModel*dModel))
	zeroGate := toBF16Bytes(make([]float32, dFF*dModel))
	zeroUp := toBF16Bytes(make([]float32, dFF*dModel))
	zeroDown := toBF16Bytes(make([]float32, dModel*dFF))
	owner := DecodeLayerWeights{
		AttnNormW: toBF16Bytes(fillConst(dModel, 1)),
		WQ:        diffusionIdentityBF16(dModel, dModel),
		WK:        diffusionIdentityBF16(dModel, dModel),
		WV:        diffusionIdentityBF16(dModel, dModel),
		WO:        zeroWO,
		MLPNormW:  toBF16Bytes(fillConst(dModel, 1)),
		WGate:     zeroGate,
		WUp:       zeroUp,
		WDown:     zeroDown,
	}
	shared := DecodeLayerWeights{
		AttnNormW: toBF16Bytes(fillConst(dModel, 1)),
		WQ:        diffusionIdentityBF16(dModel, dModel),
		WO:        diffusionIdentityBF16(dModel, dModel),
		MLPNormW:  toBF16Bytes(fillConst(dModel, 1)),
		WGate:     zeroGate,
		WUp:       zeroUp,
		WDown:     zeroDown,
	}
	g := &BF16Model{
		Layers:    []DecodeLayerWeights{owner, shared},
		Embed:     embed,
		FinalNorm: toBF16Bytes(fillConst(dModel, 1)),
		LMHead:    embed,
		Tied:      true,
	}
	arch := model.Arch{
		Hidden: dModel, Heads: nHeads, KVHeads: nKV, HeadDim: headDim, FF: dFF, Vocab: vocab,
		Eps: 1e-6, AttnScale: 1, RopeBase: 10000, RopeScale: 1, RotaryDim: headDim, RotaryDimLocal: headDim,
		Layer: []model.LayerSpec{
			{Attention: model.SlidingAttention, KVShareFrom: 0, CacheIndex: 0},
			{Attention: model.SlidingAttention, KVShareFrom: 0, CacheIndex: -1},
		},
	}
	prefixA := DiffusionLayerKV{
		K: toBF16Bytes([]float32{5, 0, 0, 0}),
		V: toBF16Bytes([]float32{0, 0, 4, 0}),
	}
	prefixB := DiffusionLayerKV{
		K: toBF16Bytes([]float32{5, 0, 0, 0}),
		V: toBF16Bytes([]float32{0, 0, -4, 0}),
	}
	mask := make([]float32, qLen*keyLen)

	gotA, err := DiffusionDenoiseForwardBF16(g, nil, arch, []int32{1, 2}, nil, []DiffusionLayerKV{prefixA, {}}, mask, mask)
	if err != nil {
		t.Fatalf("DiffusionDenoiseForwardBF16 prefix A: %v", err)
	}
	gotB, err := DiffusionDenoiseForwardBF16(g, nil, arch, []int32{1, 2}, nil, []DiffusionLayerKV{prefixB, {}}, mask, mask)
	if err != nil {
		t.Fatalf("DiffusionDenoiseForwardBF16 prefix B: %v", err)
	}
	if len(gotA) != qLen*vocab*bf16Size || len(gotB) != len(gotA) {
		t.Fatalf("logits lengths = %d/%d, want %d", len(gotA), len(gotB), qLen*vocab*bf16Size)
	}
	if bytes.Equal(gotA, gotB) {
		t.Fatal("DiffusionDenoiseForwardBF16 shared layer ignored the owner K/V prefix")
	}
}

func TestArchSessionDiffusionLayerKVPrefixCapturesOwnerRows_Good(t *testing.T) {
	requireNativeRuntime(t)
	sess := newSessionStateFixture(t)
	defer sess.Close()
	prompt := []int32{1, 2, 3}
	if err := sess.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}

	got, err := sess.DiffusionLayerKVPrefix()
	if err != nil {
		t.Fatalf("DiffusionLayerKVPrefix: %v", err)
	}
	if len(got) != len(sess.arch.Layer) {
		t.Fatalf("prefix layer count = %d, want %d", len(got), len(sess.arch.Layer))
	}
	views, err := sess.stateLayerViews()
	if err != nil {
		t.Fatalf("stateLayerViews: %v", err)
	}
	for _, view := range views {
		wantK, wantV, err := stateBlockLayerBytes(view, 0, len(prompt), sess.Pos())
		if err != nil {
			t.Fatalf("stateBlockLayerBytes layer %d: %v", view.layer, err)
		}
		kv := got[view.layer]
		if kv.PrefixStart != 0 || kv.Position != len(prompt) {
			t.Fatalf("layer %d geometry = start %d position %d, want 0/%d", view.layer, kv.PrefixStart, kv.Position, len(prompt))
		}
		if !bytes.Equal(kv.K, wantK) || !bytes.Equal(kv.V, wantV) {
			t.Fatalf("layer %d K/V prefix bytes differ from resident state block rows", view.layer)
		}
		if len(kv.K) > 0 && &kv.K[0] != &wantK[0] {
			t.Fatalf("layer %d K prefix was copied; want resident row view", view.layer)
		}
		if len(kv.V) > 0 && &kv.V[0] != &wantV[0] {
			t.Fatalf("layer %d V prefix was copied; want resident row view", view.layer)
		}
	}
}

func TestArchSessionDiffusionLayerKVPrefixCarriesSlidingWindowOffset_Good(t *testing.T) {
	requireNativeRuntime(t)
	g, arch, maxLen := sessionStateFixture(t)
	arch.SlidingWindow = 4
	arch.Layer[0].Attention = model.SlidingAttention
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	defer sess.Close()
	prompt := []int32{1, 2, 3, 4, 5, 6, 7}
	if err := sess.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}

	got, err := sess.DiffusionLayerKVPrefix()
	if err != nil {
		t.Fatalf("DiffusionLayerKVPrefix: %v", err)
	}
	view := restoredStateLayerView(t, sess, 0)
	kv := got[0]
	if kv.PrefixStart != len(prompt)-arch.SlidingWindow || kv.Position != len(prompt) {
		t.Fatalf("sliding geometry = start %d position %d, want %d/%d", kv.PrefixStart, kv.Position, len(prompt)-arch.SlidingWindow, len(prompt))
	}
	wantBytes := arch.SlidingWindow * view.rowBytes
	if len(kv.K) != wantBytes || len(kv.V) != wantBytes {
		t.Fatalf("sliding K/V bytes = %d/%d, want %d", len(kv.K), len(kv.V), wantBytes)
	}
}

func TestDiffusionSessionDenoiseMasksUseResidentPrefixSpans_Good(t *testing.T) {
	const (
		kvDim         = 2
		globalPrefix  = 6
		slidingPrefix = 3
		canvasLen     = 2
	)
	arch := model.Arch{
		KVHeads:       1,
		HeadDim:       kvDim,
		SlidingWindow: 2,
		Layer: []model.LayerSpec{
			{Attention: model.GlobalAttention, KVShareFrom: 0, CacheIndex: 0},
			{Attention: model.SlidingAttention, KVShareFrom: 1, CacheIndex: 1},
		},
	}
	layerKV := []DiffusionLayerKV{
		{
			K:        make([]byte, globalPrefix*kvDim*bf16Size),
			V:        make([]byte, globalPrefix*kvDim*bf16Size),
			Position: globalPrefix,
		},
		{
			K:           make([]byte, slidingPrefix*kvDim*bf16Size),
			V:           make([]byte, slidingPrefix*kvDim*bf16Size),
			PrefixStart: globalPrefix - slidingPrefix,
			Position:    globalPrefix,
		},
	}
	req := DiffusionDenoiseRequest{
		Prefix: 99,
		Canvas: []int32{4, 5},
	}

	globalMask, localMask, err := diffusionSessionDenoiseMasks(arch, layerKV, req)
	if err != nil {
		t.Fatalf("diffusionSessionDenoiseMasks: %v", err)
	}
	if len(globalMask) != canvasLen*(globalPrefix+canvasLen) {
		t.Fatalf("global mask length = %d, want %d", len(globalMask), canvasLen*(globalPrefix+canvasLen))
	}
	for i, v := range globalMask {
		if v != 0 {
			t.Fatalf("global mask[%d] = %f, want unmasked", i, v)
		}
	}
	wantLocalLen := canvasLen * (slidingPrefix + canvasLen)
	if len(localMask) != wantLocalLen {
		t.Fatalf("local mask length = %d, want %d", len(localMask), wantLocalLen)
	}
	negInf := float32(math.Inf(-1))
	for row := range canvasLen {
		for col := range slidingPrefix + canvasLen {
			got := localMask[row*(slidingPrefix+canvasLen)+col]
			want := float32(0)
			if col == 0 {
				want = negInf
			}
			if got != want {
				t.Fatalf("local mask[%d][%d] = %f, want %f", row, col, got, want)
			}
		}
	}
}

func TestNativeTokenModelGenerateBlockDiffusionTokensBF16_Good(t *testing.T) {
	requireNativeRuntime(t)
	g, arch, maxLen := sessionStateFixture(t)
	dModel, dFF, vocab := arch.Hidden, arch.FF, arch.Vocab
	tm, err := NewBF16TokenModel(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewBF16TokenModel: %v", err)
	}
	tm.diffusion = &model.LoadedDiffusion{
		CanvasLength:    1,
		SelfCondPreNorm: toBF16Bytes(fillConst(dModel, 1)),
		SelfCondGate:    &model.Linear{Weight: diffusionIdentityBF16(dFF, dModel), OutDim: dFF, InDim: dModel},
		SelfCondUp:      &model.Linear{Weight: diffusionIdentityBF16(dFF, dModel), OutDim: dFF, InDim: dModel},
		SelfCondDown:    &model.Linear{Weight: diffusionIdentityBF16(dModel, dFF), OutDim: dModel, InDim: dFF},
	}

	var emitted []int32
	metrics, err := tm.GenerateBlockDiffusionTokens(context.Background(), []int32{1}, BlockDiffusionOptions{
		MaxTokens: 1,
		Seed:      7,
		SeedSet:   true,
	}, func(id int32) bool {
		emitted = append(emitted, id)
		return true
	})
	if err != nil {
		t.Fatalf("GenerateBlockDiffusionTokens: %v", err)
	}
	if len(emitted) != 1 {
		t.Fatalf("emitted tokens = %v, want 1 token", emitted)
	}
	for i, id := range emitted {
		if id < 0 || id >= int32(vocab) {
			t.Fatalf("emitted[%d] = %d outside vocab", i, id)
		}
	}
	if metrics.PrefillTokens != 1 || metrics.EmittedTokens != len(emitted) || metrics.TotalSteps == 0 {
		t.Fatalf("metrics = %+v, want prefill 1 emitted %d and at least one denoise step", metrics, len(emitted))
	}
}

func TestNativeTokenModelGenerateBlockDiffusionTokensQuantPLE_Good(t *testing.T) {
	requireNativeRuntime(t)
	g, arch := pleQuantModel(t, 2, 256, 32, 0)
	dModel, dFF, vocab := arch.Hidden, arch.FF, arch.Vocab
	const maxLen = 16
	tm, err := NewQuantTokenModel(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewQuantTokenModel: %v", err)
	}
	tm.diffusion = &model.LoadedDiffusion{
		CanvasLength:    1,
		SelfCondPreNorm: toBF16Bytes(fillConst(dModel, 1)),
		SelfCondGate:    &model.Linear{Weight: diffusionIdentityBF16(dFF, dModel), OutDim: dFF, InDim: dModel},
		SelfCondUp:      &model.Linear{Weight: diffusionIdentityBF16(dFF, dModel), OutDim: dFF, InDim: dModel},
		SelfCondDown:    &model.Linear{Weight: diffusionIdentityBF16(dModel, dFF), OutDim: dModel, InDim: dFF},
	}

	var emitted []int32
	metrics, err := tm.GenerateBlockDiffusionTokens(context.Background(), []int32{1}, BlockDiffusionOptions{
		MaxTokens: 1,
		Seed:      11,
		SeedSet:   true,
	}, func(id int32) bool {
		emitted = append(emitted, id)
		return true
	})
	if err != nil {
		t.Fatalf("GenerateBlockDiffusionTokens quant PLE: %v", err)
	}
	if len(emitted) != 1 {
		t.Fatalf("emitted tokens = %v, want 1 token", emitted)
	}
	for i, id := range emitted {
		if id < 0 || id >= int32(vocab) {
			t.Fatalf("emitted[%d] = %d outside vocab", i, id)
		}
	}
	if metrics.PrefillTokens != 1 || metrics.EmittedTokens != len(emitted) || metrics.TotalSteps == 0 {
		t.Fatalf("metrics = %+v, want prefill 1 emitted %d and at least one denoise step", metrics, len(emitted))
	}
}

func TestDiffusionDenoiseForwardQuantMoE_Good(t *testing.T) {
	requireNativeRuntime(t)
	g, arch := pleQuantModel(t, 1, 128, 32, 0)
	const numExperts, topK, expertDFF = 4, 2, 128
	moe := quantMoELayerWeightsGuard(t, numExperts, topK, arch.Hidden, arch.FF, expertDFF, g.GroupSize, g.Bits)
	g.Layers[0].MoE = &moe
	arch.Layer[0].MoE = true
	arch.Experts = numExperts
	arch.TopK = topK
	arch.ExpertFF = expertDFF
	mask := []float32{0}

	logits, err := DiffusionDenoiseForwardQuant(g, nil, arch, []int32{1}, nil, []DiffusionLayerKV{{}}, mask, mask)
	if err != nil {
		t.Fatalf("DiffusionDenoiseForwardQuant MoE: %v", err)
	}
	if len(logits) != arch.Vocab*bf16Size {
		t.Fatalf("logits bytes = %d, want %d", len(logits), arch.Vocab*bf16Size)
	}
}

func TestDiffusionDenoiseForwardBF16MoE_Good(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers = 64, 2, 1, 32, 128, 32, 1
	const numExperts, topK, expertDFF = 4, 2, 128
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	moe := moeLayerWeightsFixture(numExperts, topK, dModel, dFF, expertDFF, 503)
	g.Layers[0].MoE = &moe
	arch.Layer[0].MoE = true
	arch.Experts = numExperts
	arch.TopK = topK
	arch.ExpertFF = expertDFF
	mask := []float32{0}

	logits, err := DiffusionDenoiseForwardBF16(g, nil, arch, []int32{1}, nil, []DiffusionLayerKV{{}}, mask, mask)
	if err != nil {
		t.Fatalf("DiffusionDenoiseForwardBF16 MoE: %v", err)
	}
	if len(logits) != arch.Vocab*bf16Size {
		t.Fatalf("logits bytes = %d, want %d", len(logits), arch.Vocab*bf16Size)
	}
}

func diffusionIdentityBF16(rows, cols int) []byte {
	f := make([]float32, rows*cols)
	for i := 0; i < rows && i < cols; i++ {
		f[i*cols+i] = 1
	}
	return toBF16Bytes(f)
}

func diffusionSDPAReference(q, k, v []byte, qLen, keyLen, nHeads, nKVHeads, headDim int, scale float32, mask []float32) []byte {
	grp := nHeads / nKVHeads
	out := make([]byte, nHeads*qLen*headDim*bf16Size)
	for h := range nHeads {
		kvh := h / grp
		qh := bf16HeadF32(q, h, qLen, headDim)
		kh := bf16HeadF32(k, kvh, keyLen, headDim)
		vh := bf16HeadF32(v, kvh, keyLen, headDim)
		for i := range qLen {
			scores := make([]float32, keyLen)
			maxScore := float32(math.Inf(-1))
			for j := range keyLen {
				var dot float32
				for d := range headDim {
					dot += qh[i*headDim+d] * kh[j*headDim+d]
				}
				score := dot * scale
				if len(mask) > 0 {
					score += mask[i*keyLen+j]
				}
				scores[j] = score
				if score > maxScore {
					maxScore = score
				}
			}
			var denom float32
			for j := range scores {
				scores[j] = float32(math.Exp(float64(scores[j] - maxScore)))
				denom += scores[j]
			}
			base := (h*qLen + i) * headDim * bf16Size
			for d := range headDim {
				var sum float32
				for j := range keyLen {
					sum += scores[j] / denom * vh[j*headDim+d]
				}
				b := f32ToBF16(sum)
				out[base+d*bf16Size], out[base+d*bf16Size+1] = byte(b), byte(b>>8)
			}
		}
	}
	return out
}

func TestDefaultDiffusionStepConfig_Good(t *testing.T) {
	cfg := DefaultDiffusionStepConfig(262144)
	if cfg.EntropyBound != 0.3 || cfg.MaxTemperature != 0.8 || cfg.MinTemperature != 0.4 || cfg.Exponent != 1.0 {
		t.Fatalf("default diffusion step config = %+v, want reference anneal defaults", cfg)
	}
	if cfg.TextVocabSize != 262144 {
		t.Fatalf("TextVocabSize = %d, want 262144", cfg.TextVocabSize)
	}
}

func TestWithDiffusionEncoderScalarsBF16_SwapsAndRestores_Good(t *testing.T) {
	decoder0 := toBF16Bytes([]float32{1})
	decoder1 := toBF16Bytes([]float32{2})
	encoder0 := toBF16Bytes([]float32{3})
	encoder1 := toBF16Bytes([]float32{4})
	g := &BF16Model{Layers: []DecodeLayerWeights{
		{LayerScalarW: decoder0},
		{LayerScalarW: decoder1},
	}}
	diffusion := &model.LoadedDiffusion{EncoderLayerScalars: [][]byte{encoder0, encoder1}}

	called := false
	withDiffusionEncoderScalarsBF16(g, diffusion, func() {
		called = true
		eqBytes(t, "bf16 encoder scalar 0", g.Layers[0].LayerScalarW, encoder0)
		eqBytes(t, "bf16 encoder scalar 1", g.Layers[1].LayerScalarW, encoder1)
		eqBytes(t, "bf16 parked decoder scalar 0", diffusion.EncoderLayerScalars[0], decoder0)
		eqBytes(t, "bf16 parked decoder scalar 1", diffusion.EncoderLayerScalars[1], decoder1)
	})
	if !called {
		t.Fatal("callback not invoked")
	}
	eqBytes(t, "bf16 restored decoder scalar 0", g.Layers[0].LayerScalarW, decoder0)
	eqBytes(t, "bf16 restored decoder scalar 1", g.Layers[1].LayerScalarW, decoder1)
	eqBytes(t, "bf16 restored encoder scalar 0", diffusion.EncoderLayerScalars[0], encoder0)
	eqBytes(t, "bf16 restored encoder scalar 1", diffusion.EncoderLayerScalars[1], encoder1)
}

func TestWithDiffusionEncoderScalarsQuant_SwapsAndRestores_Good(t *testing.T) {
	decoder0 := toBF16Bytes([]float32{1})
	decoder1 := toBF16Bytes([]float32{2})
	encoder0 := toBF16Bytes([]float32{3})
	encoder1 := toBF16Bytes([]float32{4})
	g := &QuantModel{Layers: []QuantizedLayerWeights{
		{LayerScalarW: decoder0},
		{LayerScalarW: decoder1},
	}}
	diffusion := &model.LoadedDiffusion{EncoderLayerScalars: [][]byte{encoder0, encoder1}}

	withDiffusionEncoderScalarsQuant(g, diffusion, func() {
		eqBytes(t, "quant encoder scalar 0", g.Layers[0].LayerScalarW, encoder0)
		eqBytes(t, "quant encoder scalar 1", g.Layers[1].LayerScalarW, encoder1)
		eqBytes(t, "quant parked decoder scalar 0", diffusion.EncoderLayerScalars[0], decoder0)
		eqBytes(t, "quant parked decoder scalar 1", diffusion.EncoderLayerScalars[1], decoder1)
	})
	eqBytes(t, "quant restored decoder scalar 0", g.Layers[0].LayerScalarW, decoder0)
	eqBytes(t, "quant restored decoder scalar 1", g.Layers[1].LayerScalarW, decoder1)
	eqBytes(t, "quant restored encoder scalar 0", diffusion.EncoderLayerScalars[0], encoder0)
	eqBytes(t, "quant restored encoder scalar 1", diffusion.EncoderLayerScalars[1], encoder1)
}

func TestWithDiffusionEncoderScalars_CountMismatchRunsUnswapped_Ugly(t *testing.T) {
	decoder0 := toBF16Bytes([]float32{1})
	g := &BF16Model{Layers: []DecodeLayerWeights{{LayerScalarW: decoder0}}}
	withDiffusionEncoderScalarsBF16(g, nil, func() {
		eqBytes(t, "bf16 mismatch scalar", g.Layers[0].LayerScalarW, decoder0)
	})
	q := &QuantModel{Layers: []QuantizedLayerWeights{{LayerScalarW: decoder0}}}
	withDiffusionEncoderScalarsQuant(q, &model.LoadedDiffusion{}, func() {
		eqBytes(t, "quant mismatch scalar", q.Layers[0].LayerScalarW, decoder0)
	})
}

func TestResolveDiffusionGenerateConfig_Good(t *testing.T) {
	cfg := resolveDiffusionGenerateConfig(DiffusionGenerateConfig{}, []int32{1, 2}, 262144)
	if cfg.CanvasLength != DefaultCanvasLength || cfg.MaxSteps != DefaultMaxSteps {
		t.Fatalf("generate defaults canvas/steps = %d/%d, want %d/%d", cfg.CanvasLength, cfg.MaxSteps, DefaultCanvasLength, DefaultMaxSteps)
	}
	if cfg.StabilityThreshold != 1 || cfg.ConfidenceThreshold != 0.005 || cfg.MaxCanvases != 1 {
		t.Fatalf("generate defaults = %+v, want stability/confidence/canvases defaults", cfg)
	}
	if len(cfg.StopTokens) != 2 || cfg.StopTokens[0] != 1 || cfg.StopTokens[1] != 2 {
		t.Fatalf("StopTokens = %v, want fallback eos tokens", cfg.StopTokens)
	}
	if cfg.Step.TextVocabSize != 262144 {
		t.Fatalf("Step.TextVocabSize = %d, want fallback vocab", cfg.Step.TextVocabSize)
	}
}

func TestRunDiffusionGenerate_OrchestratesCanvases_Good(t *testing.T) {
	const (
		textVocabSize = 16
		canvasLen     = 3
		slidingWindow = 4
	)
	var (
		prefix        = 2
		prefilled     bool
		commits       [][]int32
		truncates     []int
		steps         []DiffusionDenoiseRequest
		onSteps       int
		onCanvases    int
		seenPrevSC    bool
		canvasStepSeq = map[int]int{}
	)
	cfg := DiffusionGenerateConfig{
		Step:                DefaultDiffusionStepConfig(textVocabSize),
		CanvasLength:        canvasLen,
		MaxSteps:            4,
		MaxCanvases:         2,
		StabilityThreshold:  1,
		ConfidenceThreshold: 0.01,
		StopTokens:          []int32{9},
		OnStep: func(_ int, _ int, _ DiffusionStepResult, _ time.Duration) {
			onSteps++
		},
		OnCanvas: func(_ int, _ []int32, _ int, _ time.Duration) {
			onCanvases++
		},
	}
	ops := DiffusionGenerateOps{
		Prefill: func(context.Context) (int, error) {
			prefilled = true
			return prefix, nil
		},
		CacheOffset: func() int { return prefix },
		Denoise: func(_ context.Context, req DiffusionDenoiseRequest) (DiffusionStepResult, error) {
			if req.Prefix != prefix {
				t.Fatalf("request prefix = %d, want %d", req.Prefix, prefix)
			}
			if len(req.Canvas) != canvasLen {
				t.Fatalf("request canvas len = %d, want %d", len(req.Canvas), canvasLen)
			}
			wantKeyLen := prefix + canvasLen
			if len(req.GlobalMaskShape) != 4 || req.GlobalMaskShape[2] != canvasLen || req.GlobalMaskShape[3] != wantKeyLen {
				t.Fatalf("global mask shape = %v, want [1 1 %d %d]", req.GlobalMaskShape, canvasLen, wantKeyLen)
			}
			if len(req.LocalMaskShape) != 4 || req.LocalMaskShape[2] != canvasLen || req.LocalMaskShape[3] != wantKeyLen {
				t.Fatalf("local mask shape = %v, want [1 1 %d %d]", req.LocalMaskShape, canvasLen, wantKeyLen)
			}
			if req.StepConfig.Seed != cfg.Step.Seed+uint64(req.CanvasIndex)*0x9E3779B97F4A7C15 {
				t.Fatalf("step seed = %d, want canvas-scoped seed", req.StepConfig.Seed)
			}
			if req.CanvasIndex == 0 && req.Step == 1 && string(req.SCEmb) == "canvas-0-step-0" {
				seenPrevSC = true
			}
			steps = append(steps, req)
			canvasStepSeq[req.CanvasIndex]++
			if req.CanvasIndex == 0 {
				return DiffusionStepResult{
					Canvas:      []int32{4, 5, 6},
					Greedy:      []int32{4, 5, 6},
					SCEmb:       []byte("canvas-0-step-" + string(rune('0'+req.Step))),
					MeanEntropy: 0.001,
				}, nil
			}
			return DiffusionStepResult{
				Canvas:      []int32{7, 9, 8},
				Greedy:      []int32{7, 9, 8},
				SCEmb:       []byte("canvas-1-step"),
				MeanEntropy: 0.001,
			}, nil
		},
		TruncateTo: func(p int) error {
			truncates = append(truncates, p)
			return nil
		},
		Commit: func(_ context.Context, kept []int32) error {
			commits = append(commits, append([]int32(nil), kept...))
			prefix += len(kept)
			return nil
		},
	}

	emitted, metrics, err := RunDiffusionGenerate(context.Background(), cfg, []int32{1}, textVocabSize, slidingWindow, ops)
	if err != nil {
		t.Fatalf("RunDiffusionGenerate: %v", err)
	}
	if !prefilled {
		t.Fatal("prefill was not called")
	}
	if !core.SliceEqual(emitted, []int32{4, 5, 6, 7}) {
		t.Fatalf("emitted = %v, want [4 5 6 7]", emitted)
	}
	if len(commits) != 2 || !core.SliceEqual(commits[0], []int32{4, 5, 6}) || !core.SliceEqual(commits[1], []int32{7}) {
		t.Fatalf("commits = %v, want [[4 5 6] [7]]", commits)
	}
	if len(truncates) != 4 || truncates[0] != 2 || truncates[1] != 2 || truncates[2] != 5 || truncates[3] != 5 {
		t.Fatalf("truncates = %v, want [2 2 5 5]", truncates)
	}
	if len(steps) != 4 || onSteps != 4 || onCanvases != 2 || !seenPrevSC {
		t.Fatalf("steps/onSteps/onCanvases/seenPrevSC = %d/%d/%d/%v, want 4/4/2/true", len(steps), onSteps, onCanvases, seenPrevSC)
	}
	if metrics.PrefillTokens != 2 || metrics.EmittedTokens != 4 || metrics.Canvases != 2 || metrics.TotalSteps != 4 || !metrics.StoppedOnToken {
		t.Fatalf("metrics = %+v, want prompt=2 emitted=4 canvases=2 steps=4 stopped=true", metrics)
	}
}

func TestRunDiffusionGenerate_EmptyPromptRejected_Bad(t *testing.T) {
	_, _, err := RunDiffusionGenerate(context.Background(), DiffusionGenerateConfig{MaxCanvases: 1}, nil, 8, 4, DiffusionGenerateOps{
		Prefill: func(context.Context) (int, error) { return 0, nil },
		Denoise: func(context.Context, DiffusionDenoiseRequest) (DiffusionStepResult, error) {
			t.Fatal("denoise should not run for an empty prompt")
			return DiffusionStepResult{}, nil
		},
	})
	if err == nil || !strings.Contains(err.Error(), "prompt encoded to zero tokens") {
		t.Fatalf("RunDiffusionGenerate(empty prompt) error = %v, want zero-token rejection", err)
	}
}

func TestDiffusionInitialCanvas_DeterministicAndClamped_Good(t *testing.T) {
	a := diffusionInitialCanvas(8, 16, 123, 0)
	b := diffusionInitialCanvas(8, 16, 123, 0)
	if !core.SliceEqual(a, b) {
		t.Fatalf("initial canvas with same key differs: %v vs %v", a, b)
	}
	if len(a) != 8 {
		t.Fatalf("initial canvas len = %d, want 8", len(a))
	}
	for i, id := range a {
		if id < 0 || id >= 16 {
			t.Fatalf("initial canvas[%d] = %d, want [0,16)", i, id)
		}
	}
}

func TestDiffusionKeepUntilStop_Good(t *testing.T) {
	kept, stopped := diffusionKeepUntilStop([]int32{5, 6, 7, 8}, []int32{7, 9})
	if !stopped || !core.SliceEqual(kept, []int32{5, 6}) {
		t.Fatalf("kept/stopped = %v/%v, want [5 6]/true", kept, stopped)
	}
	kept, stopped = diffusionKeepUntilStop([]int32{5, 6}, []int32{7})
	if stopped || !core.SliceEqual(kept, []int32{5, 6}) {
		t.Fatalf("kept/stopped = %v/%v, want unchanged/false", kept, stopped)
	}
}

func TestDiffusionTokenInSet_Good(t *testing.T) {
	if !tokenInSet(106, []int32{1, 106}) {
		t.Fatal("member not found")
	}
	if tokenInSet(7, []int32{1, 106}) || tokenInSet(7, nil) {
		t.Fatal("non-member reported found")
	}
}

func TestDiffusionSelfConditionBF16_NilSignalPostNormsCanvas_Good(t *testing.T) {
	const rows, dModel, dFF = 2, 4, 6
	eps := float32(1e-6)
	h := toBF16Bytes(syntheticFloat32(rows*dModel, 11))
	ones := toBF16Bytes(fillConst(dModel, 1))
	want, err := RMSNormBF16(h, ones, rows, dModel, eps)
	if err != nil {
		t.Fatalf("RMSNormBF16 reference: %v", err)
	}

	got, err := DiffusionSelfConditionBF16(h, nil, nil, nil, nil, nil, rows, dModel, dFF, eps)
	if err != nil {
		t.Fatalf("DiffusionSelfConditionBF16(nil): %v", err)
	}
	eqBytes(t, "DiffusionSelfConditionBF16 nil signal", got, want)
}

func TestDiffusionSelfConditionBF16_WithSignalMatchesMetalFormula_Good(t *testing.T) {
	const rows, dModel, dFF = 2, 4, 6
	eps := float32(1e-6)
	h := toBF16Bytes(syntheticFloat32(rows*dModel, 21))
	scEmb := toBF16Bytes(syntheticFloat32(rows*dModel, 22))
	preNorm := toBF16Bytes(syntheticFloat32(dModel, 23))
	wGate := toBF16Bytes(syntheticFloat32(dFF*dModel, 24))
	wUp := toBF16Bytes(syntheticFloat32(dFF*dModel, 25))
	wDown := toBF16Bytes(syntheticFloat32(dModel*dFF, 26))

	normed, err := RMSNormBF16(scEmb, preNorm, rows, dModel, eps)
	if err != nil {
		t.Fatalf("RMSNormBF16 reference: %v", err)
	}
	gate, err := MatRowsBF16(wGate, normed, rows, dFF, dModel)
	if err != nil {
		t.Fatalf("MatRowsBF16 gate reference: %v", err)
	}
	up, err := MatRowsBF16(wUp, normed, rows, dFF, dModel)
	if err != nil {
		t.Fatalf("MatRowsBF16 up reference: %v", err)
	}
	gated, err := GeluGateMulBF16(gate, up)
	if err != nil {
		t.Fatalf("GeluGateMulBF16 reference: %v", err)
	}
	ffw, err := MatRowsBF16(wDown, gated, rows, dModel, dFF)
	if err != nil {
		t.Fatalf("MatRowsBF16 down reference: %v", err)
	}
	combined, err := AddBF16(h, ffw)
	if err != nil {
		t.Fatalf("AddBF16 reference: %v", err)
	}
	want, err := RMSNormBF16(combined, toBF16Bytes(fillConst(dModel, 1)), rows, dModel, eps)
	if err != nil {
		t.Fatalf("RMSNormBF16 post reference: %v", err)
	}

	got, err := DiffusionSelfConditionBF16(h, scEmb, preNorm, wGate, wUp, wDown, rows, dModel, dFF, eps)
	if err != nil {
		t.Fatalf("DiffusionSelfConditionBF16: %v", err)
	}
	eqBytes(t, "DiffusionSelfConditionBF16 formula", got, want)
}

func TestDiffusionEncodeLogitsBF16_MatchesSoftmaxEmbeddingScale_Good(t *testing.T) {
	const rows, vocab, dModel = 2, 4, 3
	logits := toBF16Bytes([]float32{
		0.25, -0.75, 1.5, 0.0,
		-1.25, 0.5, 0.75, 1.25,
	})
	embed := toBF16Bytes([]float32{
		0.20, -0.10, 0.30,
		-0.40, 0.25, 0.15,
		0.50, -0.35, 0.10,
		-0.15, 0.45, -0.25,
	})
	want := diffusionEncodeLogitsReference(bf16Floats(logits), bf16Floats(embed), rows, vocab, dModel)

	got, err := DiffusionEncodeLogitsBF16(logits, embed, rows, vocab, dModel)
	if err != nil {
		t.Fatalf("DiffusionEncodeLogitsBF16: %v", err)
	}
	eqBytes(t, "DiffusionEncodeLogitsBF16 dense encode", got, want)
}

func TestDiffusionEncodeLogitsQuant_MatchesDenseDequant_Good(t *testing.T) {
	const rows, vocab, dModel, groupSize, bits = 2, 4, 32, 32, 4
	logits := toBF16Bytes([]float32{
		0.25, -0.75, 1.5, 0.0,
		-1.25, 0.5, 0.75, 1.25,
	})
	q := quantWeightFixture(t, vocab, dModel, groupSize, bits, 51)
	dense := diffusionDequant4RowsReference(q.Packed, q.Scales, q.Biases, vocab, dModel, groupSize)
	want := diffusionEncodeLogitsReference(bf16Floats(logits), dense, rows, vocab, dModel)

	got, err := DiffusionEncodeLogitsQuant(logits, q.Packed, q.Scales, q.Biases, rows, vocab, dModel, groupSize, bits)
	if err != nil {
		t.Fatalf("DiffusionEncodeLogitsQuant: %v", err)
	}
	eqBytes(t, "DiffusionEncodeLogitsQuant", got, want)
}

func TestDiffusionSampleDenoiseStepBF16_PeakedLogitsAcceptAll_Good(t *testing.T) {
	const L, V, D = 4, 8, 4
	peaks := []int32{3, 1, 7, 0}
	logitsF := make([]float32, L*V)
	for i, p := range peaks {
		logitsF[i*V+int(p)] = 32
	}
	embed := toBF16Bytes(syntheticFloat32(V*D, 41))
	cfg := DefaultDiffusionStepConfig(V)
	cfg.Seed = 7
	prev := []int32{0, 0, 0, 0}

	res, err := DiffusionSampleDenoiseStepBF16(toBF16Bytes(logitsF), embed, prev, V, D, 0, 1.0, cfg)
	if err != nil {
		t.Fatalf("DiffusionSampleDenoiseStepBF16: %v", err)
	}
	if len(res.Canvas) != L || len(res.Greedy) != L {
		t.Fatalf("canvas/greedy lengths = %d/%d, want %d", len(res.Canvas), len(res.Greedy), L)
	}
	for i, p := range peaks {
		if res.Greedy[i] != p {
			t.Fatalf("Greedy[%d] = %d, want peak %d", i, res.Greedy[i], p)
		}
		if res.Canvas[i] != p {
			t.Fatalf("Canvas[%d] = %d, want accepted peak %d", i, res.Canvas[i], p)
		}
	}
	if res.Accepted != L {
		t.Fatalf("Accepted = %d, want all %d under near-zero entropy", res.Accepted, L)
	}
	if res.Changed != 3 {
		t.Fatalf("Changed = %d, want 3 vs previous canvas", res.Changed)
	}
	if res.MeanEntropy > 0.01 {
		t.Fatalf("MeanEntropy = %f, want ~0 for peaked logits", res.MeanEntropy)
	}
	if len(res.SCEmb) != L*D*bf16Size {
		t.Fatalf("SCEmb len = %d, want %d", len(res.SCEmb), L*D*bf16Size)
	}
}

func TestDiffusionSampleDenoiseStepQuant_PeakedLogitsAcceptAll_Good(t *testing.T) {
	const L, V, D, groupSize, bits = 4, 8, 32, 32, 4
	peaks := []int32{3, 1, 7, 0}
	logitsF := make([]float32, L*V)
	for i, p := range peaks {
		logitsF[i*V+int(p)] = 32
	}
	q := quantWeightFixture(t, V, D, groupSize, bits, 61)
	cfg := DefaultDiffusionStepConfig(V)
	cfg.Seed = 7

	res, err := DiffusionSampleDenoiseStepQuant(toBF16Bytes(logitsF), q.Packed, q.Scales, q.Biases, []int32{0, 0, 0, 0}, V, D, groupSize, bits, 0, 1.0, cfg)
	if err != nil {
		t.Fatalf("DiffusionSampleDenoiseStepQuant: %v", err)
	}
	for i, p := range peaks {
		if res.Greedy[i] != p || res.Canvas[i] != p {
			t.Fatalf("row %d greedy/canvas = %d/%d, want peak %d", i, res.Greedy[i], res.Canvas[i], p)
		}
	}
	if res.Accepted != L {
		t.Fatalf("Accepted = %d, want all %d under near-zero entropy", res.Accepted, L)
	}
	if len(res.SCEmb) != L*D*bf16Size {
		t.Fatalf("SCEmb len = %d, want %d", len(res.SCEmb), L*D*bf16Size)
	}
}

func TestDiffusionSampleDenoiseStepBF16_FlatLogitsRespectBudget_Bad(t *testing.T) {
	const L, V, D = 4, 8, 2
	embed := toBF16Bytes(syntheticFloat32(V*D, 42))
	cfg := DefaultDiffusionStepConfig(V)
	cfg.Seed = 11

	res, err := DiffusionSampleDenoiseStepBF16(toBF16Bytes(make([]float32, L*V)), embed, []int32{0, 0, 0, 0}, V, D, 0, 1.0, cfg)
	if err != nil {
		t.Fatalf("DiffusionSampleDenoiseStepBF16: %v", err)
	}
	if res.Accepted != 1 {
		t.Fatalf("Accepted = %d, want exactly 1 under the entropy budget on flat logits", res.Accepted)
	}
	if res.MeanEntropy < 1.5 {
		t.Fatalf("MeanEntropy = %f, want ~ln(8) for flat logits", res.MeanEntropy)
	}
	if len(res.SCEmb) != L*D*bf16Size {
		t.Fatalf("SCEmb len = %d, want %d", len(res.SCEmb), L*D*bf16Size)
	}
}

func TestDiffusionEncodeLogitsBF16_RejectsBadShapes_Bad(t *testing.T) {
	const rows, vocab, dModel = 2, 4, 3
	logits := toBF16Bytes(syntheticFloat32(rows*vocab, 31))
	embed := toBF16Bytes(syntheticFloat32(vocab*dModel, 32))
	if _, err := DiffusionEncodeLogitsBF16(logits[:len(logits)-1], embed, rows, vocab, dModel); err == nil {
		t.Fatal("DiffusionEncodeLogitsBF16 accepted truncated logits")
	}
	if _, err := DiffusionEncodeLogitsBF16(logits, embed[:len(embed)-1], rows, vocab, dModel); err == nil {
		t.Fatal("DiffusionEncodeLogitsBF16 accepted truncated embedding")
	}
}

func diffusionEncodeLogitsReference(logits, embed []float32, rows, vocab, dModel int) []byte {
	out := make([]float32, rows*dModel)
	scale := float32(math.Sqrt(float64(dModel)))
	for r := range rows {
		row := logits[r*vocab : (r+1)*vocab]
		maxLogit := row[0]
		for _, v := range row[1:] {
			if v > maxLogit {
				maxLogit = v
			}
		}
		probs := make([]float32, vocab)
		var denom float32
		for i, v := range row {
			p := float32(math.Exp(float64(v - maxLogit)))
			probs[i] = p
			denom += p
		}
		for d := range dModel {
			var sum float32
			for v := range vocab {
				sum += (probs[v] / denom) * embed[v*dModel+d]
			}
			out[r*dModel+d] = sum * scale
		}
	}
	return toBF16Bytes(out)
}

func diffusionDequant4RowsReference(packed, scales, biases []byte, rows, cols, groupSize int) []float32 {
	out := make([]float32, rows*cols)
	rowPacked := cols / 2
	rowSB := (cols / groupSize) * bf16Size
	for r := range rows {
		pRow := packed[r*rowPacked : (r+1)*rowPacked]
		sRow := scales[r*rowSB : (r+1)*rowSB]
		bRow := biases[r*rowSB : (r+1)*rowSB]
		for c := range cols {
			group := c / groupSize
			scale := bf16ToF32(sRow[group*bf16Size], sRow[group*bf16Size+1])
			bias := bf16ToF32(bRow[group*bf16Size], bRow[group*bf16Size+1])
			var code byte
			if c&1 == 0 {
				code = pRow[c/2] & 0x0f
			} else {
				code = pRow[c/2] >> 4
			}
			out[r*cols+c] = scale*float32(code) + bias
		}
	}
	return out
}
