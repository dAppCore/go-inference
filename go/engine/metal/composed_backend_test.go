// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"

	"dappco.re/go/inference/model/composed"
	"dappco.re/go/inference/model/qwen3"
)

func cbSyn(n, seed int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32((i*seed+11)%29-14) * 0.03
	}
	return v
}

// TestComposedDeviceVsHost runs a one-layer composed forward (attention mixer + MLP) with native's
// device-GEMM hook (init-wired) and confirms the logits match a host run (hook nil'd) within f32
// tolerance — the device path is the projection swap only. D/FF sit above composed.deviceMinWork so
// the hook genuinely engages on the MLP and head matmuls.
func TestComposedDeviceVsHost(t *testing.T) {
	if composed.ProjMatMulInto == nil {
		t.Fatal("native init did not wire composed.ProjMatMulInto")
	}
	const D, FF, vocab, heads, hd = 512, 2048, 4096, 4, 128
	m := &composed.ComposedModel{
		Embed: cbSyn(vocab*D, 1), NormF: cbSyn(D, 2), D: D, Vocab: vocab, Eps: 1e-6,
		Layers: []composed.Layer{{
			InputNorm:    cbSyn(D, 3),
			PostAttnNorm: cbSyn(D, 4),
			MLP:          &composed.MLP{Gate: cbSyn(FF*D, 5), Up: cbSyn(FF*D, 6), Down: cbSyn(D*FF, 7), FF: FF},
			Mixer: composed.NewAttnMixer(&composed.AttnWeights{
				QProj: cbSyn(heads*hd*D, 8), KProj: cbSyn(heads*hd*D, 9),
				VProj: cbSyn(heads*hd*D, 10), OProj: cbSyn(D*heads*hd, 11),
				QNorm: cbSyn(hd, 12), KNorm: cbSyn(hd, 13),
			}, composed.AttnConfig{Heads: heads, KVHeads: heads, HeadDim: hd, RotaryDim: hd / 2, RopeTheta: 1e6, NormEps: 1e-6}),
		}},
	}
	tokens := []int32{5, 9, 21}

	dev, err := composed.NewSession(m).Forward(tokens)
	if err != nil {
		t.Fatalf("device forward: %v", err)
	}
	saved, savedMLP := composed.ProjMatMulInto, composed.MLPDevice
	composed.ProjMatMulInto, composed.MLPDevice = nil, nil
	host, herr := composed.NewSession(m).Forward(tokens)
	composed.ProjMatMulInto, composed.MLPDevice = saved, savedMLP
	if herr != nil {
		t.Fatalf("host forward: %v", herr)
	}
	if len(dev) != len(host) {
		t.Fatalf("length: device %d host %d", len(dev), len(host))
	}
	for i := range dev {
		if math.Abs(float64(dev[i]-host[i])) > 1e-2*(1+math.Abs(float64(host[i]))) {
			t.Fatalf("logits[%d]: device %v host %v (device GEMM diverged)", i, dev[i], host[i])
		}
	}
	t.Logf("composed forward: device-GEMM projections match host within f32 tol over %d logits", len(dev))
}

// TestComposedAttnQKVFuseDeviceVsHost exercises the fused attention projection (q/k/v in ONE command
// buffer) at a shape whose q projection crosses composed.deviceMinWork, so AttnQKVDevice genuinely
// engages (k/v are sub-floor free riders), and confirms the logits match a pure-host run within f32
// tolerance. A call-counter around the hook asserts the fuse actually fired. heads·hd = 8·128 = 1024
// with 3 prefill tokens ⇒ q's L·D·qCols = 3·512·1024 = 1,572,864 ≥ 1<<20 opens the fuse gate; the
// KV heads (2·128 = 256) stay sub-floor, riding the fused CB for free.
func TestComposedAttnQKVFuseDeviceVsHost(t *testing.T) {
	if composed.AttnQKVDevice == nil {
		t.Fatal("native init did not wire composed.AttnQKVDevice")
	}
	const D, FF, vocab, heads, kvheads, hd = 512, 2048, 4096, 8, 2, 128
	m := &composed.ComposedModel{
		Embed: cbSyn(vocab*D, 1), NormF: cbSyn(D, 2), D: D, Vocab: vocab, Eps: 1e-6,
		Layers: []composed.Layer{{
			InputNorm:    cbSyn(D, 3),
			PostAttnNorm: cbSyn(D, 4),
			MLP:          &composed.MLP{Gate: cbSyn(FF*D, 5), Up: cbSyn(FF*D, 6), Down: cbSyn(D*FF, 7), FF: FF},
			Mixer: composed.NewAttnMixer(&composed.AttnWeights{
				QProj: cbSyn(heads*hd*D, 8), KProj: cbSyn(kvheads*hd*D, 9),
				VProj: cbSyn(kvheads*hd*D, 10), OProj: cbSyn(D*heads*hd, 11),
				QNorm: cbSyn(hd, 12), KNorm: cbSyn(hd, 13),
			}, composed.AttnConfig{Heads: heads, KVHeads: kvheads, HeadDim: hd, RotaryDim: hd / 2, RopeTheta: 1e6, NormEps: 1e-6}),
		}},
	}
	tokens := []int32{5, 9, 21}

	calls := 0
	savedFuse := composed.AttnQKVDevice
	composed.AttnQKVDevice = func(h, qW, kW, vW []float32, L, D, qCols, kvCols int) ([]float32, []float32, []float32, error) {
		calls++
		return ComposedAttnQKVDevice(h, qW, kW, vW, L, D, qCols, kvCols)
	}
	dev, err := composed.NewSession(m).Forward(tokens)
	composed.AttnQKVDevice = savedFuse
	if err != nil {
		t.Fatalf("device (fused) forward: %v", err)
	}
	if calls == 0 {
		t.Fatal("fused q/k/v hook never engaged — shape below device floor?")
	}

	savedFuse2, savedProj, savedMLP := composed.AttnQKVDevice, composed.ProjMatMulInto, composed.MLPDevice
	composed.AttnQKVDevice, composed.ProjMatMulInto, composed.MLPDevice = nil, nil, nil
	host, herr := composed.NewSession(m).Forward(tokens)
	composed.AttnQKVDevice, composed.ProjMatMulInto, composed.MLPDevice = savedFuse2, savedProj, savedMLP
	if herr != nil {
		t.Fatalf("host forward: %v", herr)
	}
	if len(dev) != len(host) {
		t.Fatalf("length: device %d host %d", len(dev), len(host))
	}
	for i := range dev {
		if math.Abs(float64(dev[i]-host[i])) > 1e-2*(1+math.Abs(float64(host[i]))) {
			t.Fatalf("logits[%d]: device %v host %v (fused q/k/v GEMM diverged)", i, dev[i], host[i])
		}
	}
	t.Logf("composed attn q/k/v fuse: %d fused-CB call(s); device matches host within f32 tol over %d logits", calls, len(dev))
}

// TestComposedResidualNormMLPFuseDeviceVsHost exercises the fused FFN-tail primitive (mixer residual add +
// post-attn RMSNorm + SwiGLU MLP + MLP residual add, all in ONE command buffer) at a shape whose MLP
// crosses composed.deviceMinWork so the tail hook genuinely engages, and confirms the logits match a
// pure-host run within f32 tolerance. A call-counter around the hook asserts the fuse actually fired.
// L·D·FF = 3·512·2048 = 3,145,728 ≥ 1<<20 opens the fuse gate.
func TestComposedResidualNormMLPFuseDeviceVsHost(t *testing.T) {
	if composed.ResidualNormMLPDevice == nil {
		t.Fatal("native init did not wire composed.ResidualNormMLPDevice")
	}
	const D, FF, vocab, heads, hd = 512, 2048, 4096, 4, 128
	m := &composed.ComposedModel{
		Embed: cbSyn(vocab*D, 1), NormF: cbSyn(D, 2), D: D, Vocab: vocab, Eps: 1e-6,
		Layers: []composed.Layer{{
			InputNorm:    cbSyn(D, 3),
			PostAttnNorm: cbSyn(D, 4),
			MLP:          &composed.MLP{Gate: cbSyn(FF*D, 5), Up: cbSyn(FF*D, 6), Down: cbSyn(D*FF, 7), FF: FF},
			Mixer: composed.NewAttnMixer(&composed.AttnWeights{
				QProj: cbSyn(heads*hd*D, 8), KProj: cbSyn(heads*hd*D, 9),
				VProj: cbSyn(heads*hd*D, 10), OProj: cbSyn(D*heads*hd, 11),
				QNorm: cbSyn(hd, 12), KNorm: cbSyn(hd, 13),
			}, composed.AttnConfig{Heads: heads, KVHeads: heads, HeadDim: hd, RotaryDim: hd / 2, RopeTheta: 1e6, NormEps: 1e-6}),
		}},
	}
	tokens := []int32{5, 9, 21}

	// The proj-fused tail (ResidualNormMLPProjDevice) supersedes the plain tail for a projMixer when both are
	// wired; disable it here so this test exercises the plain ResidualNormMLPDevice path it targets.
	savedProjTail := composed.ResidualNormMLPProjDevice
	composed.ResidualNormMLPProjDevice = nil
	defer func() { composed.ResidualNormMLPProjDevice = savedProjTail }()

	calls := 0
	savedTail := composed.ResidualNormMLPDevice
	composed.ResidualNormMLPDevice = func(h, mixOut, normW, gate, up, down []float32, L, D, FF int, eps float32) ([]float32, error) {
		calls++
		return ResidualNormMLPDevice(h, mixOut, normW, gate, up, down, L, D, FF, eps)
	}
	dev, err := composed.NewSession(m).Forward(tokens)
	composed.ResidualNormMLPDevice = savedTail
	if err != nil {
		t.Fatalf("device (fused tail) forward: %v", err)
	}
	if calls == 0 {
		t.Fatal("fused FFN-tail hook never engaged — shape below device floor?")
	}

	savedTail2, savedProj, savedMLP, savedFuse := composed.ResidualNormMLPDevice, composed.ProjMatMulInto, composed.MLPDevice, composed.AttnQKVDevice
	composed.ResidualNormMLPDevice, composed.ProjMatMulInto, composed.MLPDevice, composed.AttnQKVDevice = nil, nil, nil, nil
	host, herr := composed.NewSession(m).Forward(tokens)
	composed.ResidualNormMLPDevice, composed.ProjMatMulInto, composed.MLPDevice, composed.AttnQKVDevice = savedTail2, savedProj, savedMLP, savedFuse
	if herr != nil {
		t.Fatalf("host forward: %v", herr)
	}
	if len(dev) != len(host) {
		t.Fatalf("length: device %d host %d", len(dev), len(host))
	}
	for i := range dev {
		if math.Abs(float64(dev[i]-host[i])) > 1e-2*(1+math.Abs(float64(host[i]))) {
			t.Fatalf("logits[%d]: device %v host %v (fused FFN-tail diverged)", i, dev[i], host[i])
		}
	}
	t.Logf("composed FFN-tail fuse: %d fused-CB call(s); device matches host within f32 tol over %d logits", calls, len(dev))
}

// TestComposedResidualNormMLPProjFuseDeviceVsHost exercises the projection-fused FFN-tail primitive: the
// attention mixer's o_proj folded onto the front of the tail CB (o_proj + mixer residual + post-attn RMSNorm
// + SwiGLU MLP + MLP residual, all in ONE command buffer), and confirms the logits match a pure-host run
// within f32 tolerance. A call-counter around the hook asserts the proj-fused path actually fired — the tail
// gate L·D·FF = 3·512·2048 = 3,145,728 ≥ 1<<20 opens it; the o_proj (mixCols = heads·hd = 512) rides free.
func TestComposedResidualNormMLPProjFuseDeviceVsHost(t *testing.T) {
	if composed.ResidualNormMLPProjDevice == nil {
		t.Fatal("native init did not wire composed.ResidualNormMLPProjDevice")
	}
	const D, FF, vocab, heads, hd = 512, 2048, 4096, 4, 128
	m := &composed.ComposedModel{
		Embed: cbSyn(vocab*D, 1), NormF: cbSyn(D, 2), D: D, Vocab: vocab, Eps: 1e-6,
		Layers: []composed.Layer{{
			InputNorm:    cbSyn(D, 3),
			PostAttnNorm: cbSyn(D, 4),
			MLP:          &composed.MLP{Gate: cbSyn(FF*D, 5), Up: cbSyn(FF*D, 6), Down: cbSyn(D*FF, 7), FF: FF},
			Mixer: composed.NewAttnMixer(&composed.AttnWeights{
				QProj: cbSyn(heads*hd*D, 8), KProj: cbSyn(heads*hd*D, 9),
				VProj: cbSyn(heads*hd*D, 10), OProj: cbSyn(D*heads*hd, 11),
				QNorm: cbSyn(hd, 12), KNorm: cbSyn(hd, 13),
			}, composed.AttnConfig{Heads: heads, KVHeads: heads, HeadDim: hd, RotaryDim: hd / 2, RopeTheta: 1e6, NormEps: 1e-6}),
		}},
	}
	tokens := []int32{5, 9, 21}

	calls := 0
	savedProjTail := composed.ResidualNormMLPProjDevice
	composed.ResidualNormMLPProjDevice = func(mixerHidden, projW, h, normW, gate, up, down []float32, L, D, mixCols, FF int, eps float32) ([]float32, error) {
		calls++
		return ResidualNormMLPProjDevice(mixerHidden, projW, h, normW, gate, up, down, L, D, mixCols, FF, eps)
	}
	dev, err := composed.NewSession(m).Forward(tokens)
	composed.ResidualNormMLPProjDevice = savedProjTail
	if err != nil {
		t.Fatalf("device (proj-fused tail) forward: %v", err)
	}
	if calls == 0 {
		t.Fatal("proj-fused FFN-tail hook never engaged — shape below device floor?")
	}

	savedProjTail2, savedTail, savedProj, savedMLP, savedFuse := composed.ResidualNormMLPProjDevice, composed.ResidualNormMLPDevice, composed.ProjMatMulInto, composed.MLPDevice, composed.AttnQKVDevice
	composed.ResidualNormMLPProjDevice, composed.ResidualNormMLPDevice, composed.ProjMatMulInto, composed.MLPDevice, composed.AttnQKVDevice = nil, nil, nil, nil, nil
	host, herr := composed.NewSession(m).Forward(tokens)
	composed.ResidualNormMLPProjDevice, composed.ResidualNormMLPDevice, composed.ProjMatMulInto, composed.MLPDevice, composed.AttnQKVDevice = savedProjTail2, savedTail, savedProj, savedMLP, savedFuse
	if herr != nil {
		t.Fatalf("host forward: %v", herr)
	}
	if len(dev) != len(host) {
		t.Fatalf("length: device %d host %d", len(dev), len(host))
	}
	for i := range dev {
		if math.Abs(float64(dev[i]-host[i])) > 1e-2*(1+math.Abs(float64(host[i]))) {
			t.Fatalf("logits[%d]: device %v host %v (proj-fused FFN-tail diverged)", i, dev[i], host[i])
		}
	}
	t.Logf("composed o_proj+FFN-tail fuse: %d fused-CB call(s); device matches host within f32 tol over %d logits", calls, len(dev))
}

// TestComposedGatedDeltaProjFuseDeviceVsHost exercises the projection-fused FFN-tail for a GATED-DELTA mixer
// layer: out_proj folded onto the front of the tail CB (out_proj + mixer residual + post-attn RMSNorm +
// SwiGLU MLP + MLP residual, all in ONE command buffer), and confirms the logits match a pure-host run
// within f32 tolerance. A call-counter around the hook asserts the proj-fused path fired. The tail gate
// L·D·FF = 3·768·2048 = 4,718,592 ≥ 1<<20 opens it; the out_proj (mixCols = vDim = 512) rides free.
func TestComposedGatedDeltaProjFuseDeviceVsHost(t *testing.T) {
	if composed.ResidualNormMLPProjDevice == nil {
		t.Fatal("native init did not wire composed.ResidualNormMLPProjDevice")
	}
	const D, FF, vocab = 768, 2048, 4096
	cfg := qwen3.GatedDeltaConfig{KeyHeads: 8, ValueHeads: 8, HeadDim: 64, ConvKernel: 4, Eps: 1e-5}
	qDim, vDim := cfg.KeyHeads*cfg.HeadDim, cfg.ValueHeads*cfg.HeadDim
	convDim := 2*qDim + vDim
	gdw := &qwen3.GatedDeltaWeights{
		InProjQKV:  cbSyn(convDim*D, 11),
		ConvWeight: cbSyn(convDim*cfg.ConvKernel, 12),
		ConvBias:   cbSyn(convDim, 13),
		InProjA:    cbSyn(cfg.ValueHeads*D, 14),
		ALog:       cbSyn(cfg.ValueHeads, 15),
		DtBias:     cbSyn(cfg.ValueHeads, 16),
		InProjB:    cbSyn(cfg.ValueHeads*D, 17),
		InProjZ:    cbSyn(vDim*D, 18),
		Norm:       cbSyn(cfg.HeadDim, 19),
		OutProj:    cbSyn(D*vDim, 20),
	}
	m := &composed.ComposedModel{
		Embed: cbSyn(vocab*D, 1), NormF: cbSyn(D, 2), D: D, Vocab: vocab, Eps: 1e-6,
		Layers: []composed.Layer{{
			InputNorm:    cbSyn(D, 3),
			PostAttnNorm: cbSyn(D, 4),
			MLP:          &composed.MLP{Gate: cbSyn(FF*D, 5), Up: cbSyn(FF*D, 6), Down: cbSyn(D*FF, 7), FF: FF},
			Mixer:        composed.NewGatedDeltaMixer(gdw, cfg),
		}},
	}
	tokens := []int32{5, 9, 21}

	calls := 0
	savedProjTail := composed.ResidualNormMLPProjDevice
	composed.ResidualNormMLPProjDevice = func(mixerHidden, projW, h, normW, gate, up, down []float32, L, D, mixCols, FF int, eps float32) ([]float32, error) {
		calls++
		return ResidualNormMLPProjDevice(mixerHidden, projW, h, normW, gate, up, down, L, D, mixCols, FF, eps)
	}
	dev, err := composed.NewSession(m).Forward(tokens)
	composed.ResidualNormMLPProjDevice = savedProjTail
	if err != nil {
		t.Fatalf("device (proj-fused tail) forward: %v", err)
	}
	if calls == 0 {
		t.Fatal("gated-delta proj-fused FFN-tail hook never engaged — shape below device floor?")
	}

	savedProjTail2, savedTail, savedCProj, savedProj, savedInto, savedMLP, savedInput := composed.ResidualNormMLPProjDevice, composed.ResidualNormMLPDevice, composed.ProjMatMulInto, qwen3.ProjMatMul, qwen3.ProjMatMulInto, composed.MLPDevice, qwen3.GatedDeltaInputDevice
	composed.ResidualNormMLPProjDevice, composed.ResidualNormMLPDevice, composed.ProjMatMulInto, qwen3.ProjMatMul, qwen3.ProjMatMulInto, composed.MLPDevice, qwen3.GatedDeltaInputDevice = nil, nil, nil, nil, nil, nil, nil
	host, herr := composed.NewSession(m).Forward(tokens)
	composed.ResidualNormMLPProjDevice, composed.ResidualNormMLPDevice, composed.ProjMatMulInto, qwen3.ProjMatMul, qwen3.ProjMatMulInto, composed.MLPDevice, qwen3.GatedDeltaInputDevice = savedProjTail2, savedTail, savedCProj, savedProj, savedInto, savedMLP, savedInput
	if herr != nil {
		t.Fatalf("host forward: %v", herr)
	}
	if len(dev) != len(host) {
		t.Fatalf("length: device %d host %d", len(dev), len(host))
	}
	for i := range dev {
		if math.Abs(float64(dev[i]-host[i])) > 1e-2*(1+math.Abs(float64(host[i]))) {
			t.Fatalf("logits[%d]: device %v host %v (gated-delta proj-fused FFN-tail diverged)", i, dev[i], host[i])
		}
	}
	t.Logf("composed gated-delta out_proj+FFN-tail fuse: %d fused-CB call(s); device matches host within f32 tol over %d logits", calls, len(dev))
}

// TestComposedResidualNormMLPProjAttnInputFuseDeviceVsHost exercises the input-side mirror of the o_proj
// fuse: layer 0's proj-fused tail command buffer additionally folds in layer 1's (a full-attention mixer)
// input RMSNorm + q/k/v projections — the symmetric collapse to TestComposedResidualNormMLPProjFuseDeviceVsHost.
// A call-counter around the hook asserts the input-fuse actually fired; the full 2-layer forward's hiddens
// must still match a pure-host run within f32 tolerance. Both layers share D=512, FF=2048 (L·D·FF =
// 3,145,728 ≥ 1<<20 opens layer 0's proj-fused tail; the input-fuse rides free whatever layer 1's own q/k/v
// shape).
func TestComposedResidualNormMLPProjAttnInputFuseDeviceVsHost(t *testing.T) {
	if composed.ResidualNormMLPProjAttnInputDevice == nil {
		t.Fatal("native init did not wire composed.ResidualNormMLPProjAttnInputDevice")
	}
	const D, FF, vocab, heads, hd = 512, 2048, 4096, 4, 128
	newAttnLayer := func(seed int) composed.Layer {
		return composed.Layer{
			InputNorm:    cbSyn(D, seed),
			PostAttnNorm: cbSyn(D, seed+1),
			MLP:          &composed.MLP{Gate: cbSyn(FF*D, seed+2), Up: cbSyn(FF*D, seed+3), Down: cbSyn(D*FF, seed+4), FF: FF},
			Mixer: composed.NewAttnMixer(&composed.AttnWeights{
				QProj: cbSyn(heads*hd*D, seed+5), KProj: cbSyn(heads*hd*D, seed+6),
				VProj: cbSyn(heads*hd*D, seed+7), OProj: cbSyn(D*heads*hd, seed+8),
				QNorm: cbSyn(hd, seed+9), KNorm: cbSyn(hd, seed+10),
			}, composed.AttnConfig{Heads: heads, KVHeads: heads, HeadDim: hd, RotaryDim: hd / 2, RopeTheta: 1e6, NormEps: 1e-6}),
		}
	}
	m := &composed.ComposedModel{
		Embed: cbSyn(vocab*D, 1), NormF: cbSyn(D, 2), D: D, Vocab: vocab, Eps: 1e-6,
		Layers: []composed.Layer{newAttnLayer(20), newAttnLayer(40)},
	}
	tokens := []int32{5, 9, 21}

	calls := 0
	saved := composed.ResidualNormMLPProjAttnInputDevice
	composed.ResidualNormMLPProjAttnInputDevice = func(
		mixerHidden, projW, h, normW, gate, up, down []float32, L, D, mixCols, FF int, eps float32,
		nextNormW, nextQW, nextKW, nextVW []float32, nextQCols, nextKVCols int,
	) ([]float32, []float32, []float32, []float32, error) {
		calls++
		return ResidualNormMLPProjAttnInputDevice(mixerHidden, projW, h, normW, gate, up, down, L, D, mixCols, FF, eps, nextNormW, nextQW, nextKW, nextVW, nextQCols, nextKVCols)
	}
	dev, err := composed.NewSession(m).Forward(tokens)
	composed.ResidualNormMLPProjAttnInputDevice = saved
	if err != nil {
		t.Fatalf("device (input-fused) forward: %v", err)
	}
	if calls == 0 {
		t.Fatal("attn input-fuse hook never engaged — layer 1 not seen as a full-attention next mixer?")
	}

	savedInput, savedProjTail, savedTail, savedProj, savedMLP, savedFuse :=
		composed.ResidualNormMLPProjAttnInputDevice, composed.ResidualNormMLPProjDevice, composed.ResidualNormMLPDevice, composed.ProjMatMulInto, composed.MLPDevice, composed.AttnQKVDevice
	composed.ResidualNormMLPProjAttnInputDevice, composed.ResidualNormMLPProjDevice, composed.ResidualNormMLPDevice, composed.ProjMatMulInto, composed.MLPDevice, composed.AttnQKVDevice =
		nil, nil, nil, nil, nil, nil
	host, herr := composed.NewSession(m).Forward(tokens)
	composed.ResidualNormMLPProjAttnInputDevice, composed.ResidualNormMLPProjDevice, composed.ResidualNormMLPDevice, composed.ProjMatMulInto, composed.MLPDevice, composed.AttnQKVDevice =
		savedInput, savedProjTail, savedTail, savedProj, savedMLP, savedFuse
	if herr != nil {
		t.Fatalf("host forward: %v", herr)
	}
	if len(dev) != len(host) {
		t.Fatalf("length: device %d host %d", len(dev), len(host))
	}
	for i := range dev {
		if math.Abs(float64(dev[i]-host[i])) > 1e-2*(1+math.Abs(float64(host[i]))) {
			t.Fatalf("hidden[%d]: device %v host %v (attn input-fuse diverged)", i, dev[i], host[i])
		}
	}
	t.Logf("composed proj-tail+attn-input fuse: %d fused-CB call(s); device matches host within f32 tol over %d values", calls, len(dev))
}

// TestComposedResidualNormMLPProjGatedDeltaInputFuseDeviceVsHost exercises the input-side mirror of the
// o_proj fuse for a GATED-DELTA next layer: layer 0's (full-attention) proj-fused tail command buffer
// additionally folds in layer 1's (a gated-delta mixer) input RMSNorm + in_proj_qkv/z/a/b. A call-counter
// around the hook asserts the input-fuse actually fired; the full 2-layer forward's hiddens must still
// match a pure-host run within f32 tolerance.
func TestComposedResidualNormMLPProjGatedDeltaInputFuseDeviceVsHost(t *testing.T) {
	if composed.ResidualNormMLPProjGatedDeltaInputDevice == nil {
		t.Fatal("native init did not wire composed.ResidualNormMLPProjGatedDeltaInputDevice")
	}
	const D, FF, vocab, heads, hd = 512, 2048, 4096, 4, 128
	attnLayer := composed.Layer{
		InputNorm:    cbSyn(D, 20),
		PostAttnNorm: cbSyn(D, 21),
		MLP:          &composed.MLP{Gate: cbSyn(FF*D, 22), Up: cbSyn(FF*D, 23), Down: cbSyn(D*FF, 24), FF: FF},
		Mixer: composed.NewAttnMixer(&composed.AttnWeights{
			QProj: cbSyn(heads*hd*D, 25), KProj: cbSyn(heads*hd*D, 26),
			VProj: cbSyn(heads*hd*D, 27), OProj: cbSyn(D*heads*hd, 28),
			QNorm: cbSyn(hd, 29), KNorm: cbSyn(hd, 30),
		}, composed.AttnConfig{Heads: heads, KVHeads: heads, HeadDim: hd, RotaryDim: hd / 2, RopeTheta: 1e6, NormEps: 1e-6}),
	}
	gdCfg := qwen3.GatedDeltaConfig{KeyHeads: 8, ValueHeads: 8, HeadDim: 32, ConvKernel: 4, Eps: 1e-5}
	qDim, vDim := gdCfg.KeyHeads*gdCfg.HeadDim, gdCfg.ValueHeads*gdCfg.HeadDim
	convDim := 2*qDim + vDim
	gdw := &qwen3.GatedDeltaWeights{
		InProjQKV:  cbSyn(convDim*D, 40),
		ConvWeight: cbSyn(convDim*gdCfg.ConvKernel, 41),
		ConvBias:   cbSyn(convDim, 42),
		InProjA:    cbSyn(gdCfg.ValueHeads*D, 43),
		ALog:       cbSyn(gdCfg.ValueHeads, 44),
		DtBias:     cbSyn(gdCfg.ValueHeads, 45),
		InProjB:    cbSyn(gdCfg.ValueHeads*D, 46),
		InProjZ:    cbSyn(vDim*D, 47),
		Norm:       cbSyn(gdCfg.HeadDim, 48),
		OutProj:    cbSyn(D*vDim, 49),
	}
	gdLayer := composed.Layer{
		InputNorm:    cbSyn(D, 50),
		PostAttnNorm: cbSyn(D, 51),
		MLP:          &composed.MLP{Gate: cbSyn(FF*D, 52), Up: cbSyn(FF*D, 53), Down: cbSyn(D*FF, 54), FF: FF},
		Mixer:        composed.NewGatedDeltaMixer(gdw, gdCfg),
	}
	m := &composed.ComposedModel{
		Embed: cbSyn(vocab*D, 1), NormF: cbSyn(D, 2), D: D, Vocab: vocab, Eps: 1e-6,
		Layers: []composed.Layer{attnLayer, gdLayer},
	}
	tokens := []int32{5, 9, 21}

	calls := 0
	saved := composed.ResidualNormMLPProjGatedDeltaInputDevice
	composed.ResidualNormMLPProjGatedDeltaInputDevice = func(
		mixerHidden, projW, h, normW, gate, up, down []float32, L, D, mixCols, FF int, eps float32,
		nextNormW, nextQKVW, nextZW, nextAW, nextBW []float32, nextConvDim, nextVDim, nextVH int,
	) ([]float32, []float32, []float32, []float32, []float32, error) {
		calls++
		return ResidualNormMLPProjGatedDeltaInputDevice(mixerHidden, projW, h, normW, gate, up, down, L, D, mixCols, FF, eps, nextNormW, nextQKVW, nextZW, nextAW, nextBW, nextConvDim, nextVDim, nextVH)
	}
	dev, err := composed.NewSession(m).Forward(tokens)
	composed.ResidualNormMLPProjGatedDeltaInputDevice = saved
	if err != nil {
		t.Fatalf("device (input-fused) forward: %v", err)
	}
	if calls == 0 {
		t.Fatal("gated-delta input-fuse hook never engaged — layer 1 not seen as a gated-delta next mixer?")
	}

	savedInput, savedProjTail, savedTail, savedCProj, savedProj, savedInto, savedMLP, savedFuse, savedGDIn :=
		composed.ResidualNormMLPProjGatedDeltaInputDevice, composed.ResidualNormMLPProjDevice, composed.ResidualNormMLPDevice, composed.ProjMatMulInto, qwen3.ProjMatMul, qwen3.ProjMatMulInto, composed.MLPDevice, composed.AttnQKVDevice, qwen3.GatedDeltaInputDevice
	composed.ResidualNormMLPProjGatedDeltaInputDevice, composed.ResidualNormMLPProjDevice, composed.ResidualNormMLPDevice, composed.ProjMatMulInto, qwen3.ProjMatMul, qwen3.ProjMatMulInto, composed.MLPDevice, composed.AttnQKVDevice, qwen3.GatedDeltaInputDevice =
		nil, nil, nil, nil, nil, nil, nil, nil, nil
	host, herr := composed.NewSession(m).Forward(tokens)
	composed.ResidualNormMLPProjGatedDeltaInputDevice, composed.ResidualNormMLPProjDevice, composed.ResidualNormMLPDevice, composed.ProjMatMulInto, qwen3.ProjMatMul, qwen3.ProjMatMulInto, composed.MLPDevice, composed.AttnQKVDevice, qwen3.GatedDeltaInputDevice =
		savedInput, savedProjTail, savedTail, savedCProj, savedProj, savedInto, savedMLP, savedFuse, savedGDIn
	if herr != nil {
		t.Fatalf("host forward: %v", herr)
	}
	if len(dev) != len(host) {
		t.Fatalf("length: device %d host %d", len(dev), len(host))
	}
	for i := range dev {
		if math.Abs(float64(dev[i]-host[i])) > 1e-2*(1+math.Abs(float64(host[i]))) {
			t.Fatalf("hidden[%d]: device %v host %v (gated-delta input-fuse diverged)", i, dev[i], host[i])
		}
	}
	t.Logf("composed proj-tail+gated-delta-input fuse: %d fused-CB call(s); device matches host within f32 tol over %d values", calls, len(dev))
}
