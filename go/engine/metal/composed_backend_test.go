// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"

	"dappco.re/go/inference/model/composed"
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
