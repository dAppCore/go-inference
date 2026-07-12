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
	saved := composed.ProjMatMulInto
	composed.ProjMatMulInto = nil
	host, herr := composed.NewSession(m).Forward(tokens)
	composed.ProjMatMulInto = saved
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
