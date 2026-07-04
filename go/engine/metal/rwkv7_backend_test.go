// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"

	"dappco.re/go/inference/model/rwkv7"
)

func rwSyn(n, seed int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32((i*seed+7)%23-11) * 0.04
	}
	return v
}

// TestRWKV7BlockDeviceVsHost runs an RWKV-7 time-mix block with native's device-GEMM projections (init-
// wired) and confirms the output matches a host-matNT run (hook nil'd) within f32 tolerance — the device
// path is the projection swap only, the WKV7 recurrence unchanged.
func TestRWKV7BlockDeviceVsHost(t *testing.T) {
	if rwkv7.ProjMatMul == nil {
		t.Fatal("native init did not wire rwkv7.ProjMatMul")
	}
	cfg := rwkv7.BlockConfig{NumHeads: 2, KeyDim: 4, ValueDim: 6}
	const D, L = 8, 5
	hk, hv := cfg.NumHeads*cfg.KeyDim, cfg.NumHeads*cfg.ValueDim
	w := &rwkv7.BlockWeights{
		RProj: rwSyn(hk*D, 11), WProj: rwSyn(hk*D, 12), KProj: rwSyn(hk*D, 13),
		VProj: rwSyn(hv*D, 14), AProj: rwSyn(hk*D, 15), BProj: rwSyn(hk*D, 16),
		OutProj: rwSyn(D*hv, 17),
	}
	x := rwSyn(L*D, 1)

	dev, _, err := rwkv7.BlockForwardF32(x, w, cfg, nil, L, D)
	if err != nil {
		t.Fatalf("device block: %v", err)
	}
	saved := rwkv7.ProjMatMul
	rwkv7.ProjMatMul = nil
	host, _, herr := rwkv7.BlockForwardF32(x, w, cfg, nil, L, D)
	rwkv7.ProjMatMul = saved
	if herr != nil {
		t.Fatalf("host block: %v", herr)
	}
	for i := range dev {
		if math.Abs(float64(dev[i]-host[i])) > 1e-2*(1+math.Abs(float64(host[i]))) {
			t.Fatalf("block out[%d]: device %v, host %v (device GEMM diverged)", i, dev[i], host[i])
		}
	}
	t.Logf("rwkv7 block: device-GEMM projections match host matNT within f32 tol over %d×%d output", L, D)
}
