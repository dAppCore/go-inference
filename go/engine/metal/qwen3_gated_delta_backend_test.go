// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"

	"dappco.re/go/inference/model/qwen3"
)

func gdbSyn(n, seed int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32((i*seed+7)%23-11) * 0.04
	}
	return v
}

// TestQwen3GatedDeltaDeviceVsHost runs a Qwen 3.6 gated-delta block with native's device-GEMM projections
// (init-wired) and confirms the output matches a host-matNT run (hook nil'd) within f32 tolerance — the
// device path is the projection swap only, the delta recurrence + conv unchanged.
func TestQwen3GatedDeltaDeviceVsHost(t *testing.T) {
	if qwen3.ProjMatMul == nil {
		t.Fatal("native init did not wire qwen3.ProjMatMul")
	}
	cfg := qwen3.GatedDeltaConfig{KeyHeads: 2, ValueHeads: 4, HeadDim: 8, ConvKernel: 4, Eps: 1e-5}
	const D, L = 8, 5
	qDim, vDim, convDim := cfg.KeyHeads*cfg.HeadDim, cfg.ValueHeads*cfg.HeadDim, 2*cfg.KeyHeads*cfg.HeadDim+cfg.ValueHeads*cfg.HeadDim
	_ = qDim
	w := &qwen3.GatedDeltaWeights{
		InProjQKV:  gdbSyn(convDim*D, 11),
		ConvWeight: gdbSyn(convDim*cfg.ConvKernel, 12),
		ConvBias:   gdbSyn(convDim, 13),
		InProjA:    gdbSyn(cfg.ValueHeads*D, 14),
		ALog:       gdbSyn(cfg.ValueHeads, 15),
		DtBias:     gdbSyn(cfg.ValueHeads, 16),
		InProjB:    gdbSyn(cfg.ValueHeads*D, 17),
		InProjZ:    gdbSyn(vDim*D, 18),
		Norm:       gdbSyn(cfg.HeadDim, 19),
		OutProj:    gdbSyn(D*vDim, 20),
	}
	x := gdbSyn(L*D, 1)

	dev, _, _, err := qwen3.GatedDeltaForwardF32(x, w, cfg, nil, nil, L, D)
	if err != nil {
		t.Fatalf("device block: %v", err)
	}
	saved := qwen3.ProjMatMul
	qwen3.ProjMatMul = nil
	host, _, _, herr := qwen3.GatedDeltaForwardF32(x, w, cfg, nil, nil, L, D)
	qwen3.ProjMatMul = saved
	if herr != nil {
		t.Fatalf("host block: %v", herr)
	}
	for i := range dev {
		if math.Abs(float64(dev[i]-host[i])) > 1e-2*(1+math.Abs(float64(host[i]))) {
			t.Fatalf("block out[%d]: device %v, host %v (device GEMM diverged)", i, dev[i], host[i])
		}
	}
	t.Logf("qwen3 gated-delta: device-GEMM projections match host matNT within f32 tol over %d×%d output", L, D)
}
