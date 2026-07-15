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

// TestQwen3GatedDeltaInputFuseDeviceVsHost exercises the fused input projection (in_proj_qkv/z/a/b in
// ONE command buffer) at a decode shape (L=1) whose qkv projection crosses composed.deviceMinWork, so
// GatedDeltaInputDevice genuinely engages, and confirms the block output matches a pure-host run within
// f32 tolerance. A call-counter around the hook asserts the fuse actually fired (a below-floor shape
// would silently fall through to the per-projection path and make the test vacuous).
func TestQwen3GatedDeltaInputFuseDeviceVsHost(t *testing.T) {
	if qwen3.GatedDeltaInputDevice == nil {
		t.Fatal("native init did not wire qwen3.GatedDeltaInputDevice")
	}
	// D·convDim = 768·1536 = 1,179,648 ≥ deviceMinWork (1<<20) at L=1 ⇒ the fuse gate opens.
	cfg := qwen3.GatedDeltaConfig{KeyHeads: 8, ValueHeads: 8, HeadDim: 64, ConvKernel: 4, Eps: 1e-5}
	const D, L = 768, 1
	vDim, convDim := cfg.ValueHeads*cfg.HeadDim, 2*cfg.KeyHeads*cfg.HeadDim+cfg.ValueHeads*cfg.HeadDim
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

	calls := 0
	savedFuse := qwen3.GatedDeltaInputDevice
	qwen3.GatedDeltaInputDevice = func(x, qkvW, zW, aW, bW []float32, L, D, convDim, vDim, VH int) ([]float32, []float32, []float32, []float32, error) {
		calls++
		return GatedDeltaInputDevice(x, qkvW, zW, aW, bW, L, D, convDim, vDim, VH)
	}
	dev, _, _, err := qwen3.GatedDeltaForwardF32(x, w, cfg, nil, nil, L, D)
	qwen3.GatedDeltaInputDevice = savedFuse
	if err != nil {
		t.Fatalf("device (fused) block: %v", err)
	}
	if calls == 0 {
		t.Fatal("fused input hook never engaged — shape below device floor?")
	}

	savedFuse2, savedProj, savedInto := qwen3.GatedDeltaInputDevice, qwen3.ProjMatMul, qwen3.ProjMatMulInto
	qwen3.GatedDeltaInputDevice, qwen3.ProjMatMul, qwen3.ProjMatMulInto = nil, nil, nil
	host, _, _, herr := qwen3.GatedDeltaForwardF32(x, w, cfg, nil, nil, L, D)
	qwen3.GatedDeltaInputDevice, qwen3.ProjMatMul, qwen3.ProjMatMulInto = savedFuse2, savedProj, savedInto
	if herr != nil {
		t.Fatalf("host block: %v", herr)
	}
	if len(dev) != len(host) {
		t.Fatalf("length: device %d host %d", len(dev), len(host))
	}
	for i := range dev {
		if math.Abs(float64(dev[i]-host[i])) > 1e-2*(1+math.Abs(float64(host[i]))) {
			t.Fatalf("fused block out[%d]: device %v, host %v (fused input GEMM diverged)", i, dev[i], host[i])
		}
	}
	t.Logf("qwen3 gated-delta input fuse: %d fused-CB call(s); device matches host within f32 tol over %d×%d output", calls, L, D)
}
