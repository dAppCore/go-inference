// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"

	"dappco.re/go/inference/model/mamba2"
)

func mbSyn(n, seed int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32((i*seed+7)%23-11) * 0.04
	}
	return v
}

// TestMamba2DeviceProjMatchesReference confirms native's init wired mamba2.ProjMatMul to the steel GEMM
// and it computes the projection y = x @ Wᵀ (the [N,K]-weight convention) within f32 tolerance of a host
// f64 reference.
func TestMamba2DeviceProjMatchesReference(t *testing.T) {
	if mamba2.ProjMatMul == nil {
		t.Fatal("native init did not wire mamba2.ProjMatMul")
	}
	const M, K, N = 4, 16, 24
	x, w := mbSyn(M*K, 7), mbSyn(N*K, 5)
	dev, err := mamba2.ProjMatMul(x, w, M, K, N)
	if err != nil {
		t.Fatal(err)
	}
	if len(dev) != M*N {
		t.Fatalf("len %d, want %d", len(dev), M*N)
	}
	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			var acc float64
			for k := 0; k < K; k++ {
				acc += float64(x[m*K+k]) * float64(w[n*K+k])
			}
			if got := float64(dev[m*N+n]); math.Abs(got-acc) > 1e-3*(1+math.Abs(acc)) {
				t.Errorf("proj[%d,%d] device %v, host %v", m, n, got, acc)
			}
		}
	}
	t.Log("native wired mamba2.ProjMatMul = steel GEMM; computes x@Wᵀ within f32 tol of host reference")
}

// TestMamba2BlockDeviceVsHost runs a full Mamba-2 block with the device-GEMM projections and confirms the
// output matches a host-matNT run (hook nil'd) within f32 tolerance — the device path is the projection
// swap only, structure unchanged.
func TestMamba2BlockDeviceVsHost(t *testing.T) {
	cfg := mamba2.BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	const D, L = 8, 5
	const dInner, convDim, projDim = 16, 32, 50 // H·P, dInner+2·N, 2·dInner+2·N+H
	w := &mamba2.BlockWeights{
		InProj: mbSyn(projDim*D, 11), ConvWeight: mbSyn(convDim*4, 12), ConvBias: mbSyn(convDim, 13),
		ALog: mbSyn(2, 14), D: mbSyn(2, 15), DtBias: mbSyn(2, 16), Norm: mbSyn(dInner, 17), OutProj: mbSyn(D*dInner, 18),
	}
	x := mbSyn(L*D, 1)

	dev, _, _, err := mamba2.BlockForwardF32(x, w, cfg, nil, nil, L, D)
	if err != nil {
		t.Fatalf("device block: %v", err)
	}
	saved := mamba2.ProjMatMul
	mamba2.ProjMatMul = nil
	host, _, _, herr := mamba2.BlockForwardF32(x, w, cfg, nil, nil, L, D)
	mamba2.ProjMatMul = saved
	if herr != nil {
		t.Fatalf("host block: %v", herr)
	}
	for i := range dev {
		if math.Abs(float64(dev[i]-host[i])) > 1e-2*(1+math.Abs(float64(host[i]))) {
			t.Fatalf("block out[%d]: device %v, host %v (device GEMM diverged)", i, dev[i], host[i])
		}
	}
	t.Logf("mamba2 block: device-GEMM projections match host matNT within f32 tol over %d×%d output", L, D)
}
