// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"

	"dappco.re/go/inference/model/arch/rwkv7"
)

// rwkv7_device_parity_test.go is #36's device-GEMM parity receipt: TestRWKV7BlockDeviceVsHost
// (rwkv7_backend_test.go) already proves the device hook is wired and matches the host reference on
// block.go's simplified stand-in block at small synthetic shapes; this file adds the SAME proof at every
// GEMM shape the REAL RWKV7-Goose-World2.8-0.1B-HF checkpoint's timeMixForward/channelMixForward/lora
// chain actually drives through rwkv7.ProjMatMul during a serve (backend.go) — read from the checkpoint's
// own config.json, not guessed: hidden_size=768, head_dim=64 (H=12 heads, K=V=64 so H*K=H*V=768),
// decay_low_rank_dim=64 (w_lora), a_low_rank_dim=64 (a_lora), gate_low_rank_dim=128 (g_lora),
// v_low_rank_dim=32 (v_lora), intermediate_size=3072 (channel-mix). block.go's BlockWeights (flat
// AProj/BProj, no LoRA) was a DIFFERENT, simplified architecture used only by the retired composed
// engine's mixer adapter (#50) — so this file targets the real
// timeMixWeights/lora/channelMixWeights shapes the standalone RWKV7Session (bin/lem generate's actual
// path for a pure RWKV-7 checkpoint) drives.

// rwkv7ProjShape is one (M,K,N) class the real 0.1B checkpoint's forward pass drives through
// rwkv7.ProjMatMul, named for the real call site so a failure identifies the projection, not just three
// numbers.
type rwkv7ProjShape struct {
	name    string
	m, k, n int
}

// rwkv7ProjShapes are every distinct GEMM shape in the real 0.1B geometry, at M=1 (the live decode step)
// and M=8 (a short prefill chunk) — spanning both routes matMulF32NTInto's shape-dependent dispatch picks
// between (split-K vs the fused nt kernel; see matmul_steel.go's matMulF32NTInto comment). r_proj/k_proj/
// v_proj/o_proj are each 768x768 for this checkpoint (Dv = H*V = 768 = D, since K=V=64) — one shape entry
// covers all four real call sites.
var rwkv7ProjShapes = []rwkv7ProjShape{
	{"dense r_k_v_o_proj decode(M=1)", 1, 768, 768},
	{"dense r_k_v_o_proj prefill(M=8)", 8, 768, 768},
	{"w_lora down decode(M=1)", 1, 768, 64},
	{"w_lora down prefill(M=8)", 8, 768, 64},
	{"w_lora up decode(M=1)", 1, 64, 768},
	{"w_lora up prefill(M=8)", 8, 64, 768},
	{"a_lora down decode(M=1)", 1, 768, 64},
	{"a_lora up decode(M=1)", 1, 64, 768},
	{"g_lora down decode(M=1)", 1, 768, 128},
	{"g_lora down prefill(M=8)", 8, 768, 128},
	{"g_lora up decode(M=1)", 1, 128, 768},
	{"g_lora up prefill(M=8)", 8, 128, 768},
	{"v_lora down decode(M=1)", 1, 768, 32},
	{"v_lora up decode(M=1)", 1, 32, 768},
	{"channelmix key decode(M=1)", 1, 768, 3072},
	{"channelmix key prefill(M=8)", 8, 768, 3072},
	{"channelmix value decode(M=1)", 1, 3072, 768},
	{"channelmix value prefill(M=8)", 8, 3072, 768},
}

func rwkv7ProjSyn(n, seed int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32((i*seed+7)%23-11) * 0.04
	}
	return v
}

// TestRWKV7DeviceProjMatchesReference confirms native's init wired rwkv7.ProjMatMul to the steel GEMM and
// it computes the projection y = x @ Wᵀ within f32 tolerance of an f64 host reference, at every distinct
// GEMM shape the REAL 0.1B checkpoint's forward pass drives (rwkv7ProjShapes) — the mamba2-pair template
// (TestMamba2DeviceProjMatchesReference) run at the real geometry instead of an arbitrary shape.
func TestRWKV7DeviceProjMatchesReference(t *testing.T) {
	if rwkv7.ProjMatMul == nil {
		t.Fatal("native init did not wire rwkv7.ProjMatMul")
	}
	for _, shape := range rwkv7ProjShapes {
		t.Run(shape.name, func(t *testing.T) {
			M, K, N := shape.m, shape.k, shape.n
			x, w := rwkv7ProjSyn(M*K, 7), rwkv7ProjSyn(N*K, 5)
			dev, err := rwkv7.ProjMatMul(x, w, M, K, N)
			if err != nil {
				t.Fatal(err)
			}
			if len(dev) != M*N {
				t.Fatalf("len %d, want %d", len(dev), M*N)
			}
			var worst float64
			for m := range M {
				for n := range N {
					var acc float64
					for k := range K {
						acc += float64(x[m*K+k]) * float64(w[n*K+k])
					}
					got := float64(dev[m*N+n])
					if d := math.Abs(got - acc); d > worst {
						worst = d
					}
					if math.Abs(got-acc) > 1e-3*(1+math.Abs(acc)) {
						t.Errorf("proj[%d,%d] device %v, host %v", m, n, got, acc)
					}
				}
			}
			t.Logf("shape %s: M=%d K=%d N=%d, worst abs diff %.3e", shape.name, M, K, N, worst)
		})
	}
	t.Log("native wired rwkv7.ProjMatMul = steel GEMM; matches host f64 reference at every real 0.1B-checkpoint GEMM shape")
}
