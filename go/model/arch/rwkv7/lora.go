// SPDX-Licence-Identifier: EUPL-1.2

package rwkv7

import core "dappco.re/go"

// lora.go is the down-project -> activation -> up-project(+bias) LoRA-MLP every RWKV-7 gate (decay w,
// in-context learning rate a, value-residual mix v, output gate g) is built from — fla.layers.rwkv6.LoRA,
// reused verbatim by fla.layers.rwkv7.RWKV7Attention (`from fla.layers.rwkv6 import LoRA`). Real weight
// shapes, real forward: hidden = activation(x @ A^T); out = hidden @ B^T (+ bias). The projections go
// through this package's projMatMul hook (AX-8: host matNT by default, native's steel GEMM when linked —
// see backend.go), so a LoRA gate gets the same device acceleration as the dense projections for free.

// lora is one LoRA-MLP's real parameters: down A [Low,In], up B [Out,Low], optional Bias [Out] (nil for
// g_lora — the checkpoint's only bias=False LoRA).
type lora struct {
	A, B, Bias   []float32
	In, Low, Out int
}

// forward runs x [L,In] through the LoRA-MLP, applying act to the low-rank hidden (matching
// fla.layers.rwkv6.LoRA.forward = lora[2](activation(lora[0](x)))). act may be nil (fla's
// activation=None ⇒ nn.Identity — a_lora and v_lora). Returns the RAW [L,Out] result BEFORE any outer
// transform a caller applies on top (RWKV7Attention.forward applies its own external .sigmoid() to
// w_lora's and a_lora's/v_lora's raw output; g_lora's raw output IS the final gate, no outer transform).
func (lo *lora) forward(x []float32, L int, act func(float32) float32) ([]float32, error) {
	if len(x) != L*lo.In || lo.In <= 0 || lo.Low <= 0 || lo.Out <= 0 {
		return nil, core.NewError("rwkv7.lora.forward: bad geometry or x size")
	}
	hidden, err := projMatMul(x, lo.A, L, lo.In, lo.Low)
	if err != nil {
		return nil, err
	}
	if act != nil {
		for i := range hidden {
			hidden[i] = act(hidden[i])
		}
	}
	out, err := projMatMul(hidden, lo.B, L, lo.Low, lo.Out)
	if err != nil {
		return nil, err
	}
	if lo.Bias != nil {
		for r := range L {
			rb := r * lo.Out
			for i := range lo.Out {
				out[rb+i] += lo.Bias[i]
			}
		}
	}
	return out, nil
}
