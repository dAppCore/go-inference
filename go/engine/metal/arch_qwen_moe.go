// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// arch_qwen_moe.go decodes a qwen3_5_moe MoE FFN layer for the factory session — the SiLU switch-MLP routed
// experts + the always-on shared expert. The gemma DEVICE MoE (encMoEBlockQuantDevice) can't serve it:
// gemma's block assumes five sandwich norms, GELU experts and a router.scale, none of which qwen has.
// PreFFNorm(=post_attention_layernorm) → softmax top-k router (host-orchestrated, one device round trip via
// projQuantAttn) → ONE MoEExpertsQuantSiLU dispatch over the routed top-K → ONE degenerate-1-of-1
// MoEExpertsQuantSiLU dispatch for σ(x·SharedSigmoid)·SwiGLU_silu(shared) → residual. Was topK×3 + 3
// per-projection quant-seam round trips (host loop over sliceExpertRows + swigluSiLUHost); now 2 batched
// device dispatches — the batched ExpGate/ExpUp/ExpDown tensors are addressed by byte offset inside the
// kernel, so no host-side row-slicing is needed. A qwen MoE layer is marked by a bound shared expert, so
// gemma's device MoE is untouched.

// swigluSiLUHost computes down @ (silu(gate@x) · (up@x)) for one [d] row — a single expert's SiLU SwiGLU
// through the host quant matmul seam. gate/up are [ff,d], down is [d,ff]. No longer on the decode path
// (see encQwenMoEHalf) — kept as the correctness oracle for TestEncQwenMoESharedExpertMatchesDevice, which
// pins the fused device shared-expert call against this reference.
func swigluSiLUHost(x []float32, gate, up, down QuantWeight, d, ff int) ([]float32, error) {
	g, err := projQuantAttn(gate, x, d, ff, nil)
	if err != nil {
		return nil, err
	}
	u, err := projQuantAttn(up, x, d, ff, nil)
	if err != nil {
		return nil, err
	}
	h := make([]float32, ff)
	for f := 0; f < ff; f++ {
		s := float64(g[f])
		h[f] = float32((s / (1.0 + math.Exp(-s))) * float64(u[f]))
	}
	return projQuantAttn(down, h, ff, d, nil)
}

// sharedGateSigmoid resolves the shared expert's σ scalar gate: σ(x · SharedSigmoid). Handles the gate as
// quant (projQuantAttn) or bf16 (a host dot over the widened [d] weight); an unbound gate ⇒ σ ≡ 1.
func sharedGateSigmoid(w QuantWeight, x []float32, d int) (float64, error) {
	if len(w.Packed) == 0 {
		return 1, nil
	}
	if w.Bits > 0 {
		gl, err := projQuantAttn(w, x, d, 1, nil)
		if err != nil {
			return 0, err
		}
		return 1.0 / (1.0 + math.Exp(-float64(gl[0]))), nil
	}
	gw := bf16VecToF32(w.Packed)
	var acc float64
	for i := 0; i < d && i < len(gw); i++ {
		acc += float64(x[i]) * float64(gw[i])
	}
	return 1.0 / (1.0 + math.Exp(-acc)), nil
}

// encQwenMoEHalf computes one qwen3_5_moe MoE FFN for a single decode token: reads the post-mixer residual
// from hBuf, writes the layer output (h + routed + shared) to out. Router stays host-orchestrated (one
// device round trip); the routed top-K and the shared expert are each ONE MoEExpertsQuantSiLU dispatch.
func (s *archDecodeState) encQwenMoEHalf(li int, moe *MoEQuantLayerWeights, out metal.MTLBuffer) error {
	if moe == nil {
		return core.NewError("native.encQwenMoEHalf: nil MoE weights")
	}
	D, nE, topK, ff := s.dModel, moe.NumExperts, moe.TopK, moe.ExpertDFF
	if nE <= 0 || topK <= 0 || ff <= 0 {
		return core.NewError("native.encQwenMoEHalf: bad MoE geometry")
	}
	// sharedFF (#61): the shared expert's OWN width, from moe.SharedDFF (moeToQuant resolves it from
	// arch.SharedExpertFF, falling back to arch.ExpertFF when the arch declares no distinct width —
	// load_shared.go). A checkpoint like real llama4 Scout ships shared and routed genuinely distinct
	// (16384 vs 8192); using ff (the ROUTED width) for both — the pre-#61 bug — sized the shared
	// dispatch wrong, which MoEExpertsQuantSiLU's packed-length check turns into a hard error on every
	// decode step for such a checkpoint, not silently wrong output. Falls back to ff here too so a
	// MoEQuantLayerWeights built before this field existed (a hand-built test fixture, say) keeps the
	// historic ff-for-everything behaviour instead of a spurious zero-width dispatch.
	sharedFF := moe.SharedDFF
	if sharedFF == 0 {
		sharedFF = ff
	}
	h := bf16BufToF32(s.hBuf, 0, D)
	normed := rmsNormHostF32(h, bf16VecToF32(moe.PreFFNormW), s.eps)

	// Router: logits over every expert, softmax, top-k, renormalised over the selection (qwen default).
	logits, err := projQuantAttn(moe.Router, normed, D, nE, nil)
	if err != nil {
		return err
	}
	maxL := math.Inf(-1)
	for e := 0; e < nE; e++ {
		if float64(logits[e]) > maxL {
			maxL = float64(logits[e])
		}
	}
	probs := make([]float64, nE)
	for e := 0; e < nE; e++ {
		probs[e] = math.Exp(float64(logits[e]) - maxL)
	}
	sel := make([]int, 0, topK)
	used := make([]bool, nE)
	for k := 0; k < topK; k++ {
		best, bi := math.Inf(-1), -1
		for e := 0; e < nE; e++ {
			if !used[e] && probs[e] > best {
				best, bi = probs[e], e
			}
		}
		if bi < 0 {
			break
		}
		used[bi] = true
		sel = append(sel, bi)
	}
	denom := 0.0
	for _, e := range sel {
		denom += probs[e]
	}
	if denom == 0 {
		denom = 1
	}

	// Routed experts: ONE device dispatch computes Σ_k (probs[k]/denom) · SwiGLU_silu(expert_k) over the
	// BATCHED [nE·ff,D]/[nE·D,ff] switch_mlp tensors — MoEExpertsQuantSiLU addresses each selected expert's
	// rows by byte offset internally, so no host-side slicing (was topK×3 quant-seam round trips per layer).
	normedBF := f32ToBf16Slice(normed)
	selK := len(sel)
	idx := make([]int32, selK)
	wts := make([]byte, selK*bf16Size)
	for i, e := range sel {
		idx[i] = int32(e)
		r := f32ToBF16(float32(probs[e] / denom))
		wts[2*i], wts[2*i+1] = byte(r), byte(r>>8)
	}
	routedBytes, err := MoEExpertsQuantSiLU(normedBF, idx, wts, moe.ExpGate, moe.ExpUp, moe.ExpDown, nE, selK, D, ff, moe.ExpGate.GroupSize, moe.ExpGate.Bits)
	if err != nil {
		return err
	}
	acc := bf16ToF32Slice(routedBytes)

	// Shared expert: the SAME kernel as a degenerate 1-of-1 call — numExperts=topK=1, the combine weight IS
	// σ(normed·SharedSigmoid), so the gate scale fuses into the dispatch instead of a separate host multiply.
	// Sized off sharedFF (#61) — the shared expert's OWN width, NOT necessarily ff (the routed width);
	// see sharedFF's doc above for why the two genuinely differ on a real checkpoint.
	if len(moe.SharedGate.Packed) > 0 {
		g, err := sharedGateSigmoid(moe.SharedSigmoid, normed, D)
		if err != nil {
			return err
		}
		sr := f32ToBF16(float32(g))
		sharedBytes, err := MoEExpertsQuantSiLU(normedBF, []int32{0}, []byte{byte(sr), byte(sr >> 8)}, moe.SharedGate, moe.SharedUp, moe.SharedDown, 1, 1, D, sharedFF, moe.SharedGate.GroupSize, moe.SharedGate.Bits)
		if err != nil {
			return err
		}
		se := bf16ToF32Slice(sharedBytes)
		for d := 0; d < D; d++ {
			acc[d] += se[d]
		}
	}

	// Residual (h + routed + shared) → out, bf16.
	ob := unsafe.Slice((*byte)(out.Contents()), D*2)
	for d := 0; d < D; d++ {
		u := f32ToBF16(h[d] + acc[d])
		ob[2*d], ob[2*d+1] = byte(u), byte(u>>8)
	}
	return nil
}
