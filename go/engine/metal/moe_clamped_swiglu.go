// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// moe_clamped_swiglu.go implements GPT-OSS's clamped, sigmoid-gated SwiGLU expert activation — a THIRD
// MoE gate nonlinearity beside MoEExpertsQuant's gemma tanh-approx GELU and MoEExpertsQuantSiLU's plain
// llama/mistral/qwen silu(gate)·up. GPT-OSS's own GptOssExperts._apply_gate:
//
//	gate' = clip(gate, max=limit)               // UPPER-only clamp — no lower bound
//	up'   = clip(up, min=-limit, max=limit)      // SYMMETRIC clamp
//	glu   = gate' · sigmoid(alpha · gate')       // alpha = 1.702 ("quick-GELU" sigmoid constant) — NOT plain SiLU (no alpha term)
//	out   = glu · (up' + 1)                      // the "+1" up-shift is load-bearing; omitting it is silently wrong
//
// alpha is FIXED at 1.702 (not config-driven); limit is config.swiglu_limit (7.0 on every published
// gpt-oss checkpoint, InferenceIllusionist's MLX-4bit conversion included). Formula verified against TWO
// independent sources, fetched from source (not recalled from training data — see TestClampedSwiGLUBF16
// for the byte gate proving it against hand-computed values):
//
//	https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/gpt_oss/modeling_gpt_oss.py
//	  class GptOssExperts: "self.alpha = 1.702", "self.limit = 7.0"; _apply_gate:
//	    "gate = gate.clamp(min=None, max=self.limit); up = up.clamp(min=-self.limit, max=self.limit)"
//	    "glu = gate * torch.sigmoid(gate * self.alpha)"; forward: "gated_output = (up + 1) * glu"
//	https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/gpt_oss.py
//	  same clip/alpha/"+1" shape (mx.clip, alpha=1.702, "out_glu * (x_linear + 1)") — confirming the
//	  InferenceIllusionist MLX-4bit checkpoint's OWN conversion lineage applies the identical formula,
//	  not a simplified MLX approximation.
//
// Every op below dispatches an MLX-catalog kernel ALREADY compiled into mlx.metallib —
// vv_Maximumbfloat16/vv_Minimumbfloat16 sit in the SAME kernel family as vv_Addbfloat16/
// vv_Multiplybfloat16 (encAddBF16/encMulBF16's kernels), confirmed present by static inspection
// (`strings mlx.metallib | grep vv_`) of the compiled metallib both this repo and go-mlx ship. No NEW
// .metal kernel source or metallib rebuild is needed for this activation — see encMaxBF16/encMinBF16.

// encMaxBF16 / encMinBF16 are encAddBF16/encMulBF16's siblings over the same MLX vv_<Op>bfloat16 binary-
// kernel family (vv_Maximumbfloat16 / vv_Minimumbfloat16).
func encMaxBF16(enc metal.MTLComputeCommandEncoder, a, b, out metal.MTLBuffer, n int) error {
	return encBinaryLiteralTo(enc, "vv_Maximumbfloat16", a, b, out, 0, 0, 0, n)
}
func encMinBF16(enc metal.MTLComputeCommandEncoder, a, b, out metal.MTLBuffer, n int) error {
	return encBinaryLiteralTo(enc, "vv_Minimumbfloat16", a, b, out, 0, 0, 0, n)
}

// minBF16Const / maxBF16Const / addBF16Const are mulBF16Const's siblings: every element of a combined
// with the scalar v, driven host-side through one kernel dispatch. vv_Maximum/Minimum/Add have no
// true-scalar sibling the way Multiply does via lthn_bf16_mul_scalar (encMulScalarBF16), so materialising
// v as an n-length broadcast (matching bf16ConstBuffer / mlpScratch's c1/c044/c079/c05 constants) is the
// existing pattern, not a new one.
func minBF16Const(a []byte, n int, v float32) ([]byte, error) {
	out := make([]byte, len(a))
	if err := binaryBF16ConstIntoDirect("minBF16ConstInto", encMinBF16, a, n, v, out, false); err != nil {
		return nil, err
	}
	return out, nil
}

func maxBF16Const(a []byte, n int, v float32) ([]byte, error) {
	out := make([]byte, len(a))
	if err := binaryBF16ConstIntoDirect("maxBF16ConstInto", encMaxBF16, a, n, v, out, false); err != nil {
		return nil, err
	}
	return out, nil
}

func addBF16Const(a []byte, n int, v float32) ([]byte, error) {
	out := make([]byte, len(a))
	if err := binaryBF16ConstIntoDirect("addBF16ConstInto", encAddBF16, a, n, v, out, false); err != nil {
		return nil, err
	}
	return out, nil
}

// ClampedSwiGLUBF16 is the host-callable reference form of the formula above — gate/up are the ALREADY-
// PROJECTED expert gate_proj(x)/up_proj(x) outputs (n bf16 elements each — n is a layer's expert
// intermediate width, dFF), limit is config.swiglu_limit. Each step is its own kernel dispatch (a host
// round-trip between steps, unlike the fully-encoded encClampedSwiGLUGateMulBF16 the hot MoE dispatch
// loop uses) — this is the byte-gate test's oracle, and correct (if not the fastest) for any caller
// outside the per-token decode loop.
func ClampedSwiGLUBF16(gate, up []byte, n int, limit float32) ([]byte, error) {
	if len(gate) != n*bf16Size || len(up) != n*bf16Size {
		return nil, core.NewError("native.ClampedSwiGLUBF16: gate/up must be n bf16 values")
	}
	if limit <= 0 {
		return nil, core.NewError("native.ClampedSwiGLUBF16: limit must be > 0")
	}
	g, err := minBF16Const(gate, n, limit) // gate' = min(gate, limit) — upper-only clamp
	if err != nil {
		return nil, err
	}
	u, err := minBF16Const(up, n, limit)
	if err != nil {
		return nil, err
	}
	u, err = maxBF16Const(u, n, -limit) // up' = max(min(up,limit), -limit) — symmetric clamp
	if err != nil {
		return nil, err
	}
	scaled, err := mulBF16Const(g, n, 1.702) // alpha·gate'
	if err != nil {
		return nil, err
	}
	sig, err := SigmoidBF16(scaled) // sigmoid(alpha·gate')
	if err != nil {
		return nil, err
	}
	glu, err := MulBF16(g, sig) // gate'·sigmoid(alpha·gate')
	if err != nil {
		return nil, err
	}
	uPlus1, err := addBF16Const(u, n, 1.0) // up'+1
	if err != nil {
		return nil, err
	}
	return MulBF16(glu, uPlus1) // glu·(up'+1)
}

// clampedSwiGLUConstants are the four n-length constant broadcasts encClampedSwiGLUGateMulBF16 needs
// (limit, -limit, alpha, one) — built ONCE per MoEExpertsQuantClampedSiLU call (dFF is constant across
// the topK loop's selected experts) rather than once per expert, mirroring mlpScratch's c1/c044/c079/c05.
type clampedSwiGLUConstants struct {
	limit, negLimit, alpha, one metal.MTLBuffer
}

func newClampedSwiGLUConstants(n int, limit float32) clampedSwiGLUConstants {
	return clampedSwiGLUConstants{
		limit:    bf16ConstBuffer(n, limit),
		negLimit: bf16ConstBuffer(n, -limit),
		alpha:    bf16ConstBuffer(n, 1.702),
		one:      bf16ConstBuffer(n, 1.0),
	}
}

// encClampedSwiGLUGateMulBF16 encodes ClampedSwiGLUBF16's formula fully into an existing command encoder
// (no host round-trip) — the per-expert hot-path sibling encSiLUGateMulBF16/encGeluGateMul are.
// Deliberately clobbers gate/up in place as scratch (mirroring encSiLUGateMulBF16's 3-buffer economy: no
// caller-supplied scratch struct beyond the constants); writes the final result to out.
func encClampedSwiGLUGateMulBF16(enc metal.MTLComputeCommandEncoder, gate, up, out metal.MTLBuffer, cst clampedSwiGLUConstants, n int) error {
	if err := encMinBF16(enc, gate, cst.limit, gate, n); err != nil { // gate = min(gate, limit)
		return err
	}
	if err := encMinBF16(enc, up, cst.limit, up, n); err != nil { // up = min(up, limit)
		return err
	}
	if err := encMaxBF16(enc, up, cst.negLimit, up, n); err != nil { // up = max(up, -limit) => clip(up,-limit,limit)
		return err
	}
	if err := encMulBF16(enc, gate, cst.alpha, out, n); err != nil { // out = alpha·gate
		return err
	}
	if err := encSigmoidBF16(enc, out, out, n); err != nil { // out = sigmoid(alpha·gate)
		return err
	}
	if err := encMulBF16(enc, gate, out, out, n); err != nil { // out = gate·sigmoid(alpha·gate) = glu
		return err
	}
	if err := encAddBF16(enc, up, cst.one, up, n); err != nil { // up = up+1
		return err
	}
	return encMulBF16(enc, out, up, out, n) // out = glu·(up+1)
}

// MoEExpertsQuantClampedSiLU is MoEExpertsQuant for GPT-OSS's clamped, sigmoid-gated SwiGLU expert MLP —
// identical batched-expert dispatch (quantised gate_proj/up_proj/down_proj, router-weighted top-k
// combine) as MoEExpertsQuant/MoEExpertsQuantSiLU, but the gate nonlinearity is the clamped-sigmoid form
// documented at the top of this file, not gemma's GELU or plain SiLU. limit is config.swiglu_limit (7.0
// on every published gpt-oss checkpoint) — passed through, never hardcoded, so a future checkpoint that
// declares a different limit is served correctly rather than silently pinned to 7.0.
//
// gateBias/upBias (bf16 [numExperts×dFF]) and downBias (bf16 [numExperts×dModel]) are GPT-OSS's
// per-expert ADDITIVE projection biases (mlx-lm gpt_oss.py: SwitchGLU(..., bias=True); the real
// gpt-oss-20b MLX-4bit checkpoint ships mlp.experts.{gate,up,down}_proj.bias as BF16 [32, 2880] —
// read from the shard header). Each selected expert e adds its own [dFF]/[dModel] slice right
// after its matvec — gate'/up' BEFORE the clamp (the reference clamps the biased projection), down
// before the router-weighted combine. nil biases skip the adds — the encode stream is then
// dispatch-for-dispatch identical to the pre-bias form (the biasless-sibling regression contract;
// MoEExpertsQuant/MoEExpertsQuantSiLU are untouched entirely).
func MoEExpertsQuantClampedSiLU(x []byte, idx []int32, weights []byte, gate, up, down QuantWeight, gateBias, upBias, downBias []byte, numExperts, topK, dModel, dFF, groupSize, bits int, limit float32) ([]byte, error) {
	return moeExpertsQuantClampedInto(nil, x, idx, weights, gate, up, down, gateBias, upBias, downBias, numExperts, topK, dModel, dFF, groupSize, bits, limit, false)
}

// moeExpertsQuantClampedInto is moeExpertsQuantInto (moe.go) with the gate/up/down dispatch chain
// UNCHANGED and only the activation call swapped for encClampedSwiGLUGateMulBF16 — kept as a separate,
// self-contained function rather than a third boolean threaded through the shared moeExpertsQuantInto so
// this addition cannot perturb the three existing, already-serving callers (MoEExpertsQuant/
// MoEExpertsQuantInto/MoEExpertsQuantSiLU) even by signature change alone; gpt_oss has no live caller of
// its own yet (Config.Arch still refuses — see model/arch/openai/gptoss), so there is no serving path to
// regress by editing the shared helper, but also no need to accept that risk for a function nothing calls.
func moeExpertsQuantClampedInto(out []byte, x []byte, idx []int32, weights []byte, gate, up, down QuantWeight, gateBias, upBias, downBias []byte, numExperts, topK, dModel, dFF, groupSize, bits int, limit float32, useCallerOut bool) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != dModel*bf16Size {
		return nil, core.NewError("native.MoEExpertsQuantClampedSiLU: x must be dModel bf16 bytes")
	}
	if len(idx) != topK || len(weights) != topK*bf16Size {
		return nil, core.NewError("native.MoEExpertsQuantClampedSiLU: idx/weights length must equal topK")
	}
	if dModel%groupSize != 0 || dFF%groupSize != 0 {
		return nil, core.NewError("native.MoEExpertsQuantClampedSiLU: dModel and dFF must be multiples of groupSize")
	}
	if limit <= 0 {
		return nil, core.NewError("native.MoEExpertsQuantClampedSiLU: limit must be > 0")
	}
	// per-expert additive biases: each nil (skip — the pre-bias encode stream, dispatch for
	// dispatch) or exactly its batched [numExperts × outDim] bf16 shape.
	if len(gateBias) != 0 && len(gateBias) != numExperts*dFF*bf16Size {
		return nil, core.NewError("native.MoEExpertsQuantClampedSiLU: gateBias must be numExperts*dFF bf16 bytes or nil")
	}
	if len(upBias) != 0 && len(upBias) != numExperts*dFF*bf16Size {
		return nil, core.NewError("native.MoEExpertsQuantClampedSiLU: upBias must be numExperts*dFF bf16 bytes or nil")
	}
	if len(downBias) != 0 && len(downBias) != numExperts*dModel*bf16Size {
		return nil, core.NewError("native.MoEExpertsQuantClampedSiLU: downBias must be numExperts*dModel bf16 bytes or nil")
	}
	gatePacked, gateScale := dFF*dModel*bits/8, dFF*(dModel/groupSize)*bf16Size // per expert (gate, up)
	downPacked, downScale := dModel*dFF*bits/8, dModel*(dFF/groupSize)*bf16Size // per expert (down)
	if len(gate.Packed) != numExperts*gatePacked || len(up.Packed) != numExperts*gatePacked || len(down.Packed) != numExperts*downPacked ||
		len(gate.Scales) != numExperts*gateScale || len(up.Scales) != numExperts*gateScale || len(down.Scales) != numExperts*downScale ||
		len(gate.Biases) != numExperts*gateScale || len(up.Biases) != numExperts*gateScale || len(down.Biases) != numExperts*downScale {
		return nil, core.NewError("native.MoEExpertsQuantClampedSiLU: batched expert weight size mismatch")
	}
	for i := range idx {
		if idx[i] < 0 || int(idx[i]) >= numExperts {
			return nil, core.NewError("native.MoEExpertsQuantClampedSiLU: expert index out of range")
		}
	}
	outLen := dModel * bf16Size
	callerOut := useCallerOut && cap(out) >= outLen
	if callerOut {
		out = out[:outLen]
	} else {
		out = make([]byte, outLen)
	}
	if topK == 0 {
		clear(out)
		return out, nil
	}

	var encErr error
	withAutoreleasePool(func() {
		scratch, err := getMoEExpertsScratch(dModel, dFF, topK)
		if err != nil {
			encErr = err
			return
		}
		defer putMoEExpertsScratch(scratch)
		xBuf, ok := scratch.inputView(x)
		if !ok {
			xBuf, err = scratch.x.copyBuffer(x)
			if err != nil {
				encErr = err
				return
			}
		}
		weightsBuf, ok := scratch.weightsView(weights)
		if !ok {
			weightsBuf, err = scratch.weights.copyBuffer(weights)
			if err != nil {
				encErr = err
				return
			}
		}
		msc := scratch.mlp
		downE, scaled, acc := msc.down, scratch.scaled, scratch.acc
		cst := newClampedSwiGLUConstants(dFF, limit) // built once — dFF is constant across every selected expert
		directOut := false
		if callerOut {
			if tmp, ok := scratch.outputView(out); ok {
				acc = tmp
				directOut = true
			}
		}
		gatePackedBuf, gateScalesBuf, gateBiasesBuf := quantWeightViews(gate)
		upPackedBuf, upScalesBuf, upBiasesBuf := quantWeightViews(up)
		downPackedBuf, downScalesBuf, downBiasesBuf := quantWeightViews(down)
		// additive per-expert biases (gpt_oss): resident once (residentBytes memoises by base
		// pointer), each expert binds its slice by byte offset inside the encode loop below.
		var gateBiasBuf, upBiasBuf, downBiasBuf metal.MTLBuffer
		if len(gateBias) > 0 {
			gateBiasBuf = residentBytes(gateBias)
		}
		if len(upBias) > 0 {
			upBiasBuf = residentBytes(upBias)
		}
		if len(downBias) > 0 {
			downBiasBuf = residentBytes(downBias)
		}

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		for i := range topK {
			e := int(idx[i])
			gatePackedOff, gateScaleOff := uint(e*gatePacked), uint(e*gateScale)
			downPackedOff, downScaleOff := uint(e*downPacked), uint(e*downScale)
			if encErr = encQMVBF16(enc, gatePackedBuf.buf, gateScalesBuf.buf, gateBiasesBuf.buf, xBuf, msc.gate, gatePackedBuf.off+gatePackedOff, gateScalesBuf.off+gateScaleOff, gateBiasesBuf.off+gateScaleOff, 0, dFF, dModel, groupSize, bits); encErr != nil {
				endEncodingFast(enc)
				return
			}
			if gateBiasBuf != nil { // gate += gateBias[e] — BEFORE the clamp (the reference clamps the biased projection)
				if encErr = encAddBF16To(enc, msc.gate, gateBiasBuf, msc.gate, 0, uint(e*dFF*bf16Size), 0, dFF); encErr != nil {
					endEncodingFast(enc)
					return
				}
			}
			_ = encQMVBF16(enc, upPackedBuf.buf, upScalesBuf.buf, upBiasesBuf.buf, xBuf, msc.up, upPackedBuf.off+gatePackedOff, upScalesBuf.off+gateScaleOff, upBiasesBuf.off+gateScaleOff, 0, dFF, dModel, groupSize, bits)
			if upBiasBuf != nil { // up += upBias[e] — before the symmetric clamp
				if encErr = encAddBF16To(enc, msc.up, upBiasBuf, msc.up, 0, uint(e*dFF*bf16Size), 0, dFF); encErr != nil {
					endEncodingFast(enc)
					return
				}
			}
			if encErr = encClampedSwiGLUGateMulBF16(enc, msc.gate, msc.up, msc.gated, cst, dFF); encErr != nil {
				endEncodingFast(enc)
				return
			}
			_ = encQMVBF16(enc, downPackedBuf.buf, downScalesBuf.buf, downBiasesBuf.buf, msc.gated, downE, downPackedBuf.off+downPackedOff, downScalesBuf.off+downScaleOff, downBiasesBuf.off+downScaleOff, 0, dModel, dFF, groupSize, bits)
			if downBiasBuf != nil { // down += downBias[e] — before the router-weighted combine
				if encErr = encAddBF16To(enc, downE, downBiasBuf, downE, 0, uint(e*dModel*bf16Size), 0, dModel); encErr != nil {
					endEncodingFast(enc)
					return
				}
			}
			if i == 0 {
				if encErr = encScaleBF16(enc, downE, weightsBuf, acc, uint(i*bf16Size), weights[i*bf16Size:(i+1)*bf16Size], dModel); encErr != nil {
					endEncodingFast(enc)
					return
				}
			} else {
				if encErr = encScaleBF16(enc, downE, weightsBuf, scaled, uint(i*bf16Size), weights[i*bf16Size:(i+1)*bf16Size], dModel); encErr != nil {
					endEncodingFast(enc)
					return
				}
				_ = encAddBF16(enc, acc, scaled, acc, dModel)
			}
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if !directOut {
			copy(out, unsafe.Slice((*byte)(scratch.acc.Contents()), len(out)))
		}
	})
	return out, encErr
}
