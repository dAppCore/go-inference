// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"

	"dappco.re/go/inference/model/composed"
)

// oprojFuseEnabled gates the projection-fused FFN tail (o_proj/out_proj folded into the tail command
// buffer). Default on; LTHN_OPROJ_FUSE=0 leaves the hook unbound so forwardEmb runs the mixer's own
// projection CB + the plain tail — the "before" arm of a same-binary interleaved A/B. Same LTHN_* bench
// affordance shape as LTHN_KV_Q8_ICB / LTHN_FLASH_PROMPT.
var oprojFuseEnabled = os.Getenv("LTHN_OPROJ_FUSE") != "0"

// inputFuseEnabled gates the input-side mirror of the o_proj fuse: the NEXT layer's input RMSNorm + input
// projections folded onto the BACK of the current layer's proj-fused tail command buffer. It EXTENDS that
// tail (reimplements it inline, so it never calls ResidualNormMLPProjDevice itself) but still requires
// oprojFuseEnabled — disabling the base proj-fuse is meant to disable the whole family, not leave an
// equivalent fused path active under a different switch. Default on; LTHN_INPUT_FUSE=0 leaves these hooks
// unbound so forwardEmb falls back to the proj-fused-tail-only path — the "before" arm of a same-binary
// interleaved A/B.
var inputFuseEnabled = oprojFuseEnabled && os.Getenv("LTHN_INPUT_FUSE") != "0"

// headFuseEnabled gates the OUTPUT-side mirror of the input fuse: the model's own final RMSNorm + LM head
// GEMM folded onto the back of the LAST layer's proj-fused tail (the head's own separate command buffer
// disappears — the terminal N+1 → N collapse). Requires oprojFuseEnabled — same "disabling the base
// proj-fuse disables the whole family" rule the input fuses follow. Default on; LTHN_HEAD_FUSE=0 leaves
// the hook unbound so the last layer falls back to the plain proj-fused tail + the separate host-RMSNorm/
// device-GEMM head path (ComposedSession.headLogits) — the "before" arm of a same-binary interleaved A/B.
var headFuseEnabled = oprojFuseEnabled && os.Getenv("LTHN_HEAD_FUSE") != "0"

// composed_backend.go wires native's device GEMM into the composed stack's own projections — the
// attention mixer's q/k/v/o, the MLP/MoE matmuls, and the LM head (the largest single matmul of every
// decode step) — the same AX-8 seam as gated_delta_backend.go: composed declares the hook and runs the
// sharded host matNT by default; importing native binds the steel GEMM. The gated-delta block inside a
// composed model already rides qwen3's hook, so this closes the remaining host-only matmuls of a served
// Qwen 3.6 hybrid. Sub-floor shapes (composed.deviceMinWork) stay host-side — a tiny GEMV's
// command-buffer round-trip outweighs its compute.
func init() {
	composed.ProjMatMulInto = MatMulF32NTInto
	composed.MLPDevice = ComposedMLPDevice
	composed.AttnQKVDevice = ComposedAttnQKVDevice         // fuses the attention mixer's q/k/v (all read h) into one command buffer
	composed.ResidualNormMLPDevice = ResidualNormMLPDevice // fuses the FFN tail: mixer residual + post-attn RMSNorm + SwiGLU MLP + MLP residual into one command buffer
	if oprojFuseEnabled {
		composed.ResidualNormMLPProjDevice = ResidualNormMLPProjDevice // folds the mixer's o_proj/out_proj onto the front of that tail — one CB where the unfused path pays a standalone projection CB per layer
	}
	if inputFuseEnabled {
		composed.ResidualNormMLPProjAttnInputDevice = ResidualNormMLPProjAttnInputDevice             // folds the NEXT full-attention layer's input RMSNorm + q/k/v onto the back of that tail
		composed.ResidualNormMLPProjGatedDeltaInputDevice = ResidualNormMLPProjGatedDeltaInputDevice // folds the NEXT gated-delta layer's input RMSNorm + in_proj_qkv/z/a/b onto the back of that tail
	}
	if headFuseEnabled {
		composed.ResidualNormMLPProjHeadDevice = ResidualNormMLPProjHeadDevice // folds the model's own final RMSNorm + LM head GEMM onto the back of the LAST layer's tail
	}
}
