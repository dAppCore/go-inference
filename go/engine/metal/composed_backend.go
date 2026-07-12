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
}
