// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "dappco.re/go/inference/model/composed"

// composed_backend.go wires native's device GEMM into the composed stack's own projections — the
// attention mixer's q/k/v/o, the MLP/MoE matmuls, and the LM head (the largest single matmul of every
// decode step) — the same AX-8 seam as gated_delta_backend.go: composed declares the hook and runs the
// sharded host matNT by default; importing native binds the steel GEMM. The gated-delta block inside a
// composed model already rides qwen3's hook, so this closes the remaining host-only matmuls of a served
// Qwen 3.6 hybrid. Sub-floor shapes (composed.deviceMinWork) stay host-side — a tiny GEMV's
// command-buffer round-trip outweighs its compute.
func init() {
	composed.ProjMatMulInto = MatMulF32NTInto
}
