// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "dappco.re/go/inference/model/qwen3"

// gated_delta_backend.go wires native's device GEMM into the engine-neutral Qwen 3.6 gated-delta
// block's projections (in_proj_qkv/a/b/z + out_proj — its compute hot spot; the delta recurrence + conv
// are cheap), the same seam as mamba2/rwkv7. qwen3 declares the ProjMatMul hook and runs the pure-Go host
// matNT by default (AX-8); importing native binds it to the steel GEMM (x[M,K]@w[N,K]ᵀ, byte-identical to
// metal's projection matmul). Qwen 3.6 is a real fleet target (gemma4's peer for local inference) gated on
// native hybrid linear-attention — this readies the gated-delta block's projections for the mixer-decode
// orchestration (the composed.ComposedModel port) that will serve it.
func init() {
	qwen3.ProjMatMul = MatMulF32NT
	qwen3.ProjMatMulInto = MatMulF32NTInto              // write-into sibling: skips the projection-output alloc, byte-identical
	qwen3.GatedDeltaInputDevice = GatedDeltaInputDevice // fuses in_proj_qkv/z/a/b (all read x) into one command buffer
}
