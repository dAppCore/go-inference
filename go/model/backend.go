// SPDX-Licence-Identifier: EUPL-1.2

package model

// Backend is the backend-agnostic decode contract: a loaded model (its weights + arch
// bound at construction) that runs the arch-driven decode forward. It is the seam the
// reactive engine drives without knowing whether the compute is the no-cgo native Metal
// backend (pkg/native) or the cgo mlx-c backend (pkg/metal) — both implement it.
//
// THE TENSOR-HANDLE DECISION: activations cross the seam as bf16 []byte — the same
// lingua franca QuantMatVec uses, deliberately chosen over an abstract tensor handle.
// Native is bytes-native (zero conversion); metal converts at the boundary
// (FromRawBytes / RawBytes). Bytes keep pkg/model pure-Go and all-platforms, and they
// sidestep committing the contract to either backend's tensor type.
type Backend interface {
	// DecodeForward runs the arch decode over T input token embeddings (each the hidden
	// size in bf16 bytes) and returns T output hidden states (same shape). It is
	// whole-sequence today (the KV cache is built per call); incremental single-token
	// decode with a persistent cache is a later refinement. The output is hidden states
	// — the LM head (hidden → logits) and sampling layer on top of this seam.
	DecodeForward(inputs [][]byte) ([][]byte, error)
}
