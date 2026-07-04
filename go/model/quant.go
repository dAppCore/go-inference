// SPDX-Licence-Identifier: EUPL-1.2

// Package model is the backend-agnostic model contract — the unified home for the
// "later design choices" that pkg/metal/model accreted after the fact (its
// model↔engine seam was bolted on once the engine became reactive). A model
// declares its architecture and quantisation; a backend (pkg/metal via mlx-c,
// pkg/native no-cgo, future go-rocm) registers an implementation; the reactive
// engine talks to the contract, not the backend.
//
// It extends the registry/factory pattern already proven for quant schemes
// (pkg/scheme) and model loaders (the model registry) — one level up, for the
// backend itself. The existing quant registry keys by KIND only (a single loaded
// driver attaches its compute to the affine entry); the cross-section here keys by
// (BACKEND, kind) so two backends' compute coexist and the engine picks by the
// backend it loaded.
//
// First contract: the backend cross-section of the quant compute (below). The
// declarative architecture and the reactive-engine seam follow in later parts.
package model

import (
	core "dappco.re/go"
	"dappco.re/go/inference/scheme"
)

// QuantMatVec is the backend-agnostic quant decode projection: out = x @ Wᵀ for a
// quantised weight (transpose=true, the standard linear), single-query (decode).
// It speaks the lingua franca both backends already use — raw BYTES: bf16
// activations in/out, plus the backend's packed weight + scales + biases bytes —
// so the contract sidesteps a tensor abstraction for now (pkg/metal converts
// Array↔bytes internally; pkg/native is bytes-native). A model declaring
// quantization.kind gets its matvec from the LOADED backend's registered impl
// without knowing metal-vs-native — the backend cross-section, mirroring
// pkg/scheme's metadata registry one level down (the compute).
type QuantMatVec interface {
	scheme.QuantScheme // Kind() + Bits()
	MatVec(x, packed, scales, biases []byte, outDim, inDim, groupSize, bits int) ([]byte, error)
}

// backendQuants holds the per-(backend,kind) quant compute, keyed "backend/kind".
// A new backend or format is one RegisterBackendQuant — no central switch, no
// model edit (the pattern pkg/scheme and the model loader registry already use).
var backendQuants = core.NewRegistry[QuantMatVec]()

func bkKey(backend, kind string) string { return backend + "/" + kind }

// RegisterBackendQuant records a backend's quant-compute impl, keyed by the backend
// name and the scheme's Kind(). Called from the backend package's init() — the impl
// lives with the backend (e.g. pkg/native registers "native"/"affine"), exactly as
// pkg/metal's quant formats self-register against pkg/scheme.
func RegisterBackendQuant(backend string, q QuantMatVec) core.Result {
	return backendQuants.Set(bkKey(backend, q.Kind()), q)
}

// BackendQuant resolves the quant compute for a (backend, kind) — the loaded
// backend's implementation of the format a model declares. ok is false if no
// backend has registered that kind.
func BackendQuant(backend, kind string) (QuantMatVec, bool) {
	if r := backendQuants.Get(bkKey(backend, kind)); r.OK {
		return r.Value.(QuantMatVec), true
	}
	return nil, false
}
