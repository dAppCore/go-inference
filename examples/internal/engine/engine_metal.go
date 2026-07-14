// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// Package engine selects the GPU engine for the running platform. Blank-import
// it once and the engine registers itself with the inference registry — the
// only wiring an application needs before inference.LoadModel:
//
//	import _ "dappco.re/go/inference/examples/internal/engine"
//
// The metal engine resolves its shader library from MLX_METALLIB_PATH (see
// examples/README.md); without it LoadModel fails with "no backends available".
package engine

import (
	_ "dappco.re/go/inference/engine/metal"  // registers the "metal" backend via init()
	_ "dappco.re/go/inference/model/builtin" // registers every built-in model architecture
)
