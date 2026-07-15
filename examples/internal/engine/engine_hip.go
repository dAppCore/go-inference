// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64

// Package engine selects the GPU engine for the running platform. On
// linux/amd64 the hip engine registers the "rocm" backend — one engine, three
// build lanes: AMD (ROCm, cgo + static HIP archives), CUDA (hip-nvidia), and
// a fully static CPU lane. The repo-root Makefile owns those builds — see
// examples/pkg/README.md for the per-backend recipes.
package engine

import (
	_ "dappco.re/go/inference/engine/hip"    // registers the "rocm" backend via init()
	_ "dappco.re/go/inference/model/builtin" // registers every built-in model architecture
)
