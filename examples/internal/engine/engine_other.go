// SPDX-Licence-Identifier: EUPL-1.2

//go:build !(darwin && arm64) && !(linux && amd64)

// Package engine selects the GPU engine for the running platform. On this
// platform no engine is wired (darwin/arm64 gets metal, linux/amd64 gets hip),
// so inference.LoadModel reports "no backends available".
package engine
