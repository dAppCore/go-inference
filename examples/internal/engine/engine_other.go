// SPDX-Licence-Identifier: EUPL-1.2

//go:build !(darwin && arm64)

// Package engine selects the GPU engine for the running platform. Off
// darwin/arm64 no engine is wired yet — engine/hip (AMD, linux/amd64) joins
// once it reaches feature parity — so inference.LoadModel reports
// "no backends available". Run the examples on Apple Silicon today.
package engine
