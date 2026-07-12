// SPDX-Licence-Identifier: EUPL-1.2

//go:build !(darwin && arm64)

package main

import "dappco.re/go/inference/serving"

// setPipelinedGPUDecode is a no-op off darwin/arm64 — pipelined decode is a
// metal-engine knob; other backends (hip, llama.cpp) own their own loops.
func setPipelinedGPUDecode(bool) {}

// speculativeLoader is nil off darwin/arm64 — MTP pair-loading is the metal
// engine's; the hip lane wires its own drafter path.
var speculativeLoader serving.SpeculativeLoader
