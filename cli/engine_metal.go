// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package main

import native "dappco.re/go/inference/engine/metal" // importing registers the no-cgo Apple "metal" backend via init()

// setPipelinedGPUDecode toggles the metal engine's one-ahead pipelined decode
// (-pipeline=false forces the chained serial loop, for A/B traces).
func setPipelinedGPUDecode(on bool) { native.SetPipelinedGPUDecode(on) }

// speculativeLoader loads a target+drafter pair as one speculative unit on the
// metal engine (MTP).
var speculativeLoader = native.LoadSpeculativePair
