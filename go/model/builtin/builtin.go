// SPDX-Licence-Identifier: EUPL-1.2

// Package builtin registers the built-in model architectures with the reactive
// loader ([model.Load]) by importing each arch package for its init()-side
// [model.RegisterArch]. A serve composition blank-imports this package once and
// every built-in arch becomes resolvable by model_type — the engine stays
// arch-agnostic (it never imports an arch), and adding one is a config + that
// arch's own init().
//
// This is the go-inference home of the arch wiring that lived in go-mlx's
// register_native.go ("the serve layer now imports them explicitly") — the
// pkg/metal-typed composition root that was retired rather than ported, taking
// the wiring with it.
//
//	import _ "dappco.re/go/inference/model/builtin" // all arches resolvable
//
// The mixer/component packages (composed, deltanet, mamba2, rwkv7) are not
// listed: they carry no top-level model_type and are pulled in transitively by
// the arches that compose them.
package builtin

import (
	_ "dappco.re/go/inference/model/gemma3"  // gemma3
	_ "dappco.re/go/inference/model/gemma4"  // gemma4 / gemma4_text / gemma4_unified (+ assistant)
	_ "dappco.re/go/inference/model/mistral" // mistral
	_ "dappco.re/go/inference/model/qwen3"   // qwen3
)
