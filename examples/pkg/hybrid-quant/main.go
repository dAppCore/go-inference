// SPDX-Licence-Identifier: EUPL-1.2

// Serve a QUANTISED hybrid (Qwen 3.6 — model_type qwen3_5) with its weights kept PACKED on device. The
// composed hybrid stack alternates gated-delta linear-attention layers with full-attention layers; at 27B
// a dense f32 widening is ~110 GB (dead on arrival), so the loader carries every 2-D projection PACKED
// (MLX affine codes + scales/biases) straight to the engine's quant matvec — affine_qmv for decode,
// affine_qmm_t for prefill. This example is the lib-level acceptance for that path: LoadModel a packed
// snapshot and Generate, no HTTP serve in the way. Point it at either an MLX 4-bit pack or a 1-bit pack
// (Bonsai — its 1-bit codes are repacked to 2-bit at load, exact) — the same call serves both.
//
//	go run ./pkg/hybrid-quant -model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-27B-4bit/snapshots/<snap>
//	go run ./pkg/hybrid-quant -model ~/models/Bonsai-27B-mlx-1bit -prompt "The colour of a clear daytime sky is"
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"dappco.re/go/inference"
	_ "dappco.re/go/inference/examples/internal/engine" // registers the platform engine
)

func main() {
	model := flag.String("model", os.Getenv("LEM_MODEL"), "packed hybrid snapshot directory (config.json + *.safetensors)")
	prompt := flag.String("prompt", "The colour of a clear daytime sky is", "text to continue")
	maxTokens := flag.Int("max-tokens", 24, "tokens to generate")
	flag.Parse()
	if *model == "" {
		fmt.Fprintln(os.Stderr, "set -model (or LEM_MODEL) to a quantised hybrid snapshot directory")
		os.Exit(2)
	}

	// LoadModel routes a qwen3_5 checkpoint through the factory route (model.Load + the fused
	// arch session — the ONLY route since #50 retired the composed engine); a quantised pack keeps
	// its projections packed and is served through the quant matvec seam. The TEXT tower only —
	// image input refuses cleanly through the existing text-only machinery.
	r := inference.LoadModel(*model)
	if !r.OK {
		fmt.Fprintln(os.Stderr, "load:", r.Value)
		os.Exit(1)
	}
	m := r.Value.(inference.TextModel)
	defer m.Close()

	var reply strings.Builder
	for tok := range m.Generate(context.Background(), *prompt, inference.WithMaxTokens(*maxTokens), inference.WithTemperature(0)) {
		reply.WriteString(tok.Text)
	}
	if er := m.Err(); !er.OK {
		fmt.Fprintln(os.Stderr, "generate:", er.Value)
		os.Exit(1)
	}
	fmt.Println(*prompt + reply.String())
}
