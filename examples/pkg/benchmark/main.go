// SPDX-Licence-Identifier: EUPL-1.2

// Getting the numbers out of go-inference: every completed Generate/Chat call
// leaves a GenerateMetrics snapshot behind — token counts, prefill vs decode
// split, throughput, peak GPU memory. This is the llama-bench "tg" shape:
// one untimed warmup (load + JIT), then a measured N-token generation.
//
//	go run ./pkg/benchmark -model ~/models/gemma-4-e2b-it-4bit -tokens 256
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"dappco.re/go/inference"
	_ "dappco.re/go/inference/examples/internal/engine" // registers the platform engine
)

func main() {
	model := flag.String("model", os.Getenv("LEM_MODEL"), "model snapshot directory (config.json + *.safetensors)")
	tokens := flag.Int("tokens", 256, "tokens to generate in the measured run")
	flag.Parse()
	if *model == "" {
		fmt.Fprintln(os.Stderr, "set -model (or LEM_MODEL) to a model snapshot directory")
		os.Exit(2)
	}

	r := inference.LoadModel(*model)
	if !r.OK {
		fmt.Fprintln(os.Stderr, "load:", r.Value)
		os.Exit(1)
	}
	m := r.Value.(inference.TextModel)
	defer m.Close()

	// What loaded — architecture metadata from the checkpoint.
	info := m.Info()
	fmt.Printf("%s  %d layers  %d-bit  vocab %d\n", info.Architecture, info.NumLayers, info.QuantBits, info.VocabSize)

	ctx := context.Background()
	prompt := "Write a detailed essay about the history of navigation at sea."

	// Warmup: first generation pays one-off costs (pipeline builds, cache
	// allocation). Never time it.
	drain(m, ctx, "Warm up.", 16)

	// Measured run: temperature 0 keeps it deterministic so runs compare.
	drain(m, ctx, prompt, *tokens)

	// Metrics are valid once the iterator drains, until the next call starts.
	mt := m.Metrics()
	fmt.Printf("prompt   %4d toks   prefill %8.0f tok/s  (%s)\n", mt.PromptTokens, mt.PrefillTokensPerSec, mt.PrefillDuration.Round(1e6))
	fmt.Printf("decode   %4d toks   decode  %8.1f tok/s  (%s)\n", mt.GeneratedTokens, mt.DecodeTokensPerSec, mt.DecodeDuration.Round(1e6))
	fmt.Printf("total    %s   peak GPU %d MiB\n", mt.TotalDuration.Round(1e6), mt.PeakMemoryBytes>>20)
}

// drain runs one generation and discards the text — the metrics are the point.
func drain(m inference.TextModel, ctx context.Context, prompt string, maxTokens int) {
	off := false
	for range m.Chat(ctx,
		[]inference.Message{{Role: "user", Content: prompt}},
		inference.WithMaxTokens(maxTokens),
		inference.WithTemperature(0),
		inference.WithEnableThinking(&off),
	) {
	}
	if er := m.Err(); !er.OK {
		fmt.Fprintln(os.Stderr, "generate:", er.Value)
		os.Exit(1)
	}
}
