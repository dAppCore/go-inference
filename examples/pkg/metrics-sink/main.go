// SPDX-Licence-Identifier: EUPL-1.2

// WithMetricsSink delivers ONE generation's final GenerateMetrics as its
// stream completes — a request-scoped alternative to the global m.Metrics()
// read, which is last-writer-wins under concurrent generations against the
// same model (see the doc comment on GenerateMetrics.MetricsSink).
//
// Running two Generate/Chat calls concurrently against one loaded model is a
// real, supported shape here: LoadConfig.ParallelSlots exists precisely for
// it, and the engine's session layer (go/engine/prompt_reuse.go) opens an
// independent session per call rather than serialising through one. What is
// NOT safe to read concurrently is the model's shared last-error/last-metrics
// state — hence the sink. This example demonstrates two goroutines generating
// at once, each capturing its OWN usage through its own sink; it checks
// m.Err() only once, after both have joined, and notes below why that single
// check is a best-effort wrap-up rather than a per-request guarantee.
//
//	go run ./pkg/metrics-sink -model ~/models/gemma-4-e2b-it-4bit
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"sync"

	"dappco.re/go/inference"
	_ "dappco.re/go/inference/examples/internal/engine" // registers the platform engine
)

func main() {
	model := flag.String("model", os.Getenv("LEM_MODEL"), "model snapshot directory (config.json + *.safetensors)")
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

	prompts := []string{
		"In five words, what does a lighthouse do?",
		"In five words, what does a compass do?",
	}

	var wg sync.WaitGroup
	usage := make([]inference.GenerateMetrics, len(prompts))
	for i, prompt := range prompts {
		wg.Add(1)
		go func(i int, prompt string) {
			defer wg.Done()
			sink := inference.WithMetricsSink(func(gm inference.GenerateMetrics) { usage[i] = gm })
			for range m.Chat(context.Background(),
				[]inference.Message{{Role: "user", Content: prompt}},
				inference.WithMaxTokens(16), sink) {
			}
		}(i, prompt)
	}
	wg.Wait()

	// Best-effort wrap-up: Err() reads the model's shared last-error state, so
	// after two concurrent calls it reflects whichever finished last — good
	// enough to catch "something failed", not which request failed. A caller
	// needing per-request error detail would thread its own error channel
	// alongside the sink; this surface only makes metrics request-scoped.
	if er := m.Err(); !er.OK {
		fmt.Fprintln(os.Stderr, "generate:", er.Value)
		os.Exit(1)
	}

	for i, u := range usage {
		fmt.Printf("%q -> %d tokens, %.0f tok/s decode\n", prompts[i], u.GeneratedTokens, u.DecodeTokensPerSec)
	}
}
