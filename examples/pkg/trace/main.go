// SPDX-Licence-Identifier: EUPL-1.2

// Per-token phase timing: GenerateConfig.TraceTokenPhases asks the engine to
// report where each generated token's wall time went — GPU-busy versus
// host-serial (encode/sample/detokenise while the GPU sits idle). There is no
// WithTraceTokenPhases option constructor yet; cmd/lem's own -trace flag sets
// the field via an inline GenerateOption closure (go/decode/generate/generate.go),
// so this example mirrors that exact wiring rather than inventing one.
//
// The budget lands on m.Metrics().DecodePhases, and is nil whenever the active
// engine/decode path did not report one — an untraced path, not an error.
//
//	go run ./pkg/trace -model ~/models/gemma-4-e2b-it-4bit
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"dappco.re/go/inference"
	_ "dappco.re/go/inference/examples/internal/engine" // registers the platform engine
)

// withTraceTokenPhases mirrors cmd/lem generate.go's -trace wiring: there is
// no exported option constructor for this field, so the CLI itself builds the
// GenerateOption closure inline. Copied verbatim rather than invented.
func withTraceTokenPhases() inference.GenerateOption {
	return func(c *inference.GenerateConfig) { c.TraceTokenPhases = true }
}

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

	for range m.Generate(context.Background(), "Explain tide tables in two sentences.",
		inference.WithMaxTokens(64), inference.WithTemperature(0), withTraceTokenPhases()) {
	}
	if er := m.Err(); !er.OK {
		fmt.Fprintln(os.Stderr, "generate:", er.Value)
		os.Exit(1)
	}

	budget := m.Metrics().DecodePhases
	if budget == nil || budget.Tokens == 0 {
		fmt.Println("no phase budget reported — this engine/path is not instrumented for tracing")
		return
	}
	fmt.Printf("%d tokens · %s/token total · %s/token GPU-busy · %.0f%% GPU\n",
		budget.Tokens, budget.TotalPerToken, budget.GPUPerToken, 100*budget.GPUFraction())
	for _, phase := range budget.Phases {
		lane := "host"
		if phase.GPU {
			lane = "GPU"
		}
		fmt.Printf("  %-24s %s/token  (%s)\n", phase.Name, phase.PerToken, lane)
	}
}
