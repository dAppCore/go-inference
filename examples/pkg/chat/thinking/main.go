// SPDX-Licence-Identifier: EUPL-1.2

// The Gemma 4 thought channel: with thinking ON (the family default — an
// unset flag resolves to the vendor's thinking-on, #1847) the raw
// token stream carries the model's reasoning between channel markers, then
// the visible answer. The serve layer splits this automatically (the reply's
// `thought` field); a library caller does the same with decode/parser —
// parser.Filter for collected text (below), parser.NewProcessor for streams —
// which knows every family's markers via the model-info hint. Compare
// pkg/chat/basic, which disables the channel entirely.
//
//	go run ./pkg/chat/thinking -model ~/models/gemma-4-e2b-it-4bit
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"dappco.re/go/inference"
	"dappco.re/go/inference/decode/parser"
	_ "dappco.re/go/inference/examples/internal/engine" // registers the platform engine
)

func main() {
	model := flag.String("model", os.Getenv("LEM_MODEL"), "model snapshot directory (config.json + *.safetensors)")
	prompt := flag.String("prompt", "A farmer has 17 sheep. All but 9 run away. How many are left?", "user message")
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

	// Thinking defaults ON for Gemma 4 — no option needed. Budget generously:
	// the reasoning spends tokens before the visible answer starts.
	var raw strings.Builder
	for tok := range m.Chat(context.Background(),
		[]inference.Message{{Role: "user", Content: *prompt}},
		inference.WithMaxTokens(2048),
	) {
		raw.WriteString(tok.Text)
	}
	if er := m.Err(); !er.OK {
		fmt.Fprintln(os.Stderr, "generate:", er.Value)
		os.Exit(1)
	}

	// Split reasoning from answer with the family-aware parser: the hint from
	// ModelInfo selects the right marker set (gemma channel tokens here), and
	// ThinkingCapture strips reasoning from the visible text while handing it
	// back on the Result. parser.NewProcessor does the same incrementally for
	// per-token streams.
	hint := parser.HintFromInference(m.Info())
	res := parser.Filter(raw.String(), parser.Config{Mode: inference.ThinkingCapture}, hint)
	// Whether a reply carries a thought channel is the MODEL's choice per
	// prompt — small models often answer easy questions directly. The parser
	// is a no-op on channel-free text, so this handles both outcomes.
	if thought := strings.TrimSpace(res.Reasoning); thought != "" {
		fmt.Println("— thought —")
		fmt.Println(thought)
	} else {
		fmt.Println("— thought — (none: the model answered directly this run)")
	}
	fmt.Println("— answer —")
	fmt.Println(strings.TrimSpace(res.Text))
}
