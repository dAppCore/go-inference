// SPDX-Licence-Identifier: EUPL-1.2

// The Gemma 4 thought channel: with thinking ON (the model default) the raw
// token stream carries the model's reasoning between channel markers, then
// the visible answer:
//
//	<|channel>thought ...reasoning... <channel|> ...answer...
//
// The serve layer splits this automatically (the reply's `thought` field);
// a library caller sees the raw stream and splits it with the markers, as
// below. Compare pkg/chat/basic, which disables the channel entirely.
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
	_ "dappco.re/go/inference/examples/internal/engine" // registers the platform engine
)

// The gemma4 channel markers as they appear in decoded text.
const (
	channelOpen  = "<|channel>thought"
	channelClose = "<channel|>"
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

	// Split reasoning from answer on the channel markers.
	text := raw.String()
	thought, answer := "", text
	if rest, ok := strings.CutPrefix(strings.TrimSpace(text), channelOpen); ok {
		if t, a, found := strings.Cut(rest, channelClose); found {
			thought, answer = strings.TrimSpace(t), strings.TrimSpace(a)
		}
	}
	fmt.Println("— thought —")
	fmt.Println(thought)
	fmt.Println("— answer —")
	fmt.Println(answer)
}
