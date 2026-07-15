// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// MTP speculative decoding as a library caller: a Gemma 4 target paired with
// its drafter runs draft -> verify -> accept forwards instead of one token
// per forward. The pairing convention is a fixed suffix on the target's own
// name — target "gemma-4-<size>-it" pairs with drafter
// "gemma-4-<size>-it-assistant" (or a quantised variant such as
// "...-assistant-bf16"); LoadSpeculativePair takes the two directories
// directly rather than auto-detecting the pair (that ladder is serve/generate's
// job, not the library seam).
//
// The speculative lane only engages fully greedy: at temperature 0 the verify
// is exact (byte-identical to plain decode, just faster), so this example
// pins temperature, top_p and top_k all to 0 rather than relying on whatever
// defaults the checkpoint declares.
//
// darwin/arm64 only — this file imports engine/metal directly
// (LoadSpeculativePair), the same call serve/generate wire in through
// serving.SpeculativeLoader.
//
//	go run ./pkg/chat/mtp -model ~/models/gemma-4-e2b-it-bf16 -draft ~/models/gemma-4-e2b-it-assistant-bf16
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"dappco.re/go/inference"
	native "dappco.re/go/inference/engine/metal"

	_ "dappco.re/go/inference/model/builtin" // registers every built-in model architecture
)

func main() {
	model := flag.String("model", os.Getenv("LEM_MODEL"), "target model snapshot directory (config.json + *.safetensors)")
	draft := flag.String("draft", os.Getenv("LEM_DRAFT"), "drafter snapshot directory — the paired assistant checkpoint")
	prompt := flag.String("prompt", "In one paragraph, what makes a good lighthouse keeper?", "user message")
	flag.Parse()
	if *model == "" || *draft == "" {
		fmt.Fprintln(os.Stderr, "set -model and -draft (or LEM_MODEL/LEM_DRAFT) to a target+assistant snapshot pair")
		os.Exit(2)
	}

	// LoadSpeculativePair loads the target checkpoint once and attaches the
	// drafter as one speculative inference.TextModel; draftBlock 0 takes the
	// engine's default (5: the carried lead token + 4 draft proposals per
	// verify forward).
	m, err := native.LoadSpeculativePair(*model, *draft, 0)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load:", err)
		os.Exit(1)
	}
	defer m.Close()

	var reply strings.Builder
	for tok := range m.Chat(context.Background(),
		[]inference.Message{{Role: "user", Content: *prompt}},
		inference.WithMaxTokens(256),
		inference.WithTemperature(0),
		inference.WithTopP(0),
		inference.WithTopK(0),
	) {
		reply.WriteString(tok.Text)
	}
	if er := m.Err(); !er.OK {
		fmt.Fprintln(os.Stderr, "generate:", er.Value)
		os.Exit(1)
	}
	fmt.Println(strings.TrimSpace(reply.String()))

	// Metrics().DecodeTokensPerSec is the same field the plain path reports —
	// the speculative lane feeds the identical inference.GenerateMetrics.
	fmt.Printf("decode %.1f tok/s\n", m.Metrics().DecodeTokensPerSec)
}
