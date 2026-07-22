// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// DFlash block-diffusion speculative decoding as a library caller: the real
// z-lab convention (z-lab/Qwen3-4B-DFlash-b16 and siblings) pairs a small
// 5-layer drafter against its target through the SAME LoadSpeculativePair
// seam chat/mtp uses for the Gemma 4 assistant / Qwen MTP conventions — one
// call, family-dispatched on the drafter's config.json (see
// engine/metal/assistant_dflash_zlab.go's header for how the dispatch
// works and why the z-lab convention needs a wholly separate load path).
//
// The drafter proposes a whole BLOCK of continuation tokens per forward
// (conditioned on the target's own hidden states, fused across several
// layers) instead of one token per forward; decode/dflash.AcceptBlock then
// verifies the block against the target with the ordinary greedy
// prefix-accept rule, so the emitted sequence stays byte-identical to plain
// decode WHATEVER the drafter proposes — losslessness by construction, the
// same invariant chat/mtp's MTP pairing runs on. This example calls
// LoadSpeculativePair directly (the library seam), bypassing serving's
// DFlashEngineProbe gate the same way chat/mtp bypasses nothing extra for
// its own already-armed pairing — the gate is a SERVE-layer policy decision,
// not an engine capability check.
//
// darwin/arm64 only — this file imports engine/metal directly
// (LoadSpeculativePair), the same call serve/generate wire in through
// serving.SpeculativeLoader once DFlashEngineProbe is armed.
//
//	go run ./pkg/dflash -model ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/<snap> \
//	  -draft ~/.cache/huggingface/hub/models--z-lab--Qwen3-4B-DFlash-b16/snapshots/<snap>
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"dappco.re/go/inference"
	native "dappco.re/go/inference/engine/metal"

	_ "dappco.re/go/inference/model/builtin" // registers every built-in model architecture, incl. qwen3
)

func main() {
	model := flag.String("model", os.Getenv("LEM_MODEL"), "target model snapshot directory (Qwen/Qwen3-4B or a sibling z-lab DFlash target)")
	draft := flag.String("draft", os.Getenv("LEM_DRAFT"), "z-lab DFlash drafter snapshot directory (e.g. z-lab/Qwen3-4B-DFlash-b16)")
	prompt := flag.String("prompt", "What is the capital of France?", "user message — Qwen3-4B is chat-tuned, so this rides Chat, not a raw completion")
	maxTokens := flag.Int("max-tokens", 16, "tokens to generate")
	flag.Parse()
	if *model == "" || *draft == "" {
		fmt.Fprintln(os.Stderr, "set -model and -draft (or LEM_MODEL/LEM_DRAFT) to a Qwen3 target + its z-lab DFlash drafter snapshot pair")
		fmt.Fprintln(os.Stderr, `fetch both with:
  python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-4B'))"
  python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('z-lab/Qwen3-4B-DFlash-b16'))"`)
		os.Exit(2)
	}

	// LoadSpeculativePair loads the target once and attaches the drafter as
	// one speculative inference.TextModel; draftBlock 0 takes the drafter
	// checkpoint's own declared block_size (16 on every published z-lab
	// checkpoint — 15 candidate positions per readout, the seed token
	// carried at position 0 is never re-proposed).
	m, err := native.LoadSpeculativePair(*model, *draft, 0)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load:", err)
		os.Exit(1)
	}
	defer func() { _ = m.Close() }()

	// Qwen3 defaults to thinking mode on; the DFlash drafter was trained with
	// it off (the checkpoint's own README: "this draft model is used for
	// thinking mode disabled"), so this example disables it too.
	noThink := false
	fmt.Println("user:", *prompt)
	fmt.Print("assistant: ")
	for tok := range m.Chat(context.Background(),
		[]inference.Message{{Role: "user", Content: *prompt}},
		inference.WithMaxTokens(*maxTokens),
		inference.WithTemperature(0),
		inference.WithTopP(0),
		inference.WithTopK(0),
		inference.WithEnableThinking(&noThink),
	) {
		fmt.Print(tok.Text)
	}
	fmt.Println()
	if er := m.Err(); !er.OK {
		fmt.Fprintln(os.Stderr, "generate:", er.Value)
		os.Exit(1)
	}

	// SpeculativeMetrics reports the drafter's draft/verify counters — the
	// accept-rate receipt: how many of the drafter's proposed tokens the
	// target actually kept. AcceptedTokens == 0 is a SAFE, lossless outcome
	// (every emitted token is still the target's own — see the package doc
	// above), not a wrong answer: it means this run's drafts didn't land,
	// same as a cache miss costs a slower path rather than a wrong one.
	if p, ok := m.(inference.SpeculativeMetricsProvider); ok {
		sm := p.SpeculativeMetrics()
		fmt.Printf("drafter: %d proposed, %d accepted (%.0f%%) over %d rounds\n",
			sm.ProposedTokens, sm.AcceptedTokens, sm.AcceptanceRate*100, sm.TargetVerifyCalls)
	}
}
