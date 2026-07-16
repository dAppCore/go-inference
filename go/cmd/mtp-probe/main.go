// SPDX-Licence-Identifier: EUPL-1.2

// mtp-probe measures the real Qwen MTP head's speculative acceptance against its real base — the
// slice-B receipt and the block-verify go/no-go instrument. Temporary probe; the serve seam is the
// shipping surface.
package main

import (
	"fmt"
	"os"
	"time"

	"dappco.re/go/inference/decode/tokenizer"
	"dappco.re/go/inference/model/composed"

	// Bind the metal device hooks (steel GEMM projections, fused tails, the quant backend) into the
	// composed stack — without this import the probe measures the raw single-core host fallback, not
	// the served lane.
	_ "dappco.re/go/inference/engine/metal"
)

func main() {
	if len(os.Args) < 3 {
		fmt.Println("usage: mtp-probe <base-dir> <draft-dir> [maxNew] [prompt]")
		os.Exit(2)
	}
	base, draft := os.Args[1], os.Args[2]
	maxNew := 48
	if len(os.Args) > 3 {
		fmt.Sscanf(os.Args[3], "%d", &maxNew)
	}
	prompt := "The three primary colours are red, blue and"
	if len(os.Args) > 4 {
		prompt = os.Args[4]
	}
	t0 := time.Now()
	pair, err := composed.LoadSpeculativePairDirs(base, draft)
	if err != nil {
		fmt.Println("load pair:", err)
		os.Exit(1)
	}
	defer pair.Close()
	fmt.Printf("pair loaded in %s (checkpoint draft block %d)\n", time.Since(t0).Round(time.Millisecond), pair.DefaultDraftBlock)
	tok, err := tokenizer.LoadTokenizer(base + "/tokenizer.json")
	if err != nil {
		fmt.Println("tokenizer:", err)
		os.Exit(1)
	}
	ids := tok.Encode(prompt)
	fmt.Printf("prompt %q → %d ids\n", prompt, len(ids))
	t1 := time.Now()
	out, m, err := pair.GenerateSpeculative(ids, maxNew, -1, 0)
	if err != nil {
		fmt.Println("generate:", err)
		os.Exit(1)
	}
	dt := time.Since(t1)
	fmt.Printf("output: %q\n", tok.Decode(out))
	fmt.Printf("tokens=%d wall=%s (%.2f tok/s, per-token verify — no speed lane yet)\n",
		len(out), dt.Round(time.Millisecond), float64(len(out))/dt.Seconds())
	fmt.Printf("speculative: proposed=%d accepted=%d rejected=%d acceptance=%.1f%% draftCalls=%d baseForwards=%d\n",
		m.ProposedTokens, m.AcceptedTokens, m.RejectedTokens, m.AcceptanceRate*100, m.DraftCalls, m.TargetVerifyCalls)
	if m.DraftCalls > 0 {
		fmt.Printf("block-verify projection: committed/round=%.2f (plain=1.00) → ideal decode speedup ×%.2f at equal step cost\n",
			float64(len(out))/float64(m.DraftCalls), float64(len(out))/float64(m.DraftCalls))
	}
}
