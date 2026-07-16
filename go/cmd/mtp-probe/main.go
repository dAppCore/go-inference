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
	"dappco.re/go/inference/model"
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
	if len(os.Args) > 3 && os.Args[3] == "-parity" {
		parity(base)
		return
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
	fmt.Printf("[per-token] output: %q\n", tok.Decode(out))
	fmt.Printf("[per-token] tokens=%d wall=%s (%.2f tok/s) proposed=%d accepted=%d acceptance=%.1f%% draftCalls=%d baseForwards=%d\n",
		len(out), dt.Round(time.Millisecond), float64(len(out))/dt.Seconds(),
		m.ProposedTokens, m.AcceptedTokens, m.AcceptanceRate*100, m.DraftCalls, m.TargetVerifyCalls)

	pair.BlockVerify = true
	t2 := time.Now()
	outB, mb, err := pair.GenerateSpeculative(ids, maxNew, -1, 0)
	if err != nil {
		fmt.Println("generate (block):", err)
		os.Exit(1)
	}
	dtB := time.Since(t2)
	fmt.Printf("[block]     output: %q\n", tok.Decode(outB))
	fmt.Printf("[block]     tokens=%d wall=%s (%.2f tok/s) proposed=%d accepted=%d acceptance=%.1f%% draftCalls=%d baseForwards=%d\n",
		len(outB), dtB.Round(time.Millisecond), float64(len(outB))/dtB.Seconds(),
		mb.ProposedTokens, mb.AcceptedTokens, mb.AcceptanceRate*100, mb.DraftCalls, mb.TargetVerifyCalls)
	same := len(out) == len(outB)
	if same {
		for i := range out {
			if out[i] != outB[i] {
				same = false
				break
			}
		}
	}
	fmt.Printf("lanes agree byte-for-byte: %v  |  block speedup ×%.2f over per-token, tokens/forward %.2f vs 1.00\n",
		same, dt.Seconds()/dtB.Seconds(), float64(len(outB))/float64(mb.TargetVerifyCalls))
}

// parity measures whether the composed BATCHED forward is byte-identical to sequential single-token
// stepping on the real model with the device hooks bound — the evidence that decides whether the
// block-verify lane needs a boundary reforge (gemma4's fold lesson) or is byte-clean as built.
func parity(baseDir string) {
	tm, ok, err := model.LoadComposedDir(baseDir)
	if err != nil || !ok {
		fmt.Println("load:", ok, err)
		os.Exit(1)
	}
	cm := tm.(*composed.ComposedTokenModel).Model()
	tok, err := tokenizer.LoadTokenizer(baseDir + "/tokenizer.json")
	if err != nil {
		fmt.Println("tokenizer:", err)
		os.Exit(1)
	}
	ids := tok.Encode("The quick brown fox jumps over the lazy dog while the seven wise owls watch from the oak")
	fmt.Printf("parity probe over %d ids\n", len(ids))
	batchSess := composed.NewSession(cm)
	t0 := time.Now()
	batched, err := batchSess.Forward(ids)
	if err != nil {
		fmt.Println("batched forward:", err)
		os.Exit(1)
	}
	dtBatch := time.Since(t0)
	seqSess := composed.NewSession(cm)
	t1 := time.Now()
	var seq []float32
	for _, id := range ids {
		h, serr := seqSess.Forward([]int32{id})
		if serr != nil {
			fmt.Println("sequential forward:", serr)
			os.Exit(1)
		}
		seq = append(seq, h...)
	}
	dtSeq := time.Since(t1)
	if len(seq) != len(batched) {
		fmt.Printf("SHAPE MISMATCH: seq %d vs batched %d\n", len(seq), len(batched))
		os.Exit(1)
	}
	exact, worst, worstAt := 0, 0.0, -1
	for i := range batched {
		if batched[i] == seq[i] {
			exact++
		} else if d := abs64(float64(batched[i]) - float64(seq[i])); d > worst {
			worst, worstAt = d, i
		}
	}
	fmt.Printf("floats=%d exact=%d (%.2f%%) worst |Δ|=%g at %d\n",
		len(batched), exact, 100*float64(exact)/float64(len(batched)), worst, worstAt)
	fmt.Printf("batched %s vs sequential %s → batch advantage ×%.2f (%d rows)\n",
		dtBatch.Round(time.Millisecond), dtSeq.Round(time.Millisecond), dtSeq.Seconds()/dtBatch.Seconds(), len(ids))
	if exact == len(batched) {
		fmt.Println("VERDICT: byte-identical — block-verify needs no reforge")
	} else {
		fmt.Println("VERDICT: token-identity tier — block-verify commits need the canonical-boundary discipline")
	}
}

func abs64(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
