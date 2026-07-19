// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"os"
	"testing"

	"dappco.re/go/inference/decode/tokenizer"
)

// tq_live_e2e_test.go — the #41 S3 E2E live gate on the REAL gemma-4 e2b-4bit
// checkpoint: a turboquant:4 session decodes coherently against the mode-off
// twin (per-step top-1 agreement + max logit deviation, teacher-forced over
// the SAME token stream), and an in-context retrieval smoke proves a fact
// planted ~1K tokens back is still retrieved through the code caches (the
// global layers' long-range read IS the thing TurboQuant compresses).
//
// Gated behind LEM_REAL_E2B like the sibling real-model tests (loads ~3.5GB;
// the TQ prefill runs the per-token replay — the batched pass declines — so
// the ~1.1K-token retrieval prefill takes tens of seconds).
//
// Measured (this box, 2026-07-19, greedy, 64 teacher-forced steps, real
// e2b-4bit): top-1 agreement 62/64 with max |logit Δ| 6.31 on the 2-pass lane
// (maxLen 2048) and 62/64 with 6.76 on the single-pass lane (maxLen 512) —
// the two lanes sit in the same codec band, and the continuation stays
// coherent. Asserted floors: agreement ≥ 52/64 and max |logit Δ| ≤ 12 —
// roughly 2× margin over the observed band, still collapsing on a structural
// break (stale codes, wrong γ addressing, a dropped bind — the bring-up bug
// read 30/64 at Δ 32 here before the pass-1 ABI repack).

// TestRealE2BTurboQuantLive_Good is the live decode gate: turboquant:4 vs
// mode-off on real weights, teacher-forced.
func TestRealE2BTurboQuantLive_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if os.Getenv("LEM_REAL_E2B") == "" {
		t.Skip("set LEM_REAL_E2B=1 to run the real e2b-4bit TurboQuant live gate (loads ~3.5GB)")
	}
	dir := resolveE2B4bitDir(t)
	const maxLen = 2048 // past sdpa2PassMinKV, so the GLOBAL layers record the live 2-pass TQ pair
	const steps = 64

	lm, dm, err := loadRegistered(dir)
	if err != nil {
		t.Fatalf("loadRegistered: %v", err)
	}
	defer func() { _ = dm.Close() }()
	sb, err := buildShardBuffers(dm)
	if err != nil {
		t.Fatalf("buildShardBuffers: %v", err)
	}
	defer func() { _ = sb.Close() }()
	gs, bits := lm.Embed.GroupSize, lm.Embed.Bits
	qm, err := loadedToQuant(lm, gs, bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	arch := lm.Arch

	native, err := newArchQuantSessionShardsWithHeadConfig(qm, arch, maxLen, sb, nil, archSessionConfig{})
	if err != nil {
		t.Fatalf("native session: %v", err)
	}
	tq, err := newArchQuantSessionShardsWithHeadConfig(qm, arch, maxLen, sb, nil, archSessionConfig{kvCacheMode: "turboquant:4"})
	if err != nil {
		t.Fatalf("turboquant session: %v", err)
	}
	if tq.state.icb == nil || !tq.state.icb.hasKVTQ() {
		t.Fatal("turboquant session did not arm the TQ carrier on the real checkpoint")
	}

	tok, err := tokenizer.LoadTokenizer(dir + "/tokenizer.json")
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	prompt := append([]int32{2}, tok.Encode("The quick brown fox jumps over the lazy dog. In a quiet village by the sea,")...)
	if err := native.PrefillTokens(prompt); err != nil {
		t.Fatalf("native prefill: %v", err)
	}
	if err := tq.PrefillTokens(prompt); err != nil {
		t.Fatalf("tq prefill: %v", err)
	}

	// Teacher-forced comparison: BOTH sessions consume the native session's
	// greedy stream, so every step compares logits over the identical context.
	embScale := embedScaleOf(arch)
	head := func(hidden []byte) []float32 {
		logits, herr := LMHeadQuant(hidden, qm.FinalNorm, qm.LMHead, qm.LMHeadScales, qm.LMHeadBiases, arch.Hidden, arch.Vocab, gs, bits, arch.Eps, arch.SoftCap)
		if herr != nil {
			t.Fatalf("LMHeadQuant: %v", herr)
		}
		return bf16ToF32Slice(logits)
	}
	argmax := func(l []float32) int {
		best := 0
		for i, v := range l {
			if v > l[best] {
				best = i
			}
		}
		return best
	}
	next := prompt[len(prompt)-1]
	agree := 0
	var maxLogitDelta float64
	var text []byte
	for step := 0; step < steps; step++ {
		emb, eerr := embedTokenQuant(qm.Embed, qm.EmbedScales, qm.EmbedBiases, next, arch.Vocab, arch.Hidden, gs, bits, embScale)
		if eerr != nil {
			t.Fatalf("embed: %v", eerr)
		}
		hN, nerr := native.StepWithID(next, emb)
		if nerr != nil {
			t.Fatalf("native step %d: %v", step, nerr)
		}
		hT, terr := tq.StepWithID(next, emb)
		if terr != nil {
			t.Fatalf("tq step %d: %v", step, terr)
		}
		lN, lT := head(hN), head(hT)
		aN, aT := argmax(lN), argmax(lT)
		if aN == aT {
			agree++
		}
		stepMax := 0.0
		for i := range lN {
			if d := math.Abs(float64(lN[i]) - float64(lT[i])); d > stepMax {
				stepMax = d
			}
		}
		if stepMax > maxLogitDelta {
			maxLogitDelta = stepMax
		}
		if os.Getenv("TQ_DEBUG") != "" {
			t.Logf("step %2d: native %6d %q  tq %6d %q  agree=%v  |Δ|max %.3f", step, aN, tok.DecodeToken(int32(aN)), aT, tok.DecodeToken(int32(aT)), aN == aT, stepMax)
		}
		next = int32(aN) // teacher-force the native stream into both
		text = append(text, []byte(tok.DecodeToken(next))...)
	}
	t.Logf("turboquant:4 vs native over %d teacher-forced greedy steps: top-1 agreement %d/%d, max |logit Δ| %.4f", steps, agree, steps, maxLogitDelta)
	t.Logf("native greedy continuation: %q", string(text))
	if agree < steps-12 {
		t.Fatalf("top-1 agreement %d/%d under the observed band (≥ %d)", agree, steps, steps-12)
	}
	if maxLogitDelta > 12.0 || math.IsNaN(maxLogitDelta) {
		t.Fatalf("max |logit Δ| %.4f outside the observed band (≤ 12)", maxLogitDelta)
	}
}

// TestRealE2BTurboQuantLive_Retrieval_Good is the in-context retrieval smoke:
// a fact planted ~1K tokens back must surface through the TurboQuant code
// caches (greedy continuation names the planted code).
func TestRealE2BTurboQuantLive_Retrieval_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if os.Getenv("LEM_REAL_E2B") == "" {
		t.Skip("set LEM_REAL_E2B=1 to run the real e2b-4bit TurboQuant retrieval smoke (loads ~3.5GB; the TQ prefill is per-token)")
	}
	dir := resolveE2B4bitDir(t)
	const maxLen = 2048

	lm, dm, err := loadRegistered(dir)
	if err != nil {
		t.Fatalf("loadRegistered: %v", err)
	}
	defer func() { _ = dm.Close() }()
	sb, err := buildShardBuffers(dm)
	if err != nil {
		t.Fatalf("buildShardBuffers: %v", err)
	}
	defer func() { _ = sb.Close() }()
	gs, bits := lm.Embed.GroupSize, lm.Embed.Bits
	qm, err := loadedToQuant(lm, gs, bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	arch := lm.Arch

	tq, err := newArchQuantSessionShardsWithHeadConfig(qm, arch, maxLen, sb, nil, archSessionConfig{kvCacheMode: "turboquant:4"})
	if err != nil {
		t.Fatalf("turboquant session: %v", err)
	}
	tok, err := tokenizer.LoadTokenizer(dir + "/tokenizer.json")
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}

	// Plant the fact, pad ~1K tokens of neutral prose, then ask.
	fact := "<start_of_turn>user\nRemember this: the secret code is 7412. "
	filler := ""
	for len(tok.Encode(fact+filler)) < 1050 {
		filler += "The library stood at the corner of the old town square, its shelves heavy with maps and letters. Every morning the caretaker opened the tall windows and let the sea air move through the reading rooms. "
	}
	question := "\nWhat is the secret code? Answer with just the number.<end_of_turn>\n<start_of_turn>model\n"
	ids := append([]int32{2}, tok.Encode(fact+filler+question)...)
	if len(ids) > maxLen-32 {
		ids = ids[:maxLen-32]
	}
	if err := tq.PrefillTokens(ids); err != nil {
		t.Fatalf("tq prefill (%d tokens): %v", len(ids), err)
	}
	gen, err := tq.GenerateFromCache(16, -1)
	if err != nil {
		t.Fatalf("GenerateFromCache: %v", err)
	}
	out := tok.Decode(gen)
	t.Logf("retrieval over %d prefilled tokens (fact ~%d back): %q", len(ids), len(ids)-60, out)
	if !containsStr(out, "7412") {
		t.Fatalf("planted code 7412 not retrieved through the TQ cache: %q", out)
	}
}

// containsStr is a tiny local contains (the test file keeps the banned-import
// discipline of the package: core's helpers cover production code; tests may
// use them too, but this avoids importing core for one call).
func containsStr(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
