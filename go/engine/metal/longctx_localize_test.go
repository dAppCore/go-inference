// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"strconv"
	"testing"

	"dappco.re/go/inference/model"
)

// bf16NaNScan counts bf16 NaNs in raw bytes and reports the first NaN's element index.
// bf16 NaN = exponent all-ones (0x7F80) with a non-zero mantissa (0x007F).
func bf16NaNScan(b []byte) (count, firstIdx int) {
	firstIdx = -1
	for i := 0; i+1 < len(b); i += 2 {
		h := uint16(b[i]) | uint16(b[i+1])<<8
		if h&0x7F80 == 0x7F80 && h&0x007F != 0 {
			if firstIdx < 0 {
				firstIdx = i / 2
			}
			count++
		}
	}
	return count, firstIdx
}

// TestE2BQ4LongContextPrefillCorruption localises the ~52K-token E2B corruption at REAL scale
// (task #343): PrefillTokens a deep synthetic prompt on the e2b-4bit checkpoint, then scan the
// session's per-layer KV views + the retained hidden for NaN and run the greedy head — the
// failing sweep run died with `direct argmax returned invalid token -1` (all-NaN logits) or an
// instant EOS, NON-deterministically, so each rep opens a fresh session and reports which layer
// (and which cache row) first carries NaN. Set E2B_Q4_DIR; LONGCTX_TOKENS (default 52000) and
// LONGCTX_REPS (default 3) size the hunt.
func TestE2BQ4LongContextPrefillCorruption(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	dir := os.Getenv("E2B_Q4_DIR")
	if dir == "" {
		t.Skip("set E2B_Q4_DIR to the e2b-4bit snapshot dir")
	}
	nTokens := 52000
	if v := os.Getenv("LONGCTX_TOKENS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			nTokens = n
		}
	}
	reps := 3
	if v := os.Getenv("LONGCTX_REPS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			reps = n
		}
	}
	const maxLen = 131072
	nm, err := LoadTokenModelDir(dir, maxLen)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	// deterministic pseudorandom prompt ids in the sane vocab interior (content is
	// irrelevant to a structural/raced corruption; ids just need to be valid tokens).
	ids := make([]int32, nTokens)
	x := uint32(0x9E3779B9)
	for i := range ids {
		x ^= x << 13
		x ^= x >> 17
		x ^= x << 5
		ids[i] = int32(1000 + x%200000)
	}

	fails := 0
	for rep := 0; rep < reps; rep++ {
		ns, serr := nm.(model.SessionModel).OpenSession()
		if serr != nil {
			t.Fatalf("rep %d session: %v", rep, serr)
		}
		s := ns.(*ArchSession)
		if s.state.icb == nil {
			t.Fatal("expected an ICB-eligible session — the corruption lives on the recorded-ICB prompt lane")
		}
		// the FULL CLI shape: a cold Generate (the "warm" pass) then a SECOND Generate of
		// the same prompt on the same session — an exact prompt-cache hit at depth, the
		// rollback/cached lane. The sweep failed in both passes on different runs.
		gen, perr := s.Generate(ids, 8, -1)
		if perr != nil {
			t.Logf("rep %d: cold Generate error: %v", rep, perr)
			fails++
			_ = s.Close()
			continue
		}
		t.Logf("rep %d: cold Generate produced %d tokens: %v", rep, len(gen), gen)
		gen2, perr2 := s.GenerateCached(ids, 8, -1)
		if perr2 != nil {
			t.Logf("rep %d: cached Generate error: %v", rep, perr2)
			fails++
			_ = s.Close()
			continue
		}
		if len(gen2) < 2 {
			t.Logf("rep %d: cached Generate produced only %d tokens: %v (the CLI's n<2 failure)", rep, len(gen2), gen2)
			fails++
		} else {
			t.Logf("rep %d: cached Generate produced %d tokens: %v", rep, len(gen2), gen2)
		}

		// scan every layer's KV view (ICB sessions expose the replay caches here)
		views, verr := s.stateLayerViews()
		if verr != nil {
			t.Fatalf("rep %d stateLayerViews: %v", rep, verr)
		}
		nanLayers := 0
		for _, v := range views {
			kc, kf := bf16NaNScan(v.keyBytes)
			vc, vf := bf16NaNScan(v.valueBytes)
			if kc > 0 || vc > 0 {
				nanLayers++
				sp := s.state.specs[v.layer]
				at := "sliding"
				if sp.Attention == model.GlobalAttention {
					at = "GLOBAL "
				}
				t.Logf("rep %d: L%2d %s owns=%v shareFrom=%d  K-NaN=%d(first elem %d)  V-NaN=%d(first elem %d)",
					rep, v.layer, at, sp.OwnsCache(), sp.KVShareFrom, kc, kf, vc, vf)
			}
		}
		// retained hidden + the greedy head — the exact op that returned -1 in the sweep
		hidden := s.retainedHidden
		hc, hf := bf16NaNScan(hidden)
		tok, ok, gerr := s.headEnc.greedy(hidden, nil)
		verdict := "OK"
		if gerr != nil || !ok || tok < 0 {
			verdict = "FAIL"
			fails++
		}
		t.Logf("rep %d: hidden NaN=%d(first %d)  head greedy tok=%d ok=%v err=%v  nanLayers=%d  => %s",
			rep, hc, hf, tok, ok, gerr, nanLayers, verdict)
		if verdict == "OK" && nanLayers > 0 {
			t.Logf("rep %d: NOTE — KV NaNs present but head survived (silent-corruption mode)", rep)
		}
		_ = s.Close()
	}
	t.Logf("=> %d/%d reps corrupt at %d tokens", fails, reps, nTokens)
}
