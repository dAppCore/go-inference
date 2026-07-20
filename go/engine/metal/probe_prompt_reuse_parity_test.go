// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"strconv"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/decode/tokenizer"
)

// TestProbePromptReuseParity (LTHN_PROBE_MODEL-gated) is the #1845 instrument
// and its regression gate: on a REAL checkpoint it replays the serve shape
// that broke reuse determinism — a warmup request sharing a literal token
// prefix with the next prompt — and asserts the REUSE CONTRACT (a cached
// prefill decodes the same bytes as a fresh one; hard Errorf), while logging
// the raw-session diagnostics that localised the mechanism.
//
// VERDICT (gemma-4-31B-it-4bit @ context 6144): the divergence was
// never the reuse bookkeeping — a plain PrefillTokens(prefix)+AppendTokens
// (suffix) split diverges identically with NO reuse machinery involved. The
// batched landing forward is intra-batch TILE-POSITION sensitive: the same
// absolute row lands a few bf16 ulps apart depending on its offset within the
// projection/rope/norm batch (both fresh and reuse take qmm_t — it is not a
// kernel-choice mismatch; K max|Δ| ~0.004-0.016, V ~0.06-0.13 on 31B, from the
// APPEND boundary onward). bf16 caches absorb that inside greedy margins, but
// the q8 store quantises the wobble discontinuously and amplifies it ~4×,
// flipping knife-edge tokens (the #1845 decline).
//
// FIX (#1846): reuseCanonicalLanding routes every prefill/append through the
// POSITION-INVARIANT per-token lane, landing each row at batch offset 0 — the
// only shape whose reused prefix+suffix is byte-identical to a fresh whole
// prefill (TestProbeCanonicalLanding is the KV-level receipt; the batched
// forward cannot be block-anchored to byte-identity on sliding-window models).
// The reuse arms below arm it and REQUIRE engagement (reused>0) + token parity
// against a canonical fresh; the split-canonical arm shows the same mechanism
// makes the raw split go parity. The cost is a decode-speed (per-token) prefill,
// so the mode is opt-in — the batched diagnostics below keep their own batched
// ground truth and still log the tier (expected, non-fatal).
//
//	LTHN_PROBE_MODEL=<snapshot dir> [LTHN_PROBE_CONTEXT=6144] \
//	  go test -tags metal_runtime ./engine/metal/ -run TestProbePromptReuseParity -v
func TestProbePromptReuseParity(t *testing.T) {
	requireNativeRuntime(t)
	dir := os.Getenv("LTHN_PROBE_MODEL")
	if dir == "" {
		t.Skip("LTHN_PROBE_MODEL not set")
	}
	maxLen := 6144
	if v := os.Getenv("LTHN_PROBE_CONTEXT"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n <= 0 {
			t.Fatalf("LTHN_PROBE_CONTEXT=%q: not a positive integer", v)
		}
		maxLen = n
	}

	tm, err := LoadTokenModelDir(dir, maxLen)
	if err != nil {
		t.Fatalf("LoadTokenModelDir: %v", err)
	}
	ntm := tm.(*NativeTokenModel)
	defer ntm.Close()

	tok, err := tokenizer.LoadTokenizer(core.PathJoin(dir, "tokenizer.json"))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	// The live repro's texts (raw — the template is irrelevant to parity, the
	// shared literal prefix is what arms the rollback).
	warmIDs := tok.Encode("Write the integers from 90001 to 90801 separated by single spaces. Output only the numbers, nothing else.")
	aIDs := tok.Encode("Write the integers from 1 to 800 separated by single spaces. Output only the numbers, nothing else.")
	lcp := 0
	for lcp < len(warmIDs) && lcp < len(aIDs) && warmIDs[lcp] == aIDs[lcp] {
		lcp++
	}
	t.Logf("warm=%d toks, A=%d toks, shared token prefix=%d", len(warmIDs), len(aIDs), lcp)
	if lcp == 0 {
		t.Fatalf("prompts share no token prefix — the probe shape is wrong")
	}

	const warmGen = 24
	const parityGen = 48

	openArch := func(label string) *ArchSession {
		stepper, oerr := ntm.OpenSession()
		if oerr != nil {
			t.Fatalf("OpenSession(%s): %v", label, oerr)
		}
		return stepper.(*ArchSession)
	}
	boundaryLogits := func(s *ArchSession) []float32 {
		if len(s.retainedLogits) == 0 {
			return nil
		}
		return bf16BytesToF32(s.retainedLogits)
	}

	// FRESH — the ground truth (batched landing) for the mechanism diagnostics.
	fresh := openArch("fresh")
	defer fresh.Close()
	if err := fresh.PrefillTokens(aIDs); err != nil {
		t.Fatalf("fresh PrefillTokens: %v", err)
	}
	freshToks, err := fresh.GenerateFromCache(parityGen, -1)
	if err != nil {
		t.Fatalf("fresh GenerateFromCache: %v", err)
	}

	// FRESH-CANONICAL — the ground truth for the REUSE arms (#1846). Reuse now
	// engages under position-invariant canonical landing, so its reference is a
	// whole prefill in the SAME landing mode; the reused cache is byte-identical
	// to it (TestProbeCanonicalLanding is the KV-level receipt).
	freshCanon := openArch("fresh-canonical")
	freshCanon.reuseCanonicalLanding = true
	defer freshCanon.Close()
	if err := freshCanon.PrefillTokens(aIDs); err != nil {
		t.Fatalf("fresh-canonical PrefillTokens: %v", err)
	}
	freshCanonLogits := boundaryLogits(freshCanon)
	freshCanonToks, err := freshCanon.GenerateFromCache(parityGen, -1)
	if err != nil {
		t.Fatalf("fresh-canonical GenerateFromCache: %v", err)
	}

	run := func(label string, decodeWarm bool) {
		s := openArch(label)
		s.reuseCanonicalLanding = true // #1846: reuse engages under canonical landing
		defer s.Close()
		if err := s.PrefillTokens(warmIDs); err != nil {
			t.Fatalf("%s PrefillTokens(warm): %v", label, err)
		}
		if decodeWarm {
			if _, err := s.GenerateFromCache(warmGen, -1); err != nil {
				t.Fatalf("%s warm GenerateFromCache: %v", label, err)
			}
		}
		reused, err := s.PrefillTokensCached(aIDs)
		if err != nil {
			t.Fatalf("%s PrefillTokensCached: %v", label, err)
		}
		if reused == 0 {
			t.Errorf("%s: reuse DECLINED (reused=0) — canonical landing must ENGAGE q8 reuse (#1846)", label)
			return
		}
		gotLogits := boundaryLogits(s)
		got, err := s.GenerateFromCache(parityGen, -1)
		if err != nil {
			t.Fatalf("%s GenerateFromCache: %v", label, err)
		}

		div := -1
		n := len(got)
		if len(freshCanonToks) < n {
			n = len(freshCanonToks)
		}
		for i := 0; i < n; i++ {
			if got[i] != freshCanonToks[i] {
				div = i
				break
			}
		}
		if div < 0 && len(got) != len(freshCanonToks) {
			div = n
		}
		if div < 0 {
			t.Logf("%s: PARITY over %d tokens (reused=%d, ENGAGED)", label, len(got), reused)
			return
		}
		// Divergence: characterise it. The boundary logits cover step 0 only;
		// a later divergence still reports the max boundary delta (the drift
		// indicator) plus the token ids either side.
		maxD, at := float32(0), -1
		for i := range freshCanonLogits {
			if i >= len(gotLogits) {
				break
			}
			d := freshCanonLogits[i] - gotLogits[i]
			if d < 0 {
				d = -d
			}
			if d > maxD {
				maxD, at = d, i
			}
		}
		t.Errorf("%s: DIVERGED at token %d (fresh=%d got=%d, reused=%d); boundary-logit max|Δ|=%g at vocab %d",
			label, div, tokAt(freshCanonToks, div), tokAt(got, div), reused, maxD, at)
	}

	run("reuse-after-decode", true)
	run("reuse-no-decode", false)

	// SPLIT-PREFILL — no reuse machinery at all: a fresh session prefills the
	// shared prefix then APPENDS the rest, exactly the multi-turn shape the
	// continuity and -state lanes use. Parity here pins the bug to the
	// truncate/stale-row interaction; divergence here means AppendTokens'
	// split prefill itself breaks on this model — a much wider defect than
	// prompt reuse.
	// The minimal repro is the split prefill alone; run it under a toggle
	// matrix, each config with its OWN fresh ground truth (decode paths are
	// not numerically identical to each other, so cross-config comparison
	// would confound). Whichever kill switch restores within-config parity
	// names the subsystem.
	freshArm := func(label string) []int32 {
		s := openArch(label)
		defer s.Close()
		if err := s.PrefillTokens(aIDs); err != nil {
			t.Fatalf("%s PrefillTokens: %v", label, err)
		}
		toks, gerr := s.GenerateFromCache(parityGen, -1)
		if gerr != nil {
			t.Fatalf("%s GenerateFromCache: %v", label, gerr)
		}
		return toks
	}
	splitArm := func(label string) []int32 {
		split := openArch(label)
		defer split.Close()
		if err := split.PrefillTokens(aIDs[:lcp]); err != nil {
			t.Fatalf("%s PrefillTokens(prefix): %v", label, err)
		}
		if err := split.AppendTokens(aIDs[lcp:]); err != nil {
			t.Fatalf("%s AppendTokens(suffix): %v", label, err)
		}
		toks, gerr := split.GenerateFromCache(parityGen, -1)
		if gerr != nil {
			t.Fatalf("%s GenerateFromCache: %v", label, gerr)
		}
		return toks
	}
	compare := func(label string, hard bool, want, got []int32) {
		for i := 0; i < len(got) && i < len(want); i++ {
			if got[i] != want[i] {
				if hard {
					t.Errorf("%s: DIVERGED at token %d (fresh=%d split=%d)", label, i, want[i], got[i])
				} else {
					t.Logf("%s: diverged at token %d (fresh=%d split=%d) — the documented batch-shape numeric tier (raw session ops, not the reuse contract)", label, i, want[i], got[i])
				}
				return
			}
		}
		t.Logf("%s: PARITY over %d tokens", label, len(got))
	}
	compare("baseline", false, freshToks, splitArm("split-prefill"))

	// CANONICAL SPLIT — the same split prefill under position-invariant landing
	// (#1846). This is the SAME mechanism the reuse arms use: byte-identical
	// landing regardless of batch shape, so the split diagnostic goes PARITY
	// (hard) rather than the batched tier the baseline arm logs.
	canonSplit := func() []int32 {
		s := openArch("split-canonical")
		s.reuseCanonicalLanding = true
		defer s.Close()
		if err := s.PrefillTokens(aIDs[:lcp]); err != nil {
			t.Fatalf("split-canonical PrefillTokens(prefix): %v", err)
		}
		if err := s.AppendTokens(aIDs[lcp:]); err != nil {
			t.Fatalf("split-canonical AppendTokens(suffix): %v", err)
		}
		toks, gerr := s.GenerateFromCache(parityGen, -1)
		if gerr != nil {
			t.Fatalf("split-canonical GenerateFromCache: %v", gerr)
		}
		return toks
	}
	compare("split-canonical", true, freshCanonToks, canonSplit())

	icbDisabledForTest = true
	compare("noicb", false, freshArm("fresh-noicb"), splitArm("split-noicb"))
	icbDisabledForTest = false

	prefillSkipSharedOffForTest = true
	compare("noskipshared", false, freshArm("fresh-noskip"), splitArm("split-noskip"))
	prefillSkipSharedOffForTest = false

	chainedGPUInputsDisabled = true
	compare("nochained", false, freshArm("fresh-nochained"), splitArm("split-nochained"))
	chainedGPUInputsDisabled = false

	// THE SURGICAL ARM: same WHOLE prompt, one session prefilled through the
	// ICB batch path (the pos==0 default on this model), one with the host
	// batched-dense path forced (toggle held ONLY across the prefill; decode
	// runs identically on both). Divergence here pins the root cause to the
	// two prefill implementations disagreeing numerically — the mix that
	// split/append sessions are built from.
	hostPrefill := openArch("fresh-hostprefill")
	defer hostPrefill.Close()
	icbDisabledForTest = true
	err = hostPrefill.PrefillTokens(aIDs)
	icbDisabledForTest = false
	if err != nil {
		t.Fatalf("fresh-hostprefill PrefillTokens: %v", err)
	}
	hostToks, err := hostPrefill.GenerateFromCache(parityGen, -1)
	if err != nil {
		t.Fatalf("fresh-hostprefill GenerateFromCache: %v", err)
	}
	compare("icb-vs-host-prefill", true, freshToks, hostToks)

	// CHUNK ARM: a fresh whole prefill FORCED through the chunked path
	// (cap=lcp so the first chunk boundary lands exactly where the split's
	// append boundary does). An append at pos>0 is structurally a chunk, so
	// divergence here pins the bug to the multi-call/chunk position machinery
	// independent of TruncateTo/AppendTokens bookkeeping.
	chunked := openArch("fresh-chunked")
	defer chunked.Close()
	chunked.prefillChunkRowsCap = lcp
	if err := chunked.PrefillTokens(aIDs); err != nil {
		t.Fatalf("fresh-chunked PrefillTokens: %v", err)
	}
	chunkToks, err := chunked.GenerateFromCache(parityGen, -1)
	if err != nil {
		t.Fatalf("fresh-chunked GenerateFromCache: %v", err)
	}
	compare("chunked-vs-whole", true, freshToks, chunkToks)

	// HIDDEN ARM: the boundary hidden (last-position residual after prefill)
	// compared f32-wise between a whole prefill and a split prefill — a large
	// delta = structural distortion of the appended forward (mask/rope); a
	// tiny one = accumulation-order numerics.
	hFresh := openArch("hidden-fresh")
	defer hFresh.Close()
	if err := hFresh.PrefillTokens(aIDs); err != nil {
		t.Fatalf("hidden-fresh PrefillTokens: %v", err)
	}
	hSplit := openArch("hidden-split")
	defer hSplit.Close()
	if err := hSplit.PrefillTokens(aIDs[:lcp]); err != nil {
		t.Fatalf("hidden-split PrefillTokens(prefix): %v", err)
	}
	if err := hSplit.AppendTokens(aIDs[lcp:]); err != nil {
		t.Fatalf("hidden-split AppendTokens: %v", err)
	}
	// FLASH-APPEND ARM: the same split, but the appended suffix forced through
	// prefillPromptRetainedInPool — the lane a FRESH prefill takes (and the
	// lane restored -state sessions already append through). Parity here means
	// the fix is routing appends onto the fresh lane; divergence means the
	// flash lane itself breaks at pos>0.
	fa := openArch("split-flashappend")
	defer fa.Close()
	if err := fa.PrefillTokens(aIDs[:lcp]); err != nil {
		t.Fatalf("split-flashappend PrefillTokens(prefix): %v", err)
	}
	fa.resetRetainedLogits()
	faHidden, err := fa.prefillPromptRetainedInPool(aIDs[lcp:])
	if err != nil {
		t.Fatalf("split-flashappend prefillPromptRetainedInPool: %v", err)
	}
	fa.cachedIDs = append(fa.cachedIDs, aIDs[lcp:]...)
	fa.clearCachedPromptHidden()
	fa.rememberRetainedHidden(faHidden)
	faToks, err := fa.GenerateFromCache(parityGen, -1)
	if err != nil {
		t.Fatalf("split-flashappend GenerateFromCache: %v", err)
	}
	compare("flash-append-vs-whole", false, freshToks, faToks)

	fh := bf16BytesToF32(hFresh.retainedHidden)
	sh := bf16BytesToF32(hSplit.retainedHidden)
	if len(fh) == 0 || len(sh) == 0 || len(fh) != len(sh) {
		t.Logf("hidden arm: retained hidden unavailable (fresh=%d split=%d f32s)", len(fh), len(sh))
	} else {
		maxD, at, nDiff := float32(0), -1, 0
		for i := range fh {
			d := fh[i] - sh[i]
			if d < 0 {
				d = -d
			}
			if d > 0 {
				nDiff++
			}
			if d > maxD {
				maxD, at = d, i
			}
		}
		t.Logf("boundary hidden: %d/%d components differ, max|Δ|=%g at dim %d (fresh=%g split=%g)",
			nDiff, len(fh), maxD, at, fh[at], sh[at])
	}

	// KV-DIFF ARM: the caches themselves, row by row (views dequantise the q8
	// mirrors), between the whole and split sessions — WHICH layers and WHICH
	// rows disagree localises the landing that differs. Prefix rows differing
	// = the short first call landed different bytes; suffix rows = the append
	// landed different bytes.
	fViews, err := hFresh.stateLayerViews()
	if err != nil {
		t.Fatalf("fresh stateLayerViews: %v", err)
	}
	sViews, err := hSplit.stateLayerViews()
	if err != nil {
		t.Fatalf("split stateLayerViews: %v", err)
	}
	if len(fViews) != len(sViews) {
		t.Fatalf("view counts differ: fresh=%d split=%d", len(fViews), len(sViews))
	}
	rows := len(aIDs)
	logged := 0
	for vi := range fViews {
		fv, sv := fViews[vi], sViews[vi]
		diffRows := func(fb, sb []byte, kind string) {
			firstRow, rowMax := -1, float32(0)
			for r := 0; r < rows; r++ {
				off := r * fv.rowBytes
				if off+fv.rowBytes > len(fb) || off+fv.rowBytes > len(sb) {
					break
				}
				fr := bf16BytesToF32(fb[off : off+fv.rowBytes])
				sr := bf16BytesToF32(sb[off : off+fv.rowBytes])
				for i := range fr {
					d := fr[i] - sr[i]
					if d < 0 {
						d = -d
					}
					if d > 0 && firstRow < 0 {
						firstRow = r
					}
					if d > rowMax {
						rowMax = d
					}
				}
			}
			if firstRow >= 0 && logged < 12 {
				t.Logf("kvdiff L%02d %s: first differing row=%d (prefix boundary=%d) max|Δ|=%g", fv.layer, kind, firstRow, lcp, rowMax)
				logged++
			}
		}
		diffRows(fv.keyBytes, sv.keyBytes, "K")
		diffRows(fv.valueBytes, sv.valueBytes, "V")
	}
	if logged == 0 {
		t.Logf("kvdiff: all owner-layer K/V rows 0..%d byte-identical between whole and split", rows-1)
	}
}

func tokAt(toks []int32, i int) int32 {
	if i < 0 || i >= len(toks) {
		return -1
	}
	return toks[i]
}
