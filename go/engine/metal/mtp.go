// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"os"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// mtp.go — speculative (multi-token-prediction) decode over two ArchSessions, a fast DRAFT
// proposing K tokens and the TARGET verifying them in one pass over its OWN resident cache.
// The result is TOKEN-IDENTICAL to plain greedy Generate on the target: a draft is only ever
// an ACCELERATOR — every position is decided by the target's greedy argmax, so a wrong draft
// token merely falls back to the token the target would have emitted anyway. Correctness does
// NOT depend on draft quality (a divergent draft just accepts nothing and runs at greedy speed).
//
// Why this is the native multi-token forward against the session cache: the target's verify runs
// the K draft tokens through StepWithID, which writes each token's K/V into the session's growing
// cache at the live position and advances — exactly what Generate does internally, so the verified
// hiddens are byte-identical to stepping the same tokens one at a time. There is no separate fused
// multi-token kernel to drive (DecodeForward/DecodeForwardArch allocate their OWN caches and step
// internally, so they cannot target a session's resident cache); K sequential steps over the
// session cache IS the cache-faithful batched forward, and the win is replacing K target head+step
// rounds with one draft+verify round whenever the draft guesses right.
//
// Accept rule (standard MTP, matched to plain greedy): with h the hidden of the last committed
// token, the target's next token is T0 = greedy(head(h)). The draft proposes d0..d_{K-1}; each
// d_i is stepped on the TARGET to get hidden_i, and the target's greedy there is T_{i+1} =
// greedy(head(hidden_i)). Accept the longest prefix where d_i == T_i (i.e. the draft matched what
// the target would have emitted), then emit ONE bonus correction token T_j (the target's greedy at
// the first mismatch, or after a full-length accept) and step it. The cursor hidden h for the next
// round is the bonus token's hidden. Every emitted id is a target greedy id, so the stream equals
// target.Generate(promptIDs, maxNew, eos) token for token.
//
// Cache rollback on reject: the target steps all K draft tokens (advancing pos by K), then pos is
// reset to the accepted length + the committed bonus token. The rejected suffix's K/V rows are
// simply overwritten by the next write at that position (stepToken writes at pos and SDPA attends a
// pos+1 window — see decode_forward_arch.go), so resetting pos is a complete rollback. This is exact
// for owner caches. Sliding-ring/device-paged caches keep only the visible window; rollback stores
// the absolute offset separately and the batched bridge syncs the physical ring slots before and
// after verification, so a speculative window may straddle a ring wrap.

// MTPResult reports a speculative decode: the generated ids (target-greedy, identical to plain
// Generate) plus the acceptance accounting — how many draft tokens were proposed vs accepted, and
// how many draft+verify rounds ran. Drafted/Accepted give the realised acceptance rate; with a
// perfect draft Accepted≈Drafted and Rounds is small, with a useless draft Accepted is ~0 and the
// stream is unchanged (still correct, just no speedup).
type MTPResult struct {
	Tokens   []int32
	Drafted  int // total draft tokens proposed across all rounds
	Accepted int // draft tokens that matched the target's greedy (the realised speculative win)
	Rounds   int // draft→verify rounds executed
}

// MTPDecode speculatively decodes up to maxNew tokens on target, using draft to propose K tokens
// per round, returning the target-greedy token stream (token-identical to target.Generate) plus the
// acceptance stats. eosID < 0 disables early stop. Both sessions are advanced as a side effect: the
// target ends positioned exactly after the committed sequence (prompt + emitted tokens), the draft
// after its last proposal — drive each from a single goroutine (the ArchSession contract).
//
// The two sessions are independent caches: typically draft is a small/cheap model and target the
// real one, but for correctness they may share weights (the draft then accepts everything and the
// speedup is maximal) or diverge wildly (nothing accepts, greedy speed) — the output is the same.
// mtpDiagForTest prints the first rounds' t0/drafts/greedys (#352 instrument; env LTHN_MTP_DIAG).
var mtpDiagForTest = os.Getenv("LTHN_MTP_DIAG") != ""

// mtpNoFusedForTest forces the legacy per-op drafter path (#352 A/B; env LTHN_MTP_NOFUSED).
var mtpNoFusedForTest = os.Getenv("LTHN_MTP_NOFUSED") != ""

// mtpVerifyFoldDisabled restores the per-row verify lane (LTHN_MTP_VERIFY_FOLD=0):
// the MTP verify otherwise takes the batched fold at small K — weights swept once
// per block instead of once per drafted row.
var mtpVerifyFoldDisabled = os.Getenv("LTHN_MTP_VERIFY_FOLD") == "0"

// mtpReengageDisabled restores the permanent low-accept bail (LTHN_MTP_REENGAGE=0):
// the speculative loop otherwise runs plain for a bounded cooldown after a bail and
// re-probes drafting, staying engaged only when the probe's measured emitted-token
// rate is at least the plain stretch's (#299).
var mtpReengageDisabled = os.Getenv("LTHN_MTP_REENGAGE") == "0"

// mtpDiagDraftCalls counts draft-block invocations for the #352 instrument (single decode goroutine).
var mtpDiagDraftCalls int

// mtpDiagVerifyRowsCalls counts verify-rows invocations for the #352 instrument.
var mtpDiagVerifyRowsCalls int

// mtpDiagBF16Stats summarises a bf16 buffer for the #352 instrument.
func mtpDiagBF16Stats(b []byte) string {
	n := len(b) / 2
	var sum, maxAbs float64
	nan := 0
	for i := 0; i < n; i++ {
		bits := uint32(b[2*i]) | uint32(b[2*i+1])<<8
		f := float64(math.Float32frombits(bits << 16))
		if math.IsNaN(f) {
			nan++
			continue
		}
		a := math.Abs(f)
		sum += a
		if a > maxAbs {
			maxAbs = a
		}
	}
	return core.Sprintf("sum|x|=%.1f max|x|=%.3f nan=%d n=%d", sum, maxAbs, nan, n)
}

func MTPDecode(target, draft *ArchSession, promptIDs []int32, maxNew, eosID, k int) (*MTPResult, error) {
	return MTPDecodeEach(target, draft, promptIDs, maxNew, eosID, k, nil)
}

// MTPDecodeEach is MTPDecode with a streaming token sink. yield receives each
// committed token in order; returning false stops after that token and returns
// the partial result.
func MTPDecodeEach(target, draft *ArchSession, promptIDs []int32, maxNew, eosID, k int, yield func(int32) bool) (*MTPResult, error) {
	if target == nil || draft == nil {
		return nil, core.NewError("native.MTPDecode: nil target/draft session")
	}
	if len(promptIDs) == 0 {
		return nil, core.NewError("native.MTPDecode: empty prompt")
	}
	if maxNew <= 0 {
		return nil, core.NewError("native.MTPDecode: maxNew must be > 0")
	}
	if k <= 0 {
		return nil, core.NewError("native.MTPDecode: k must be > 0")
	}
	// The loop caps each draft block to the remaining emitted-token budget, so neither cache needs
	// rows beyond the final committed prompt+generated sequence.
	if target.pos+len(promptIDs)+maxNew > target.maxLen {
		return nil, core.NewError("native.MTPDecode: target sequence would exceed maxLen cache rows")
	}
	if draft.pos+len(promptIDs)+maxNew > draft.maxLen {
		return nil, core.NewError("native.MTPDecode: draft sequence would exceed maxLen cache rows")
	}

	res := &MTPResult{Tokens: make([]int32, 0, maxNew)}
	targetStartPos := target.pos
	draftStartPos := draft.pos
	var verifyStack [16]int32
	verifyIDs := verifyStack[:1]
	if k+1 > len(verifyStack) {
		verifyIDs = make([]int32, 1, k+1)
	}
	var greedyStack [16]int32
	greedyIDs := greedyStack[:0]
	if k+1 > len(greedyStack) {
		greedyIDs = make([]int32, 0, k+1)
	}

	// prefill the prompt into BOTH sessions; keep the target's last hidden as the cursor h. The
	// draft is advanced in lockstep so its cache holds the same committed history before it proposes.
	hidden, ok, err := target.prefillMTPPrompt(promptIDs, true)
	if err != nil {
		return nil, err
	}
	if !ok {
		for i, id := range promptIDs {
			h, err := target.stepID(id)
			if err != nil {
				return nil, err
			}
			if i == len(promptIDs)-1 {
				hidden = h
			}
		}
	}
	if _, ok, err = draft.prefillMTPPrompt(promptIDs, false); err != nil {
		return nil, err
	} else if !ok {
		for _, id := range promptIDs {
			if _, err := draft.stepID(id); err != nil {
				return nil, err
			}
		}
	}
	if hidden == nil {
		return nil, core.NewError("native.MTPDecode: prompt prefill produced no cursor hidden")
	}

	// each round: read the target's greedy at the cursor (T0, always committed), let the draft
	// propose K continuations, verify them against the target's cache, commit the accepted run plus
	// one bonus correction, and carry the bonus's hidden as the next cursor — until maxNew/eos.
	for len(res.Tokens) < maxNew {
		res.Rounds++
		draftPos0 := draft.pos // draft cache position at round start; the committed run is replayed from here to keep the draft aligned with the target

		// the token the target emits at the cursor (round's first committed token); this is T0.
		t0, err := target.greedyOf(hidden)
		if err != nil {
			return nil, err
		}

		// DRAFT: propose K tokens. The draft seeds from t0 (the token actually being committed),
		// stepping its own cache; quality is irrelevant to correctness. We stop drafting early if the
		// committed sequence would already reach maxNew — no point proposing tokens we can't emit.
		room := maxNew - len(res.Tokens) // tokens still emittable INCLUDING t0
		nDraft := max(
			// -1: t0 itself occupies one emit slot
			min(k,

				room-1), 0)
		verifyIDs = verifyIDs[:1]
		verifyIDs[0] = t0
		drafts := verifyIDs[1:1]
		seed := t0
		for d := 0; d < nDraft; d++ {
			dh, err := draft.stepID(seed)
			if err != nil {
				return nil, err
			}
			nd, err := draft.greedyOf(dh)
			if err != nil {
				return nil, err
			}
			drafts = append(drafts, nd)
			seed = nd
		}
		verifyIDs = verifyIDs[:1+len(drafts)]
		res.Drafted += len(drafts)

		// VERIFY: run [t0, drafts...] through the TARGET's cache from the current pos in one pass of
		// sequential steps (the multi-token forward against the resident cache). After stepping token
		// x at a position, the target's greedy of that hidden is what it would emit AFTER x — i.e. the
		// expected value of the NEXT proposed token. So:
		//   step t0           → expect drafts[0]
		//   step drafts[0]    → expect drafts[1]
		//   ...
		// accept the longest prefix of drafts that matches, then the first mismatch's expected token
		// is the bonus correction. posBefore lets us roll the target cache back to the committed length.
		posBefore := target.pos
		commitLen := 1             // t0 is always committed (it's the target's own greedy)
		bonusHidden := []byte(nil) // filled when we step the committed bonus token below
		accepted := 0
		var bonus int32
		// compute the target's greedy after each of [t0, drafts...]. The BATCHED path runs all of them
		// through the resident stack in ONE pass over the cache (the speculative-decode speedup — one
		// submit, weights resident, vs K stepGreedy rounds); it declines (batched=false) for models
		// outside the dense path (PLE/MoE/recorded-ICB/shared-KV), where we step sequentially. Both
		// produce the identical greedys, so the accept/reject and the emitted stream are unchanged.
		greedys, batched, verr := target.verifyBatchedInto(verifyIDs, greedyIDs[:len(verifyIDs)])
		if verr != nil {
			return nil, verr
		}
		if batched {
			bonus = greedys[0] // greedys[i] = target's greedy AFTER the i-th verified token
			for d := 0; d < len(drafts); d++ {
				if drafts[d] != greedys[d] { // mismatch: target diverges here, drafts[d] rejected
					bonus = greedys[d]
					break
				}
				commitLen++
				accepted++
				bonus = greedys[d+1]
			}
		} else {
			expected, err := target.stepGreedy(t0)
			if err != nil {
				return nil, err
			}
			bonus = expected // if drafts is empty, the bonus IS the target's next greedy after t0
			for d := 0; d < len(drafts); d++ {
				if drafts[d] != expected { // mismatch: target diverges here, drafts[d] rejected
					bonus = expected
					break
				}
				// accepted: drafts[d] is exactly the target's greedy — commit it and step the target to
				// get the NEXT expected token (and a fresh bonus in case this was the last draft).
				commitLen++
				accepted++
				expected, err = target.stepGreedy(drafts[d])
				if err != nil {
					return nil, err
				}
				bonus = expected
			}
		}
		if mtpDiagForTest && res.Rounds <= 3 {
			nativeTraceLog(core.Sprintf("mtp-diag round %d: t0=%d drafts=%v greedys=%v batched=%v accepted=%d\n",
				res.Rounds, t0, drafts, greedys[:len(verifyIDs)], batched, accepted))
		}
		res.Accepted += accepted

		// roll the target cache back to just the committed run (t0 + accepted drafts); the rejected
		// suffix's K/V is overwritten by the bonus step below / the next round.
		target.pos = posBefore + commitLen
		if err := target.truncateSpeculativeKV(target.pos); err != nil {
			return nil, err
		}

		// keep the DRAFT cache aligned with the committed run too — otherwise it drifts a slot every
		// round (it proposes nDraft tokens but only `accepted` commit, and on a FULL accept it proposed
		// the last token without ever stepping it into its cache), the next round's proposals continue
		// from the wrong context, and acceptance collapses. The committed tokens [t0, accepted drafts]
		// are already resident in the draft cache from proposing at rows [draftPos0 .. draftPos0+accepted]
		// — except the full-accept case, where the final committed draft was proposed but not stepped, so
		// step it now to fill that row. Then roll the draft to the committed length so the bonus below
		// lands at the same position the target wrote it.
		// commit the accepted run, honouring maxNew/eos as plain Generate would.
		stop := false
		emittedCommitLen := 0
		for _, id := range verifyIDs[:commitLen] {
			if !emitMTPToken(res, yield, id) {
				stop = true
				emittedCommitLen++
				break
			}
			emittedCommitLen++
			if (eosID >= 0 && int(id) == eosID) || len(res.Tokens) >= maxNew {
				stop = true
				break
			}
		}
		if stop {
			target.pos = posBefore + emittedCommitLen
			if err := target.truncateSpeculativeKV(target.pos); err != nil {
				return nil, err
			}
			if err := draft.retainMTPCommittedBoundary(draftPos0, verifyIDs[:emittedCommitLen]); err != nil {
				return nil, err
			}
			if batched && emittedCommitLen > 0 {
				if err := target.rememberDenseBatchRetainedHidden(emittedCommitLen - 1); err != nil {
					return nil, err
				}
			}
			break
		}

		if accepted == len(drafts) && accepted > 0 {
			if _, err = draft.stepID(drafts[accepted-1]); err != nil {
				return nil, err
			}
		}
		draft.pos = draftPos0 + commitLen
		if err := draft.truncateSpeculativeKV(draft.pos); err != nil {
			return nil, err
		}

		// commit the bonus correction token (the target's greedy after the accepted run) and step it
		// on BOTH sessions, so each cache holds it and the next round's cursor is its hidden.
		yieldOK := emitMTPToken(res, yield, bonus)
		if bonusHidden, err = target.stepID(bonus); err != nil {
			return nil, err
		}
		if _, err = draft.stepID(bonus); err != nil {
			return nil, err
		}
		hidden = bonusHidden
		if !yieldOK {
			break
		}
		if (eosID >= 0 && int(bonus) == eosID) || len(res.Tokens) >= maxNew {
			break
		}
	}

	target.appendKnownResidentIDs(targetStartPos, promptIDs, res.Tokens)
	draft.appendKnownResidentIDs(draftStartPos, promptIDs, res.Tokens)
	return res, nil
}

// MTPDecodeSampled is the target-sampled sibling of MTPDecode: draft proposes
// continuations, but the target sampler decides every committed token in the
// same order as GenerateSampledEach. The draft sampler is separate so proposal
// draws cannot perturb the target RNG stream. stopTokens mirrors
// GenerateSampledEach; pass nil to disable stop-token early exit.
func MTPDecodeSampled(target, draft *ArchSession, promptIDs []int32, maxNew int, stopTokens []int32, targetSampler, draftSampler *model.Sampler, params model.SampleParams, k int) (*MTPResult, error) {
	return MTPDecodeSampledEach(target, draft, promptIDs, maxNew, stopTokens, targetSampler, draftSampler, params, k, nil)
}

// MTPDecodeSampledEach is MTPDecodeSampled with a streaming token sink. yield
// receives every target-sampled committed token in order; returning false stops
// after that token and returns the partial result.
func MTPDecodeSampledEach(target, draft *ArchSession, promptIDs []int32, maxNew int, stopTokens []int32, targetSampler, draftSampler *model.Sampler, params model.SampleParams, k int, yield func(int32) bool) (*MTPResult, error) {
	if target == nil || draft == nil {
		return nil, core.NewError("native.MTPDecodeSampled: nil target/draft session")
	}
	if targetSampler == nil {
		return nil, core.NewError("native.MTPDecodeSampled: nil target sampler")
	}
	if draftSampler == nil {
		return nil, core.NewError("native.MTPDecodeSampled: nil draft sampler")
	}
	if targetSampler == draftSampler {
		return nil, core.NewError("native.MTPDecodeSampled: target and draft samplers must be distinct")
	}
	if len(promptIDs) == 0 {
		return nil, core.NewError("native.MTPDecodeSampled: empty prompt")
	}
	if maxNew <= 0 {
		return nil, core.NewError("native.MTPDecodeSampled: maxNew must be > 0")
	}
	if k <= 0 {
		return nil, core.NewError("native.MTPDecodeSampled: k must be > 0")
	}
	if target.pos+len(promptIDs)+maxNew > target.maxLen {
		return nil, core.NewError("native.MTPDecodeSampled: target sequence would exceed maxLen cache rows")
	}
	if draft.pos+len(promptIDs)+maxNew > draft.maxLen {
		return nil, core.NewError("native.MTPDecodeSampled: draft sequence would exceed maxLen cache rows")
	}

	res := &MTPResult{Tokens: make([]int32, 0, maxNew)}
	targetStartPos := target.pos
	draftStartPos := draft.pos
	history := target.sampleHistoryScratchFor(params, maxNew)
	finalHistory := history
	defer func() { target.sampleHistory = finalHistory }()

	hidden, ok, err := target.prefillMTPPrompt(promptIDs, true)
	if err != nil {
		return nil, err
	}
	if !ok {
		for i, id := range promptIDs {
			h, err := target.stepID(id)
			if err != nil {
				return nil, err
			}
			if i == len(promptIDs)-1 {
				hidden = h
			}
		}
	}
	if _, ok, err = draft.prefillMTPPrompt(promptIDs, false); err != nil {
		return nil, err
	} else if !ok {
		for _, id := range promptIDs {
			if _, err := draft.stepID(id); err != nil {
				return nil, err
			}
		}
	}
	if hidden == nil {
		return nil, core.NewError("native.MTPDecodeSampled: prompt prefill produced no cursor hidden")
	}

	var verifyStack [16]int32
	verifyIDs := verifyStack[:1]
	if k+1 > len(verifyStack) {
		verifyIDs = make([]int32, 1, k+1)
	}

	for len(res.Tokens) < maxNew {
		res.Rounds++
		draftPos0 := draft.pos

		pickParams := target.mtpSamplePickParams(params, stopTokens, len(res.Tokens))
		t0, err := target.sampleMTPTokenFromHidden(hidden, targetSampler, pickParams, history)
		if err != nil {
			return nil, err
		}

		room := maxNew - len(res.Tokens)
		nDraft := max(min(k, room-1), 0)
		verifyIDs = verifyIDs[:1]
		verifyIDs[0] = t0
		drafts := verifyIDs[1:1]
		draftHistory := history
		if params.RepeatPenalty > 1 {
			draftHistory = draft.sampleHistoryScratchFor(params, maxNew)
			draftHistory = append(draftHistory, history...)
			draftHistory = append(draftHistory, t0)
		}
		seed := t0
		for d := 0; d < nDraft; d++ {
			dh, err := draft.stepID(seed)
			if err != nil {
				return nil, err
			}
			draftParams := draft.mtpSamplePickParams(params, stopTokens, len(res.Tokens)+1+d)
			nd, err := draft.sampleMTPTokenFromHidden(dh, draftSampler, draftParams, draftHistory)
			if err != nil {
				return nil, err
			}
			drafts = append(drafts, nd)
			if params.RepeatPenalty > 1 {
				draftHistory = append(draftHistory, nd)
			}
			seed = nd
		}
		verifyIDs = verifyIDs[:1+len(drafts)]
		res.Drafted += len(drafts)

		posBefore := target.pos
		commitLen := 0
		accepted := 0
		bonusOK := false
		stopped := false
		var bonus int32
		batchedHiddens, batched, err := target.verifyBatchedHiddens(verifyIDs)
		if err != nil {
			return nil, err
		}
		if batched {
			if len(batchedHiddens) != len(verifyIDs) {
				return nil, core.NewError("native.MTPDecodeSampled: sampled batched verify hidden count mismatch")
			}
			hidden = batchedHiddens[0]
			commitLen = 1
			if !emitMTPToken(res, yield, t0) {
				stopped = true
			}
			if params.RepeatPenalty > 1 {
				history = append(history, t0)
				finalHistory = history
			}
			if nativeTokenInSet(t0, stopTokens) || len(res.Tokens) >= maxNew {
				stopped = true
			}
			if !stopped {
				expectedParams := target.mtpSamplePickParams(params, stopTokens, len(res.Tokens))
				expected, sampleErr := target.sampleMTPTokenFromDenseBatchRowOrHidden(0, hidden, targetSampler, expectedParams, history)
				if sampleErr != nil {
					return nil, sampleErr
				}
				for d, draftID := range drafts {
					if draftID != expected {
						bonus = expected
						bonusOK = true
						break
					}
					commitLen++
					accepted++
					hidden = batchedHiddens[d+1]
					if !emitMTPToken(res, yield, draftID) {
						stopped = true
						break
					}
					if params.RepeatPenalty > 1 {
						history = append(history, draftID)
						finalHistory = history
					}
					if nativeTokenInSet(draftID, stopTokens) || len(res.Tokens) >= maxNew {
						stopped = true
						break
					}
					expectedParams = target.mtpSamplePickParams(params, stopTokens, len(res.Tokens))
					expected, sampleErr = target.sampleMTPTokenFromDenseBatchRowOrHidden(d+1, hidden, targetSampler, expectedParams, history)
					if sampleErr != nil {
						return nil, sampleErr
					}
				}
				if !stopped && !bonusOK {
					bonus = expected
					bonusOK = true
				}
			}
		} else {
			hidden, err = target.stepID(t0)
			if err != nil {
				return nil, err
			}
			commitLen = 1
			if !emitMTPToken(res, yield, t0) {
				stopped = true
			}
			if params.RepeatPenalty > 1 {
				history = append(history, t0)
				finalHistory = history
			}
			if nativeTokenInSet(t0, stopTokens) || len(res.Tokens) >= maxNew {
				stopped = true
			}

			if !stopped {
				expectedParams := target.mtpSamplePickParams(params, stopTokens, len(res.Tokens))
				expected, sampleErr := target.sampleMTPTokenFromHidden(hidden, targetSampler, expectedParams, history)
				if sampleErr != nil {
					return nil, sampleErr
				}
				for _, draftID := range drafts {
					if draftID != expected {
						bonus = expected
						bonusOK = true
						break
					}
					hidden, err = target.stepID(draftID)
					if err != nil {
						return nil, err
					}
					commitLen++
					accepted++
					if !emitMTPToken(res, yield, draftID) {
						stopped = true
						break
					}
					if params.RepeatPenalty > 1 {
						history = append(history, draftID)
						finalHistory = history
					}
					if nativeTokenInSet(draftID, stopTokens) || len(res.Tokens) >= maxNew {
						stopped = true
						break
					}
					expectedParams = target.mtpSamplePickParams(params, stopTokens, len(res.Tokens))
					expected, sampleErr = target.sampleMTPTokenFromHidden(hidden, targetSampler, expectedParams, history)
					if sampleErr != nil {
						return nil, sampleErr
					}
				}
				if !stopped && !bonusOK {
					bonus = expected
					bonusOK = true
				}
			}
		}
		res.Accepted += accepted
		target.pos = posBefore + commitLen
		if err := target.truncateSpeculativeKV(target.pos); err != nil {
			return nil, err
		}

		if stopped {
			if batched {
				if err := target.rememberDenseBatchRetainedHidden(commitLen - 1); err != nil {
					return nil, err
				}
			}
			if err := draft.retainMTPCommittedBoundary(draftPos0, verifyIDs[:commitLen]); err != nil {
				return nil, err
			}
			break
		}
		if len(drafts) == 0 {
			if _, err = draft.stepID(t0); err != nil {
				return nil, err
			}
		} else if accepted == len(drafts) && accepted > 0 {
			if _, err = draft.stepID(drafts[accepted-1]); err != nil {
				return nil, err
			}
		}
		draft.pos = draftPos0 + commitLen
		if err := draft.truncateSpeculativeKV(draft.pos); err != nil {
			return nil, err
		}
		if !bonusOK {
			return nil, core.NewError("native.MTPDecodeSampled: sampled verify produced no bonus token")
		}

		yieldOK := emitMTPToken(res, yield, bonus)
		if params.RepeatPenalty > 1 {
			history = append(history, bonus)
			finalHistory = history
		}
		if hidden, err = target.stepID(bonus); err != nil {
			return nil, err
		}
		if _, err = draft.stepID(bonus); err != nil {
			return nil, err
		}
		if !yieldOK {
			break
		}
		if nativeTokenInSet(bonus, stopTokens) || len(res.Tokens) >= maxNew {
			break
		}
	}

	target.appendKnownResidentIDs(targetStartPos, promptIDs, res.Tokens)
	draft.appendKnownResidentIDs(draftStartPos, promptIDs, res.Tokens)
	return res, nil
}

func emitMTPToken(res *MTPResult, yield func(int32) bool, id int32) bool {
	res.Tokens = append(res.Tokens, id)
	return yield == nil || yield(id)
}

func (s *ArchSession) mtpSamplePickParams(params model.SampleParams, stopTokens []int32, generated int) model.SampleParams {
	pick := params
	if params.MinTokensBeforeStop > 0 && generated < params.MinTokensBeforeStop {
		pick.SuppressTokens = s.suppressionTokensScratch(params.SuppressTokens, stopTokens)
	}
	return pick
}

func (s *ArchSession) sampleMTPTokenFromHidden(hidden []byte, sampler *model.Sampler, params model.SampleParams, history []int32) (int32, error) {
	var (
		token int32
		err   error
	)
	withAutoreleasePool(func() {
		token, err = s.sampleMTPTokenFromHiddenInPool(hidden, sampler, params, history)
	})
	return token, err
}

func (s *ArchSession) sampleMTPTokenFromHiddenInPool(hidden []byte, sampler *model.Sampler, params model.SampleParams, history []int32) (int32, error) {
	if sampledGreedyParamsEligible(params) {
		return s.headGreedyOrLogits(hidden, params.SuppressTokens, nil, nil, false)
	}
	if sampledTopOneGreedyParamsEligible(params, history) {
		sampler.Draw()
		return s.headGreedyOrLogits(hidden, params.SuppressTokens, nil, nil, false)
	}
	if s.sampleTopKTokenParamsEligible(params) {
		token, ok, err := s.sampleTopKTokenFromHiddenInPool(hidden, params, sampler.Draw(), history)
		if err != nil || ok {
			return token, err
		}
		return 0, core.NewError("native.MTPDecodeSampled: TopK token path declined after eligibility check")
	}
	if s.sampleLogitsTokenParamsEligible(params) && !sampleLogitsTokenCPUPreferred(params, s.arch.Vocab) {
		token, ok, err := s.sampleLogitsTokenFromHiddenInPool(hidden, params, sampler.Draw(), history)
		if err != nil || ok {
			return token, err
		}
		return 0, core.NewError("native.MTPDecodeSampled: logits token path declined after eligibility check")
	}
	if candidateLogits, candidateIDs, ok, err := s.sampleTopKCandidatesFromHiddenWithHistoryInPool(hidden, params, history); err != nil {
		return 0, err
	} else if ok {
		return sampler.SampleCandidates(candidateLogits, candidateIDs, params)
	}
	logits, err := s.headLogitsScratch(hidden, false)
	if err != nil {
		return 0, err
	}
	pickLogits := logits
	if params.RepeatPenalty > 1 {
		pickLogits, err = s.repeatPenaltyLogitsScratch(logits, s.arch.Vocab, history, params.RepeatPenalty)
		if err != nil {
			return 0, err
		}
	}
	if sampleLogitsTokenCPUPreferred(params, s.arch.Vocab) {
		return sampleSmallVocabBF16(pickLogits, s.arch.Vocab, sampler, params)
	}
	return s.sampleVocabBF16(pickLogits, s.arch.Vocab, sampler, params)
}

func (s *ArchSession) sampleMTPTokenFromDenseBatchRowOrHidden(row int, hidden []byte, sampler *model.Sampler, params model.SampleParams, history []int32) (int32, error) {
	token, ok, err := s.sampleMTPTokenFromDenseBatchRow(row, sampler, params, history)
	if err != nil || ok {
		return token, err
	}
	return s.sampleMTPTokenFromHidden(hidden, sampler, params, history)
}

func (s *ArchSession) sampleMTPTokenFromDenseBatchRow(row int, sampler *model.Sampler, params model.SampleParams, history []int32) (token int32, ok bool, err error) {
	withAutoreleasePool(func() {
		token, ok, err = s.sampleMTPTokenFromDenseBatchRowInPool(row, sampler, params, history)
	})
	return token, ok, err
}

func (s *ArchSession) sampleMTPTokenFromDenseBatchRowInPool(row int, sampler *model.Sampler, params model.SampleParams, history []int32) (int32, bool, error) {
	if sampler == nil {
		return 0, true, core.NewError("native.MTPDecodeSampled: nil sampler")
	}
	rowBuf, rowOff, ok, err := s.denseBatchHiddenRowBuffer(row)
	if err != nil || !ok {
		return 0, ok, err
	}
	if sampledGreedyParamsEligible(params) {
		if !s.canUseDirectHeadGreedy() {
			return 0, false, nil
		}
		token, ok, err := s.headEnc.greedyBufferAtInPool(rowBuf, rowOff, params.SuppressTokens)
		if err != nil || ok {
			return token, ok, err
		}
		return 0, true, core.NewError("native.MTPDecodeSampled: dense-row greedy path declined after eligibility check")
	}
	if sampledTopOneGreedyParamsEligible(params, history) {
		if !s.canUseDirectHeadGreedy() {
			return 0, false, nil
		}
		sampler.Draw()
		token, ok, err := s.headEnc.greedyBufferAtInPool(rowBuf, rowOff, params.SuppressTokens)
		if err != nil || ok {
			return token, ok, err
		}
		return 0, true, core.NewError("native.MTPDecodeSampled: dense-row TopK=1 greedy path declined after eligibility check")
	}
	if s.sampleTopKTokenParamsEligible(params) {
		token, ok, err := s.headEnc.sampleTopKTokenBufferAtInPool(rowBuf, rowOff, params, sampler.Draw(), history)
		if err != nil || ok {
			return token, ok, err
		}
		return 0, true, core.NewError("native.MTPDecodeSampled: dense-row TopK token path declined after eligibility check")
	}
	if s.sampleLogitsTokenParamsEligible(params) && !sampleLogitsTokenCPUPreferred(params, s.arch.Vocab) {
		token, ok, err := s.headEnc.sampleLogitsTokenBufferAtInPool(rowBuf, rowOff, params, sampler.Draw(), history)
		if err != nil || ok {
			return token, ok, err
		}
		return 0, true, core.NewError("native.MTPDecodeSampled: dense-row logits token path declined after eligibility check")
	}
	return 0, false, nil
}

func (s *ArchSession) retainMTPCommittedBoundary(start int, ids []int32) error {
	if s == nil {
		return core.NewError("native.MTPDecode: nil draft session")
	}
	if start < 0 || start+len(ids) > s.maxLen {
		return core.NewError("native.MTPDecode: committed draft boundary would exceed maxLen cache rows")
	}
	s.pos = start
	if err := s.truncateSpeculativeKV(s.pos); err != nil {
		return err
	}
	for _, id := range ids {
		if _, err := s.stepID(id); err != nil {
			return err
		}
	}
	return nil
}

func (s *ArchSession) prefillMTPPrompt(ids []int32, readLast bool) ([]byte, bool, error) {
	if len(ids) == 0 {
		return nil, false, core.NewError("native.MTPDecode: empty prompt")
	}
	if s.perLayerInput != nil || s.pos+len(ids) > s.maxLen {
		return nil, false, nil
	}
	batchIDs := ids
	if readLast {
		batchIDs = ids
	}
	if len(batchIDs) == 0 {
		return nil, false, nil
	}
	var embStack [16][]byte
	var embs [][]byte
	if len(batchIDs) <= len(embStack) {
		embs = embStack[:len(batchIDs)]
	} else {
		embs = make([][]byte, len(batchIDs))
	}
	if s.canUseEmbedScratch() {
		rowBytes := s.arch.Hidden * bf16Size
		need := len(batchIDs) * rowBytes
		if cap(s.embedScratch) < need {
			s.embedScratch = make([]byte, need)
		} else {
			s.embedScratch = s.embedScratch[:need]
		}
		for i, id := range batchIDs {
			dst := s.embedScratch[i*rowBytes : (i+1)*rowBytes]
			emb, err := s.embedInto(dst, id)
			if err != nil {
				return nil, false, err
			}
			if len(emb) != rowBytes {
				return nil, false, core.NewError("native.MTPDecode: embedInto returned wrong hidden size")
			}
			embs[i] = emb
		}
	} else {
		for i, id := range batchIDs {
			emb, err := s.embed(id)
			if err != nil {
				return nil, false, err
			}
			embs[i] = emb
		}
	}
	var (
		hidden  []byte
		hiddens [][]byte
		ok      bool
		err     error
	)
	if readLast {
		dst := s.sampleHidden
		retained := false
		if pinned, pinnedOK := s.ensureRetainedHiddenPinned(s.arch.Hidden * bf16Size); pinnedOK {
			s.resetRetainedLogits()
			dst = pinned.bytes[:s.arch.Hidden*bf16Size]
			retained = true
		}
		withAutoreleasePool(func() {
			hidden, ok, err = s.state.stepTokensBatchedDenseLastInto(embs, s.pos, dst)
		})
		if err != nil || !ok {
			return nil, ok, err
		}
		if retained {
			s.sampleHidden = nil
			s.retainedHidden = hidden
		} else {
			s.sampleHidden = hidden
		}
		s.pos += len(batchIDs)
		return hidden, true, nil
	}
	withAutoreleasePool(func() {
		hiddens, ok, err = s.state.stepTokensBatchedDenseResult(embs, s.pos, false, false, nil, nil)
	})
	if err != nil || !ok {
		return nil, ok, err
	}
	s.pos += len(batchIDs)
	if len(hiddens) != 0 {
		return nil, false, core.NewError("native.MTPDecode: dense prompt prefill returned incomplete hiddens")
	}
	return nil, true, nil
}

// stepID embeds token id and steps it through the session's resident cache at the current position,
// advancing pos. It retains the returned hidden in the session's no-copy boundary buffer when possible,
// so the following greedy/head path can bind it directly. PLE models thread the id correctly.
func (s *ArchSession) stepID(id int32) ([]byte, error) {
	var (
		hidden []byte
		err    error
	)
	withAutoreleasePool(func() {
		hidden, err = s.stepIDRetainedInPool(id)
	})
	return hidden, err
}

// greedyOf returns the greedy argmax id plain Generate would emit at this hidden.
func (s *ArchSession) greedyOf(hidden []byte) (int32, error) {
	return s.greedyFromHiddenInPool(hidden, nil)
}

// stepGreedy steps token id on the session cache and returns the greedy argmax of the resulting
// hidden — the target's expected NEXT token after id. It is stepID followed by greedyOf, the verify
// inner loop's unit of work.
func (s *ArchSession) stepGreedy(id int32) (int32, error) {
	h, err := s.stepID(id)
	if err != nil {
		return 0, err
	}
	return s.greedyOf(h)
}
