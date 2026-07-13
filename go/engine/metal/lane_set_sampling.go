// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/model"
	"github.com/tmc/apple/metal"
)

// lane_set_sampling.go — per-lane sampler state for the laneSet (the #35
// serve-integration sampling rung). Phase 1 of laneSet.Step produces one token
// per lane from its current hidden; a lane whose LaneSpec selects a non-greedy
// discipline owns a model.Sampler (its OWN RNG stream, seeded from
// LaneSpec.SampleSeed) plus repeat-penalty history, and samples through the
// SAME route ladder the classic sampled generate runs — so a lane's token
// stream is token-identical to GenerateSampledEach on an identical session
// (TestLaneSetSampledMatchesGenerateSampledEach pins it).

// laneSampled reports whether cfg selects a non-greedy decode discipline,
// mirroring the classic serve path's pick EXACTLY (engine SessionHandle
// generate: sampled iff temperature, min-p, or repeat-penalty engage —
// TopK/TopP alone with temperature 0 stay greedy there, so they stay greedy
// here; CB must never serve a different discipline than the plain path would).
func laneSampled(cfg inference.SamplerConfig) bool {
	return cfg.Temperature > 0 || cfg.MinP > 0 || cfg.RepeatPenalty > 1
}

// laneSampleParams maps the neutral SamplerConfig onto the model sampler
// params, the same field-for-field mapping the classic serve path applies.
func laneSampleParams(cfg inference.SamplerConfig) model.SampleParams {
	return model.SampleParams{
		Temperature:   cfg.Temperature,
		TopK:          cfg.TopK,
		TopP:          cfg.TopP,
		MinP:          cfg.MinP,
		RepeatPenalty: cfg.RepeatPenalty,
	}
}

// sampledNextFromHiddenInPool picks ONE sampled token from hidden — the
// carry-less single-step twin of generateSampledFromHiddenInPool's route
// ladder, in the same route order with the same one-Draw-per-token RNG
// consumption, so a step-at-a-time caller (the laneSet's phase 1) draws the
// identical sequence the classic loop draws. history is caller-owned
// (per-lane) and consulted only when RepeatPenalty engages; the caller
// appends the picked token itself.
func (s *ArchSession) sampledNextFromHiddenInPool(hidden []byte, sampler *model.Sampler, params model.SampleParams, history []int32) (int32, error) {
	if sampledGreedyParamsEligible(params) {
		return s.headGreedyOrLogits(hidden, params.SuppressTokens, nil, nil, false)
	}
	if sampledTopOneGreedyParamsEligible(params, history) {
		sampler.Draw()
		return s.headGreedyOrLogits(hidden, params.SuppressTokens, nil, nil, false)
	}
	if s.sampleTopKTokenParamsEligible(params) {
		draw := sampler.Draw()
		next, ok, err := s.sampleTopKTokenFromHiddenInPool(hidden, params, draw, history)
		if err != nil {
			return 0, err
		}
		if !ok {
			return 0, core.NewError("native.ArchSession.sampledNextFromHiddenInPool: TopK token path declined after eligibility check")
		}
		return next, nil
	}
	if s.sampleLogitsTokenParamsEligible(params) && !sampleLogitsTokenCPUPreferred(params, s.arch.Vocab) {
		draw := sampler.Draw()
		next, ok, err := s.sampleLogitsTokenFromHiddenInPool(hidden, params, draw, history)
		if err != nil {
			return 0, err
		}
		if !ok {
			return 0, core.NewError("native.ArchSession.sampledNextFromHiddenInPool: logits token path declined after eligibility check")
		}
		return next, nil
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

// phase1SampledTopKRows runs Phase 1 for an all-sampled lane set whose every
// lane the ladder would route to the GPU topK-token path, as ONE batched
// submission: each lane's topK sample chain (norm + head + k-select + pick)
// encodes into the same command buffer against the lane's own hidden and
// scratch, committed and waited once — the per-lane path paid K separate
// commit+wait round-trips that serialised in the drive loop (measured 0.77×
// greedy at K=4/8 on the 26B probe). Per-lane RNG order is preserved: one
// Draw per lane in lane order, exactly the ladder's consumption, so tokens
// are identical to the per-lane path. ok=false — fewer than two lanes, a
// greedy lane, or a lane another route would claim first — keeps the
// per-lane ladder, byte-identically, decided BEFORE any lane's RNG draws (a
// post-draw abandonment would desync the samplers from the per-lane path, so
// post-draw failures are ERRORS, never silent fallbacks).
func (ls *laneSet) phase1SampledTopKRows(lanes []*decodeLane) ([]int32, bool, error) {
	if len(lanes) < 2 {
		return nil, false, nil
	}
	for _, lane := range lanes {
		if lane.sampler == nil {
			return nil, false, nil
		}
		p := lane.sampleParams
		if sampledGreedyParamsEligible(p) || sampledTopOneGreedyParamsEligible(p, lane.sampleHistory) {
			return nil, false, nil // the ladder takes an earlier route for this lane
		}
		if !lane.sess.sampleTopKTokenParamsEligible(p) {
			return nil, false, nil
		}
	}
	he := lanes[0].sess.headEnc
	if he == nil {
		return nil, false, nil
	}
	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	scratches := make([]*headTopKScratch, 0, len(lanes))
	hiddenScratches := make([]*headHiddenScratch, 0, len(lanes))
	release := func() {
		for _, sc := range scratches {
			he.putTopKScratch(sc)
		}
		for _, hs := range hiddenScratches {
			he.putHiddenScratch(hs)
		}
	}
	for _, lane := range lanes {
		hiddenBuf := lane.sess.retainedHiddenBufferFor(lane.hidden)
		if hiddenBuf == nil {
			hs, buf, herr := he.hiddenBuffer(lane.hidden)
			if herr != nil {
				endEncodingFast(enc)
				release()
				return nil, false, herr
			}
			hiddenScratches = append(hiddenScratches, hs)
			hiddenBuf = buf
		}
		draw := lane.sampler.Draw()
		sc, ok, err := he.encodeTopKSampleAtFast(enc, hiddenBuf, 0, lane.sampleParams, draw, lane.sampleHistory)
		if sc != nil {
			scratches = append(scratches, sc)
		}
		if err != nil || !ok {
			endEncodingFast(enc)
			release()
			if err == nil {
				err = core.NewError("native.laneSet.phase1SampledTopKRows: topK encode declined after eligibility (post-draw — cannot fall back without desyncing RNG)")
			}
			return nil, false, err
		}
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)
	out := make([]int32, len(lanes))
	for i, sc := range scratches {
		tok := sc.token()
		if tok < 0 || int(tok) >= ls.vocab {
			release()
			return nil, false, core.NewError("native.laneSet.phase1SampledTopKRows: sampled invalid token")
		}
		out[i] = tok
	}
	release()
	ls.sampledRowsCount++
	return out, true, nil
}

// phase1SampledLogitsRows runs Phase 1 for an all-sampled lane set whose
// every lane the ladder would route to the FINAL fallback — full-vocab
// logits + host top-k/sample (the big-vocab serve shape: every GPU-select
// route declines via hostTopKSamplePreferred) — as ONE batched submission:
// each lane's logits chain (norm + lm_head + softcap) encodes into the same
// command buffer with the per-lane path's IDENTICAL kernels, waited once;
// the host tails (repeat penalty, top-k scan, draw) then run per lane in
// lane order, exactly as the ladder runs them. Decided BEFORE any RNG use;
// tokens are identical to the per-lane path because the logits bytes are.
func (ls *laneSet) phase1SampledLogitsRows(lanes []*decodeLane) ([]int32, bool, error) {
	if len(lanes) < 2 {
		return nil, false, nil
	}
	for _, lane := range lanes {
		if lane.sampler == nil {
			return nil, false, nil
		}
		p := lane.sampleParams
		sess := lane.sess
		if sampledGreedyParamsEligible(p) || sampledTopOneGreedyParamsEligible(p, lane.sampleHistory) {
			return nil, false, nil
		}
		if sess.sampleTopKTokenParamsEligible(p) {
			return nil, false, nil
		}
		if sess.sampleLogitsTokenParamsEligible(p) && !sampleLogitsTokenCPUPreferred(p, sess.arch.Vocab) {
			return nil, false, nil
		}
		if sess.sampleTopKParamsEligible(p) {
			return nil, false, nil // the candidates route claims this lane
		}
		if sess.headEnc == nil {
			return nil, false, nil
		}
	}
	he := lanes[0].sess.headEnc
	hiddenBufs := make([]metal.MTLBuffer, len(lanes))
	outs := make([][]byte, len(lanes))
	var hiddenScratches []*headHiddenScratch
	defer func() {
		for _, hs := range hiddenScratches {
			he.putHiddenScratch(hs)
		}
	}()
	for i, lane := range lanes {
		sess := lane.sess
		buf := sess.retainedHiddenBufferFor(lane.hidden)
		if buf == nil {
			hs, b, err := he.hiddenBuffer(lane.hidden)
			if err != nil {
				return nil, false, err
			}
			hiddenScratches = append(hiddenScratches, hs)
			buf = b
		}
		hiddenBufs[i] = buf
		// Mirror headLogitsScratch's session-owned logits staging so the host
		// tail reads the same backing the per-lane route would.
		if cap(sess.sampleHeadLogits) < sess.arch.Vocab*bf16Size {
			sess.sampleHeadLogits = make([]byte, sess.arch.Vocab*bf16Size)
		}
		sess.sampleHeadLogits = sess.sampleHeadLogits[:sess.arch.Vocab*bf16Size]
		outs[i] = sess.sampleHeadLogits
	}
	if err := he.encodeLogitsRowsInPool(hiddenBufs, false, outs); err != nil {
		return nil, false, err
	}
	out := make([]int32, len(lanes))
	for i, lane := range lanes {
		sess := lane.sess
		p := lane.sampleParams
		pickLogits := outs[i]
		if p.RepeatPenalty > 1 {
			var err error
			pickLogits, err = sess.repeatPenaltyLogitsScratch(outs[i], sess.arch.Vocab, lane.sampleHistory, p.RepeatPenalty)
			if err != nil {
				return nil, false, err
			}
		}
		var tok int32
		var err error
		if sampleLogitsTokenCPUPreferred(p, sess.arch.Vocab) {
			tok, err = sampleSmallVocabBF16(pickLogits, sess.arch.Vocab, lane.sampler, p)
		} else {
			tok, err = sess.sampleVocabBF16(pickLogits, sess.arch.Vocab, lane.sampler, p)
		}
		if err != nil {
			return nil, false, err
		}
		out[i] = tok
	}
	ls.sampledRowsCount++
	return out, true, nil
}
