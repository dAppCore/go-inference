// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/model"
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
