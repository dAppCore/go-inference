// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"dappco.re/go/inference/eval/score/lek"
	"dappco.re/go/inference/train"
)

// lekScoreFunc returns the score-cascade hook backed by the LEK phonetics/
// heuristic scorer (dappco.re/go/inference/eval/score/lek). It is the
// go-inference driver's scorer: the train library stays scorer-neutral (a nil
// hook disables the cascade), and cmd/lem — the binary that owns the model and
// picks the scorer — injects this concrete one into the sft and ssd configs.
//
// Each (prompt, text) is scored as a PAIR (lek.ScorePair) so the cross-text
// Echo dimension is available. The mapping mirrors go-mlx's original cascade
// wiring: the response side supplies LEK / sycophancy tier / hostility, and the
// differential supplies Echo. Step, Prompt, Text and At are stamped by the
// cascade, so this adapter leaves them zero.
//
//	cfg.Score = lekScoreFunc() // arms the cascade with the real LEK scorer
func lekScoreFunc() train.ScoreFunc {
	return func(prompt, text string) train.ScoreRecord {
		pair := lek.ScorePair(prompt, text)
		rec := train.ScoreRecord{}
		if r := pair.Response; r.LEK != nil {
			rec.LEK = r.LEK.LEKScore
		}
		if r := pair.Response; r.Sycophancy != nil {
			rec.Tier = r.Sycophancy.Tier
		}
		if r := pair.Response; r.Hostility != nil {
			rec.Hostility = r.Hostility.Score
		}
		if pair.Differential != nil {
			rec.Echo = pair.Differential.Echo
		}
		return rec
	}
}
