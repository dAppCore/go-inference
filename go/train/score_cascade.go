// SPDX-Licence-Identifier: EUPL-1.2

// The score cascade (#50): a scorer rides the SFT eval loop (and the SSD
// sampling phase), so the checkpoint is not a guess — it's from analysis.
// Each pass scores every (prompt, output) pair AT GENERATION TIME (the score
// is part of the data point, never recomputed — data-is-the-return) and
// appends the full vector to a JSONL sidecar; saved checkpoints are annotated
// with the windowed composite, and the run reports the best checkpoint by the
// cascade read: the highest windowed mean — composite climbing AND holding
// across the window is the "vectors tighten without losing range" signature,
// read simply.
//
// Ported from go-mlx/go/train/score_cascade.go with ONE deliberate change: the
// concrete lem-scorer (go-mlx's mlx/pkg/score — an 11-file phonetics/cmudict
// subsystem) has no go-inference home, so this package takes the scorer as an
// injected [ScoreFunc] hook rather than importing one. The cascade MACHINERY
// (windowed composite, sidecar, best-checkpoint read) is engine- and
// scorer-neutral; a driver that has a scorer supplies it, and a driver that
// does not leaves scoring off (the hook is nil → the cascade is a no-op).

package train

import (
	"time"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

// ScoreRecord is one scored generation — the immortalised vector. It is the
// engine-neutral subset of go-mlx's SFTScoreRecord: the phonetics ImprintScores
// pointer is dropped (it belonged to the concrete lem-scorer), leaving the
// headline dimensions every scorer can report as plain scalars.
type ScoreRecord struct {
	Step      int     `json:"step"`
	Prompt    string  `json:"prompt"`
	Text      string  `json:"text"`
	LEK       float64 `json:"lek"`
	Tier      int     `json:"sycophancy_tier"`
	Hostility float64 `json:"hostility"`
	Echo      float64 `json:"echo"`
	At        int64   `json:"at_unix"`
}

// ScoreFunc scores one (prompt, response) pair, returning the dimensions the
// cascade reads. The caller fills LEK/Tier/Hostility/Echo; Step and At are
// stamped by the cascade. A nil ScoreFunc disables scoring.
//
//	cascade := newScoreCascade(sidecar, 3, func(prompt, text string) train.ScoreRecord {
//	    return train.ScoreRecord{LEK: myScorer.Score(prompt, text)}
//	})
type ScoreFunc func(prompt, text string) ScoreRecord

// scoreCascade accumulates eval scores and answers the cascade read.
type scoreCascade struct {
	scoreFn     ScoreFunc
	sidecarPath string
	window      int
	records     []ScoreRecord
	// perStep is the mean LEK composite across the probes of one eval
	// pass, in eval order — the cascade's raw curve.
	perStep []struct {
		Step int
		Mean float64
	}
}

// scoreWindowDefault is the stability window: the cascade judges a step by the
// mean composite across this many trailing eval passes, so one lucky
// generation never crowns a checkpoint.
const scoreWindowDefault = 3

func newScoreCascade(sidecarPath string, window int, scoreFn ScoreFunc) *scoreCascade {
	if window <= 0 {
		window = scoreWindowDefault
	}
	return &scoreCascade{scoreFn: scoreFn, sidecarPath: sidecarPath, window: window}
}

// recordPass scores one eval pass (all probes at one step) and appends the
// vectors to the sidecar. A nil scoreFn makes this a no-op.
func (c *scoreCascade) recordPass(step int, evals []SFTEvalResult) {
	if c == nil || c.scoreFn == nil || len(evals) == 0 {
		return
	}
	sum := 0.0
	count := 0
	now := time.Now().Unix()
	for _, ev := range evals {
		if ev.Step != step {
			continue
		}
		rec := c.scoreFn(ev.Prompt, ev.Text)
		rec.Step = step
		rec.Prompt = ev.Prompt
		rec.Text = ev.Text
		rec.At = now
		c.records = append(c.records, rec)
		sum += rec.LEK
		count++
	}
	if count == 0 {
		return
	}
	c.perStep = append(c.perStep, struct {
		Step int
		Mean float64
	}{Step: step, Mean: sum / float64(count)})
	c.appendSidecar(step)
}

// appendSidecar writes this pass's records as JSONL. Append-only and
// best-effort: a sidecar write failure never interrupts training.
func (c *scoreCascade) appendSidecar(step int) {
	if c.sidecarPath == "" {
		return
	}
	// Presize the accumulator from the raw field sizes so the per-row appends
	// don't regrow it geometrically (the -alloc_space FLAT profile put the
	// whole cost on the append line): each row's JSON is the two text fields
	// plus a fixed scaffold of keys/quoting/newline and the six numeric fields
	// (~128 bytes covers the {"step":,"prompt":,"text":,"lek":,
	// "sycophancy_tier":,"hostility":,"echo":,"at_unix":} frame and its digits).
	estimate := 0
	for _, rec := range c.records {
		if rec.Step != step {
			continue
		}
		estimate += len(rec.Prompt) + len(rec.Text) + 128
	}
	out := make([]byte, 0, estimate)
	for _, rec := range c.records {
		if rec.Step != step {
			continue
		}
		encoded := core.JSONMarshal(rec)
		if !encoded.OK {
			continue
		}
		out = append(out, encoded.Value.([]byte)...)
		out = append(out, '\n')
	}
	if len(out) == 0 {
		return
	}
	w, err := coreio.Local.Append(c.sidecarPath)
	if err != nil {
		return
	}
	defer func() { _ = w.Close() }()
	_, _ = w.Write(out)
}

// compositeAt answers the cascade read for a step: the mean composite over the
// trailing window of eval passes at or before it. Steps before any eval read
// as 0.
func (c *scoreCascade) compositeAt(step int) float64 {
	if c == nil || len(c.perStep) == 0 {
		return 0
	}
	end := -1
	for i, p := range c.perStep {
		if p.Step <= step {
			end = i
		}
	}
	if end < 0 {
		return 0
	}
	start := max(end-c.window+1, 0)
	sum := 0.0
	for i := start; i <= end; i++ {
		sum += c.perStep[i].Mean
	}
	return sum / float64(end-start+1)
}

// best returns the eval step with the highest windowed composite and that
// composite. Ties crown the LATER step — equal windowed quality with more
// training behind it is the stabler artefact. False when nothing was scored.
func (c *scoreCascade) best() (int, float64, bool) {
	if c == nil || len(c.perStep) == 0 {
		return 0, 0, false
	}
	bestStep, bestMean := 0, -1.0
	for _, p := range c.perStep {
		if w := c.compositeAt(p.Step); w >= bestMean {
			bestMean = w
			bestStep = p.Step
		}
	}
	return bestStep, bestMean, true
}
