// SPDX-Licence-Identifier: EUPL-1.2

// Capture-first (#97): the model's response IS the data point, so the raw
// return text is captured at the moment it exists — independent of any
// scoring. Scoring after the fact over captured text is archaeology and
// always legitimate; a missed capture is a data point that never existed.
// Rows are JSONL, append-only, shaped to be scoreable later. Ported from
// go-mlx/go/train/capture.go verbatim — capture is engine-free (plain text
// + a JSONL sidecar), so it belongs in the shared engine-neutral train
// package unchanged.

package train

import (
	"time"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

// CaptureRow is one captured generation: the bare prompt, the raw return
// text, and when it was born. Scores are deliberately absent — they can
// be attached after the fact; the capture cannot.
type CaptureRow struct {
	Step   int    `json:"step"`
	Prompt string `json:"prompt"`
	Text   string `json:"text"`
	At     int64  `json:"at_unix"`
}

// appendCaptureRows writes one pass's generations to the capture sidecar.
// Best-effort like the score sidecar: a disk hiccup never interrupts a
// training run. Returns how many rows landed, for honesty counters.
func appendCaptureRows(path string, evals []SFTEvalResult) int {
	if path == "" || len(evals) == 0 {
		return 0
	}
	now := time.Now().Unix()
	// Presize the accumulator from the raw field sizes so the per-row appends
	// don't regrow it geometrically: each row's JSON is the two text fields
	// plus a fixed scaffold of keys/quoting/newline (~64 bytes covers the
	// step/at digits and the {"step":,"prompt":,"text":,"at_unix":} frame).
	estimate := 0
	for _, ev := range evals {
		estimate += len(ev.Prompt) + len(ev.Text) + 64
	}
	out := make([]byte, 0, estimate)
	rows := 0
	for _, ev := range evals {
		encoded := core.JSONMarshal(CaptureRow{Step: ev.Step, Prompt: ev.Prompt, Text: ev.Text, At: now})
		if !encoded.OK {
			continue
		}
		out = append(out, encoded.Bytes()...)
		out = append(out, '\n')
		rows++
	}
	if len(out) == 0 {
		return 0
	}
	w, err := coreio.Local.Append(path)
	if err != nil {
		return 0
	}
	defer func() { _ = w.Close() }()
	if _, err := w.Write(out); err != nil {
		return 0
	}
	return rows
}
