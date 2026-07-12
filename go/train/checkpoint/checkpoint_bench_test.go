// SPDX-Licence-Identifier: EUPL-1.2

// Benchmark for the per-checkpoint dirname format. FormatStepDir builds the
// "step-NNNNNN" directory name every time a checkpoint lands, and the source
// deliberately uses strconv.AppendInt with in-place zero-padding to avoid
// fmt's reflection path on this hot loop — this bench pins that floor.

package checkpoint

import "testing"

// BenchmarkCheckpoint_FormatStepDir measures the padded-step dirname build for
// a mid-run step (six-digit padded) — the dominant zero-pad branch.
func BenchmarkCheckpoint_FormatStepDir(b *testing.B) {
	b.ReportAllocs()
	var dir string
	for i := 0; i < b.N; i++ {
		dir = FormatStepDir(1200)
	}
	_ = dir
}
