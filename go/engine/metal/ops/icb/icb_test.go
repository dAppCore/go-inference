// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package icb

import "testing"

func TestRecorderRangeCoversAllocatedCommands(t *testing.T) {
	const count = 37
	r := Recorder{Range: commandRange(count)}
	if r.Range.Location != 0 || r.Range.Length != count {
		t.Fatalf("range = {%d %d}, want {0 %d}", r.Range.Location, r.Range.Length, count)
	}
}
