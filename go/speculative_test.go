// SPDX-Licence-Identifier: EUPL-1.2

package inference

import (
	"testing"
	"time"
)

type speculativeModel struct {
	metrics SpeculativeMetrics
}

func (m *speculativeModel) SpeculativeMetrics() SpeculativeMetrics { return m.metrics }

func TestSpeculative_SpeculativeMetricsProvider_Good(t *testing.T) {
	want := SpeculativeMetrics{
		DraftTokenSchedule: []int{4, 4, 2},
		ProposedTokens:     10,
		AcceptedTokens:     7,
		RejectedTokens:     3,
		TargetVerifyCalls:  3,
		TargetCalls:        4,
		DraftCalls:         3,
		AcceptanceRate:     0.7,
		WallDuration:       250 * time.Millisecond,
		PeakMemoryBytes:    2 << 30,
	}
	var provider any = &speculativeModel{metrics: want}

	p, ok := provider.(SpeculativeMetricsProvider)

	checkTrue(t, ok)
	checkEqual(t, want, p.SpeculativeMetrics())
}

func TestSpeculative_SpeculativeMetrics_BadNotEngaged(t *testing.T) {
	var provider any = &speculativeModel{}

	p, ok := provider.(SpeculativeMetricsProvider)

	checkTrue(t, ok)
	// zero-valued ProposedTokens is the "speculation did not engage" contract
	checkEqual(t, 0, p.SpeculativeMetrics().ProposedTokens)
}

func TestSpeculative_SpeculativeMetricsProvider_UglyNonProvider(t *testing.T) {
	var provider any = struct{}{}

	_, ok := provider.(SpeculativeMetricsProvider)

	checkFalse(t, ok)
}
