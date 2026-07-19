// SPDX-Licence-Identifier: EUPL-1.2

package inference

import (
	"testing"
	"time"

	core "dappco.re/go"
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

// stubSpeculativePairBackend layers SpeculativePairBackend over stubBackend —
// the shape the metal engine's metalBackend provides by delegating to its own
// pair-loading machinery.
type stubSpeculativePairBackend struct {
	stubBackend
	pairErr error
}

func (s *stubSpeculativePairBackend) LoadSpeculativePair(targetPath, draftPath string, draftBlock int, opts ...LoadOption) (TextModel, error) {
	if s.pairErr != nil {
		return nil, s.pairErr
	}
	return &stubTextModel{backend: s.name, path: targetPath}, nil
}

// TestSpeculative_SpeculativePairBackend_Good pins the discovery seam
// train/tune uses: a registered Backend implementing SpeculativePairBackend is
// found via Get + a type assertion (no model loaded yet), and its
// LoadSpeculativePair call reaches the concrete backend.
func TestSpeculative_SpeculativePairBackend_Good(t *testing.T) {
	resetBackends(t)
	b := &stubSpeculativePairBackend{stubBackend: stubBackend{name: "pair_backend", available: true}}
	Register(b)

	got, ok := Get("pair_backend")
	checkTrue(t, ok)

	spl, ok := got.(SpeculativePairBackend)
	checkTrue(t, ok)

	tm, err := spl.LoadSpeculativePair("/models/target", "/models/draft", 5)
	checkNoError(t, err)
	checkNotNil(t, tm)
	checkEqual(t, "/models/target", tm.(*stubTextModel).path)
}

// TestSpeculative_SpeculativePairBackend_Bad pins the propagated-failure arm: a
// backend that implements the capability but fails to load the pair (a bad
// checkpoint, an unreachable drafter) returns the error rather than a nil
// TextModel with no error.
func TestSpeculative_SpeculativePairBackend_Bad(t *testing.T) {
	resetBackends(t)
	wantErr := core.NewError("attach drafter: no such file")
	b := &stubSpeculativePairBackend{stubBackend: stubBackend{name: "pair_backend", available: true}, pairErr: wantErr}
	Register(b)

	got, ok := Get("pair_backend")
	checkTrue(t, ok)
	spl, ok := got.(SpeculativePairBackend)
	checkTrue(t, ok)

	tm, err := spl.LoadSpeculativePair("/models/target", "/models/draft", 5)

	checkError(t, err)
	checkNil(t, tm)
}

// TestSpeculative_SpeculativePairBackend_UglyNonImplementor pins the absence
// case: an ordinary Backend that does NOT implement the capability fails the
// type assertion cleanly (ok=false) rather than panicking — the shape RunTune
// probes for before deciding the MTP sweep is blocked.
func TestSpeculative_SpeculativePairBackend_UglyNonImplementor(t *testing.T) {
	resetBackends(t)
	b := &stubBackend{name: "plain_backend", available: true}
	Register(b)

	got, ok := Get("plain_backend")
	checkTrue(t, ok)

	_, ok = got.(SpeculativePairBackend)
	checkFalse(t, ok)
}
