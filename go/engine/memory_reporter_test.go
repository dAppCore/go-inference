// SPDX-Licence-Identifier: EUPL-1.2

package engine

import (
	"context"
	"testing"
)

// memoryReportingTokenModel is a fakeTokenModel that also implements the
// MemoryReporter capability — the double the setMetrics fold is pinned with.
type memoryReportingTokenModel struct {
	fakeTokenModel
	peak   uint64
	active uint64
}

func (m *memoryReportingTokenModel) PeakMemoryBytes() uint64   { return m.peak }
func (m *memoryReportingTokenModel) ActiveMemoryBytes() uint64 { return m.active }

// TestModel_Metrics_MemoryReporter_Good pins the seam: a TokenModel with the
// MemoryReporter capability lands its counters on GenerateMetrics after a
// generation (mantis #1843 — the fields were documented but never populated).
func TestModel_Metrics_MemoryReporter_Good(t *testing.T) {
	tm := &memoryReportingTokenModel{peak: 7 << 20, active: 5 << 20}
	m := newTestModel(t, tm)
	for range m.Generate(context.Background(), "hello") {
	}
	if r := m.Err(); !r.OK {
		t.Fatalf("generate failed: %+v", r)
	}
	mt := m.Metrics()
	if mt.PeakMemoryBytes != 7<<20 || mt.ActiveMemoryBytes != 5<<20 {
		t.Fatalf("memory counters = peak %d active %d, want the reporter's 7MiB/5MiB", mt.PeakMemoryBytes, mt.ActiveMemoryBytes)
	}
}

// TestModel_Metrics_MemoryReporter_Bad pins the honest absence: an engine
// WITHOUT the capability leaves both counters at zero — never fabricated.
func TestModel_Metrics_MemoryReporter_Bad(t *testing.T) {
	m := newTestModel(t, &fakeTokenModel{})
	for range m.Generate(context.Background(), "hello") {
	}
	if r := m.Err(); !r.OK {
		t.Fatalf("generate failed: %+v", r)
	}
	mt := m.Metrics()
	if mt.PeakMemoryBytes != 0 || mt.ActiveMemoryBytes != 0 {
		t.Fatalf("capability-less engine reported memory: peak %d active %d, want 0/0", mt.PeakMemoryBytes, mt.ActiveMemoryBytes)
	}
}
