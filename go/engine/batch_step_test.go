// SPDX-Licence-Identifier: EUPL-1.2

package engine

import (
	"context"
	"testing"

	"dappco.re/go/inference"
)

// laneCapableTokenModel is a fakeTokenModel that ALSO implements LaneSetOpener,
// so TextModel over it surfaces the inference.BatchStepModel capability.
type laneCapableTokenModel struct {
	fakeTokenModel
	opened int
	lane   *fakeLaneSet
}

func (m *laneCapableTokenModel) OpenLaneSet(cfg inference.LaneSetConfig) (inference.LaneSet, error) {
	m.opened++
	m.lane = &fakeLaneSet{maxLanes: cfg.MaxLanes}
	return m.lane, nil
}

// fakeLaneSet is a no-op inference.LaneSet — the delegation target under test.
type fakeLaneSet struct {
	maxLanes int
	closed   bool
}

func (l *fakeLaneSet) Prepare(context.Context, inference.LaneSpec) (inference.LaneHandle, error) {
	return inference.LaneHandle{ID: 1}, nil
}
func (l *fakeLaneSet) Step(context.Context) ([]inference.LaneStep, error) { return nil, nil }
func (l *fakeLaneSet) Retire(inference.LaneHandle) error                  { return nil }
func (l *fakeLaneSet) Active() int                                        { return 0 }
func (l *fakeLaneSet) BatchForwardCount() uint64                          { return 0 }
func (l *fakeLaneSet) Close() error                                       { l.closed = true; return nil }

func TestBatchStepUnavailableWithoutOpener(t *testing.T) {
	m := &TextModel{tm: &fakeTokenModel{}, maxLen: 4096}
	if m.BatchStepAvailable() {
		t.Fatal("BatchStepAvailable should be false for an engine without LaneSetOpener")
	}
	if _, err := m.OpenLaneSet(inference.LaneSetConfig{MaxLanes: 4}); err == nil {
		t.Fatal("OpenLaneSet should refuse (not fall back to serial) when the engine has no lane-set backend")
	}
	// The capability is a plain interface assertion — it must still be reachable
	// on the value even when unavailable (probe-then-check, not wrapper-strip).
	if _, ok := inference.TextModel(m).(inference.BatchStepModel); !ok {
		t.Fatal("TextModel must satisfy inference.BatchStepModel so callers can probe availability")
	}
}

func TestBatchStepAvailableWithOpener(t *testing.T) {
	cap := &laneCapableTokenModel{}
	m := &TextModel{tm: cap, maxLen: 4096}
	if !m.BatchStepAvailable() {
		t.Fatal("BatchStepAvailable should be true for a lane-set-capable engine with the kill switch unset")
	}
	ls, err := m.OpenLaneSet(inference.LaneSetConfig{MaxLanes: 3})
	if err != nil {
		t.Fatalf("OpenLaneSet: %v", err)
	}
	if cap.opened != 1 {
		t.Fatalf("OpenLaneSet should delegate to the engine opener exactly once, got %d", cap.opened)
	}
	if cap.lane.maxLanes != 3 {
		t.Fatalf("config should thread through to the opener, got MaxLanes %d", cap.lane.maxLanes)
	}
	_ = ls.Close()
}

func TestBatchStepKillSwitchUnbinds(t *testing.T) {
	cap := &laneCapableTokenModel{}
	m := &TextModel{tm: cap, maxLen: 4096}
	// The EXACT value "0" disables the capability, even on a capable engine.
	t.Setenv("LTHN_CB_STEP", "0")
	if m.BatchStepAvailable() {
		t.Fatal("LTHN_CB_STEP=0 must report the capability unavailable")
	}
	if _, err := m.OpenLaneSet(inference.LaneSetConfig{MaxLanes: 2}); err == nil {
		t.Fatal("LTHN_CB_STEP=0 must make OpenLaneSet refuse")
	}
	if cap.opened != 0 {
		t.Fatalf("kill switch must prevent any delegation to the engine opener, got %d", cap.opened)
	}
}

func TestBatchStepKillSwitchOnlyExactZero(t *testing.T) {
	cap := &laneCapableTokenModel{}
	m := &TextModel{tm: cap, maxLen: 4096}
	// Any value other than the exact "0" leaves the capability bound (unset is
	// the common case; "1"/"on" must not be read as "off").
	t.Setenv("LTHN_CB_STEP", "1")
	if !m.BatchStepAvailable() {
		t.Fatal("LTHN_CB_STEP=1 must leave the capability bound")
	}
	if _, err := m.OpenLaneSet(inference.LaneSetConfig{MaxLanes: 2}); err != nil {
		t.Fatalf("LTHN_CB_STEP=1 OpenLaneSet: %v", err)
	}
}
