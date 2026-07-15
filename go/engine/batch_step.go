// SPDX-Licence-Identifier: EUPL-1.2

package engine

import (
	"os"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// LaneSetOpener is the engine-level capability a concrete [TokenModel]
// implements when its backend can advance K INDEPENDENT decode lanes through
// one batched forward — the metal multi-session owner. It is defined here (not
// in the neutral inference package) so a backend implements it structurally
// without importing engine; TextModel surfaces it to the neutral
// [inference.BatchStepModel] contract the scheduler probes.
//
// A backend that cannot batch across the session dimension simply does not
// implement this — TextModel then reports the capability unavailable and every
// caller stays on its existing single-session path, no contract change.
type LaneSetOpener interface {
	OpenLaneSet(cfg inference.LaneSetConfig) (inference.LaneSet, error)
}

// batchStepKillSwitch reports the LTHN_CB_STEP=0 kill switch. The EXACT value
// "0" disables multi-session batched stepping; unset or any other value leaves
// the capability bound. Checked here (the engine surface) so the gate has one
// home and the metal owner never has to re-decide it.
func batchStepKillSwitch() bool {
	return os.Getenv("LTHN_CB_STEP") == "0"
}

// BatchStepAvailable implements [inference.BatchStepModel]: true only when the
// loaded engine implements [LaneSetOpener] AND the kill switch is not set. The
// scheduler probes this before building a step coordinator, so LTHN_CB_STEP=0
// leaves the request path byte-for-byte unchanged.
func (m *TextModel) BatchStepAvailable() bool {
	if m == nil || m.tm == nil || batchStepKillSwitch() {
		return false
	}
	_, ok := m.tm.(LaneSetOpener)
	return ok
}

// OpenLaneSet implements [inference.BatchStepModel]. It refuses with a clear
// error — never a silent serial-loop fallback — when the capability is
// unavailable (kill switch set, or the loaded engine has no lane-set backend),
// so a caller that reached here without checking BatchStepAvailable still can
// never mistake a serial degrade for a real batched owner.
func (m *TextModel) OpenLaneSet(cfg inference.LaneSetConfig) (inference.LaneSet, error) {
	if m == nil || m.tm == nil {
		return nil, core.NewError("engine.TextModel.OpenLaneSet: model is not initialised")
	}
	if batchStepKillSwitch() {
		return nil, core.NewError("engine.TextModel.OpenLaneSet: LTHN_CB_STEP=0 — multi-session batched step disabled")
	}
	opener, ok := m.tm.(LaneSetOpener)
	if !ok {
		return nil, core.NewError("engine.TextModel.OpenLaneSet: loaded engine does not support multi-session batched stepping")
	}
	return opener.OpenLaneSet(cfg)
}

var _ inference.BatchStepModel = (*TextModel)(nil)
