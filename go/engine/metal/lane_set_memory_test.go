// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/inference/model"
)

// laneMemoryTestArch builds a single-layer, single-head synthetic arch so a
// lane's KV cost is one clean number (kvHeads × headDim × bf16Size × 2) —
// admitLaneMemoryBudget's own tests care about the ADMISSION decision, not
// the KV-shape arithmetic (sessionKVBytesAt's own tests, load_context_ram_test.go,
// already cover global/sliding shape).
func laneMemoryTestArch(kvHeads, headDim int) model.Arch {
	return model.Arch{
		Layer:   model.DeriveLayers([]string{"full_attention"}, 0),
		KVHeads: kvHeads, HeadDim: headDim,
	}
}

// TestAdmitLaneMemoryBudget_Good admits a tiny lane onto a roomy box.
func TestAdmitLaneMemoryBudget_Good(t *testing.T) {
	arch := laneMemoryTestArch(1, 1)
	if err := admitLaneMemoryBudget(arch, 1000, 0, 64<<30, 1); err != nil {
		t.Fatalf("admitLaneMemoryBudget(tiny lane, roomy box) = %v, want nil", err)
	}
}

// TestAdmitLaneMemoryBudget_Bad declines cleanly (a real error, not a panic
// or a silent admit) when the model's own weights already crowd the box —
// there is no room for even one lane's KV cache.
func TestAdmitLaneMemoryBudget_Bad(t *testing.T) {
	err := admitLaneMemoryBudget(laneMemoryTestArch(1, 1), 1000, 16<<30, 16<<30, 1)
	if err == nil {
		t.Fatal("admitLaneMemoryBudget(weights crowd the box) = nil, want a declined error")
	}
}

// TestAdmitLaneMemoryBudget_Ugly is the ticket's exact scenario: a per-lane
// KV cost that comfortably admits at a small lane count is DECLINED once
// enough lanes stack up — the co-residency gap a per-request/single-session
// check alone can never see, because no single admission looks over-budget
// on its own. It also pins the fail-OPEN contract: an unmeasured box
// (ramBytes==0) never declines, no matter how large liveLanes is.
func TestAdmitLaneMemoryBudget_Ugly(t *testing.T) {
	arch := laneMemoryTestArch(1, 1024)
	const ctxLen, ramBytes, weightsBytes = 100000, uint64(64) << 30, uint64(0)

	// Fails open on an unmeasured box regardless of scale.
	if err := admitLaneMemoryBudget(arch, ctxLen, weightsBytes, 0, 1_000_000); err != nil {
		t.Fatalf("admitLaneMemoryBudget(ramBytes=0) = %v, want nil (fails open, unmeasured)", err)
	}

	// Derive the real fits/doesn't-fit boundary from the same primitives
	// admitLaneMemoryBudget itself calls, so this stays correct even if the
	// reserve/fixed constants ever change.
	perLane := sessionKVBytesAt(arch, ctxLen)
	deviceBudget, ok := sessionMemoryBudgetBytes(weightsBytes, ramBytes)
	if !ok || perLane == 0 {
		t.Fatalf("test fixture invalid: sessionMemoryBudgetBytes ok=%v sessionKVBytesAt=%d", ok, perLane)
	}
	fitsAt := int64(deviceBudget / perLane)
	if fitsAt < 1 {
		t.Fatalf("test fixture too tight: fitsAt=%d lanes", fitsAt)
	}

	if err := admitLaneMemoryBudget(arch, ctxLen, weightsBytes, ramBytes, fitsAt); err != nil {
		t.Fatalf("admitLaneMemoryBudget(%d lanes, within budget) = %v, want nil", fitsAt, err)
	}
	over := fitsAt + 1000 // comfortably past the boundary regardless of rounding
	if err := admitLaneMemoryBudget(arch, ctxLen, weightsBytes, ramBytes, over); err == nil {
		t.Fatalf("admitLaneMemoryBudget(%d lanes, over budget) = nil, want a declined error", over)
	}
}
