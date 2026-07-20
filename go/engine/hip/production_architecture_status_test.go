// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"testing"

	core "dappco.re/go"
)

// TestProductionArchitectureStatus_DefaultProductionArchitectureStatus_CountsReconcile_Good
// asserts the report's native/gap counts and ID slices stay internally
// consistent against rocmCapabilityArchitectures, and that every id the
// report calls native truly has a native ROCm runtime (per
// supportedNativeArchitecture) while every reported gap truly does not.
func TestProductionArchitectureStatus_DefaultProductionArchitectureStatus_CountsReconcile_Good(t *testing.T) {
	report := DefaultProductionArchitectureStatus()
	core.AssertEqual(t, len(rocmCapabilityArchitectures), report.TotalArchitectures)
	core.AssertEqual(t, report.TotalArchitectures, report.NativeArchitectures+report.MetadataOnlyArchitectures)
	core.AssertEqual(t, report.MetadataOnlyArchitectures, len(report.RemainingGaps))
	core.AssertEqual(t, report.NativeArchitectures, len(report.NativeIDs))
	core.AssertEqual(t, report.MetadataOnlyArchitectures, len(report.MetadataOnlyIDs))
	for _, id := range report.NativeIDs {
		core.AssertTrue(t, supportedNativeArchitecture(id), id)
	}
	for _, gap := range report.RemainingGaps {
		core.AssertFalse(t, supportedNativeArchitecture(gap.ID), gap.ID)
	}
}

// TestProductionArchitectureStatus_DefaultProductionArchitectureStatus_ComposedRetiredGapsCarryTruthfulNotes_Good
// asserts every architecture whose only prior ROCm runtime was the retired
// composed detour (#50) — the gated-delta hybrid family — is reported as a
// gap, never claimed native, and carries the truthful, canonically-sourced
// explanation naming the retirement. Before this fix, ProductionArchitectureGap.Notes
// was declared but never populated (this file was last touched at the
// original hip-engine landing, bd50aca3, and was never revisited by the
// composed-strip or the truth-pass merges), so every gap silently reported an
// empty Notes regardless of why the architecture had no native runtime.
func TestProductionArchitectureStatus_DefaultProductionArchitectureStatus_ComposedRetiredGapsCarryTruthfulNotes_Good(t *testing.T) {
	report := DefaultProductionArchitectureStatus()
	gapNotes := make(map[string][]string, len(report.RemainingGaps))
	for _, gap := range report.RemainingGaps {
		gapNotes[gap.ID] = gap.Notes
	}

	for _, id := range []string{"qwen3_6", "qwen3_6_moe", "qwen3_next"} {
		notes, ok := gapNotes[id]
		core.AssertTrue(t, ok, id+" must be reported as a gap, not a native architecture")
		core.AssertNotEmpty(t, notes, id)
		core.AssertContains(t, notes[0], "composed route retired (#50)", id)
		core.AssertContains(t, notes[0], "factory-native port pending", id)
	}

	for _, id := range []string{"deepseek", "deepseek_r1", "mixtral", "qwen3_moe"} {
		notes, ok := gapNotes[id]
		core.AssertTrue(t, ok, id+" must be reported as a gap, not a native architecture")
		core.AssertNotEmpty(t, notes, id)
		core.AssertContains(t, notes[0], "model-integrated expert decode remains pending", id)
	}
}

// TestProductionArchitectureStatus_DefaultProductionArchitectureStatus_NativeArchitecturesNeverGaps_Bad
// asserts architectures with a genuine native ROCm runtime never surface as
// gaps — including ids this file's older special-cased branches
// (productionArchitectureMissingNative, productionArchitectureNextWork) still
// mention text for ("bert", "gpt-oss", "kimi"): those branches are dead code
// for a native id because DefaultProductionArchitectureStatus's loop skips
// straight to NativeIDs before productionArchitectureGap is ever called.
func TestProductionArchitectureStatus_DefaultProductionArchitectureStatus_NativeArchitecturesNeverGaps_Bad(t *testing.T) {
	report := DefaultProductionArchitectureStatus()
	gapIDs := make(map[string]bool, len(report.RemainingGaps))
	for _, gap := range report.RemainingGaps {
		gapIDs[gap.ID] = true
	}
	for _, id := range []string{"kimi", "gpt-oss", "bert", "bert_rerank", "minimax_m2", "gemma4"} {
		core.AssertFalse(t, gapIDs[id], id)
	}
}

// TestProductionArchitectureStatus_ProductionArchitectureGapNotes_UnknownID_Ugly asserts
// an id outside the built-in architecture-profile catalogue reports no notes
// rather than panicking or fabricating text.
func TestProductionArchitectureStatus_ProductionArchitectureGapNotes_UnknownID_Ugly(t *testing.T) {
	core.AssertEmpty(t, productionArchitectureGapNotes("zzz_not_a_real_architecture_zzz"))
}
