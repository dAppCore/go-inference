// SPDX-Licence-Identifier: EUPL-1.2

package serving

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// tunedProfile builds a tuning profile carrying blockLabel under the draft-block
// key — the shape `lthn-mlx tune` / `lem tune` writes and loadTunedDraftBlock
// reads.
func tunedProfile(modelPath, machineHash, blockLabel string, created int64) inference.TuningProfile {
	return inference.TuningProfile{
		Key: inference.TuningProfileKey{
			MachineHash: machineHash,
			Model:       inference.ModelIdentity{Path: modelPath},
		},
		Candidate:     inference.TuningCandidate{Labels: map[string]string{tuneDraftBlockLabel: blockLabel}},
		CreatedAtUnix: created,
	}
}

// writeProfileJSON writes p as a JSON profile file at dir/name.
func writeProfileJSON(t *testing.T, dir, name string, p inference.TuningProfile) {
	t.Helper()
	if r := core.WriteFile(core.JoinPath(dir, name), []byte(core.JSONMarshalString(p)), 0o644); !r.OK {
		t.Fatalf("write profile %s: %v", name, r.Error())
	}
}

// TestServing_ServeProfile_LoadTunedDraftBlock_Good pins the resolution winner:
// among several matching profiles the NEWEST (highest CreatedAtUnix) wins, and
// its file path is returned so the boot notice can name the source.
func TestServing_ServeProfile_LoadTunedDraftBlock_Good(t *testing.T) {
	dir := t.TempDir()
	const model = "/models/Lemma-v2-e2b"
	writeProfileJSON(t, dir, "old.json", tunedProfile(model, "", "3", 100))
	writeProfileJSON(t, dir, "new.json", tunedProfile(model, "", "5", 200))
	// a non-matching model must not shadow the winner even if it is newer.
	writeProfileJSON(t, dir, "other.json", tunedProfile("/models/gemma", "", "7", 999))

	block, path := loadTunedDraftBlock(dir, model, "")
	if block != 5 {
		t.Fatalf("draft block = %d, want 5 (the newest matching profile)", block)
	}
	if !core.HasSuffix(path, "new.json") {
		t.Fatalf("winner path = %q, want the newest profile file (new.json)", path)
	}
}

// TestServing_ServeProfile_LoadTunedDraftBlock_Bad pins the skip-and-survive
// paths: an absent directory, a malformed profile file, and a model-path
// mismatch all yield the no-tune signal (0, "") without stopping the scan — a
// corrupt profile must never block a serve from booting.
func TestServing_ServeProfile_LoadTunedDraftBlock_Bad(t *testing.T) {
	// Absent: an empty directory has no matching profile.
	if block, path := loadTunedDraftBlock(t.TempDir(), "/models/x", ""); block != 0 || path != "" {
		t.Fatalf("empty dir = (%d, %q), want (0, \"\")", block, path)
	}

	// Malformed + mismatch: neither yields a winner, and the garbage file does
	// not abort the scan of the sibling.
	dir := t.TempDir()
	if r := core.WriteFile(core.JoinPath(dir, "corrupt.json"), []byte("{not json"), 0o644); !r.OK {
		t.Fatalf("write corrupt: %v", r.Error())
	}
	writeProfileJSON(t, dir, "wrongmodel.json", tunedProfile("/models/other", "", "4", 10))
	if block, path := loadTunedDraftBlock(dir, "/models/wanted", ""); block != 0 || path != "" {
		t.Fatalf("malformed+mismatch = (%d, %q), want (0, \"\")", block, path)
	}
}

// TestServing_ServeProfile_LoadTunedDraftBlock_Ugly pins the guards that reject
// a matching-model profile: an out-of-range block (must be 2..8), an
// unparseable label, and the machine-hash gate (skip only when BOTH sides carry
// a hash and they differ; an empty hash on either side is a wildcard).
func TestServing_ServeProfile_LoadTunedDraftBlock_Ugly(t *testing.T) {
	const model = "/models/Lemma"

	t.Run("block out of range and unparseable are skipped", func(t *testing.T) {
		dir := t.TempDir()
		writeProfileJSON(t, dir, "low.json", tunedProfile(model, "", "1", 1))    // < 2
		writeProfileJSON(t, dir, "high.json", tunedProfile(model, "", "9", 2))   // > 8
		writeProfileJSON(t, dir, "junk.json", tunedProfile(model, "", "abc", 3)) // unparseable
		if block, path := loadTunedDraftBlock(dir, model, ""); block != 0 || path != "" {
			t.Fatalf("guarded profiles = (%d, %q), want (0, \"\")", block, path)
		}
	})

	t.Run("machine-hash gate", func(t *testing.T) {
		cases := []struct {
			name        string
			profileHash string
			callerHash  string
			wantMatched bool
		}{
			{"both match", "M1", "M1", true},
			{"both differ", "M1", "M2", false},
			{"caller wildcard", "M1", "", true},
			{"profile wildcard", "", "M2", true},
		}
		for _, tc := range cases {
			dir := t.TempDir()
			writeProfileJSON(t, dir, "p.json", tunedProfile(model, tc.profileHash, "4", 1))
			block, _ := loadTunedDraftBlock(dir, model, tc.callerHash)
			if matched := block == 4; matched != tc.wantMatched {
				t.Errorf("%s: block = %d (matched=%v), want matched=%v", tc.name, block, matched, tc.wantMatched)
			}
		}
	})
}

// TestServing_ServeProfile_ResolveServeDraftBlock_Good pins the resolution
// precedence in resolveServeDraftBlock: an explicit flag wins, no-auto and an
// inactive drafter stand the lookup down, and an active drafter with a matching
// tuned profile returns the block plus a source-named note.
func TestServing_ServeProfile_ResolveServeDraftBlock_Good(t *testing.T) {
	active := DraftDetection{Source: DraftSourceFlag, DraftPath: "/models/draft"}
	inactive := DraftDetection{}
	const model = "/models/Lemma"

	// explicit flag wins — the profile scan is skipped entirely.
	if block, note := resolveServeDraftBlock(active, model, 6, false, t.TempDir(), ""); block != 6 || note != "" {
		t.Fatalf("explicit flag = (%d, %q), want (6, \"\")", block, note)
	}
	// no-auto stands the lookup down.
	if block, note := resolveServeDraftBlock(active, model, 0, true, t.TempDir(), ""); block != 0 || note != "" {
		t.Fatalf("no-auto = (%d, %q), want (0, \"\")", block, note)
	}
	// inactive drafter: nothing to tune.
	if block, note := resolveServeDraftBlock(inactive, model, 0, false, t.TempDir(), ""); block != 0 || note != "" {
		t.Fatalf("inactive = (%d, %q), want (0, \"\")", block, note)
	}

	// active + matching profile → block and a "tuned: <path>" note.
	dir := t.TempDir()
	writeProfileJSON(t, dir, "p.json", tunedProfile(model, "", "5", 1))
	block, note := resolveServeDraftBlock(active, model, 0, false, dir, "")
	if block != 5 {
		t.Fatalf("tuned block = %d, want 5", block)
	}
	if !core.HasPrefix(note, "tuned: ") || !core.HasSuffix(note, "p.json") {
		t.Fatalf("tuned note = %q, want a \"tuned: <...p.json>\" source line", note)
	}
}

// TestServing_ServeProfile_StandardTuningProfileDir_Good pins the canonical
// tuned-profile directory — ~/Lethean/lem/tuning, the path tune writes to and
// serve resolves from.
func TestServing_ServeProfile_StandardTuningProfileDir_Good(t *testing.T) {
	if got := standardTuningProfileDir(); !core.HasSuffix(got, core.JoinPath("Lethean", "lem", "tuning")) {
		t.Fatalf("standardTuningProfileDir = %q, want a .../Lethean/lem/tuning path", got)
	}
}

// TestServing_ServeProfile_WriteTunedDraftBlockProfile_Good pins the
// write/read round trip end to end: a profile written by
// WriteTunedDraftBlockProfile is resolved straight back by
// loadTunedDraftBlock/resolveServeDraftBlock — the exact contract `lem tune`'s
// MTP block sweep and a later serve boot share.
func TestServing_ServeProfile_WriteTunedDraftBlockProfile_Good(t *testing.T) {
	dir := t.TempDir()
	const model = "/models/Lemma-v2-e2b"
	measurements := inference.TuningMeasurements{DecodeTokensPerSec: 88.5, PromptTokens: 12, GeneratedTokens: 64}
	score := inference.ScoreTuningMeasurements(inference.TuningWorkloadChat, measurements)

	path, err := WriteTunedDraftBlockProfile(dir, model, "machine-1", inference.TuningWorkloadChat, 6, measurements, score, 12345)
	if err != nil {
		t.Fatalf("WriteTunedDraftBlockProfile: %v", err)
	}
	if !core.HasSuffix(path, ".json") {
		t.Fatalf("written path = %q, want a .json file", path)
	}

	data := core.ReadFile(path)
	if !data.OK {
		t.Fatalf("read written profile: %v", data.Error())
	}
	var profile inference.TuningProfile
	if r := core.JSONUnmarshal(data.Value.([]byte), &profile); !r.OK {
		t.Fatalf("unmarshal written profile: %v", r.Error())
	}
	if profile.Key.Model.Path != model || profile.Key.MachineHash != "machine-1" {
		t.Fatalf("profile key = %+v, want model %q machine-1", profile.Key, model)
	}
	if profile.Candidate.Labels[tuneDraftBlockLabel] != "6" {
		t.Fatalf("profile block label = %q, want %q", profile.Candidate.Labels[tuneDraftBlockLabel], "6")
	}

	// The read side (matched by content, not filename) resolves the same block.
	block, resolvedPath := loadTunedDraftBlock(dir, model, "machine-1")
	if block != 6 || resolvedPath != path {
		t.Fatalf("loadTunedDraftBlock = (%d, %q), want (6, %q)", block, resolvedPath, path)
	}
}

// TestServing_ServeProfile_WriteTunedDraftBlockProfile_Bad pins the guard: an
// out-of-range block (loadTunedDraftBlock would silently discard it) is
// refused up front rather than written as a profile serve can never apply.
func TestServing_ServeProfile_WriteTunedDraftBlockProfile_Bad(t *testing.T) {
	dir := t.TempDir()
	for _, block := range []int{0, 1, 9, -3} {
		if _, err := WriteTunedDraftBlockProfile(dir, "/models/x", "", inference.TuningWorkloadChat, block, inference.TuningMeasurements{}, inference.TuningScore{}, 1); err == nil {
			t.Fatalf("WriteTunedDraftBlockProfile(block=%d) = nil error, want a range refusal", block)
		}
	}
	if files := core.PathGlob(core.JoinPath(dir, "*.json")); len(files) != 0 {
		t.Fatalf("a refused write must leave no file behind, found %v", files)
	}
}
