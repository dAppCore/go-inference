// SPDX-Licence-Identifier: EUPL-1.2

// serve_profile.go is the tuned-profile draft-block resolution ported out of
// lthn-mlx's serve command so the business logic lives in a go-inference
// library. A serve picks the MTP draft block from (in order): an explicit
// --draft-block flag, the newest matching tuned profile written by `lthn-mlx
// tune` / `lem tune`, then the engine default. Profiles are plain JSON scanned
// off disk — engine-neutral, no weights opened.

package serving

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
)

// tuneDraftBlockLabel is the tuning-candidate label key that carries the proven
// MTP draft block. Matches the key `lthn-mlx tune` / `lem tune` writes.
const tuneDraftBlockLabel = "mtp_draft_block"

// standardTuningProfileDir returns ~/Lethean/lem/tuning — the canonical
// directory tuned profiles are written to and resolved from.
func standardTuningProfileDir() string {
	return core.PathJoin(core.Env("HOME"), "Lethean", "lem", "tuning")
}

// resolveServeDraftBlock picks the draft block the MTP lane will run: an
// explicit flagBlock wins, then the newest matching tuned profile (matched on
// model path + machine hash), then the engine default (0, resolved by the
// loader). noAutoProfile stands the profile lookup down. machineHash is the
// caller's machine identity ("" accepts only hash-less profiles). Returns the
// block plus an operator note naming the source when a profile applied.
func resolveServeDraftBlock(detection DraftDetection, modelPath string, flagBlock int, noAutoProfile bool, profileDir, machineHash string) (int, string) {
	if !detection.Active() || flagBlock > 0 || noAutoProfile {
		return flagBlock, ""
	}
	dir := core.Trim(profileDir)
	if dir == "" {
		dir = standardTuningProfileDir()
	}
	block, path := loadTunedDraftBlock(dir, modelPath, machineHash)
	if block <= 0 {
		return 0, ""
	}
	return block, "tuned: " + path
}

// loadTunedDraftBlock scans dir for tuning profiles matching the model path
// (and machine hash, when both sides carry one) and returns the newest winner's
// draft block. Unparseable files are skipped — a corrupt profile must never
// stop a serve from booting.
func loadTunedDraftBlock(dir, modelPath, machineHash string) (int, string) {
	bestBlock, bestPath := 0, ""
	var bestCreated int64 = -1
	for _, path := range core.PathGlob(core.JoinPath(dir, "*.json")) {
		data := core.ReadFile(path)
		if !data.OK {
			continue
		}
		raw, ok := data.Value.([]byte)
		if !ok {
			continue
		}
		var profile inference.TuningProfile
		if result := core.JSONUnmarshal(raw, &profile); !result.OK {
			continue
		}
		if profile.Key.Model.Path != modelPath {
			continue
		}
		if profile.Key.MachineHash != "" && machineHash != "" && profile.Key.MachineHash != machineHash {
			continue
		}
		blockLabel := profile.Candidate.Labels[tuneDraftBlockLabel]
		parsed := core.ParseInt(core.Trim(blockLabel), 10, 32)
		if !parsed.OK {
			continue
		}
		block := int(parsed.Value.(int64))
		if block < 2 || block > 8 {
			continue
		}
		if profile.CreatedAtUnix > bestCreated {
			bestCreated = profile.CreatedAtUnix
			bestBlock = block
			bestPath = path
		}
	}
	return bestBlock, bestPath
}
