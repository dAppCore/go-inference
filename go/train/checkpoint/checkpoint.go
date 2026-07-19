// SPDX-Licence-Identifier: EUPL-1.2

// Package checkpoint is the shared checkpoint-metadata engine behind
// distill/grpo/train's own CheckpointMetadata sidecars. It owns only the
// mechanics every domain's Save/Load/LoadResume shared byte-for-byte
// before this package existed — marshal-and-write, read-and-unmarshal,
// the soft-missing-file semantics --resume relies on, and the
// "step-NNNNNN" checkpoint-dirname convention — never the metadata shape
// itself: distill, grpo, and train each stamp a genuinely different set
// of fields into their own CheckpointMetadata/CheckpointSnapshot types
// (different training regime, different portable run state, and a
// different JSON field order that a shared embedded type would disturb),
// so those types stay domain-owned and are not reproduced here.
//
// A leaf package: imports only core (dappco.re/go), nothing else from
// this module, so distill/grpo/train can each depend on it without any
// risk of an import cycle.
//
//	err := checkpoint.Save(sidecarPath, meta)
//	loaded, err := checkpoint.Load[MyMetadata](sidecarPath)
//	resumed, err := checkpoint.LoadResume[MyMetadata](sidecarPath)
//	dir := checkpoint.FormatStepDir(step)
package checkpoint

import (
	"strconv"

	core "dappco.re/go"
)

// Save marshals meta as indented JSON and writes it to sidecarPath,
// creating sidecarPath's parent directory first if it does not already
// exist. This is the write-side engine shared by every domain package's
// own SaveCheckpointMetadata (distill/grpo/train) — each wrapper computes
// its own sidecar path and normalises its own Version/Path defaults
// before calling in, so Save itself stays domain-agnostic: it never
// reads or sets a field on meta.
//
//	err := checkpoint.Save(checkpointMetadataPath(path), meta)
func Save(sidecarPath string, meta any) error {
	dir := core.PathDir(sidecarPath)
	if dir != "" && dir != "." {
		if result := core.MkdirAll(dir, 0o755); !result.OK {
			return core.E("CheckpointMetadata.Save", "ensure metadata dir", result.Err())
		}
	}
	data := core.JSONMarshalIndent(meta, "", "  ")
	if !data.OK {
		return core.E("CheckpointMetadata.Save", "marshal metadata", data.Err())
	}
	if result := core.WriteFile(sidecarPath, data.Bytes(), 0o600); !result.OK {
		return core.E("CheckpointMetadata.Save", "write metadata", result.Err())
	}
	return nil
}

// Load reads and JSON-decodes the sidecar at sidecarPath into a freshly
// allocated T, returning a hard error when the sidecar is missing or
// unparsable. This is the read-side engine shared by every domain
// package's own LoadCheckpointMetadata; LoadResume below is the soft-
// missing-file variant a driver's --resume flow needs instead.
//
//	meta, err := checkpoint.Load[CheckpointMetadata](checkpointMetadataPath(path))
func Load[T any](sidecarPath string) (*T, error) {
	read := core.ReadFile(sidecarPath)
	if !read.OK {
		return nil, read.Err()
	}
	var meta T
	if result := core.JSONUnmarshal(read.Bytes(), &meta); !result.OK {
		return nil, core.E("LoadCheckpointMetadata", "parse metadata", result.Err())
	}
	return &meta, nil
}

// LoadResume reads and JSON-decodes the sidecar at sidecarPath, returning
// (nil, nil) when the sidecar does not exist yet — the soft-missing-file
// semantics every domain's --resume flow relies on: a driver's own loop
// treats an absent resume checkpoint as "start fresh" rather than an
// error. A corrupt sidecar (exists but fails to parse) is still a real
// error, never swallowed.
//
//	meta, err := checkpoint.LoadResume[CheckpointMetadata](checkpointMetadataPath(path))
func LoadResume[T any](sidecarPath string) (*T, error) {
	read := core.ReadFile(sidecarPath)
	if !read.OK {
		err := read.Err()
		if core.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var meta T
	if result := core.JSONUnmarshal(read.Bytes(), &meta); !result.OK {
		return nil, core.E("LoadResumeMetadata", "parse metadata", result.Err())
	}
	return &meta, nil
}

// FormatStepDir builds the conventional "step-NNNNNN" checkpoint dirname
// shared by every domain package's own FormatStepDir (distill/grpo/
// train) — zero-padded to 6 digits below 100000, natural width at or
// above it. Uses strconv.AppendInt with explicit zero padding, avoiding
// fmt's reflection path on the per-checkpoint hot loop. Digit count is
// computed in place instead of via a throwaway strconv.AppendInt(nil,
// ...) so the function allocates exactly once — the returned string
// itself.
//
//	dir := core.PathJoin(cfg.CheckpointDir, checkpoint.FormatStepDir(step))
func FormatStepDir(step int) string {
	const prefix = "step-"
	const padTo = 6
	buf := make([]byte, 0, len(prefix)+20)
	buf = append(buf, prefix...)
	if step >= 0 && step < 100000 {
		digits := 1
		for n := step / 10; n > 0; n /= 10 {
			digits++
		}
		for i := digits; i < padTo; i++ {
			buf = append(buf, '0')
		}
	}
	buf = strconv.AppendInt(buf, int64(step), 10)
	return string(buf)
}
