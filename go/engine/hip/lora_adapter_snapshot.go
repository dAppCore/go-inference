// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"crypto/sha256"
	"encoding/hex"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// NativeLoRAAdapterSnapshotConfig describes a loadable LoRA adapter snapshot
// produced from the packed LoRA AdamW state.
type NativeLoRAAdapterSnapshotConfig struct {
	Format string
	Name   string
	Target string
	Rows   int
	Cols   int
	Rank   int
	Alpha  float32
	Bias   []float32
}

// SaveNativeLoRAAdapterSnapshot writes the current packed LoRA A/B parameters
// as a ROCm loadable adapter JSON file.
func SaveNativeLoRAAdapterSnapshot(path string, state *NativeAdamWState, cfg NativeLoRAAdapterSnapshotConfig) (inference.AdapterIdentity, error) {
	if path == "" {
		return inference.AdapterIdentity{}, core.NewError("rocm: LoRA adapter snapshot path is required")
	}
	cfg = normalizeNativeLoRAAdapterSnapshotConfig(cfg)
	if err := validateNativeLoRAAdapterSnapshotConfig(cfg); err != nil {
		return inference.AdapterIdentity{}, err
	}
	loraA, loraB, err := nativeLoRAAdamWStateViews(state, cfg.Rows, cfg.Cols, cfg.Rank)
	if err != nil {
		return inference.AdapterIdentity{}, err
	}
	if len(cfg.Bias) != 0 && !rocmFloat32SliceFinite(cfg.Bias) {
		return inference.AdapterIdentity{}, core.NewError("rocm: LoRA adapter snapshot bias values must be finite")
	}

	payload, err := marshalNativeLoRAAdapterSnapshot(loraA, loraB, cfg)
	if err != nil {
		return inference.AdapterIdentity{}, err
	}
	if err := ensureNativeAdamWStateDir(path); err != nil {
		return inference.AdapterIdentity{}, err
	}
	if result := core.WriteFile(path, payload, 0o644); !result.OK {
		return inference.AdapterIdentity{}, core.E("rocm.LoRA.AdapterSnapshot", "write adapter", nativeAdamWResultError(result))
	}
	sum := sha256.Sum256(payload)
	hash := hex.EncodeToString(sum[:])
	return inference.AdapterIdentity{
		Path:       path,
		Hash:       hash,
		Format:     cfg.Format,
		Rank:       cfg.Rank,
		Alpha:      cfg.Alpha,
		TargetKeys: []string{cfg.Target},
		Labels: map[string]string{
			"adapter_file":        path,
			"adapter_alpha":       core.Sprintf("%g", cfg.Alpha),
			"adapter_format":      cfg.Format,
			"adapter_hash":        hash,
			"adapter_name":        cfg.Name,
			"adapter_rank":        core.Sprintf("%d", cfg.Rank),
			"adapter_snapshot":    "lora_adamw_state",
			"adapter_target":      cfg.Target,
			"adapter_target_cols": core.Sprintf("%d", cfg.Cols),
			"adapter_target_rows": core.Sprintf("%d", cfg.Rows),
			"adapter_track":       "loadable_json",
			"target":              cfg.Target,
			"target_cols":         core.Sprintf("%d", cfg.Cols),
			"target_rows":         core.Sprintf("%d", cfg.Rows),
			"trainer_interface":   "not_implemented",
		},
	}, nil
}

// SaveNativeLoRAAdapterSnapshotTrackStep loads a packed LoRA AdamW state from an
// append-only optimizer track step and writes it as a loadable adapter snapshot.
func SaveNativeLoRAAdapterSnapshotTrackStep(trackPath string, step int, snapshotPath string, cfg NativeLoRAAdapterSnapshotConfig) (inference.AdapterIdentity, NativeAdamWTrackRecord, error) {
	state, record, err := LoadNativeAdamWStateTrackStep(trackPath, step)
	if err != nil {
		return inference.AdapterIdentity{}, NativeAdamWTrackRecord{}, err
	}
	identity, err := SaveNativeLoRAAdapterSnapshot(snapshotPath, state, cfg)
	if err != nil {
		return inference.AdapterIdentity{}, NativeAdamWTrackRecord{}, err
	}
	identity, err = addNativeLoRAAdapterSnapshotTrackLabels(identity, trackPath, record, "LoadNativeAdamWStateTrackStep", 0)
	if err != nil {
		return inference.AdapterIdentity{}, NativeAdamWTrackRecord{}, err
	}
	return identity, record, nil
}

// SaveNativeLoRAAdapterSnapshotTrackLast writes the latest complete optimizer
// track frame as a loadable LoRA adapter snapshot.
func SaveNativeLoRAAdapterSnapshotTrackLast(trackPath string, snapshotPath string, cfg NativeLoRAAdapterSnapshotConfig) (inference.AdapterIdentity, NativeAdamWTrackRecord, error) {
	state, record, frames, err := loadLastNativeAdamWStateTrackWithFrameCount(trackPath)
	if err != nil {
		return inference.AdapterIdentity{}, NativeAdamWTrackRecord{}, err
	}
	identity, err := SaveNativeLoRAAdapterSnapshot(snapshotPath, state, cfg)
	if err != nil {
		return inference.AdapterIdentity{}, NativeAdamWTrackRecord{}, err
	}
	identity, err = addNativeLoRAAdapterSnapshotTrackLabels(identity, trackPath, record, "LoadLastNativeAdamWStateTrack", frames)
	if err != nil {
		return inference.AdapterIdentity{}, NativeAdamWTrackRecord{}, err
	}
	return identity, record, nil
}

func addNativeLoRAAdapterSnapshotTrackLabels(identity inference.AdapterIdentity, trackPath string, record NativeAdamWTrackRecord, helper string, frames int) (inference.AdapterIdentity, error) {
	if frames <= 0 {
		records, err := ListNativeAdamWStateTrack(trackPath)
		if err != nil {
			return inference.AdapterIdentity{}, err
		}
		frames = len(records)
	}
	if identity.Labels == nil {
		identity.Labels = map[string]string{}
	}
	identity.Labels["adapter_track_source"] = "adamw_append_only"
	identity.Labels["adapter_track_format"] = "rocm_adamw_track_v1"
	identity.Labels["adapter_track_container"] = NativeAdamWTrackContainer(trackPath)
	identity.Labels["adapter_track_path"] = trackPath
	identity.Labels["adapter_track_offset"] = core.Sprintf("%d", record.Offset)
	identity.Labels["adapter_track_payload_bytes"] = core.Sprintf("%d", record.PayloadSize)
	identity.Labels["adapter_track_step"] = core.Sprintf("%d", record.Step)
	identity.Labels["adapter_track_frames"] = core.Sprintf("%d", frames)
	identity.Labels["adapter_track_load_helper"] = helper
	if helper == "LoadNativeAdamWStateTrackStep" {
		identity.Labels["adapter_track_load_step_helper"] = helper
	}
	return identity, nil
}

func normalizeNativeLoRAAdapterSnapshotConfig(cfg NativeLoRAAdapterSnapshotConfig) NativeLoRAAdapterSnapshotConfig {
	if cfg.Format == "" {
		cfg.Format = rocmTinyLoRAFormat
	}
	if cfg.Name == "" {
		cfg.Name = cfg.Format
	}
	if cfg.Target == "" {
		if cfg.Format == rocmClassifierLoRAFormat {
			cfg.Target = "classifier.weight"
		} else {
			cfg.Target = "output.weight"
		}
	}
	return cfg
}

func validateNativeLoRAAdapterSnapshotConfig(cfg NativeLoRAAdapterSnapshotConfig) error {
	switch cfg.Format {
	case rocmTinyLoRAFormat, rocmSmallLoRAFormat, rocmClassifierLoRAFormat:
	default:
		return core.NewError("rocm: LoRA adapter snapshot format is unsupported")
	}
	if cfg.Rows <= 0 || cfg.Cols <= 0 || cfg.Rank <= 0 {
		return core.NewError("rocm: LoRA adapter snapshot rows, cols, and rank must be positive")
	}
	if !hipQ8ScaleIsPositiveFinite(cfg.Alpha) {
		return core.NewError("rocm: LoRA adapter snapshot alpha must be positive and finite")
	}
	if len(cfg.Bias) != 0 && len(cfg.Bias) != cfg.Rows {
		return core.NewError("rocm: LoRA adapter snapshot bias length must match rows")
	}
	return nil
}

func marshalNativeLoRAAdapterSnapshot(loraA, loraB []float32, cfg NativeLoRAAdapterSnapshotConfig) ([]byte, error) {
	switch cfg.Format {
	case rocmClassifierLoRAFormat:
		file := hipClassifierLoRAAdapterFile{
			Format:     cfg.Format,
			Name:       cfg.Name,
			Target:     cfg.Target,
			Rank:       cfg.Rank,
			Alpha:      cfg.Alpha,
			HiddenSize: cfg.Cols,
			NumLabels:  cfg.Rows,
			LoRAA:      loraA,
			LoRAB:      loraB,
			Bias:       cfg.Bias,
		}
		encoded := core.JSONMarshalIndent(file, "", "  ")
		if !encoded.OK {
			return nil, core.E("rocm.LoRA.AdapterSnapshot", "marshal classifier adapter", nativeAdamWResultError(encoded))
		}
		return encoded.Value.([]byte), nil
	default:
		file := hipTinyLoRAAdapterFile{
			Format:     cfg.Format,
			Name:       cfg.Name,
			Target:     cfg.Target,
			Rank:       cfg.Rank,
			Alpha:      cfg.Alpha,
			HiddenSize: cfg.Cols,
			VocabSize:  cfg.Rows,
			LoRAA:      loraA,
			LoRAB:      loraB,
			Bias:       cfg.Bias,
		}
		encoded := core.JSONMarshalIndent(file, "", "  ")
		if !encoded.OK {
			return nil, core.E("rocm.LoRA.AdapterSnapshot", "marshal adapter", nativeAdamWResultError(encoded))
		}
		return encoded.Value.([]byte), nil
	}
}
