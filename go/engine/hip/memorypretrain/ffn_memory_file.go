// SPDX-Licence-Identifier: EUPL-1.2

package memorypretrain

import core "dappco.re/go"

const (
	// FFNMemoryBankFileKind identifies ROCm hierarchical FFN memory parameter files.
	FFNMemoryBankFileKind = "go-rocm/memorypretrain-ffn-memory"
	// GoMLXFFNMemoryBankFileKind identifies sibling go-mlx FFN memory files.
	// The bank payload schema is shared so ROCm training and serving lanes can
	// consume memory tables built on the Metal backend.
	GoMLXFFNMemoryBankFileKind = "go-mlx/memorypretrain-ffn-memory"
	// FFNMemoryBankFileVersion is the JSON envelope schema version.
	FFNMemoryBankFileVersion = 1
)

var (
	errFFNMemoryBankNil                    = core.NewError("memorypretrain: FFN memory bank is nil")
	errFFNMemoryBankFileCoreResult         = core.NewError("memorypretrain: core file operation failed")
	errFFNMemoryBankFileUnsupportedVersion = core.NewError("memorypretrain: unsupported FFN memory bank file version")
	errFFNMemoryBankFileInvalidKind        = core.NewError("memorypretrain: invalid FFN memory bank file kind")
)

type ffnMemoryBankFileEnvelope struct {
	Version int           `json:"version"`
	Kind    string        `json:"kind"`
	Bank    FFNMemoryBank `json:"bank"`
}

// Save writes bank to path using the versioned go-rocm FFN memory bank JSON
// envelope.
func (bank *FFNMemoryBank) Save(path string) error {
	return SaveFFNMemoryBank(path, bank)
}

// SaveFFNMemoryBank writes bank to path using a versioned JSON envelope.
func SaveFFNMemoryBank(path string, bank *FFNMemoryBank) error {
	if path == "" {
		return core.NewError("memorypretrain: FFN memory bank path is required")
	}
	if err := validateFFNMemoryBank(bank); err != nil {
		return err
	}
	envelope := ffnMemoryBankFileEnvelope{
		Version: FFNMemoryBankFileVersion,
		Kind:    FFNMemoryBankFileKind,
		Bank:    *bank,
	}
	encoded := core.JSONMarshalIndent(envelope, "", "  ")
	if !encoded.OK {
		return core.E("memorypretrain.SaveFFNMemoryBank", "marshal bank", memoryPretrainResultError(encoded))
	}
	dir := core.PathDir(path)
	if dir != "" && dir != "." {
		if result := core.MkdirAll(dir, 0o755); !result.OK {
			return core.E("memorypretrain.SaveFFNMemoryBank", "create bank directory", memoryPretrainResultError(result))
		}
	}
	if result := core.WriteFile(path, encoded.Value.([]byte), 0o644); !result.OK {
		return core.E("memorypretrain.SaveFFNMemoryBank", "write bank", memoryPretrainResultError(result))
	}
	return nil
}

// LoadFFNMemoryBank reads a versioned go-rocm or go-mlx FFN memory bank JSON
// envelope from path and validates the memory table before returning it.
func LoadFFNMemoryBank(path string) (*FFNMemoryBank, error) {
	if path == "" {
		return nil, core.NewError("memorypretrain: FFN memory bank path is required")
	}
	read := core.ReadFile(path)
	if !read.OK {
		return nil, core.E("memorypretrain.LoadFFNMemoryBank", "read bank", memoryPretrainResultError(read))
	}
	var envelope ffnMemoryBankFileEnvelope
	if result := core.JSONUnmarshal(read.Value.([]byte), &envelope); !result.OK {
		return nil, core.E("memorypretrain.LoadFFNMemoryBank", "parse bank", memoryPretrainResultError(result))
	}
	if envelope.Version <= 0 || envelope.Version > FFNMemoryBankFileVersion {
		return nil, errFFNMemoryBankFileUnsupportedVersion
	}
	if !isCompatibleFFNMemoryBankFileKind(envelope.Kind) {
		return nil, errFFNMemoryBankFileInvalidKind
	}
	bank := &envelope.Bank
	if err := validateFFNMemoryBank(bank); err != nil {
		return nil, err
	}
	return bank, nil
}

func isCompatibleFFNMemoryBankFileKind(kind string) bool {
	return kind == FFNMemoryBankFileKind || kind == GoMLXFFNMemoryBankFileKind
}

func validateFFNMemoryBank(bank *FFNMemoryBank) error {
	if bank == nil {
		return errFFNMemoryBankNil
	}
	if bank.HiddenSize <= 0 {
		return core.NewError("memorypretrain: FFN memory bank hidden size is required")
	}
	bank.Config = normaliseFFNMemoryConfig(bank.Config)
	if bank.Config.HiddenSize != bank.HiddenSize {
		return core.Errorf("memorypretrain: FFN memory bank hidden size %d does not match config %d", bank.HiddenSize, bank.Config.HiddenSize)
	}
	if err := validateFFNMemoryConfig(bank.Config); err != nil {
		return err
	}
	if len(bank.Layers) != bank.Config.Layers {
		return core.Errorf("memorypretrain: FFN memory bank layers %d does not match config %d", len(bank.Layers), bank.Config.Layers)
	}
	for layerID := range bank.Layers {
		if err := validateFFNMemoryLayer(&bank.Layers[layerID], bank.Config, layerID); err != nil {
			return err
		}
	}
	return nil
}

func validateFFNMemoryLayer(layer *FFNMemoryLayer, cfg FFNMemoryConfig, layerID int) error {
	if layer.Layer != layerID {
		return core.Errorf("memorypretrain: FFN memory layer %d has id %d", layerID, layer.Layer)
	}
	if len(layer.Levels) != len(cfg.MemoryLevels) {
		return core.Errorf("memorypretrain: FFN memory layer %d levels %d does not match config %d", layerID, len(layer.Levels), len(cfg.MemoryLevels))
	}
	for levelID := range layer.Levels {
		level := &layer.Levels[levelID]
		if level.Name != cfg.MemoryLevels[levelID] {
			return core.Errorf("memorypretrain: FFN memory layer %d level %d name %q does not match %q", layerID, levelID, level.Name, cfg.MemoryLevels[levelID])
		}
		if level.NumClusters != cfg.NumClusters[levelID] {
			return core.Errorf("memorypretrain: FFN memory layer %d level %s clusters %d does not match %d", layerID, level.Name, level.NumClusters, cfg.NumClusters[levelID])
		}
		if level.AddedGenericSize != cfg.AddedGenericSize {
			return core.Errorf("memorypretrain: FFN memory layer %d level %s generic size %d does not match %d", layerID, level.Name, level.AddedGenericSize, cfg.AddedGenericSize)
		}
		if level.MemoryTokens <= 0 {
			return core.Errorf("memorypretrain: FFN memory layer %d level %s token count must be positive", layerID, level.Name)
		}
		if err := validateFFNMemoryLevel(level, cfg.HiddenSize, 0); err != nil {
			return err
		}
	}
	return nil
}
