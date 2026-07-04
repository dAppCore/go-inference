// SPDX-Licence-Identifier: EUPL-1.2

package memorypretrain

import core "dappco.re/go"

const (
	// BankFileKind identifies hierarchical-memory pretraining bank files.
	BankFileKind = "go-rocm/memorypretrain-bank"
	// GoMLXBankFileKind identifies sibling go-mlx hierarchical-memory banks.
	// The bank payload schema is shared so ROCm training lanes can consume
	// memory banks built on the Metal backend without rebuilding embeddings.
	GoMLXBankFileKind = "go-mlx/memorypretrain-bank"
	// BankFileVersion is the JSON envelope schema version.
	BankFileVersion = 1
)

var (
	errBankNil                    = core.NewError("memorypretrain: bank is nil")
	errBankFileCoreResult         = core.NewError("memorypretrain: core file operation failed")
	errBankFileUnsupportedVersion = core.NewError("memorypretrain: unsupported bank file version")
	errBankFileInvalidKind        = core.NewError("memorypretrain: invalid bank file kind")
)

type bankFileEnvelope struct {
	Version int    `json:"version"`
	Kind    string `json:"kind"`
	Bank    Bank   `json:"bank"`
}

// Save writes bank to path using the versioned go-rocm memory-pretraining bank
// JSON envelope.
func (bank *Bank) Save(path string) error {
	return SaveBank(path, bank)
}

// SaveBank writes bank to path using the versioned go-rocm memory-pretraining
// bank JSON envelope.
func SaveBank(path string, bank *Bank) error {
	if path == "" {
		return core.NewError("memorypretrain: bank path is required")
	}
	if err := validateBank(bank); err != nil {
		return err
	}
	envelope := bankFileEnvelope{
		Version: BankFileVersion,
		Kind:    BankFileKind,
		Bank:    *bank,
	}
	encoded := core.JSONMarshalIndent(envelope, "", "  ")
	if !encoded.OK {
		return core.E("memorypretrain.SaveBank", "marshal bank", memoryPretrainResultError(encoded))
	}
	dir := core.PathDir(path)
	if dir != "" && dir != "." {
		if result := core.MkdirAll(dir, 0o755); !result.OK {
			return core.E("memorypretrain.SaveBank", "create bank directory", memoryPretrainResultError(result))
		}
	}
	if result := core.WriteFile(path, encoded.Value.([]byte), 0o644); !result.OK {
		return core.E("memorypretrain.SaveBank", "write bank", memoryPretrainResultError(result))
	}
	return nil
}

// LoadBank reads a versioned go-rocm memory-pretraining bank JSON envelope from
// path and validates the bank structure before returning it.
func LoadBank(path string) (*Bank, error) {
	if path == "" {
		return nil, core.NewError("memorypretrain: bank path is required")
	}
	read := core.ReadFile(path)
	if !read.OK {
		return nil, core.E("memorypretrain.LoadBank", "read bank", memoryPretrainResultError(read))
	}
	var envelope bankFileEnvelope
	if result := core.JSONUnmarshal(read.Value.([]byte), &envelope); !result.OK {
		return nil, core.E("memorypretrain.LoadBank", "parse bank", memoryPretrainResultError(result))
	}
	if envelope.Version <= 0 || envelope.Version > BankFileVersion {
		return nil, errBankFileUnsupportedVersion
	}
	if !isCompatibleBankFileKind(envelope.Kind) {
		return nil, errBankFileInvalidKind
	}
	bank := &envelope.Bank
	if err := validateBank(bank); err != nil {
		return nil, err
	}
	return bank, nil
}

func isCompatibleBankFileKind(kind string) bool {
	return kind == BankFileKind || kind == GoMLXBankFileKind
}

func validateBank(bank *Bank) error {
	if bank == nil {
		return errBankNil
	}
	if bank.Dimension <= 0 {
		return core.NewError("memorypretrain: bank dimension is required")
	}
	dim, err := validateBlocks(bank.Blocks)
	if err != nil {
		return err
	}
	if dim != bank.Dimension {
		return core.Errorf("memorypretrain: bank dimension %d does not match block dimension %d", bank.Dimension, dim)
	}
	if len(bank.Nodes) == 0 {
		return core.NewError("memorypretrain: bank nodes are required")
	}
	if bank.Root < 0 || bank.Root >= len(bank.Nodes) {
		return core.NewError("memorypretrain: bank root is out of range")
	}
	bank.Config = normaliseBuildConfig(bank.Config)
	for i := range bank.Nodes {
		if err := validateBankNode(bank, i); err != nil {
			return err
		}
	}
	return nil
}

func validateBankNode(bank *Bank, idx int) error {
	node := bank.Nodes[idx]
	if node.ID != idx {
		return core.Errorf("memorypretrain: bank node %d has id %d", idx, node.ID)
	}
	if idx == bank.Root && node.Parent != -1 {
		return core.Errorf("memorypretrain: bank root node parent %d is invalid", node.Parent)
	}
	if idx != bank.Root && node.Parent == idx {
		return core.Errorf("memorypretrain: bank node %d cannot parent itself", idx)
	}
	if node.Parent < -1 || node.Parent >= len(bank.Nodes) {
		return core.Errorf("memorypretrain: bank node %d parent %d is out of range", idx, node.Parent)
	}
	if len(node.Centroid) != bank.Dimension {
		return core.Errorf("memorypretrain: bank node %d centroid dimension %d does not match %d", idx, len(node.Centroid), bank.Dimension)
	}
	for _, child := range node.Children {
		if child < 0 || child >= len(bank.Nodes) {
			return core.Errorf("memorypretrain: bank node %d child %d is out of range", idx, child)
		}
	}
	for _, blockID := range node.BlockIDs {
		if blockID < 0 || blockID >= len(bank.Blocks) {
			return core.Errorf("memorypretrain: bank node %d block %d is out of range", idx, blockID)
		}
	}
	return nil
}

func memoryPretrainResultError(result core.Result) error {
	if result.OK {
		return nil
	}
	if err, ok := result.Value.(error); ok {
		return err
	}
	return errBankFileCoreResult
}
