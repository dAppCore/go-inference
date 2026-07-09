//go:build linux && amd64 && rocm_legacy_server

package hip

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/engine/hip/internal/gguf"
)

// rocmBackend implements inference.Backend for AMD ROCm GPUs.
type rocmBackend struct{}

const defaultContextLengthCap = 4096

func (b *rocmBackend) Name() string { return "rocm" }

// Available reports whether ROCm GPU inference can run on this machine.
// Checks for the ROCm kernel driver (/dev/kfd) and a findable llama-server binary.
func (b *rocmBackend) Available() bool {
	if r := core.Stat("/dev/kfd"); !r.OK {
		return false
	}
	if _, err := findLlamaServer(); err != nil {
		return false
	}
	return true
}

// LoadModel loads a GGUF model onto the AMD GPU via llama-server.
// Model architecture is read from GGUF metadata (replacing filename-based guessing).
// If no context length is specified, use the model native context window. When
// metadata omits the native context, fall back to 4096.
func (b *rocmBackend) LoadModel(path string, opts ...inference.LoadOption) (
	inference.TextModel,
	error,
) {
	loadConfig := inference.ApplyLoadOpts(opts)

	binary, err := findLlamaServer()
	if err != nil {
		return nil, err
	}

	metadata, err := gguf.ReadMetadata(path)
	if err != nil {
		return nil, core.E("rocm.LoadModel", "read model metadata", err)
	}

	contextLength := resolveContextLength(loadConfig.ContextLen, metadata)

	modelServer, err := startServer(serverStartConfig{
		BinaryPath:        binary,
		ModelPath:         path,
		GPULayerCount:     loadConfig.GPULayers,
		ContextSize:       contextLength,
		ParallelSlotCount: loadConfig.ParallelSlots,
	})
	if err != nil {
		return nil, err
	}

	return &rocmModel{
		server:        modelServer,
		modelPath:     path,
		modelType:     metadata.Architecture,
		modelInfo:     modelInfoFromMetadata(metadata),
		contextLength: contextLength,
	}, nil
}

func resolveContextLength(requestedContextLength int, metadata gguf.Metadata) int {
	if requestedContextLength > 0 {
		return requestedContextLength
	}
	if metadata.ContextLength == 0 {
		return defaultContextLengthCap
	}
	return int(metadata.ContextLength)
}

func modelInfoFromMetadata(metadata gguf.Metadata) inference.ModelInfo {
	quantBits, quantGroup := quantisationFromFileType(metadata.FileType)
	return inference.ModelInfo{
		Architecture: metadata.Architecture,
		NumLayers:    int(metadata.BlockCount),
		QuantBits:    quantBits,
		QuantGroup:   quantGroup,
	}
}

func quantisationFromFileType(fileType uint32) (bits, groupSize int) {
	fileTypeName := gguf.FileTypeName(fileType)

	switch {
	case core.HasPrefix(fileTypeName, "Q4_"):
		return 4, 32
	case core.HasPrefix(fileTypeName, "Q5_"):
		return 5, 32
	case core.HasPrefix(fileTypeName, "Q8_"):
		return 8, 32
	case core.HasPrefix(fileTypeName, "Q2_"):
		return 2, 16
	case core.HasPrefix(fileTypeName, "Q3_"):
		return 3, 32
	case core.HasPrefix(fileTypeName, "Q6_"):
		return 6, 64
	case fileTypeName == "F16":
		return 16, 0
	case fileTypeName == "F32":
		return 32, 0
	default:
		return 0, 0
	}
}
