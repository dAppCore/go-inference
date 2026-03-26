package inference

import "dappco.re/go/core"

// LoRAConfig specifies LoRA adapter parameters for fine-tuning.
type LoRAConfig struct {
	Rank       int      // Decomposition rank (default 8)
	Alpha      float32  // Scaling factor (default 16)
	TargetKeys []string // Projection layer suffixes to target (default: q_proj, v_proj)
	BFloat16   bool     // Use BFloat16 for adapter weights (mixed precision)
}

// DefaultLoRAConfig returns standard LoRA parameters for LLM fine-tuning.
func DefaultLoRAConfig() LoRAConfig {
	return LoRAConfig{
		Rank:       8,
		Alpha:      16,
		TargetKeys: []string{"q_proj", "v_proj"},
	}
}

// Adapter holds trainable LoRA parameters applied to a model.
// The concrete type is backend-specific (e.g. *metal.LoRAAdapter for go-mlx).
type Adapter interface {
	// TotalParams returns the total number of trainable parameters.
	TotalParams() int

	// Save writes adapter weights to a safetensors file.
	Save(path string) error
}

// TrainableModel extends TextModel with LoRA fine-tuning capabilities.
//
// Use type assertion to check if a loaded model supports training:
//
//	tm, ok := model.(inference.TrainableModel)
//
// Backend-specific training operations (optimisers, gradient computation,
// tensor creation) are provided by the backend package directly
// (e.g. go-mlx for Metal, go-rocm for AMD).
type TrainableModel interface {
	TextModel

	// ApplyLoRA injects LoRA adapters into target projection layers.
	// Returns an Adapter that holds references to all trainable parameters.
	ApplyLoRA(cfg LoRAConfig) Adapter

	// Encode tokenises text into token IDs using the model's tokeniser.
	Encode(text string) []int32

	// Decode converts token IDs back to text.
	Decode(ids []int32) string

	// NumLayers returns the number of transformer layers.
	NumLayers() int
}

// LoadTrainable loads a model that supports training.
// Returns an error if the backend does not support fine-tuning.
func LoadTrainable(path string, opts ...LoadOption) (TrainableModel, error) {
	model, err := LoadModel(path, opts...)
	if err != nil {
		return nil, err
	}
	tm, ok := model.(TrainableModel)
	if !ok {
		model.Close()
		return nil, core.NewError(core.Sprintf("inference: backend %q does not support training", model.ModelType()))
	}
	return tm, nil
}
