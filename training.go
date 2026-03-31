package inference

import "fmt"

// inference.LoRAConfig{Rank: 16, Alpha: 32, TargetKeys: []string{"q_proj", "k_proj", "v_proj"}}
// inference.LoRAConfig{Rank: 8, Alpha: 16, BFloat16: true} // mixed-precision adapter
type LoRAConfig struct {
	Rank       int      // Decomposition rank (default 8)
	Alpha      float32  // Scaling factor (default 16)
	TargetKeys []string // Projection layer suffixes to target (default: q_proj, v_proj)
	BFloat16   bool     // Use BFloat16 for adapter weights (mixed precision)
}

// config := inference.DefaultLoRAConfig() // Rank=8, Alpha=16, TargetKeys=["q_proj","v_proj"]
func DefaultLoRAConfig() LoRAConfig {
	return LoRAConfig{
		Rank:       8,
		Alpha:      16,
		TargetKeys: []string{"q_proj", "v_proj"},
	}
}

// The concrete type is backend-specific (e.g. *metal.LoRAAdapter for go-mlx).
//
//	adapter := trainableModel.ApplyLoRA(inference.DefaultLoRAConfig())
//	fmt.Printf("%d trainable parameters\n", adapter.TotalParams())
//	adapter.Save("/models/lora/domain-v2/adapter.safetensors")
type Adapter interface {
	// TotalParams is the sum of all injected adapter weight elements.
	//
	//	fmt.Printf("%d trainable params\n", adapter.TotalParams())
	TotalParams() int

	// Save persists adapter weights to a safetensors file at path.
	//
	//	adapter.Save("/models/lora/epoch3/adapter.safetensors")
	Save(path string) error
}

// Use type assertion to check if a loaded model supports training:
//
//	trainableModel, ok := model.(inference.TrainableModel)
//	if !ok { return fmt.Errorf("backend does not support training") }
//
// Backend-specific training operations (optimisers, gradient computation,
// tensor creation) are provided by the backend package directly
// (e.g. go-mlx for Metal, go-rocm for AMD).
type TrainableModel interface {
	TextModel

	// ApplyLoRA injects LoRA adapters into target projection layers.
	//
	//	adapter := trainableModel.ApplyLoRA(inference.LoRAConfig{Rank: 16, Alpha: 32})
	//	fmt.Printf("%d trainable params\n", adapter.TotalParams())
	ApplyLoRA(config LoRAConfig) Adapter

	// Encode tokenises text into token IDs using the model's tokeniser.
	//
	//	ids := trainableModel.Encode("Hello, world!") // [1, 22172, 29892, 3186, 29991]
	Encode(text string) []int32

	// Decode converts token IDs back to text.
	//
	//	text := trainableModel.Decode([]int32{1, 22172, 29892}) // "Hello,"
	Decode(ids []int32) string

	// NumLayers is the transformer depth used to size per-layer LoRA matrices.
	//
	//	layers := trainableModel.NumLayers() // e.g. 26 for gemma3-1b
	NumLayers() int
}

// trainableModel, err := inference.LoadTrainable("/models/gemma3-1b")
// adapter := trainableModel.ApplyLoRA(inference.DefaultLoRAConfig())
func LoadTrainable(path string, options ...LoadOption) (TrainableModel, error) {
	loadedModel, err := LoadModel(path, options...)
	if err != nil {
		return nil, err
	}
	trainableModel, ok := loadedModel.(TrainableModel)
	if !ok {
		loadedModel.Close()
		return nil, fmt.Errorf("inference: backend %q does not support training", loadedModel.ModelType())
	}
	return trainableModel, nil
}
