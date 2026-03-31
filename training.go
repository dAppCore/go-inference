package inference

import "fmt"

// config := inference.LoRAConfig{Rank: 16, Alpha: 32, TargetKeys: []string{"q_proj", "k_proj", "v_proj"}}
// config := inference.LoRAConfig{Rank: 8, Alpha: 16, BFloat16: true}
type LoRAConfig struct {
	Rank       int      // LoRA rank (default 8)
	Alpha      float32  // LoRA scaling factor (default 16)
	TargetKeys []string // Projection suffixes to target
	BFloat16   bool     // Use BFloat16 adapter weights
}

// config := inference.DefaultLoRAConfig() // Rank=8, Alpha=16, TargetKeys=["q_proj","v_proj"]
func DefaultLoRAConfig() LoRAConfig {
	return LoRAConfig{
		Rank:       8,
		Alpha:      16,
		TargetKeys: []string{"q_proj", "v_proj"},
	}
}

// adapter := trainableModel.ApplyLoRA(inference.DefaultLoRAConfig())
// fmt.Printf("%d trainable parameters\n", adapter.TotalParams())
// adapter.Save("/models/lora/domain-v2/adapter.safetensors")
type Adapter interface {
	// fmt.Printf("%d trainable params\n", adapter.TotalParams())
	TotalParams() int

	// adapter.Save("/models/lora/epoch3/adapter.safetensors")
	Save(path string) error
}

// trainableModel, ok := model.(inference.TrainableModel)
// if !ok { return fmt.Errorf("backend does not support training") }
type TrainableModel interface {
	TextModel

	// adapter := trainableModel.ApplyLoRA(inference.LoRAConfig{Rank: 16, Alpha: 32})
	// fmt.Printf("%d trainable params\n", adapter.TotalParams())
	ApplyLoRA(config LoRAConfig) Adapter

	// ids := trainableModel.Encode("Hello, world!") // [1, 22172, 29892, 3186, 29991]
	Encode(text string) []int32

	// text := trainableModel.Decode([]int32{1, 22172, 29892}) // "Hello,"
	Decode(ids []int32) string

	// layers := trainableModel.NumLayers() // 26 for gemma3-1b
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
