package inference

import (
	core "dappco.re/go"
)

// inference.LoRAConfig{Rank: 16, Alpha: 32, TargetKeys: []string{"q_proj", "k_proj", "v_proj"}}
// inference.LoRAConfig{Rank: 8, Alpha: 16, BFloat16: true} // mixed-precision adapter
type LoRAConfig struct {
	Rank       int      // Decomposition rank (default 8)
	Alpha      float32  // Scaling factor (default 16)
	TargetKeys []string // Projection layer suffixes to target (default: q_proj, v_proj)
	BFloat16   bool     // Use BFloat16 for adapter weights (mixed precision)
}

// cfg := inference.DefaultLoRAConfig() // Rank=8, Alpha=16, TargetKeys=["q_proj","v_proj"]
func DefaultLoRAConfig() LoRAConfig {
	return LoRAConfig{
		Rank:       8,
		Alpha:      16,
		TargetKeys: []string{"q_proj", "v_proj"},
	}
}

// The concrete type is backend-specific (e.g. *metal.LoRAAdapter for go-mlx).
//
//	adapter := tm.ApplyLoRA(inference.DefaultLoRAConfig())
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
//	tm, ok := model.(inference.TrainableModel)
//	if !ok { return errors.New("backend does not support training") }
//
// Backend-specific training operations (optimisers, gradient computation,
// tensor creation) are provided by the backend package directly
// (e.g. go-mlx for Metal, go-rocm for AMD).
type TrainableModel interface {
	TextModel

	// ApplyLoRA injects LoRA adapters into target projection layers.
	//
	//	adapter := tm.ApplyLoRA(inference.LoRAConfig{Rank: 16, Alpha: 32})
	//	fmt.Printf("%d trainable params\n", adapter.TotalParams())
	ApplyLoRA(cfg LoRAConfig) Adapter

	// Encode tokenises text into token IDs using the model's tokeniser.
	//
	//	ids := tm.Encode("Hello, world!") // [1, 22172, 29892, 3186, 29991]
	Encode(text string) []int32

	// Decode converts token IDs back to text.
	//
	//	text := tm.Decode([]int32{1, 22172, 29892}) // "Hello,"
	Decode(ids []int32) string

	// NumLayers is the transformer depth used to size per-layer LoRA matrices.
	//
	//	layers := tm.NumLayers() // e.g. 26 for gemma3-1b
	NumLayers() int
}

// r := inference.LoadTrainable("/models/gemma3-1b")
// adapter := r.Value.(inference.TrainableModel).ApplyLoRA(inference.DefaultLoRAConfig())
func LoadTrainable(path string, opts ...LoadOption) core.Result {
	modelResult := LoadModel(path, opts...)
	if !modelResult.OK {
		return modelResult
	}
	model, ok := modelResult.Value.(TextModel)
	if !ok {
		return core.Fail(core.E("inference.LoadTrainable", "load returned a non-model value", nil))
	}
	if model == nil {
		return core.Fail(core.E("inference.LoadTrainable", "load returned a nil model", nil))
	}
	modelType := model.ModelType()
	tm, ok := model.(TrainableModel)
	if !ok {
		closeResult := core.ResultOf(nil, model.Close())
		if !closeResult.OK {
			return core.Fail(core.Wrap(closeResult.Value.(error), "inference.LoadTrainable", "close non-trainable model"))
		}
		return core.Fail(core.E("inference.LoadTrainable", "backend "+core.Sprintf("%q", modelType)+" does not support training", nil))
	}
	return core.Ok(tm)
}
