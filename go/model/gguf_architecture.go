// SPDX-Licence-Identifier: EUPL-1.2

package model

// GGUFArchitecture maps a checkpoint's HF model_type onto the canonical GGUF
// general.architecture identifier the llama.cpp ecosystem recognises. The two
// naming schemes drift — HF says "gemma3_text", GGUF says "gemma3" — and an
// exported GGUF carrying the HF spelling is refused at load ("unknown model
// architecture"), so emitted identifiers must be the ecosystem's own. Every
// entry here is verified against llama.cpp's compiled-in architecture table;
// an unmapped model_type returns ok=false and the caller decides whether to
// emit the raw name (honest, likely refused) or stop.
//
// The gguf package itself stays arch-free (it writes whatever architecture
// string it is given); this mapping is model-side knowledge.
func GGUFArchitecture(modelType string) (string, bool) {
	switch modelType {
	case "gemma", "gemma_text":
		return "gemma", true
	case "gemma2", "gemma2_text":
		return "gemma2", true
	case "gemma3", "gemma3_text":
		return "gemma3", true
	case "gemma3n", "gemma3n_text":
		return "gemma3n", true
	case "gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text":
		return "gemma4", true
	case "gemma4_assistant", "gemma4_unified_assistant":
		return "gemma4-assistant", true
	case "llama":
		return "llama", true
	case "llama4", "llama4_text":
		return "llama4", true
	case "mistral", "ministral", "ministral3":
		return "mistral", true
	case "qwen2":
		return "qwen2", true
	case "qwen2_moe":
		return "qwen2moe", true
	case "qwen3":
		return "qwen3", true
	case "qwen3_moe":
		return "qwen3moe", true
	case "qwen3_5", "qwen3_5_text":
		return "qwen35", true
	case "gpt2":
		return "gpt2", true
	case "phi3":
		return "phi3", true
	default:
		return "", false
	}
}
