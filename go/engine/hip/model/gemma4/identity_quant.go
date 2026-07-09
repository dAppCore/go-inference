// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"strconv"
	"strings"

	"dappco.re/go/inference"
	rocmprofile "dappco.re/go/inference/engine/hip/profile"
)

// IsSizeQuantIdentity reports whether architecture belongs to Gemma-4 text or
// attached-assistant identities that can use the size/quant matrix.
func IsSizeQuantIdentity(architecture string) bool {
	switch rocmprofile.Gemma4ArchitectureID(architecture) {
	case "gemma4", "gemma4_text", "gemma4_unified", AssistantArchitecture:
		return true
	default:
		return false
	}
}

func IsAssistantArchitecture(architecture string) bool {
	return rocmprofile.Gemma4ArchitectureID(architecture) == AssistantArchitecture
}

func ModelPackSize(model inference.ModelIdentity, path string) string {
	return modelPackSize(model, path, false)
}

func ModelPackSizeWithGeometry(model inference.ModelIdentity, path string) string {
	return modelPackSize(model, path, true)
}

func modelPackSize(model inference.ModelIdentity, path string, includeGeometry bool) string {
	if model.Labels["gemma4_size"] != "" {
		return CanonicalSize(model.Labels["gemma4_size"])
	}
	normalizedPath := strings.ToLower(strings.ReplaceAll(path, "-", "_"))
	switch {
	case strings.Contains(normalizedPath, "26b") && strings.Contains(normalizedPath, "a4b"):
		return "26B-A4B"
	case strings.Contains(normalizedPath, "31b"):
		return "31B"
	case strings.Contains(normalizedPath, "12b"):
		return "12B"
	case strings.Contains(normalizedPath, "e4b"):
		return "E4B"
	case strings.Contains(normalizedPath, "e2b"):
		return "E2B"
	case includeGeometry && model.NumLayers == 64 && model.HiddenSize == 4096:
		return "31B"
	case includeGeometry && model.NumLayers == 48 && model.HiddenSize == 3840:
		return "12B"
	case includeGeometry && model.NumLayers == 26 && model.HiddenSize == 2304:
		return "E4B"
	case includeGeometry && model.NumLayers == 35 && model.HiddenSize == e2bHiddenSize:
		return "E2B"
	default:
		return ""
	}
}

func NormalizeSizeQuantMode(size, mode string) string {
	normalizedSize := strings.ToLower(strings.TrimSpace(size))
	normalizedMode := strings.ToLower(strings.TrimSpace(mode))
	if normalizedSize == "26b-a4b" || normalizedSize == "31b" {
		switch normalizedMode {
		case "bf16":
			return "bf16-status"
		case "q8":
			return "q8-status"
		case "q6":
			return "q6-status"
		case "q5":
			return "q5-status"
		case "q4":
			return "q4-status"
		}
	}
	return mode
}

func ModelPackQuantMode(model inference.ModelIdentity) string {
	return modelPackQuantMode(model, false)
}

func ModelPackQuantModeWithGeometry(model inference.ModelIdentity) string {
	return modelPackQuantMode(model, true)
}

func modelPackQuantMode(model inference.ModelIdentity, includeGeometry bool) string {
	if model.Labels["gemma4_quant_mode"] != "" {
		if IsAssistantArchitecture(model.Architecture) {
			return denormalizeStatusQuantMode(model.Labels["gemma4_quant_mode"])
		}
		return CanonicalQuantMode(modelPackSize(model, model.Path, includeGeometry), model.Labels["gemma4_quant_mode"])
	}
	quantType := strings.ToLower(strings.TrimSpace(model.QuantType))
	switch {
	case strings.Contains(quantType, "mxfp8"):
		return "mxfp8"
	case strings.Contains(quantType, "mxfp4"):
		return "mxfp4"
	case strings.Contains(quantType, "bf16") || strings.Contains(quantType, "bfloat16") || model.QuantBits == 16:
		return "bf16"
	case model.QuantBits == 5:
		return "q5"
	case model.QuantBits > 0:
		return "q" + strconv.Itoa(model.QuantBits)
	default:
		return ""
	}
}

func ModelPackQuantModeForPath(model inference.ModelIdentity, path string) string {
	return modelPackQuantModeForPath(model, path, false)
}

func ModelPackQuantModeForPathWithGeometry(model inference.ModelIdentity, path string) string {
	return modelPackQuantModeForPath(model, path, true)
}

func modelPackQuantModeForPath(model inference.ModelIdentity, path string, includeGeometry bool) string {
	if model.Labels["gemma4_quant_mode"] != "" {
		if IsAssistantArchitecture(model.Architecture) {
			return denormalizeStatusQuantMode(model.Labels["gemma4_quant_mode"])
		}
		return CanonicalQuantMode(modelPackSize(model, path, includeGeometry), model.Labels["gemma4_quant_mode"])
	}
	pathMode := PathQuantMode(path)
	switch pathMode {
	case "mxfp8", "mxfp4":
		return pathMode
	}
	if mode := modelPackQuantMode(model, includeGeometry); mode != "" {
		return mode
	}
	if pathMode != "" {
		return pathMode
	}
	if IsAssistantArchitecture(model.Architecture) && strings.Contains(strings.ToLower(path), "assistant") {
		return AssistantQuantMode
	}
	return ""
}

func PathQuantMode(path string) string {
	normalized := strings.ToLower(strings.TrimSpace(path))
	switch {
	case normalized == "":
		return ""
	case strings.Contains(normalized, "mxfp8"):
		return "mxfp8"
	case strings.Contains(normalized, "mxfp4"):
		return "mxfp4"
	case strings.Contains(normalized, "nvfp4"):
		return "nvfp4"
	case strings.Contains(normalized, "bf16") || strings.Contains(normalized, "bfloat16"):
		return "bf16"
	case pathHasQuantToken(normalized, "8bit", "8-bit", "8_bit", "q8", "q8_0"):
		return "q8"
	case pathHasQuantToken(normalized, "6bit", "6-bit", "6_bit", "q6"):
		return "q6"
	case pathHasQuantToken(normalized, "5bit", "5-bit", "5_bit", "q5"):
		return "q5"
	case pathHasQuantToken(normalized, "4bit", "4-bit", "4_bit", "q4", "q4_0", "q4_k_m"):
		return "q4"
	default:
		return ""
	}
}

func ModelWithInferredQuantMode(model inference.ModelIdentity, mode string) inference.ModelIdentity {
	if model.QuantType == "" {
		switch mode {
		case "bf16":
			model.QuantType = "bf16"
		case "mxfp8", "mxfp4", "nvfp4":
			model.QuantType = mode
		case "q8", "q8-status":
			model.QuantType = "q8"
		case "q6", "q6-status":
			model.QuantType = "q6"
		case "q5", "q5-status":
			model.QuantType = "q5"
		case "q4", "q4-status":
			model.QuantType = "q4"
		}
	}
	if model.QuantBits <= 0 {
		switch mode {
		case "bf16":
			model.QuantBits = 16
		case "mxfp8", "q8", "q8-status":
			model.QuantBits = 8
		case "q6", "q6-status":
			model.QuantBits = 6
		case "q5", "q5-status":
			model.QuantBits = 5
		case "mxfp4", "nvfp4", "q4", "q4-status":
			model.QuantBits = 4
		}
	}
	if model.QuantGroup <= 0 {
		switch mode {
		case "mxfp8", "mxfp4", "nvfp4":
			model.QuantGroup = 32
		case "q8", "q6", "q5", "q4":
			model.QuantGroup = 64
		}
	}
	return model
}

func CanonicalQuantMode(size, mode string) string {
	mode = NormalizeSizeQuantMode(size, strings.TrimSpace(mode))
	if mode == "" {
		return ""
	}
	if support, ok := QuantModeSupportBySize(size, mode); ok {
		return support.Mode
	}
	return mode
}

func pathHasQuantToken(path string, tokens ...string) bool {
	for _, token := range tokens {
		if strings.Contains(path, token) {
			return true
		}
	}
	return false
}
