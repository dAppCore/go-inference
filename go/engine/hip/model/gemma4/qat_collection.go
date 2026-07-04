// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"strings"

	"dappco.re/go/inference"
)

const (
	QATCollectionID     = "mlx-community/gemma-4-qat"
	MTPQATCollectionID  = "mlx-community/gemma-4-mtp-qat"
	QATCollectionURL    = "https://huggingface.co/collections/mlx-community/gemma-4-qat"
	MTPQATCollectionURL = "https://huggingface.co/collections/mlx-community/gemma-4-mtp-qat"
)

type QATCollectionEntry struct {
	CollectionID   string
	CollectionURL  string
	ModelID        string
	Size           string
	QuantMode      string
	QuantSuffix    string
	Bits           int
	QuantGroup     int
	Assistant      bool
	Runtime        string
	GenerateStatus string
	RunnableOnCard bool
}

var qatCollectionSizes = []string{"E2B", "E4B", "26B-A4B", "31B", "12B"}

var qatCollectionQuantSuffixes = []struct {
	mode   string
	suffix string
}{
	{mode: "q4", suffix: "4bit"},
	{mode: "q5", suffix: "5bit"},
	{mode: "q6", suffix: "6bit"},
	{mode: "q8", suffix: "8bit"},
	{mode: "bf16", suffix: "bf16"},
	{mode: "mxfp4", suffix: "mxfp4"},
	{mode: "nvfp4", suffix: "nvfp4"},
	{mode: "mxfp8", suffix: "mxfp8"},
}

func DefaultQATTargetCollection() []QATCollectionEntry {
	return defaultQATCollection(false)
}

func DefaultMTPQATCollection() []QATCollectionEntry {
	return defaultQATCollection(true)
}

func QATCollectionEntryForModelID(modelID string) (QATCollectionEntry, bool) {
	normalized := strings.ToLower(strings.TrimSpace(modelID))
	if normalized == "" {
		return QATCollectionEntry{}, false
	}
	assistant := strings.Contains(normalized, "-it-qat-assistant-")
	target := strings.Contains(normalized, "-it-qat-") && !assistant
	if !target && !assistant {
		return QATCollectionEntry{}, false
	}
	size := ModelPackSize(inference.ModelIdentity{}, modelID)
	rawMode := PathQuantMode(modelID)
	if size == "" || rawMode == "" {
		return QATCollectionEntry{}, false
	}
	mode := rawMode
	if !assistant {
		mode = NormalizeSizeQuantMode(size, mode)
	}
	return QATCollectionEntryFor(size, mode, assistant)
}

func QATCollectionEntryFor(size, mode string, assistant bool) (QATCollectionEntry, bool) {
	size = CanonicalSize(size)
	mode = strings.ToLower(strings.TrimSpace(mode))
	if size == "" || mode == "" {
		return QATCollectionEntry{}, false
	}
	rawMode := denormalizeStatusQuantMode(mode)
	suffix, ok := qatQuantSuffix(rawMode)
	if !ok {
		return QATCollectionEntry{}, false
	}
	var support QuantModeSupport
	if assistant {
		support, ok = MTPAssistantQuantModeSupport(size, rawMode)
	} else {
		support, ok = QATTargetQuantModeSupport(size, mode)
	}
	if !ok {
		return QATCollectionEntry{}, false
	}
	sizeSupport, ok := SizeQuantSupportBySize(size)
	if !ok {
		return QATCollectionEntry{}, false
	}
	collectionID := QATCollectionID
	collectionURL := QATCollectionURL
	if assistant {
		collectionID = MTPQATCollectionID
		collectionURL = MTPQATCollectionURL
	}
	return QATCollectionEntry{
		CollectionID:   collectionID,
		CollectionURL:  collectionURL,
		ModelID:        QATCollectionModelID(size, rawMode, assistant),
		Size:           size,
		QuantMode:      support.Mode,
		QuantSuffix:    suffix,
		Bits:           quantModeBits(rawMode),
		QuantGroup:     quantModeGroup(rawMode),
		Assistant:      assistant,
		Runtime:        support.Runtime,
		GenerateStatus: support.GenerateStatus,
		RunnableOnCard: assistant || sizeSupport.RunnableOnCard,
	}, true
}

func QATTargetQuantModeSupport(size, mode string) (QuantModeSupport, bool) {
	size = CanonicalSize(size)
	mode = strings.ToLower(strings.TrimSpace(mode))
	if size == "" || mode == "" {
		return QuantModeSupport{}, false
	}
	rawMode := denormalizeStatusQuantMode(mode)
	if _, ok := qatQuantSuffix(rawMode); !ok {
		return QuantModeSupport{}, false
	}
	if size == "26B-A4B" || size == "31B" {
		return QuantModeSupport{
			Mode:           NormalizeSizeQuantMode(size, rawMode),
			Runtime:        RuntimePlanned,
			GenerateStatus: GeneratePlannedOnly,
			Notes:          "recognized Gemma-4 QAT collection pack; too large for this card",
		}, true
	}
	switch rawMode {
	case "bf16":
		return QuantModeSupport{Mode: rawMode, Runtime: RuntimeBF16, GenerateStatus: GenerateLoadOnly, Notes: "Gemma-4 QAT BF16 correctness anchor"}, true
	case "q8", "q6", "q4":
		return QuantModeSupport{Mode: rawMode, Runtime: RuntimeMLXAffine, GenerateStatus: GenerateLinked, Notes: "Gemma-4 QAT MLX-affine generate path"}, true
	case "q5", "mxfp8", "mxfp4", "nvfp4":
		return QuantModeSupport{Mode: rawMode, Runtime: RuntimePlanned, GenerateStatus: GeneratePlannedOnly, Notes: "Gemma-4 QAT collection pack recognized; native generate is not promoted"}, true
	default:
		return QuantModeSupport{}, false
	}
}

func QATCollectionModelID(size, mode string, assistant bool) string {
	size = CanonicalSize(size)
	if size == "" {
		size = "E2B"
	}
	mode = denormalizeStatusQuantMode(strings.ToLower(strings.TrimSpace(mode)))
	suffix, ok := qatQuantSuffix(mode)
	if !ok {
		suffix = "6bit"
	}
	if assistant {
		return "mlx-community/gemma-4-" + size + "-it-qat-assistant-" + suffix
	}
	return "mlx-community/gemma-4-" + size + "-it-qat-" + suffix
}

func DenormalizedQuantModeForCollection(mode string) string {
	return denormalizeStatusQuantMode(mode)
}

func defaultQATCollection(assistant bool) []QATCollectionEntry {
	out := make([]QATCollectionEntry, 0, len(qatCollectionSizes)*len(qatCollectionQuantSuffixes))
	for _, size := range qatCollectionSizes {
		for _, quant := range qatCollectionQuantSuffixes {
			mode := quant.mode
			if !assistant {
				mode = NormalizeSizeQuantMode(size, mode)
			}
			entry, ok := QATCollectionEntryFor(size, mode, assistant)
			if ok {
				out = append(out, entry)
			}
		}
	}
	return out
}

func qatQuantSuffix(mode string) (string, bool) {
	mode = denormalizeStatusQuantMode(strings.ToLower(strings.TrimSpace(mode)))
	for _, quant := range qatCollectionQuantSuffixes {
		if quant.mode == mode {
			return quant.suffix, true
		}
	}
	return "", false
}

func denormalizeStatusQuantMode(mode string) string {
	return strings.TrimSuffix(strings.ToLower(strings.TrimSpace(mode)), "-status")
}

func quantModeBits(mode string) int {
	switch denormalizeStatusQuantMode(mode) {
	case "bf16":
		return 16
	case "mxfp8", "q8":
		return 8
	case "q6":
		return 6
	case "q5":
		return 5
	case "mxfp4", "nvfp4", "q4":
		return 4
	default:
		return 0
	}
}

func quantModeGroup(mode string) int {
	switch denormalizeStatusQuantMode(mode) {
	case "mxfp8", "mxfp4", "nvfp4":
		return 32
	case "q8", "q6", "q5", "q4":
		return 64
	default:
		return 0
	}
}
