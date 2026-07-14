// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import "strings"

const (
	RuntimeMLXAffine    = "mlx_affine"
	RuntimeBF16         = "bf16"
	RuntimeGGUF         = "gguf"
	RuntimePlanned      = "planned_status"
	GenerateLinked      = "linked"
	GenerateLoadOnly    = "load_only"
	GeneratePlannedOnly = "planned_only"
)

// SizeQuantSupport declares the Gemma-4 size/quant support matrix that model
// pack inspection and production quant routing react to.
type SizeQuantSupport struct {
	Size             string
	ModelIDPrefix    string
	Runtime          string
	QuantModes       []string
	QuantModeSupport []QuantModeSupport
	RunnableOnCard   bool
	Notes            string
}

type QuantModeSupport struct {
	Mode           string
	Runtime        string
	GenerateStatus string
	Notes          string
}

var sizeQuantMatrix = []SizeQuantSupport{
	{Size: "E2B", ModelIDPrefix: "gemma-4-E2B-it", Runtime: RuntimeMLXAffine, QuantModes: []string{"bf16", "q8", "q6", "q4", "mxfp8", "mxfp4"}, QuantModeSupport: smallQuantModeSupport(), RunnableOnCard: true, Notes: "primary production size"},
	{Size: "E4B", ModelIDPrefix: "gemma-4-E4B-it", Runtime: RuntimeMLXAffine, QuantModes: []string{"bf16", "q8", "q6", "q4", "mxfp8", "mxfp4"}, QuantModeSupport: smallQuantModeSupport(), RunnableOnCard: true, Notes: "same quant ladder as E2B"},
	{Size: "12B", ModelIDPrefix: "gemma-4-12B-it", Runtime: RuntimeMLXAffine, QuantModes: []string{"q6", "q4"}, QuantModeSupport: []QuantModeSupport{{Mode: "q6", Runtime: RuntimeMLXAffine, GenerateStatus: GenerateLinked, Notes: "q6 target on this card"}, {Mode: "q4", Runtime: RuntimeMLXAffine, GenerateStatus: GenerateLinked, Notes: "QAT constrained 12B target on this card"}}, RunnableOnCard: true, Notes: "q6 and QAT q4 targets on this card"},
	{Size: "26B-A4B", ModelIDPrefix: "gemma-4-26B-A4B-it", Runtime: RuntimePlanned, QuantModes: []string{"q8-status", "q6-status", "q4-status"}, QuantModeSupport: largeStatusQuantModeSupport(), RunnableOnCard: false, Notes: "too large for this RX 7800 XT target"},
	{Size: "31B", ModelIDPrefix: "gemma-4-31B-it", Runtime: RuntimePlanned, QuantModes: []string{"q8-status", "q6-status", "q4-status"}, QuantModeSupport: largeStatusQuantModeSupport(), RunnableOnCard: false, Notes: "too large for this RX 7800 XT target"},
}

func DefaultSizeQuantSupport() []SizeQuantSupport {
	out := make([]SizeQuantSupport, len(sizeQuantMatrix))
	for i, entry := range sizeQuantMatrix {
		out[i] = CloneSizeQuantSupport(entry)
	}
	return out
}

func SizeQuantSupportBySize(size string) (SizeQuantSupport, bool) {
	needle := strings.ToLower(strings.TrimSpace(size))
	for _, entry := range sizeQuantMatrix {
		if strings.ToLower(entry.Size) == needle {
			return CloneSizeQuantSupport(entry), true
		}
	}
	return SizeQuantSupport{}, false
}

func CanonicalSize(size string) string {
	size = strings.TrimSpace(size)
	if size == "" {
		return ""
	}
	if entry, ok := SizeQuantSupportBySize(size); ok {
		return entry.Size
	}
	return size
}

func QuantModeSupportBySize(size, mode string) (QuantModeSupport, bool) {
	entry, ok := SizeQuantSupportBySize(size)
	if !ok {
		return QuantModeSupport{}, false
	}
	needle := strings.ToLower(strings.TrimSpace(mode))
	for _, quant := range entry.QuantModeSupport {
		if strings.ToLower(quant.Mode) == needle {
			return quant, true
		}
	}
	return QuantModeSupport{}, false
}

func CloneSizeQuantSupport(entry SizeQuantSupport) SizeQuantSupport {
	entry.QuantModes = append([]string(nil), entry.QuantModes...)
	entry.QuantModeSupport = append([]QuantModeSupport(nil), entry.QuantModeSupport...)
	return entry
}

func smallQuantModeSupport() []QuantModeSupport {
	return []QuantModeSupport{
		{Mode: "bf16", Runtime: RuntimeBF16, GenerateStatus: GenerateLinked, Notes: "native dense BF16 text generation"},
		{Mode: "q8", Runtime: RuntimeMLXAffine, GenerateStatus: GenerateLinked, Notes: "quality MLX-affine generate path"},
		{Mode: "q6", Runtime: RuntimeMLXAffine, GenerateStatus: GenerateLinked, Notes: "production MLX-affine generate path"},
		{Mode: "q4", Runtime: RuntimeMLXAffine, GenerateStatus: GenerateLinked, Notes: "constrained MLX-affine generate path"},
		{Mode: "mxfp8", Runtime: RuntimePlanned, GenerateStatus: GeneratePlannedOnly, Notes: "research pack; native dequant/generate not promoted"},
		{Mode: "mxfp4", Runtime: RuntimePlanned, GenerateStatus: GeneratePlannedOnly, Notes: "research pack; native dequant/generate not promoted"},
	}
}

func largeStatusQuantModeSupport() []QuantModeSupport {
	return []QuantModeSupport{
		{Mode: "q8-status", Runtime: RuntimePlanned, GenerateStatus: GeneratePlannedOnly, Notes: "recognized status-only pack; too large for this RX 7800 XT target"},
		{Mode: "q6-status", Runtime: RuntimePlanned, GenerateStatus: GeneratePlannedOnly, Notes: "recognized status-only pack; too large for this RX 7800 XT target"},
		{Mode: "q4-status", Runtime: RuntimePlanned, GenerateStatus: GeneratePlannedOnly, Notes: "recognized status-only pack; too large for this RX 7800 XT target"},
	}
}
