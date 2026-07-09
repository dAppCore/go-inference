// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"slices"
	"strconv"
	"strings"

	"dappco.re/go/inference"
)

const (
	ProductionLaneModelID                   = "mlx-community/gemma-4-e2b-it-6bit"
	ProductionLaneArchivedBaselineModelID   = "mlx-community/gemma-4-e2b-it-4bit"
	ProductionLaneCurrentQualityModelID     = "lmstudio-community/gemma-4-E2B-it-MLX-8bit"
	ProductionLaneCurrentModelID            = "lmstudio-community/gemma-4-E2B-it-MLX-6bit"
	ProductionLaneCurrentConstrainedModelID = "lmstudio-community/gemma-4-E2B-it-MLX-4bit"
	ProductionLaneQualityQuantBits          = 8
	ProductionLaneProductDefaultQuantBits   = 6
	ProductionLaneConstrainedQuantBits      = 4
	ProductionLaneLongContextLength         = 32768
	ProductionActiveParameterEstimate       = 2300000000

	productionQuantizationGiB = 1024 * 1024 * 1024
)

// ProductionQuantizationPackSupport is the Gemma-4 pack matrix used by ROCm
// inspection, quant-loader routing, benchmark selection, and app defaults.
type ProductionQuantizationPackSupport struct {
	Name             string
	Size             string
	ModelID          string
	LockedModelID    string
	SourceCollection string
	Bits             int
	QuantMode        string
	QuantGroup       int
	Runtime          string
	GenerateStatus   string
	ProductRole      string
	Supported        bool
	RunnableOnCard   bool
	RequiresBench    bool
	RequiresNative   bool
}

type ProductionQuantizationTier struct {
	Name                          string
	ModelID                       string
	Bits                          int
	QuantMode                     string
	QuantGroup                    int
	ProductDefault                bool
	QualityFirst                  bool
	ConstrainedOnly               bool
	ArchivedControl               bool
	StepDownToBits                int
	ActiveWeightReadBytesPerToken uint64
	MinimumWorkingSetBytes        uint64
	LongContextWorkingSetBytes    uint64
}

type ProductionQuantizationSelectionInput struct {
	Device              inference.MachineDeviceInfo
	ContextLength       int
	QualityFirst        bool
	ConstrainedFallback bool
}

type ProductionQuantizationChoice struct {
	Tier                       ProductionQuantizationTier
	Fits                       bool
	RequestedBits              int
	WorkingSetBytes            uint64
	RequiredWorkingSet         uint64
	LongContextSelection       bool
	StepDownFromBits           int
	StepDownWorkingSetBytes    uint64
	StepDownRequiredWorkingSet uint64
	Reason                     string
}

var productionQuantizationPackSupport = []ProductionQuantizationPackSupport{
	{Name: "mxfp4", Size: "E2B", ModelID: "mlx-community/gemma-4-e2b-it-mxfp4", Bits: 4, QuantMode: "mxfp4", QuantGroup: 32, Runtime: RuntimePlanned, GenerateStatus: GeneratePlannedOnly, ProductRole: "research", Supported: true, RunnableOnCard: true, RequiresBench: true},
	{Name: "mxfp8", Size: "E2B", ModelID: "mlx-community/gemma-4-e2b-it-mxfp8", Bits: 8, QuantMode: "mxfp8", QuantGroup: 32, Runtime: RuntimePlanned, GenerateStatus: GeneratePlannedOnly, ProductRole: "research", Supported: true, RunnableOnCard: true, RequiresBench: true},
	{Name: "4bit", Size: "E2B", ModelID: ProductionLaneCurrentConstrainedModelID, LockedModelID: ProductionLaneArchivedBaselineModelID, Bits: ProductionLaneConstrainedQuantBits, QuantMode: "affine", QuantGroup: 64, Runtime: RuntimeMLXAffine, GenerateStatus: GenerateLinked, ProductRole: "constrained", Supported: true, RunnableOnCard: true},
	{Name: "6bit", Size: "E2B", ModelID: ProductionLaneCurrentModelID, LockedModelID: ProductionLaneModelID, Bits: ProductionLaneProductDefaultQuantBits, QuantMode: "affine", QuantGroup: 64, Runtime: RuntimeMLXAffine, GenerateStatus: GenerateLinked, ProductRole: "default", Supported: true, RunnableOnCard: true},
	{Name: "8bit", Size: "E2B", ModelID: ProductionLaneCurrentQualityModelID, LockedModelID: "mlx-community/gemma-4-e2b-it-8bit", Bits: ProductionLaneQualityQuantBits, QuantMode: "affine", QuantGroup: 64, Runtime: RuntimeMLXAffine, GenerateStatus: GenerateLinked, ProductRole: "quality", Supported: true, RunnableOnCard: true},
	{Name: "bf16", Size: "E2B", ModelID: "mlx-community/gemma-4-e2b-it-bf16", Bits: 16, QuantMode: "bf16", Runtime: RuntimeBF16, GenerateStatus: GenerateLoadOnly, ProductRole: "quality-control", Supported: true, RunnableOnCard: true, RequiresBench: true, RequiresNative: true},
	{Name: "e4b-bf16", Size: "E4B", ModelID: "mlx-community/gemma-4-e4b-it-bf16", Bits: 16, QuantMode: "bf16", Runtime: RuntimeBF16, GenerateStatus: GenerateLoadOnly, ProductRole: "quality-control", Supported: true, RunnableOnCard: true, RequiresBench: true, RequiresNative: true},
	{Name: "e4b-mxfp8", Size: "E4B", ModelID: "mlx-community/gemma-4-e4b-it-mxfp8", Bits: 8, QuantMode: "mxfp8", QuantGroup: 32, Runtime: RuntimePlanned, GenerateStatus: GeneratePlannedOnly, ProductRole: "research", Supported: true, RunnableOnCard: true, RequiresBench: true},
	{Name: "e4b-mxfp4", Size: "E4B", ModelID: "mlx-community/gemma-4-e4b-it-mxfp4", Bits: 4, QuantMode: "mxfp4", QuantGroup: 32, Runtime: RuntimePlanned, GenerateStatus: GeneratePlannedOnly, ProductRole: "research", Supported: true, RunnableOnCard: true, RequiresBench: true},
	{Name: "e4b-8bit", Size: "E4B", ModelID: "lmstudio-community/gemma-4-E4B-it-MLX-8bit", Bits: 8, QuantMode: "affine", QuantGroup: 64, Runtime: RuntimeMLXAffine, GenerateStatus: GenerateLinked, ProductRole: "quality", Supported: true, RunnableOnCard: true, RequiresBench: true},
	{Name: "e4b-6bit", Size: "E4B", ModelID: "lmstudio-community/gemma-4-E4B-it-MLX-6bit", Bits: 6, QuantMode: "affine", QuantGroup: 64, Runtime: RuntimeMLXAffine, GenerateStatus: GenerateLinked, ProductRole: "default", Supported: true, RunnableOnCard: true, RequiresBench: true},
	{Name: "e4b-4bit", Size: "E4B", ModelID: "lmstudio-community/gemma-4-E4B-it-MLX-4bit", Bits: 4, QuantMode: "affine", QuantGroup: 64, Runtime: RuntimeMLXAffine, GenerateStatus: GenerateLinked, ProductRole: "constrained", Supported: true, RunnableOnCard: true, RequiresBench: true},
	{Name: "12b-6bit", Size: "12B", ModelID: "mlx-community/gemma-4-12b-it-6bit", Bits: 6, QuantMode: "affine", QuantGroup: 64, Runtime: RuntimeMLXAffine, GenerateStatus: GenerateLinked, ProductRole: "largest-local-target", Supported: true, RunnableOnCard: true, RequiresBench: true},
	{Name: "12b-qat-4bit", Size: "12B", ModelID: "mlx-community/gemma-4-12B-it-qat-4bit", SourceCollection: QATCollectionID, Bits: 4, QuantMode: "affine", QuantGroup: 64, Runtime: RuntimeMLXAffine, GenerateStatus: GenerateLinked, ProductRole: "constrained", Supported: true, RunnableOnCard: true, RequiresBench: true},
	{Name: "26b-a4b-8bit", Size: "26B-A4B", ModelID: "lmstudio-community/gemma-4-26B-A4B-it-MLX-8bit", Bits: 8, QuantMode: "q8-status", Runtime: RuntimePlanned, GenerateStatus: GeneratePlannedOnly, ProductRole: "status-only", Supported: true},
	{Name: "26b-a4b-6bit", Size: "26B-A4B", ModelID: "lmstudio-community/gemma-4-26B-A4B-it-MLX-6bit", Bits: 6, QuantMode: "q6-status", Runtime: RuntimePlanned, GenerateStatus: GeneratePlannedOnly, ProductRole: "status-only", Supported: true},
	{Name: "26b-a4b-4bit", Size: "26B-A4B", ModelID: "lmstudio-community/gemma-4-26B-A4B-it-MLX-4bit", Bits: 4, QuantMode: "q4-status", Runtime: RuntimePlanned, GenerateStatus: GeneratePlannedOnly, ProductRole: "status-only", Supported: true},
	{Name: "31b-8bit", Size: "31B", ModelID: "lmstudio-community/gemma-4-31B-it-MLX-8bit", Bits: 8, QuantMode: "q8-status", Runtime: RuntimePlanned, GenerateStatus: GeneratePlannedOnly, ProductRole: "status-only", Supported: true},
	{Name: "31b-6bit", Size: "31B", ModelID: "lmstudio-community/gemma-4-31B-it-MLX-6bit", Bits: 6, QuantMode: "q6-status", Runtime: RuntimePlanned, GenerateStatus: GeneratePlannedOnly, ProductRole: "status-only", Supported: true},
	{Name: "31b-4bit", Size: "31B", ModelID: "lmstudio-community/gemma-4-31B-it-MLX-4bit", Bits: 4, QuantMode: "q4-status", Runtime: RuntimePlanned, GenerateStatus: GeneratePlannedOnly, ProductRole: "status-only", Supported: true},
}

var productionQuantizationTiers = []ProductionQuantizationTier{
	{
		Name:                          "quality",
		ModelID:                       ProductionLaneCurrentQualityModelID,
		Bits:                          ProductionLaneQualityQuantBits,
		QuantMode:                     "affine",
		QuantGroup:                    64,
		QualityFirst:                  true,
		StepDownToBits:                ProductionLaneProductDefaultQuantBits,
		ActiveWeightReadBytesPerToken: ProductionQuantizationActiveWeightReadBytes(ProductionLaneQualityQuantBits),
		MinimumWorkingSetBytes:        32 * productionQuantizationGiB,
		LongContextWorkingSetBytes:    64 * productionQuantizationGiB,
	},
	{
		Name:                          "default",
		ModelID:                       ProductionLaneCurrentModelID,
		Bits:                          ProductionLaneProductDefaultQuantBits,
		QuantMode:                     "affine",
		QuantGroup:                    64,
		ProductDefault:                true,
		StepDownToBits:                ProductionLaneConstrainedQuantBits,
		ActiveWeightReadBytesPerToken: ProductionQuantizationActiveWeightReadBytes(ProductionLaneProductDefaultQuantBits),
		MinimumWorkingSetBytes:        16 * productionQuantizationGiB,
		LongContextWorkingSetBytes:    24 * productionQuantizationGiB,
	},
	{
		Name:                          "constrained",
		ModelID:                       ProductionLaneCurrentConstrainedModelID,
		Bits:                          ProductionLaneConstrainedQuantBits,
		QuantMode:                     "affine",
		QuantGroup:                    64,
		ConstrainedOnly:               true,
		ArchivedControl:               true,
		ActiveWeightReadBytesPerToken: ProductionQuantizationActiveWeightReadBytes(ProductionLaneConstrainedQuantBits),
		MinimumWorkingSetBytes:        8 * productionQuantizationGiB,
		LongContextWorkingSetBytes:    12 * productionQuantizationGiB,
	},
}

// DefaultProductionQuantizationPackSupport returns every Gemma-4 pack type the
// runtime recognises for product selection, benchmark selection, or validation.
func DefaultProductionQuantizationPackSupport() []ProductionQuantizationPackSupport {
	return append([]ProductionQuantizationPackSupport(nil), productionQuantizationPackSupport...)
}

func DefaultProductionQuantizationTiers() []ProductionQuantizationTier {
	return append([]ProductionQuantizationTier(nil), productionQuantizationTiers...)
}

func ProductionQuantizationPackByName(name string) (ProductionQuantizationPackSupport, bool) {
	needle := strings.ToLower(strings.TrimSpace(name))
	if needle == "" {
		return ProductionQuantizationPackSupport{}, false
	}
	for _, pack := range productionQuantizationPackSupport {
		if strings.ToLower(pack.Name) == needle || strings.ToLower(pack.ModelID) == needle {
			return pack, true
		}
	}
	if pack, ok := ProductionQuantizationPackAlias(name); ok {
		return pack, true
	}
	return ProductionQuantizationPackSupport{}, false
}

func ProductionQuantizationPacksBySize(size string) []ProductionQuantizationPackSupport {
	needle := strings.ToLower(strings.TrimSpace(size))
	if needle == "" {
		return nil
	}
	var out []ProductionQuantizationPackSupport
	for _, pack := range productionQuantizationPackSupport {
		if strings.ToLower(pack.Size) == needle {
			out = append(out, pack)
		}
	}
	return out
}

func ApplyProductionQuantizationPackSupportLabels(labels map[string]string) {
	if labels == nil {
		return
	}
	sizes := make([]string, 0, 3)
	linked := make([]string, 0, len(productionQuantizationPackSupport))
	loadOnly := make([]string, 0, 2)
	planned := make([]string, 0, 3)
	runnable := 0
	for _, pack := range productionQuantizationPackSupport {
		sizes = appendUniqueString(sizes, pack.Size)
		if pack.RunnableOnCard {
			runnable++
		}
		packName := ProductionQuantizationPackLabelName(pack)
		switch pack.GenerateStatus {
		case GenerateLinked:
			linked = append(linked, packName)
		case GenerateLoadOnly:
			loadOnly = append(loadOnly, packName)
		case GeneratePlannedOnly:
			planned = append(planned, packName)
		}
	}
	labels["production_quant_pack_count"] = strconv.Itoa(len(productionQuantizationPackSupport))
	labels["production_quant_runnable_pack_count"] = strconv.Itoa(runnable)
	labels["production_quant_pack_sizes"] = strings.Join(sizes, ",")
	labels["production_quant_linked_generate_packs"] = strings.Join(linked, ",")
	labels["production_quant_load_only_packs"] = strings.Join(loadOnly, ",")
	labels["production_quant_planned_packs"] = strings.Join(planned, ",")
}

func ProductionQuantizationPackLabelName(pack ProductionQuantizationPackSupport) string {
	mode := pack.QuantMode
	if mode == "affine" && pack.Bits > 0 {
		mode = "q" + strconv.Itoa(pack.Bits)
	}
	if pack.ProductRole == "mtp-assistant" && mode != "" {
		mode = "assistant-" + mode
	}
	if pack.Size == "" {
		return mode
	}
	return pack.Size + ":" + mode
}

func ProductionQuantizationPackAlias(name string) (ProductionQuantizationPackSupport, bool) {
	if entry, ok := QATCollectionEntryForModelID(name); ok {
		return productionQuantizationPackFromQATEntry(entry), true
	}
	if strings.Contains(strings.ToLower(name), "assistant") {
		return ProductionQuantizationAssistantPackForModel(inference.ModelIdentity{
			Architecture: AssistantArchitecture,
			Path:         name,
		})
	}
	model := inference.ModelIdentity{
		Architecture: "gemma4_text",
		Path:         name,
	}
	size := ModelPackSize(model, model.Path)
	mode := ModelPackQuantModeForPath(model, model.Path)
	mode = NormalizeSizeQuantMode(size, mode)
	if productionQuantizationAliasIsGGUF(name) {
		return ProductionQuantizationGGUFPackAlias(name, size, mode)
	}
	if size == "" {
		return ProductionQuantizationPackSupport{}, false
	}
	packs := ProductionQuantizationPacksBySize(size)
	if mode == "" && len(packs) == 1 {
		return packs[0], true
	}
	for _, pack := range packs {
		if ProductionQuantizationPackMode(pack) == mode {
			return pack, true
		}
	}
	return ProductionQuantizationPackSupport{}, false
}

func ProductionQuantizationGGUFPackAlias(name, size, mode string) (ProductionQuantizationPackSupport, bool) {
	if size == "" || mode == "" {
		return ProductionQuantizationPackSupport{}, false
	}
	support, ok := QuantModeSupportBySize(size, mode)
	if !ok {
		return ProductionQuantizationPackSupport{}, false
	}
	sizeSupport, ok := SizeQuantSupportBySize(size)
	if !ok {
		return ProductionQuantizationPackSupport{}, false
	}
	model := ModelWithInferredQuantMode(inference.ModelIdentity{Architecture: "gemma4_text"}, mode)
	return ProductionQuantizationPackSupport{
		Name:           "gguf-" + strings.ToLower(mode),
		Size:           size,
		ModelID:        name,
		Bits:           model.QuantBits,
		QuantMode:      mode,
		QuantGroup:     model.QuantGroup,
		Runtime:        RuntimeGGUF,
		GenerateStatus: GenerateLoadOnly,
		ProductRole:    "load-only",
		Supported:      true,
		RunnableOnCard: sizeSupport.RunnableOnCard && support.GenerateStatus != GeneratePlannedOnly,
	}, true
}

func ProductionQuantizationPackForModel(model inference.ModelIdentity) (ProductionQuantizationPackSupport, bool) {
	if entry, ok := QATCollectionEntryForModelID(firstNonEmptyString(model.Path, model.ID)); ok && !entry.Assistant {
		return productionQuantizationPackFromQATEntry(entry), true
	}
	if IsAssistantArchitecture(model.Architecture) {
		return ProductionQuantizationAssistantPackForModel(model)
	}
	if !IsSizeQuantIdentity(model.Architecture) {
		return ProductionQuantizationPackSupport{}, false
	}
	model = modelWithInferredPathQuant(model)
	size := ModelPackSize(model, model.Path)
	mode := ModelPackQuantModeForPath(model, model.Path)
	mode = NormalizeSizeQuantMode(size, mode)
	if size == "" {
		return ProductionQuantizationPackSupport{}, false
	}
	for _, pack := range productionQuantizationPackSupport {
		if pack.Size != size {
			continue
		}
		if mode != "" {
			if mode == ProductionQuantizationPackMode(pack) {
				return pack, true
			}
			continue
		}
		if bits := modelQuantBits(model); bits > 0 && pack.Bits == bits {
			return pack, true
		}
	}
	return ProductionQuantizationPackSupport{}, false
}

func ProductionQuantizationAssistantPackForModel(model inference.ModelIdentity) (ProductionQuantizationPackSupport, bool) {
	if !IsAssistantArchitecture(model.Architecture) {
		return ProductionQuantizationPackSupport{}, false
	}
	if entry, ok := QATCollectionEntryForModelID(firstNonEmptyString(model.Path, model.ID)); ok && entry.Assistant {
		return productionQuantizationPackFromQATEntry(entry), true
	}
	model = modelWithInferredPathQuant(model)
	size := ModelPackSize(model, model.Path)
	mode := ModelPackQuantModeForPath(model, model.Path)
	if size == "" {
		return ProductionQuantizationPackSupport{}, false
	}
	support, ok := MTPAssistantQuantModeSupport(size, mode)
	if !ok {
		return ProductionQuantizationPackSupport{}, false
	}
	modelID := firstNonEmptyString(model.Path, MTPAssistantPath(size, support.Mode))
	return ProductionQuantizationPackSupport{
		Name:           MTPAssistantPackNameForQuant(size, support.Mode),
		Size:           size,
		ModelID:        modelID,
		Bits:           quantModeBits(support.Mode),
		QuantMode:      productionQuantizationPackQuantMode(support.Mode),
		QuantGroup:     quantModeGroup(support.Mode),
		Runtime:        support.Runtime,
		GenerateStatus: support.GenerateStatus,
		ProductRole:    "mtp-assistant",
		Supported:      true,
		RunnableOnCard: true,
	}, true
}

func ProductionQuantizationPackMode(pack ProductionQuantizationPackSupport) string {
	if pack.QuantMode == "affine" && pack.Bits > 0 {
		return "q" + strconv.Itoa(pack.Bits)
	}
	return pack.QuantMode
}

func ProductionQuantizationPackBySizeRole(size, role string) (ProductionQuantizationPackSupport, bool) {
	for _, pack := range productionQuantizationPackSupport {
		if pack.Size == size && pack.ProductRole == role {
			return pack, true
		}
	}
	return ProductionQuantizationPackSupport{}, false
}

func productionQuantizationPackFromQATEntry(entry QATCollectionEntry) ProductionQuantizationPackSupport {
	return ProductionQuantizationPackSupport{
		Name:             productionQuantizationQATPackName(entry),
		Size:             entry.Size,
		ModelID:          entry.ModelID,
		SourceCollection: entry.CollectionID,
		Bits:             entry.Bits,
		QuantMode:        productionQuantizationPackQuantMode(entry.QuantMode),
		QuantGroup:       entry.QuantGroup,
		Runtime:          entry.Runtime,
		GenerateStatus:   entry.GenerateStatus,
		ProductRole:      productionQuantizationQATProductRole(entry),
		Supported:        true,
		RunnableOnCard:   entry.RunnableOnCard,
		RequiresBench:    !entry.Assistant && entry.GenerateStatus == GenerateLinked,
		RequiresNative:   !entry.Assistant && entry.GenerateStatus == GenerateLoadOnly,
	}
}

func productionQuantizationQATPackName(entry QATCollectionEntry) string {
	name := strings.ToLower(entry.Size) + "-qat-" + entry.QuantSuffix
	if entry.Assistant {
		name = strings.ToLower(entry.Size) + "-qat-assistant-" + entry.QuantSuffix
	}
	return name
}

func productionQuantizationQATProductRole(entry QATCollectionEntry) string {
	if entry.Assistant {
		return "mtp-assistant"
	}
	if !entry.RunnableOnCard {
		return "status-only"
	}
	switch entry.QuantMode {
	case "q8":
		return "quality"
	case "q6":
		if entry.Size == "12B" {
			return "largest-local-target"
		}
		return "default"
	case "q4":
		return "constrained"
	case "bf16":
		return "quality-control"
	default:
		return "research"
	}
}

func productionQuantizationPackQuantMode(mode string) string {
	switch denormalizeStatusQuantMode(mode) {
	case "q8", "q6", "q5", "q4":
		return "affine"
	default:
		return mode
	}
}

func SelectProductionQuantizationTier(input ProductionQuantizationSelectionInput) ProductionQuantizationChoice {
	defaultTier := ProductionQuantizationTierByBits(ProductionLaneProductDefaultQuantBits)
	qualityTier := ProductionQuantizationTierByBits(ProductionLaneQualityQuantBits)
	constrainedTier := ProductionQuantizationTierByBits(ProductionLaneConstrainedQuantBits)
	workingSet := productionQuantizationWorkingSet(input.Device)
	longContext := input.ContextLength >= ProductionLaneLongContextLength
	requestedBits := ProductionLaneProductDefaultQuantBits
	if input.QualityFirst {
		requestedBits = ProductionLaneQualityQuantBits
	}
	if input.ConstrainedFallback {
		return productionQuantizationChoice(constrainedTier, workingSet, longContext, ProductionLaneConstrainedQuantBits, "constrained fallback requested")
	}
	if input.QualityFirst {
		if workingSet == 0 {
			return productionQuantizationStepDownChoice(defaultTier, qualityTier, workingSet, longContext, requestedBits, "quality q8 requires measured memory headroom; using q6 default")
		}
		choice := productionQuantizationChoice(qualityTier, workingSet, longContext, requestedBits, "quality tier selected with sufficient headroom")
		if choice.Fits {
			return choice
		}
		defaultChoice := productionQuantizationStepDownChoice(defaultTier, qualityTier, workingSet, longContext, requestedBits, "quality q8 does not fit requested memory/context; using q6 default")
		if defaultChoice.Fits {
			return defaultChoice
		}
	}
	choice := productionQuantizationChoice(defaultTier, workingSet, longContext, requestedBits, "default q6 tier selected")
	if choice.Fits {
		return choice
	}
	fallback := productionQuantizationStepDownChoice(constrainedTier, defaultTier, workingSet, longContext, requestedBits, "q6 does not fit requested memory/context; using q4 fallback")
	if fallback.Fits {
		return fallback
	}
	fallback.Reason = "q4 is the smallest supported tier but still exceeds the measured working set"
	return fallback
}

func ProductionQuantizationTierByBits(bits int) ProductionQuantizationTier {
	for _, tier := range productionQuantizationTiers {
		if tier.Bits == bits {
			return tier
		}
	}
	return ProductionQuantizationTier{}
}

func ProductionQuantizationActiveWeightReadBytes(bits int) uint64 {
	if bits <= 0 {
		return 0
	}
	return (uint64(ProductionActiveParameterEstimate)*uint64(bits) + 7) / 8
}

func productionQuantizationAliasIsGGUF(name string) bool {
	return strings.Contains(strings.ToLower(strings.TrimSpace(name)), "gguf")
}

func productionQuantizationChoice(tier ProductionQuantizationTier, workingSet uint64, longContext bool, requestedBits int, reason string) ProductionQuantizationChoice {
	required := productionQuantizationRequiredWorkingSet(tier, longContext)
	fits := workingSet == 0 || required == 0 || workingSet >= required
	return ProductionQuantizationChoice{
		Tier:                 tier,
		Fits:                 fits,
		RequestedBits:        requestedBits,
		WorkingSetBytes:      workingSet,
		RequiredWorkingSet:   required,
		LongContextSelection: longContext,
		Reason:               reason,
	}
}

func productionQuantizationStepDownChoice(tier, failedTier ProductionQuantizationTier, workingSet uint64, longContext bool, requestedBits int, reason string) ProductionQuantizationChoice {
	choice := productionQuantizationChoice(tier, workingSet, longContext, requestedBits, reason)
	choice.StepDownFromBits = failedTier.Bits
	choice.StepDownWorkingSetBytes = workingSet
	choice.StepDownRequiredWorkingSet = productionQuantizationRequiredWorkingSet(failedTier, longContext)
	return choice
}

func productionQuantizationRequiredWorkingSet(tier ProductionQuantizationTier, longContext bool) uint64 {
	required := tier.MinimumWorkingSetBytes
	if longContext && tier.LongContextWorkingSetBytes > required {
		required = tier.LongContextWorkingSetBytes
	}
	return required
}

func productionQuantizationWorkingSet(device inference.MachineDeviceInfo) uint64 {
	if device.MaxRecommendedWorkingSetSize > 0 {
		return device.MaxRecommendedWorkingSetSize
	}
	return device.MemorySize
}

func modelWithInferredPathQuant(model inference.ModelIdentity) inference.ModelIdentity {
	mode := ModelPackQuantModeForPath(model, model.Path)
	if mode == "" {
		return model
	}
	return ModelWithInferredQuantMode(model, mode)
}

func modelQuantBits(model inference.ModelIdentity) int {
	if model.QuantBits > 0 {
		return model.QuantBits
	}
	switch ModelPackQuantMode(model) {
	case "bf16":
		return 16
	case "mxfp8", "q8", "q8-status":
		return 8
	case "q6", "q6-status":
		return 6
	case "q5", "q5-status":
		return 5
	case "mxfp4", "nvfp4", "q4", "q4-status":
		return 4
	default:
		return 0
	}
}

func appendUniqueString(values []string, value string) []string {
	if value == "" {
		return values
	}
	if slices.Contains(values, value) {
		return values
	}
	return append(values, value)
}
