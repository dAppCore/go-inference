// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"strconv"
	"strings"

	"dappco.re/go/inference"
)

const (
	AssistantArchitecture                   = "gemma4_assistant"
	AssistantQuantMode                      = "bf16"
	AssistantLayerCount                     = 4
	AssistantTokenOrderingVocabSize         = 262144
	AssistantOrderedEmbeddingCentroids      = 2048
	AssistantCentroidIntermediateTopK       = 32
	AssistantOrderedEmbeddingCentroidsLabel = "2048"
	AssistantCentroidIntermediateTopKLabel  = "32"
	AssistantTokenOrderingDType             = "int64"
	AssistantTokenOrderingShape             = "2048x128"
	OfficialE2BTargetModelID                = "google/gemma-4-E2B-it"
	OfficialE2BTargetRevision               = "905e84b50c4d2a365ebde34e685027578e6728db"
	OfficialE2BAssistantModelID             = "google/gemma-4-E2B-it-assistant"
	OfficialE2BAssistantRevision            = "5810c41a67974da9c7bd6f3e6c69d5d13854d9f0"
	OfficialE2BSourceCheckedAt              = "2026-05-31"
	OfficialE2BTargetConfigSHA256           = "1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330"
	OfficialE2BAssistantConfigSHA256        = "7f42f559a6a69ffaeaf6b61a1ece3a562a2ed5ad00b8d30f16917ba5ab1bcbe9"
	e2bHiddenSize                           = 1536
)

// AssistantConfig carries the assistant-shape fields ROCm needs from
// config.json without making the model package depend on a backend config type.
type AssistantConfig struct {
	BackboneHiddenSize       int
	NumCentroids             int
	CentroidIntermediateTopK int
	UseOrderedEmbeddings     bool
	UseOrderedEmbeddingsSet  bool
	NumLayers                int
	VocabSize                int
}

// PairEvidence is the model-owned Gemma-4 target/assistant compatibility
// surface. Backends can fill it from a loaded model, an inspection, or labels.
type PairEvidence struct {
	TargetSize              string
	TargetQuantMode         string
	TargetQuantGroup        int
	TargetRuntime           string
	TargetGenerateStatus    string
	AssistantSize           string
	AssistantQuantMode      string
	AssistantQuantGroup     int
	AssistantRuntime        string
	AssistantGenerateStatus string
}

// MTPAssistantPath returns the Gemma-4 attached drafter model id for the
// target size and assistant quant mode. BF16 keeps the official Google
// assistant id; QAT assistant modes resolve into the mlx-community MTP-QAT
// collection.
func MTPAssistantPath(size, mode string) string {
	size = CanonicalSize(size)
	if size == "" {
		size = "E2B"
	}
	mode = denormalizeStatusQuantMode(strings.ToLower(strings.TrimSpace(mode)))
	if mode == "" || mode == AssistantQuantMode {
		return "google/gemma-4-" + size + "-it-assistant"
	}
	if _, ok := MTPAssistantQuantModeSupport(size, mode); ok {
		return QATCollectionModelID(size, mode, true)
	}
	return "google/gemma-4-" + size + "-it-assistant"
}

func MTPAssistantPackName(size string) string {
	return MTPAssistantPackNameForQuant(size, AssistantQuantMode)
}

func MTPAssistantPackNameForQuant(size, mode string) string {
	size = CanonicalSize(size)
	if size == "" {
		size = "E2B"
	}
	mode = denormalizeStatusQuantMode(strings.ToLower(strings.TrimSpace(mode)))
	if mode == "" {
		mode = AssistantQuantMode
	}
	suffix, ok := qatQuantSuffix(mode)
	if !ok {
		suffix = mode
	}
	return strings.ToLower(size) + "-assistant-" + suffix
}

func MTPAssistantQuantModeSupport(size, mode string) (QuantModeSupport, bool) {
	size = CanonicalSize(size)
	mode = denormalizeStatusQuantMode(strings.ToLower(strings.TrimSpace(mode)))
	if size == "" || mode == "" {
		return QuantModeSupport{}, false
	}
	suffix, ok := qatQuantSuffix(mode)
	if !ok || suffix == "" {
		return QuantModeSupport{}, false
	}
	runtime := RuntimeMLXAffine
	if mode == AssistantQuantMode {
		runtime = RuntimeBF16
	}
	if mode == "q5" || mode == "mxfp8" || mode == "mxfp4" || mode == "nvfp4" {
		runtime = RuntimePlanned
	}
	return QuantModeSupport{
		Mode:           mode,
		Runtime:        runtime,
		GenerateStatus: GenerateLoadOnly,
		Notes:          "Gemma-4 MTP assistant loads as an attached drafter; native attached execution is gated separately",
	}, true
}

func MTPAssistantHiddenSizeForTarget(size string, targetHidden int) int {
	if targetHidden > 0 {
		return targetHidden
	}
	switch CanonicalSize(size) {
	case "E4B":
		return 2304
	case "12B":
		return 3840
	case "26B-A4B", "31B":
		return 4096
	default:
		return e2bHiddenSize
	}
}

func MTPAssistantLabels(size string, labels map[string]string) map[string]string {
	return MTPAssistantLabelsForModel(size, AssistantQuantMode, MTPAssistantPath(size, AssistantQuantMode), labels)
}

func MTPAssistantLabelsForModel(size, mode, modelID string, labels map[string]string) map[string]string {
	out := cloneStringMap(labels)
	if out == nil {
		out = map[string]string{}
	}
	size = CanonicalSize(size)
	if size == "" {
		size = "E2B"
	}
	support, ok := MTPAssistantQuantModeSupport(size, mode)
	if !ok {
		support = QuantModeSupport{
			Mode:           AssistantQuantMode,
			Runtime:        RuntimeBF16,
			GenerateStatus: GenerateLoadOnly,
		}
	}
	mode = support.Mode
	if strings.TrimSpace(modelID) == "" {
		modelID = MTPAssistantPath(size, mode)
	}
	out["gemma4_size"] = size
	out["gemma4_quant_mode"] = mode
	out["gemma4_runtime"] = support.Runtime
	out["gemma4_generate_status"] = support.GenerateStatus
	out["gemma4_pack_supported"] = "true"
	out["gemma4_runnable_on_card"] = "true"
	out["production_quant_size"] = size
	out["production_quant_pack"] = size + ":assistant-" + denormalizeStatusQuantMode(mode)
	out["production_quant_pack_name"] = MTPAssistantPackNameForQuant(size, mode)
	out["production_quant_tier"] = "mtp-assistant"
	out["production_quant_model"] = modelID
	out["production_quant_mode"] = mode
	out["production_quant_bits"] = strconv.Itoa(quantModeBits(mode))
	if group := quantModeGroup(mode); group > 0 {
		out["production_quant_group"] = strconv.Itoa(group)
	}
	out["production_quant_runtime"] = support.Runtime
	out["production_quant_generate_status"] = support.GenerateStatus
	out["production_quant_supported"] = "true"
	out["production_quant_runnable_on_card"] = "true"
	out["production_quant_mtp_assistant"] = "true"
	out["production_quant_assistant_model"] = modelID
	out["production_quant_target_family"] = "gemma4"
	if entry, ok := QATCollectionEntryForModelID(modelID); ok && entry.Assistant {
		out["production_quant_collection"] = entry.CollectionID
	}
	return out
}

func ApplyAssistantConfigLabels(labels map[string]string, cfg AssistantConfig) (map[string]string, bool) {
	if labels == nil {
		labels = map[string]string{}
	}
	if cfg.BackboneHiddenSize > 0 {
		labels["attached_drafter_assistant_backbone_hidden_size"] = strconv.Itoa(cfg.BackboneHiddenSize)
	}
	if cfg.NumCentroids > 0 {
		labels["attached_drafter_assistant_centroids"] = strconv.Itoa(cfg.NumCentroids)
	}
	if cfg.CentroidIntermediateTopK > 0 {
		labels["attached_drafter_assistant_centroid_intermediate_top_k"] = strconv.Itoa(cfg.CentroidIntermediateTopK)
	}
	if cfg.UseOrderedEmbeddingsSet {
		labels["attached_drafter_assistant_ordered_embeddings"] = strconv.FormatBool(cfg.UseOrderedEmbeddings)
	}
	if cfg.NumLayers > 0 {
		labels["attached_drafter_assistant_layer_count"] = strconv.Itoa(cfg.NumLayers)
		labels["attached_drafter_assistant_four_layer_drafter"] = strconv.FormatBool(cfg.NumLayers == AssistantLayerCount)
	}
	if cfg.NumCentroids > 0 && cfg.VocabSize > 0 && cfg.VocabSize%cfg.NumCentroids == 0 {
		labels["attached_drafter_assistant_token_ordering_shape"] = strconv.Itoa(cfg.NumCentroids) + "x" + strconv.Itoa(cfg.VocabSize/cfg.NumCentroids)
	}
	return labels, AssistantConfigContradictsOfficial(cfg, labels)
}

func AssistantConfigContradictsOfficial(cfg AssistantConfig, labels map[string]string) bool {
	if cfg.NumCentroids > 0 && cfg.NumCentroids != AssistantOrderedEmbeddingCentroids {
		return true
	}
	if cfg.CentroidIntermediateTopK > 0 && cfg.CentroidIntermediateTopK != AssistantCentroidIntermediateTopK {
		return true
	}
	if cfg.UseOrderedEmbeddingsSet && !cfg.UseOrderedEmbeddings {
		return true
	}
	if cfg.NumLayers > 0 && cfg.NumLayers != AssistantLayerCount {
		return true
	}
	if labelValue(labels, "attached_drafter_assistant_token_ordering_shape") != "" &&
		labelValue(labels, "attached_drafter_assistant_token_ordering_shape") != AssistantTokenOrderingShape {
		return true
	}
	return false
}

func PairEvidenceFromIdentities(target, assistant inference.ModelIdentity) PairEvidence {
	return PairEvidence{
		TargetSize:              CanonicalSize(target.Labels["gemma4_size"]),
		TargetQuantMode:         strings.ToLower(strings.TrimSpace(target.Labels["gemma4_quant_mode"])),
		TargetQuantGroup:        target.QuantGroup,
		TargetRuntime:           target.Labels["gemma4_runtime"],
		TargetGenerateStatus:    target.Labels["gemma4_generate_status"],
		AssistantSize:           CanonicalSize(assistant.Labels["gemma4_size"]),
		AssistantQuantMode:      strings.ToLower(strings.TrimSpace(assistant.Labels["gemma4_quant_mode"])),
		AssistantQuantGroup:     assistant.QuantGroup,
		AssistantRuntime:        assistant.Labels["gemma4_runtime"],
		AssistantGenerateStatus: assistant.Labels["gemma4_generate_status"],
	}
}

func OfficialPairVerified(target, assistant inference.ModelIdentity) bool {
	return OfficialPairEvidenceVerified(PairEvidenceFromIdentities(target, assistant))
}

func FamilyPairVerified(target, assistant inference.ModelIdentity) bool {
	return FamilyPairEvidenceVerified(PairEvidenceFromIdentities(target, assistant))
}

func OfficialPairEvidenceVerified(evidence PairEvidence) bool {
	return evidence.TargetSize == "E2B" &&
		evidence.TargetQuantMode == "q6" &&
		evidence.TargetQuantGroup == 64 &&
		evidence.TargetRuntime == RuntimeMLXAffine &&
		evidence.TargetGenerateStatus == GenerateLinked &&
		evidence.AssistantSize == "E2B" &&
		evidence.AssistantQuantMode == AssistantQuantMode &&
		evidence.AssistantRuntime == RuntimeBF16 &&
		evidence.AssistantGenerateStatus == GenerateLoadOnly
}

func FamilyPairEvidenceVerified(evidence PairEvidence) bool {
	if evidence.TargetSize == "" || evidence.AssistantSize == "" || evidence.TargetSize != evidence.AssistantSize {
		return false
	}
	if evidence.TargetQuantMode == "" || evidence.TargetQuantGroup <= 0 {
		return false
	}
	if !targetRuntimeCompatibleWithAttachedDrafter(evidence.TargetRuntime) || evidence.TargetGenerateStatus != GenerateLinked {
		return false
	}
	if evidence.AssistantGenerateStatus != GenerateLoadOnly {
		return false
	}
	assistantSupport, ok := MTPAssistantQuantModeSupport(evidence.AssistantSize, evidence.AssistantQuantMode)
	if !ok || assistantSupport.Runtime == RuntimePlanned {
		return false
	}
	if !assistantRuntimeCompatibleWithAttachedDrafter(evidence.AssistantRuntime, evidence.AssistantQuantMode) {
		return false
	}
	if !assistantQuantModeCompatibleWithTarget(evidence.TargetQuantMode, evidence.AssistantQuantMode) {
		return false
	}
	targetSupport, ok := QuantModeSupportBySize(evidence.TargetSize, evidence.TargetQuantMode)
	if !ok {
		targetSupport, ok = QATTargetQuantModeSupport(evidence.TargetSize, evidence.TargetQuantMode)
	}
	return ok && targetSupport.Runtime != RuntimePlanned && targetSupport.GenerateStatus == GenerateLinked
}

func targetRuntimeCompatibleWithAttachedDrafter(runtime string) bool {
	switch strings.ToLower(strings.TrimSpace(runtime)) {
	case RuntimeMLXAffine, RuntimeGGUF:
		return true
	default:
		return false
	}
}

func assistantRuntimeCompatibleWithAttachedDrafter(runtime, mode string) bool {
	runtime = strings.ToLower(strings.TrimSpace(runtime))
	mode = denormalizeStatusQuantMode(strings.ToLower(strings.TrimSpace(mode)))
	if mode == AssistantQuantMode {
		return runtime == RuntimeBF16 || runtime == RuntimeGGUF
	}
	return runtime == RuntimeMLXAffine || runtime == RuntimeGGUF
}

func assistantQuantModeCompatibleWithTarget(targetMode, assistantMode string) bool {
	targetMode = denormalizeStatusQuantMode(strings.ToLower(strings.TrimSpace(targetMode)))
	assistantMode = denormalizeStatusQuantMode(strings.ToLower(strings.TrimSpace(assistantMode)))
	if assistantMode == AssistantQuantMode || assistantMode == targetMode {
		return true
	}
	targetBits := quantModeBits(targetMode)
	assistantBits := quantModeBits(assistantMode)
	return targetBits > 0 && assistantBits > 0 && strings.HasPrefix(targetMode, "q") && strings.HasPrefix(assistantMode, "q") && assistantBits >= targetBits
}

func ApplyPairVerificationLabels(labels map[string]string, target, assistant inference.ModelIdentity, dotted bool) {
	if labels == nil {
		return
	}
	key := "attached_drafter_official_pair_verified"
	familyKey := "attached_drafter_gemma4_family_pair_verified"
	if dotted {
		key = "attached.drafter.official_pair_verified"
		familyKey = "attached.drafter.gemma4_family_pair_verified"
	}
	labels[key] = strconv.FormatBool(OfficialPairVerified(target, assistant))
	labels[familyKey] = strconv.FormatBool(FamilyPairVerified(target, assistant))
}

func ApplyOfficialPairLockLabels(labels map[string]string, target, assistant inference.ModelIdentity, dotted bool) {
	if labels == nil {
		return
	}
	prefix := "attached_drafter_"
	if dotted {
		prefix = "attached.drafter."
	}
	for _, key := range []string{
		prefix + "official_target_model_id",
		prefix + "official_target_revision",
		prefix + "official_assistant_model_id",
		prefix + "official_assistant_revision",
	} {
		delete(labels, key)
	}
	pairVerified := OfficialPairVerified(target, assistant)
	labels[prefix+"official_pair_verified"] = strconv.FormatBool(pairVerified)
	labels[prefix+"gemma4_family_pair_verified"] = strconv.FormatBool(FamilyPairVerified(target, assistant))
	if !pairVerified {
		return
	}
	labels[prefix+"official_assistant_model_id"] = OfficialE2BAssistantModelID
	labels[prefix+"official_assistant_revision"] = OfficialE2BAssistantRevision
	labels[prefix+"official_target_model_id"] = OfficialE2BTargetModelID
	labels[prefix+"official_target_revision"] = OfficialE2BTargetRevision
}
