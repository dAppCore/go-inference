// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"strconv"
	"strings"

	"dappco.re/go/inference"
	"dappco.re/go/inference/engine/hip/internal/registry"
	"dappco.re/go/inference/engine/hip/profile"
)

const (
	AttachedDrafterRegistryContract = "rocm-attached-drafter-registry-v1"

	AttachedDrafterRouteName       = "mtp-attached-drafter-route"
	AttachedDrafterRuntimeMetadata = "metadata"
	AttachedDrafterRuntimeHIP      = "hip"

	AttachedDrafterGemma4RuntimeMLXAffine    = "mlx_affine"
	AttachedDrafterGemma4RuntimeBF16         = "bf16"
	AttachedDrafterGemma4GenerateLinked      = "linked"
	AttachedDrafterGemma4GenerateLoadOnly    = "load_only"
	AttachedDrafterDefaultDraftTokens        = 4
	AttachedDrafterMinimumRetainedTurns      = 20
	AttachedDrafterAssistantCentroids        = 2048
	AttachedDrafterAssistantIntermediateTopK = 32
)

var attachedDrafterGemma4TargetQuantModes = []string{"q8", "q6", "q5", "q4", "bf16", "mxfp8", "mxfp4", "nvfp4"}
var attachedDrafterGemma4AssistantQuantModes = []string{"bf16", "q8", "q6", "q5", "q4", "mxfp8", "mxfp4", "nvfp4"}

type AttachedDrafterRouteStatus string

const (
	AttachedDrafterRouteNativePending   AttachedDrafterRouteStatus = "native_pending"
	AttachedDrafterRouteAttachedOnly    AttachedDrafterRouteStatus = "attached_only"
	AttachedDrafterRoutePlannedMetadata AttachedDrafterRouteStatus = "planned_metadata"
)

// AttachedDrafterRoute is the folder-owned MTP target/assistant pairing route.
// Model packages can register target or assistant pairing metadata without
// importing the root rocm package.
type AttachedDrafterRoute struct {
	Contract                          string                         `json:"contract,omitempty"`
	Name                              string                         `json:"name,omitempty"`
	Architecture                      string                         `json:"architecture,omitempty"`
	Family                            string                         `json:"family,omitempty"`
	Runtime                           string                         `json:"runtime,omitempty"`
	RuntimeStatus                     inference.FeatureRuntimeStatus `json:"runtime_status,omitempty"`
	Status                            AttachedDrafterRouteStatus     `json:"status,omitempty"`
	Reference                         string                         `json:"reference,omitempty"`
	Mode                              string                         `json:"mode,omitempty"`
	Role                              string                         `json:"role,omitempty"`
	TargetArchitecture                string                         `json:"target_architecture,omitempty"`
	AssistantArchitecture             string                         `json:"assistant_architecture,omitempty"`
	TargetFamily                      string                         `json:"target_family,omitempty"`
	AssistantFamily                   string                         `json:"assistant_family,omitempty"`
	TargetRuntime                     string                         `json:"target_runtime,omitempty"`
	AssistantRuntime                  string                         `json:"assistant_runtime,omitempty"`
	TargetGenerateStatus              string                         `json:"target_generate_status,omitempty"`
	AssistantGenerateStatus           string                         `json:"assistant_generate_status,omitempty"`
	NativeAttachment                  string                         `json:"native_attachment,omitempty"`
	ExecutionStatus                   string                         `json:"execution_status,omitempty"`
	Fallback                          string                         `json:"fallback,omitempty"`
	Registered                        bool                           `json:"registered,omitempty"`
	NativeRuntime                     bool                           `json:"native_runtime,omitempty"`
	Target                            bool                           `json:"target,omitempty"`
	Assistant                         bool                           `json:"assistant,omitempty"`
	AttachedOnly                      bool                           `json:"attached_only,omitempty"`
	StandaloneGeneration              bool                           `json:"standalone_generation,omitempty"`
	PairValidation                    bool                           `json:"pair_validation,omitempty"`
	FamilyPairRequired                bool                           `json:"family_pair_required,omitempty"`
	OfficialPairKnown                 bool                           `json:"official_pair_known,omitempty"`
	OfficialPairLocked                bool                           `json:"official_pair_locked,omitempty"`
	SameSizeRequired                  bool                           `json:"same_size_required,omitempty"`
	SameTokenizerRequired             bool                           `json:"same_tokenizer_required,omitempty"`
	HiddenSizeMatchRequired           bool                           `json:"hidden_size_match_required,omitempty"`
	VocabMatchRequired                bool                           `json:"vocab_match_required,omitempty"`
	LayerTypeMatchRequired            bool                           `json:"layer_type_match_required,omitempty"`
	RetainedStateRequired             bool                           `json:"retained_state_required,omitempty"`
	RuntimeOwnedKV                    bool                           `json:"runtime_owned_kv,omitempty"`
	PromptReplayRefused               bool                           `json:"prompt_replay_refused,omitempty"`
	DraftDetection                    bool                           `json:"draft_detection,omitempty"`
	ExplicitDraft                     bool                           `json:"explicit_draft,omitempty"`
	AutoDetectAssistantDir            bool                           `json:"auto_detect_assistant_dir,omitempty"`
	AutoDetectSiblingPair             bool                           `json:"auto_detect_sibling_pair,omitempty"`
	AutoDetectMTPDir                  bool                           `json:"auto_detect_mtp_dir,omitempty"`
	AutoDetectMTPSiblingGGUF          bool                           `json:"auto_detect_mtp_sibling_gguf,omitempty"`
	TuneProfile                       bool                           `json:"tune_profile,omitempty"`
	FourLayerDrafter                  bool                           `json:"four_layer_drafter,omitempty"`
	OrderedEmbeddings                 bool                           `json:"ordered_embeddings,omitempty"`
	CentroidRouting                   bool                           `json:"centroid_routing,omitempty"`
	BorrowTargetKV                    bool                           `json:"borrow_target_kv,omitempty"`
	VerifyForward                     bool                           `json:"verify_forward,omitempty"`
	NativeGeneration                  bool                           `json:"native_generation,omitempty"`
	NativeStateGeneration             bool                           `json:"native_state_generation,omitempty"`
	FallbackRefused                   bool                           `json:"fallback_refused,omitempty"`
	Staged                            bool                           `json:"staged,omitempty"`
	Planned                           bool                           `json:"planned,omitempty"`
	DefaultDraftTokens                int                            `json:"default_draft_tokens,omitempty"`
	DefaultDraftBlock                 int                            `json:"default_draft_block,omitempty"`
	MinimumRetainedTurns              int                            `json:"minimum_retained_turns,omitempty"`
	AssistantCentroids                int                            `json:"assistant_centroids,omitempty"`
	AssistantCentroidIntermediateTopK int                            `json:"assistant_centroid_intermediate_top_k,omitempty"`
	AssistantTokenOrderingShape       []int                          `json:"assistant_token_ordering_shape,omitempty"`
	TargetSizes                       []string                       `json:"target_sizes,omitempty"`
	TargetQuantModes                  []string                       `json:"target_quant_modes,omitempty"`
	AssistantQuantModes               []string                       `json:"assistant_quant_modes,omitempty"`
	AssistantModelIDs                 []string                       `json:"assistant_model_ids,omitempty"`
	DetectionSources                  []string                       `json:"detection_sources,omitempty"`
	RequiredDraftTokenSweeps          []int                          `json:"required_draft_token_sweeps,omitempty"`
	TunableDraftBlocks                []int                          `json:"tunable_draft_blocks,omitempty"`
	RequiredMetrics                   []string                       `json:"required_metrics,omitempty"`
	Capabilities                      []inference.CapabilityID       `json:"capabilities,omitempty"`
	Labels                            map[string]string              `json:"labels,omitempty"`
}

func (route AttachedDrafterRoute) Matched() bool {
	return route.Contract != "" && route.Name != "" && route.Architecture != "" && route.Registered
}

func (route AttachedDrafterRoute) Clone() AttachedDrafterRoute {
	route.AssistantTokenOrderingShape = append([]int(nil), route.AssistantTokenOrderingShape...)
	route.TargetSizes = append([]string(nil), route.TargetSizes...)
	route.TargetQuantModes = append([]string(nil), route.TargetQuantModes...)
	route.AssistantQuantModes = append([]string(nil), route.AssistantQuantModes...)
	route.AssistantModelIDs = append([]string(nil), route.AssistantModelIDs...)
	route.DetectionSources = append([]string(nil), route.DetectionSources...)
	route.RequiredDraftTokenSweeps = append([]int(nil), route.RequiredDraftTokenSweeps...)
	route.TunableDraftBlocks = append([]int(nil), route.TunableDraftBlocks...)
	route.RequiredMetrics = append([]string(nil), route.RequiredMetrics...)
	route.Capabilities = append([]inference.CapabilityID(nil), route.Capabilities...)
	route.Labels = cloneStringMap(route.Labels)
	return route
}

func (route AttachedDrafterRoute) WithLabels(labels map[string]string) AttachedDrafterRoute {
	route = route.withLabels(labels)
	route.finalize()
	return route.Clone()
}

var registeredAttachedDrafters = registry.NewOrdered[string, AttachedDrafterRoute]()

func RegisterAttachedDrafterRoute(route AttachedDrafterRoute) {
	route = NormalizeAttachedDrafterRoute(route)
	if !route.Matched() {
		return
	}
	registeredAttachedDrafters.Put(route.Architecture, route)
}

func RegisteredAttachedDrafterArchitectures() []string {
	return registeredAttachedDrafters.Keys()
}

func RegisteredAttachedDrafterRoutes() []AttachedDrafterRoute {
	return registeredAttachedDrafterSnapshot()
}

func ReplaceRegisteredAttachedDrafterRoutes(routes []AttachedDrafterRoute) {
	order := make([]string, 0, len(routes))
	values := make(map[string]AttachedDrafterRoute, len(routes))
	for _, route := range routes {
		route = NormalizeAttachedDrafterRoute(route)
		if !route.Matched() {
			continue
		}
		if _, ok := values[route.Architecture]; !ok {
			order = append(order, route.Architecture)
		}
		values[route.Architecture] = route
	}
	registeredAttachedDrafters.Restore(order, values)
}

func RegisteredAttachedDrafterRouteForArchitecture(architecture string) (AttachedDrafterRoute, bool) {
	return registeredAttachedDrafterForArchitecture(architecture)
}

func AttachedDrafterRouteForArchitecture(architecture string) (AttachedDrafterRoute, bool) {
	architecture = profile.ArchitectureID(architecture)
	if architecture == "" {
		return AttachedDrafterRoute{}, false
	}
	if route, ok := registeredAttachedDrafterForArchitecture(architecture); ok {
		return route, true
	}
	architectureProfile, ok := profile.LookupArchitectureProfile(architecture)
	if !ok {
		return AttachedDrafterRoute{}, false
	}
	route := staticAttachedDrafterRoute(architectureProfile.ID, firstNonEmpty(architectureProfile.Family, architectureProfile.ID, "gemma4"), architectureProfile)
	if !route.Matched() {
		return AttachedDrafterRoute{}, false
	}
	return route, true
}

func AttachedDrafterRouteForIdentity(path string, identity inference.ModelIdentity) (AttachedDrafterRoute, bool) {
	if identity.Path == "" {
		identity.Path = path
	}
	architecture := firstNonEmpty(
		identity.Labels["engine_architecture_profile"],
		identity.Labels["architecture_model_type"],
		identity.Labels["engine_architecture_resolved"],
		identity.Labels["architecture_resolved"],
		identity.Architecture,
	)
	route, ok := AttachedDrafterRouteForArchitecture(architecture)
	if ok {
		return route.WithLabels(identity.Labels), true
	}
	route = staticAttachedDrafterRoute(attachedDrafterArchitecture(architecture, identity.Labels), "gemma4", profile.ArchitectureProfile{})
	route = route.WithLabels(identity.Labels)
	if !route.Matched() {
		return AttachedDrafterRoute{}, false
	}
	return route, true
}

func AttachedDrafterRouteForInfo(path string, info inference.ModelInfo, labels map[string]string) (AttachedDrafterRoute, bool) {
	return AttachedDrafterRouteForIdentity(path, inference.ModelIdentity{
		Path:         path,
		Architecture: info.Architecture,
		VocabSize:    info.VocabSize,
		NumLayers:    info.NumLayers,
		HiddenSize:   info.HiddenSize,
		QuantBits:    info.QuantBits,
		QuantGroup:   info.QuantGroup,
		Labels:       cloneStringMap(labels),
	})
}

func AttachedDrafterRouteForInspection(inspection *inference.ModelPackInspection) (AttachedDrafterRoute, bool) {
	if inspection == nil {
		return AttachedDrafterRoute{}, false
	}
	identity := inspection.Model
	if identity.Path == "" {
		identity.Path = inspection.Path
	}
	labels := mergeAttachedDrafterLabels(identity.Labels, inspection.Labels)
	identity.Labels = labels
	return AttachedDrafterRouteForIdentity(identity.Path, identity)
}

func DefaultAttachedDrafterRoutes() []AttachedDrafterRoute {
	profiles := profile.DefaultGemma4ArchitectureSettings()
	routes := make([]AttachedDrafterRoute, 0, len(profiles)+len(registeredAttachedDrafters.Keys()))
	seen := map[string]int{}
	for _, architectureProfile := range profiles {
		route, ok := AttachedDrafterRouteForArchitecture(architectureProfile.ID)
		if !ok {
			continue
		}
		seen[route.Architecture] = len(routes)
		routes = append(routes, route)
	}
	for _, route := range registeredAttachedDrafterSnapshot() {
		if !route.Matched() {
			continue
		}
		if index, ok := seen[route.Architecture]; ok {
			routes[index] = route.Clone()
			continue
		}
		seen[route.Architecture] = len(routes)
		routes = append(routes, route.Clone())
	}
	return cloneAttachedDrafterRoutes(routes)
}

func NormalizeAttachedDrafterRoute(route AttachedDrafterRoute) AttachedDrafterRoute {
	route.Architecture = profile.ArchitectureID(route.Architecture)
	if route.Architecture == "" {
		return AttachedDrafterRoute{}
	}
	architectureProfile, hasProfile := profile.LookupArchitectureProfile(route.Architecture)
	if route.Contract == "" {
		route.Contract = AttachedDrafterRegistryContract
	}
	if route.Name == "" {
		route.Name = AttachedDrafterRouteName
	}
	if route.Family == "" && hasProfile {
		route.Family = firstNonEmpty(architectureProfile.Family, architectureProfile.ID)
	}
	if route.Family == "" {
		route.Family = route.Architecture
	}
	if route.Runtime == "" {
		route.Runtime = AttachedDrafterRuntimeMetadata
	}
	if route.Reference == "" {
		route.Reference = "registered_attached_drafter"
	}
	if route.Mode == "" {
		route.Mode = "mtp_attached_drafter"
	}

	assistantRoute := route.Assistant || route.AttachedOnly || route.Role == "assistant" || attachedDrafterGemma4AssistantArchitecture(route.Architecture)
	if hasProfile && architectureProfile.AttachedOnly {
		assistantRoute = true
	}
	targetRoute := route.Target || route.Role == "target" || (!assistantRoute && !route.Assistant)
	if assistantRoute {
		route.Role = "assistant"
		route.Assistant = true
		route.AttachedOnly = true
		route.Target = false
	} else if targetRoute {
		route.Role = "target"
		route.Target = true
		route.Assistant = false
		route.AttachedOnly = false
	}

	if route.TargetArchitecture == "" {
		if route.Target {
			route.TargetArchitecture = route.Architecture
		} else {
			route.TargetArchitecture = "gemma4_text"
		}
	}
	if route.AssistantArchitecture == "" {
		if route.Assistant {
			route.AssistantArchitecture = route.Architecture
		} else {
			route.AssistantArchitecture = "gemma4_assistant"
		}
	}
	route.TargetFamily = firstNonEmpty(route.TargetFamily, route.Family)
	route.AssistantFamily = firstNonEmpty(route.AssistantFamily, route.Family)
	route.TargetRuntime = firstNonEmpty(route.TargetRuntime, AttachedDrafterGemma4RuntimeMLXAffine)
	route.AssistantRuntime = firstNonEmpty(route.AssistantRuntime, AttachedDrafterGemma4RuntimeBF16)
	route.TargetGenerateStatus = firstNonEmpty(route.TargetGenerateStatus, AttachedDrafterGemma4GenerateLinked)
	route.AssistantGenerateStatus = firstNonEmpty(route.AssistantGenerateStatus, AttachedDrafterGemma4GenerateLoadOnly)

	nativeRequested := route.NativeRuntime || route.NativeAttachment == KernelStatusLinked || route.Runtime == AttachedDrafterRuntimeHIP || route.ExecutionStatus == "ready"
	if nativeRequested {
		route.NativeAttachment = firstNonEmpty(route.NativeAttachment, KernelStatusLinked)
		route.NativeGeneration = true
		route.NativeStateGeneration = true
	} else {
		route.NativeAttachment = firstNonEmpty(route.NativeAttachment, KernelStatusNotLinked)
		route.ExecutionStatus = firstNonEmpty(route.ExecutionStatus, KernelStatusNotLinked)
		route.Fallback = firstNonEmpty(route.Fallback, "refused")
		route.FallbackRefused = true
	}

	route.PairValidation = true
	route.FamilyPairRequired = true
	route.OfficialPairKnown = true
	route.OfficialPairLocked = true
	route.SameSizeRequired = true
	route.SameTokenizerRequired = true
	route.HiddenSizeMatchRequired = false
	route.VocabMatchRequired = true
	route.LayerTypeMatchRequired = true
	route.RetainedStateRequired = true
	route.RuntimeOwnedKV = true
	route.PromptReplayRefused = true
	route.DraftDetection = true
	route.ExplicitDraft = true
	route.AutoDetectAssistantDir = true
	route.AutoDetectSiblingPair = true
	route.AutoDetectMTPDir = true
	route.AutoDetectMTPSiblingGGUF = true
	route.TuneProfile = true
	route.FourLayerDrafter = true
	route.OrderedEmbeddings = true
	route.CentroidRouting = true
	route.BorrowTargetKV = true
	route.VerifyForward = true
	route.StandaloneGeneration = false
	route.DefaultDraftTokens = firstPositiveInt(route.DefaultDraftTokens, AttachedDrafterDefaultDraftTokens)
	route.DefaultDraftBlock = firstPositiveInt(route.DefaultDraftBlock, 5)
	route.MinimumRetainedTurns = firstPositiveInt(route.MinimumRetainedTurns, AttachedDrafterMinimumRetainedTurns)
	route.AssistantCentroids = firstPositiveInt(route.AssistantCentroids, AttachedDrafterAssistantCentroids)
	route.AssistantCentroidIntermediateTopK = firstPositiveInt(route.AssistantCentroidIntermediateTopK, AttachedDrafterAssistantIntermediateTopK)
	if len(route.AssistantTokenOrderingShape) == 0 {
		route.AssistantTokenOrderingShape = []int{route.AssistantCentroids, 128}
	}
	if len(route.TargetSizes) == 0 {
		route.TargetSizes = []string{"E2B", "E4B", "12B", "26B-A4B", "31B"}
	}
	if len(route.TargetQuantModes) == 0 {
		route.TargetQuantModes = append([]string(nil), attachedDrafterGemma4TargetQuantModes...)
	}
	if len(route.AssistantQuantModes) == 0 {
		route.AssistantQuantModes = append([]string(nil), attachedDrafterGemma4AssistantQuantModes...)
	}
	if len(route.DetectionSources) == 0 {
		route.DetectionSources = []string{"flag", "assistant-dir", "assistant-pair", "mtp-dir", "mtp-sibling-gguf"}
	}
	if len(route.RequiredDraftTokenSweeps) == 0 {
		route.RequiredDraftTokenSweeps = []int{1, 2, 4}
	}
	if len(route.TunableDraftBlocks) == 0 {
		route.TunableDraftBlocks = []int{4, 5, 6}
	}
	if len(route.RequiredMetrics) == 0 {
		route.RequiredMetrics = defaultAttachedDrafterRequiredMetrics()
	}
	if len(route.AssistantModelIDs) == 0 {
		for _, size := range route.TargetSizes {
			route.AssistantModelIDs = append(route.AssistantModelIDs, gemma4MTPAssistantPaths(size)...)
		}
	}
	route.finalize()
	return route.Clone()
}

func registeredAttachedDrafterForArchitecture(architecture string) (AttachedDrafterRoute, bool) {
	route, ok := registeredAttachedDrafters.Get(profile.ArchitectureID(architecture))
	if !ok {
		return AttachedDrafterRoute{}, false
	}
	return route.Clone(), true
}

func registeredAttachedDrafterSnapshot() []AttachedDrafterRoute {
	routes := registeredAttachedDrafters.Values()
	out := make([]AttachedDrafterRoute, 0, len(routes))
	for _, route := range routes {
		out = append(out, route.Clone())
	}
	return out
}

func staticAttachedDrafterRoute(architecture, family string, architectureProfile profile.ArchitectureProfile) AttachedDrafterRoute {
	architecture = profile.ArchitectureID(architecture)
	route := AttachedDrafterRoute{
		Contract:                          AttachedDrafterRegistryContract,
		Name:                              AttachedDrafterRouteName,
		Architecture:                      architecture,
		Family:                            firstNonEmpty(family, "gemma4"),
		Runtime:                           AttachedDrafterRuntimeMetadata,
		RuntimeStatus:                     inference.FeatureRuntimeMetadataOnly,
		Reference:                         "go_mlx_gemma4_assistant_pair",
		Mode:                              "mtp_attached_drafter",
		TargetArchitecture:                "gemma4_text",
		AssistantArchitecture:             "gemma4_assistant",
		TargetFamily:                      "gemma4",
		AssistantFamily:                   "gemma4",
		TargetRuntime:                     AttachedDrafterGemma4RuntimeMLXAffine,
		AssistantRuntime:                  AttachedDrafterGemma4RuntimeBF16,
		TargetGenerateStatus:              AttachedDrafterGemma4GenerateLinked,
		AssistantGenerateStatus:           AttachedDrafterGemma4GenerateLoadOnly,
		NativeAttachment:                  KernelStatusNotLinked,
		ExecutionStatus:                   KernelStatusNotLinked,
		Fallback:                          "refused",
		PairValidation:                    true,
		FamilyPairRequired:                true,
		OfficialPairKnown:                 true,
		OfficialPairLocked:                true,
		SameSizeRequired:                  true,
		SameTokenizerRequired:             true,
		HiddenSizeMatchRequired:           false,
		VocabMatchRequired:                true,
		LayerTypeMatchRequired:            true,
		RetainedStateRequired:             true,
		RuntimeOwnedKV:                    true,
		PromptReplayRefused:               true,
		DraftDetection:                    true,
		ExplicitDraft:                     true,
		AutoDetectAssistantDir:            true,
		AutoDetectSiblingPair:             true,
		AutoDetectMTPDir:                  true,
		AutoDetectMTPSiblingGGUF:          true,
		TuneProfile:                       true,
		FourLayerDrafter:                  true,
		OrderedEmbeddings:                 true,
		CentroidRouting:                   true,
		BorrowTargetKV:                    true,
		VerifyForward:                     true,
		FallbackRefused:                   true,
		DefaultDraftTokens:                AttachedDrafterDefaultDraftTokens,
		DefaultDraftBlock:                 5,
		MinimumRetainedTurns:              AttachedDrafterMinimumRetainedTurns,
		AssistantCentroids:                AttachedDrafterAssistantCentroids,
		AssistantCentroidIntermediateTopK: AttachedDrafterAssistantIntermediateTopK,
		AssistantTokenOrderingShape:       []int{AttachedDrafterAssistantCentroids, 128},
		TargetSizes:                       []string{"E2B", "E4B", "12B", "26B-A4B", "31B"},
		TargetQuantModes:                  append([]string(nil), attachedDrafterGemma4TargetQuantModes...),
		AssistantQuantModes:               append([]string(nil), attachedDrafterGemma4AssistantQuantModes...),
		DetectionSources:                  []string{"flag", "assistant-dir", "assistant-pair", "mtp-dir", "mtp-sibling-gguf"},
		RequiredDraftTokenSweeps:          []int{1, 2, 4},
		TunableDraftBlocks:                []int{4, 5, 6},
		RequiredMetrics:                   defaultAttachedDrafterRequiredMetrics(),
	}
	for _, size := range route.TargetSizes {
		route.AssistantModelIDs = append(route.AssistantModelIDs, gemma4MTPAssistantPaths(size)...)
	}
	if architectureProfile.ID != "" {
		route.NativeRuntime = architectureProfile.NativeRuntime && route.NativeAttachment == KernelStatusLinked
	}
	switch {
	case attachedDrafterGemma4AssistantArchitecture(architecture):
		route.Role = "assistant"
		route.Assistant = true
		route.AttachedOnly = true
		route.Target = false
		route.TargetArchitecture = "gemma4_text"
	case attachedDrafterGemma4Architecture(architecture):
		route.Role = "target"
		route.Target = true
		route.Assistant = false
		route.AttachedOnly = false
		route.TargetArchitecture = architecture
	default:
		route.Architecture = firstNonEmpty(architecture, route.Architecture)
	}
	route.finalize()
	return route.Clone()
}

func (route AttachedDrafterRoute) withLabels(labels map[string]string) AttachedDrafterRoute {
	if len(labels) == 0 {
		return route
	}
	route.Reference = firstNonEmpty(labels["engine_attached_drafter_reference"], labels["attached_drafter_reference"], labels["attached.drafter.reference"], route.Reference)
	route.Mode = firstNonEmpty(labels["engine_attached_drafter_mode"], labels["attached_drafter_mode"], labels["attached.drafter.mode"], route.Mode)
	route.Role = firstNonEmpty(labels["engine_attached_drafter_role"], labels["attached_drafter_role"], labels["attached.drafter.role"], route.Role)
	if route.Role == "assistant" {
		route.Assistant = true
		route.AttachedOnly = true
		route.Target = false
	} else if route.Role == "target" {
		route.Target = true
		route.Assistant = false
		route.AttachedOnly = false
	}
	route.TargetArchitecture = firstNonEmpty(labels["engine_attached_drafter_target_architecture"], labels["target_architecture"], route.TargetArchitecture)
	route.AssistantArchitecture = firstNonEmpty(labels["engine_attached_drafter_assistant_architecture"], labels["assistant_architecture"], route.AssistantArchitecture)
	route.TargetRuntime = firstNonEmpty(labels["attached_drafter_target_gemma4_runtime"], labels["attached.drafter.target.gemma4_runtime"], labels["gemma4_runtime"], route.TargetRuntime)
	route.TargetGenerateStatus = firstNonEmpty(labels["attached_drafter_target_gemma4_generate_status"], labels["attached.drafter.target.gemma4_generate_status"], labels["gemma4_generate_status"], route.TargetGenerateStatus)
	route.AssistantRuntime = firstNonEmpty(labels["attached_drafter_assistant_gemma4_runtime"], labels["attached.drafter.assistant.gemma4_runtime"], labels["assistant_gemma4_runtime"], route.AssistantRuntime)
	route.AssistantGenerateStatus = firstNonEmpty(labels["attached_drafter_assistant_gemma4_generate_status"], labels["attached.drafter.assistant.gemma4_generate_status"], labels["assistant_gemma4_generate_status"], route.AssistantGenerateStatus)
	route.NativeAttachment = firstNonEmpty(labels["engine_attached_drafter_native_attachment"], labels["attached_drafter_native_attachment"], labels["attached.drafter.native_attachment"], route.NativeAttachment)
	route.ExecutionStatus = firstNonEmpty(labels["engine_attached_drafter_execution_status"], labels["attached_drafter_execution_status"], labels["attached.drafter.execution_status"], route.ExecutionStatus)
	route.Fallback = firstNonEmpty(labels["engine_attached_drafter_fallback"], labels["attached_drafter_fallback"], labels["attached.drafter.fallback"], route.Fallback)
	if labels["attached_drafter_retained_state_required"] == "true" || labels["attached.drafter.retained_state_required"] == "true" {
		route.RetainedStateRequired = true
	}
	if labels["attached_drafter_prompt_replay_fallback"] == "forbidden" || labels["attached.drafter.prompt_replay_fallback"] == "forbidden" {
		route.PromptReplayRefused = true
	}
	if labels["attached_drafter_official_pair_verified"] == "true" || labels["attached.drafter.official_pair_verified"] == "true" {
		route.OfficialPairLocked = true
	}
	if labels["attached_drafter_gemma4_family_pair_verified"] == "true" || labels["attached.drafter.gemma4_family_pair_verified"] == "true" {
		route.FamilyPairRequired = true
	}
	if tokens := attachedDrafterLabelInt(labels["speculative_draft_tokens"]); tokens > 0 {
		route.DefaultDraftTokens = tokens
	}
	if block := attachedDrafterLabelInt(firstNonEmpty(labels["reactive_draft_block"], labels["mtp_draft_block"])); block > 0 {
		route.DefaultDraftBlock = block
	}
	if route.Architecture == "" {
		route.Architecture = profile.ArchitectureID(firstNonEmpty(labels["engine_architecture_profile"], labels["architecture_model_type"], labels["engine_architecture_resolved"], labels["architecture_resolved"]))
	}
	return route
}

func (route *AttachedDrafterRoute) finalize() {
	if route == nil {
		return
	}
	route.Architecture = profile.ArchitectureID(route.Architecture)
	route.Registered = route.Architecture != "" && (route.Target || route.Assistant)
	route.NativeRuntime = route.Registered && route.NativeAttachment == KernelStatusLinked && route.NativeGeneration
	if route.NativeRuntime {
		route.Runtime = AttachedDrafterRuntimeHIP
		route.RuntimeStatus = inference.FeatureRuntimeExperimental
		route.Status = AttachedDrafterRouteNativePending
		route.ExecutionStatus = "ready"
		route.Staged = false
		route.Planned = false
	} else if route.Registered {
		route.Runtime = firstNonEmpty(route.Runtime, AttachedDrafterRuntimeMetadata)
		if route.RuntimeStatus == "" {
			route.RuntimeStatus = inference.FeatureRuntimeMetadataOnly
		}
		if route.AttachedOnly {
			route.Status = AttachedDrafterRouteAttachedOnly
		} else {
			route.Status = AttachedDrafterRouteNativePending
		}
		route.ExecutionStatus = firstNonEmpty(route.ExecutionStatus, KernelStatusNotLinked)
		route.Staged = true
		route.Planned = true
	}
	if route.Fallback == "" && route.FallbackRefused {
		route.Fallback = "refused"
	}
	route.FallbackRefused = route.FallbackRefused || route.Fallback == "refused"
	if route.DefaultDraftTokens == 0 {
		route.DefaultDraftTokens = AttachedDrafterDefaultDraftTokens
	}
	if route.DefaultDraftBlock == 0 {
		route.DefaultDraftBlock = 5
	}
	route.Capabilities = attachedDrafterRouteCapabilities(*route)
	route.Labels = attachedDrafterRouteLabels(*route)
}

func attachedDrafterArchitecture(architecture string, labels map[string]string) string {
	if architecture := profile.ArchitectureID(architecture); architecture != "" {
		return architecture
	}
	return profile.ArchitectureID(firstNonEmpty(labels["engine_architecture_profile"], labels["architecture_model_type"], labels["engine_architecture_resolved"], labels["architecture_resolved"]))
}

func attachedDrafterGemma4Architecture(architecture string) bool {
	switch profile.Gemma4ArchitectureID(architecture) {
	case "gemma4", "gemma4_text", "gemma4_unified":
		return true
	default:
		return false
	}
}

func attachedDrafterGemma4AssistantArchitecture(architecture string) bool {
	return profile.Gemma4ArchitectureID(architecture) == "gemma4_assistant"
}

func attachedDrafterRouteCapabilities(route AttachedDrafterRoute) []inference.CapabilityID {
	if !route.Matched() {
		return nil
	}
	capabilities := []inference.CapabilityID{inference.CapabilitySpeculativeDecode}
	if route.RetainedStateRequired {
		capabilities = append(capabilities, inference.CapabilityStateBundle, inference.CapabilityStateWake, inference.CapabilityStateSleep, inference.CapabilityStateFork)
	}
	return capabilities
}

// AttachedDrafterRouteCapabilities returns the model-owned capability contract
// for an attached-drafter route.
func AttachedDrafterRouteCapabilities(route AttachedDrafterRoute) []inference.CapabilityID {
	return append([]inference.CapabilityID(nil), attachedDrafterRouteCapabilities(route)...)
}

func attachedDrafterRouteLabels(route AttachedDrafterRoute) map[string]string {
	if !route.Matched() {
		return nil
	}
	labels := map[string]string{
		"engine_attached_drafter_route_contract":             route.Contract,
		"engine_attached_drafter_route":                      route.Name,
		"engine_attached_drafter_runtime":                    route.Runtime,
		"engine_attached_drafter_status":                     string(route.Status),
		"engine_attached_drafter_mode":                       route.Mode,
		"engine_attached_drafter_role":                       route.Role,
		"engine_attached_drafter_registered":                 strconv.FormatBool(route.Registered),
		"engine_attached_drafter_native_runtime":             strconv.FormatBool(route.NativeRuntime),
		"engine_attached_drafter_target":                     strconv.FormatBool(route.Target),
		"engine_attached_drafter_assistant":                  strconv.FormatBool(route.Assistant),
		"engine_attached_drafter_attached_only":              strconv.FormatBool(route.AttachedOnly),
		"engine_attached_drafter_standalone_generation":      strconv.FormatBool(route.StandaloneGeneration),
		"engine_attached_drafter_pair_validation":            strconv.FormatBool(route.PairValidation),
		"engine_attached_drafter_family_pair_required":       strconv.FormatBool(route.FamilyPairRequired),
		"engine_attached_drafter_official_pair_known":        strconv.FormatBool(route.OfficialPairKnown),
		"engine_attached_drafter_official_pair_locked":       strconv.FormatBool(route.OfficialPairLocked),
		"engine_attached_drafter_same_size_required":         strconv.FormatBool(route.SameSizeRequired),
		"engine_attached_drafter_same_tokenizer_required":    strconv.FormatBool(route.SameTokenizerRequired),
		"engine_attached_drafter_hidden_size_match_required": strconv.FormatBool(route.HiddenSizeMatchRequired),
		"engine_attached_drafter_vocab_match_required":       strconv.FormatBool(route.VocabMatchRequired),
		"engine_attached_drafter_layer_type_match_required":  strconv.FormatBool(route.LayerTypeMatchRequired),
		"engine_attached_drafter_retained_state_required":    strconv.FormatBool(route.RetainedStateRequired),
		"engine_attached_drafter_runtime_owned_kv":           strconv.FormatBool(route.RuntimeOwnedKV),
		"engine_attached_drafter_prompt_replay_refused":      strconv.FormatBool(route.PromptReplayRefused),
		"engine_attached_drafter_draft_detection":            strconv.FormatBool(route.DraftDetection),
		"engine_attached_drafter_explicit_draft":             strconv.FormatBool(route.ExplicitDraft),
		"engine_attached_drafter_auto_assistant_dir":         strconv.FormatBool(route.AutoDetectAssistantDir),
		"engine_attached_drafter_auto_sibling_pair":          strconv.FormatBool(route.AutoDetectSiblingPair),
		"engine_attached_drafter_auto_mtp_dir":               strconv.FormatBool(route.AutoDetectMTPDir),
		"engine_attached_drafter_auto_mtp_sibling_gguf":      strconv.FormatBool(route.AutoDetectMTPSiblingGGUF),
		"engine_attached_drafter_tune_profile":               strconv.FormatBool(route.TuneProfile),
		"engine_attached_drafter_four_layer_drafter":         strconv.FormatBool(route.FourLayerDrafter),
		"engine_attached_drafter_ordered_embeddings":         strconv.FormatBool(route.OrderedEmbeddings),
		"engine_attached_drafter_centroid_routing":           strconv.FormatBool(route.CentroidRouting),
		"engine_attached_drafter_borrow_target_kv":           strconv.FormatBool(route.BorrowTargetKV),
		"engine_attached_drafter_verify_forward":             strconv.FormatBool(route.VerifyForward),
		"engine_attached_drafter_native_generation":          strconv.FormatBool(route.NativeGeneration),
		"engine_attached_drafter_native_state_generation":    strconv.FormatBool(route.NativeStateGeneration),
		"engine_attached_drafter_fallback_refused":           strconv.FormatBool(route.FallbackRefused),
		"engine_attached_drafter_staged":                     strconv.FormatBool(route.Staged),
		"engine_attached_drafter_planned":                    strconv.FormatBool(route.Planned),
		"engine_attached_drafter_target_sizes":               joinNonEmptyStrings(route.TargetSizes, ","),
		"engine_attached_drafter_target_quant_modes":         joinNonEmptyStrings(route.TargetQuantModes, ","),
		"engine_attached_drafter_assistant_quant_modes":      joinNonEmptyStrings(route.AssistantQuantModes, ","),
		"engine_attached_drafter_assistant_models":           joinNonEmptyStrings(route.AssistantModelIDs, ","),
		"engine_attached_drafter_detection_sources":          joinNonEmptyStrings(route.DetectionSources, ","),
		"engine_attached_drafter_capabilities":               attachedDrafterCapabilityLabels(route.Capabilities),
		"engine_attached_drafter_required_metrics":           joinNonEmptyStrings(route.RequiredMetrics, ","),
	}
	setStringLabel(labels, "engine_attached_drafter_architecture", route.Architecture)
	setStringLabel(labels, "engine_attached_drafter_family", route.Family)
	setStringLabel(labels, "engine_attached_drafter_runtime_status", string(route.RuntimeStatus))
	setStringLabel(labels, "engine_attached_drafter_reference", route.Reference)
	setStringLabel(labels, "engine_attached_drafter_target_architecture", route.TargetArchitecture)
	setStringLabel(labels, "engine_attached_drafter_assistant_architecture", route.AssistantArchitecture)
	setStringLabel(labels, "engine_attached_drafter_target_family", route.TargetFamily)
	setStringLabel(labels, "engine_attached_drafter_assistant_family", route.AssistantFamily)
	setStringLabel(labels, "engine_attached_drafter_target_runtime", route.TargetRuntime)
	setStringLabel(labels, "engine_attached_drafter_assistant_runtime", route.AssistantRuntime)
	setStringLabel(labels, "engine_attached_drafter_target_generate_status", route.TargetGenerateStatus)
	setStringLabel(labels, "engine_attached_drafter_assistant_generate_status", route.AssistantGenerateStatus)
	setStringLabel(labels, "engine_attached_drafter_native_attachment", route.NativeAttachment)
	setStringLabel(labels, "engine_attached_drafter_execution_status", route.ExecutionStatus)
	setStringLabel(labels, "engine_attached_drafter_fallback", route.Fallback)
	if route.RuntimeOwnedKV {
		labels["engine_attached_drafter_state_source"] = "rocm_state_session_runtime_kv"
	}
	if route.PromptReplayRefused {
		labels["engine_attached_drafter_prompt_replay_fallback"] = "forbidden"
	}
	setIntLabel(labels, "engine_attached_drafter_default_draft_tokens", route.DefaultDraftTokens)
	setIntLabel(labels, "engine_attached_drafter_default_draft_block", route.DefaultDraftBlock)
	setIntLabel(labels, "engine_attached_drafter_minimum_retained_turns", route.MinimumRetainedTurns)
	setIntLabel(labels, "engine_attached_drafter_assistant_centroids", route.AssistantCentroids)
	setIntLabel(labels, "engine_attached_drafter_assistant_centroid_intermediate_top_k", route.AssistantCentroidIntermediateTopK)
	if len(route.AssistantTokenOrderingShape) > 0 {
		labels["engine_attached_drafter_assistant_token_ordering_dtype"] = "int64"
		labels["engine_attached_drafter_assistant_token_ordering_shape"] = attachedDrafterIntLabels(route.AssistantTokenOrderingShape, "x")
	}
	if len(route.RequiredDraftTokenSweeps) > 0 {
		labels["engine_attached_drafter_required_draft_token_sweeps"] = attachedDrafterIntLabels(route.RequiredDraftTokenSweeps, ",")
	}
	if len(route.TunableDraftBlocks) > 0 {
		labels["engine_attached_drafter_tunable_draft_blocks"] = attachedDrafterIntLabels(route.TunableDraftBlocks, ",")
	}
	return labels
}

// AttachedDrafterRouteLabels returns the normalized model-owned label contract
// for an attached-drafter route.
func AttachedDrafterRouteLabels(route AttachedDrafterRoute) map[string]string {
	route = NormalizeAttachedDrafterRoute(route)
	return cloneStringMap(route.Labels)
}

func attachedDrafterLabelInt(value string) int {
	value = strings.TrimSpace(value)
	if value == "" || value == "backend_default" {
		return 0
	}
	parsed, err := strconv.Atoi(value)
	if err != nil || parsed < 0 {
		return 0
	}
	return parsed
}

func attachedDrafterIntLabels(values []int, sep string) string {
	if len(values) == 0 {
		return ""
	}
	out := make([]string, 0, len(values))
	for _, value := range values {
		if value > 0 {
			out = append(out, strconv.Itoa(value))
		}
	}
	return joinNonEmptyStrings(out, sep)
}

func attachedDrafterCapabilityLabels(capabilities []inference.CapabilityID) string {
	if len(capabilities) == 0 {
		return ""
	}
	values := make([]string, 0, len(capabilities))
	for _, capability := range capabilities {
		if capability != "" {
			values = append(values, string(capability))
		}
	}
	return joinNonEmptyStrings(values, ",")
}

func gemma4MTPAssistantPaths(size string) []string {
	size = gemma4MTPAssistantSize(size)
	paths := []string{gemma4MTPAssistantPath(size)}
	for _, mode := range attachedDrafterGemma4AssistantQuantModes {
		paths = append(paths, gemma4MTPQATAssistantPath(size, mode))
	}
	return paths
}

func gemma4MTPAssistantPath(size string) string {
	size = gemma4MTPAssistantSize(size)
	if size == "" {
		size = "E2B"
	}
	return "google/gemma-4-" + size + "-it-assistant"
}

func gemma4MTPQATAssistantPath(size, mode string) string {
	size = gemma4MTPAssistantSize(size)
	if size == "" {
		size = "E2B"
	}
	suffix := gemma4MTPQATQuantSuffix(mode)
	if suffix == "" {
		suffix = "bf16"
	}
	return "mlx-community/gemma-4-" + size + "-it-qat-assistant-" + suffix
}

func gemma4MTPAssistantSize(size string) string {
	size = strings.TrimSpace(size)
	switch strings.ToLower(size) {
	case "26b-a4b":
		return "26B-A4B"
	default:
		return strings.ToUpper(size)
	}
}

func gemma4MTPQATQuantSuffix(mode string) string {
	switch strings.TrimSuffix(strings.ToLower(strings.TrimSpace(mode)), "-status") {
	case "q8":
		return "8bit"
	case "q6":
		return "6bit"
	case "q5":
		return "5bit"
	case "q4":
		return "4bit"
	case "bf16":
		return "bf16"
	case "mxfp8":
		return "mxfp8"
	case "mxfp4":
		return "mxfp4"
	case "nvfp4":
		return "nvfp4"
	default:
		return ""
	}
}

func defaultAttachedDrafterRequiredMetrics() []string {
	return []string{
		"retained_workflow",
		"turns",
		"greedy_output_matches",
		"quality_flags",
		"speculative_draft_model_path",
		"speculative_draft_tokens",
		"target_only_visible_tokens_per_sec",
		"mtp_visible_tokens_per_sec",
		"mtp_target_tokens_per_sec",
		"mtp_warm_decode_tokens_per_sec",
		"target_only_wall_duration",
		"mtp_wall_duration",
		"target_only_restore_duration",
		"mtp_restore_duration",
		"target_only_peak_memory_bytes",
		"mtp_peak_memory_bytes",
		"target_only_active_plus_cache_memory_bytes",
		"mtp_active_plus_cache_memory_bytes",
		"target_only_energy_joules",
		"mtp_energy_joules",
		"same_load_policy",
		"target_only_cache_mode",
		"mtp_cache_mode",
		"mtp_observed_draft_token_sweeps",
		"mtp_proposed_tokens",
		"mtp_accepted_tokens",
		"mtp_rejected_tokens",
		"mtp_target_verify_calls",
		"mtp_draft_calls",
		"attached_drafter_retained_state_entrypoint",
		"attached_drafter_retained_state_required",
		"attached_drafter_state_source",
		"attached_drafter_prompt_replay_fallback",
		"attached_drafter_target_gemma4_size",
		"attached_drafter_target_gemma4_quant_mode",
		"attached_drafter_target_gemma4_quant_group",
		"attached_drafter_target_gemma4_runtime",
		"attached_drafter_target_gemma4_generate_status",
		"attached_drafter_target_production_quant_model",
		"attached_drafter_assistant_gemma4_size",
		"attached_drafter_assistant_gemma4_quant_mode",
		"attached_drafter_assistant_gemma4_runtime",
		"attached_drafter_assistant_gemma4_generate_status",
		"attached_drafter_assistant_production_quant_model",
		"attached_drafter_assistant_production_quant_pack",
		"attached_drafter_assistant_production_quant_tier",
		"attached_drafter_assistant_production_quant_mtp_assistant",
		"assistant_architecture",
		"assistant_ordered_embeddings",
		"assistant_centroids",
		"assistant_centroid_intermediate_top_k",
		"assistant_four_layer_drafter",
		"assistant_token_ordering_dtype",
		"assistant_token_ordering_shape",
		"gemma4_family_pair_verified",
	}
}

func mergeAttachedDrafterLabels(left, right map[string]string) map[string]string {
	out := cloneStringMap(left)
	if out == nil {
		out = map[string]string{}
	}
	for key, value := range right {
		if value != "" {
			out[key] = value
		}
	}
	return out
}

func cloneAttachedDrafterRoutes(routes []AttachedDrafterRoute) []AttachedDrafterRoute {
	out := append([]AttachedDrafterRoute(nil), routes...)
	for i := range out {
		out[i] = out[i].Clone()
	}
	return out
}
