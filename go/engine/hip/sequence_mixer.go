// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"slices"
	"strings"

	core "dappco.re/go"
	"dappco.re/go/inference"
	rocmmodel "dappco.re/go/inference/engine/hip/model"
)

const (
	SequenceMixerRuntimePlannedHIP       = rocmmodel.SequenceMixerRuntimePlannedHIP
	SequenceMixerRegistryContract        = rocmmodel.SequenceMixerRegistryContract
	SequenceMixerStateKVCache            = rocmmodel.SequenceMixerStateKVCache
	SequenceMixerStateRecurrent          = rocmmodel.SequenceMixerStateRecurrent
	SequenceMixerStateContract           = rocmmodel.SequenceMixerStateContract
	SequenceMixerStateSlotsContract      = rocmmodel.SequenceMixerStateSlotsContract
	SequenceMixerCachePlanContract       = rocmmodel.SequenceMixerCachePlanContract
	SequenceMixerCacheFactoryContract    = rocmmodel.SequenceMixerCacheFactoryContract
	SequenceMixerCacheModeDefault        = rocmmodel.SequenceMixerCacheModeDefault
	SequenceMixerCacheModeRecurrent      = rocmmodel.SequenceMixerCacheModeRecurrent
	SequenceMixerCacheModeMLALatent      = rocmmodel.SequenceMixerCacheModeMLALatent
	SequenceMixerCacheModeCompaction     = rocmmodel.SequenceMixerCacheModeCompaction
	SequenceMixerCacheModeCompactionFull = rocmmodel.SequenceMixerCacheModeCompactionFull
	SequenceMixerRequiredLeavesContract  = rocmmodel.SequenceMixerRequiredLeavesContract
)

// SequenceMixerFamily describes one config-composed sequence mixer kind ROCm
// can recognise and plan for. Model metadata lives in go/model; the root name
// remains the public API surface for consumers.
type SequenceMixerFamily = rocmmodel.SequenceMixerFamily

// SequenceMixerSubpathPlan records checkpoint-derived mixer sublayer routing.
type SequenceMixerSubpathPlan = rocmmodel.SequenceMixerSubpathPlan

// SequenceMixerLayerPlan is the model-owned side of the config-composed loader
// contract: one normalized mixer kind, state shape, and checkpoint subpath.
type SequenceMixerLayerPlan = rocmmodel.SequenceMixerLayerPlan

// SequenceMixerCacheLayerPlan is the cache-holder side of go-mlx's composed
// NewCache contract.
type SequenceMixerCacheLayerPlan = rocmmodel.SequenceMixerCacheLayerPlan

// SequenceMixerCachePlan records the per-layer cache holders needed by the
// config-composed mixer stack.
type SequenceMixerCachePlan = rocmmodel.SequenceMixerCachePlan

// SequenceMixerLoadPlan is the inspected plan a HIP/CUDA/CPU backend can
// consume without rediscovering config and tensor routing decisions.
type SequenceMixerLoadPlan = rocmmodel.SequenceMixerLoadPlan

// DefaultSequenceMixerFamilies returns the active go-mlx-style sequence-mixer
// registry surface: generic softmax plus the nine FLA sequence-mixer families,
// with any registered ROCm extension families applied.
func DefaultSequenceMixerFamilies() []SequenceMixerFamily {
	return rocmmodel.DefaultSequenceMixerFamilies()
}

// RegisterSequenceMixerFamily registers or replaces a sequence-mixer family in
// the ROCm planning registry. Registered families mirror go-mlx mixer-loader
// self-registration: the config declares a layer kind, and the registry supplies
// the state/cache contract plus the required checkpoint leaves used by planning.
func RegisterSequenceMixerFamily(family SequenceMixerFamily, requiredLeaves []string) {
	rocmmodel.RegisterSequenceMixerFamily(family, requiredLeaves)
}

// RegisteredSequenceMixerFamilyKinds returns extension family kinds in
// resolution order. Built-ins are not included.
func RegisteredSequenceMixerFamilyKinds() []string {
	return rocmmodel.RegisteredSequenceMixerFamilyKinds()
}

// SequenceMixerFamilyByKind resolves a normalized mixer kind.
func SequenceMixerFamilyByKind(kind string) (SequenceMixerFamily, bool) {
	return rocmmodel.SequenceMixerFamilyByKind(kind)
}

// DefaultSequenceMixerCacheFactoryModes returns the go-mlx cache factory modes
// ROCm can plan for. "default" is the standard growing KV cache, "recurrent" is
// the fixed recurrent holder, and "mla-latent" is MLA's compressed-latent KV
// store.
func DefaultSequenceMixerCacheFactoryModes() []string {
	return rocmmodel.DefaultSequenceMixerCacheFactoryModes()
}

// SequenceMixerCacheModeForKind resolves the cache factory mode a registered
// mixer kind needs. Consumers can use this before building a full load plan when
// they already know the config's normalized mixer kind.
func SequenceMixerCacheModeForKind(kind string) (string, bool) {
	return rocmmodel.SequenceMixerCacheModeForKind(kind)
}

// SequenceMixerStateSlotsForKind returns the recurrent holder slots a mixer
// kind threads through go-mlx's cache factory. KV-cache mixers return an empty
// slot list with ok=true because their holder shape is implicit in the KV cache.
func SequenceMixerStateSlotsForKind(kind string) ([]string, bool) {
	return rocmmodel.SequenceMixerStateSlotsForKind(kind)
}

// SequenceMixerRequiredLeaves returns the bare checkpoint leaf names a composed
// mixer family needs below its discovered layer subpath.
func SequenceMixerRequiredLeaves(kind string) ([]string, bool) {
	return rocmmodel.SequenceMixerRequiredLeaves(kind)
}

func sequenceMixerRequiredLeaves(kind string) ([]string, bool) {
	return rocmmodel.SequenceMixerRequiredLeaves(kind)
}

func sequenceMixerRegisteredKinds() []string {
	return rocmmodel.SequenceMixerFamilyKinds()
}

func sequenceMixerFLAKinds() []string {
	return rocmmodel.SequenceMixerFLAKinds()
}

func sequenceMixerRegisteredStateEntries() []string {
	return rocmmodel.SequenceMixerRegisteredStateEntries()
}

func sequenceMixerRegisteredCacheModeEntries() []string {
	return rocmmodel.SequenceMixerRegisteredCacheModeEntries()
}

func sequenceMixerRegisteredStateSlotEntries() []string {
	return rocmmodel.SequenceMixerRegisteredStateSlotEntries()
}

func sequenceMixerStateSlotCountEntries() []string {
	return rocmmodel.SequenceMixerStateSlotCountEntries()
}

func sequenceMixerCacheFactoryModes() []string {
	return rocmmodel.DefaultSequenceMixerCacheFactoryModes()
}

func sequenceMixerRequiredLeafEntries() []string {
	return rocmmodel.SequenceMixerRequiredLeafEntries()
}

func sequenceMixerLayerCounts(layerTypes []string) map[string]int {
	return rocmmodel.SequenceMixerLayerCounts(layerTypes)
}

func sequenceMixerUniqueKinds(layerTypes []string) []string {
	return rocmmodel.SequenceMixerUniqueKinds(layerTypes)
}

func rocmApplySequenceMixerConfigLabels(labels map[string]string, layerTypes []string, layerTypesSource string) {
	if labels == nil || len(layerTypes) == 0 {
		return
	}
	counts := sequenceMixerLayerCounts(layerTypes)
	declared := sequenceMixerUniqueKinds(layerTypes)
	if len(declared) == 0 {
		return
	}
	registered := make([]string, 0, len(declared))
	unregistered := make([]string, 0)
	flaKinds := make([]string, 0)
	flaLayers := 0
	for _, kind := range declared {
		family, ok := SequenceMixerFamilyByKind(kind)
		if !ok {
			unregistered = append(unregistered, kind)
			continue
		}
		registered = append(registered, kind)
		if family.Source == "fla" {
			flaKinds = append(flaKinds, kind)
			flaLayers += counts[kind]
		}
	}
	labels["sequence_mixer_registry"] = "rocm_planning"
	labels["sequence_mixer_registry_contract"] = SequenceMixerRegistryContract
	labels["sequence_mixer_registry_kinds"] = core.Join(",", sequenceMixerRegisteredKinds()...)
	labels["sequence_mixer_state_contract"] = SequenceMixerStateContract
	labels["sequence_mixer_registered_states"] = core.Join(",", sequenceMixerRegisteredStateEntries()...)
	labels["sequence_mixer_state_slots_contract"] = SequenceMixerStateSlotsContract
	labels["sequence_mixer_registered_state_slots"] = core.Join(",", sequenceMixerRegisteredStateSlotEntries()...)
	labels["sequence_mixer_state_slot_counts"] = core.Join(",", sequenceMixerStateSlotCountEntries()...)
	labels["sequence_mixer_cache_factory_contract"] = SequenceMixerCacheFactoryContract
	labels["sequence_mixer_cache_factory_modes"] = core.Join(",", sequenceMixerCacheFactoryModes()...)
	labels["sequence_mixer_registered_cache_modes"] = core.Join(",", sequenceMixerRegisteredCacheModeEntries()...)
	labels["sequence_mixer_required_leaves_contract"] = SequenceMixerRequiredLeavesContract
	labels["sequence_mixer_required_leaves"] = core.Join(",", sequenceMixerRequiredLeafEntries()...)
	labels["sequence_mixer_loader_status"] = "registered_contract"
	labels["sequence_mixer_runtime"] = SequenceMixerRuntimePlannedHIP
	labels["sequence_mixer_declared_kinds"] = core.Join(",", declared...)
	if layerTypesSource != "" {
		labels["sequence_mixer_layer_types_source"] = layerTypesSource
	}
	if len(registered) > 0 {
		labels["sequence_mixer_registered_declared_kinds"] = core.Join(",", registered...)
	}
	if len(unregistered) > 0 {
		labels["sequence_mixer_unregistered_declared_kinds"] = core.Join(",", unregistered...)
	} else if len(declared) > 0 {
		labels["sequence_mixer_load_plan_candidate"] = "true"
	}
	if counts["full_attention"] > 0 {
		labels["sequence_mixer_full_attention_layers"] = core.Sprintf("%d", counts["full_attention"])
	}
	if len(flaKinds) > 0 {
		labels["sequence_mixer_fla"] = "true"
		labels["sequence_mixer_fla_kinds"] = core.Join(",", flaKinds...)
		labels["sequence_mixer_fla_layers"] = core.Sprintf("%d", flaLayers)
	}
}

func rocmApplySequenceMixerConfigErrorLabels(labels map[string]string, layerTypesSource string, err error) {
	if labels == nil || err == nil {
		return
	}
	labels["sequence_mixer_registry"] = "rocm_planning"
	labels["sequence_mixer_registry_contract"] = SequenceMixerRegistryContract
	labels["sequence_mixer_registry_kinds"] = core.Join(",", sequenceMixerRegisteredKinds()...)
	labels["sequence_mixer_state_contract"] = SequenceMixerStateContract
	labels["sequence_mixer_registered_states"] = core.Join(",", sequenceMixerRegisteredStateEntries()...)
	labels["sequence_mixer_state_slots_contract"] = SequenceMixerStateSlotsContract
	labels["sequence_mixer_registered_state_slots"] = core.Join(",", sequenceMixerRegisteredStateSlotEntries()...)
	labels["sequence_mixer_state_slot_counts"] = core.Join(",", sequenceMixerStateSlotCountEntries()...)
	labels["sequence_mixer_cache_factory_contract"] = SequenceMixerCacheFactoryContract
	labels["sequence_mixer_cache_factory_modes"] = core.Join(",", sequenceMixerCacheFactoryModes()...)
	labels["sequence_mixer_registered_cache_modes"] = core.Join(",", sequenceMixerRegisteredCacheModeEntries()...)
	labels["sequence_mixer_required_leaves_contract"] = SequenceMixerRequiredLeavesContract
	labels["sequence_mixer_required_leaves"] = core.Join(",", sequenceMixerRequiredLeafEntries()...)
	labels["sequence_mixer_loader_status"] = "registered_contract"
	labels["sequence_mixer_runtime"] = SequenceMixerRuntimePlannedHIP
	if layerTypesSource != "" {
		labels["sequence_mixer_layer_types_source"] = layerTypesSource
	}
	rocmApplySequenceMixerLoadPlanLabels(labels, SequenceMixerLoadPlan{
		Contract: SequenceMixerRegistryContract,
		Runtime:  SequenceMixerRuntimePlannedHIP,
	}, err)
}

func rocmApplySequenceMixerCapabilityLabels(capability *inference.Capability) {
	if capability == nil {
		return
	}
	if capability.Labels == nil {
		capability.Labels = map[string]string{}
	}
	capability.Labels["sequence_mixer_registry"] = "rocm_planning"
	capability.Labels["sequence_mixer_registry_contract"] = SequenceMixerRegistryContract
	capability.Labels["sequence_mixer_registry_kinds"] = core.Join(",", sequenceMixerRegisteredKinds()...)
	capability.Labels["sequence_mixer_fla_kinds"] = core.Join(",", sequenceMixerFLAKinds()...)
	capability.Labels["sequence_mixer_state_contract"] = SequenceMixerStateContract
	capability.Labels["sequence_mixer_registered_states"] = core.Join(",", sequenceMixerRegisteredStateEntries()...)
	capability.Labels["sequence_mixer_state_slots_contract"] = SequenceMixerStateSlotsContract
	capability.Labels["sequence_mixer_registered_state_slots"] = core.Join(",", sequenceMixerRegisteredStateSlotEntries()...)
	capability.Labels["sequence_mixer_state_slot_counts"] = core.Join(",", sequenceMixerStateSlotCountEntries()...)
	capability.Labels["sequence_mixer_cache_factory_contract"] = SequenceMixerCacheFactoryContract
	capability.Labels["sequence_mixer_cache_factory_modes"] = core.Join(",", sequenceMixerCacheFactoryModes()...)
	capability.Labels["sequence_mixer_registered_cache_modes"] = core.Join(",", sequenceMixerRegisteredCacheModeEntries()...)
	capability.Labels["sequence_mixer_required_leaves_contract"] = SequenceMixerRequiredLeavesContract
	capability.Labels["sequence_mixer_required_leaves"] = core.Join(",", sequenceMixerRequiredLeafEntries()...)
	capability.Labels["sequence_mixer_cache_plan_contract"] = SequenceMixerCachePlanContract
	capability.Labels["sequence_mixer_cache_holders"] = core.Join(",", SequenceMixerStateKVCache, SequenceMixerStateRecurrent)
	capability.Labels["sequence_mixer_runtime"] = SequenceMixerRuntimePlannedHIP
	capability.Labels["sequence_mixer_hip_kernels"] = hipKernelStatusNotLinked
	capability.Labels["sequence_mixer_subpath_discovery"] = "safetensors"
}

// BuildSequenceMixerLoadPlan validates a config-composed mixer plan the same
// way go-mlx's composed runner does before load: every layer must declare a
// registered mixer kind, the layer count must match, and checkpoint subpath
// discovery must produce either one deterministic owner or a bare layout.
func BuildSequenceMixerLoadPlan(layerTypes []string, tensorNames []string, numLayers int) (SequenceMixerLoadPlan, error) {
	return rocmmodel.BuildSequenceMixerLoadPlan(layerTypes, tensorNames, numLayers)
}

// BuildSequenceMixerCachePlan resolves only the cache side of a composed
// sequence-mixer plan. It is the ROCm planning counterpart to go-mlx's cache
// factory front door: the caller supplies registered mixer layers and ROCm
// returns the per-layer cache holder plus concrete factory mode.
func BuildSequenceMixerCachePlan(layers []SequenceMixerLayerPlan) (SequenceMixerCachePlan, error) {
	return buildSequenceMixerCachePlan(layers)
}

func buildSequenceMixerCachePlan(layers []SequenceMixerLayerPlan) (SequenceMixerCachePlan, error) {
	return rocmmodel.BuildSequenceMixerCachePlan(layers)
}

func sequenceMixerCacheHolderForState(state string) (string, error) {
	switch state {
	case SequenceMixerStateKVCache, SequenceMixerStateRecurrent:
		return state, nil
	default:
		return "", core.NewError("unsupported sequence mixer state " + state)
	}
}

func sequenceMixerCacheModeForLayer(layer SequenceMixerLayerPlan) (string, error) {
	family, ok := SequenceMixerFamilyByKind(layer.Kind)
	if !ok {
		return "", core.NewError("unregistered sequence mixer kind " + layer.Kind)
	}
	if family.State != layer.State {
		return "", core.NewError("sequence mixer state mismatch for " + layer.Kind)
	}
	if family.CacheMode != "" {
		return family.CacheMode, nil
	}
	switch layer.State {
	case SequenceMixerStateRecurrent:
		return SequenceMixerCacheModeRecurrent, nil
	case SequenceMixerStateKVCache:
		return SequenceMixerCacheModeDefault, nil
	default:
		return "", core.NewError("unsupported sequence mixer state " + layer.State)
	}
}

func sequenceMixerStateSlotsForLayer(layer SequenceMixerLayerPlan) ([]string, error) {
	family, ok := SequenceMixerFamilyByKind(layer.Kind)
	if !ok {
		return nil, core.NewError("unregistered sequence mixer kind " + layer.Kind)
	}
	if family.State != layer.State {
		return nil, core.NewError("sequence mixer state mismatch for " + layer.Kind)
	}
	if len(layer.StateSlots) == 0 {
		return append([]string(nil), family.StateSlots...), nil
	}
	if !slices.Equal(layer.StateSlots, family.StateSlots) {
		return nil, core.NewError("sequence mixer state slots mismatch for " + layer.Kind)
	}
	return append([]string(nil), layer.StateSlots...), nil
}

func sequenceMixerLoadPlanFromInspection(inspection *inference.ModelPackInspection, tensors []nativeTensorInfo) (*SequenceMixerLoadPlan, error) {
	if inspection == nil || inspection.Labels["sequence_mixer_load_plan_status"] != "valid" {
		return nil, nil
	}
	names := make([]string, 0, len(tensors))
	for _, tensor := range tensors {
		names = append(names, tensor.Name)
	}
	plan, err := BuildSequenceMixerLoadPlan(sequenceMixerLayerTypesFromLabels(inspection.Labels), names, inspection.Model.NumLayers)
	if err != nil {
		return nil, err
	}
	return cloneSequenceMixerLoadPlan(&plan), nil
}

func cloneSequenceMixerLoadPlan(plan *SequenceMixerLoadPlan) *SequenceMixerLoadPlan {
	return rocmmodel.CloneSequenceMixerLoadPlan(plan)
}

func cloneSequenceMixerCachePlan(plan SequenceMixerCachePlan) SequenceMixerCachePlan {
	return plan.Clone()
}

// DiscoverSequenceMixerSubpaths finds the checkpoint sublayer that owns each
// layer's mixer weights. Like go-mlx's composed loader, only the MLP sublayer is
// excluded; any other nested sub-projection is a candidate owner and multiple
// owners are refused instead of guessed. No subpath means bare leaves.
func DiscoverSequenceMixerSubpaths(names []string, numLayers int) SequenceMixerSubpathPlan {
	return rocmmodel.DiscoverSequenceMixerSubpaths(names, numLayers)
}

func rocmApplySequenceMixerSafetensorsPlanLabels(inspection *inference.ModelPackInspection, path string) error {
	if inspection == nil {
		return nil
	}
	if inspection.Labels["sequence_mixer_load_plan_status"] == "invalid" {
		return core.NewError(inspection.Labels["sequence_mixer_load_plan_error"])
	}
	if inspection.Labels["sequence_mixer_load_plan_candidate"] != "true" {
		return nil
	}
	tensors, err := readROCmSafetensorsNativeTensors(path)
	if err != nil {
		return err
	}
	names := make([]string, 0, len(tensors))
	for _, tensor := range tensors {
		names = append(names, tensor.Name)
	}
	layerTypes := sequenceMixerLayerTypesFromLabels(inspection.Labels)
	loadPlan, err := BuildSequenceMixerLoadPlan(layerTypes, names, inspection.Model.NumLayers)
	rocmApplySequenceMixerLoadPlanLabels(inspection.Labels, loadPlan, err)
	plan := loadPlan.Subpaths
	if plan.LayerCount == 0 {
		return nil
	}
	inspection.Labels["sequence_mixer_subpath_discovery"] = "safetensors"
	if len(plan.Ambiguous) > 0 {
		inspection.Labels["sequence_mixer_subpath_status"] = "ambiguous"
		inspection.Labels["sequence_mixer_subpath_ambiguous_layers"] = sequenceMixerAmbiguousSubpathCSV(plan.Ambiguous)
		return err
	}
	inspection.Labels["sequence_mixer_subpath_count"] = core.Sprintf("%d", len(plan.Subpaths))
	if len(plan.Subpaths) == 0 {
		inspection.Labels["sequence_mixer_subpath_status"] = "bare"
		return err
	}
	inspection.Labels["sequence_mixer_subpath_status"] = "ok"
	inspection.Labels["sequence_mixer_subpaths"] = sequenceMixerSubpathCSV(plan.Subpaths)
	return err
}

func sequenceMixerLayerTypesFromLabels(labels map[string]string) []string {
	raw := labels["attention_layer_types"]
	if raw == "" {
		return nil
	}
	parts := strings.Split(raw, ",")
	layerTypes := make([]string, 0, len(parts))
	for _, part := range parts {
		if kind := NormalizeDenseLayerType(part); kind != "" {
			layerTypes = append(layerTypes, kind)
		}
	}
	return layerTypes
}

func rocmApplySequenceMixerLoadPlanLabels(labels map[string]string, plan SequenceMixerLoadPlan, err error) {
	if labels == nil {
		return
	}
	labels["sequence_mixer_load_plan"] = SequenceMixerRuntimePlannedHIP
	labels["sequence_mixer_load_plan_contract"] = SequenceMixerRegistryContract
	if err != nil {
		labels["sequence_mixer_load_plan_status"] = "invalid"
		labels["sequence_mixer_load_plan_error"] = err.Error()
		return
	}
	labels["sequence_mixer_load_plan_status"] = "valid"
	labels["sequence_mixer_load_plan_layers"] = core.Sprintf("%d", len(plan.Layers))
	labels["sequence_mixer_load_plan_entries"] = sequenceMixerLoadPlanCSV(plan.Layers)
	labels["sequence_mixer_cache_plan_contract"] = plan.Cache.Contract
	labels["sequence_mixer_cache_factory_contract"] = SequenceMixerCacheFactoryContract
	labels["sequence_mixer_cache_factory_modes"] = core.Join(",", sequenceMixerCacheFactoryModes()...)
	labels["sequence_mixer_registered_cache_modes"] = core.Join(",", sequenceMixerRegisteredCacheModeEntries()...)
	labels["sequence_mixer_state_slots_contract"] = SequenceMixerStateSlotsContract
	labels["sequence_mixer_registered_state_slots"] = core.Join(",", sequenceMixerRegisteredStateSlotEntries()...)
	labels["sequence_mixer_state_slot_counts"] = core.Join(",", sequenceMixerStateSlotCountEntries()...)
	labels["sequence_mixer_cache_plan_layers"] = core.Sprintf("%d", len(plan.Cache.Layers))
	labels["sequence_mixer_cache_plan_entries"] = sequenceMixerCachePlanCSV(plan.Cache.Layers)
	if slots := sequenceMixerCachePlanSlotCSV(plan.Cache.Layers); slots != "" {
		labels["sequence_mixer_cache_plan_state_slots"] = slots
	}
}

func sequenceMixerSubpathCSV(subpaths map[int]string) string {
	return rocmmodel.SequenceMixerSubpathCSV(subpaths)
}

func sequenceMixerLoadPlanCSV(layers []SequenceMixerLayerPlan) string {
	return rocmmodel.SequenceMixerLoadPlanCSV(layers)
}

func sequenceMixerCachePlanCSV(layers []SequenceMixerCacheLayerPlan) string {
	return rocmmodel.SequenceMixerCachePlanCSV(layers)
}

func sequenceMixerCachePlanSlotCSV(layers []SequenceMixerCacheLayerPlan) string {
	return rocmmodel.SequenceMixerCachePlanSlotCSV(layers)
}

func cloneSequenceMixerFamily(family SequenceMixerFamily) SequenceMixerFamily {
	return family.Clone()
}

func cloneSequenceMixerFamilies(families []SequenceMixerFamily) []SequenceMixerFamily {
	return rocmmodel.CloneSequenceMixerFamilies(families)
}

func cloneSequenceMixerLayerPlans(layers []SequenceMixerLayerPlan) []SequenceMixerLayerPlan {
	return rocmmodel.CloneSequenceMixerLayerPlans(layers)
}

func cloneSequenceMixerCacheLayerPlans(layers []SequenceMixerCacheLayerPlan) []SequenceMixerCacheLayerPlan {
	return rocmmodel.CloneSequenceMixerCacheLayerPlans(layers)
}

func sequenceMixerAmbiguousSubpathCSV(ambiguous map[int][]string) string {
	return rocmmodel.SequenceMixerAmbiguousSubpathCSV(ambiguous)
}
