// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"slices"
	"sort"
	"strconv"
	"strings"

	core "dappco.re/go"
)

const hipSequenceMixerOperation = "rocm.hip.SequenceMixer"

type hipSequenceMixerBindings struct {
	Contract string
	Runtime  string
	Cache    SequenceMixerCachePlan
	Layers   []hipSequenceMixerLayerBinding
}

type hipSequenceMixerLayerBinding struct {
	Plan    SequenceMixerLayerPlan
	Tensors map[string]hipTensor
}

func (model *hipLoadedModel) bindSequenceMixerPlan() error {
	if model == nil {
		return core.E(hipSequenceMixerOperation, "loaded model is required", nil)
	}
	plan := model.sequenceMixerPlan
	if plan == nil {
		model.sequenceMixerBindings = nil
		return nil
	}
	if plan.Contract != SequenceMixerRegistryContract {
		return core.E(hipSequenceMixerOperation, "unsupported sequence mixer contract "+plan.Contract, nil)
	}
	if plan.Runtime != SequenceMixerRuntimePlannedHIP {
		return core.E(hipSequenceMixerOperation, "unsupported sequence mixer runtime "+plan.Runtime, nil)
	}
	cachePlan, err := sequenceMixerCachePlanForLoadPlan(plan)
	if err != nil {
		return err
	}
	bindings := &hipSequenceMixerBindings{
		Contract: plan.Contract,
		Runtime:  plan.Runtime,
		Cache:    cachePlan,
		Layers:   make([]hipSequenceMixerLayerBinding, 0, len(plan.Layers)),
	}
	for _, layerPlan := range plan.Layers {
		layerPlan.Kind = NormalizeDenseLayerType(layerPlan.Kind)
		layerPlan.Subpath = NormalizeDenseLayerType(layerPlan.Subpath)
		binding, err := model.bindSequenceMixerLayer(layerPlan)
		if err != nil {
			return err
		}
		bindings.Layers = append(bindings.Layers, binding)
	}
	model.sequenceMixerBindings = bindings
	return nil
}

func sequenceMixerCachePlanForLoadPlan(plan *SequenceMixerLoadPlan) (SequenceMixerCachePlan, error) {
	if plan == nil {
		return SequenceMixerCachePlan{}, core.E(hipSequenceMixerOperation, "sequence mixer plan is required", nil)
	}
	if plan.Cache.Contract == "" && len(plan.Cache.Layers) == 0 {
		return buildSequenceMixerCachePlan(plan.Layers)
	}
	if plan.Cache.Contract != SequenceMixerCachePlanContract {
		return SequenceMixerCachePlan{}, core.E(hipSequenceMixerOperation, "unsupported sequence mixer cache plan contract "+plan.Cache.Contract, nil)
	}
	if len(plan.Cache.Layers) != len(plan.Layers) {
		return SequenceMixerCachePlan{}, core.E(hipSequenceMixerOperation, core.Sprintf("sequence mixer cache plan layers %d != mixer layers %d", len(plan.Cache.Layers), len(plan.Layers)), nil)
	}
	cache := cloneSequenceMixerCachePlan(plan.Cache)
	for index, cacheLayer := range cache.Layers {
		layer := plan.Layers[index]
		holder, err := sequenceMixerCacheHolderForState(layer.State)
		if err != nil {
			return SequenceMixerCachePlan{}, err
		}
		mode, err := sequenceMixerCacheModeForLayer(layer)
		if err != nil {
			return SequenceMixerCachePlan{}, err
		}
		slots, err := sequenceMixerStateSlotsForLayer(layer)
		if err != nil {
			return SequenceMixerCachePlan{}, err
		}
		if len(layer.StateSlots) == 0 && len(slots) > 0 {
			plan.Layers[index].StateSlots = append([]string(nil), slots...)
			layer.StateSlots = plan.Layers[index].StateSlots
		}
		if cacheLayer.Mode == "" {
			cache.Layers[index].Mode = mode
			cacheLayer.Mode = mode
		}
		if len(cacheLayer.StateSlots) == 0 && len(slots) > 0 {
			cache.Layers[index].StateSlots = append([]string(nil), slots...)
			cacheLayer.StateSlots = cache.Layers[index].StateSlots
		}
		if cacheLayer.Layer != layer.Layer ||
			cacheLayer.Kind != layer.Kind ||
			cacheLayer.State != layer.State ||
			cacheLayer.Holder != holder ||
			cacheLayer.Mode != mode ||
			!slices.Equal(cacheLayer.StateSlots, slots) {
			return SequenceMixerCachePlan{}, core.E(hipSequenceMixerOperation, core.Sprintf("sequence mixer cache plan mismatch at layer %d", layer.Layer), nil)
		}
	}
	return cache, nil
}

func (model *hipLoadedModel) bindSequenceMixerLayer(plan SequenceMixerLayerPlan) (hipSequenceMixerLayerBinding, error) {
	if plan.Layer < 0 {
		return hipSequenceMixerLayerBinding{}, core.E(hipSequenceMixerOperation, "sequence mixer layer must be non-negative", nil)
	}
	family, ok := SequenceMixerFamilyByKind(plan.Kind)
	if !ok {
		return hipSequenceMixerLayerBinding{}, core.E(hipSequenceMixerOperation, "unregistered sequence mixer kind "+plan.Kind, nil)
	}
	plan.Kind = family.Kind
	plan.State = family.State
	plan.StateSlots = append([]string(nil), family.StateSlots...)
	plan.Source = family.Source
	if plan.Runtime == "" {
		plan.Runtime = family.Runtime
	}
	if plan.Runtime != SequenceMixerRuntimePlannedHIP {
		return hipSequenceMixerLayerBinding{}, core.E(hipSequenceMixerOperation, "unsupported sequence mixer layer runtime "+plan.Runtime, nil)
	}
	tensors := model.sequenceMixerTensorsForLayer(plan.Layer, plan.Subpath)
	requiredLeaves, ok := sequenceMixerRequiredLeaves(plan.Kind)
	if !ok {
		return hipSequenceMixerLayerBinding{}, core.E(hipSequenceMixerOperation, "unmapped sequence mixer kind "+plan.Kind, nil)
	}
	for _, leaf := range requiredLeaves {
		tensor, ok := model.sequenceMixerTensorByCanonical(plan.Layer, plan.Subpath, leaf)
		if !ok {
			tensor, ok = tensors[leaf]
		}
		if !ok {
			return hipSequenceMixerLayerBinding{}, core.E(hipSequenceMixerOperation, core.Sprintf("layer %d %s missing %s tensor", plan.Layer, plan.Kind, leaf), nil)
		}
		tensors[leaf] = tensor
	}
	return hipSequenceMixerLayerBinding{
		Plan:    plan,
		Tensors: tensors,
	}, nil
}

func (model *hipLoadedModel) sequenceMixerTensorByCanonical(layer int, subpath, leaf string) (hipTensor, bool) {
	if model == nil || leaf == "" {
		return hipTensor{}, false
	}
	canonical := core.Sprintf("model.layers.%d", layer)
	if subpath != "" {
		canonical += "." + NormalizeDenseLayerType(subpath)
	}
	canonical += "." + leaf
	for _, candidate := range DenseWeightNameCandidates(canonical) {
		tensor, ok := model.tensors[candidate]
		if ok && tensor.pointer != 0 {
			return tensor, true
		}
	}
	return hipTensor{}, false
}

func (model *hipLoadedModel) sequenceMixerTensorsForLayer(layer int, subpath string) map[string]hipTensor {
	tensors := map[string]hipTensor{}
	if model == nil {
		return tensors
	}
	names := make([]string, 0, len(model.tensors))
	for name := range model.tensors {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		tensor := model.tensors[name]
		leaf, ok := sequenceMixerTensorLeaf(name, layer, subpath)
		if !ok || tensor.pointer == 0 {
			continue
		}
		tensors[leaf] = tensor
	}
	return tensors
}

func sequenceMixerTensorLeaf(name string, layer int, subpath string) (string, bool) {
	index := strings.Index(name, "model.layers.")
	if index < 0 {
		return "", false
	}
	parts := strings.Split(name[index+len("model.layers."):], ".")
	if len(parts) < 2 {
		return "", false
	}
	layerID, err := strconv.Atoi(parts[0])
	if err != nil || layerID != layer {
		return "", false
	}
	subpath = NormalizeDenseLayerType(subpath)
	if subpath != "" {
		if len(parts) < 3 || NormalizeDenseLayerType(parts[1]) != subpath {
			return "", false
		}
		leaf := strings.Join(parts[2:], ".")
		return leaf, leaf != ""
	}
	leafStart := 1
	if ignoredSequenceMixerSubpath(parts[1]) {
		return "", false
	}
	leaf := strings.Join(parts[leafStart:], ".")
	return leaf, leaf != ""
}

func ignoredSequenceMixerSubpath(value string) bool {
	switch NormalizeDenseLayerType(value) {
	case "", "mlp", "block_sparse_moe", "shared_experts":
		return true
	default:
		return false
	}
}
