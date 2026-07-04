// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"slices"
	"strconv"
	"strings"

	core "dappco.re/go"
	"dappco.re/go/inference/engine/hip/internal/registry"
	rocmscheme "dappco.re/go/inference/engine/hip/scheme"
)

const (
	SequenceMixerRuntimePlannedHIP       = "planned_hip"
	SequenceMixerRegistryContract        = "go_mlx_config_composed_mixer_registry"
	SequenceMixerStateKVCache            = "kv-cache"
	SequenceMixerStateRecurrent          = "recurrent"
	SequenceMixerStateContract           = "go_mlx_scheme_state_kind"
	SequenceMixerStateSlotsContract      = "go_mlx_recurrent_state_slots"
	SequenceMixerCachePlanContract       = "go_mlx_composed_cache_state_plan"
	SequenceMixerCacheFactoryContract    = "go_mlx_cache_factory"
	SequenceMixerCacheModeDefault        = rocmscheme.CacheModeDefault
	SequenceMixerCacheModeRecurrent      = rocmscheme.CacheModeRecurrent
	SequenceMixerCacheModeMLALatent      = rocmscheme.CacheModeMLALatent
	SequenceMixerCacheModeCompaction     = rocmscheme.CacheModeCompaction
	SequenceMixerCacheModeCompactionFull = rocmscheme.CacheModeCompactionFull
	SequenceMixerRequiredLeavesContract  = "go_mlx_composed_mixer_required_leaves"

	SequenceMixerLoaderRouteName = "sequence-mixer-loader"
)

// SequenceMixerFamily describes one config-composed sequence mixer kind ROCm
// can recognise and plan for. It is model-owned metadata; runtime packages bind
// the plan to HIP/CUDA/CPU tensors later.
type SequenceMixerFamily struct {
	Kind       string   `json:"kind"`
	State      string   `json:"state"`
	CacheMode  string   `json:"cache_mode"`
	StateSlots []string `json:"state_slots,omitempty"`
	Source     string   `json:"source"`
	Runtime    string   `json:"runtime"`
}

// Clone returns a copy with independent state-slot storage.
func (family SequenceMixerFamily) Clone() SequenceMixerFamily {
	family.StateSlots = append([]string(nil), family.StateSlots...)
	return family
}

// CloneSequenceMixerFamilies returns independent family copies.
func CloneSequenceMixerFamilies(families []SequenceMixerFamily) []SequenceMixerFamily {
	out := append([]SequenceMixerFamily(nil), families...)
	for index := range out {
		out[index] = out[index].Clone()
	}
	return out
}

type SequenceMixerRegistration struct {
	Family         SequenceMixerFamily `json:"family"`
	RequiredLeaves []string            `json:"required_leaves,omitempty"`
}

func (registration SequenceMixerRegistration) Clone() SequenceMixerRegistration {
	return SequenceMixerRegistration{
		Family:         registration.Family.Clone(),
		RequiredLeaves: append([]string(nil), registration.RequiredLeaves...),
	}
}

// SequenceMixerSubpathPlan records checkpoint-derived mixer sublayer routing.
type SequenceMixerSubpathPlan struct {
	LayerCount int              `json:"layer_count"`
	Subpaths   map[int]string   `json:"subpaths,omitempty"`
	Ambiguous  map[int][]string `json:"ambiguous,omitempty"`
}

// Clone returns a copy with independent subpath maps.
func (plan SequenceMixerSubpathPlan) Clone() SequenceMixerSubpathPlan {
	out := SequenceMixerSubpathPlan{
		LayerCount: plan.LayerCount,
		Subpaths:   make(map[int]string, len(plan.Subpaths)),
		Ambiguous:  make(map[int][]string, len(plan.Ambiguous)),
	}
	for layer, subpath := range plan.Subpaths {
		out.Subpaths[layer] = subpath
	}
	for layer, ambiguous := range plan.Ambiguous {
		out.Ambiguous[layer] = append([]string(nil), ambiguous...)
	}
	return out
}

// SequenceMixerLayerPlan is the model-owned side of the config-composed loader
// contract: one normalized mixer kind, state shape, and checkpoint subpath.
type SequenceMixerLayerPlan struct {
	Layer      int      `json:"layer"`
	Kind       string   `json:"kind"`
	State      string   `json:"state"`
	StateSlots []string `json:"state_slots,omitempty"`
	Source     string   `json:"source"`
	Runtime    string   `json:"runtime"`
	Subpath    string   `json:"subpath,omitempty"`
}

// Clone returns a copy with independent state-slot storage.
func (plan SequenceMixerLayerPlan) Clone() SequenceMixerLayerPlan {
	plan.StateSlots = append([]string(nil), plan.StateSlots...)
	return plan
}

// CloneSequenceMixerLayerPlans returns independent layer-plan copies.
func CloneSequenceMixerLayerPlans(layers []SequenceMixerLayerPlan) []SequenceMixerLayerPlan {
	out := append([]SequenceMixerLayerPlan(nil), layers...)
	for index := range out {
		out[index] = out[index].Clone()
	}
	return out
}

type SequenceMixerCacheLayerPlan struct {
	Layer      int      `json:"layer"`
	Kind       string   `json:"kind"`
	State      string   `json:"state"`
	Holder     string   `json:"holder"`
	Mode       string   `json:"mode"`
	StateSlots []string `json:"state_slots,omitempty"`
}

// Clone returns a copy with independent state-slot storage.
func (plan SequenceMixerCacheLayerPlan) Clone() SequenceMixerCacheLayerPlan {
	plan.StateSlots = append([]string(nil), plan.StateSlots...)
	return plan
}

// CloneSequenceMixerCacheLayerPlans returns independent cache-layer copies.
func CloneSequenceMixerCacheLayerPlans(layers []SequenceMixerCacheLayerPlan) []SequenceMixerCacheLayerPlan {
	out := append([]SequenceMixerCacheLayerPlan(nil), layers...)
	for index := range out {
		out[index] = out[index].Clone()
	}
	return out
}

type SequenceMixerCachePlan struct {
	Contract string                        `json:"contract"`
	Layers   []SequenceMixerCacheLayerPlan `json:"layers"`
}

// Clone returns a copy with independent cache-layer storage.
func (plan SequenceMixerCachePlan) Clone() SequenceMixerCachePlan {
	return SequenceMixerCachePlan{
		Contract: plan.Contract,
		Layers:   CloneSequenceMixerCacheLayerPlans(plan.Layers),
	}
}

type SequenceMixerLoadPlan struct {
	Contract string                   `json:"contract"`
	Runtime  string                   `json:"runtime"`
	Layers   []SequenceMixerLayerPlan `json:"layers"`
	Subpaths SequenceMixerSubpathPlan `json:"subpaths"`
	Cache    SequenceMixerCachePlan   `json:"cache"`
}

// Clone returns a copy with independent layers, subpath maps, and cache plan.
func (plan SequenceMixerLoadPlan) Clone() SequenceMixerLoadPlan {
	return SequenceMixerLoadPlan{
		Contract: plan.Contract,
		Runtime:  plan.Runtime,
		Layers:   CloneSequenceMixerLayerPlans(plan.Layers),
		Subpaths: plan.Subpaths.Clone(),
		Cache:    plan.Cache.Clone(),
	}
}

// CloneSequenceMixerLoadPlan returns an independent copy, preserving nil.
func CloneSequenceMixerLoadPlan(plan *SequenceMixerLoadPlan) *SequenceMixerLoadPlan {
	if plan == nil {
		return nil
	}
	cloned := plan.Clone()
	return &cloned
}

// SequenceMixerLoaderRoute is the model-owned route view for go-mlx's
// mixer-loader registry surface.
type SequenceMixerLoaderRoute struct {
	Contract       string            `json:"contract,omitempty"`
	Name           string            `json:"name,omitempty"`
	Kind           string            `json:"kind,omitempty"`
	Loader         string            `json:"loader,omitempty"`
	State          string            `json:"state,omitempty"`
	CacheMode      string            `json:"cache_mode,omitempty"`
	StateSlots     []string          `json:"state_slots,omitempty"`
	Source         string            `json:"source,omitempty"`
	Runtime        string            `json:"runtime,omitempty"`
	RequiredLeaves []string          `json:"required_leaves,omitempty"`
	Registered     bool              `json:"registered,omitempty"`
	NativeRuntime  bool              `json:"native_runtime,omitempty"`
	Planned        bool              `json:"planned,omitempty"`
	Labels         map[string]string `json:"labels,omitempty"`
}

func (route SequenceMixerLoaderRoute) Matched() bool {
	return route.Contract != "" && route.Kind != "" && route.Loader != ""
}

func (route SequenceMixerLoaderRoute) Clone() SequenceMixerLoaderRoute {
	route.StateSlots = append([]string(nil), route.StateSlots...)
	route.RequiredLeaves = append([]string(nil), route.RequiredLeaves...)
	route.Labels = cloneStringMap(route.Labels)
	return route
}

type registeredSequenceMixerFamily struct {
	Family         SequenceMixerFamily
	RequiredLeaves []string
}

type sequenceMixerSchemeInfo struct {
	kind      string
	state     rocmscheme.StateKind
	cacheMode string
}

func (mixer sequenceMixerSchemeInfo) Kind() string { return mixer.kind }
func (mixer sequenceMixerSchemeInfo) State() rocmscheme.StateKind {
	return mixer.state
}
func (mixer sequenceMixerSchemeInfo) CacheMode() string { return mixer.cacheMode }

type sequenceMixerCacheSchemeInfo struct {
	mode   string
	serves rocmscheme.StateKind
}

func (cache sequenceMixerCacheSchemeInfo) Mode() string { return cache.mode }
func (cache sequenceMixerCacheSchemeInfo) Serves() rocmscheme.StateKind {
	return cache.serves
}

func (registration registeredSequenceMixerFamily) clone() registeredSequenceMixerFamily {
	return registeredSequenceMixerFamily{
		Family:         registration.Family.Clone(),
		RequiredLeaves: append([]string(nil), registration.RequiredLeaves...),
	}
}

var registeredSequenceMixerFamilies = registry.NewOrdered[string, registeredSequenceMixerFamily]()

func NormalizeSequenceMixerKind(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	value = strings.ReplaceAll(value, "-", "_")
	value = strings.ReplaceAll(value, ".", "_")
	return strings.ReplaceAll(value, " ", "_")
}

// RegisterSequenceMixerFamily registers or replaces a sequence-mixer family in
// the model-owned planning registry.
func RegisterSequenceMixerFamily(family SequenceMixerFamily, requiredLeaves []string) {
	family = normalizeSequenceMixerFamily(family)
	if family.Kind == "" {
		return
	}
	registerSequenceMixerFamilyScheme(family)
	registeredSequenceMixerFamilies.Put(family.Kind, registeredSequenceMixerFamily{
		Family:         family.Clone(),
		RequiredLeaves: normalizedSequenceMixerRequiredLeaves(requiredLeaves),
	})
}

func RegisteredSequenceMixerFamilyKinds() []string {
	return registeredSequenceMixerFamilies.Keys()
}

func RegisteredSequenceMixerFamilies() []SequenceMixerRegistration {
	registrations := registeredSequenceMixerFamilies.Values()
	out := make([]SequenceMixerRegistration, 0, len(registrations))
	for _, registration := range registrations {
		out = append(out, SequenceMixerRegistration{
			Family:         registration.Family.Clone(),
			RequiredLeaves: append([]string(nil), registration.RequiredLeaves...),
		})
	}
	return out
}

func ReplaceRegisteredSequenceMixerFamilies(registrations []SequenceMixerRegistration) {
	order := make([]string, 0, len(registrations))
	values := make(map[string]registeredSequenceMixerFamily, len(registrations))
	for _, registration := range registrations {
		family := normalizeSequenceMixerFamily(registration.Family)
		if family.Kind == "" {
			continue
		}
		if _, ok := values[family.Kind]; !ok {
			order = append(order, family.Kind)
		}
		registerSequenceMixerFamilyScheme(family)
		values[family.Kind] = registeredSequenceMixerFamily{
			Family:         family.Clone(),
			RequiredLeaves: normalizedSequenceMixerRequiredLeaves(registration.RequiredLeaves),
		}
	}
	registeredSequenceMixerFamilies.Restore(order, values)
}

func DefaultSequenceMixerFamilies() []SequenceMixerFamily {
	families := CloneSequenceMixerFamilies(builtinSequenceMixerFamilies())
	index := make(map[string]int, len(families))
	for i, family := range families {
		index[family.Kind] = i
	}
	for _, registration := range registeredSequenceMixerFamilies.Values() {
		family := registration.Family.Clone()
		family.CacheMode = sequenceMixerCacheModeForFamily(family)
		if existing, ok := index[family.Kind]; ok {
			families[existing] = family
			continue
		}
		index[family.Kind] = len(families)
		families = append(families, family)
	}
	for i := range families {
		families[i].CacheMode = sequenceMixerCacheModeForFamily(families[i])
	}
	return CloneSequenceMixerFamilies(families)
}

func builtinSequenceMixerFamilies() []SequenceMixerFamily {
	return []SequenceMixerFamily{
		normalizeSequenceMixerFamily(SequenceMixerFamily{Kind: "full_attention", State: SequenceMixerStateKVCache, Source: "generic_softmax", Runtime: SequenceMixerRuntimePlannedHIP}),
		normalizeSequenceMixerFamily(SequenceMixerFamily{Kind: "mamba2", State: SequenceMixerStateRecurrent, StateSlots: []string{"conv_state", "ssm_state"}, Source: "fla", Runtime: SequenceMixerRuntimePlannedHIP}),
		normalizeSequenceMixerFamily(SequenceMixerFamily{Kind: "rwkv7", State: SequenceMixerStateRecurrent, StateSlots: []string{"wkv_state"}, Source: "fla", Runtime: SequenceMixerRuntimePlannedHIP}),
		normalizeSequenceMixerFamily(SequenceMixerFamily{Kind: "gla", State: SequenceMixerStateRecurrent, StateSlots: []string{"gated_linear_state"}, Source: "fla", Runtime: SequenceMixerRuntimePlannedHIP}),
		normalizeSequenceMixerFamily(SequenceMixerFamily{Kind: "retnet", State: SequenceMixerStateRecurrent, StateSlots: []string{"retention_state"}, Source: "fla", Runtime: SequenceMixerRuntimePlannedHIP}),
		normalizeSequenceMixerFamily(SequenceMixerFamily{Kind: "deltanet", State: SequenceMixerStateRecurrent, StateSlots: []string{"value_memory_state"}, Source: "fla", Runtime: SequenceMixerRuntimePlannedHIP}),
		normalizeSequenceMixerFamily(SequenceMixerFamily{Kind: "gsa", State: SequenceMixerStateRecurrent, StateSlots: []string{"slot_key_state", "slot_value_state"}, Source: "fla", Runtime: SequenceMixerRuntimePlannedHIP}),
		normalizeSequenceMixerFamily(SequenceMixerFamily{Kind: "nsa", State: SequenceMixerStateKVCache, Source: "fla", Runtime: SequenceMixerRuntimePlannedHIP}),
		normalizeSequenceMixerFamily(SequenceMixerFamily{Kind: "moba", State: SequenceMixerStateKVCache, Source: "fla", Runtime: SequenceMixerRuntimePlannedHIP}),
		normalizeSequenceMixerFamily(SequenceMixerFamily{Kind: "mla", State: SequenceMixerStateKVCache, Source: "fla", Runtime: SequenceMixerRuntimePlannedHIP}),
	}
}

func SequenceMixerFamilyByKind(kind string) (SequenceMixerFamily, bool) {
	kind = NormalizeSequenceMixerKind(kind)
	for _, family := range DefaultSequenceMixerFamilies() {
		if family.Kind == kind {
			family.CacheMode = sequenceMixerCacheModeForFamily(family)
			return family.Clone(), true
		}
	}
	return SequenceMixerFamily{}, false
}

func DefaultSequenceMixerCacheFactoryModes() []string {
	return rocmscheme.CacheModes()
}

func SequenceMixerCacheModeForKind(kind string) (string, bool) {
	family, ok := SequenceMixerFamilyByKind(kind)
	if !ok || family.CacheMode == "" {
		return "", false
	}
	return family.CacheMode, true
}

func SequenceMixerStateSlotsForKind(kind string) ([]string, bool) {
	family, ok := SequenceMixerFamilyByKind(kind)
	if !ok {
		return nil, false
	}
	return append([]string(nil), family.StateSlots...), true
}

func SequenceMixerRequiredLeaves(kind string) ([]string, bool) {
	kind = NormalizeSequenceMixerKind(kind)
	if leaves, ok := registeredSequenceMixerRequiredLeaves(kind); ok {
		return leaves, true
	}
	leaves, ok := sequenceMixerRequiredLeavesByKind[kind]
	if !ok {
		return nil, false
	}
	return append([]string(nil), leaves...), true
}

var sequenceMixerRequiredLeavesByKind = map[string][]string{
	"full_attention": {"q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"},
	"mamba2":         {"in_proj.weight", "out_proj.weight", "conv1d.weight", "A_log"},
	"rwkv7":          {"receptance.weight", "key.weight", "value.weight", "output.weight", "decay.weight", "a_proj.weight", "b_proj.weight"},
	"gla":            {"q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight", "gk_proj.weight"},
	"retnet":         {"q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"},
	"deltanet":       {"q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight", "b_proj.weight"},
	"gsa":            {"q_proj.weight", "k_proj.weight", "v_proj.weight", "f_proj.weight", "g_proj.weight", "o_proj.weight"},
	"nsa":            {"q_proj.weight", "k_proj.weight", "v_proj.weight", "g_proj.weight", "o_proj.weight"},
	"moba":           {"q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"},
	"mla":            {"kv_a_proj_with_mqa.weight", "kv_b_proj.weight", "q_a_proj.weight", "q_b_proj.weight", "o_proj.weight"},
}

func SequenceMixerFamilyKinds() []string {
	families := DefaultSequenceMixerFamilies()
	kinds := make([]string, 0, len(families))
	for _, family := range families {
		kinds = append(kinds, family.Kind)
	}
	return kinds
}

func SequenceMixerFLAKinds() []string {
	families := DefaultSequenceMixerFamilies()
	kinds := make([]string, 0, len(families))
	for _, family := range families {
		if family.Source == "fla" {
			kinds = append(kinds, family.Kind)
		}
	}
	return kinds
}

func SequenceMixerRegisteredStateEntries() []string {
	families := DefaultSequenceMixerFamilies()
	entries := make([]string, 0, len(families))
	for _, family := range families {
		entries = append(entries, family.Kind+":"+family.State)
	}
	return entries
}

func SequenceMixerRegisteredCacheModeEntries() []string {
	families := DefaultSequenceMixerFamilies()
	entries := make([]string, 0, len(families))
	for _, family := range families {
		entries = append(entries, family.Kind+":"+family.CacheMode)
	}
	return entries
}

func SequenceMixerRegisteredStateSlotEntries() []string {
	families := DefaultSequenceMixerFamilies()
	entries := make([]string, 0, len(families))
	for _, family := range families {
		if len(family.StateSlots) == 0 {
			continue
		}
		entries = append(entries, family.Kind+":"+core.Join("|", family.StateSlots...))
	}
	return entries
}

func SequenceMixerStateSlotCountEntries() []string {
	families := DefaultSequenceMixerFamilies()
	entries := make([]string, 0, len(families))
	for _, family := range families {
		if family.State != SequenceMixerStateRecurrent {
			continue
		}
		entries = append(entries, family.Kind+":"+strconv.Itoa(len(family.StateSlots)))
	}
	return entries
}

func SequenceMixerRequiredLeafEntries() []string {
	families := DefaultSequenceMixerFamilies()
	entries := make([]string, 0, len(families))
	for _, family := range families {
		leaves, ok := SequenceMixerRequiredLeaves(family.Kind)
		if !ok {
			continue
		}
		entries = append(entries, family.Kind+":"+core.Join("|", leaves...))
	}
	return entries
}

func SequenceMixerLayerCounts(layerTypes []string) map[string]int {
	counts := make(map[string]int, len(layerTypes))
	for _, layerType := range layerTypes {
		kind := NormalizeSequenceMixerKind(layerType)
		if kind == "" {
			continue
		}
		counts[kind]++
	}
	return counts
}

func SequenceMixerUniqueKinds(layerTypes []string) []string {
	seen := map[string]bool{}
	kinds := make([]string, 0, len(layerTypes))
	for _, layerType := range layerTypes {
		kind := NormalizeSequenceMixerKind(layerType)
		if kind == "" || seen[kind] {
			continue
		}
		seen[kind] = true
		kinds = append(kinds, kind)
	}
	return kinds
}

func NormalizeSequenceMixerLayerTypes(values []string) []string {
	out := make([]string, 0, len(values))
	for _, value := range values {
		if normalized := NormalizeSequenceMixerKind(value); normalized != "" {
			out = append(out, normalized)
		}
	}
	return out
}

func DefaultSequenceMixerLoaderRoutes() []SequenceMixerLoaderRoute {
	families := DefaultSequenceMixerFamilies()
	routes := make([]SequenceMixerLoaderRoute, 0, len(families))
	for _, family := range families {
		route := SequenceMixerLoaderRouteForFamily(family)
		if !route.Matched() {
			continue
		}
		routes = append(routes, route)
	}
	return routes
}

func SequenceMixerLoaderRouteForKind(kind string) (SequenceMixerLoaderRoute, bool) {
	family, ok := SequenceMixerFamilyByKind(kind)
	if !ok {
		return SequenceMixerLoaderRoute{}, false
	}
	return SequenceMixerLoaderRouteForFamily(family), true
}

func SequenceMixerLoaderRouteForFamily(family SequenceMixerFamily) SequenceMixerLoaderRoute {
	family = normalizeSequenceMixerFamily(family)
	leaves, _ := SequenceMixerRequiredLeaves(family.Kind)
	route := SequenceMixerLoaderRoute{
		Contract:       SequenceMixerRegistryContract,
		Name:           SequenceMixerLoaderRouteName,
		Kind:           family.Kind,
		Loader:         family.Kind,
		State:          family.State,
		CacheMode:      family.CacheMode,
		StateSlots:     append([]string(nil), family.StateSlots...),
		Source:         family.Source,
		Runtime:        family.Runtime,
		RequiredLeaves: leaves,
		Registered:     true,
		NativeRuntime:  false,
		Planned:        family.Runtime == SequenceMixerRuntimePlannedHIP,
	}
	route.Labels = sequenceMixerLoaderRouteLabels(route)
	return route.Clone()
}

// BuildSequenceMixerLoadPlan validates a config-composed mixer plan the same
// way go-mlx's composed runner does before load.
func BuildSequenceMixerLoadPlan(layerTypes []string, tensorNames []string, numLayers int) (SequenceMixerLoadPlan, error) {
	plan := SequenceMixerLoadPlan{
		Contract: SequenceMixerRegistryContract,
		Runtime:  SequenceMixerRuntimePlannedHIP,
	}
	if numLayers <= 0 {
		numLayers = len(layerTypes)
	}
	plan.Subpaths = DiscoverSequenceMixerSubpaths(tensorNames, numLayers)
	if numLayers <= 0 {
		return plan, core.NewError("num_hidden_layers must be > 0")
	}
	if len(layerTypes) != numLayers {
		return plan, core.NewError(core.Sprintf("layer_types length %d != num_hidden_layers %d", len(layerTypes), numLayers))
	}
	if len(plan.Subpaths.Ambiguous) > 0 {
		return plan, core.NewError("sequence mixer subpath is ambiguous: " + SequenceMixerAmbiguousSubpathCSV(plan.Subpaths.Ambiguous))
	}
	tensorNameSet := make(map[string]bool, len(tensorNames))
	for _, name := range tensorNames {
		tensorNameSet[name] = true
	}
	plan.Layers = make([]SequenceMixerLayerPlan, 0, numLayers)
	for layer, raw := range layerTypes {
		kind := NormalizeSequenceMixerKind(raw)
		family, ok := SequenceMixerFamilyByKind(kind)
		if !ok {
			return plan, core.NewError(core.Sprintf("layer %d: unregistered mixer kind %q", layer, kind))
		}
		subpath := plan.Subpaths.Subpaths[layer]
		if missing := sequenceMixerMissingRequiredLeaves(tensorNameSet, layer, family.Kind, subpath); len(missing) > 0 {
			return plan, core.NewError(core.Sprintf("layer %d %s missing required mixer tensors %s", layer, family.Kind, core.Join(",", missing...)))
		}
		plan.Layers = append(plan.Layers, SequenceMixerLayerPlan{
			Layer:      layer,
			Kind:       family.Kind,
			State:      family.State,
			StateSlots: append([]string(nil), family.StateSlots...),
			Source:     family.Source,
			Runtime:    family.Runtime,
			Subpath:    subpath,
		})
	}
	cache, err := BuildSequenceMixerCachePlan(plan.Layers)
	if err != nil {
		return plan, err
	}
	plan.Cache = cache
	return plan, nil
}

func BuildSequenceMixerCachePlan(layers []SequenceMixerLayerPlan) (SequenceMixerCachePlan, error) {
	plan := SequenceMixerCachePlan{
		Contract: SequenceMixerCachePlanContract,
		Layers:   make([]SequenceMixerCacheLayerPlan, 0, len(layers)),
	}
	for _, layer := range layers {
		holder, err := sequenceMixerCacheHolderForState(layer.State)
		if err != nil {
			return plan, core.E("model.SequenceMixerCachePlan", core.Sprintf("layer %d %s", layer.Layer, layer.Kind), err)
		}
		mode, err := sequenceMixerCacheModeForLayer(layer)
		if err != nil {
			return plan, core.E("model.SequenceMixerCachePlan", core.Sprintf("layer %d %s", layer.Layer, layer.Kind), err)
		}
		slots, err := sequenceMixerStateSlotsForLayer(layer)
		if err != nil {
			return plan, core.E("model.SequenceMixerCachePlan", core.Sprintf("layer %d %s", layer.Layer, layer.Kind), err)
		}
		plan.Layers = append(plan.Layers, SequenceMixerCacheLayerPlan{
			Layer:      layer.Layer,
			Kind:       layer.Kind,
			State:      layer.State,
			Holder:     holder,
			Mode:       mode,
			StateSlots: slots,
		})
	}
	return plan, nil
}

// DiscoverSequenceMixerSubpaths finds the checkpoint sublayer that owns each
// layer's mixer weights. Feed-forward owners are ignored.
func DiscoverSequenceMixerSubpaths(names []string, numLayers int) SequenceMixerSubpathPlan {
	plan := SequenceMixerSubpathPlan{
		Subpaths:  map[int]string{},
		Ambiguous: map[int][]string{},
	}
	layerSubs := map[int]map[string]struct{}{}
	maxLayer := -1
	for _, name := range names {
		layer, subpath, ok := sequenceMixerTensorSubpath(name)
		if !ok {
			continue
		}
		if layer > maxLayer {
			maxLayer = layer
		}
		if layerSubs[layer] == nil {
			layerSubs[layer] = map[string]struct{}{}
		}
		layerSubs[layer][subpath] = struct{}{}
	}
	if numLayers <= 0 {
		numLayers = maxLayer + 1
	}
	plan.LayerCount = numLayers
	for layer, subs := range layerSubs {
		if numLayers > 0 && layer >= numLayers {
			continue
		}
		switch len(subs) {
		case 0:
			continue
		case 1:
			for subpath := range subs {
				plan.Subpaths[layer] = subpath
			}
		default:
			values := make([]string, 0, len(subs))
			for subpath := range subs {
				values = append(values, subpath)
			}
			slices.Sort(values)
			plan.Ambiguous[layer] = values
		}
	}
	return plan
}

func SequenceMixerWeightNameCandidates(name string) []string {
	candidates := []string{name}
	if strings.HasPrefix(name, "model.") {
		suffix := strings.TrimPrefix(name, "model.")
		return append(candidates,
			"language_model."+name,
			"language_model.model."+suffix,
			"model.language_model."+suffix,
			"model.language_model.model."+suffix,
		)
	}
	return append(candidates,
		"model."+name,
		"language_model."+name,
		"language_model.model."+name,
		"model.language_model."+name,
		"model.language_model.model."+name,
	)
}

func SequenceMixerHasResolvedWeightName(names map[string]bool, name string) bool {
	for _, candidate := range SequenceMixerWeightNameCandidates(name) {
		if names[candidate] {
			return true
		}
	}
	return false
}

func SequenceMixerSubpathCSV(subpaths map[int]string) string {
	layers := make([]int, 0, len(subpaths))
	for layer := range subpaths {
		layers = append(layers, layer)
	}
	slices.Sort(layers)
	parts := make([]string, 0, len(layers))
	for _, layer := range layers {
		parts = append(parts, core.Sprintf("%d:%s", layer, subpaths[layer]))
	}
	return core.Join(",", parts...)
}

func SequenceMixerLoadPlanCSV(layers []SequenceMixerLayerPlan) string {
	parts := make([]string, 0, len(layers))
	for _, layer := range layers {
		subpath := layer.Subpath
		if subpath == "" {
			subpath = "bare"
		}
		parts = append(parts, core.Sprintf("%d:%s:%s:%s:%s", layer.Layer, layer.Kind, layer.State, subpath, layer.Runtime))
	}
	return core.Join(",", parts...)
}

func SequenceMixerCachePlanCSV(layers []SequenceMixerCacheLayerPlan) string {
	parts := make([]string, 0, len(layers))
	for _, layer := range layers {
		parts = append(parts, core.Sprintf("%d:%s:%s:%s", layer.Layer, layer.Kind, layer.Holder, layer.Mode))
	}
	return core.Join(",", parts...)
}

func SequenceMixerCachePlanSlotCSV(layers []SequenceMixerCacheLayerPlan) string {
	parts := make([]string, 0, len(layers))
	for _, layer := range layers {
		if len(layer.StateSlots) == 0 {
			continue
		}
		parts = append(parts, core.Sprintf("%d:%s:%s", layer.Layer, layer.Kind, core.Join("|", layer.StateSlots...)))
	}
	return core.Join(",", parts...)
}

func SequenceMixerAmbiguousSubpathCSV(ambiguous map[int][]string) string {
	layers := make([]int, 0, len(ambiguous))
	for layer := range ambiguous {
		layers = append(layers, layer)
	}
	slices.Sort(layers)
	parts := make([]string, 0, len(layers))
	for _, layer := range layers {
		parts = append(parts, core.Sprintf("%d:%s", layer, core.Join("|", ambiguous[layer]...)))
	}
	return core.Join(",", parts...)
}

func normalizeSequenceMixerFamily(family SequenceMixerFamily) SequenceMixerFamily {
	family.Kind = NormalizeSequenceMixerKind(family.Kind)
	if family.Kind == "" {
		return SequenceMixerFamily{}
	}
	switch family.State {
	case SequenceMixerStateKVCache:
		if family.CacheMode == "" {
			family.CacheMode = rocmscheme.CacheModeForMixer(sequenceMixerSchemeInfo{
				kind:  family.Kind,
				state: rocmscheme.StateKVCache,
			})
		}
	case SequenceMixerStateRecurrent:
		if family.CacheMode == "" {
			family.CacheMode = rocmscheme.CacheModeForMixer(sequenceMixerSchemeInfo{
				kind:  family.Kind,
				state: rocmscheme.StateRecurrent,
			})
		}
	default:
		return SequenceMixerFamily{}
	}
	if family.Source == "" {
		family.Source = "registered"
	}
	if family.Runtime == "" {
		family.Runtime = SequenceMixerRuntimePlannedHIP
	}
	family.StateSlots = append([]string(nil), family.StateSlots...)
	return family
}

func registerSequenceMixerFamilyScheme(family SequenceMixerFamily) {
	state, ok := sequenceMixerSchemeStateForString(family.State)
	if !ok {
		return
	}
	rocmscheme.RegisterMixer(sequenceMixerSchemeInfo{
		kind:      family.Kind,
		state:     state,
		cacheMode: family.CacheMode,
	})
	if family.CacheMode == "" {
		return
	}
	if _, ok := rocmscheme.CacheFor(family.CacheMode); ok {
		return
	}
	rocmscheme.RegisterCache(sequenceMixerCacheSchemeInfo{
		mode:   family.CacheMode,
		serves: state,
	})
}

func sequenceMixerCacheModeForFamily(family SequenceMixerFamily) string {
	if mixer, ok := rocmscheme.MixerFor(family.Kind); ok {
		if state, ok := sequenceMixerSchemeStateForString(family.State); ok && mixer.State() == state {
			if mode := rocmscheme.CacheModeForMixer(mixer); mode != "" {
				return mode
			}
		}
	}
	return strings.ToLower(strings.TrimSpace(family.CacheMode))
}

func sequenceMixerSchemeStateForString(state string) (rocmscheme.StateKind, bool) {
	switch state {
	case SequenceMixerStateKVCache:
		return rocmscheme.StateKVCache, true
	case SequenceMixerStateRecurrent:
		return rocmscheme.StateRecurrent, true
	default:
		return rocmscheme.StateNone, false
	}
}

func sequenceMixerStateForScheme(state rocmscheme.StateKind) (string, bool) {
	switch state {
	case rocmscheme.StateKVCache:
		return SequenceMixerStateKVCache, true
	case rocmscheme.StateRecurrent:
		return SequenceMixerStateRecurrent, true
	default:
		return "", false
	}
}

func registeredSequenceMixerRequiredLeaves(kind string) ([]string, bool) {
	kind = NormalizeSequenceMixerKind(kind)
	registration, ok := registeredSequenceMixerFamilies.Get(kind)
	if !ok || len(registration.RequiredLeaves) == 0 {
		return nil, false
	}
	return append([]string(nil), registration.RequiredLeaves...), true
}

func normalizedSequenceMixerRequiredLeaves(leaves []string) []string {
	out := make([]string, 0, len(leaves))
	seen := map[string]bool{}
	for _, leaf := range leaves {
		leaf = strings.TrimSpace(leaf)
		if leaf == "" || seen[leaf] {
			continue
		}
		seen[leaf] = true
		out = append(out, leaf)
	}
	return out
}

func sequenceMixerLoaderRouteLabels(route SequenceMixerLoaderRoute) map[string]string {
	if !route.Matched() {
		return nil
	}
	labels := map[string]string{
		"engine_mixer_loader_contract":          route.Contract,
		"engine_mixer_loader":                   route.Loader,
		"engine_mixer_loader_kind":              route.Kind,
		"engine_mixer_loader_registered":        strconv.FormatBool(route.Registered),
		"engine_mixer_loader_native":            strconv.FormatBool(route.NativeRuntime),
		"engine_mixer_loader_planned":           strconv.FormatBool(route.Planned),
		"engine_mixer_cache_factory_contract":   SequenceMixerCacheFactoryContract,
		"engine_mixer_cache_factory_modes":      core.Join(",", DefaultSequenceMixerCacheFactoryModes()...),
		"engine_mixer_state_slots_contract":     SequenceMixerStateSlotsContract,
		"engine_mixer_registered_state_slots":   core.Join(",", SequenceMixerRegisteredStateSlotEntries()...),
		"engine_mixer_state_slot_counts":        core.Join(",", SequenceMixerStateSlotCountEntries()...),
		"engine_mixer_required_leaves_contract": SequenceMixerRequiredLeavesContract,
	}
	if route.State != "" {
		labels["engine_mixer_loader_state"] = route.State
	}
	if route.CacheMode != "" {
		labels["engine_mixer_loader_cache_mode"] = route.CacheMode
	}
	if len(route.StateSlots) > 0 {
		labels["engine_mixer_loader_state_slots"] = core.Join(",", route.StateSlots...)
		labels["engine_mixer_loader_state_slot_count"] = strconv.Itoa(len(route.StateSlots))
	}
	if route.Source != "" {
		labels["engine_mixer_loader_source"] = route.Source
	}
	if route.Runtime != "" {
		labels["engine_mixer_loader_runtime"] = route.Runtime
	}
	if len(route.RequiredLeaves) > 0 {
		labels["engine_mixer_loader_required_leaves"] = core.Join(",", route.RequiredLeaves...)
	}
	return labels
}

func sequenceMixerMissingRequiredLeaves(tensorNames map[string]bool, layer int, kind, subpath string) []string {
	required, ok := SequenceMixerRequiredLeaves(kind)
	if !ok {
		return []string{"<unmapped:" + NormalizeSequenceMixerKind(kind) + ">"}
	}
	missing := make([]string, 0)
	for _, leaf := range required {
		if SequenceMixerHasResolvedWeightName(tensorNames, sequenceMixerRequiredTensorName(layer, subpath, leaf)) {
			continue
		}
		missing = append(missing, leaf)
	}
	return missing
}

func sequenceMixerRequiredTensorName(layer int, subpath, leaf string) string {
	name := core.Sprintf("model.layers.%d", layer)
	if normalized := NormalizeSequenceMixerKind(subpath); normalized != "" {
		name += "." + normalized
	}
	return name + "." + leaf
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
	mixer, ok := rocmscheme.MixerFor(layer.Kind)
	if !ok {
		if family.CacheMode != "" {
			return family.CacheMode, nil
		}
		return "", core.NewError("unregistered sequence mixer scheme " + layer.Kind)
	}
	mixerState, ok := sequenceMixerStateForScheme(mixer.State())
	if !ok {
		return "", core.NewError("unsupported sequence mixer scheme state for " + layer.Kind)
	}
	if mixerState != layer.State {
		return "", core.NewError("sequence mixer scheme state mismatch for " + layer.Kind)
	}
	cache, ok := rocmscheme.CacheForMixer(mixer)
	if !ok {
		return "", core.NewError("unregistered sequence mixer cache scheme " + rocmscheme.CacheModeForMixer(mixer))
	}
	if !rocmscheme.Compatible(mixer, cache) {
		return "", core.NewError("sequence mixer cache scheme mismatch for " + layer.Kind)
	}
	return cache.Mode(), nil
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

func sequenceMixerTensorSubpath(name string) (int, string, bool) {
	const prefix = "model.layers."
	if !strings.HasPrefix(name, prefix) {
		return 0, "", false
	}
	parts := strings.Split(name[len(prefix):], ".")
	if len(parts) < 4 {
		return 0, "", false
	}
	layer, err := strconv.Atoi(parts[0])
	if err != nil || layer < 0 {
		return 0, "", false
	}
	subpath := NormalizeSequenceMixerKind(parts[1])
	if sequenceMixerIgnoredSubpath(subpath) {
		return 0, "", false
	}
	return layer, subpath, true
}

func sequenceMixerIgnoredSubpath(subpath string) bool {
	switch NormalizeSequenceMixerKind(subpath) {
	case "", "mlp", "ffn", "feed_forward", "feedforward", "block_sparse_moe", "sparse_moe", "moe", "experts":
		return true
	default:
		return false
	}
}
