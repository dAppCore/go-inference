// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"strconv"
	"strings"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/engine/hip/internal/registry"
)

const (
	QuantSchemeRegistryContract = "go_mlx_weight_quant_scheme_registry"

	QuantSchemeRouteName         = "weight-quant-scheme"
	QuantSchemeRuntimeMetadata   = "metadata"
	QuantSchemeRuntimePlannedHIP = "planned_hip"

	QuantRuntimeMLXAffine = "mlx_affine"
	QuantRuntimeBF16      = "bf16"
	QuantRuntimeGGUF      = "gguf"
	QuantRuntimePlanned   = "planned_status"

	QuantGenerateLinked      = "linked"
	QuantGenerateLoadOnly    = "load_only"
	QuantGeneratePlannedOnly = "planned_only"

	QuantLoaderRegistryContract = "rocm-quant-loader-registry-v1"

	QuantLoaderRouteName              = "weight-quant-loader"
	QuantLoaderFamilyGemma4           = "gemma4"
	QuantLoaderArchitectureGemma4Text = "gemma4_text"
)

// QuantScheme is the model-owned weight-quant scheme catalogue entry. It lets
// families self-register quant metadata without importing the root rocm package.
type QuantScheme struct {
	Contract      string                         `json:"contract,omitempty"`
	Name          string                         `json:"name,omitempty"`
	Kind          string                         `json:"kind,omitempty"`
	Bits          int                            `json:"bits,omitempty"`
	Loader        string                         `json:"loader,omitempty"`
	Source        string                         `json:"source,omitempty"`
	Runtime       string                         `json:"runtime,omitempty"`
	RuntimeStatus inference.FeatureRuntimeStatus `json:"runtime_status,omitempty"`
	Registered    bool                           `json:"registered,omitempty"`
	NativeRuntime bool                           `json:"native_runtime,omitempty"`
	MetadataOnly  bool                           `json:"metadata_only,omitempty"`
	Planned       bool                           `json:"planned,omitempty"`
	Labels        map[string]string              `json:"labels,omitempty"`
}

func (scheme QuantScheme) Matched() bool {
	return scheme.Contract != "" && scheme.Kind != ""
}

func (scheme QuantScheme) Clone() QuantScheme {
	scheme.Labels = cloneStringMap(scheme.Labels)
	return scheme
}

var registeredQuantSchemes = registry.NewOrdered[string, QuantScheme]()

func RegisterQuantScheme(scheme QuantScheme) {
	scheme = NormalizeQuantScheme(scheme)
	if !scheme.Matched() {
		return
	}
	registeredQuantSchemes.Put(scheme.Kind, scheme)
}

func RegisteredQuantSchemeKinds() []string {
	return registeredQuantSchemes.Keys()
}

func RegisteredQuantSchemes() []QuantScheme {
	return registeredQuantSchemeSnapshot()
}

func ReplaceRegisteredQuantSchemes(schemes []QuantScheme) {
	order := make([]string, 0, len(schemes))
	values := make(map[string]QuantScheme, len(schemes))
	for _, scheme := range schemes {
		scheme = NormalizeQuantScheme(scheme)
		if !scheme.Matched() {
			continue
		}
		if _, ok := values[scheme.Kind]; !ok {
			order = append(order, scheme.Kind)
		}
		values[scheme.Kind] = scheme.Clone()
	}
	registeredQuantSchemes.Restore(order, values)
}

func DefaultQuantSchemes() []QuantScheme {
	schemes := builtinQuantSchemes()
	index := make(map[string]int, len(schemes))
	for i, scheme := range schemes {
		index[scheme.Kind] = i
	}
	for _, scheme := range registeredQuantSchemeSnapshot() {
		if existing, ok := index[scheme.Kind]; ok {
			schemes[existing] = scheme
			continue
		}
		index[scheme.Kind] = len(schemes)
		schemes = append(schemes, scheme)
	}
	return cloneQuantSchemes(schemes)
}

func QuantSchemeForKind(kind string) (QuantScheme, bool) {
	kind = NormalizeQuantSchemeKind(kind)
	if kind == "" {
		return QuantScheme{}, false
	}
	for _, scheme := range DefaultQuantSchemes() {
		if scheme.Kind == kind {
			return scheme.Clone(), true
		}
	}
	return QuantScheme{}, false
}

func DefaultQuantSchemeKinds() []string {
	return QuantSchemeKinds(DefaultQuantSchemes())
}

func NormalizeQuantScheme(scheme QuantScheme) QuantScheme {
	scheme.Kind = NormalizeQuantSchemeKind(scheme.Kind)
	if scheme.Kind == "" {
		return QuantScheme{}
	}
	if scheme.Contract == "" {
		scheme.Contract = QuantSchemeRegistryContract
	}
	if scheme.Name == "" {
		scheme.Name = QuantSchemeRouteName
	}
	if scheme.Loader == "" {
		scheme.Loader = scheme.Kind
	}
	if scheme.Source == "" {
		scheme.Source = "registered"
	}
	if scheme.Runtime == "" {
		switch {
		case scheme.MetadataOnly:
			scheme.Runtime = QuantSchemeRuntimeMetadata
		case scheme.Planned:
			scheme.Runtime = QuantSchemeRuntimePlannedHIP
		default:
			scheme.Runtime = QuantRuntimeMLXAffine
		}
	}
	if scheme.RuntimeStatus == "" {
		switch {
		case scheme.MetadataOnly:
			scheme.RuntimeStatus = inference.FeatureRuntimeMetadataOnly
		case scheme.Planned:
			scheme.RuntimeStatus = inference.FeatureRuntimePlanned
		case scheme.NativeRuntime:
			scheme.RuntimeStatus = inference.FeatureRuntimeNative
		default:
			scheme.RuntimeStatus = inference.FeatureRuntimeExperimental
		}
	}
	scheme.Registered = true
	scheme.Labels = quantSchemeLabels(scheme)
	return scheme.Clone()
}

func NormalizeQuantSchemeKind(kind string) string {
	kind = strings.ToLower(strings.TrimSpace(kind))
	kind = strings.ReplaceAll(kind, "-", "_")
	kind = strings.TrimPrefix(kind, "mlx_")
	kind = strings.TrimPrefix(kind, "weight_")
	switch kind {
	case "q4", "q6", "q8", "affine_q4", "affine_q6", "affine_q8", "mlx":
		return "affine"
	case "fp16", "f16", "bfloat16":
		return "bf16"
	case "jang", "mxtq":
		return "jangtq"
	default:
		return kind
	}
}

func QuantSchemeKinds(schemes []QuantScheme) []string {
	kinds := make([]string, 0, len(schemes))
	for _, scheme := range schemes {
		if scheme.Kind != "" {
			kinds = append(kinds, scheme.Kind)
		}
	}
	return kinds
}

func QuantSchemeKindsCSV(schemes []QuantScheme) string {
	return core.Join(",", QuantSchemeKinds(schemes)...)
}

func builtinQuantSchemes() []QuantScheme {
	return []QuantScheme{
		quantScheme("affine", 0, "gemma4_affine", "go-mlx", QuantRuntimeMLXAffine, inference.FeatureRuntimeExperimental, true, true, false, false),
		quantScheme("bf16", 16, "gemma4_bf16", "dense", QuantRuntimeBF16, inference.FeatureRuntimeNative, true, true, false, false),
		quantScheme("mxfp4", 4, "autoround_mxfp4", "autoround", QuantSchemeRuntimePlannedHIP, inference.FeatureRuntimePlanned, true, false, false, true),
		quantScheme("mxfp8", 8, "autoround_mxfp8", "autoround", QuantSchemeRuntimePlannedHIP, inference.FeatureRuntimePlanned, true, false, false, true),
		quantScheme("nvfp4", 4, "autoround_nvfp4", "autoround", QuantSchemeRuntimePlannedHIP, inference.FeatureRuntimePlanned, true, false, false, true),
		quantScheme("q4_0", 4, "gguf_q4_0", "gguf", QuantSchemeRuntimeMetadata, inference.FeatureRuntimeMetadataOnly, true, false, true, false),
		quantScheme("jangtq", 2, "minimax_m2_jangtq", "minimax_m2", QuantSchemeRuntimeMetadata, inference.FeatureRuntimeMetadataOnly, true, false, true, false),
	}
}

func quantScheme(kind string, bits int, loader, source, runtime string, status inference.FeatureRuntimeStatus, registered, nativeRuntime, metadataOnly, planned bool) QuantScheme {
	scheme := QuantScheme{
		Contract:      QuantSchemeRegistryContract,
		Name:          QuantSchemeRouteName,
		Kind:          kind,
		Bits:          bits,
		Loader:        loader,
		Source:        source,
		Runtime:       runtime,
		RuntimeStatus: status,
		Registered:    registered,
		NativeRuntime: nativeRuntime,
		MetadataOnly:  metadataOnly,
		Planned:       planned,
	}
	scheme.Labels = quantSchemeLabels(scheme)
	return scheme
}

func registeredQuantSchemeSnapshot() []QuantScheme {
	registeredSchemes := registeredQuantSchemes.Values()
	out := make([]QuantScheme, 0, len(registeredSchemes))
	for _, scheme := range registeredSchemes {
		out = append(out, scheme.Clone())
	}
	return out
}

func quantSchemeLabels(scheme QuantScheme) map[string]string {
	if !scheme.Matched() {
		return nil
	}
	labels := map[string]string{
		"engine_quant_scheme_contract":      scheme.Contract,
		"engine_quant_scheme":               scheme.Name,
		"engine_quant_scheme_kind":          scheme.Kind,
		"engine_quant_scheme_registered":    strconv.FormatBool(scheme.Registered),
		"engine_quant_scheme_native":        strconv.FormatBool(scheme.NativeRuntime),
		"engine_quant_scheme_metadata_only": strconv.FormatBool(scheme.MetadataOnly),
		"engine_quant_scheme_planned":       strconv.FormatBool(scheme.Planned),
	}
	if scheme.Bits > 0 {
		labels["engine_quant_scheme_bits"] = strconv.Itoa(scheme.Bits)
	}
	if scheme.Loader != "" {
		labels["engine_quant_scheme_loader"] = scheme.Loader
	}
	if scheme.Source != "" {
		labels["engine_quant_scheme_source"] = scheme.Source
	}
	if scheme.Runtime != "" {
		labels["engine_quant_scheme_runtime"] = scheme.Runtime
	}
	if scheme.RuntimeStatus != "" {
		labels["engine_quant_scheme_runtime_status"] = string(scheme.RuntimeStatus)
	}
	return labels
}

func cloneQuantSchemes(schemes []QuantScheme) []QuantScheme {
	out := append([]QuantScheme(nil), schemes...)
	for i := range out {
		out[i] = out[i].Clone()
	}
	return out
}

// QuantLoaderPack is the production pack metadata needed to synthesize a
// concrete quant-loader route without importing root production-lane code.
type QuantLoaderPack struct {
	Name           string
	Size           string
	ModelID        string
	LockedModelID  string
	Bits           int
	QuantMode      string
	QuantGroup     int
	Runtime        string
	GenerateStatus string
	ProductRole    string
	Supported      bool
	RunnableOnCard bool
	RequiresBench  bool
	RequiresNative bool
}

// QuantLoaderRoute is the model-owned weight-quant loader route. Root ROCm
// converts production-pack rows into this shape; model families can also
// self-register extension routes directly.
type QuantLoaderRoute struct {
	Contract       string            `json:"contract,omitempty"`
	Name           string            `json:"name,omitempty"`
	Family         string            `json:"family,omitempty"`
	Architecture   string            `json:"architecture,omitempty"`
	Size           string            `json:"size,omitempty"`
	Pack           string            `json:"pack,omitempty"`
	PackName       string            `json:"pack_name,omitempty"`
	ModelID        string            `json:"model_id,omitempty"`
	LockedModelID  string            `json:"locked_model_id,omitempty"`
	Mode           string            `json:"mode,omitempty"`
	Bits           int               `json:"bits,omitempty"`
	Group          int               `json:"group,omitempty"`
	ProductRole    string            `json:"product_role,omitempty"`
	Loader         string            `json:"loader,omitempty"`
	Runtime        string            `json:"runtime,omitempty"`
	GenerateStatus string            `json:"generate_status,omitempty"`
	Target         string            `json:"target,omitempty"`
	Registered     bool              `json:"registered,omitempty"`
	NativeRuntime  bool              `json:"native_runtime,omitempty"`
	RunnableOnCard bool              `json:"runnable_on_card,omitempty"`
	Staged         bool              `json:"staged,omitempty"`
	LoadOnly       bool              `json:"load_only,omitempty"`
	Planned        bool              `json:"planned,omitempty"`
	RequiresBench  bool              `json:"requires_bench,omitempty"`
	RequiresNative bool              `json:"requires_native,omitempty"`
	Labels         map[string]string `json:"labels,omitempty"`
}

func (route QuantLoaderRoute) Matched() bool {
	return route.Contract != "" && route.Pack != "" && route.Loader != ""
}

func (route QuantLoaderRoute) Clone() QuantLoaderRoute {
	route.Labels = cloneStringMap(route.Labels)
	return route
}

var registeredQuantLoaders = registry.NewOrdered[string, QuantLoaderRoute]()

func RegisterQuantLoaderRoute(route QuantLoaderRoute) {
	route = NormalizeQuantLoaderRoute(route)
	if !route.Matched() {
		return
	}
	registeredQuantLoaders.Put(QuantLoaderRouteKey(route.Pack), route)
}

func RegisteredQuantLoaderRoutePacks() []string {
	registeredRoutes := registeredQuantLoaders.Values()
	out := make([]string, 0, len(registeredRoutes))
	for _, route := range registeredRoutes {
		out = append(out, route.Pack)
	}
	return out
}

func RegisteredQuantLoaderRoutes() []QuantLoaderRoute {
	return registeredQuantLoaderSnapshot()
}

func ReplaceRegisteredQuantLoaderRoutes(routes []QuantLoaderRoute) {
	order := make([]string, 0, len(routes))
	values := make(map[string]QuantLoaderRoute, len(routes))
	for _, route := range routes {
		route = NormalizeQuantLoaderRoute(route)
		if !route.Matched() {
			continue
		}
		key := QuantLoaderRouteKey(route.Pack)
		if _, ok := values[key]; !ok {
			order = append(order, key)
		}
		values[key] = route.Clone()
	}
	registeredQuantLoaders.Restore(order, values)
}

func DefaultQuantLoaderRoutesForPacks(packs []QuantLoaderPack) []QuantLoaderRoute {
	routes := make([]QuantLoaderRoute, 0, len(packs)+len(registeredQuantLoaders.Keys()))
	seen := map[string]int{}
	for _, pack := range packs {
		route := QuantLoaderRouteForPack(pack)
		if !route.Matched() {
			continue
		}
		seen[QuantLoaderRouteKey(route.Pack)] = len(routes)
		routes = append(routes, route)
	}
	for _, route := range registeredQuantLoaderSnapshot() {
		key := QuantLoaderRouteKey(route.Pack)
		if idx, ok := seen[key]; ok {
			routes[idx] = route.Clone()
			continue
		}
		seen[key] = len(routes)
		routes = append(routes, route.Clone())
	}
	return cloneQuantLoaderRoutes(routes)
}

func QuantLoaderRouteForPack(pack QuantLoaderPack) QuantLoaderRoute {
	mode := QuantLoaderPackMode(pack)
	packLabel := QuantLoaderPackLabelName(pack)
	route := QuantLoaderRoute{
		Contract:       QuantLoaderRegistryContract,
		Name:           QuantLoaderRouteName,
		Family:         QuantLoaderFamilyGemma4,
		Architecture:   QuantLoaderArchitectureGemma4Text,
		Size:           pack.Size,
		Pack:           packLabel,
		PackName:       pack.Name,
		ModelID:        pack.ModelID,
		LockedModelID:  pack.LockedModelID,
		Mode:           mode,
		Bits:           pack.Bits,
		Group:          pack.QuantGroup,
		ProductRole:    pack.ProductRole,
		Loader:         QuantLoaderNameForPack(pack, mode),
		Runtime:        pack.Runtime,
		GenerateStatus: pack.GenerateStatus,
		Target:         QuantLoaderTargetForStatus(pack.GenerateStatus, pack.RunnableOnCard),
		Registered:     pack.Supported,
		NativeRuntime:  QuantLoaderPackNativeRuntime(pack),
		RunnableOnCard: pack.RunnableOnCard,
		Staged:         pack.GenerateStatus != QuantGenerateLinked,
		LoadOnly:       pack.GenerateStatus == QuantGenerateLoadOnly,
		Planned:        pack.GenerateStatus == QuantGeneratePlannedOnly,
		RequiresBench:  pack.RequiresBench,
		RequiresNative: pack.RequiresNative,
	}
	route.Labels = quantLoaderRouteLabels(route)
	return route.Clone()
}

func RegisteredQuantLoaderRouteForToken(token string) (QuantLoaderRoute, bool) {
	token = QuantLoaderRouteKey(token)
	if token == "" {
		return QuantLoaderRoute{}, false
	}
	for _, route := range registeredQuantLoaders.Values() {
		if QuantLoaderRouteMatchesToken(route, token) {
			return route.Clone(), true
		}
	}
	return QuantLoaderRoute{}, false
}

func NormalizeQuantLoaderRoute(route QuantLoaderRoute) QuantLoaderRoute {
	route.Pack = strings.TrimSpace(route.Pack)
	route.PackName = strings.TrimSpace(route.PackName)
	route.Mode = NormalizeQuantLoaderMode(route.Mode)
	route.Size = strings.TrimSpace(route.Size)
	if route.Pack == "" {
		switch {
		case route.Size != "" && route.Mode != "":
			route.Pack = route.Size + ":" + route.Mode
		case route.Mode != "":
			route.Pack = route.Mode
		case route.PackName != "":
			route.Pack = route.PackName
		case route.Loader != "":
			route.Pack = route.Loader
		}
	}
	if route.Pack == "" {
		return QuantLoaderRoute{}
	}
	if route.Contract == "" {
		route.Contract = QuantLoaderRegistryContract
	}
	if route.Name == "" {
		route.Name = QuantLoaderRouteName
	}
	if route.Family == "" {
		route.Family = "registered"
	}
	if route.Loader == "" {
		route.Loader = strings.ReplaceAll(QuantLoaderRouteKey(route.Pack), ":", "_")
	}
	if route.Runtime == "" {
		switch {
		case route.Planned:
			route.Runtime = QuantRuntimePlanned
		case route.LoadOnly:
			route.Runtime = QuantRuntimeBF16
		default:
			route.Runtime = QuantRuntimeMLXAffine
		}
	}
	if route.GenerateStatus == "" {
		switch {
		case route.Planned:
			route.GenerateStatus = QuantGeneratePlannedOnly
		case route.LoadOnly:
			route.GenerateStatus = QuantGenerateLoadOnly
		default:
			route.GenerateStatus = QuantGenerateLinked
		}
	}
	route.Target = firstNonEmpty(route.Target, QuantLoaderTargetForStatus(route.GenerateStatus, route.RunnableOnCard))
	route.Registered = true
	route.Planned = route.GenerateStatus == QuantGeneratePlannedOnly
	route.LoadOnly = route.GenerateStatus == QuantGenerateLoadOnly
	route.Staged = route.GenerateStatus != QuantGenerateLinked
	if !route.Planned && route.Runtime != QuantRuntimePlanned && route.Runtime != QuantRuntimeGGUF && route.Runtime != QuantSchemeRuntimeMetadata {
		route.NativeRuntime = true
	}
	route.Labels = quantLoaderRouteLabels(route)
	return route.Clone()
}

func QuantLoaderPackMode(pack QuantLoaderPack) string {
	if pack.QuantMode == "affine" && pack.Bits > 0 {
		return "q" + strconv.Itoa(pack.Bits)
	}
	return pack.QuantMode
}

func QuantLoaderPackLabelName(pack QuantLoaderPack) string {
	mode := QuantLoaderPackMode(pack)
	if pack.ProductRole == "mtp-assistant" && mode != "" {
		mode = "assistant-" + mode
	}
	if pack.Size == "" {
		return mode
	}
	return pack.Size + ":" + mode
}

func QuantLoaderNameForPack(pack QuantLoaderPack, mode string) string {
	switch {
	case pack.ProductRole == "mtp-assistant":
		return "gemma4_assistant_bf16"
	case pack.Runtime == QuantRuntimeGGUF:
		return "gemma4_gguf"
	case pack.QuantMode == "affine":
		return "gemma4_affine"
	case strings.HasSuffix(mode, "-status"):
		return "gemma4_status"
	case mode != "":
		return "gemma4_" + strings.ReplaceAll(mode, "-", "_")
	default:
		return "gemma4_quant"
	}
}

func QuantLoaderPackNativeRuntime(pack QuantLoaderPack) bool {
	return pack.GenerateStatus != QuantGeneratePlannedOnly && pack.Runtime != QuantRuntimePlanned && pack.Runtime != QuantRuntimeGGUF
}

func QuantLoaderTargetForStatus(status string, runnableOnCard bool) string {
	switch status {
	case QuantGenerateLinked:
		return "generate"
	case QuantGenerateLoadOnly:
		return "load"
	case QuantGeneratePlannedOnly:
		if !runnableOnCard {
			return "metadata"
		}
		return "planned"
	default:
		return "metadata"
	}
}

func QuantLoaderRouteMatchesToken(route QuantLoaderRoute, token string) bool {
	candidates := []string{
		route.Pack,
		route.PackName,
		route.Mode,
		route.Loader,
		route.ModelID,
		route.LockedModelID,
	}
	if route.Size != "" && route.Mode != "" {
		candidates = append(candidates, route.Size+":"+route.Mode)
	}
	for _, candidate := range candidates {
		if QuantLoaderRouteKey(candidate) == token {
			return true
		}
	}
	return false
}

func QuantLoaderIdentityTokens(model inference.ModelIdentity) []string {
	labels := model.Labels
	candidates := []string{
		model.ID,
		model.Path,
		model.QuantType,
		labels["engine_quant_loader_pack"],
		labels["engine_quant_loader_pack_name"],
		labels["engine_quant_loader_mode"],
		labels["production_quant_pack"],
		labels["production_quant_mode"],
		labels["gemma4_quant_mode"],
		labels["quant_type"],
	}
	if model.QuantBits > 0 {
		candidates = append(candidates, "q"+strconv.Itoa(model.QuantBits))
	}
	out := make([]string, 0, len(candidates))
	seen := map[string]bool{}
	for _, candidate := range candidates {
		key := QuantLoaderRouteKey(candidate)
		if key == "" || seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, key)
	}
	return out
}

func QuantLoaderRouteKey(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	value = strings.ReplaceAll(value, "-", "_")
	value = strings.ReplaceAll(value, " ", "_")
	return value
}

func NormalizeQuantLoaderMode(mode string) string {
	mode = strings.ToLower(strings.TrimSpace(mode))
	mode = strings.ReplaceAll(mode, "-", "_")
	mode = strings.ReplaceAll(mode, " ", "_")
	return mode
}

func registeredQuantLoaderSnapshot() []QuantLoaderRoute {
	registeredRoutes := registeredQuantLoaders.Values()
	out := make([]QuantLoaderRoute, 0, len(registeredRoutes))
	for _, route := range registeredRoutes {
		out = append(out, route.Clone())
	}
	return out
}

func quantLoaderRouteLabels(route QuantLoaderRoute) map[string]string {
	if !route.Matched() {
		return nil
	}
	labels := map[string]string{
		"engine_quant_loader_contract":         route.Contract,
		"engine_quant_loader":                  route.Loader,
		"engine_quant_loader_registered":       strconv.FormatBool(route.Registered),
		"engine_quant_loader_native":           strconv.FormatBool(route.NativeRuntime),
		"engine_quant_loader_runnable_on_card": strconv.FormatBool(route.RunnableOnCard),
		"engine_quant_loader_staged":           strconv.FormatBool(route.Staged),
		"engine_quant_loader_load_only":        strconv.FormatBool(route.LoadOnly),
		"engine_quant_loader_planned":          strconv.FormatBool(route.Planned),
		"engine_quant_loader_requires_bench":   strconv.FormatBool(route.RequiresBench),
		"engine_quant_loader_requires_native":  strconv.FormatBool(route.RequiresNative),
	}
	if route.Family != "" {
		labels["engine_quant_loader_family"] = route.Family
	}
	if route.Architecture != "" {
		labels["engine_quant_loader_architecture"] = route.Architecture
	}
	if route.Size != "" {
		labels["engine_quant_loader_size"] = route.Size
	}
	if route.Pack != "" {
		labels["engine_quant_loader_pack"] = route.Pack
	}
	if route.PackName != "" {
		labels["engine_quant_loader_pack_name"] = route.PackName
	}
	if route.ModelID != "" {
		labels["engine_quant_loader_model"] = route.ModelID
	}
	if route.LockedModelID != "" {
		labels["engine_quant_loader_locked_model"] = route.LockedModelID
	}
	if route.Mode != "" {
		labels["engine_quant_loader_mode"] = route.Mode
	}
	if route.Bits > 0 {
		labels["engine_quant_loader_bits"] = strconv.Itoa(route.Bits)
	}
	if route.Group > 0 {
		labels["engine_quant_loader_group"] = strconv.Itoa(route.Group)
	}
	if route.ProductRole != "" {
		labels["engine_quant_loader_product_role"] = route.ProductRole
	}
	if route.Runtime != "" {
		labels["engine_quant_loader_runtime"] = route.Runtime
	}
	if route.GenerateStatus != "" {
		labels["engine_quant_loader_generate_status"] = route.GenerateStatus
	}
	if route.Target != "" {
		labels["engine_quant_loader_target"] = route.Target
	}
	return labels
}

// QuantLoaderRouteLabels returns the normalized model-owned label contract for
// a quant-loader route.
func QuantLoaderRouteLabels(route QuantLoaderRoute) map[string]string {
	route = NormalizeQuantLoaderRoute(route)
	return cloneStringMap(route.Labels)
}

func cloneQuantLoaderRoutes(routes []QuantLoaderRoute) []QuantLoaderRoute {
	out := append([]QuantLoaderRoute(nil), routes...)
	for i := range out {
		out[i] = out[i].Clone()
	}
	return out
}
