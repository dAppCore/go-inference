// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"strings"

	"dappco.re/go/inference"
	rocmmodel "dappco.re/go/inference/engine/hip/model"
)

const (
	ROCmQuantLoaderRegistryContract = rocmmodel.QuantLoaderRegistryContract

	rocmQuantLoaderRegistryRouteName = rocmmodel.QuantLoaderRouteName
	rocmQuantLoaderFamilyGemma4      = rocmmodel.QuantLoaderFamilyGemma4
	rocmQuantLoaderArchitecture      = rocmmodel.QuantLoaderArchitectureGemma4Text
)

// ROCmQuantLoaderRoute is the production quant-pack route consumers can use
// before model load. It mirrors go-mlx's weight-quant loader registry at the
// contract layer while preserving ROCm's current linked/load-only/planned
// runtime status for each pack.
type ROCmQuantLoaderRoute = rocmmodel.QuantLoaderRoute

func DefaultROCmQuantLoaderRoutes() []ROCmQuantLoaderRoute {
	return rocmQuantLoaderRoutesFromModel(rocmmodel.DefaultQuantLoaderRoutesForPacks(rocmQuantLoaderPacksToModel(DefaultProductionQuantizationPackSupport())))
}

// RegisterROCmQuantLoaderRoute registers or replaces a concrete quant-loader
// route. It mirrors go-mlx's quant loader registry at the ROCm API layer: a
// quant format or production pack can register how it should be loaded without
// editing the built-in Gemma-4 matrix.
func RegisterROCmQuantLoaderRoute(route ROCmQuantLoaderRoute) {
	route = normalizeRegisteredROCmQuantLoaderRoute(route)
	if !route.Matched() {
		return
	}
	rocmmodel.RegisterQuantLoaderRoute(route)
}

// RegisteredROCmQuantLoaderRoutePacks returns extension route packs in
// resolution order. Built-in production packs are intentionally not included.
func RegisteredROCmQuantLoaderRoutePacks() []string {
	return rocmmodel.RegisteredQuantLoaderRoutePacks()
}

func registeredROCmQuantLoaderRouteSnapshot() []ROCmQuantLoaderRoute {
	return rocmQuantLoaderRoutesFromModel(rocmmodel.RegisteredQuantLoaderRoutes())
}

func registeredROCmQuantLoaderRouteForToken(token string) (ROCmQuantLoaderRoute, bool) {
	route, ok := rocmmodel.RegisteredQuantLoaderRouteForToken(token)
	if !ok {
		return ROCmQuantLoaderRoute{}, false
	}
	return rocmQuantLoaderRouteFromModel(route), true
}

func normalizeRegisteredROCmQuantLoaderRoute(route ROCmQuantLoaderRoute) ROCmQuantLoaderRoute {
	if route.Architecture != "" {
		route.Architecture = ROCmArchitectureID(route.Architecture)
	}
	return rocmmodel.NormalizeQuantLoaderRoute(route).Clone()
}

func ROCmQuantLoaderRouteForPack(pack ProductionQuantizationPackSupport) ROCmQuantLoaderRoute {
	return rocmQuantLoaderRouteFromModel(rocmmodel.QuantLoaderRouteForPack(rocmQuantLoaderPackToModel(pack)))
}

func ROCmQuantLoaderRouteForMode(mode string) (ROCmQuantLoaderRoute, bool) {
	needle := strings.ToLower(strings.TrimSpace(mode))
	if needle == "" {
		return ROCmQuantLoaderRoute{}, false
	}
	if route, ok := registeredROCmQuantLoaderRouteForToken(needle); ok {
		return route.Clone(), true
	}
	if pack, ok := ProductionQuantizationPackByName(needle); ok {
		return ROCmQuantLoaderRouteForPack(pack), true
	}
	for _, pack := range DefaultProductionQuantizationPackSupport() {
		if strings.ToLower(rocmGemma4ProductionQuantPackMode(pack)) == needle ||
			strings.ToLower(pack.QuantMode) == needle ||
			strings.ToLower(pack.Name) == needle {
			return ROCmQuantLoaderRouteForPack(pack), true
		}
	}
	return ROCmQuantLoaderRoute{}, false
}

func ROCmQuantLoaderRouteForIdentity(path string, model inference.ModelIdentity) (ROCmQuantLoaderRoute, bool) {
	if model.Path == "" {
		model.Path = path
	}
	for _, token := range rocmQuantLoaderIdentityTokens(model) {
		if route, ok := registeredROCmQuantLoaderRouteForToken(token); ok {
			return route.Clone(), true
		}
	}
	pack, ok := rocmGemma4ProductionQuantPackForModel(model)
	if !ok {
		return ROCmQuantLoaderRoute{}, false
	}
	return ROCmQuantLoaderRouteForPack(pack), true
}

func ROCmQuantLoaderRouteForProfile(profile ROCmModelProfile) (ROCmQuantLoaderRoute, bool) {
	return ROCmQuantLoaderRouteForIdentity(profile.Model.Path, profile.Model)
}

func ROCmQuantLoaderRouteForInspection(inspection *inference.ModelPackInspection) (ROCmQuantLoaderRoute, bool) {
	if inspection == nil {
		return ROCmQuantLoaderRoute{}, false
	}
	return ROCmQuantLoaderRouteForIdentity(inspection.Path, inspection.Model)
}

func rocmApplyROCmQuantLoaderRouteLabels(labels map[string]string, route ROCmQuantLoaderRoute) map[string]string {
	if labels == nil {
		labels = map[string]string{}
	}
	if !route.Matched() {
		return labels
	}
	for key, value := range rocmQuantLoaderRouteLabels(route) {
		if value != "" {
			labels[key] = value
		}
	}
	return labels
}

func rocmQuantLoaderNameForPack(pack ProductionQuantizationPackSupport, mode string) string {
	return rocmmodel.QuantLoaderNameForPack(rocmQuantLoaderPackToModel(pack), mode)
}

func rocmQuantLoaderNativeRuntime(pack ProductionQuantizationPackSupport) bool {
	return rocmmodel.QuantLoaderPackNativeRuntime(rocmQuantLoaderPackToModel(pack))
}

func rocmQuantLoaderTarget(pack ProductionQuantizationPackSupport) string {
	return rocmQuantLoaderTargetForStatus(pack.GenerateStatus, pack.RunnableOnCard)
}

func rocmQuantLoaderTargetForStatus(status string, runnableOnCard bool) string {
	return rocmmodel.QuantLoaderTargetForStatus(status, runnableOnCard)
}

func rocmQuantLoaderRouteMatchesToken(route ROCmQuantLoaderRoute, token string) bool {
	return rocmmodel.QuantLoaderRouteMatchesToken(route, token)
}

func rocmQuantLoaderIdentityTokens(model inference.ModelIdentity) []string {
	return rocmmodel.QuantLoaderIdentityTokens(model)
}

func rocmQuantLoaderRouteKey(value string) string {
	return rocmmodel.QuantLoaderRouteKey(value)
}

func normalizeROCmQuantLoaderMode(mode string) string {
	return rocmmodel.NormalizeQuantLoaderMode(mode)
}

func rocmQuantLoaderRouteLabels(route ROCmQuantLoaderRoute) map[string]string {
	return rocmmodel.QuantLoaderRouteLabels(route)
}

func rocmQuantLoaderPackToModel(pack ProductionQuantizationPackSupport) rocmmodel.QuantLoaderPack {
	return rocmmodel.QuantLoaderPack{
		Name:           pack.Name,
		Size:           pack.Size,
		ModelID:        pack.ModelID,
		LockedModelID:  pack.LockedModelID,
		Bits:           pack.Bits,
		QuantMode:      pack.QuantMode,
		QuantGroup:     pack.QuantGroup,
		Runtime:        pack.Runtime,
		GenerateStatus: pack.GenerateStatus,
		ProductRole:    pack.ProductRole,
		Supported:      pack.Supported,
		RunnableOnCard: pack.RunnableOnCard,
		RequiresBench:  pack.RequiresBench,
		RequiresNative: pack.RequiresNative,
	}
}

func rocmQuantLoaderPacksToModel(packs []ProductionQuantizationPackSupport) []rocmmodel.QuantLoaderPack {
	out := make([]rocmmodel.QuantLoaderPack, 0, len(packs))
	for _, pack := range packs {
		out = append(out, rocmQuantLoaderPackToModel(pack))
	}
	return out
}

func rocmQuantLoaderRouteFromModel(route rocmmodel.QuantLoaderRoute) ROCmQuantLoaderRoute {
	if route.Labels == nil {
		route.Labels = rocmmodel.QuantLoaderRouteLabels(route)
	}
	return route.Clone()
}

func rocmQuantLoaderRoutesFromModel(routes []rocmmodel.QuantLoaderRoute) []ROCmQuantLoaderRoute {
	out := make([]ROCmQuantLoaderRoute, 0, len(routes))
	for _, route := range routes {
		out = append(out, rocmQuantLoaderRouteFromModel(route))
	}
	return out
}
