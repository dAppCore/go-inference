// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"dappco.re/go/inference"
	rocmmodel "dappco.re/go/inference/engine/hip/model"
)

const (
	ROCmModelRuntimeContractRegistryContract = rocmmodel.RuntimeContractRegistryContract
	rocmModelRuntimeContractRouteName        = rocmmodel.RuntimeContractRouteName
)

type ROCmModelRuntimeContractID = rocmmodel.RuntimeContractID

const (
	ROCmRuntimeContractLastTokenLogits          = rocmmodel.RuntimeContractLastTokenLogits
	ROCmRuntimeContractGreedyToken              = rocmmodel.RuntimeContractGreedyToken
	ROCmRuntimeContractSuppressedGreedyToken    = rocmmodel.RuntimeContractSuppressedGreedyToken
	ROCmRuntimeContractQueryHeads               = rocmmodel.RuntimeContractQueryHeads
	ROCmRuntimeContractLoRALinearResolver       = rocmmodel.RuntimeContractLoRALinearResolver
	ROCmRuntimeContractDenseSplitParts          = rocmmodel.RuntimeContractDenseSplitParts
	ROCmRuntimeContractCacheTopology            = rocmmodel.RuntimeContractCacheTopology
	ROCmRuntimeContractAttentionCacheLayout     = rocmmodel.RuntimeContractAttentionCacheLayout
	ROCmRuntimeContractModelCloser              = rocmmodel.RuntimeContractModelCloser
	ROCmRuntimeContractFixedSlidingPrefillLimit = rocmmodel.RuntimeContractFixedSlidingPrefillLimit
	ROCmRuntimeContractFixedSlidingCache        = rocmmodel.RuntimeContractFixedSlidingCache
	ROCmRuntimeContractThoughtChannelSuppressor = rocmmodel.RuntimeContractThoughtChannelSuppressor
	ROCmRuntimeContractModelInfoReporter        = rocmmodel.RuntimeContractModelInfoReporter
	ROCmRuntimeContractMoETextRuntimeReporter   = rocmmodel.RuntimeContractMoETextRuntimeReporter
	ROCmRuntimeContractDecodeUnavailableReport  = rocmmodel.RuntimeContractDecodeUnavailableReport
	ROCmRuntimeContractHybridAttentionCachePlan = rocmmodel.RuntimeContractHybridAttentionCachePlan
)

// ROCmModelRuntimeContractRoute reports go-mlx-compatible optional model
// contracts for a resolved ROCm model profile. The contract is model-owned; the
// ROCm root alias keeps the consumer-facing API stable.
type ROCmModelRuntimeContractRoute = rocmmodel.RuntimeContractRoute

func RegisterROCmModelRuntimeContractRoute(route ROCmModelRuntimeContractRoute) {
	route = normalizeROCmModelRuntimeContractRoute(route)
	if !route.Matched() {
		return
	}
	rocmmodel.RegisterRuntimeContractRoute(route)
}

func RegisteredROCmModelRuntimeContractRouteArchitectures() []string {
	return rocmmodel.RegisteredRuntimeContractArchitectures()
}

func DefaultROCmModelRuntimeContractRoutes() []ROCmModelRuntimeContractRoute {
	return rocmModelRuntimeContractRoutesFromModel(rocmmodel.DefaultRuntimeContractRoutes())
}

func ROCmModelRuntimeContractRouteForArchitecture(architecture string) (ROCmModelRuntimeContractRoute, bool) {
	route, ok := rocmmodel.RuntimeContractRouteForArchitecture(architecture)
	if !ok {
		return ROCmModelRuntimeContractRoute{}, false
	}
	return route.Clone(), true
}

func ROCmModelRuntimeContractRouteForIdentity(path string, model inference.ModelIdentity) (ROCmModelRuntimeContractRoute, bool) {
	route, ok := rocmmodel.RuntimeContractRouteForIdentity(path, model)
	if !ok {
		return ROCmModelRuntimeContractRoute{}, false
	}
	return route.Clone(), true
}

func ROCmModelRuntimeContractRouteForInfo(path string, info inference.ModelInfo, labels map[string]string) (ROCmModelRuntimeContractRoute, bool) {
	route, ok := rocmmodel.RuntimeContractRouteForInfo(path, info, labels)
	if !ok {
		return ROCmModelRuntimeContractRoute{}, false
	}
	return route.Clone(), true
}

func ROCmModelRuntimeContractRouteForInspection(inspection *inference.ModelPackInspection) (ROCmModelRuntimeContractRoute, bool) {
	route, ok := rocmmodel.RuntimeContractRouteForInspection(inspection)
	if !ok {
		return ROCmModelRuntimeContractRoute{}, false
	}
	return route.Clone(), true
}

func normalizeROCmModelRuntimeContractRoute(route ROCmModelRuntimeContractRoute) ROCmModelRuntimeContractRoute {
	return rocmmodel.NormalizeRuntimeContractRoute(route).Clone()
}

func rocmModelRuntimeContractRoutesFromModel(routes []rocmmodel.RuntimeContractRoute) []ROCmModelRuntimeContractRoute {
	out := make([]ROCmModelRuntimeContractRoute, 0, len(routes))
	for _, route := range routes {
		if route.Matched() {
			out = append(out, route.Clone())
		}
	}
	return out
}

func rocmApplyROCmModelRuntimeContractRouteLabels(labels map[string]string, route ROCmModelRuntimeContractRoute) map[string]string {
	if labels == nil {
		labels = map[string]string{}
	}
	if !route.Matched() {
		return labels
	}
	for key, value := range rocmmodel.RuntimeContractRouteLabels(route) {
		if value != "" {
			labels[key] = value
		}
	}
	return labels
}

func rocmModelRuntimeContractIDsCSV(ids []ROCmModelRuntimeContractID) string {
	return rocmmodel.RuntimeContractIDsCSV(ids)
}
