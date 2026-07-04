// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"dappco.re/go/inference"
	rocmmodel "dappco.re/go/inference/engine/hip/model"
)

const (
	ROCmCacheFactoryRouteContract = rocmmodel.CacheFactoryRouteContract
	ROCmCacheFactoryRouteName     = rocmmodel.CacheFactoryRouteName

	ROCmCacheRuntimeHIP      = rocmmodel.CacheRuntimeHIP
	ROCmCacheRuntimeMetadata = rocmmodel.CacheRuntimeMetadata
	ROCmCacheRuntimePlanned  = rocmmodel.CacheRuntimePlanned
	ROCmCacheRuntimeRetained = rocmmodel.CacheRuntimeRetained
	ROCmCacheRuntimeAttached = rocmmodel.CacheRuntimeAttached

	ROCmCacheModeDefault        = rocmmodel.SequenceMixerCacheModeDefault
	ROCmCacheModeRecurrent      = rocmmodel.SequenceMixerCacheModeRecurrent
	ROCmCacheModeMLALatent      = rocmmodel.SequenceMixerCacheModeMLALatent
	ROCmCacheModeCompaction     = rocmmodel.SequenceMixerCacheModeCompaction
	ROCmCacheModeCompactionFull = rocmmodel.SequenceMixerCacheModeCompactionFull
	ROCmCacheModeBlockPrefix    = rocmmodel.CacheModeBlockPrefix
	ROCmCacheModeRetained       = rocmmodel.CacheModeRetained
	ROCmCacheModeAttached       = rocmmodel.CacheModeAttached
	ROCmCacheModeFP16           = rocmmodel.CacheModeFP16
	ROCmCacheModeQ8             = rocmmodel.CacheModeQ8
	ROCmCacheModeKQ8VQ4         = rocmmodel.CacheModeKQ8VQ4
	ROCmCacheModePaged          = rocmmodel.CacheModePaged
	ROCmCacheModeFixed          = rocmmodel.CacheModeFixed
	ROCmCacheModeTurboQuant     = rocmmodel.CacheModeTurboQuant
)

// ROCmCacheModeRoute describes a cache/state holder the ROCm cache factory can
// plan for. It aliases the model-owned route so the root package exposes the
// same public API contract without duplicating registry state.
type ROCmCacheModeRoute = rocmmodel.CacheModeRoute

// ROCmCacheRoute is the model-owned cache factory answer for a concrete
// architecture/profile, exposed at the root API beside quant and mixer routes.
type ROCmCacheRoute = rocmmodel.CacheRoute

func DefaultROCmCacheModeRoutes() []ROCmCacheModeRoute {
	return rocmmodel.DefaultCacheModeRoutes()
}

func ROCmCacheModeRouteForMode(mode string) (ROCmCacheModeRoute, bool) {
	return rocmmodel.CacheModeRouteForMode(mode)
}

func ROCmCacheRouteForArchitecture(architecture string) (ROCmCacheRoute, bool) {
	return rocmmodel.CacheRouteForArchitecture(architecture)
}

func ROCmCacheRouteForIdentity(path string, model inference.ModelIdentity) (ROCmCacheRoute, bool) {
	return rocmmodel.CacheRouteForIdentity(path, model)
}

func ROCmCacheRouteForInfo(path string, info inference.ModelInfo, labels map[string]string) (ROCmCacheRoute, bool) {
	return rocmmodel.CacheRouteForInfo(path, info, labels)
}

func ROCmCacheRouteForInspection(inspection *inference.ModelPackInspection) (ROCmCacheRoute, bool) {
	return rocmmodel.CacheRouteForInspection(inspection)
}

func ROCmCacheRouteForProfile(profile ROCmModelProfile) (ROCmCacheRoute, bool) {
	plan := ROCmModelRoutePlanForProfile(profile)
	if !plan.Matched() || !plan.CacheRoute.Matched() {
		return ROCmCacheRoute{}, false
	}
	return plan.CacheRoute.Clone(), true
}

func ROCmCacheRouteForModel(model inference.TextModel) (ROCmCacheRoute, bool) {
	plan, ok := ROCmModelRoutePlanForModel(model)
	if !ok || !plan.CacheRoute.Matched() {
		return ROCmCacheRoute{}, false
	}
	return plan.CacheRoute.Clone(), true
}

func rocmApplyROCmCacheRouteLabels(labels map[string]string, route ROCmCacheRoute) {
	if !route.Matched() {
		return
	}
	for key, value := range route.Labels {
		if value != "" {
			labels[key] = value
		}
	}
}
