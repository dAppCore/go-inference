// SPDX-Licence-Identifier: EUPL-1.2

package hip

import rocmmodel "dappco.re/go/inference/engine/hip/model"

const (
	ROCmSequenceMixerLoaderRegistryContract = SequenceMixerRegistryContract
)

// ROCmSequenceMixerLoaderRoute is the public route view for go-mlx's
// mixer-loader registry surface. The route metadata is model-owned; the ROCm
// alias preserves the root API name used by consumers.
type ROCmSequenceMixerLoaderRoute = rocmmodel.SequenceMixerLoaderRoute

func DefaultROCmSequenceMixerLoaderRoutes() []ROCmSequenceMixerLoaderRoute {
	return cloneROCmSequenceMixerLoaderRoutes(rocmmodel.DefaultSequenceMixerLoaderRoutes())
}

func ROCmSequenceMixerLoaderRouteForKind(kind string) (ROCmSequenceMixerLoaderRoute, bool) {
	route, ok := rocmmodel.SequenceMixerLoaderRouteForKind(kind)
	if !ok {
		return ROCmSequenceMixerLoaderRoute{}, false
	}
	return route.Clone(), true
}

func rocmSequenceMixerLoaderRouteFromModel(route rocmmodel.SequenceMixerLoaderRoute) ROCmSequenceMixerLoaderRoute {
	return route.Clone()
}

func rocmSequenceMixerLoaderRoutesFromModel(routes []rocmmodel.SequenceMixerLoaderRoute) []ROCmSequenceMixerLoaderRoute {
	return cloneROCmSequenceMixerLoaderRoutes(routes)
}

func cloneROCmSequenceMixerLoaderRoutes(routes []ROCmSequenceMixerLoaderRoute) []ROCmSequenceMixerLoaderRoute {
	out := make([]ROCmSequenceMixerLoaderRoute, 0, len(routes))
	for _, route := range routes {
		if route.Matched() {
			out = append(out, route.Clone())
		}
	}
	return out
}
