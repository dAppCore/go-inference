// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"dappco.re/go/inference"
	rocmmodel "dappco.re/go/inference/engine/hip/model"
)

const (
	ROCmQuantSchemeRegistryContract = rocmmodel.QuantSchemeRegistryContract

	rocmQuantSchemeRegistryRouteName = rocmmodel.QuantSchemeRouteName
	rocmQuantSchemeRuntimeMetadata   = rocmmodel.QuantSchemeRuntimeMetadata
	rocmQuantSchemeRuntimePlannedHIP = rocmmodel.QuantSchemeRuntimePlannedHIP
)

// ROCmQuantScheme is the pure weight-quant scheme catalogue entry that mirrors
// go-mlx's scheme.QuantFor contract. Concrete model-pack routes still live in
// ROCmQuantLoaderRoute; this smaller surface lets consumers react to a model's
// declared quantization kind before selecting a concrete loader.
type ROCmQuantScheme struct {
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

func (scheme ROCmQuantScheme) Matched() bool {
	return scheme.Contract != "" && scheme.Kind != ""
}

func (scheme ROCmQuantScheme) clone() ROCmQuantScheme {
	scheme.Labels = cloneStringMap(scheme.Labels)
	return scheme
}

func DefaultROCmQuantSchemes() []ROCmQuantScheme {
	return rocmQuantSchemesFromModel(rocmmodel.DefaultQuantSchemes())
}

// RegisterROCmQuantScheme registers or replaces a weight-quantization scheme in
// the ROCm catalogue. It mirrors go-mlx's quant-loader registration at the
// contract layer: quant formats can self-register their metadata without adding
// another central switch.
func RegisterROCmQuantScheme(scheme ROCmQuantScheme) {
	rocmmodel.RegisterQuantScheme(rocmQuantSchemeToModel(scheme))
}

// RegisteredROCmQuantSchemeKinds returns extension scheme kinds in resolution
// order. Built-in schemes are intentionally not included.
func RegisteredROCmQuantSchemeKinds() []string {
	return rocmmodel.RegisteredQuantSchemeKinds()
}

func registeredROCmQuantSchemeSnapshot() []ROCmQuantScheme {
	return rocmQuantSchemesFromModel(rocmmodel.RegisteredQuantSchemes())
}

func normalizeRegisteredROCmQuantScheme(scheme ROCmQuantScheme) ROCmQuantScheme {
	return rocmQuantSchemeFromModel(rocmmodel.NormalizeQuantScheme(rocmQuantSchemeToModel(scheme)))
}

func ROCmQuantSchemeForKind(kind string) (ROCmQuantScheme, bool) {
	scheme, ok := rocmmodel.QuantSchemeForKind(kind)
	if !ok {
		return ROCmQuantScheme{}, false
	}
	return rocmQuantSchemeFromModel(scheme), true
}

func DefaultROCmQuantSchemeKinds() []string {
	return rocmmodel.DefaultQuantSchemeKinds()
}

func normalizeROCmQuantSchemeKind(kind string) string {
	return rocmmodel.NormalizeQuantSchemeKind(kind)
}

func rocmQuantSchemeKinds(schemes []ROCmQuantScheme) []string {
	return rocmmodel.QuantSchemeKinds(rocmQuantSchemesToModel(schemes))
}

func rocmQuantSchemeLabels(scheme ROCmQuantScheme) map[string]string {
	converted := rocmmodel.NormalizeQuantScheme(rocmQuantSchemeToModel(scheme))
	return cloneStringMap(converted.Labels)
}

func cloneROCmQuantSchemes(schemes []ROCmQuantScheme) []ROCmQuantScheme {
	out := append([]ROCmQuantScheme(nil), schemes...)
	for i := range out {
		out[i] = out[i].clone()
	}
	return out
}

func rocmQuantSchemeKindsCSV(schemes []ROCmQuantScheme) string {
	return rocmmodel.QuantSchemeKindsCSV(rocmQuantSchemesToModel(schemes))
}

func rocmQuantSchemeToModel(scheme ROCmQuantScheme) rocmmodel.QuantScheme {
	return rocmmodel.QuantScheme{
		Contract:      scheme.Contract,
		Name:          scheme.Name,
		Kind:          scheme.Kind,
		Bits:          scheme.Bits,
		Loader:        scheme.Loader,
		Source:        scheme.Source,
		Runtime:       scheme.Runtime,
		RuntimeStatus: scheme.RuntimeStatus,
		Registered:    scheme.Registered,
		NativeRuntime: scheme.NativeRuntime,
		MetadataOnly:  scheme.MetadataOnly,
		Planned:       scheme.Planned,
		Labels:        cloneStringMap(scheme.Labels),
	}
}

func rocmQuantSchemeFromModel(scheme rocmmodel.QuantScheme) ROCmQuantScheme {
	return ROCmQuantScheme{
		Contract:      scheme.Contract,
		Name:          scheme.Name,
		Kind:          scheme.Kind,
		Bits:          scheme.Bits,
		Loader:        scheme.Loader,
		Source:        scheme.Source,
		Runtime:       scheme.Runtime,
		RuntimeStatus: scheme.RuntimeStatus,
		Registered:    scheme.Registered,
		NativeRuntime: scheme.NativeRuntime,
		MetadataOnly:  scheme.MetadataOnly,
		Planned:       scheme.Planned,
		Labels:        cloneStringMap(scheme.Labels),
	}
}

func rocmQuantSchemesToModel(schemes []ROCmQuantScheme) []rocmmodel.QuantScheme {
	out := make([]rocmmodel.QuantScheme, 0, len(schemes))
	for _, scheme := range schemes {
		out = append(out, rocmQuantSchemeToModel(scheme))
	}
	return out
}

func rocmQuantSchemesFromModel(schemes []rocmmodel.QuantScheme) []ROCmQuantScheme {
	out := make([]ROCmQuantScheme, 0, len(schemes))
	for _, scheme := range schemes {
		out = append(out, rocmQuantSchemeFromModel(scheme))
	}
	return out
}
