// SPDX-Licence-Identifier: EUPL-1.2

package hip

// ROCmNativeModelLoaderRegistration is the public, copy-safe view of an actual
// native loader registration. Route metadata remains the consumer contract; this
// view proves that a standalone route also has a live ROCm loader behind it.
type ROCmNativeModelLoaderRegistration struct {
	Architecture  string               `json:"architecture,omitempty"`
	Loader        string               `json:"loader,omitempty"`
	Route         ROCmModelLoaderRoute `json:"route"`
	Registered    bool                 `json:"registered,omitempty"`
	NativeRuntime bool                 `json:"native_runtime,omitempty"`
	Standalone    bool                 `json:"standalone,omitempty"`
	TextGenerate  bool                 `json:"text_generate,omitempty"`
}

func (registration ROCmNativeModelLoaderRegistration) clone() ROCmNativeModelLoaderRegistration {
	registration.Route = registration.Route.Clone()
	return registration
}
