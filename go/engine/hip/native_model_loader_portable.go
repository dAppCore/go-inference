// SPDX-Licence-Identifier: EUPL-1.2

//go:build !linux || !amd64 || rocm_legacy_server

package hip

// RegisteredROCmNativeModelLoaderRegistrations returns no native loaders on
// portable builds. Route and profile registries remain available for planning.
func RegisteredROCmNativeModelLoaderRegistrations() []ROCmNativeModelLoaderRegistration {
	return nil
}

// ROCmNativeModelLoaderRegistrationForArchitecture reports no native loader on
// portable builds. Use ROCmModelLoaderRouteForArchitecture for metadata.
func ROCmNativeModelLoaderRegistrationForArchitecture(string) (ROCmNativeModelLoaderRegistration, bool) {
	return ROCmNativeModelLoaderRegistration{}, false
}
