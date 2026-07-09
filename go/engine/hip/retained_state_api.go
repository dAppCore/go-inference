// SPDX-Licence-Identifier: EUPL-1.2

package hip

import "dappco.re/go/inference"

// ROCmRetainedStateStatus is the copy-safe, application-facing summary of a
// model's retained decode contract. It is derived from the state-context route
// so CLI, daemon, and API consumers make the same runtime-owned KV decision.
type ROCmRetainedStateStatus struct {
	Contract                string                         `json:"contract,omitempty"`
	Architecture            string                         `json:"architecture,omitempty"`
	Family                  string                         `json:"family,omitempty"`
	Runtime                 string                         `json:"runtime,omitempty"`
	RuntimeStatus           inference.FeatureRuntimeStatus `json:"runtime_status,omitempty"`
	Status                  ROCmStateContextRouteStatus    `json:"status,omitempty"`
	StateSession            bool                           `json:"state_session,omitempty"`
	SleepState              bool                           `json:"sleep_state,omitempty"`
	WakeState               bool                           `json:"wake_state,omitempty"`
	ForkState               bool                           `json:"fork_state,omitempty"`
	CaptureState            bool                           `json:"capture_state,omitempty"`
	RestoreState            bool                           `json:"restore_state,omitempty"`
	RuntimeOwnedKV          bool                           `json:"runtime_owned_kv,omitempty"`
	PromptReplayRefused     bool                           `json:"prompt_replay_refused,omitempty"`
	RemainingContextDefault bool                           `json:"remaining_context_default,omitempty"`
	ModelContextWindow      bool                           `json:"model_context_window,omitempty"`
	DeviceKVState           bool                           `json:"device_kv_state,omitempty"`
	RetainedStateRequired   bool                           `json:"retained_state_required,omitempty"`
	AttachedDrafterState    bool                           `json:"attached_drafter_state,omitempty"`
	DefaultDeviceKVMode     string                         `json:"default_device_kv_mode,omitempty"`
	CacheModes              []string                       `json:"cache_modes,omitempty"`
	StateBackends           []string                       `json:"state_backends,omitempty"`
	Labels                  map[string]string              `json:"labels,omitempty"`
}

func (status ROCmRetainedStateStatus) RuntimeOwnedDecodeReady() bool {
	return status.StateSession &&
		status.RuntimeOwnedKV &&
		status.DeviceKVState &&
		status.RetainedStateRequired &&
		status.PromptReplayRefused
}

func ROCmRetainedStateForIdentity(path string, model inference.ModelIdentity) (ROCmRetainedStateStatus, bool) {
	route, ok := ROCmStateContextRouteForIdentity(path, model)
	if !ok {
		return ROCmRetainedStateStatus{}, false
	}
	return rocmRetainedStateStatusFromRoute(route), true
}

func ROCmRetainedStateForInfo(path string, info inference.ModelInfo, labels map[string]string) (ROCmRetainedStateStatus, bool) {
	route, ok := ROCmStateContextRouteForInfo(path, info, labels)
	if !ok {
		return ROCmRetainedStateStatus{}, false
	}
	return rocmRetainedStateStatusFromRoute(route), true
}

func ROCmRetainedStateForInspection(inspection *inference.ModelPackInspection) (ROCmRetainedStateStatus, bool) {
	route, ok := ROCmStateContextRouteForInspection(inspection)
	if !ok {
		return ROCmRetainedStateStatus{}, false
	}
	return rocmRetainedStateStatusFromRoute(route), true
}

func ROCmRetainedStateForModel(model inference.TextModel) (ROCmRetainedStateStatus, bool) {
	profile, ok := ResolveROCmModelProfileForModel(model)
	if !ok {
		return ROCmRetainedStateStatus{}, false
	}
	route := profile.StateContextRoute
	if !route.Matched() {
		route = ROCmStateContextRouteForProfile(profile)
	}
	if !route.Matched() {
		return ROCmRetainedStateStatus{}, false
	}
	return rocmRetainedStateStatusFromRoute(route), true
}

func rocmRetainedStateStatusFromRoute(route ROCmStateContextRoute) ROCmRetainedStateStatus {
	route = route.Clone()
	labels := cloneStringMap(route.Labels)
	if labels == nil {
		labels = rocmApplyROCmStateContextRouteLabels(nil, route)
	}
	return ROCmRetainedStateStatus{
		Contract:                route.Contract,
		Architecture:            route.Architecture,
		Family:                  route.Family,
		Runtime:                 route.Runtime,
		RuntimeStatus:           route.RuntimeStatus,
		Status:                  route.Status,
		StateSession:            route.StateSession,
		SleepState:              route.SleepState,
		WakeState:               route.WakeState,
		ForkState:               route.ForkState,
		CaptureState:            route.CaptureState,
		RestoreState:            route.RestoreState,
		RuntimeOwnedKV:          route.RuntimeOwnedKV,
		PromptReplayRefused:     route.PromptReplayRefused,
		RemainingContextDefault: route.RemainingContextDefault,
		ModelContextWindow:      route.ModelContextWindow,
		DeviceKVState:           route.DeviceKVState,
		RetainedStateRequired:   route.RetainedStateRequired,
		AttachedDrafterState:    route.AttachedDrafterState,
		DefaultDeviceKVMode:     route.DefaultDeviceKVMode,
		CacheModes:              append([]string(nil), route.CacheModes...),
		StateBackends:           append([]string(nil), route.StateBackends...),
		Labels:                  labels,
	}
}
