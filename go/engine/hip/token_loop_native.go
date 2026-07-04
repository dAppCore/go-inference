// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import "dappco.re/go/inference"

// ROCmRuntimeTokenModel is the loaded native ROCm analogue of go-mlx's
// SessionModel: callers drive text generation through inference.TextModel, and
// the model exposes its retained StateSession for runtime-owned KV lifecycle
// operations. The HIP stepper remains package-local; this interface is the safe
// application contract.
type ROCmRuntimeTokenModel interface {
	inference.TextModel
	RuntimeStateSession() *StateSession
	ResetState() error
}

func ROCmRuntimeTokenSession(model inference.TextModel) (*StateSession, bool) {
	runtime, ok := model.(ROCmRuntimeTokenModel)
	if !ok {
		return nil, false
	}
	session := runtime.RuntimeStateSession()
	if session == nil {
		return nil, false
	}
	return session, true
}

var _ ROCmRuntimeTokenModel = (*rocmModel)(nil)
