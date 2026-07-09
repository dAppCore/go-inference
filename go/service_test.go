// SPDX-License-Identifier: EUPL-1.2

package inference

import (
	"testing"

	core "dappco.re/go"
)

// TestNewService_RegistersInferenceService — happy path for canonical factory.
// v1 Options is empty; package behaviour driven by global Backend registry
// independently managed via init() in each backend package.
func TestNewService_RegistersInferenceService(t *testing.T) {
	c := core.New(core.WithService(NewService(Options{})))
	if !c.Service("inference").OK {
		t.Fatal("inference service not registered via NewService")
	}
}

// TestRegisterCore_Imperative — defaults shorthand. Named RegisterCore (not
// Register) to avoid collision with the existing package-level
// `func Register(b Backend)` used by backend implementations to self-register.
func TestRegisterCore_Imperative(t *testing.T) {
	c := core.New(core.WithService(RegisterCore))
	if !c.Service("inference").OK {
		t.Fatal("inference service not registered via RegisterCore")
	}
}
