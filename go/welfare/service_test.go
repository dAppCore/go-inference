// SPDX-Licence-Identifier: EUPL-1.2

package welfare

import core "dappco.re/go"

// TestService_Register_Good pins the core-registration hook: Register builds a
// defaulted welfare Service and hands it back as an OK result — the value
// core.WithName("welfare", welfare.Register) mounts.
func TestService_Register_Good(t *core.T) {
	result := Register(nil)
	core.AssertTrue(t, result.OK)

	svc, ok := result.Value.(*Service)
	core.AssertTrue(t, ok)
	if svc == nil {
		t.Fatal("Register produced a nil Service")
	}
	// The service must be usable — defaults applied, ready to gate turns.
	core.AssertEqual(t, "Welfare", svc.ServiceName())
}

// TestService_ServiceName_Good pins the Wails binding name the desktop shell
// looks the welfare subsystem up under.
func TestService_ServiceName_Good(t *core.T) {
	core.AssertEqual(t, "Welfare", New(Config{}).ServiceName())
}
