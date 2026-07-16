// SPDX-Licence-Identifier: EUPL-1.2

package admin

import (
	"net/http"
	"net/http/httptest"

	core "dappco.re/go"
)

// ExampleNewMux demonstrates mounting the admin control plane and calling
// its always-present /v1/admin/machine route.
func ExampleNewMux() {
	mux := NewMux(Config{})
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, PathMachine, nil))
	core.Println(rec.Code)
	// Output:
	// 200
}

// ExampleMachineHash demonstrates the machine-identity token a reload caller
// must echo back as confirm_machine.
func ExampleMachineHash() {
	core.Println(core.HasPrefix(MachineHash(), "lem-"))
	// Output:
	// true
}
