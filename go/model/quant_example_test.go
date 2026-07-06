// SPDX-Licence-Identifier: EUPL-1.2

package model

import core "dappco.re/go"

// ExampleBackendQuant shows the (backend,kind) cross-section: a backend registers its
// quant compute under its own name, and the engine resolves a weight's matvec by the
// backend it loaded + the kind the model declares. Two backends can register the SAME
// kind ("affine") without colliding — the fact that distinguishes this registry from
// pkg/scheme's kind-only one. (fakeQuant stands in for a real impl, which lives in
// pkg/native / pkg/metal.) The example re-registers its own backends, so the global
// registry's prior state cannot perturb the result — it is deterministic.
func ExampleBackendQuant() {
	// a backend's init() does this once — here two backends, same kind.
	RegisterBackendQuant("native", fakeQuant{kind: "affine", bits: 4, tag: 0x1})
	RegisterBackendQuant("metal", fakeQuant{kind: "affine", bits: 4, tag: 0x2})

	// the engine resolves by (loaded backend, declared kind):
	q, ok := BackendQuant("native", "affine")
	core.Println(ok) // native registered "affine"
	if ok {
		core.Println(q.Bits()) // the nominal width of the resolved impl
	}

	// an unregistered (backend,kind) is detectable, not a panic:
	_, missing := BackendQuant("rocm", "affine")
	core.Println(missing) // no rocm backend has registered yet
	// Output:
	// true
	// 4
	// false
}

// ExampleRegisterBackendQuant shows a backend's init()-time self-registration: it
// records its quant compute keyed by its own name plus the scheme's Kind(), returning a
// core.Result so a registration mistake (never happens today — Set on this registry
// cannot fail) is still visible through the ordinary error-handling shape.
func ExampleRegisterBackendQuant() {
	r := RegisterBackendQuant("example-native", fakeQuant{kind: "affine", bits: 4, tag: 0x7})
	core.Println(r.OK)
	// Output: true
}
