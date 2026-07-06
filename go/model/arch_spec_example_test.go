// SPDX-Licence-Identifier: EUPL-1.2

package model

import core "dappco.re/go"

// ExampleRegisterArch shows the reactive registration a model package's init() does: one
// spec claims every alias it ships (a bare id plus any multimodal wrapper aliases), and
// the reactive loader resolves any of them through LookupArch.
func ExampleRegisterArch() {
	RegisterArch(ArchSpec{
		ModelTypes: []string{"examplearch", "examplearch_text"},
		Parse:      func([]byte) (ArchConfig, error) { return fakeArchConfig{}, nil },
	})
	_, ok := LookupArch("examplearch_text")
	core.Println(ok) // the wrapper alias resolves to the same registration
	// Output: true
}

// ExampleLookupArch shows the reactive loader's dispatch: a registered model_type
// resolves to its spec; an unregistered one misses cleanly (ok=false), so the loader can
// report "no architecture registered" rather than panic.
func ExampleLookupArch() {
	RegisterArch(ArchSpec{ModelTypes: []string{"examplearch2"}})
	_, ok := LookupArch("examplearch2")
	core.Println(ok)
	_, ok = LookupArch("never-registered")
	core.Println(ok)
	// Output:
	// true
	// false
}
