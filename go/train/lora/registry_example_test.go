// SPDX-Licence-Identifier: EUPL-1.2

// Runnable examples for registry.go — kept separate from registry_test.go so
// the godoc-attached usage snippets stay readable.

package lora

import core "dappco.re/go"

// ExampleNewRegistry shows the empty catalogue ready to accept Register.
func ExampleNewRegistry() {
	reg := NewRegistry()
	core.Println(len(reg.List()))
	// Output:
	// 0
}

// ExampleRegistry_Register shows cataloguing an adapter under its Name.
func ExampleRegistry_Register() {
	reg := NewRegistry()
	err := reg.Register(AdapterRef{Name: "support-tone", Path: "/adapters/support"})
	core.Println(err)
	// Output:
	// <nil>
}

// ExampleRegistry_Unregister shows removing a free (unreferenced) adapter
// from the catalogue.
func ExampleRegistry_Unregister() {
	reg := NewRegistry()
	_ = reg.Register(AdapterRef{Name: "support-tone", Path: "/adapters/support"})
	err := reg.Unregister("support-tone")
	core.Println(err)
	// Output:
	// <nil>
}

// ExampleRegistry_Get shows resolving a registered adapter by name.
func ExampleRegistry_Get() {
	reg := NewRegistry()
	_ = reg.Register(AdapterRef{Name: "support-tone", Path: "/adapters/support"})
	ref, _ := reg.Get("support-tone")
	core.Println(ref.Path)
	// Output:
	// /adapters/support
}

// ExampleRegistry_List shows the sorted snapshot of every registered
// adapter.
func ExampleRegistry_List() {
	reg := NewRegistry()
	_ = reg.Register(AdapterRef{Name: "beta", Path: "/adapters/beta"})
	_ = reg.Register(AdapterRef{Name: "alpha", Path: "/adapters/alpha"})
	for _, r := range reg.List() {
		core.Println(r.Name)
	}
	// Output:
	// alpha
	// beta
}

// ExampleRegistry_Acquire shows taking an in-flight lease on an adapter: the
// Pool will not evict it while the lease is outstanding.
func ExampleRegistry_Acquire() {
	reg := NewRegistry()
	_ = reg.Register(AdapterRef{Name: "support-tone", Path: "/adapters/support"})
	id, err := reg.Acquire("support-tone")
	core.Println(err)
	core.Println(reg.RefCount(id))
	// Output:
	// <nil>
	// 1
}

// ExampleRegistry_Release shows dropping the lease taken by Acquire.
func ExampleRegistry_Release() {
	reg := NewRegistry()
	_ = reg.Register(AdapterRef{Name: "support-tone", Path: "/adapters/support"})
	id, _ := reg.Acquire("support-tone")
	reg.Release(id)
	core.Println(reg.RefCount(id))
	// Output:
	// 0
}

// ExampleRegistry_RefCount shows the outstanding-lease count for an adapter.
func ExampleRegistry_RefCount() {
	reg := NewRegistry()
	_ = reg.Register(AdapterRef{Name: "support-tone", Path: "/adapters/support"})
	id, _ := reg.Acquire("support-tone")
	_, _ = reg.Acquire("support-tone")
	core.Println(reg.RefCount(id))
	// Output:
	// 2
}

// ExampleRegistry_InUse shows the eviction-eligibility check: an adapter
// with any outstanding lease is in use.
func ExampleRegistry_InUse() {
	reg := NewRegistry()
	_ = reg.Register(AdapterRef{Name: "support-tone", Path: "/adapters/support"})
	id, _ := reg.Acquire("support-tone")
	core.Println(reg.InUse(id))
	reg.Release(id)
	core.Println(reg.InUse(id))
	// Output:
	// true
	// false
}
