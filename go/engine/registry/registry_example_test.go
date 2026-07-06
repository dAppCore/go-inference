// SPDX-Licence-Identifier: EUPL-1.2

package registry_test

import (
	"fmt"

	registry "dappco.re/go/inference/engine/registry"
)

func ExampleNew() {
	r := registry.New()
	fmt.Println(len(r.List()))
	// Output: 0
}

func ExampleNewWithStore() {
	s := registry.NewMemStore()
	s.Put(registry.Entry{ID: "gemma-4-4b-it", Aliases: []string{"lemma"}})
	r := registry.NewWithStore(s)
	fmt.Println(r.Resolve("lemma").OK)
	// Output: true
}

func ExampleRegistry_Put() {
	r := registry.New()
	res := r.Put(registry.Entry{ID: "gemma-4-4b-it", Aliases: []string{"lemma"}})
	fmt.Println(res.OK)
	// Output: true
}

func ExampleRegistry_Get() {
	r := registry.New()
	r.Put(registry.Entry{ID: "gemma-4-4b-it"})
	fmt.Println(r.Get("gemma-4-4b-it").OK)
	// Output: true
}

func ExampleRegistry_Resolve() {
	r := registry.New()
	r.Put(registry.Entry{ID: "gemma-4-4b-it", Aliases: []string{"lemma"}})
	e := r.Resolve("lemma").Value.(registry.Entry)
	fmt.Println(e.ID)
	// Output: gemma-4-4b-it
}

func ExampleRegistry_List() {
	r := registry.New()
	r.Put(registry.Entry{ID: "zeta"})
	r.Put(registry.Entry{ID: "alpha"})
	for _, e := range r.List() {
		fmt.Println(e.ID)
	}
	// Output:
	// alpha
	// zeta
}

func ExampleRegistry_Filter() {
	r := registry.New()
	r.Put(registry.Entry{ID: "tool-model", Capabilities: registry.Capabilities{Tools: true}})
	r.Put(registry.Entry{ID: "plain-model"})
	fits := r.Filter(registry.Filter{Tools: true})
	fmt.Println(len(fits), fits[0].ID)
	// Output: 1 tool-model
}

func ExampleRegistry_Delete() {
	r := registry.New()
	r.Put(registry.Entry{ID: "gemma-4-4b-it"})
	r.Delete("gemma-4-4b-it")
	fmt.Println(r.Resolve("gemma-4-4b-it").OK)
	// Output: false
}

func ExampleRegistry_FitsDevice() {
	r := registry.New()
	r.Put(registry.Entry{ID: "gemma-4-4b-it", MemoryBytes: 4_500_000_000})
	r.Put(registry.Entry{ID: "gemma-4-31b-it", MemoryBytes: 24_000_000_000})
	fits := r.FitsDevice(8 << 30)
	fmt.Println(len(fits), fits[0].ID)
	// Output: 1 gemma-4-4b-it
}

func ExampleRegistry_FitsDeviceWith() {
	r := registry.New()
	r.Put(registry.Entry{
		ID:           "vision-model",
		MemoryBytes:  4_500_000_000,
		Capabilities: registry.Capabilities{Vision: true},
	})
	r.Put(registry.Entry{ID: "text-model", MemoryBytes: 4_000_000_000})
	fits := r.FitsDeviceWith(8<<30, registry.Filter{Vision: true})
	fmt.Println(len(fits), fits[0].ID)
	// Output: 1 vision-model
}
