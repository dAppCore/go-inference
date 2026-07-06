// SPDX-Licence-Identifier: EUPL-1.2

package registry_test

import (
	"fmt"

	registry "dappco.re/go/inference/engine/registry"
)

func ExampleNewMemStore() {
	s := registry.NewMemStore()
	fmt.Println(len(s.List()))
	// Output: 0
}

func ExampleMemStore_Put() {
	s := registry.NewMemStore()
	res := s.Put(registry.Entry{ID: "gemma-4-4b-it", MemoryBytes: 4_500_000_000})
	fmt.Println(res.OK)
	// Output: true
}

func ExampleMemStore_Get() {
	s := registry.NewMemStore()
	s.Put(registry.Entry{ID: "gemma-4-4b-it"})
	res := s.Get("gemma-4-4b-it")
	fmt.Println(res.OK, res.Value.(registry.Entry).ID)
	// Output: true gemma-4-4b-it
}

func ExampleMemStore_List() {
	s := registry.NewMemStore()
	s.Put(registry.Entry{ID: "zeta"})
	s.Put(registry.Entry{ID: "alpha"})
	for _, e := range s.List() {
		fmt.Println(e.ID)
	}
	// Output:
	// alpha
	// zeta
}

func ExampleMemStore_Delete() {
	s := registry.NewMemStore()
	s.Put(registry.Entry{ID: "gemma-4-4b-it"})
	res := s.Delete("gemma-4-4b-it")
	fmt.Println(res.OK, len(s.List()))
	// Output: true 0
}
