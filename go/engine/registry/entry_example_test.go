// SPDX-Licence-Identifier: EUPL-1.2

package registry_test

import (
	"fmt"

	registry "dappco.re/go/inference/engine/registry"
)

func ExampleEntry() {
	e := registry.Entry{
		ID:           "gemma-4-4b-it",
		Aliases:      []string{"lemma"},
		Architecture: "gemma4",
		Format:       registry.FormatGGUF,
		MemoryBytes:  4_500_000_000,
		Status:       registry.StatusReady,
	}
	fmt.Println(e.ID, e.Format, e.Status)
	// Output: gemma-4-4b-it gguf ready
}

func ExampleFilter() {
	r := registry.New()
	r.Put(registry.Entry{
		ID:           "gemma-4-4b-it",
		Capabilities: registry.Capabilities{Tools: true},
		Status:       registry.StatusReady,
	})
	fits := r.Filter(registry.Filter{Tools: true, ReadyOnly: true})
	fmt.Println(len(fits))
	// Output: 1
}
