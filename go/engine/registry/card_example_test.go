// SPDX-Licence-Identifier: EUPL-1.2

package registry_test

import (
	"fmt"

	registry "dappco.re/go/inference/engine/registry"
)

func ExampleRegistry_SetCard() {
	r := registry.New()
	r.Put(registry.Entry{ID: "gemma-4-4b-it"})
	res := r.SetCard("gemma-4-4b-it", registry.ModelCard{IntendedUse: "Ethical instruction following."})
	fmt.Println(res.OK)
	// Output: true
}

func ExampleRegistry_GetCard() {
	r := registry.New()
	r.Put(registry.Entry{ID: "gemma-4-4b-it"})
	r.SetCard("gemma-4-4b-it", registry.ModelCard{IntendedUse: "Ethical instruction following."})
	card := r.GetCard("gemma-4-4b-it").Value.(registry.ModelCard)
	fmt.Println(card.IntendedUse)
	// Output: Ethical instruction following.
}
