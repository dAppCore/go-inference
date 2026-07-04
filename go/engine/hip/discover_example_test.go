package hip

import core "dappco.re/go"

func ExampleDiscoverModels() {
	models, err := DiscoverModels(core.TempDir())
	core.Println(err == nil, len(models) >= 0)
	// Output: true true
}
