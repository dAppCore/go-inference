// SPDX-Licence-Identifier: EUPL-1.2

package kv_test

import (
	"fmt"

	"dappco.re/go/inference/engine/scheme"
	"dappco.re/go/inference/kv"
)

// A KV data format plugs in through the scheme registry: resolve the layer's
// cache mode with scheme.CacheFor, then probe for kv.CacheProvider to apply
// its wire invariants. Adding a format is registering a scheme — never a
// switch arm in the codec.
func ExampleCacheProvider() {
	cacheScheme, _ := scheme.CacheFor("turboquant")
	if provider, ok := cacheScheme.(kv.CacheProvider); ok {
		err := provider.ValidateLayer(&kv.LayerSnapshot{
			CacheMode:          "turboquant",
			TurboQuantPayloads: [][]byte{{0x01}},
		})
		fmt.Printf("turboquant provider validates: %v\n", err == nil)
	}
	// Output: turboquant provider validates: true
}
