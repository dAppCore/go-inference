// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import "fmt"

// Example shows gemma4Metadata deriving the block count and the per-layer
// feed-forward length array from a checkpoint's config.json.
func Example_gemma4Metadata() {
	entries, _ := gemma4Metadata([]byte(gemma4TestConfig), gemma4TestFeedForward, 15, "")
	for _, e := range entries {
		switch e.Key {
		case "gemma4.block_count":
			fmt.Println(e.Key, "=", e.Value)
		case "gemma4.feed_forward_length":
			fmt.Println(e.Key, "len", len(e.Value.([]int32)))
		}
	}
	// Output:
	// gemma4.block_count = 6
	// gemma4.feed_forward_length len 6
}
