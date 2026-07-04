// SPDX-Licence-Identifier: EUPL-1.2

package spine_test

import (
	"fmt"

	"dappco.re/go/inference/spine"
)

func ExamplePromptChunksToString() {
	chunks := func(yield func(string) bool) {
		for _, s := range []string{"Hello", ", ", "world"} {
			if !yield(s) {
				return
			}
		}
	}
	fmt.Println(spine.PromptChunksToString(chunks))
	// Output: Hello, world
}
