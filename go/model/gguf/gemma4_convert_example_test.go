// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import "fmt"

// Example shows the computed rope_freqs mask: the first partial-rotary pairs
// rotate (1.0) and the rest are disabled (1e30).
func Example_gemma4RopeFreqsTensor() {
	tensor, _ := gemma4RopeFreqsTensor(512, 0.25)
	freqs := gemma4DecodeF32(tensor.Data)
	fmt.Println("len", len(freqs), "first", freqs[0], "at64", freqs[64])
	// Output: len 256 first 1 at64 1e+30
}
