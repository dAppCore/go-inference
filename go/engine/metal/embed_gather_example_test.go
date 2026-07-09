// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"

	core "dappco.re/go"
)

// ExampleEmbedGatherQuantBF16 shows the GPU dequant-gather call shape: one
// token's affine-quantised embedding row (packed codes + per-group
// scales/biases) in, the dModel bf16 row times embedScale out. The call needs
// MLX_METALLIB_PATH set, so the example guards on it (no Output: directive —
// the GPU dispatch is exercised under the test gate).
func ExampleEmbedGatherQuantBF16() {
	if os.Getenv(MetallibPathEnv) == "" {
		return
	}
	const vocab, dModel, groupSize, bits = 256, 512, 64, 4
	packed, scales, biases := embedGatherQuantFixture(vocab, dModel, groupSize, bits)

	row, err := EmbedGatherQuantBF16(42, packed, scales, biases, dModel, groupSize, bits, 0.5)
	if err != nil {
		return
	}
	core.Println(len(row)) // dModel bf16 values, 2 bytes each
}

// ExampleEmbedGatherQuantBF16Into is ExampleEmbedGatherQuantBF16 with
// caller-owned output storage: pass a slice with capacity for dModel bf16
// bytes and the kernel writes it directly (no per-call row allocation on the
// chained decode's input seam).
func ExampleEmbedGatherQuantBF16Into() {
	if os.Getenv(MetallibPathEnv) == "" {
		return
	}
	const vocab, dModel, groupSize, bits = 256, 512, 64, 4
	packed, scales, biases := embedGatherQuantFixture(vocab, dModel, groupSize, bits)
	out := make([]byte, dModel*bf16Size)

	row, err := EmbedGatherQuantBF16Into(out, 42, packed, scales, biases, dModel, groupSize, bits, 0.5)
	if err != nil {
		return
	}
	core.Println(len(row)) // dModel bf16 values, backed by out
}
