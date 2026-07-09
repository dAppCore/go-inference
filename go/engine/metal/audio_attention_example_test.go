// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"

	core "dappco.re/go"
)

// ExampleAudioAttention shows the Conformer chunked relative-position attention call shape: a
// bf16 [T,hidden] turn in, the same shape out. The call needs MLX_METALLIB_PATH set, so the
// example guards on it (no Output: directive — the GPU dispatch is exercised under the test
// gate).
func ExampleAudioAttention() {
	if os.Getenv(MetallibPathEnv) == "" {
		return
	}
	const hid, H, D, chunk, past, future, T = 16, 2, 8, 4, 2, 1, 6
	w := audioAttentionWeightsFixture(hid, H, D, past+1)
	cfg := audioAttentionCfgFixture(hid, H, D, chunk, past, future)
	x := toBF16Bytes(syntheticFloat32(T*hid, 17))

	out, err := AudioAttention(x, w, cfg)
	if err != nil {
		return
	}
	core.Println(len(out)) // T*hidden*2 bytes
}

// ExampleAudioAttentionF32 is the fp32-tower sibling of ExampleAudioAttention, used where the
// Conformer runs promoted to float32 after the gradient-clamp.
func ExampleAudioAttentionF32() {
	if os.Getenv(MetallibPathEnv) == "" {
		return
	}
	const hid, H, D, chunk, past, future, T = 16, 2, 8, 4, 2, 1, 6
	w := audioAttentionWeightsFixture(hid, H, D, past+1)
	cfg := audioAttentionCfgFixture(hid, H, D, chunk, past, future)
	x := syntheticFloat32(T*hid, 17)

	out, err := AudioAttentionF32(x, w, cfg)
	if err != nil {
		return
	}
	core.Println(len(out)) // T*hidden float32 elements
}
