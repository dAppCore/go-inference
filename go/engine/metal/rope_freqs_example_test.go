// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"

	core "dappco.re/go"
)

// ExampleRoPEFreqsBF16 shows the explicit-frequency (YaRN long-context spectrum) rotary
// embedding call shape: invFreqs carries rotaryDim/2 per-dimension inverse frequencies. The call
// needs MLX_METALLIB_PATH set, so the example guards on it (no Output: directive — the GPU
// dispatch is exercised under the test gate).
func ExampleRoPEFreqsBF16() {
	if os.Getenv(MetallibPathEnv) == "" {
		return
	}
	const b, nHeads, headDim, rotaryDim = 1, 8, 64, 64
	x := toBF16Bytes(syntheticFloat32(b*nHeads*headDim, 5))
	invFreqs := plainRopeInvFreqs(10000, rotaryDim)

	out, err := RoPEFreqsBF16(x, b, nHeads, headDim, rotaryDim, invFreqs, 1, 7, false)
	if err != nil {
		return
	}
	core.Println(len(out)) // b*nHeads*headDim*2 bytes
}

// ExampleRoPEFreqsBF16Into is ExampleRoPEFreqsBF16 with caller-owned output storage.
func ExampleRoPEFreqsBF16Into() {
	if os.Getenv(MetallibPathEnv) == "" {
		return
	}
	const b, nHeads, headDim, rotaryDim = 1, 8, 64, 64
	x := toBF16Bytes(syntheticFloat32(b*nHeads*headDim, 5))
	invFreqs := plainRopeInvFreqs(10000, rotaryDim)
	out := make([]byte, len(x))

	got, err := RoPEFreqsBF16Into(out, x, b, nHeads, headDim, rotaryDim, invFreqs, 1, 7, false)
	if err != nil {
		return
	}
	core.Println(len(got)) // b*nHeads*headDim*2 bytes, reusing out's backing
}
