// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"

	core "dappco.re/go"
)

// ExampleQKNormRopeBF16 shows the fused per-head QK-norm + RoPE call shape: out[head] =
// RoPE(RMSNorm(x[head], weight), offset), one dispatch. The call needs MLX_METALLIB_PATH set,
// so the example guards on it (no Output: directive — the GPU dispatch is exercised under the
// test gate).
func ExampleQKNormRopeBF16() {
	if os.Getenv(MetallibPathEnv) == "" {
		return
	}
	const nHeads, headDim, rotaryDim = 8, 256, 128
	const eps, scale, log2Theta = float32(1e-6), float32(1.0), float32(13.2877) // log2(10000)
	x := toBF16Bytes(syntheticFloat32(nHeads*headDim, headDim+1))
	w := toBF16Bytes(syntheticFloat32(headDim, headDim+7))

	out, err := QKNormRopeBF16(x, w, nHeads, headDim, rotaryDim, 7, scale, eps, log2Theta, nil)
	if err != nil {
		return
	}
	core.Println(len(out)) // nHeads*headDim*2 bytes
}

// ExampleQKNormRopeBF16Into is ExampleQKNormRopeBF16 with caller-owned output storage.
func ExampleQKNormRopeBF16Into() {
	if os.Getenv(MetallibPathEnv) == "" {
		return
	}
	const nHeads, headDim, rotaryDim = 8, 256, 128
	const eps, scale, log2Theta = float32(1e-6), float32(1.0), float32(13.2877)
	x := toBF16Bytes(syntheticFloat32(nHeads*headDim, headDim+1))
	w := toBF16Bytes(syntheticFloat32(headDim, headDim+7))
	out := make([]byte, len(x))

	got, err := QKNormRopeBF16Into(out, x, w, nHeads, headDim, rotaryDim, 7, scale, eps, log2Theta, nil)
	if err != nil {
		return
	}
	core.Println(len(got)) // nHeads*headDim*2 bytes, reusing out's backing
}
