// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"

	core "dappco.re/go"
)

// ExampleDiffusionSDPA shows the block-diffusion canvas attention call shape: q attends the full
// [nKVHeads, keyLen, headDim] cache under an optional additive mask (0 = attend, -Inf = blocked).
// The call needs MLX_METALLIB_PATH set, so the example guards on it (no Output: directive — the
// GPU dispatch is exercised under the test gate).
func ExampleDiffusionSDPA() {
	if os.Getenv(MetallibPathEnv) == "" {
		return
	}
	const qLen, keyLen, nHeads, nKVHeads, headDim = 2, 3, 2, 1, 8
	q := toBF16Bytes(syntheticFloat32(nHeads*qLen*headDim, 3))
	k := toBF16Bytes(syntheticFloat32(nKVHeads*keyLen*headDim, 5))
	v := toBF16Bytes(syntheticFloat32(nKVHeads*keyLen*headDim, 7))

	out, err := DiffusionSDPA(q, k, v, qLen, keyLen, nHeads, nKVHeads, headDim, 0.125, nil)
	if err != nil {
		return
	}
	core.Println(len(out)) // 96 bytes: nHeads*qLen*headDim*2
}
