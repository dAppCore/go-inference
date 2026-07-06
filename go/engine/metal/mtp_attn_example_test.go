// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"

	core "dappco.re/go"
)

// ExampleSDPACausalBF16 shows the MTP batched-verify causal attention call shape: H query heads
// (the K draft rows) attend Hkv key/value heads causally over their own [qL,kL] window. The
// call needs MLX_METALLIB_PATH set, so the example guards on it (no Output: directive — the GPU
// dispatch is exercised under the test gate).
func ExampleSDPACausalBF16() {
	if os.Getenv(MetallibPathEnv) == "" {
		return
	}
	const H, Hkv, qL, kL, D = 4, 2, 3, 5, 16
	scale := sdpaScale(D)
	q := toBF16Bytes(syntheticFloat32(H*qL*D, 3))
	k := toBF16Bytes(syntheticFloat32(Hkv*kL*D, 5))
	v := toBF16Bytes(syntheticFloat32(Hkv*kL*D, 7))

	out, err := SDPACausalBF16(q, k, v, H, Hkv, qL, kL, D, scale)
	if err != nil {
		return
	}
	core.Println(len(out)) // H*qL*D*2 bytes
}

// ExampleSDPACausalBF16Into is ExampleSDPACausalBF16 with caller-owned output storage.
func ExampleSDPACausalBF16Into() {
	if os.Getenv(MetallibPathEnv) == "" {
		return
	}
	const H, Hkv, qL, kL, D = 4, 2, 3, 5, 16
	scale := sdpaScale(D)
	q := toBF16Bytes(syntheticFloat32(H*qL*D, 3))
	k := toBF16Bytes(syntheticFloat32(Hkv*kL*D, 5))
	v := toBF16Bytes(syntheticFloat32(Hkv*kL*D, 7))
	out := make([]byte, H*qL*D*bf16Size)

	got, err := SDPACausalBF16Into(out, q, k, v, H, Hkv, qL, kL, D, scale)
	if err != nil {
		return
	}
	core.Println(len(got)) // H*qL*D*2 bytes, reusing out's backing
}
