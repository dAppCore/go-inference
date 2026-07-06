// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"

	core "dappco.re/go"
)

// ExampleAttentionBlock shows the fused attention-half-of-a-decode-step call shape: rmsnorm ->
// wQ projection -> rope -> sdpa over the KV cache -> wO projection -> residual add, one command
// buffer. The call needs MLX_METALLIB_PATH set, so the example guards on it (no Output:
// directive — the GPU dispatch is exercised under the test gate).
func ExampleAttentionBlock() {
	if os.Getenv(MetallibPathEnv) == "" {
		return
	}
	const dModel, nHeads, nKV, headDim, kvLen = 64, 1, 1, 64, 2
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	qDim := nHeads * headDim
	x := toBF16Bytes(syntheticFloat32(dModel, 3))
	normW := toBF16Bytes(syntheticFloat32(dModel, 5))
	wQ := toBF16Bytes(syntheticFloat32(qDim*dModel, 7))
	wO := toBF16Bytes(syntheticFloat32(dModel*qDim, 11))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 13))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 17))

	out, err := AttentionBlock(x, normW, wQ, wO, kCache, vCache, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps)
	if err != nil {
		return
	}
	core.Println(len(out)) // dModel*2 bytes
}

// ExampleAttentionBlockInto is ExampleAttentionBlock with caller-owned output storage.
func ExampleAttentionBlockInto() {
	if os.Getenv(MetallibPathEnv) == "" {
		return
	}
	const dModel, nHeads, nKV, headDim, kvLen = 64, 1, 1, 64, 2
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	qDim := nHeads * headDim
	x := toBF16Bytes(syntheticFloat32(dModel, 3))
	normW := toBF16Bytes(syntheticFloat32(dModel, 5))
	wQ := toBF16Bytes(syntheticFloat32(qDim*dModel, 7))
	wO := toBF16Bytes(syntheticFloat32(dModel*qDim, 11))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 13))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 17))
	out := make([]byte, dModel*bf16Size)

	got, err := AttentionBlockInto(out, x, normW, wQ, wO, kCache, vCache, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps)
	if err != nil {
		return
	}
	core.Println(len(got)) // dModel*2 bytes, reusing out's backing
}
