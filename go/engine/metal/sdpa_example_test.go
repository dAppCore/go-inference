// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"

	core "dappco.re/go"
)

// ExampleSDPA shows the single-query decode attention call shape: q (b,nHeads,1,headDim) attends
// k/v (b,nKVHeads,kvLen,headDim). The call needs MLX_METALLIB_PATH set, so the example guards on
// it (no Output: directive — the GPU dispatch is exercised under the test gate).
func ExampleSDPA() {
	if os.Getenv(MetallibPathEnv) == "" {
		return
	}
	const b, nHeads, nKV, headDim, kvLen = 1, 8, 4, 64, 16
	q := toBF16Bytes(syntheticFloat32(b*nHeads*headDim, 3))
	k := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 5))
	v := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 7))

	out, err := SDPA(q, k, v, b, nHeads, nKV, headDim, kvLen, 0.125)
	if err != nil {
		return
	}
	core.Println(len(out)) // b*nHeads*headDim*2 bytes
}

// ExampleSDPAInto is ExampleSDPA with caller-owned output storage.
func ExampleSDPAInto() {
	if os.Getenv(MetallibPathEnv) == "" {
		return
	}
	const b, nHeads, nKV, headDim, kvLen = 1, 8, 4, 64, 16
	q := toBF16Bytes(syntheticFloat32(b*nHeads*headDim, 3))
	k := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 5))
	v := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 7))
	out := make([]byte, b*nHeads*headDim*bf16Size)

	got, err := SDPAInto(out, q, k, v, b, nHeads, nKV, headDim, kvLen, 0.125)
	if err != nil {
		return
	}
	core.Println(len(got)) // b*nHeads*headDim*2 bytes, reusing out's backing
}

// ExampleSDPA2Pass shows the long-context two-pass SDPA — same call shape as ExampleSDPA, but
// the cache reduction fans over multiple threadgroups so it keeps scaling where the single-pass
// kernel's one-threadgroup-per-head reduction degrades.
func ExampleSDPA2Pass() {
	if os.Getenv(MetallibPathEnv) == "" {
		return
	}
	const b, nHeads, nKV, headDim, kvLen = 1, 4, 2, 64, 8
	q := toBF16Bytes(syntheticFloat32(b*nHeads*headDim, 3))
	k := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 5))
	v := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 7))

	out, err := SDPA2Pass(q, k, v, b, nHeads, nKV, headDim, kvLen, 0.125)
	if err != nil {
		return
	}
	core.Println(len(out)) // b*nHeads*headDim*2 bytes
}

// ExampleSDPA2PassInto is ExampleSDPA2Pass with caller-owned output storage.
func ExampleSDPA2PassInto() {
	if os.Getenv(MetallibPathEnv) == "" {
		return
	}
	const b, nHeads, nKV, headDim, kvLen = 1, 4, 2, 64, 8
	q := toBF16Bytes(syntheticFloat32(b*nHeads*headDim, 3))
	k := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 5))
	v := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 7))
	out := make([]byte, b*nHeads*headDim*bf16Size)

	got, err := SDPA2PassInto(out, q, k, v, b, nHeads, nKV, headDim, kvLen, 0.125)
	if err != nil {
		return
	}
	core.Println(len(got)) // b*nHeads*headDim*2 bytes, reusing out's backing
}
