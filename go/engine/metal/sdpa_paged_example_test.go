// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"

	core "dappco.re/go"
)

// ExampleSDPAPagedBF16 shows the paged-KV-cache attention call shape: keyPages/valuePages are
// head-major [nKVHeads, pageLen, headDim] pages, attended without host-side concatenation. The
// call needs MLX_METALLIB_PATH set (and the lthn_kernels custom metallib), so the example guards
// on it (no Output: directive — the GPU dispatch is exercised under the test gate).
func ExampleSDPAPagedBF16() {
	if os.Getenv(MetallibPathEnv) == "" {
		return
	}
	const nHeads, nKVHeads, headDim = 4, 2, 64
	q := toBF16Bytes(syntheticFloat32(nHeads*headDim, 3))
	kPages := [][]byte{toBF16Bytes(syntheticFloat32(nKVHeads*3*headDim, 5))}
	vPages := [][]byte{toBF16Bytes(syntheticFloat32(nKVHeads*3*headDim, 7))}

	out, err := SDPAPagedBF16(q, kPages, vPages, nHeads, nKVHeads, headDim, 0.125)
	if err != nil {
		return
	}
	core.Println(len(out)) // nHeads*headDim*2 bytes
}

// ExampleSDPAPagedBF16Into is ExampleSDPAPagedBF16 with caller-owned output storage.
func ExampleSDPAPagedBF16Into() {
	if os.Getenv(MetallibPathEnv) == "" {
		return
	}
	const nHeads, nKVHeads, headDim = 4, 2, 64
	q := toBF16Bytes(syntheticFloat32(nHeads*headDim, 3))
	kPages := [][]byte{toBF16Bytes(syntheticFloat32(nKVHeads*3*headDim, 5))}
	vPages := [][]byte{toBF16Bytes(syntheticFloat32(nKVHeads*3*headDim, 7))}
	out := make([]byte, nHeads*headDim*bf16Size)

	got, err := SDPAPagedBF16Into(out, q, kPages, vPages, nHeads, nKVHeads, headDim, 0.125)
	if err != nil {
		return
	}
	core.Println(len(got)) // nHeads*headDim*2 bytes, reusing out's backing
}
