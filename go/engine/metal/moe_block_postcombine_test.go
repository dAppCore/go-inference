// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"
)

// moe_block_postcombine_test.go drives the GPU compute path of
// moeBlockPostCombineBF16 — the gemma4 MoE post-combine tail: RMSNorm(h1)·post1
// + RMSNorm(h2)·post2, RMSNorm(sum)·post, then the residual add of h. The
// package's existing test only reaches the input-guard branches, so the whole
// autorelease-pool encode (and the scratch pool it drives) sat dark. The
// reference runs the same public RMSNorm/Add primitives; the single-row vs rows
// RMSNorm kernels agree to a few bf16 ULPs, so parity is asserted within a tight
// tolerance rather than byte-for-byte.

func moePostCombineReference(t *testing.T, h, h1, h2, post1, post2, post []byte, dModel int, eps float32) []byte {
	t.Helper()
	must := func(b []byte, err error) []byte {
		if err != nil {
			t.Fatalf("moePostCombine reference op: %v", err)
		}
		return b
	}
	h1n := must(RMSNormBF16(h1, post1, 1, dModel, eps))
	h2n := must(RMSNormBF16(h2, post2, 1, dModel, eps))
	comb := must(AddBF16(h1n, h2n))
	ffr := must(RMSNormBF16(comb, post, 1, dModel, eps))
	return must(AddBF16(h, ffr))
}

func TestMoEBlockPostCombineBF16MatchesReference(t *testing.T) {
	requireNativeRuntime(t)
	const dModel = 64
	const eps = float32(1e-5)
	h := toBF16Bytes(syntheticFloat32(dModel, 29))
	h1 := toBF16Bytes(syntheticFloat32(dModel, 31))
	h2 := toBF16Bytes(syntheticFloat32(dModel, 37))
	post1 := toBF16Bytes(syntheticFloat32(dModel, 41))
	post2 := toBF16Bytes(syntheticFloat32(dModel, 43))
	post := toBF16Bytes(syntheticFloat32(dModel, 47))

	want := moePostCombineReference(t, h, h1, h2, post1, post2, post, dModel, eps)

	// Two invocations: the second pulls the scratch the first returned to the
	// pool, exercising the getMoEBlockPostCombineScratch reuse branch.
	for pass := range 2 {
		got, err := moeBlockPostCombineBF16(h, h1, h2, post1, bufView{}, post2, bufView{}, post, bufView{}, dModel, eps)
		if err != nil {
			t.Fatalf("pass %d: moeBlockPostCombineBF16: %v", pass, err)
		}
		if len(got) != dModel*bf16Size {
			t.Fatalf("pass %d: output bytes = %d, want %d", pass, len(got), dModel*bf16Size)
		}
		var maxDiff float64
		for i := 0; i+1 < len(want); i += bf16Size {
			g := float64(bf16ToF32(got[i], got[i+1]))
			w := float64(bf16ToF32(want[i], want[i+1]))
			if d := math.Abs(g - w); d > maxDiff {
				maxDiff = d
			}
		}
		if maxDiff > 0.02 {
			t.Fatalf("pass %d: post-combine vs reference maxDiff = %.5f (> 0.02)", pass, maxDiff)
		}
	}
}
