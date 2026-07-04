// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/inference/model"
)

// BenchmarkVerifyBatchedVsSequential measures the MTP batched verify against the sequential path it
// replaces: K query tokens through the resident stack in ONE command buffer (stepTokensBatchedDense)
// vs K separate stepToken calls = K command-buffer submits. Same kernels, byte-identical output (see
// TestStepTokensBatchedDense) — this isolates the submit/sync overhead the batch removes. AX-11:
// synthetic weights, no model load.
func BenchmarkVerifyBatchedVsSequential(b *testing.B) {
	requireNativeRuntime(b)
	const dModel, nHeads, nKV, headDim, dFF = 512, 8, 4, 64, 1024
	const nL, maxLen, prefix, K = 6, 64, 8, 4
	base, scale, eps := float32(10000), float32(0.125), float32(1e-5)

	layers := make([]DecodeLayerWeights, nL)
	types := make([]string, nL)
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
		types[li] = "full_attention"
	}
	specs := model.DeriveLayers(types, 0)
	emb := func(seed int) []byte {
		f := make([]float32, dModel)
		for j := range f {
			f[j] = float32((j*(seed+3)+5)%97-48) * 0.02
		}
		return toBF16Bytes(f)
	}
	embs := make([][]byte, prefix+K)
	for i := range embs {
		embs[i] = emb(i + 1)
	}
	build := func() *archDecodeState {
		lb, moe, err := buildBF16ArchLayerBufs(layers, specs, dModel, nHeads, nKV, headDim, dFF, maxLen, 0, nil)
		if err != nil {
			b.Fatalf("buildBF16ArchLayerBufs: %v", err)
		}
		st := newArchDecodeState(specs, lb, moe, dModel, nHeads, nKV, headDim, dFF, 0, headDim, headDim, base, base, scale, eps, false, 0)
		return &st
	}

	b.Run("sequential-Kx-stepToken", func(b *testing.B) {
		for n := 0; n < b.N; n++ {
			withAutoreleasePool(func() {
				st := build()
				for i := 0; i < prefix; i++ {
					if _, err := st.stepToken(embs[i], i); err != nil {
						b.Fatal(err)
					}
				}
				for i := 0; i < K; i++ {
					if _, err := st.stepToken(embs[prefix+i], prefix+i); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	})

	b.Run("batched-1x-stepTokensBatchedDense", func(b *testing.B) {
		for n := 0; n < b.N; n++ {
			withAutoreleasePool(func() {
				st := build()
				for i := 0; i < prefix; i++ {
					if _, err := st.stepToken(embs[i], i); err != nil {
						b.Fatal(err)
					}
				}
				if _, ok, err := st.stepTokensBatchedDense(embs[prefix:prefix+K], prefix); err != nil || !ok {
					b.Fatalf("stepTokensBatchedDense ok=%v err=%v", ok, err)
				}
			})
		}
	})
}
