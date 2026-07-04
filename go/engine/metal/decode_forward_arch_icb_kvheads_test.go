// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// TestDecodeForwardArchICBQuantPerLayerKVHeads is the FAST synthetic reproduction of the 12B/31B
// non-uniform-kvHeads ICB divergence (the real-model TestRealModelICBvsReencodeParity catches 14/24
// token diffs — a recorder-vs-stepToken cache-stride mismatch). The existing ICB forward parity cases
// all use UNIFORM kvHeads; this is the missing case: a sliding layer (GQA) + a global layer (MQA, fewer
// kv heads), the geometry that gates 12B/31B to the slow re-encode path. DecodeForwardArchICBQuant must
// equal DecodeForwardArchQuant (the correct re-encode oracle) byte-for-byte; a divergence here pins the
// bug on a fixture that builds in milliseconds instead of an 18GB model load.
func TestDecodeForwardArchICBQuantPerLayerKVHeads(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, headDim, globalHeadDim, dFF, gs, bits = 512, 8, 64, 128, 1024, 64, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	const maxLen = 8
	const slidingKV, globalKV = 2, 1 // sliding GQA (ratio 4) + global MQA (ratio 8) — the 12B/31B mix

	mkInputs := func(n int) [][]byte {
		in := make([][]byte, n)
		for i := range in {
			f := make([]float32, dModel)
			for j := range f {
				f[j] = float32((j*(i+3)+5)%97-48) * 0.02
			}
			in[i] = toBF16Bytes(f)
		}
		return in
	}

	// specs: layer 0 sliding (GQA kv=2, headDim=64), layer 1 global/full (MQA kv=1, headDim=128) —
	// the real 12B/31B geometry: the global layer varies BOTH kvHeads AND head dim from the sliding base.
	specs := model.DeriveLayers([]string{"sliding_attention", "full_attention"}, 0)
	specs[0].KVHeads, specs[0].HeadDim = slidingKV, headDim
	specs[1].KVHeads, specs[1].HeadDim = globalKV, globalHeadDim
	// weights sized to each layer's own (kvHeads, headDim): Q/O at nHeads·hd, K/V at kvHeads·hd.
	ql := []QuantizedLayerWeights{
		buildQuantLayer(t, dModel, nHeads, slidingKV, headDim, dFF, gs, bits, 100),
		buildQuantLayer(t, dModel, nHeads, globalKV, globalHeadDim, dFF, gs, bits, 200),
	}

	const T, slidingWindow = 6, 3
	inputs := mkInputs(T)
	// the sliding kvHeads is the default; the global layer's spec overrides it to globalKV.
	got, err := DecodeForwardArchICBQuant(inputs, ql, specs, dModel, nHeads, slidingKV, headDim, maxLen, dFF, slidingWindow, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArchICBQuant: %v", err)
	}
	want, err := DecodeForwardArchQuant(inputs, ql, specs, dModel, nHeads, slidingKV, headDim, maxLen, dFF, slidingWindow, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArchQuant: %v", err)
	}
	for tok := 0; tok < T; tok++ {
		eqBytes(t, core.Sprintf("per-layer-kvHeads tok%d", tok), got[tok], want[tok])
	}
	t.Logf("non-uniform kvHeads (sliding GQA kv=%d / global MQA kv=%d): ICB replay ≡ DecodeForwardArchQuant byte-for-byte — the 12B/31B mix records correctly", slidingKV, globalKV)
}
