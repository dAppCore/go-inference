// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// ExampleArchSession_DumpKVRows shows the returned shape: one row per
// (attention head, cached token), headDim wide. Unlike most of this
// package's examples it needs no Metal device — DumpKVRows' geometry read
// stays on stateLayerViewsRefreshing's cache-hit fast path when
// stateBlockViews is already populated, so this fixture is plain host bytes.
func ExampleArchSession_DumpKVRows() {
	const heads, headDim, tokens = 1, 3, 2
	rowBytes := heads * headDim * bf16Size
	arch := model.Arch{Layer: []model.LayerSpec{{Attention: model.GlobalAttention, CacheIndex: 0}}}
	session := &ArchSession{
		arch:  arch,
		pos:   tokens,
		state: archDecodeState{specs: arch.Layer},
		stateBlockViews: []sessionStateLayerView{
			{
				layer: 0, kvHeads: heads, headDim: headDim, rowBytes: rowBytes, cacheIndex: 0,
				cacheMode: nativeStateCacheModeFixed, cacheRows: tokens,
				keyBytes:   f32ToBf16Slice([]float32{1, 2, 3, 4, 5, 6}),
				valueBytes: f32ToBf16Slice([]float32{7, 8, 9, 10, 11, 12}),
			},
		},
	}

	keys, values, err := session.DumpKVRows(0)
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("rows:", len(keys), "d:", len(keys[0]))
	core.Println("keys[0]:", keys[0])
	core.Println("keys[1]:", keys[1])
	core.Println("values[0]:", values[0])
	// Output:
	// rows: 2 d: 3
	// keys[0]: [1 2 3]
	// keys[1]: [4 5 6]
	// values[0]: [7 8 9]
}
