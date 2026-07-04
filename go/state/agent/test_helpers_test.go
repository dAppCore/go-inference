// SPDX-Licence-Identifier: EUPL-1.2

package agent

import "dappco.re/go/inference/kv"

func kvSnapshotBlocksTestSnapshot() *kv.Snapshot {
	return &kv.Snapshot{
		Version:       kv.SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1, 2, 3, 4},
		Generated:     []int32{4},
		TokenOffset:   4,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        4,
		HeadDim:       2,
		NumQueryHeads: 1,
		LogitShape:    []int32{1, 1, 3},
		Logits:        []float32{0.1, 0.2, 0.7},
		Layers: []kv.LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []kv.HeadSnapshot{{
				Key:   []float32{10, 11, 12, 13, 14, 15, 16, 17},
				Value: []float32{20, 21, 22, 23, 24, 25, 26, 27},
			}},
		}},
	}
}
