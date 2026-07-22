// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/inference/model"
)

// dumpKVRowsTestBytes builds one token's row bytes from vals — heads*headDim
// float32 values in [head0's headDim...][head1's headDim...] order, the
// token-major layout stateBlockLayerBytes returns per cache row. Every value
// used by these tests is a small integer, exact under a bf16 round trip.
func dumpKVRowsTestBytes(vals ...float32) []byte {
	return f32ToBf16Slice(vals)
}

// dumpKVRowsTestArch returns a one-layer arch whose single layer owns cache
// slot 0 — the minimum s.state.specs shape that keeps stateLayerViewsRefreshing
// on its cache-hit fast path once stateBlockViews is pre-populated by hand
// (assistant_load_test.go's ArchSession literal pattern; no Metal device
// needed since the fast path never calls snapshotCacheViews).
func dumpKVRowsTestArch(attn model.AttentionType) model.Arch {
	return model.Arch{Layer: []model.LayerSpec{{Attention: attn, CacheIndex: 0}}}
}

// TestArchSessionDumpKVRows_Good checks the head-major-then-token-major row
// order against a hand-built 2-head, 3-token fixture for both K and V.
func TestArchSessionDumpKVRows_Good(t *testing.T) {
	const heads, headDim, tokens = 2, 2, 3
	rowBytes := heads * headDim * bf16Size
	var key, value []byte
	for tok := 0; tok < tokens; tok++ {
		key = append(key, dumpKVRowsTestBytes(float32(4*tok+1), float32(4*tok+2), float32(4*tok+3), float32(4*tok+4))...)
		value = append(value, dumpKVRowsTestBytes(float32(100+4*tok+1), float32(100+4*tok+2), float32(100+4*tok+3), float32(100+4*tok+4))...)
	}
	arch := dumpKVRowsTestArch(model.GlobalAttention)
	session := &ArchSession{
		arch:  arch,
		pos:   tokens,
		state: archDecodeState{specs: arch.Layer},
		stateBlockViews: []sessionStateLayerView{
			{
				layer: 0, kvHeads: heads, headDim: headDim, rowBytes: rowBytes, cacheIndex: 0,
				cacheMode: nativeStateCacheModeFixed, cacheRows: tokens, keyBytes: key, valueBytes: value,
			},
		},
	}

	keys, values, err := session.DumpKVRows(0)
	if err != nil {
		t.Fatalf("DumpKVRows: %v", err)
	}
	if len(keys) != heads*tokens || len(values) != heads*tokens {
		t.Fatalf("DumpKVRows returned %d/%d rows, want %d (heads*tokens)", len(keys), len(values), heads*tokens)
	}
	wantKeys := [][2]float32{{1, 2}, {5, 6}, {9, 10}, {3, 4}, {7, 8}, {11, 12}}
	wantValues := [][2]float32{{101, 102}, {105, 106}, {109, 110}, {103, 104}, {107, 108}, {111, 112}}
	for i, want := range wantKeys {
		if keys[i][0] != want[0] || keys[i][1] != want[1] {
			t.Errorf("keys[%d] = %v, want %v", i, keys[i], want)
		}
	}
	for i, want := range wantValues {
		if values[i][0] != want[0] || values[i][1] != want[1] {
			t.Errorf("values[%d] = %v, want %v", i, values[i], want)
		}
	}
}

// TestArchSessionDumpKVRows_Bad checks the three input-error paths: a nil
// session, an empty (pos=0) cache, and a layer index that shares another
// layer's cache rather than owning one (gemma4's KV-shared tail layers).
func TestArchSessionDumpKVRows_Bad(t *testing.T) {
	t.Run("nil_session", func(t *testing.T) {
		var session *ArchSession
		if _, _, err := session.DumpKVRows(0); err == nil {
			t.Fatal("DumpKVRows accepted a nil session")
		}
	})
	t.Run("empty_cache", func(t *testing.T) {
		session := &ArchSession{}
		if _, _, err := session.DumpKVRows(0); err == nil {
			t.Fatal("DumpKVRows accepted pos=0 (empty cache)")
		}
	})
	t.Run("shared_layer", func(t *testing.T) {
		arch := model.Arch{Layer: []model.LayerSpec{
			{Attention: model.GlobalAttention, CacheIndex: 0},
			{Attention: model.GlobalAttention, CacheIndex: -1, KVShareFrom: 0}, // shares layer 0's cache
		}}
		session := &ArchSession{
			arch:  arch,
			pos:   1,
			state: archDecodeState{specs: arch.Layer},
			stateBlockViews: []sessionStateLayerView{
				{
					layer: 0, kvHeads: 1, headDim: 2, rowBytes: 2 * bf16Size, cacheIndex: 0,
					cacheMode: nativeStateCacheModeFixed, cacheRows: 1,
					keyBytes: dumpKVRowsTestBytes(1, 2), valueBytes: dumpKVRowsTestBytes(3, 4),
				},
			},
		}
		if _, _, err := session.DumpKVRows(1); err == nil {
			t.Fatal("DumpKVRows accepted a shared (non-owning) layer index")
		}
	})
}

// TestArchSessionDumpKVRows_Ugly checks a wrapped SLIDING cache: pos=5 over a
// 2-row physical window means logical tokens 3,4 are resident, physically
// stored at slot (token%cacheRows) — slot0 holds token4, slot1 holds token3.
// DumpKVRows must return them in ASCENDING logical order (token3 then
// token4), proving it reuses stateBlockLayerBytes' de-wrap rather than
// reading cache bytes in raw physical order.
func TestArchSessionDumpKVRows_Ugly(t *testing.T) {
	const headDim = 2
	rowBytes := headDim * bf16Size
	var physical []byte
	physical = append(physical, dumpKVRowsTestBytes(7, 8)...) // slot0 = token4 (newest)
	physical = append(physical, dumpKVRowsTestBytes(5, 6)...) // slot1 = token3 (oldest resident)
	arch := dumpKVRowsTestArch(model.SlidingAttention)
	session := &ArchSession{
		arch:  arch,
		pos:   5,
		state: archDecodeState{specs: arch.Layer},
		stateBlockViews: []sessionStateLayerView{
			{
				layer: 0, kvHeads: 1, headDim: headDim, rowBytes: rowBytes, cacheIndex: 0,
				cacheMode: nativeStateCacheModeFixed, maxSize: 2, cacheRows: 2,
				keyBytes: physical, valueBytes: append([]byte(nil), physical...),
			},
		},
	}

	keys, _, err := session.DumpKVRows(0)
	if err != nil {
		t.Fatalf("DumpKVRows: %v", err)
	}
	if len(keys) != 2 {
		t.Fatalf("DumpKVRows returned %d rows, want 2 (the sliding window)", len(keys))
	}
	if keys[0][0] != 5 || keys[0][1] != 6 {
		t.Errorf("keys[0] = %v, want [5 6] (token 3, oldest in the window)", keys[0])
	}
	if keys[1][0] != 7 || keys[1][1] != 8 {
		t.Errorf("keys[1] = %v, want [7 8] (token 4, newest)", keys[1])
	}
}
