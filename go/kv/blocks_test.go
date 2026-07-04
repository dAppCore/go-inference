// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"context"
	stdio "io"
	"math"
	"testing"

	core "dappco.re/go"
	state "dappco.re/go/inference/state"
	filestore "dappco.re/go/inference/state/filestore"
)

func TestKVSnapshotBlocks_Good_SplitAndAssemble(t *testing.T) {
	snapshot := kvSnapshotBlocksTestSnapshot()

	blocks, err := snapshot.SplitBlocks(2)
	if err != nil {
		t.Fatalf("SplitBlocks() error = %v", err)
	}
	if len(blocks) != 2 {
		t.Fatalf("blocks len = %d, want 2", len(blocks))
	}
	if blocks[0].Index != 0 || blocks[0].TokenStart != 0 || blocks[0].TokenCount != 2 {
		t.Fatalf("block[0] metadata = %+v", blocks[0])
	}
	if got := blocks[0].Snapshot.Tokens; len(got) != 2 || got[0] != 1 || got[1] != 2 {
		t.Fatalf("block[0] tokens = %v, want [1 2]", got)
	}
	if got := blocks[0].Snapshot.Layers[0].Heads[0].Key; len(got) != 4 || got[0] != 10 || got[3] != 13 {
		t.Fatalf("block[0] key = %v, want first token range", got)
	}
	if len(blocks[0].Snapshot.Logits) != 0 {
		t.Fatalf("block[0] logits = %v, want logits only on final block", blocks[0].Snapshot.Logits)
	}
	if got := blocks[1].Snapshot.Layers[0].Heads[0].Value; len(got) != 4 || got[0] != 24 || got[3] != 27 {
		t.Fatalf("block[1] value = %v, want second token range", got)
	}

	assembled, err := AssembleBlocks(blocks)
	if err != nil {
		t.Fatalf("AssembleBlocks() error = %v", err)
	}
	if assembled.SeqLen != snapshot.SeqLen || assembled.TokenOffset != snapshot.TokenOffset {
		t.Fatalf("assembled seq/offset = %d/%d, want %d/%d", assembled.SeqLen, assembled.TokenOffset, snapshot.SeqLen, snapshot.TokenOffset)
	}
	if len(assembled.Tokens) != 4 || assembled.Tokens[0] != 1 || assembled.Tokens[3] != 4 {
		t.Fatalf("assembled tokens = %v, want original tokens", assembled.Tokens)
	}
	head, ok := assembled.Head(0, 0)
	if !ok {
		t.Fatal("assembled Head(0,0) ok = false")
	}
	if len(head.Key) != 8 || head.Key[0] != 10 || head.Key[7] != 17 || head.Value[0] != 20 || head.Value[7] != 27 {
		t.Fatalf("assembled head = %+v, want original key/value", head)
	}
	if len(assembled.Logits) != 3 || assembled.Logits[2] != 0.7 {
		t.Fatalf("assembled logits = %v, want final logits", assembled.Logits)
	}
}

func TestKVSnapshotBlocks_Good_TurboQuantPayloadsStayWhole(t *testing.T) {
	snapshot := kvSnapshotBlocksTestSnapshot()
	snapshot.Layers[0].CacheMode = "turboquant"
	snapshot.Layers[0].TurboQuantPayloads = [][]byte{
		[]byte(`{"layout":{"page_tokens":2},"data":"first"}`),
		[]byte(`{"layout":{"page_tokens":2},"data":"second"}`),
	}
	snapshot.Layers[0].Heads = nil

	blocks, err := snapshot.SplitBlocks(2)
	if err != nil {
		t.Fatalf("SplitBlocks(turboquant) error = %v", err)
	}
	if len(blocks) != 1 || blocks[0].TokenStart != 0 || blocks[0].TokenCount != len(snapshot.Tokens) {
		t.Fatalf("blocks = %+v, want one whole compressed block", blocks)
	}
	if got := blocks[0].Snapshot.Layers[0].TurboQuantPayloads; len(got) != 2 || string(got[1]) != string(snapshot.Layers[0].TurboQuantPayloads[1]) {
		t.Fatalf("block payloads = %q, want original compressed payloads", got)
	}
	assembled, err := AssembleBlocks(blocks)
	if err != nil {
		t.Fatalf("AssembleBlocks(turboquant) error = %v", err)
	}
	if assembled.Layers[0].CacheMode != "turboquant" || len(assembled.Layers[0].TurboQuantPayloads) != 2 {
		t.Fatalf("assembled compressed layer = mode:%q payloads:%d, want turboquant/2", assembled.Layers[0].CacheMode, len(assembled.Layers[0].TurboQuantPayloads))
	}

	store := state.NewInMemoryStore(nil)
	bundle, err := snapshot.SaveStateBlocks(context.Background(), store, StateBlockOptions{BlockSize: 2})
	if err != nil {
		t.Fatalf("SaveStateBlocks(turboquant) error = %v", err)
	}
	if len(bundle.Blocks) != 1 {
		t.Fatalf("state blocks = %d, want one whole compressed block", len(bundle.Blocks))
	}
	loaded, err := LoadFromStateBlocks(context.Background(), store, bundle)
	if err != nil {
		t.Fatalf("LoadFromStateBlocks(turboquant) error = %v", err)
	}
	if loaded.Layers[0].CacheMode != "turboquant" || len(loaded.Layers[0].TurboQuantPayloads) != 2 {
		t.Fatalf("loaded compressed layer = mode:%q payloads:%d, want turboquant/2", loaded.Layers[0].CacheMode, len(loaded.Layers[0].TurboQuantPayloads))
	}
	if string(loaded.Layers[0].TurboQuantPayloads[0]) != string(snapshot.Layers[0].TurboQuantPayloads[0]) {
		t.Fatalf("loaded first payload = %q, want %q", loaded.Layers[0].TurboQuantPayloads[0], snapshot.Layers[0].TurboQuantPayloads[0])
	}
}

func TestKVSnapshotBlocks_Good_RangeBlocksStopsEarly(t *testing.T) {
	snapshot := kvSnapshotBlocksTestSnapshot()
	seen := []int{}

	err := snapshot.RangeBlocks(1, func(block Block) bool {
		seen = append(seen, block.Index)
		return len(seen) < 2
	})

	if err != nil {
		t.Fatalf("RangeBlocks() error = %v", err)
	}
	if len(seen) != 2 || seen[0] != 0 || seen[1] != 1 {
		t.Fatalf("seen blocks = %v, want [0 1]", seen)
	}
}

func TestKVSnapshotBlocks_Good_SplitsMixedHeadDims(t *testing.T) {
	snapshot := kvSnapshotBlocksTestSnapshot()
	snapshot.Layers[0].Heads[0].Key = []float32{
		10, 11, 12,
		13, 14, 15,
		16, 17, 18,
		19, 20, 21,
	}
	snapshot.Layers[0].Heads[0].Value = []float32{
		30,
		31,
		32,
		33,
	}

	blocks, err := snapshot.SplitBlocks(2)
	if err != nil {
		t.Fatalf("SplitBlocks() error = %v", err)
	}
	if got := blocks[0].Snapshot.Layers[0].Heads[0].Key; len(got) != 6 || got[0] != 10 || got[5] != 15 {
		t.Fatalf("block[0] mixed key = %v, want first two 3-wide tokens", got)
	}
	if got := blocks[1].Snapshot.Layers[0].Heads[0].Value; len(got) != 2 || got[0] != 32 || got[1] != 33 {
		t.Fatalf("block[1] mixed value = %v, want final two 1-wide tokens", got)
	}
}

func TestKVSnapshotBlocks_Good_SplitsLayerSuffixWindows(t *testing.T) {
	snapshot := kvSnapshotBlocksTestSnapshot()
	snapshot.Tokens = []int32{1, 2, 3, 4, 5}
	snapshot.TokenOffset = 5
	snapshot.SeqLen = 5
	snapshot.Layers[0].Heads[0].Key = []float32{10, 11, 12, 13, 14, 15, 16, 17, 18, 19}
	snapshot.Layers[0].Heads[0].Value = []float32{20, 21, 22, 23, 24, 25, 26, 27, 28, 29}
	snapshot.NumLayers = 2
	snapshot.Layers = append(snapshot.Layers, LayerSnapshot{
		Layer:      1,
		CacheIndex: 1,
		Heads: []HeadSnapshot{{
			Key:   []float32{100, 101, 102, 103},
			Value: []float32{200, 201, 202, 203},
		}},
	})

	blocks, err := snapshot.SplitBlocks(2)
	if err != nil {
		t.Fatalf("SplitBlocks() error = %v", err)
	}
	if len(blocks[0].Snapshot.Layers[1].Heads) != 0 {
		t.Fatalf("block[0] layer 1 heads = %d, want omitted before suffix window", len(blocks[0].Snapshot.Layers[1].Heads))
	}
	last := blocks[len(blocks)-1]
	if got := last.Snapshot.Layers[1].Heads[0].Key; len(got) != 2 || got[0] != 102 || got[1] != 103 {
		t.Fatalf("last block suffix key = %v, want final suffix token", got)
	}

	assembled, err := AssembleBlocks(blocks)
	if err != nil {
		t.Fatalf("AssembleBlocks() error = %v", err)
	}
	if assembled.SeqLen != 5 || len(assembled.Tokens) != 5 {
		t.Fatalf("assembled metadata = %+v, want global sequence retained", assembled)
	}
	head, ok := assembled.Head(1, 0)
	if !ok {
		t.Fatal("assembled Head(1,0) ok = false")
	}
	if len(head.Key) != 4 || head.Key[0] != 100 || head.Value[3] != 203 {
		t.Fatalf("assembled suffix head = %+v, want retained local cache", head)
	}
}

func TestKVSnapshotBlocks_Good_SplitAndAssembleNativeDType(t *testing.T) {
	snapshot := kvSnapshotBlocksTestSnapshot()
	head := &snapshot.Layers[0].Heads[0]
	head.KeyDType = "float16"
	head.ValueDType = "bfloat16"
	for _, value := range head.Key {
		head.KeyBytes = appendUint16LE(head.KeyBytes, float32ToFloat16(value))
	}
	for _, value := range head.Value {
		head.ValueBytes = appendUint16LE(head.ValueBytes, uint16(math.Float32bits(value)>>16))
	}

	blocks, err := snapshot.SplitBlocks(2)
	if err != nil {
		t.Fatalf("SplitBlocks() error = %v", err)
	}

	if got := len(blocks[0].Snapshot.Layers[0].Heads[0].KeyBytes); got != 8 {
		t.Fatalf("block[0] key bytes = %d, want two tokens x dim two x f16", got)
	}
	if blocks[0].Snapshot.Layers[0].Heads[0].KeyDType != "float16" {
		t.Fatalf("block[0] key dtype = %q, want float16", blocks[0].Snapshot.Layers[0].Heads[0].KeyDType)
	}
	assembled, err := AssembleBlocks(blocks)
	if err != nil {
		t.Fatalf("AssembleBlocks() error = %v", err)
	}
	assembledHead := assembled.Layers[0].Heads[0]
	if !equalBytes(assembledHead.KeyBytes, head.KeyBytes) || !equalBytes(assembledHead.ValueBytes, head.ValueBytes) {
		t.Fatalf("assembled native bytes = %d/%d, want original %d/%d", len(assembledHead.KeyBytes), len(assembledHead.ValueBytes), len(head.KeyBytes), len(head.ValueBytes))
	}
}

func TestKVSnapshotBlocks_Bad_RejectsInvalidHeadShape(t *testing.T) {
	snapshot := kvSnapshotBlocksTestSnapshot()
	snapshot.Layers[0].Heads[0].Key = snapshot.Layers[0].Heads[0].Key[:7]

	_, err := snapshot.SplitBlocks(2)

	if err == nil {
		t.Fatal("SplitBlocks() error = nil, want invalid head shape error")
	}
}

func TestKVSnapshotStateBlocks_Good_SaveLoadRoundTrip(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	snapshot := kvSnapshotBlocksTestSnapshot()

	bundle, err := snapshot.SaveStateBlocks(context.Background(), store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingQ8,
		URI:        "mlx://session/blocks",
		Labels:     []string{"session-kv-block"},
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks() error = %v", err)
	}
	if bundle.Kind != StateBlockBundleKind || len(bundle.Blocks) != 2 || bundle.BlockSize != 2 {
		t.Fatalf("bundle = %+v, want two State KV blocks", bundle)
	}
	if bundle.Blocks[0].State.ChunkID == bundle.Blocks[1].State.ChunkID {
		t.Fatalf("block refs = %+v, want distinct State chunks", bundle.Blocks)
	}
	if bundle.Blocks[0].PayloadEncoding != kvSnapshotStatePayloadRaw || bundle.Blocks[0].PayloadByteCount == 0 {
		t.Fatalf("block payload metadata = %+v, want raw binary payload", bundle.Blocks[0])
	}
	chunk, err := state.ResolveBytes(context.Background(), store, bundle.Blocks[0].State.ChunkID)
	if err != nil {
		t.Fatalf("ResolveBytes(block chunk) error = %v", err)
	}
	if len(chunk.Data) != bundle.Blocks[0].PayloadByteCount || core.Contains(chunk.Text, `"block_index":0`) {
		t.Fatalf("block chunk = text %q data %d, want raw binary payload", chunk.Text, len(chunk.Data))
	}

	loaded, err := LoadFromStateBlocks(context.Background(), store, bundle)
	if err != nil {
		t.Fatalf("LoadFromStateBlocks() error = %v", err)
	}
	if loaded.TokenOffset != snapshot.TokenOffset || len(loaded.Tokens) != len(snapshot.Tokens) {
		t.Fatalf("loaded metadata = %+v, want original token state", loaded)
	}
	head, ok := loaded.Head(0, 0)
	if !ok {
		t.Fatal("loaded Head(0,0) ok = false")
	}
	if len(head.Key) != 8 || head.Key[0] < 9.99 || head.Key[7] < 16.99 || head.Value[7] < 26.99 {
		t.Fatalf("loaded head = %+v, want original q8-ish values", head)
	}
}

func TestKVSnapshotStateBlocks_Good_TextStoreUsesEnvelopeFallback(t *testing.T) {
	store := &textOnlyStateStore{store: state.NewInMemoryStore(nil)}
	snapshot := kvSnapshotBlocksTestSnapshot()

	bundle, err := snapshot.SaveStateBlocks(context.Background(), store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingQ8,
		URI:        "mlx://session/text-blocks",
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks(text store) error = %v", err)
	}
	if bundle.Blocks[0].PayloadEncoding != kvSnapshotStatePayloadJSONBase64 {
		t.Fatalf("payload encoding = %q, want JSON/base64 fallback", bundle.Blocks[0].PayloadEncoding)
	}
	chunk, err := state.Resolve(context.Background(), store, bundle.Blocks[0].State.ChunkID)
	if err != nil {
		t.Fatalf("Resolve(block chunk) error = %v", err)
	}
	if !core.Contains(chunk.Text, `"kind":"`+KVSnapshotStateBlockKind+`"`) || !core.Contains(chunk.Text, `"block_index":0`) {
		t.Fatalf("block chunk = %s, want block envelope", chunk.Text)
	}
	loaded, err := LoadFromStateBlocks(context.Background(), store, bundle)
	if err != nil {
		t.Fatalf("LoadFromStateBlocks(text store) error = %v", err)
	}
	if loaded.TokenOffset != snapshot.TokenOffset || len(loaded.Tokens) != len(snapshot.Tokens) {
		t.Fatalf("loaded metadata = %+v, want original token state", loaded)
	}
}

func TestKVSnapshotStateBlocks_Good_SaveNativeRawOnlyWithoutFloat32(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	snapshot := kvSnapshotBlocksTestSnapshot()
	head := &snapshot.Layers[0].Heads[0]
	for _, value := range head.Key {
		head.KeyBytes = appendUint16LE(head.KeyBytes, float32ToFloat16(value))
	}
	for _, value := range head.Value {
		head.ValueBytes = appendUint16LE(head.ValueBytes, uint16(math.Float32bits(value)>>16))
	}
	head.Key = nil
	head.Value = nil
	head.KeyDType = "float16"
	head.ValueDType = "bfloat16"

	blocks, err := snapshot.SplitBlocks(2)
	if err != nil {
		t.Fatalf("SplitBlocks(native raw-only) error = %v", err)
	}
	if len(blocks) != 2 || blocks[0].Hash == "" {
		t.Fatalf("raw-only split blocks = %+v, want hashed streamed blocks", blocks)
	}

	bundle, err := snapshot.SaveStateBlocks(context.Background(), store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingNative,
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks(native raw-only) error = %v", err)
	}
	loaded, err := LoadFromStateBlocksWithOptions(context.Background(), store, bundle, LoadOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("LoadFromStateBlocksWithOptions(raw-only) error = %v", err)
	}
	loadedHead := loaded.Layers[0].Heads[0]
	if len(loadedHead.Key) != 0 || len(loadedHead.Value) != 0 {
		t.Fatalf("loaded float32 key/value lengths = %d/%d, want raw-only", len(loadedHead.Key), len(loadedHead.Value))
	}
	if loadedHead.KeyDType != "float16" || loadedHead.ValueDType != "bfloat16" {
		t.Fatalf("loaded dtypes = %q/%q, want float16/bfloat16", loadedHead.KeyDType, loadedHead.ValueDType)
	}
	if len(loadedHead.KeyBytes) != 16 || len(loadedHead.ValueBytes) != 16 {
		t.Fatalf("loaded raw bytes = %d/%d, want four tokens x dim two x two bytes", len(loadedHead.KeyBytes), len(loadedHead.ValueBytes))
	}
}

func TestKVSnapshotStateBlocks_Good_SaveNativeLayerRawOnlyWithoutHeadDuplication(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	keyBytes := []byte{
		1, 0, 2, 0, 3, 0, 4, 0,
		5, 0, 6, 0, 7, 0, 8, 0,
	}
	valueBytes := []byte{
		11, 0, 12, 0, 13, 0, 14, 0,
		15, 0, 16, 0, 17, 0, 18, 0,
	}
	snapshot := &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1, 2, 3, 4},
		TokenOffset:   4,
		NumLayers:     1,
		NumHeads:      2,
		SeqLen:        4,
		HeadDim:       1,
		NumQueryHeads: 2,
		Layers: []LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			KeyDType:   "float16",
			KeyBytes:   keyBytes,
			KeyShape:   []int32{1, 2, 4, 1},
			ValueDType: "float16",
			ValueBytes: valueBytes,
			ValueShape: []int32{1, 2, 4, 1},
			Heads:      make([]HeadSnapshot, 2),
		}},
	}

	blocks, err := snapshot.SplitBlocks(2)
	if err != nil {
		t.Fatalf("SplitBlocks(native layer raw-only) error = %v", err)
	}
	if got := blocks[0].Snapshot.Layers[0].KeyBytes; !equalBytes(got, []byte{1, 0, 2, 0, 5, 0, 6, 0}) {
		t.Fatalf("block[0] layer key bytes = %v, want first two tokens for both heads", got)
	}
	bundle, err := snapshot.SaveStateBlocks(context.Background(), store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingNative,
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks(native layer raw-only) error = %v", err)
	}
	loaded, err := LoadFromStateBlocksWithOptions(context.Background(), store, bundle, LoadOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("LoadFromStateBlocksWithOptions(native layer raw-only) error = %v", err)
	}
	layer := loaded.Layers[0]
	if !equalBytes(layer.KeyBytes, keyBytes) || !equalBytes(layer.ValueBytes, valueBytes) {
		t.Fatalf("assembled layer bytes = %v/%v, want original slabs", layer.KeyBytes, layer.ValueBytes)
	}
	if len(layer.Heads) != 2 || len(layer.Heads[0].KeyBytes) != 0 {
		t.Fatalf("assembled heads = %+v, want no duplicated per-head bytes", layer.Heads)
	}
}

func TestKVSnapshotStateBlocks_Good_NativeLayerRawPayloadBytesAreState(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	keyBytes := []byte{
		1, 0, 2, 0, 3, 0, 4, 0,
		5, 0, 6, 0, 7, 0, 8, 0,
	}
	valueBytes := []byte{
		11, 0, 12, 0, 13, 0, 14, 0,
		15, 0, 16, 0, 17, 0, 18, 0,
	}
	snapshot := &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1, 2, 3, 4},
		TokenOffset:   4,
		NumLayers:     1,
		NumHeads:      2,
		SeqLen:        4,
		HeadDim:       1,
		NumQueryHeads: 2,
		Layers: []LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			KeyDType:   "float16",
			KeyBytes:   keyBytes,
			KeyShape:   []int32{1, 2, 4, 1},
			ValueDType: "float16",
			ValueBytes: valueBytes,
			ValueShape: []int32{1, 2, 4, 1},
			Heads:      make([]HeadSnapshot, 2),
		}},
	}
	wantBlocks, err := snapshot.SplitBlocks(2)
	if err != nil {
		t.Fatalf("SplitBlocks(native payload contract) error = %v", err)
	}
	bundle, err := snapshot.SaveStateBlocks(context.Background(), store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingNative,
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks(native payload contract) error = %v", err)
	}
	if len(bundle.Blocks) != len(wantBlocks) {
		t.Fatalf("saved blocks = %d, want %d", len(bundle.Blocks), len(wantBlocks))
	}
	for i, wantBlock := range wantBlocks {
		wantPayload, err := wantBlock.Snapshot.bytesWithOptions(SaveOptions{KVEncoding: EncodingNative})
		if err != nil {
			t.Fatalf("bytesWithOptions(block %d) error = %v", i, err)
		}
		ref := bundle.Blocks[i]
		if ref.PayloadEncoding != kvSnapshotStatePayloadRaw {
			t.Fatalf("block %d payload encoding = %q, want raw bytes", i, ref.PayloadEncoding)
		}
		if ref.PayloadByteCount != len(wantPayload) {
			t.Fatalf("block %d payload bytes = %d, want exact native block bytes %d", i, ref.PayloadByteCount, len(wantPayload))
		}
		chunk, err := state.ResolveBytes(context.Background(), store, ref.State.ChunkID)
		if err != nil {
			t.Fatalf("ResolveBytes(block %d) error = %v", i, err)
		}
		if !equalBytes(chunk.Data, wantPayload) {
			t.Fatalf("block %d raw payload diverged from native block bytes", i)
		}
	}
	loaded, err := LoadFromStateBlocksWithOptions(context.Background(), store, bundle, LoadOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("LoadFromStateBlocksWithOptions(native payload contract) error = %v", err)
	}
	layer := loaded.Layers[0]
	if !equalBytes(layer.KeyBytes, keyBytes) || !equalBytes(layer.ValueBytes, valueBytes) {
		t.Fatalf("loaded native slabs = %v/%v, want original State bytes", layer.KeyBytes, layer.ValueBytes)
	}
	if len(layer.Heads) != 2 || len(layer.Heads[0].KeyBytes) != 0 || len(layer.Heads[0].Key) != 0 {
		t.Fatalf("loaded heads = %+v, want native slabs without duplicated head payload", layer.Heads)
	}
}

func TestKVSnapshotStateBlocks_Good_SaveNativeLayerSingleHeadRawOnly(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	keyBytes := []byte{1, 0, 2, 0, 3, 0, 4, 0}
	valueBytes := []byte{11, 0, 12, 0, 13, 0, 14, 0}
	snapshot := &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1, 2, 3, 4},
		TokenOffset:   4,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        4,
		HeadDim:       1,
		NumQueryHeads: 1,
		Layers: []LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			KeyDType:   "float16",
			KeyBytes:   keyBytes,
			KeyShape:   []int32{1, 1, 4, 1},
			ValueDType: "float16",
			ValueBytes: valueBytes,
			ValueShape: []int32{1, 1, 4, 1},
			Heads:      make([]HeadSnapshot, 1),
		}},
	}

	bundle, err := snapshot.SaveStateBlocks(context.Background(), store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingNative,
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks(native single-head layer raw-only) error = %v", err)
	}
	loaded, err := LoadFromStateBlocksWithOptions(context.Background(), store, bundle, LoadOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("LoadFromStateBlocksWithOptions(native single-head layer raw-only) error = %v", err)
	}
	layer := loaded.Layers[0]
	if !equalBytes(layer.KeyBytes, keyBytes) || !equalBytes(layer.ValueBytes, valueBytes) {
		t.Fatalf("assembled single-head layer bytes = %v/%v, want original slabs", layer.KeyBytes, layer.ValueBytes)
	}
	if len(layer.Heads) != 1 || len(layer.Heads[0].KeyBytes) != 0 {
		t.Fatalf("assembled heads = %+v, want no duplicated per-head bytes", layer.Heads)
	}
}

func TestKVSnapshotStateBlocks_Good_SaveNativeLayerTokenMajorRawOnly(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	keyBytes := []byte{
		1, 0, 2, 0,
		3, 0, 4, 0,
		5, 0, 6, 0,
		7, 0, 8, 0,
	}
	valueBytes := []byte{
		11, 0, 12, 0,
		13, 0, 14, 0,
		15, 0, 16, 0,
		17, 0, 18, 0,
	}
	snapshot := &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1, 2, 3, 4},
		TokenOffset:   4,
		NumLayers:     1,
		NumHeads:      2,
		SeqLen:        4,
		HeadDim:       1,
		NumQueryHeads: 2,
		Layers: []LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			KeyDType:   "bfloat16",
			KeyBytes:   keyBytes,
			KeyShape:   []int32{4, 2, 1},
			ValueDType: "bfloat16",
			ValueBytes: valueBytes,
			ValueShape: []int32{4, 2, 1},
			Heads:      make([]HeadSnapshot, 2),
		}},
	}

	bundle, err := snapshot.SaveStateBlocks(context.Background(), store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingNative,
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks(native token-major layer raw-only) error = %v", err)
	}
	loaded, err := LoadFromStateBlocksWithOptions(context.Background(), store, bundle, LoadOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("LoadFromStateBlocksWithOptions(native token-major layer raw-only) error = %v", err)
	}
	layer := loaded.Layers[0]
	if !equalBytes(layer.KeyBytes, keyBytes) || !equalBytes(layer.ValueBytes, valueBytes) {
		t.Fatalf("assembled token-major layer bytes = %v/%v, want original slabs", layer.KeyBytes, layer.ValueBytes)
	}
	if len(layer.KeyShape) != 3 || layer.KeyShape[0] != 4 || layer.KeyShape[1] != 2 || layer.KeyShape[2] != 1 {
		t.Fatalf("assembled token-major key shape = %v, want [4 2 1]", layer.KeyShape)
	}
	if len(layer.Heads) != 2 || len(layer.Heads[0].KeyBytes) != 0 {
		t.Fatalf("assembled token-major heads = %+v, want no duplicated per-head bytes", layer.Heads)
	}
}

func TestKVSnapshotStateBlocks_Good_SaveNativeRawOnlyToFileStore(t *testing.T) {
	ctx := context.Background()
	path := core.PathJoin(t.TempDir(), "kv-blocks.mvlog")
	store, err := filestore.Create(ctx, path)
	if err != nil {
		t.Fatalf("filestore.Create() error = %v", err)
	}
	snapshot := kvSnapshotBlocksTestSnapshot()
	head := &snapshot.Layers[0].Heads[0]
	for _, value := range head.Key {
		head.KeyBytes = appendUint16LE(head.KeyBytes, float32ToFloat16(value))
	}
	for _, value := range head.Value {
		head.ValueBytes = appendUint16LE(head.ValueBytes, uint16(math.Float32bits(value)>>16))
	}
	head.Key = nil
	head.Value = nil
	head.KeyDType = "float16"
	head.ValueDType = "bfloat16"

	bundle, err := snapshot.SaveStateBlocks(ctx, store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingNative,
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks(file native raw-only) error = %v", err)
	}
	if len(bundle.Blocks) != 2 || bundle.Blocks[0].State.Codec != filestore.CodecFile {
		t.Fatalf("bundle refs = %+v, want file-backed block refs", bundle.Blocks)
	}
	if bundle.Blocks[0].PayloadEncoding != kvSnapshotStatePayloadRaw || bundle.Blocks[0].PayloadByteCount == 0 {
		t.Fatalf("bundle payload = %+v, want raw file-backed payload", bundle.Blocks[0])
	}
	rawChunk, err := state.ResolveBytes(ctx, store, bundle.Blocks[0].State.ChunkID)
	if err != nil {
		t.Fatalf("ResolveBytes(file block) error = %v", err)
	}
	if len(rawChunk.Data) != bundle.Blocks[0].PayloadByteCount || core.Contains(rawChunk.Text, `"data"`) {
		t.Fatalf("raw file chunk = text %q data %d, want binary payload", rawChunk.Text, len(rawChunk.Data))
	}
	if err := store.Close(); err != nil {
		t.Fatalf("filestore.Close() error = %v", err)
	}
	if stat := core.Stat(path); !stat.OK || stat.Value.(core.FsFileInfo).Size() == 0 {
		t.Fatalf("file-backed store stat = %+v, want non-empty file", stat)
	}

	reopened, err := filestore.Open(ctx, path)
	if err != nil {
		t.Fatalf("filestore.Open() error = %v", err)
	}
	defer reopened.Close()
	loaded, err := LoadFromStateBlocksWithOptions(ctx, reopened, bundle, LoadOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("LoadFromStateBlocksWithOptions(file raw-only) error = %v", err)
	}
	loadedHead := loaded.Layers[0].Heads[0]
	if len(loadedHead.Key) != 0 || len(loadedHead.Value) != 0 {
		t.Fatalf("loaded float32 key/value lengths = %d/%d, want raw-only", len(loadedHead.Key), len(loadedHead.Value))
	}
	if len(loadedHead.KeyBytes) != 16 || len(loadedHead.ValueBytes) != 16 {
		t.Fatalf("loaded raw bytes = %d/%d, want file-backed native bytes", len(loadedHead.KeyBytes), len(loadedHead.ValueBytes))
	}
}

func TestKVSnapshotStateBlocks_Good_LoadNativeRawOnlyFromRegionStore(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	sourcePath := core.PathJoin(dir, "kv-blocks.mvlog")
	containerPath := core.PathJoin(dir, "session.kv")
	store, err := filestore.Create(ctx, sourcePath)
	if err != nil {
		t.Fatalf("filestore.Create() error = %v", err)
	}
	snapshot := kvSnapshotBlocksTestSnapshot()
	head := &snapshot.Layers[0].Heads[0]
	for _, value := range head.Key {
		head.KeyBytes = appendUint16LE(head.KeyBytes, float32ToFloat16(value))
	}
	for _, value := range head.Value {
		head.ValueBytes = appendUint16LE(head.ValueBytes, uint16(math.Float32bits(value)>>16))
	}
	head.Key = nil
	head.Value = nil
	head.KeyDType = "float16"
	head.ValueDType = "bfloat16"

	bundle, err := snapshot.SaveStateBlocks(ctx, store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingNative,
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks(region source) error = %v", err)
	}
	if err := store.Close(); err != nil {
		t.Fatalf("filestore.Close() error = %v", err)
	}
	read := core.ReadFile(sourcePath)
	if !read.OK {
		t.Fatalf("ReadFile(source) error = %s", read.Error())
	}
	prefix := []byte("KVST-region-head")
	payload := read.Value.([]byte)
	container := append(append(append([]byte(nil), prefix...), payload...), []byte("tail")...)
	if write := core.WriteFile(containerPath, container, 0o600); !write.OK {
		t.Fatalf("WriteFile(container) error = %s", write.Error())
	}

	region, err := filestore.OpenRegionWithSegmentAlias(ctx, containerPath, int64(len(prefix)), int64(len(payload)), sourcePath)
	if err != nil {
		t.Fatalf("OpenRegionWithSegmentAlias() error = %v", err)
	}
	defer region.Close()
	loaded, err := LoadFromStateBlocksWithOptions(ctx, region, bundle, LoadOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("LoadFromStateBlocksWithOptions(region raw-only) error = %v", err)
	}
	loadedHead := loaded.Layers[0].Heads[0]
	if len(loadedHead.Key) != 0 || len(loadedHead.Value) != 0 {
		t.Fatalf("loaded region float32 key/value lengths = %d/%d, want raw-only", len(loadedHead.Key), len(loadedHead.Value))
	}
	if len(loadedHead.KeyBytes) != 16 || len(loadedHead.ValueBytes) != 16 {
		t.Fatalf("loaded region raw bytes = %d/%d, want file-backed native bytes", len(loadedHead.KeyBytes), len(loadedHead.ValueBytes))
	}
}

func TestKVSnapshotStateBlocks_Good_UsesStreamingBinaryWriter(t *testing.T) {
	store := &streamRecordingStateStore{store: state.NewInMemoryStore(nil)}
	snapshot := kvSnapshotBlocksTestSnapshot()

	bundle, err := snapshot.SaveStateBlocks(context.Background(), store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingNative,
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks(streaming) error = %v", err)
	}
	if store.streamPuts != len(bundle.Blocks) || store.textPuts != 0 {
		t.Fatalf("writes = stream %d text %d for %d blocks, want streaming raw block writes", store.streamPuts, store.textPuts, len(bundle.Blocks))
	}
	if bundle.Blocks[0].PayloadEncoding != kvSnapshotStatePayloadRaw || bundle.Blocks[0].PayloadByteCount == 0 {
		t.Fatalf("block payload = %+v, want raw streamed payload", bundle.Blocks[0])
	}
	if len(store.streamOpts) != len(bundle.Blocks) {
		t.Fatalf("stream opts = %d, want one per block", len(store.streamOpts))
	}
	if _, ok := store.streamOpts[0].Tags["kv_hash"]; ok {
		t.Fatalf("stream metadata tags = %+v, want no blank kv_hash before payload is hashed", store.streamOpts[0].Tags)
	}
	if store.streamOpts[0].Tags["payload_encoding"] != kvSnapshotStatePayloadRaw {
		t.Fatalf("stream metadata payload_encoding = %q, want raw", store.streamOpts[0].Tags["payload_encoding"])
	}
	chunk, err := state.ResolveBytes(context.Background(), store, bundle.Blocks[0].State.ChunkID)
	if err != nil {
		t.Fatalf("ResolveBytes(streamed block) error = %v", err)
	}
	if len(chunk.Data) != bundle.Blocks[0].PayloadByteCount {
		t.Fatalf("streamed payload bytes = %d, want %d", len(chunk.Data), bundle.Blocks[0].PayloadByteCount)
	}
	loaded, err := LoadFromStateBlocksWithOptions(context.Background(), store, bundle, LoadOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("LoadFromStateBlocksWithOptions(streaming) error = %v", err)
	}
	if len(loaded.Tokens) != len(snapshot.Tokens) || loaded.TokenOffset != snapshot.TokenOffset {
		t.Fatalf("loaded metadata = %+v, want original token state", loaded)
	}
}

func TestKVSnapshotStateBlocks_Good_SaveStreamInfersBundleMetadata(t *testing.T) {
	store := &streamRecordingStateStore{store: state.NewInMemoryStore(nil)}
	snapshot := kvSnapshotBlocksTestSnapshot()

	bundle, err := SaveStateBlocksFromStream(context.Background(), store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingNative,
		URI:        "mlx://streamed/session",
	}, func(yield func(Block) (bool, error)) error {
		return snapshot.walkBlocks(2, false, yield)
	})

	if err != nil {
		t.Fatalf("SaveStateBlocksFromStream() error = %v", err)
	}
	if bundle.Architecture != snapshot.Architecture || bundle.TokenCount != len(snapshot.Tokens) || bundle.TokenOffset != snapshot.TokenOffset {
		t.Fatalf("bundle metadata = %+v, want snapshot metadata", bundle)
	}
	if bundle.NumLayers != snapshot.NumLayers || bundle.NumHeads != snapshot.NumHeads || bundle.HeadDim != snapshot.HeadDim || bundle.SeqLen != snapshot.SeqLen {
		t.Fatalf("bundle shape = %+v, want snapshot shape", bundle)
	}
	if len(bundle.Blocks) != 2 || store.streamPuts != 2 {
		t.Fatalf("bundle blocks = %d stream writes = %d, want two streamed blocks", len(bundle.Blocks), store.streamPuts)
	}
	if bundle.SnapshotHash == "" {
		t.Fatal("bundle SnapshotHash is empty")
	}
	loaded, err := LoadFromStateBlocksWithOptions(context.Background(), store, bundle, LoadOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("LoadFromStateBlocksWithOptions(stream bundle) error = %v", err)
	}
	if len(loaded.Tokens) != len(snapshot.Tokens) || loaded.TokenOffset != snapshot.TokenOffset {
		t.Fatalf("loaded metadata = %+v, want original token state", loaded)
	}
}

func TestKVSnapshotStateBlocks_Good_StreamReusesPrefixBlocks(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	parent := kvSnapshotBlocksTestSnapshot()
	parentBundle, err := parent.SaveStateBlocks(ctx, store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingNative,
		URI:        "mlx://parent",
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks(parent) error = %v", err)
	}
	child := kvSnapshotBlocksTestSnapshot()
	child.Tokens[2] = 9
	child.Tokens[3] = 10
	child.Generated = []int32{10}
	child.Layers[0].Heads[0].Key[4] = 90
	child.Layers[0].Heads[0].Key[5] = 91
	child.Layers[0].Heads[0].Key[6] = 92
	child.Layers[0].Heads[0].Key[7] = 93
	child.Layers[0].Heads[0].Value[4] = 100
	child.Layers[0].Heads[0].Value[5] = 101
	child.Layers[0].Heads[0].Value[6] = 102
	child.Layers[0].Heads[0].Value[7] = 103

	childBundle, err := SaveStateBlocksFromStream(ctx, store, StateBlockOptions{
		BlockSize:         2,
		KVEncoding:        EncodingNative,
		URI:               "mlx://child",
		ReusePrefix:       parentBundle,
		ReusePrefixTokens: 2,
	}, func(yield func(Block) (bool, error)) error {
		return child.walkBlocks(2, false, yield)
	})
	if err != nil {
		t.Fatalf("SaveStateBlocksFromStream(child reuse) error = %v", err)
	}
	if childBundle.ReusedBlocks != 1 {
		t.Fatalf("child reused blocks = %d, want 1", childBundle.ReusedBlocks)
	}
	if childBundle.Blocks[0].State.ChunkID != parentBundle.Blocks[0].State.ChunkID {
		t.Fatalf("child first block ref = %+v, want parent first ref %+v", childBundle.Blocks[0], parentBundle.Blocks[0])
	}
	if childBundle.Blocks[1].State.ChunkID == parentBundle.Blocks[1].State.ChunkID {
		t.Fatalf("child second block reused parent ref %+v, want new suffix block", childBundle.Blocks[1])
	}
	loaded, err := LoadFromStateBlocksWithOptions(ctx, store, childBundle, LoadOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("LoadFromStateBlocksWithOptions(child reuse) error = %v", err)
	}
	if len(loaded.Tokens) != 4 || loaded.Tokens[0] != 1 || loaded.Tokens[2] != 9 || loaded.Tokens[3] != 10 {
		t.Fatalf("loaded child tokens = %v, want reused prefix plus new suffix", loaded.Tokens)
	}
}

func TestKVSnapshotStateBlocks_Bad_SaveStreamErrors(t *testing.T) {
	snapshot := kvSnapshotBlocksTestSnapshot()
	store := &streamRecordingStateStore{store: state.NewInMemoryStore(nil)}
	if _, err := SaveStateBlocksFromStream(context.Background(), nil, StateBlockOptions{}, func(func(Block) (bool, error)) error {
		return nil
	}); err == nil {
		t.Fatal("SaveStateBlocksFromStream(nil store) error = nil")
	}
	if _, err := SaveStateBlocksFromStream(context.Background(), store, StateBlockOptions{}, nil); err == nil {
		t.Fatal("SaveStateBlocksFromStream(nil stream) error = nil")
	}
	if _, err := SaveStateBlocksFromStream(context.Background(), store, StateBlockOptions{}, func(func(Block) (bool, error)) error {
		return nil
	}); err == nil {
		t.Fatal("SaveStateBlocksFromStream(empty stream) error = nil")
	}
	if _, err := SaveStateBlocksFromStream(context.Background(), store, StateBlockOptions{}, func(yield func(Block) (bool, error)) error {
		_, err := yield(Block{Index: 0, TokenStart: 0, TokenCount: 1})
		return err
	}); err == nil {
		t.Fatal("SaveStateBlocksFromStream(nil block snapshot) error = nil")
	}

	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := SaveStateBlocksFromStream(cancelled, store, StateBlockOptions{}, func(yield func(Block) (bool, error)) error {
		return snapshot.walkBlocks(2, false, yield)
	}); err == nil {
		t.Fatal("SaveStateBlocksFromStream(cancelled context) error = nil")
	}

	writerStore := &failingStreamStateStore{}
	if _, err := SaveStateBlocksFromStream(context.Background(), writerStore, StateBlockOptions{}, func(yield func(Block) (bool, error)) error {
		return snapshot.walkBlocks(2, false, yield)
	}); err == nil {
		t.Fatal("SaveStateBlocksFromStream(writer failure) error = nil")
	}
}

func TestKVSnapshotStateBlocks_Bad_ValidationAndLoadErrors(t *testing.T) {
	if _, err := LoadFromStateBlocks(context.Background(), nil, &StateBlockBundle{}); err == nil {
		t.Fatal("LoadFromStateBlocks(nil store) error = nil")
	}
	if _, err := LoadFromStateBlocks(context.Background(), state.NewInMemoryStore(nil), nil); err == nil {
		t.Fatal("LoadFromStateBlocks(nil bundle) error = nil")
	}
	for _, bundle := range []*StateBlockBundle{
		{Version: StateBlockVersion + 1, Kind: StateBlockBundleKind, TokenCount: 1, Blocks: []StateBlockRef{{}}},
		{Version: StateBlockVersion, Kind: "wrong", TokenCount: 1, Blocks: []StateBlockRef{{}}},
		{Version: StateBlockVersion, Kind: StateBlockBundleKind, Blocks: []StateBlockRef{{}}},
		{Version: StateBlockVersion, Kind: StateBlockBundleKind, TokenCount: 1},
	} {
		if err := ValidateStateBlockBundle(bundle); err == nil {
			t.Fatalf("ValidateStateBlockBundle(%+v) error = nil", bundle)
		}
	}
	if err := ValidateStateBlockBundle(nil); err == nil {
		t.Fatal("ValidateStateBlockBundle(nil) error = nil")
	}
	if _, err := LoadPrefixFromStateBlocks(context.Background(), nil, &StateBlockBundle{}, 1); err == nil {
		t.Fatal("LoadPrefixFromStateBlocks(nil store) error = nil")
	}
}

func TestKVSnapshotStateBlocks_Bad_RawBlockIntegrity(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	ref, err := store.PutBytes(context.Background(), []byte(kvSnapshotMagic), state.PutOptions{})
	if err != nil {
		t.Fatalf("PutBytes() error = %v", err)
	}
	blockRef := StateBlockRef{
		Index:            0,
		TokenStart:       0,
		TokenCount:       1,
		KVHash:           "not-the-hash",
		PayloadEncoding:  kvSnapshotStatePayloadRaw,
		PayloadByteCount: len(kvSnapshotMagic),
		State:            ref,
	}
	if _, err := loadRawKVSnapshotStateBlockWithOptions(context.Background(), store, blockRef, LoadOptions{}); err == nil {
		t.Fatal("loadRawKVSnapshotStateBlockWithOptions(hash mismatch) error = nil")
	}
	blockRef.KVHash = ""
	blockRef.PayloadByteCount++
	if _, err := loadRawKVSnapshotStateBlockWithOptions(context.Background(), store, blockRef, LoadOptions{}); err == nil {
		t.Fatal("loadRawKVSnapshotStateBlockWithOptions(length mismatch) error = nil")
	}
}

func TestKVSnapshotStateBlocks_Bad_EnvelopeIntegrity(t *testing.T) {
	for _, envelope := range []kvSnapshotStateBlockEnvelope{
		{Version: StateBlockVersion + 1, Kind: KVSnapshotStateBlockKind, BinaryEncoding: "base64"},
		{Version: StateBlockVersion, Kind: "wrong", BinaryEncoding: "base64"},
		{Version: StateBlockVersion, Kind: KVSnapshotStateBlockKind, BinaryEncoding: "hex"},
		{Version: StateBlockVersion, Kind: KVSnapshotStateBlockKind, BinaryEncoding: "base64", Data: "not base64"},
		{Version: StateBlockVersion, Kind: KVSnapshotStateBlockKind, BinaryEncoding: "base64", Data: core.Base64Encode([]byte("x")), PayloadByteCount: 2},
		{Version: StateBlockVersion, Kind: KVSnapshotStateBlockKind, BinaryEncoding: "base64", Data: core.Base64Encode([]byte("x")), KVHash: "bad"},
	} {
		if _, err := decodeKVSnapshotStateBlockEnvelope(envelope, ""); err == nil {
			t.Fatalf("decodeKVSnapshotStateBlockEnvelope(%+v) error = nil", envelope)
		}
	}
	data := []byte("x")
	envelope := kvSnapshotStateBlockEnvelope{
		Version:        StateBlockVersion,
		Kind:           KVSnapshotStateBlockKind,
		BinaryEncoding: "base64",
		Data:           core.Base64Encode(data),
	}
	if _, err := decodeKVSnapshotStateBlockEnvelope(envelope, "wrong-ref-hash"); err == nil {
		t.Fatal("decodeKVSnapshotStateBlockEnvelope(ref hash mismatch) error = nil")
	}
}

func TestKVSnapshotStateBlocks_Good_LoadPrefixOnlyReadsNeededBlocks(t *testing.T) {
	source := state.NewInMemoryStore(nil)
	snapshot := kvSnapshotBlocksTestSnapshot()
	bundle, err := snapshot.SaveStateBlocks(context.Background(), source, StateBlockOptions{BlockSize: 2})
	if err != nil {
		t.Fatalf("SaveStateBlocks() error = %v", err)
	}
	store := &recordingStateStore{store: source}

	loaded, err := LoadPrefixFromStateBlocks(context.Background(), store, bundle, 2)
	if err != nil {
		t.Fatalf("LoadPrefixFromStateBlocks() error = %v", err)
	}

	if len(store.resolved) != 1 || store.resolved[0] != bundle.Blocks[0].State.ChunkID {
		t.Fatalf("resolved chunks = %v, want only first block chunk %d", store.resolved, bundle.Blocks[0].State.ChunkID)
	}
	if loaded.TokenOffset != 2 || loaded.SeqLen != 2 || len(loaded.Tokens) != 2 || loaded.Tokens[0] != 1 || loaded.Tokens[1] != 2 {
		t.Fatalf("loaded prefix metadata = %+v, want first two tokens", loaded)
	}
	head, ok := loaded.Head(0, 0)
	if !ok {
		t.Fatal("loaded Head(0,0) ok = false")
	}
	if len(head.Key) != 4 || head.Key[0] < 9.99 || head.Key[3] < 12.99 {
		t.Fatalf("loaded prefix head = %+v, want first block key/value tensors", head)
	}
	if len(loaded.Logits) != 0 {
		t.Fatalf("loaded prefix logits = %v, want no logits for non-final prefix", loaded.Logits)
	}
}

func TestKVSnapshotStateBlocks_Good_LoadPartialPrefixSlicesCoveringBlock(t *testing.T) {
	source := state.NewInMemoryStore(nil)
	snapshot := kvSnapshotBlocksTestSnapshot()
	bundle, err := snapshot.SaveStateBlocks(context.Background(), source, StateBlockOptions{BlockSize: 2})
	if err != nil {
		t.Fatalf("SaveStateBlocks() error = %v", err)
	}

	loaded, err := LoadPrefixFromStateBlocks(context.Background(), source, bundle, 3)
	if err != nil {
		t.Fatalf("LoadPrefixFromStateBlocks() error = %v", err)
	}

	if loaded.TokenOffset != 3 || loaded.SeqLen != 3 || len(loaded.Tokens) != 3 || loaded.Tokens[2] != 3 {
		t.Fatalf("loaded prefix metadata = %+v, want first three tokens", loaded)
	}
	head, ok := loaded.Head(0, 0)
	if !ok {
		t.Fatal("loaded Head(0,0) ok = false")
	}
	if len(head.Key) != 6 || head.Key[0] < 9.99 || head.Key[5] < 14.99 {
		t.Fatalf("loaded prefix head = %+v, want sliced first three tokens", head)
	}
	if len(loaded.Logits) != 0 {
		t.Fatalf("loaded prefix logits = %v, want no logits for partial final block", loaded.Logits)
	}
}

func TestKVSnapshotStateBlocks_Good_LoadPrefixTokensSkipsKVAssembly(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	first := stateTokenOnlyTestSnapshot([]int32{1, 2}, 2, 2)
	second := stateTokenOnlyTestSnapshot([]int32{3, 4}, 4, 1)
	bundle, err := SaveStateBlocksFromStream(ctx, store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingNative,
	}, func(yield func(Block) (bool, error)) error {
		ok, err := yield(Block{Index: 0, TokenStart: 0, TokenCount: 2, Snapshot: first})
		if err != nil || !ok {
			return err
		}
		_, err = yield(Block{Index: 1, TokenStart: 2, TokenCount: 2, Snapshot: second})
		return err
	})
	if err != nil {
		t.Fatalf("SaveStateBlocksFromStream() error = %v", err)
	}

	if _, err := LoadPrefixFromStateBlocksWithOptions(ctx, store, bundle, 4, LoadOptions{RawKVOnly: true}); err == nil {
		t.Fatal("LoadPrefixFromStateBlocksWithOptions(mismatched shapes) error = nil")
	}
	tokens, err := LoadPrefixTokensFromStateBlocksWithOptions(ctx, store, bundle, 4, LoadOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("LoadPrefixTokensFromStateBlocksWithOptions() error = %v", err)
	}
	if len(tokens) != 4 || tokens[0] != 1 || tokens[3] != 4 {
		t.Fatalf("tokens = %v, want [1 2 3 4]", tokens)
	}
}

type recordingStateStore struct {
	store    state.Store
	resolved []int
}

func (s *recordingStateStore) Get(ctx context.Context, chunkID int) (string, error) {
	s.resolved = append(s.resolved, chunkID)
	return s.store.Get(ctx, chunkID)
}

func (s *recordingStateStore) Resolve(ctx context.Context, chunkID int) (state.Chunk, error) {
	s.resolved = append(s.resolved, chunkID)
	return state.Resolve(ctx, s.store, chunkID)
}

type textOnlyStateStore struct {
	store *state.InMemoryStore
}

func (s *textOnlyStateStore) Get(ctx context.Context, chunkID int) (string, error) {
	return s.store.Get(ctx, chunkID)
}

func (s *textOnlyStateStore) Resolve(ctx context.Context, chunkID int) (state.Chunk, error) {
	return s.store.Resolve(ctx, chunkID)
}

func (s *textOnlyStateStore) ResolveURI(ctx context.Context, uri string) (state.Chunk, error) {
	return s.store.ResolveURI(ctx, uri)
}

func (s *textOnlyStateStore) Put(ctx context.Context, text string, opts state.PutOptions) (state.ChunkRef, error) {
	return s.store.Put(ctx, text, opts)
}

type streamRecordingStateStore struct {
	store      *state.InMemoryStore
	streamPuts int
	textPuts   int
	streamOpts []state.PutOptions
}

func (s *streamRecordingStateStore) Get(ctx context.Context, chunkID int) (string, error) {
	return s.store.Get(ctx, chunkID)
}

func (s *streamRecordingStateStore) Resolve(ctx context.Context, chunkID int) (state.Chunk, error) {
	return s.store.Resolve(ctx, chunkID)
}

func (s *streamRecordingStateStore) ResolveBytes(ctx context.Context, chunkID int) (state.Chunk, error) {
	return s.store.ResolveBytes(ctx, chunkID)
}

func (s *streamRecordingStateStore) Put(ctx context.Context, text string, opts state.PutOptions) (state.ChunkRef, error) {
	s.textPuts++
	return s.store.Put(ctx, text, opts)
}

func (s *streamRecordingStateStore) PutBytesStream(ctx context.Context, payloadSize int, opts state.PutOptions, write func(stdio.Writer) error) (state.ChunkRef, error) {
	s.streamPuts++
	s.streamOpts = append(s.streamOpts, opts)
	writer := &streamRecordingWriter{data: make([]byte, 0, payloadSize)}
	if err := write(writer); err != nil {
		return state.ChunkRef{}, err
	}
	if len(writer.data) != payloadSize {
		return state.ChunkRef{}, core.NewError("stream payload size mismatch")
	}
	return s.store.PutBytes(ctx, writer.data, opts)
}

type streamRecordingWriter struct {
	data []byte
}

func (w *streamRecordingWriter) Write(data []byte) (int, error) {
	w.data = append(w.data, data...)
	return len(data), nil
}

type failingStreamStateStore struct{}

func (s *failingStreamStateStore) Put(context.Context, string, state.PutOptions) (state.ChunkRef, error) {
	return state.ChunkRef{}, core.NewError("unexpected text write")
}

func (s *failingStreamStateStore) PutBytesStream(ctx context.Context, payloadSize int, opts state.PutOptions, write func(stdio.Writer) error) (state.ChunkRef, error) {
	err := write(failingStreamWriter{})
	if err == nil {
		err = core.NewError("expected writer failure")
	}
	return state.ChunkRef{}, err
}

type failingStreamWriter struct{}

func (failingStreamWriter) Write([]byte) (int, error) {
	return 0, core.NewError("stream writer failed")
}

// failingGetStateStore implements the minimal state.Store contract (Get only)
// and fails every resolve. Because state.Resolve / ResolveBytes / BorrowRefBytes
// all fall through to Get for a plain Store, one double drives the resolve-error
// arm of every block load entry point.
type failingGetStateStore struct{}

func (failingGetStateStore) Get(context.Context, int) (string, error) {
	return "", core.NewError("resolve refused")
}

func kvSnapshotBlocksTestSnapshot() *Snapshot {
	return &Snapshot{
		Version:       SnapshotVersion,
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
		Layers: []LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []HeadSnapshot{{
				Key:   []float32{10, 11, 12, 13, 14, 15, 16, 17},
				Value: []float32{20, 21, 22, 23, 24, 25, 26, 27},
			}},
		}},
	}
}

func stateTokenOnlyTestSnapshot(tokens []int32, tokenOffset, headDim int) *Snapshot {
	key := make([]float32, len(tokens)*headDim)
	value := make([]float32, len(tokens)*headDim)
	for i := range key {
		key[i] = float32(i + tokenOffset)
		value[i] = float32(i + tokenOffset + 100)
	}
	return &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        append([]int32(nil), tokens...),
		TokenOffset:   tokenOffset,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        len(tokens),
		HeadDim:       headDim,
		NumQueryHeads: 1,
		Layers: []LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []HeadSnapshot{{
				Key:   key,
				Value: value,
			}},
		}},
	}
}

// kvSnapshotBlocksTestBundle saves the 4-token fixture as a 2-block State bundle
// into a fresh in-memory store, returning both for round-trip and error tests.
func kvSnapshotBlocksTestBundle(t *testing.T) (*state.InMemoryStore, *StateBlockBundle) {
	t.Helper()
	store := state.NewInMemoryStore(nil)
	bundle, err := kvSnapshotBlocksTestSnapshot().SaveStateBlocks(context.Background(), store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingQ8,
		URI:        "mlx://session/blocks",
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks() error = %v", err)
	}
	return store, bundle
}

// TestKVSnapshotBlocks_MemvidAliases_Good asserts every deprecated Memvid-named
// block alias forwards transparently to its canonical State counterpart: a save
// via one name is loadable via the other, and the manifest survives a
// save-bundle / load-bundle round trip through the deprecated entry points.
func TestKVSnapshotBlocks_MemvidAliasesForward(t *testing.T) {
	ctx := context.Background()

	// SaveMemvidBlocks (alias of SaveStateBlocks) → LoadFromStateBlocks.
	store := state.NewInMemoryStore(nil)
	bundle, err := kvSnapshotBlocksTestSnapshot().SaveMemvidBlocks(ctx, store, StateBlockOptions{BlockSize: 2, KVEncoding: EncodingQ8})
	if err != nil {
		t.Fatalf("SaveMemvidBlocks() error = %v", err)
	}
	if len(bundle.Blocks) != 2 || bundle.Kind != MemvidBlockBundleKind {
		t.Fatalf("SaveMemvidBlocks() bundle = %+v, want two blocks", bundle)
	}

	// SaveMemvidBlocksFromStream (alias of SaveStateBlocksFromStream).
	streamStore := state.NewInMemoryStore(nil)
	streamBundle, err := SaveMemvidBlocksFromStream(ctx, streamStore, StateBlockOptions{BlockSize: 2, KVEncoding: EncodingQ8}, func(yield func(Block) (bool, error)) error {
		_, err := yield(Block{Index: 0, TokenStart: 0, TokenCount: 4, Snapshot: kvSnapshotBlocksTestSnapshot()})
		return err
	})
	if err != nil || len(streamBundle.Blocks) == 0 {
		t.Fatalf("SaveMemvidBlocksFromStream() = %+v, err = %v", streamBundle, err)
	}

	// SaveMemvidBlockBundle (alias) → LoadMemvidBlockBundle (alias).
	bundleRef, err := SaveMemvidBlockBundle(ctx, store, bundle, "mlx://session/memvid-manifest")
	if err != nil {
		t.Fatalf("SaveMemvidBlockBundle() error = %v", err)
	}
	if bundleRef.ChunkID == 0 {
		t.Fatalf("SaveMemvidBlockBundle() ref = %+v, want written chunk", bundleRef)
	}
	reloaded, err := LoadMemvidBlockBundle(ctx, store, "mlx://session/memvid-manifest")
	if err != nil {
		t.Fatalf("LoadMemvidBlockBundle() error = %v", err)
	}
	if reloaded.SnapshotHash != bundle.SnapshotHash || len(reloaded.Blocks) != len(bundle.Blocks) {
		t.Fatalf("LoadMemvidBlockBundle() = %+v, want bundle round trip", reloaded)
	}

	// ValidateMemvidBlockBundle (alias of ValidateStateBlockBundle).
	if err := ValidateMemvidBlockBundle(bundle); err != nil {
		t.Fatalf("ValidateMemvidBlockBundle(valid) error = %v", err)
	}
	if err := ValidateMemvidBlockBundle(&MemvidBlockBundle{}); err == nil {
		t.Fatal("ValidateMemvidBlockBundle(empty) error = nil, want validation error")
	}

	// LoadFromMemvidBlocks / LoadFromMemvidBlocksWithOptions (aliases).
	loaded, err := LoadFromMemvidBlocks(ctx, store, bundle)
	if err != nil {
		t.Fatalf("LoadFromMemvidBlocks() error = %v", err)
	}
	if len(loaded.Tokens) != 4 {
		t.Fatalf("LoadFromMemvidBlocks() tokens = %d, want 4", len(loaded.Tokens))
	}
	if _, err := LoadFromMemvidBlocksWithOptions(ctx, store, bundle, LoadOptions{}); err != nil {
		t.Fatalf("LoadFromMemvidBlocksWithOptions() error = %v", err)
	}

	// LoadPrefixFromMemvidBlocks / WithOptions (aliases).
	prefix, err := LoadPrefixFromMemvidBlocks(ctx, store, bundle, 2)
	if err != nil || len(prefix.Tokens) != 2 {
		t.Fatalf("LoadPrefixFromMemvidBlocks() = %+v, err = %v, want 2 tokens", prefix, err)
	}
	if _, err := LoadPrefixFromMemvidBlocksWithOptions(ctx, store, bundle, 2, LoadOptions{}); err != nil {
		t.Fatalf("LoadPrefixFromMemvidBlocksWithOptions() error = %v", err)
	}

	// LoadMemvidBlockWithOptions (alias of LoadStateBlockWithOptions).
	block, err := LoadMemvidBlockWithOptions(ctx, store, bundle.Blocks[0], LoadOptions{})
	if err != nil || block.TokenCount != 2 {
		t.Fatalf("LoadMemvidBlockWithOptions() = %+v, err = %v, want first block", block, err)
	}
}

// TestKVSnapshotBlocks_LoadStateBlockBundle_Bad covers the bundle-load guard
// branches: nil store, blank URI, and a missing URI.
func TestKVSnapshotBlocks_LoadStateBlockBundle_Bad(t *testing.T) {
	ctx := context.Background()
	store, _ := kvSnapshotBlocksTestBundle(t)

	if _, err := LoadStateBlockBundle(ctx, nil, "mlx://x"); err == nil {
		t.Fatal("LoadStateBlockBundle(nil store) error = nil")
	}
	if _, err := LoadStateBlockBundle(ctx, store, ""); err == nil {
		t.Fatal("LoadStateBlockBundle(blank URI) error = nil")
	}
	if _, err := LoadStateBlockBundle(ctx, store, "mlx://does-not-exist"); err == nil {
		t.Fatal("LoadStateBlockBundle(missing URI) error = nil")
	}
}

// TestKVSnapshotBlocks_LoadPrefixFromStateBlocksWithOptions_Bad exercises the
// uncovered guard and edge branches: nil store, an oversized prefix, an exact
// full prefix (delegates to the full load), and a zero prefix.
func TestKVSnapshotBlocks_LoadPrefixFromStateBlocksWithOptions_Bad(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	if _, err := LoadPrefixFromStateBlocksWithOptions(ctx, nil, bundle, 1, LoadOptions{}); err == nil {
		t.Fatal("LoadPrefixFromStateBlocksWithOptions(nil store) error = nil")
	}
	if _, err := LoadPrefixFromStateBlocksWithOptions(ctx, store, bundle, bundle.TokenCount+1, LoadOptions{}); err == nil {
		t.Fatal("LoadPrefixFromStateBlocksWithOptions(oversized prefix) error = nil")
	}
	// Exact full prefix: delegates to the full block load, returns all tokens.
	full, err := LoadPrefixFromStateBlocksWithOptions(ctx, store, bundle, bundle.TokenCount, LoadOptions{})
	if err != nil || len(full.Tokens) != bundle.TokenCount {
		t.Fatalf("LoadPrefixFromStateBlocksWithOptions(full) = %+v, err = %v", full, err)
	}
	// Zero prefix is treated as the full bundle.
	zero, err := LoadPrefixFromStateBlocksWithOptions(ctx, store, bundle, 0, LoadOptions{})
	if err != nil || len(zero.Tokens) != bundle.TokenCount {
		t.Fatalf("LoadPrefixFromStateBlocksWithOptions(zero) = %+v, err = %v", zero, err)
	}
}

// TestKVSnapshotBlocks_LoadPrefixTokens_GoodBadUgly covers the token-only prefix
// path: a partial prefix (Good), guard errors (Bad), and a manifest with
// non-contiguous block indices that trips the contiguity check (Ugly).
func TestKVSnapshotBlocks_LoadPrefixTokens_GoodBadUgly(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	// Good: mid-block prefix returns exactly the requested token count.
	tokens, err := LoadPrefixTokensFromStateBlocks(ctx, store, bundle, 3)
	if err != nil {
		t.Fatalf("LoadPrefixTokensFromStateBlocks() error = %v", err)
	}
	if len(tokens) != 3 || tokens[0] != 1 || tokens[2] != 3 {
		t.Fatalf("tokens = %v, want first three", tokens)
	}

	// Bad: nil store and an oversized prefix.
	if _, err := LoadPrefixTokensFromStateBlocks(ctx, nil, bundle, 1); err == nil {
		t.Fatal("LoadPrefixTokensFromStateBlocks(nil store) error = nil")
	}
	if _, err := LoadPrefixTokensFromStateBlocks(ctx, store, bundle, bundle.TokenCount+1); err == nil {
		t.Fatal("LoadPrefixTokensFromStateBlocks(oversized) error = nil")
	}

	// Ugly: tamper the manifest so block indices are non-contiguous.
	broken := *bundle
	broken.Blocks = append([]StateBlockRef(nil), bundle.Blocks...)
	broken.Blocks[0].Index = 5
	if _, err := LoadPrefixTokensFromStateBlocks(ctx, store, &broken, 4); err == nil {
		t.Fatal("LoadPrefixTokensFromStateBlocks(non-contiguous) error = nil")
	}
}

// TestKVSnapshotBlocks_LoadStateBlockTokens_Good covers the token-only single
// block loader and its WithOptions sibling: tokens are returned without K/V
// assembly.
func TestKVSnapshotBlocks_LoadStateBlockTokens_Good(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	block, err := LoadStateBlockTokens(ctx, store, bundle.Blocks[0])
	if err != nil {
		t.Fatalf("LoadStateBlockTokens() error = %v", err)
	}
	if block.TokenCount != 2 || block.Index != 0 || len(block.Tokens) != 2 || block.Tokens[0] != 1 {
		t.Fatalf("block = %+v, want first two token IDs", block)
	}

	withOpts, err := LoadStateBlockTokensWithOptions(ctx, store, bundle.Blocks[1], LoadOptions{})
	if err != nil {
		t.Fatalf("LoadStateBlockTokensWithOptions() error = %v", err)
	}
	if withOpts.TokenStart != 2 || len(withOpts.Tokens) != 2 || withOpts.Tokens[0] != 3 {
		t.Fatalf("block = %+v, want second block tokens", withOpts)
	}
}

// TestKVSnapshotBlocks_TokensFromTextStore_Good drives the JSON/base64 envelope
// branch of the token loaders. A text-only store cannot accept raw binary, so
// SaveStateBlocks falls back to base64-wrapped envelopes — LoadStateBlockTokens
// and LoadPrefixTokens then take their envelope-decode paths rather than the raw
// fast path.
func TestKVSnapshotBlocks_TokensFromTextStoreEnvelope(t *testing.T) {
	ctx := context.Background()
	store := &textOnlyStateStore{store: state.NewInMemoryStore(nil)}
	bundle, err := kvSnapshotBlocksTestSnapshot().SaveStateBlocks(ctx, store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingQ8,
		URI:        "mlx://session/text-tokens",
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks(text store) error = %v", err)
	}
	if bundle.Blocks[0].PayloadEncoding != kvSnapshotStatePayloadJSONBase64 {
		t.Fatalf("payload encoding = %q, want JSON/base64 fallback", bundle.Blocks[0].PayloadEncoding)
	}

	block, err := LoadStateBlockTokens(ctx, store, bundle.Blocks[0])
	if err != nil {
		t.Fatalf("LoadStateBlockTokens(envelope) error = %v", err)
	}
	if block.TokenCount != 2 || len(block.Tokens) != 2 || block.Tokens[1] != 2 {
		t.Fatalf("block = %+v, want first block tokens via envelope", block)
	}

	tokens, err := LoadPrefixTokensFromStateBlocks(ctx, store, bundle, 4)
	if err != nil {
		t.Fatalf("LoadPrefixTokensFromStateBlocks(envelope) error = %v", err)
	}
	if len(tokens) != 4 || tokens[3] != 4 {
		t.Fatalf("tokens = %v, want all four via envelope path", tokens)
	}
}

// TestKVSnapshotBlocks_TokensFromTextStore_Ugly tampers a text-store manifest so
// the envelope-path metadata checks fail: a ref whose recorded TokenCount no
// longer matches the stored block trips errTokenBlockMetadata / count guards.
func TestKVSnapshotBlocks_TokensFromTextStoreTampered(t *testing.T) {
	ctx := context.Background()
	store := &textOnlyStateStore{store: state.NewInMemoryStore(nil)}
	bundle, err := kvSnapshotBlocksTestSnapshot().SaveStateBlocks(ctx, store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingQ8,
		URI:        "mlx://session/text-ugly",
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks(text store) error = %v", err)
	}

	// Loading a single block whose ref hash no longer matches the stored
	// envelope must fail the envelope hash check.
	badHash := bundle.Blocks[0]
	badHash.KVHash = "sha256:not-the-stored-hash"
	if _, err := LoadStateBlockTokensWithOptions(ctx, store, badHash, LoadOptions{}); err == nil {
		t.Fatal("LoadStateBlockTokensWithOptions(bad hash) error = nil")
	}

	// Tamper the manifest's recorded per-block TokenCount: the envelope still
	// decodes 2 tokens but the ref claims 1, so the prefix loader's
	// block-token-count check rejects it.
	broken := *bundle
	broken.Blocks = append([]StateBlockRef(nil), bundle.Blocks...)
	broken.Blocks[0].TokenCount = 1
	if _, err := LoadPrefixTokensFromStateBlocks(ctx, store, &broken, 4); err == nil {
		t.Fatal("LoadPrefixTokensFromStateBlocks(tampered token count) error = nil")
	}
}

// TestKVSnapshotBlocks_LoadPrefixPartialSlice_Good drives the partial-prefix
// slicing path of LoadPrefixFromStateBlocksWithOptions: a prefix that lands
// inside the final covering block forces the SliceBlock trim branch.
func TestKVSnapshotBlocks_LoadPrefixPartialSliceCovering(t *testing.T) {
	store, bundle := kvSnapshotBlocksTestBundle(t)

	// prefix 1 lands inside the first 2-token block — the loader reads the
	// covering block then trims it to a single token.
	loaded, err := LoadPrefixFromStateBlocksWithOptions(context.Background(), store, bundle, 1, LoadOptions{})
	if err != nil {
		t.Fatalf("LoadPrefixFromStateBlocksWithOptions(partial) error = %v", err)
	}
	if len(loaded.Tokens) != 1 || loaded.Tokens[0] != 1 {
		t.Fatalf("loaded = %+v, want single trimmed token", loaded)
	}
	if len(loaded.Generated) != 0 || len(loaded.Logits) != 0 {
		t.Fatalf("loaded = %+v, want terminal state cleared for non-final prefix", loaded)
	}
}

// TestKVSnapshotBlocks_SaveStateBlocks_Bad covers the SaveStateBlocks guard
// branches: nil snapshot, nil store, and an unsupported KV encoding.
// TestKVSnapshotBlocks_LoadFromStateBlocks_Ugly drives the load-path validation
// branches over a real bundle: a bad version, a wrong kind, and a manifest whose
// block refs are reordered so the contiguity / out-of-order checks reject it.
func TestKVSnapshotBlocks_LoadFromStateBlocks_Ugly(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	badVersion := *bundle
	badVersion.Version = StateBlockVersion + 1
	if _, err := LoadFromStateBlocks(ctx, store, &badVersion); err == nil {
		t.Fatal("LoadFromStateBlocks(bad version) error = nil")
	}

	badKind := *bundle
	badKind.Kind = "not-a-kv-bundle"
	if _, err := LoadFromStateBlocks(ctx, store, &badKind); err == nil {
		t.Fatal("LoadFromStateBlocks(bad kind) error = nil")
	}

	// Reorder the block refs: block index 1 is presented first, so the
	// in-order index check (ref.Index != index) rejects the manifest.
	reordered := *bundle
	reordered.Blocks = []StateBlockRef{bundle.Blocks[1], bundle.Blocks[0]}
	if _, err := LoadFromStateBlocks(ctx, store, &reordered); err == nil {
		t.Fatal("LoadFromStateBlocks(reordered blocks) error = nil")
	}

	// A bundle whose recorded TokenOffset disagrees with the assembled
	// snapshot's offset trips the offset-mismatch guard.
	badOffset := *bundle
	badOffset.TokenOffset = bundle.TokenOffset + 1000
	if _, err := LoadFromStateBlocks(ctx, store, &badOffset); err == nil {
		t.Fatal("LoadFromStateBlocks(offset mismatch) error = nil")
	}
}

// The trusted-prefix sleep lane: parent blocks below the boundary graft by
// reference with no capture and no hash. The stream asserts the capture side
// was never asked for the grafted range (BlockStartToken semantics).
func TestKVSnapshotStateBlocks_Good_TrustedPrefixGraftsWithoutCapture(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	parent := kvSnapshotBlocksTestSnapshot()
	parentBundle, err := parent.SaveStateBlocks(ctx, store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingNative,
		URI:        "mlx://trusted/parent",
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks(parent) error = %v", err)
	}

	opts := StateBlockOptions{
		BlockSize:          2,
		KVEncoding:         EncodingNative,
		URI:                "mlx://trusted/child",
		ReusePrefix:        parentBundle,
		ReusePrefixTokens:  2,
		ReusePrefixTrusted: true,
	}
	if boundary := TrustedReuseBoundary(opts, 2); boundary != 2 {
		t.Fatalf("TrustedReuseBoundary = %d, want 2", boundary)
	}

	child := kvSnapshotBlocksTestSnapshot()
	captured := []int{}
	childBundle, err := SaveStateBlocksFromStream(ctx, store, opts, func(yield func(Block) (bool, error)) error {
		// Mirror the capture side: BlockStartToken skips blocks ending at or
		// before the trusted boundary.
		return child.walkBlocks(2, false, func(block Block) (bool, error) {
			if block.TokenStart+block.TokenCount <= 2 {
				return true, nil
			}
			captured = append(captured, block.TokenStart)
			return yield(block)
		})
	})
	if err != nil {
		t.Fatalf("SaveStateBlocksFromStream(trusted) error = %v", err)
	}
	if len(captured) != 1 || captured[0] != 2 {
		t.Fatalf("captured starts = %v, want only the post-boundary block [2]", captured)
	}
	if childBundle.ReusedBlocks != 1 || len(childBundle.Blocks) != 2 {
		t.Fatalf("bundle reused=%d blocks=%d, want 1 grafted + 1 streamed", childBundle.ReusedBlocks, len(childBundle.Blocks))
	}
	if childBundle.Blocks[0].State.ChunkID != parentBundle.Blocks[0].State.ChunkID {
		t.Fatalf("grafted ref = %+v, want parent ref %+v", childBundle.Blocks[0], parentBundle.Blocks[0])
	}
	if childBundle.Blocks[0].KVHash != parentBundle.Blocks[0].KVHash {
		t.Fatalf("grafted hash = %q, want parent hash %q carried", childBundle.Blocks[0].KVHash, parentBundle.Blocks[0].KVHash)
	}
	loaded, err := LoadFromStateBlocksWithOptions(ctx, store, childBundle, LoadOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("LoadFromStateBlocksWithOptions(trusted bundle) error = %v", err)
	}
	if len(loaded.Tokens) != 4 {
		t.Fatalf("loaded tokens = %v, want full 4-token prefix", loaded.Tokens)
	}
}

func TestKVSnapshotStateBlocks_Good_TrustedBoundaryMatrix(t *testing.T) {
	parent := &StateBlockBundle{
		BlockSize:  2,
		TokenCount: 5,
		Blocks: []StateBlockRef{
			{Index: 0, TokenStart: 0, TokenCount: 2},
			{Index: 1, TokenStart: 2, TokenCount: 2},
			{Index: 2, TokenStart: 4, TokenCount: 1}, // partial tail — never grafted
		},
	}
	cases := []struct {
		name string
		opts StateBlockOptions
		size int
		want int
	}{
		{"untrusted", StateBlockOptions{ReusePrefix: parent}, 2, 0},
		{"trusted full", StateBlockOptions{ReusePrefix: parent, ReusePrefixTrusted: true}, 2, 4},
		{"trusted capped", StateBlockOptions{ReusePrefix: parent, ReusePrefixTrusted: true, ReusePrefixTokens: 3}, 2, 2},
		{"block size mismatch", StateBlockOptions{ReusePrefix: parent, ReusePrefixTrusted: true}, 4, 0},
		{"no parent", StateBlockOptions{ReusePrefixTrusted: true}, 2, 0},
	}
	for _, tc := range cases {
		if got := TrustedReuseBoundary(tc.opts, tc.size); got != tc.want {
			t.Errorf("%s: boundary = %d, want %d", tc.name, got, tc.want)
		}
	}
}

// TestBlocks_LoadKVSnapshotStateBlock_Good covers the unexported convenience
// wrapper loadKVSnapshotStateBlock (blocks.go), which forwards to
// LoadStateBlockWithOptions with default LoadOptions. A real saved block is
// loaded back and asserted equal to the canonical WithOptions result.
func TestBlocks_LoadKVSnapshotStateBlockWrapper(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	block, err := loadKVSnapshotStateBlock(ctx, store, bundle.Blocks[0])
	if err != nil {
		t.Fatalf("loadKVSnapshotStateBlock() error = %v", err)
	}
	if block.Index != 0 || block.TokenCount != 2 || block.Snapshot == nil {
		t.Fatalf("block = %+v, want first block with snapshot", block)
	}
	// Equivalence with the explicit-options entry point.
	viaOpts, err := LoadStateBlockWithOptions(ctx, store, bundle.Blocks[0], LoadOptions{})
	if err != nil || viaOpts.TokenCount != block.TokenCount || viaOpts.Index != block.Index {
		t.Fatalf("LoadStateBlockWithOptions() = %+v / %v, want match wrapper", viaOpts, err)
	}
}

// TestBlocks_PrefixLoaders_InvalidBundleWithStore_Bad covers the
// ValidateStateBlockBundle error-return blocks that the existing _Bad tests miss:
// LoadPrefixFromStateBlocksWithOptions and LoadPrefixTokensFromStateBlocksWithOptions.
// Both fire only when the bundle is invalid AND the store is non-nil — the prior
// tests pass either a nil store (short-circuits before validate) or a valid
// bundle (validate returns nil).
func TestBlocks_PrefixLoadersInvalidBundleWithStore(t *testing.T) {
	ctx := context.Background()
	store, _ := kvSnapshotBlocksTestBundle(t)

	if _, err := LoadPrefixFromStateBlocksWithOptions(ctx, store, &StateBlockBundle{}, 1, LoadOptions{}); err == nil {
		t.Fatal("LoadPrefixFromStateBlocksWithOptions(invalid bundle, valid store) error = nil, want validate error")
	}
	if _, err := LoadPrefixTokensFromStateBlocksWithOptions(ctx, store, &StateBlockBundle{}, 1, LoadOptions{}); err == nil {
		t.Fatal("LoadPrefixTokensFromStateBlocksWithOptions(invalid bundle, valid store) error = nil, want validate error")
	}
}

// TestBlocks_LoadPrefixPartial_Good drives loadAndAssembleStateBlockPrefix's
// mid-block trim body: a 3-token prefix over a 4-token / 2-block bundle covers
// the first whole block and trims the second to one token via SliceBlock, then
// assembles the partial result. This is the prompt-cache warmup-to-a-partial-
// prefix path the full-bundle and zero-prefix delegations never reach.
func TestBlocks_LoadPrefixPartialTrim(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	prefix, err := LoadPrefixFromStateBlocks(ctx, store, bundle, 3)
	if err != nil {
		t.Fatalf("LoadPrefixFromStateBlocks(3 of 4) error = %v", err)
	}
	if len(prefix.Tokens) != 3 {
		t.Fatalf("prefix tokens = %d (%v), want 3", len(prefix.Tokens), prefix.Tokens)
	}
	// A non-final prefix omits the terminal logits (ClearTerminalState ran).
	if len(prefix.Logits) != 0 {
		t.Fatalf("partial prefix Logits = %v, want cleared", prefix.Logits)
	}
}

// TestBlocks_AssembleStateBlocks_MetadataMismatch_Bad tampers a saved bundle's
// ref metadata so it still passes ValidateStateBlockBundle but diverges from the
// stored block, driving loadAndAssembleStateBlocks' post-load guards:
// errBlockMetadataMismatch (ref.TokenStart no longer matches the decoded block)
// and errBlocksNotContiguous (a zero TokenCount in the up-front order check).
func TestBlocks_AssembleStateBlocksMetadataMismatch(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	// errBlocksNotContiguous: zero the first block's TokenCount so the
	// up-front contiguity scan rejects it before any load.
	notContiguous := *bundle
	notContiguous.Blocks = append([]StateBlockRef(nil), bundle.Blocks...)
	notContiguous.Blocks[0].TokenCount = 0
	if _, err := LoadFromStateBlocks(ctx, store, &notContiguous); err != errBlocksNotContiguous {
		t.Fatalf("LoadFromStateBlocks(zero token count) error = %v, want errBlocksNotContiguous", err)
	}

	// errBlockMetadataMismatch: keep the bundle contiguous and well-ordered
	// (so the up-front scan passes) but shift the recorded TokenStart of both
	// refs by a constant. The decoded block envelopes still carry the original
	// starts, so the per-block metadata check trips after load.
	shifted := *bundle
	shifted.Blocks = append([]StateBlockRef(nil), bundle.Blocks...)
	for i := range shifted.Blocks {
		shifted.Blocks[i].TokenStart += 100
	}
	if _, err := LoadFromStateBlocks(ctx, store, &shifted); err == nil {
		t.Fatal("LoadFromStateBlocks(shifted token starts) error = nil, want metadata mismatch")
	}
}

// TestBlocks_StateBlockPrefixCoverage_Bad exercises stateBlockPrefixCoverage's
// guard arms directly with hand-built bundles: out-of-order index, a
// non-contiguous gap, and a prefix that no block covers (the requested prefix
// exceeds the summed token count of the covering blocks).
func TestBlocks_StateBlockPrefixCoverageGuards(t *testing.T) {
	// errPrefixNoCoveringBlocks: empty bundle.
	if _, err := stateBlockPrefixCoverage(&StateBlockBundle{}, 2); err != errPrefixNoCoveringBlocks {
		t.Fatalf("stateBlockPrefixCoverage(empty) = %v, want errPrefixNoCoveringBlocks", err)
	}

	// errBlocksOutOfOrder: first ref claims index 5.
	outOfOrder := &StateBlockBundle{
		TokenCount: 4,
		Blocks: []StateBlockRef{
			{Index: 5, TokenStart: 0, TokenCount: 2},
			{Index: 1, TokenStart: 2, TokenCount: 2},
		},
	}
	if _, err := stateBlockPrefixCoverage(outOfOrder, 4); err != errBlocksOutOfOrder {
		t.Fatalf("stateBlockPrefixCoverage(out of order) = %v, want errBlocksOutOfOrder", err)
	}

	// errBlocksNotContiguous: second ref starts past the running cursor.
	gap := &StateBlockBundle{
		TokenCount: 4,
		Blocks: []StateBlockRef{
			{Index: 0, TokenStart: 0, TokenCount: 2},
			{Index: 1, TokenStart: 3, TokenCount: 2}, // gap: expected start 2
		},
	}
	if _, err := stateBlockPrefixCoverage(gap, 4); err != errBlocksNotContiguous {
		t.Fatalf("stateBlockPrefixCoverage(gap) = %v, want errBlocksNotContiguous", err)
	}

	// errPrefixBlocksNoCover: a single 2-token block can't cover a 4-token
	// prefix — the loop exhausts the blocks with totalTokens < prefixTokens.
	short := &StateBlockBundle{
		TokenCount: 2,
		Blocks: []StateBlockRef{
			{Index: 0, TokenStart: 0, TokenCount: 2},
		},
	}
	if _, err := stateBlockPrefixCoverage(short, 4); err != errPrefixBlocksNoCover {
		t.Fatalf("stateBlockPrefixCoverage(short) = %v, want errPrefixBlocksNoCover", err)
	}

	// Good: a 2-block bundle covers a 3-token prefix with the first 2 blocks.
	ok := &StateBlockBundle{
		TokenCount: 4,
		Blocks: []StateBlockRef{
			{Index: 0, TokenStart: 0, TokenCount: 2},
			{Index: 1, TokenStart: 2, TokenCount: 2},
		},
	}
	if n, err := stateBlockPrefixCoverage(ok, 3); err != nil || n != 2 {
		t.Fatalf("stateBlockPrefixCoverage(3 of 4) = %d/%v, want 2 blocks", n, err)
	}
}

// TestBlocks_LoadResolveFailure_Bad drives the resolve-error arm of every block
// load entry point with a single store double whose Get always fails. A valid
// bundle (saved to a real store first) supplies well-formed refs, so the only
// failure is the resolve itself — covering the error returns in
// LoadStateBlockWithOptions, LoadStateBlockTokensWithOptions,
// loadAndAssembleStateBlocks and the prefix loaders at once. Both the Q8
// envelope path and the native raw path are exercised.
func TestBlocks_LoadResolveFailurePaths(t *testing.T) {
	ctx := context.Background()
	failing := failingGetStateStore{}

	// Q8 (envelope) bundle: LoadStateBlockWithOptions resolves via state.Resolve.
	_, q8Bundle := kvSnapshotBlocksTestBundle(t)
	if _, err := LoadStateBlockWithOptions(ctx, failing, q8Bundle.Blocks[0], LoadOptions{}); err == nil {
		t.Fatal("LoadStateBlockWithOptions(failing store) error = nil, want resolve error")
	}
	if _, err := LoadStateBlockTokensWithOptions(ctx, failing, q8Bundle.Blocks[0], LoadOptions{}); err == nil {
		t.Fatal("LoadStateBlockTokensWithOptions(failing store) error = nil, want resolve error")
	}
	if _, err := LoadFromStateBlocks(ctx, failing, q8Bundle); err == nil {
		t.Fatal("LoadFromStateBlocks(failing store) error = nil, want resolve error")
	}
	if _, err := LoadPrefixFromStateBlocks(ctx, failing, q8Bundle, 2); err == nil {
		t.Fatal("LoadPrefixFromStateBlocks(failing store) error = nil, want resolve error")
	}
	if _, err := LoadPrefixTokensFromStateBlocks(ctx, failing, q8Bundle, 2); err == nil {
		t.Fatal("LoadPrefixTokensFromStateBlocks(failing store) error = nil, want resolve error")
	}

	// Native (raw payload) bundle: the raw load path resolves via
	// state.BorrowRefBytes, which also falls through to Get.
	nativeStore := state.NewInMemoryStore(nil)
	nativeBundle, err := kvSnapshotBlocksTestSnapshot().SaveStateBlocks(ctx, nativeStore, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingNative,
		URI:        "mlx://session/native-resolve",
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks(native) error = %v", err)
	}
	if nativeBundle.Blocks[0].PayloadEncoding != kvSnapshotStatePayloadRaw {
		t.Fatalf("native block payload encoding = %q, want raw", nativeBundle.Blocks[0].PayloadEncoding)
	}
	if _, err := LoadStateBlockWithOptions(ctx, failing, nativeBundle.Blocks[0], LoadOptions{}); err == nil {
		t.Fatal("LoadStateBlockWithOptions(raw, failing store) error = nil, want resolve error")
	}
	if _, err := LoadStateBlockTokensWithOptions(ctx, failing, nativeBundle.Blocks[0], LoadOptions{}); err == nil {
		t.Fatal("LoadStateBlockTokensWithOptions(raw, failing store) error = nil, want resolve error")
	}
}

// TestBlocks_ReusableStateBlockRef_Miss covers reusableKVSnapshotStateBlockRef's
// non-reuse arms (the cache-miss lane): a parent with a mismatched KVEncoding,
// a block whose range falls outside the reuse limit, and an untrusted child
// whose hashed content diverges from the parent at the same range so the
// hash-match loop falls through to "no reuse". Each returns ok=false.
func TestBlocks_ReusableStateBlockRef_Miss(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	parent := kvSnapshotBlocksTestSnapshot()
	parentBundle, err := parent.SaveStateBlocks(ctx, store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingNative,
		URI:        "mlx://reuse-miss-parent",
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks(parent) error = %v", err)
	}

	block := Block{Index: 0, TokenStart: 0, TokenCount: 2, Snapshot: kvSnapshotBlocksTestSnapshot()}

	// Encoding mismatch: parent recorded EncodingNative, ask for Q8 → no reuse.
	if _, _, ok, err := reusableKVSnapshotStateBlockRef(block, StateBlockOptions{ReusePrefix: parentBundle}, EncodingQ8); err != nil || ok {
		t.Fatalf("reusable(encoding mismatch) = ok %v / err %v, want no reuse", ok, err)
	}

	// Out-of-limit range: a block ending past ReusePrefixTokens cannot reuse.
	outOfLimit := Block{Index: 0, TokenStart: 0, TokenCount: 4, Snapshot: kvSnapshotBlocksTestSnapshot()}
	if _, _, ok, err := reusableKVSnapshotStateBlockRef(outOfLimit, StateBlockOptions{ReusePrefix: parentBundle, ReusePrefixTokens: 2}, EncodingNative); err != nil || ok {
		t.Fatalf("reusable(out of limit) = ok %v / err %v, want no reuse", ok, err)
	}

	// Hash divergence: an untrusted child whose block content differs from the
	// parent at the same range hashes to a non-matching digest → fall through.
	diverged := kvSnapshotBlocksTestSnapshot()
	diverged.Layers[0].Heads[0].Key[0] = 42 // perturb the captured K so the hash differs
	divergedBlock := Block{Index: 0, TokenStart: 0, TokenCount: 2, Snapshot: diverged}
	_, hash, ok, err := reusableKVSnapshotStateBlockRef(divergedBlock, StateBlockOptions{ReusePrefix: parentBundle}, EncodingNative)
	if err != nil || ok {
		t.Fatalf("reusable(hash diverged) = ok %v / err %v, want no reuse", ok, err)
	}
	if hash == "" {
		t.Fatal("reusable(hash diverged) returned empty hash, want the computed digest")
	}

	// Nil parent / empty parent both short-circuit to no reuse.
	if _, _, ok, _ := reusableKVSnapshotStateBlockRef(block, StateBlockOptions{}, EncodingNative); ok {
		t.Fatal("reusable(nil parent) = ok true, want no reuse")
	}
}

// TestBlocks_AssembleBlocks_Mismatch_Bad drives appendKVSnapshotBlock's
// consistency guards by assembling a valid first block with a deliberately
// divergent second block: architecture, shape (HeadDim/NumHeads/NumLayers),
// layer count, per-layer cache-mode, MaxSize, and head-count mismatches each
// surface their specific error. block[0] establishes the assembled skeleton, so
// the mutated block[1] trips the guard during the fold.
func TestBlocks_AssembleBlocks_Mismatch_Bad(t *testing.T) {
	// twoBlocks returns a fresh, valid, contiguous 2-block pair (4 tokens).
	twoBlocks := func() []Block {
		blocks, err := kvSnapshotBlocksTestSnapshot().SplitBlocks(2)
		if err != nil {
			t.Fatalf("SplitBlocks() error = %v", err)
		}
		if len(blocks) != 2 {
			t.Fatalf("SplitBlocks() = %d blocks, want 2", len(blocks))
		}
		return blocks
	}

	cases := []struct {
		name    string
		perturb func(second *Snapshot)
		want    error
	}{
		{"arch", func(s *Snapshot) { s.Architecture = "different_model" }, errBlockArchMismatch},
		{"headDim", func(s *Snapshot) { s.HeadDim++ }, errBlockShapeMismatch},
		{"numHeads", func(s *Snapshot) { s.NumHeads++ }, errBlockShapeMismatch},
		{"numLayers", func(s *Snapshot) { s.NumLayers++ }, errBlockShapeMismatch},
		{"cacheMode", func(s *Snapshot) { s.Layers[0].CacheMode = "turboquant" }, errBlockMetadataMismatch},
		{"maxSize", func(s *Snapshot) { s.Layers[0].MaxSize = 4096 }, errBlockMetadataMismatch},
		{"headCount", func(s *Snapshot) {
			s.Layers[0].Heads = append(s.Layers[0].Heads, HeadSnapshot{Key: []float32{1}, Value: []float32{1}})
		}, errBlockHeadCountMismatch},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			blocks := twoBlocks()
			// Establish the skeleton from block[0]; set its cache mode/max size
			// so the per-layer mismatch guards have a non-empty value to clash
			// against (otherwise the divergent value just gets adopted).
			if tc.want == errBlockMetadataMismatch {
				blocks[0].Snapshot.Layers[0].CacheMode = "fp16"
				blocks[0].Snapshot.Layers[0].MaxSize = 2048
			}
			tc.perturb(blocks[1].Snapshot)
			if _, err := AssembleBlocks(blocks); err != tc.want {
				t.Fatalf("AssembleBlocks(%s mismatch) error = %v, want %v", tc.name, err, tc.want)
			}
		})
	}
}

// TestBlocks_AppendLayerRawBlock_Bad drives appendKVSnapshotLayerRawBlock's
// shape + dtype guards directly: an unsupported dtype, a non-4D shape, a byte
// length that disagrees with the shape, a dtype that changes between arrivals,
// and a second-arrival shape whose B/H/D dims diverge from the first. The
// no-op empty-raw path is the Good anchor.
func TestBlocks_AppendLayerRawBlockGuards(t *testing.T) {
	// Good: empty raw is a no-op that leaves the destination untouched.
	var dt string
	var by []byte
	var sh []int32
	if err := appendKVSnapshotLayerRawBlock(&dt, &by, &sh, "float16", nil, []int32{1, 1, 2, 1}); err != nil {
		t.Fatalf("appendKVSnapshotLayerRawBlock(empty raw) error = %v, want nil no-op", err)
	}

	raw := cvtRawF16(2, 1) // 2 f16 values = 4 bytes, shape [1,1,2,1]
	good := []int32{1, 1, 2, 1}

	// Unsupported dtype.
	if err := appendKVSnapshotLayerRawBlock(&dt, &by, &sh, "nonsense", raw, good); err != errUnsupportedLayerRawTensor {
		t.Fatalf("append(bad dtype) = %v, want errUnsupportedLayerRawTensor", err)
	}
	// Non-4D shape.
	if err := appendKVSnapshotLayerRawBlock(&dt, &by, &sh, "float16", raw, []int32{2, 1}); err != errUnsupportedLayerRawTensor {
		t.Fatalf("append(non-4D shape) = %v, want errUnsupportedLayerRawTensor", err)
	}
	// Byte length disagrees with shape (shape claims 4 values, raw has 2).
	if err := appendKVSnapshotLayerRawBlock(&dt, &by, &sh, "float16", raw, []int32{1, 1, 4, 1}); err != errLayerRawTensorShape {
		t.Fatalf("append(len mismatch) = %v, want errLayerRawTensorShape", err)
	}

	// First valid arrival establishes dtype + shape.
	var dDType string
	var dBytes []byte
	var dShape []int32
	if err := appendKVSnapshotLayerRawBlock(&dDType, &dBytes, &dShape, "float16", raw, good); err != nil {
		t.Fatalf("append(first arrival) error = %v", err)
	}
	// Dtype change on a subsequent arrival.
	if err := appendKVSnapshotLayerRawBlock(&dDType, &dBytes, &dShape, "bfloat16", raw, good); err != errLayerRawDtypeMismatch {
		t.Fatalf("append(dtype change) = %v, want errLayerRawDtypeMismatch", err)
	}
	// Second-arrival B/H/D divergence (D goes from 1 to 2).
	raw2 := cvtRawF16(2, 2) // 4 values, shape [1,1,2,2]
	if err := appendKVSnapshotLayerRawBlock(&dDType, &dBytes, &dShape, "float16", raw2, []int32{1, 1, 2, 2}); err != errLayerRawTensorShape {
		t.Fatalf("append(dim divergence) = %v, want errLayerRawTensorShape", err)
	}
}

// TestBlocks_SliceBlock_Bad covers sliceBlockInternal's range guard and the
// compressed-payload full-range requirement: an inverted or out-of-bounds range
// is rejected, and a snapshot carrying TurboQuant payloads refuses a partial
// slice (compressed blocks must be taken whole).
func TestBlocks_SliceBlock_Bad(t *testing.T) {
	snapshot := kvSnapshotBlocksTestSnapshot()

	// Inverted range (end <= start).
	if _, err := snapshot.SliceBlock(2, 2, 0, false); err != errBlockRangeInvalid {
		t.Fatalf("SliceBlock(2,2) = %v, want errBlockRangeInvalid", err)
	}
	// Out-of-bounds end.
	if _, err := snapshot.SliceBlock(0, len(snapshot.Tokens)+1, 0, false); err != errBlockRangeInvalid {
		t.Fatalf("SliceBlock(over end) = %v, want errBlockRangeInvalid", err)
	}

	// A compressed-payload layer refuses a partial-range slice.
	compressed := kvSnapshotBlocksTestSnapshot()
	compressed.Layers[0].TurboQuantPayloads = [][]byte{{1, 2, 3, 4}}
	if _, err := compressed.SliceBlock(0, 2, 0, false); err != errBlockCompressedPayloadSplit {
		t.Fatalf("SliceBlock(compressed partial) = %v, want errBlockCompressedPayloadSplit", err)
	}
	// The same compressed snapshot sliced at its full range succeeds.
	if _, err := compressed.SliceBlock(0, len(compressed.Tokens), 0, true); err != nil {
		t.Fatalf("SliceBlock(compressed full range) error = %v, want success", err)
	}
}

// TestBlocks_SplitBlocks_Guards_Bad drives walkBlocks' precondition guards via
// the public SplitBlocks: a nil receiver, a non-positive block size, a snapshot
// whose token count disagrees with its effective sequence length, and one with
// no head dimension each return a specific guard error before any slicing.
func TestBlocks_SplitBlocks_Guards_Bad(t *testing.T) {
	// Nil receiver.
	if _, err := (*Snapshot)(nil).SplitBlocks(2); err != errSnapshotNil {
		t.Fatalf("SplitBlocks(nil) = %v, want errSnapshotNil", err)
	}
	// Non-positive block size.
	if _, err := kvSnapshotBlocksTestSnapshot().SplitBlocks(0); err != errBlockSizeTooSmall {
		t.Fatalf("SplitBlocks(0) = %v, want errBlockSizeTooSmall", err)
	}
	// Token count disagrees with SeqLen.
	mismatch := kvSnapshotBlocksTestSnapshot()
	mismatch.Tokens = mismatch.Tokens[:1] // SeqLen still 4
	if _, err := mismatch.SplitBlocks(2); err != errBlockSplitNeedsTokens {
		t.Fatalf("SplitBlocks(token/seqlen mismatch) = %v, want errBlockSplitNeedsTokens", err)
	}
	// No head dimension.
	noHeadDim := kvSnapshotBlocksTestSnapshot()
	noHeadDim.HeadDim = 0
	if _, err := noHeadDim.SplitBlocks(2); err != errBlockSplitNeedsHeadDim {
		t.Fatalf("SplitBlocks(no head dim) = %v, want errBlockSplitNeedsHeadDim", err)
	}
}

// TestBlocks_LoadPrefixNative_Partial_Good drives the native (layer-raw) prefix
// assembly path: a native-dtype bundle loaded to a partial prefix exercises
// loadAndAssembleStateBlockPrefix's SliceBlock-trim plus the raw layer-slab
// assembly arms (the float32 partial-prefix test never touches the native code).
func TestBlocks_LoadPrefixNativePartialAssembly(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	bundle, err := kvSnapshotBlocksTestSnapshot().SaveStateBlocks(ctx, store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingNative,
		URI:        "mlx://native-prefix",
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks(native) error = %v", err)
	}

	prefix, err := LoadPrefixFromStateBlocksWithOptions(ctx, store, bundle, 3, LoadOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("LoadPrefixFromStateBlocksWithOptions(native, 3 of 4) error = %v", err)
	}
	if len(prefix.Tokens) != 3 {
		t.Fatalf("native prefix tokens = %d, want 3", len(prefix.Tokens))
	}
}

// --- blocks.go canonical AX-7 triplets -------------------------------------

// TestBlocks_Snapshot_SplitBlocks_Good splits the four-token fixture into two
// blocks and asserts the block metadata and per-block token/tensor slices.
func TestBlocks_Snapshot_SplitBlocks_Good(t *testing.T) {
	snapshot := kvSnapshotBlocksTestSnapshot()

	blocks, err := snapshot.SplitBlocks(2)
	if err != nil {
		t.Fatalf("SplitBlocks() error = %v", err)
	}
	if len(blocks) != 2 {
		t.Fatalf("blocks len = %d, want 2", len(blocks))
	}
	if blocks[0].Index != 0 || blocks[0].TokenStart != 0 || blocks[0].TokenCount != 2 {
		t.Fatalf("block[0] metadata = %+v", blocks[0])
	}
	if got := blocks[0].Snapshot.Tokens; len(got) != 2 || got[0] != 1 || got[1] != 2 {
		t.Fatalf("block[0] tokens = %v, want [1 2]", got)
	}
	if got := blocks[1].Snapshot.Layers[0].Heads[0].Value; len(got) != 4 || got[0] != 24 || got[3] != 27 {
		t.Fatalf("block[1] value = %v, want second token range", got)
	}
}

// TestBlocks_Snapshot_SplitBlocks_Bad drives SplitBlocks' precondition guards: a
// nil receiver, a non-positive block size, a token/seq-len mismatch, and a
// missing head dimension each return a specific guard error before any slicing.
func TestBlocks_Snapshot_SplitBlocks_Bad(t *testing.T) {
	if _, err := (*Snapshot)(nil).SplitBlocks(2); err != errSnapshotNil {
		t.Fatalf("SplitBlocks(nil) = %v, want errSnapshotNil", err)
	}
	if _, err := kvSnapshotBlocksTestSnapshot().SplitBlocks(0); err != errBlockSizeTooSmall {
		t.Fatalf("SplitBlocks(0) = %v, want errBlockSizeTooSmall", err)
	}
	mismatch := kvSnapshotBlocksTestSnapshot()
	mismatch.Tokens = mismatch.Tokens[:1] // SeqLen still 4
	if _, err := mismatch.SplitBlocks(2); err != errBlockSplitNeedsTokens {
		t.Fatalf("SplitBlocks(token/seqlen mismatch) = %v, want errBlockSplitNeedsTokens", err)
	}
	noHeadDim := kvSnapshotBlocksTestSnapshot()
	noHeadDim.HeadDim = 0
	if _, err := noHeadDim.SplitBlocks(2); err != errBlockSplitNeedsHeadDim {
		t.Fatalf("SplitBlocks(no head dim) = %v, want errBlockSplitNeedsHeadDim", err)
	}
}

// TestBlocks_Snapshot_SplitBlocks_Ugly splits a compressed (TurboQuant) snapshot
// whose payloads cannot be partially sliced: SplitBlocks must keep the whole
// snapshot in a single block rather than cutting the compressed layer.
func TestBlocks_Snapshot_SplitBlocks_Ugly(t *testing.T) {
	snapshot := kvSnapshotBlocksTestSnapshot()
	snapshot.Layers[0].CacheMode = "turboquant"
	snapshot.Layers[0].TurboQuantPayloads = [][]byte{{1, 2, 3, 4}}
	snapshot.Layers[0].Heads = nil

	blocks, err := snapshot.SplitBlocks(2)
	if err != nil {
		t.Fatalf("SplitBlocks(turboquant) error = %v", err)
	}
	if len(blocks) != 1 || blocks[0].TokenCount != len(snapshot.Tokens) {
		t.Fatalf("blocks = %+v, want one whole compressed block", blocks)
	}
}

// TestBlocks_Snapshot_RangeBlocks_Good iterates blocks and asserts RangeBlocks
// visits them in index order until the yield callback returns false.
func TestBlocks_Snapshot_RangeBlocks_Good(t *testing.T) {
	snapshot := kvSnapshotBlocksTestSnapshot()
	seen := []int{}

	err := snapshot.RangeBlocks(1, func(block Block) bool {
		seen = append(seen, block.Index)
		return len(seen) < 2
	})
	if err != nil {
		t.Fatalf("RangeBlocks() error = %v", err)
	}
	if len(seen) != 2 || seen[0] != 0 || seen[1] != 1 {
		t.Fatalf("seen blocks = %v, want [0 1]", seen)
	}
}

// TestBlocks_Snapshot_RangeBlocks_Bad asserts RangeBlocks surfaces the same
// precondition guard as SplitBlocks: a non-positive block size is rejected.
func TestBlocks_Snapshot_RangeBlocks_Bad(t *testing.T) {
	snapshot := kvSnapshotBlocksTestSnapshot()

	err := snapshot.RangeBlocks(0, func(Block) bool { return true })
	if err != errBlockSizeTooSmall {
		t.Fatalf("RangeBlocks(0) = %v, want errBlockSizeTooSmall", err)
	}
}

// TestBlocks_Snapshot_RangeBlocks_Ugly asserts RangeBlocks reports the nil-receiver
// guard rather than panicking.
func TestBlocks_Snapshot_RangeBlocks_Ugly(t *testing.T) {
	if err := (*Snapshot)(nil).RangeBlocks(2, func(Block) bool { return true }); err != errSnapshotNil {
		t.Fatalf("RangeBlocks(nil) = %v, want errSnapshotNil", err)
	}
}

// TestBlocks_Snapshot_SliceBlock_Good slices the first two-token window of the
// fixture and asserts the slice carries exactly that token range.
func TestBlocks_Snapshot_SliceBlock_Good(t *testing.T) {
	snapshot := kvSnapshotBlocksTestSnapshot()

	slice, err := snapshot.SliceBlock(0, 2, 0, false)
	if err != nil {
		t.Fatalf("SliceBlock(0,2) error = %v", err)
	}
	if len(slice.Tokens) != 2 || slice.Tokens[0] != 1 || slice.Tokens[1] != 2 {
		t.Fatalf("SliceBlock(0,2) tokens = %v, want first two", slice.Tokens)
	}
}

// TestBlocks_Snapshot_SliceBlock_Bad drives SliceBlock's range guards: an
// inverted range and an out-of-bounds end both return errBlockRangeInvalid.
func TestBlocks_Snapshot_SliceBlock_Bad(t *testing.T) {
	snapshot := kvSnapshotBlocksTestSnapshot()

	if _, err := snapshot.SliceBlock(2, 2, 0, false); err != errBlockRangeInvalid {
		t.Fatalf("SliceBlock(2,2) = %v, want errBlockRangeInvalid", err)
	}
	if _, err := snapshot.SliceBlock(0, len(snapshot.Tokens)+1, 0, false); err != errBlockRangeInvalid {
		t.Fatalf("SliceBlock(over end) = %v, want errBlockRangeInvalid", err)
	}
}

// TestBlocks_Snapshot_SliceBlock_Ugly asserts a compressed-payload layer refuses
// a partial-range slice but accepts a full-range slice.
func TestBlocks_Snapshot_SliceBlock_Ugly(t *testing.T) {
	compressed := kvSnapshotBlocksTestSnapshot()
	compressed.Layers[0].TurboQuantPayloads = [][]byte{{1, 2, 3, 4}}

	if _, err := compressed.SliceBlock(0, 2, 0, false); err != errBlockCompressedPayloadSplit {
		t.Fatalf("SliceBlock(compressed partial) = %v, want errBlockCompressedPayloadSplit", err)
	}
	if _, err := compressed.SliceBlock(0, len(compressed.Tokens), 0, true); err != nil {
		t.Fatalf("SliceBlock(compressed full range) error = %v, want success", err)
	}
}

// TestBlocks_ValidateStateBlockBundle_Good asserts a freshly saved bundle passes
// ValidateStateBlockBundle.
func TestBlocks_ValidateStateBlockBundle_Good(t *testing.T) {
	_, bundle := kvSnapshotBlocksTestBundle(t)

	if err := ValidateStateBlockBundle(bundle); err != nil {
		t.Fatalf("ValidateStateBlockBundle(valid) error = %v", err)
	}
}

// TestBlocks_ValidateStateBlockBundle_Bad asserts ValidateStateBlockBundle
// rejects an empty bundle (zero version, blank kind, no blocks).
func TestBlocks_ValidateStateBlockBundle_Bad(t *testing.T) {
	if err := ValidateStateBlockBundle(&StateBlockBundle{}); err == nil {
		t.Fatal("ValidateStateBlockBundle(empty) error = nil, want validation error")
	}
}

// TestBlocks_ValidateStateBlockBundle_Ugly asserts ValidateStateBlockBundle
// rejects a nil bundle pointer rather than dereferencing it.
func TestBlocks_ValidateStateBlockBundle_Ugly(t *testing.T) {
	if err := ValidateStateBlockBundle(nil); err != errBundleNil {
		t.Fatalf("ValidateStateBlockBundle(nil) = %v, want errBundleNil", err)
	}
}

// TestBlocks_ValidateMemvidBlockBundle_Good asserts the deprecated
// ValidateMemvidBlockBundle alias passes a valid bundle.
func TestBlocks_ValidateMemvidBlockBundle_Good(t *testing.T) {
	_, bundle := kvSnapshotBlocksTestBundle(t)

	if err := ValidateMemvidBlockBundle(bundle); err != nil {
		t.Fatalf("ValidateMemvidBlockBundle(valid) error = %v", err)
	}
}

// TestBlocks_ValidateMemvidBlockBundle_Bad asserts the deprecated
// ValidateMemvidBlockBundle alias rejects an empty bundle.
func TestBlocks_ValidateMemvidBlockBundle_Bad(t *testing.T) {
	if err := ValidateMemvidBlockBundle(&MemvidBlockBundle{}); err == nil {
		t.Fatal("ValidateMemvidBlockBundle(empty) error = nil, want validation error")
	}
}

// TestBlocks_ValidateMemvidBlockBundle_Ugly asserts the deprecated
// ValidateMemvidBlockBundle alias rejects a nil bundle.
func TestBlocks_ValidateMemvidBlockBundle_Ugly(t *testing.T) {
	if err := ValidateMemvidBlockBundle(nil); err != errBundleNil {
		t.Fatalf("ValidateMemvidBlockBundle(nil) = %v, want errBundleNil", err)
	}
}

// TestBlocks_ClearTerminalState_Good asserts ClearTerminalState strips the
// generated tokens, logit shape and logits from a snapshot.
func TestBlocks_ClearTerminalState_Good(t *testing.T) {
	snapshot := kvSnapshotBlocksTestSnapshot()

	ClearTerminalState(snapshot)

	if snapshot.Generated != nil || snapshot.LogitShape != nil || snapshot.Logits != nil {
		t.Fatalf("ClearTerminalState() = generated %v / logitShape %v / logits %v, want all nil", snapshot.Generated, snapshot.LogitShape, snapshot.Logits)
	}
}

// TestBlocks_ClearTerminalState_Bad asserts ClearTerminalState leaves the
// non-terminal fields (tokens, layers) intact while clearing terminal state.
func TestBlocks_ClearTerminalState_Bad(t *testing.T) {
	snapshot := kvSnapshotBlocksTestSnapshot()

	ClearTerminalState(snapshot)

	if len(snapshot.Tokens) != 4 || len(snapshot.Layers) != 1 {
		t.Fatalf("ClearTerminalState() removed non-terminal data: tokens %v layers %d", snapshot.Tokens, len(snapshot.Layers))
	}
}

// TestBlocks_ClearTerminalState_Ugly asserts ClearTerminalState is a safe no-op
// on a nil snapshot.
func TestBlocks_ClearTerminalState_Ugly(t *testing.T) {
	ClearTerminalState(nil)

	// A snapshot that already has no terminal state stays empty-safe.
	bare := &Snapshot{Tokens: []int32{1}}
	ClearTerminalState(bare)
	if bare.Generated != nil || bare.Logits != nil {
		t.Fatalf("ClearTerminalState(bare) = %+v, want terminal fields nil", bare)
	}
}

// TestBlocks_LoadStateBlockWithOptions_Good loads the first block of the fixture
// bundle and asserts the recovered block metadata + snapshot, matching the
// unexported wrapper.
func TestBlocks_LoadStateBlockWithOptions_Good(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	block, err := LoadStateBlockWithOptions(ctx, store, bundle.Blocks[0], LoadOptions{})
	if err != nil {
		t.Fatalf("LoadStateBlockWithOptions() error = %v", err)
	}
	if block.Index != 0 || block.TokenCount != 2 || block.Snapshot == nil {
		t.Fatalf("LoadStateBlockWithOptions() block = %+v, want first block with snapshot", block)
	}
}

// TestBlocks_LoadStateBlockWithOptions_Bad asks LoadStateBlockWithOptions to
// resolve a block ref that points at no chunk; the resolve must fail.
func TestBlocks_LoadStateBlockWithOptions_Bad(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)

	if _, err := LoadStateBlockWithOptions(ctx, store, StateBlockRef{State: state.ChunkRef{ChunkID: 9999}}, LoadOptions{}); err == nil {
		t.Fatal("LoadStateBlockWithOptions(missing chunk) error = nil, want resolve error")
	}
}

// TestBlocks_LoadStateBlockWithOptions_Ugly feeds LoadStateBlockWithOptions a
// block ref whose recorded KV hash does not match the stored payload, tripping
// the envelope hash guard.
func TestBlocks_LoadStateBlockWithOptions_Ugly(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	tampered := bundle.Blocks[0]
	tampered.KVHash = "sha256:not-the-real-hash"
	if _, err := LoadStateBlockWithOptions(ctx, store, tampered, LoadOptions{}); err == nil {
		t.Fatal("LoadStateBlockWithOptions(hash mismatch) error = nil, want hash error")
	}
}

// TestBlocks_LoadMemvidBlockWithOptions_Good asserts the deprecated
// LoadMemvidBlockWithOptions alias loads the first fixture block.
func TestBlocks_LoadMemvidBlockWithOptions_Good(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	block, err := LoadMemvidBlockWithOptions(ctx, store, bundle.Blocks[0], LoadOptions{})
	if err != nil || block.TokenCount != 2 {
		t.Fatalf("LoadMemvidBlockWithOptions() = %+v, err = %v, want first block", block, err)
	}
}

// TestBlocks_LoadMemvidBlockWithOptions_Bad asks the deprecated alias to resolve
// a missing chunk; the resolve must fail.
func TestBlocks_LoadMemvidBlockWithOptions_Bad(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)

	if _, err := LoadMemvidBlockWithOptions(ctx, store, StateBlockRef{State: state.ChunkRef{ChunkID: 4242}}, LoadOptions{}); err == nil {
		t.Fatal("LoadMemvidBlockWithOptions(missing chunk) error = nil, want resolve error")
	}
}

// TestBlocks_LoadMemvidBlockWithOptions_Ugly feeds the deprecated alias a ref
// whose hash does not match the stored payload, tripping the envelope guard.
func TestBlocks_LoadMemvidBlockWithOptions_Ugly(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	tampered := bundle.Blocks[0]
	tampered.KVHash = "sha256:wrong"
	if _, err := LoadMemvidBlockWithOptions(ctx, store, tampered, LoadOptions{}); err == nil {
		t.Fatal("LoadMemvidBlockWithOptions(hash mismatch) error = nil, want hash error")
	}
}

// TestBlocks_LoadStateBlockTokens_Good loads only the token IDs of the first
// fixture block (no K/V assembly).
func TestBlocks_LoadStateBlockTokens_Good(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	block, err := LoadStateBlockTokens(ctx, store, bundle.Blocks[0])
	if err != nil {
		t.Fatalf("LoadStateBlockTokens() error = %v", err)
	}
	if block.TokenCount != 2 || block.Index != 0 || len(block.Tokens) != 2 || block.Tokens[0] != 1 {
		t.Fatalf("LoadStateBlockTokens() block = %+v, want first two token IDs", block)
	}
}

// TestBlocks_LoadStateBlockTokens_Bad asks LoadStateBlockTokens to resolve a
// block ref with no backing chunk; the resolve must fail.
func TestBlocks_LoadStateBlockTokens_Bad(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)

	if _, err := LoadStateBlockTokens(ctx, store, StateBlockRef{State: state.ChunkRef{ChunkID: 7777}}); err == nil {
		t.Fatal("LoadStateBlockTokens(missing chunk) error = nil, want resolve error")
	}
}

// TestBlocks_LoadStateBlockTokens_Ugly feeds LoadStateBlockTokens a ref whose
// recorded hash mismatches the stored block envelope, tripping the hash guard.
func TestBlocks_LoadStateBlockTokens_Ugly(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	tampered := bundle.Blocks[0]
	tampered.KVHash = "sha256:nope"
	if _, err := LoadStateBlockTokens(ctx, store, tampered); err == nil {
		t.Fatal("LoadStateBlockTokens(hash mismatch) error = nil, want hash error")
	}
}

// TestBlocks_LoadStateBlockTokensWithOptions_Good loads the second fixture
// block's tokens with options and asserts the token-start and IDs.
func TestBlocks_LoadStateBlockTokensWithOptions_Good(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	block, err := LoadStateBlockTokensWithOptions(ctx, store, bundle.Blocks[1], LoadOptions{})
	if err != nil {
		t.Fatalf("LoadStateBlockTokensWithOptions() error = %v", err)
	}
	if block.TokenStart != 2 || len(block.Tokens) != 2 || block.Tokens[0] != 3 {
		t.Fatalf("LoadStateBlockTokensWithOptions() block = %+v, want second block tokens", block)
	}
}

// TestBlocks_LoadStateBlockTokensWithOptions_Bad asks the loader to resolve a
// ref with no backing chunk; the resolve must fail.
func TestBlocks_LoadStateBlockTokensWithOptions_Bad(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)

	if _, err := LoadStateBlockTokensWithOptions(ctx, store, StateBlockRef{State: state.ChunkRef{ChunkID: 5555}}, LoadOptions{}); err == nil {
		t.Fatal("LoadStateBlockTokensWithOptions(missing chunk) error = nil, want resolve error")
	}
}

// TestBlocks_LoadStateBlockTokensWithOptions_Ugly feeds the loader a ref whose
// hash mismatches the stored envelope, tripping the hash guard.
func TestBlocks_LoadStateBlockTokensWithOptions_Ugly(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	tampered := bundle.Blocks[1]
	tampered.KVHash = "sha256:bad"
	if _, err := LoadStateBlockTokensWithOptions(ctx, store, tampered, LoadOptions{}); err == nil {
		t.Fatal("LoadStateBlockTokensWithOptions(hash mismatch) error = nil, want hash error")
	}
}

// TestBlocks_StateBlockChunkRef_Good asserts StateBlockChunkRef returns the
// State ref when it is populated.
func TestBlocks_StateBlockChunkRef_Good(t *testing.T) {
	ref := StateBlockRef{State: state.ChunkRef{ChunkID: 42}}

	got := StateBlockChunkRef(ref)
	if got.ChunkID != 42 {
		t.Fatalf("StateBlockChunkRef(state set) = %+v, want State ref chunk 42", got)
	}
}

// TestBlocks_StateBlockChunkRef_Bad asserts StateBlockChunkRef falls back to the
// Memvid ref when the State ref is entirely zero.
func TestBlocks_StateBlockChunkRef_Bad(t *testing.T) {
	ref := StateBlockRef{Memvid: state.ChunkRef{ChunkID: 7}}

	got := StateBlockChunkRef(ref)
	if got.ChunkID != 7 {
		t.Fatalf("StateBlockChunkRef(only memvid) = %+v, want Memvid ref chunk 7", got)
	}
}

// TestBlocks_StateBlockChunkRef_Ugly asserts StateBlockChunkRef returns a zero
// ref when neither State nor Memvid is populated.
func TestBlocks_StateBlockChunkRef_Ugly(t *testing.T) {
	got := StateBlockChunkRef(StateBlockRef{})
	if got != (state.ChunkRef{}) {
		t.Fatalf("StateBlockChunkRef(empty) = %+v, want zero ChunkRef", got)
	}
}

// TestBlocks_EffectiveSeqLen_Good asserts EffectiveSeqLen returns the populated
// SeqLen field when set.
func TestBlocks_EffectiveSeqLen_Good(t *testing.T) {
	if got := EffectiveSeqLen(&Snapshot{SeqLen: 9}); got != 9 {
		t.Fatalf("EffectiveSeqLen(SeqLen=9) = %d, want 9", got)
	}
}

// TestBlocks_EffectiveSeqLen_Bad asserts EffectiveSeqLen falls back to the token
// count when SeqLen is zero.
func TestBlocks_EffectiveSeqLen_Bad(t *testing.T) {
	if got := EffectiveSeqLen(&Snapshot{Tokens: []int32{1, 2, 3}}); got != 3 {
		t.Fatalf("EffectiveSeqLen(zero SeqLen) = %d, want token count 3", got)
	}
}

// TestBlocks_EffectiveSeqLen_Ugly asserts EffectiveSeqLen returns 0 for a nil
// snapshot rather than panicking.
func TestBlocks_EffectiveSeqLen_Ugly(t *testing.T) {
	if got := EffectiveSeqLen(nil); got != 0 {
		t.Fatalf("EffectiveSeqLen(nil) = %d, want 0", got)
	}
}
