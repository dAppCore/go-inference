// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"math"
	"testing"

	"dappco.re/go/inference/kv"
)

func seededKVBytes(n int, seed uint32) []byte {
	out := make([]byte, n)
	for i := range out {
		seed = seed*1664525 + 1013904223
		out[i] = byte(seed >> 24)
	}
	return out
}

func TestArchSessionCaptureKV_Nil(t *testing.T) {
	var session *ArchSession
	if _, err := session.CaptureKV(); err == nil {
		t.Fatal("CaptureKV accepted a nil session")
	}
}

func TestArchSessionCaptureKVWithOptions_InvalidBlockStart(t *testing.T) {
	session := &ArchSession{}
	if _, err := session.CaptureKVWithOptions(kv.CaptureOptions{BlockStartToken: -1}); err == nil {
		t.Fatal("CaptureKVWithOptions accepted a negative block start")
	}
}

func TestArchSessionKVBlockSource_Nil(t *testing.T) {
	var session *ArchSession
	if _, err := session.KVBlockSource(4, kv.CaptureOptions{}); err == nil {
		t.Fatal("KVBlockSource accepted a nil session")
	}
}

func TestArchSessionRangeKVBlocks_NilYield(t *testing.T) {
	if err := (&ArchSession{}).RangeKVBlocks(4, kv.CaptureOptions{}, nil); err == nil {
		t.Fatal("RangeKVBlocks accepted a nil yield function")
	}
}

func TestArchSessionRestoreKV_NilSnapshot(t *testing.T) {
	if err := (&ArchSession{}).RestoreKV(nil); err == nil {
		t.Fatal("RestoreKV accepted a nil snapshot")
	}
}

func TestNativeKVTokenRowsToLayerSlab_Good(t *testing.T) {
	src := seededKVBytes(3*2*2*bf16Size, 17)
	got := make([]byte, len(src))
	nativeKVTokenRowsToLayerSlab(got, src, 3, 2, 2)
	want := append(append(append([]byte{}, src[0:4]...), src[8:12]...), src[16:20]...)
	want = append(want, src[4:8]...)
	want = append(want, src[12:16]...)
	want = append(want, src[20:24]...)
	if string(got) != string(want) {
		t.Fatalf("slab = %v, want %v", got, want)
	}
}

func TestNativeKVLayerSlabToTokenRows_Ugly(t *testing.T) {
	src := seededKVBytes(4*3*2*bf16Size, 91)
	rows := make([]byte, len(src))
	slab := make([]byte, len(src))
	nativeKVTokenRowsToLayerSlab(slab, src, 4, 3, 2)
	nativeKVLayerSlabToTokenRows(rows, slab, 4, 3, 2)
	if string(rows) != string(src) {
		t.Fatal("row/slab transpose did not round-trip varied fixture")
	}
}

func TestNativeKVLayerSlabPrefix_Good(t *testing.T) {
	src := seededKVBytes(2*4*3*bf16Size, 123)
	got, err := nativeKVLayerSlabPrefix(src, 4, 2, 2, 3)
	if err != nil {
		t.Fatalf("nativeKVLayerSlabPrefix: %v", err)
	}
	want := append(append([]byte{}, src[:12]...), src[24:36]...)
	if string(got) != string(want) {
		t.Fatalf("prefix = %v, want %v", got, want)
	}
}

func TestNativeKVLayerSlabPrefix_Bad(t *testing.T) {
	if _, err := nativeKVLayerSlabPrefix(seededKVBytes(8, 3), 2, 3, 1, 2); err == nil {
		t.Fatal("prefix beyond sequence must fail")
	}
}

func TestNativeKVLayerSlabPrefix_Ugly(t *testing.T) {
	if _, err := nativeKVLayerSlabPrefix(seededKVBytes(7, 5), 2, 1, 1, 2); err == nil {
		t.Fatal("malformed slab size must fail")
	}
}

func TestNativeKVSliceBlockPrefix_Good(t *testing.T) {
	const heads, seqLen, headDim = 2, 4, 2
	key := seededKVBytes(heads*seqLen*headDim*bf16Size, 201)
	value := seededKVBytes(heads*seqLen*headDim*bf16Size, 203)
	snapshot := &kv.Snapshot{
		Tokens:   []int32{10, 11, 12, 13},
		SeqLen:   seqLen,
		NumHeads: heads,
		HeadDim:  headDim,
		Layers: []kv.LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			CacheMode:  nativeStateCacheModeFixed,
			KeyDType:   nativeKVSnapshotDTypeBF16,
			KeyBytes:   key,
			KeyShape:   []int32{1, heads, seqLen, headDim},
			ValueDType: nativeKVSnapshotDTypeBF16,
			ValueBytes: value,
			ValueShape: []int32{1, heads, seqLen, headDim},
		}},
	}

	got, err := nativeKVSliceBlockPrefix(snapshot, 2, 1)
	if err != nil {
		t.Fatalf("nativeKVSliceBlockPrefix: %v", err)
	}
	if !idsEqual(got.Tokens, []int32{10, 11}) || got.TokenOffset != 3 || got.SeqLen != 2 {
		t.Fatalf("prefix metadata = tokens %v offset %d seq %d", got.Tokens, got.TokenOffset, got.SeqLen)
	}
	wantKey, err := nativeKVLayerSlabPrefix(key, seqLen, 2, heads, headDim)
	if err != nil {
		t.Fatalf("nativeKVLayerSlabPrefix(key): %v", err)
	}
	wantValue, err := nativeKVLayerSlabPrefix(value, seqLen, 2, heads, headDim)
	if err != nil {
		t.Fatalf("nativeKVLayerSlabPrefix(value): %v", err)
	}
	if !bytes.Equal(got.Layers[0].KeyBytes, wantKey) || !bytes.Equal(got.Layers[0].ValueBytes, wantValue) {
		t.Fatal("nativeKVSliceBlockPrefix did not preserve the expected raw prefix")
	}
}

func TestNativeKVSliceBlockPrefix_Bad(t *testing.T) {
	snapshot := &kv.Snapshot{
		Tokens: []int32{1, 2},
		SeqLen: 2,
		Layers: []kv.LayerSnapshot{{TurboQuantPayloads: [][]byte{{1, 2, 3}}}},
	}
	if _, err := nativeKVSliceBlockPrefix(snapshot, 1, 0); err == nil {
		t.Fatal("partial TurboQuant block prefix must remain unsupported")
	}
}

func TestNativeKVLayerSnapshotPrefixSlabs_Good(t *testing.T) {
	const heads, seqLen, prefixTokens, headDim = 2, 4, 2, 2
	key := seededKVBytes(heads*seqLen*headDim*bf16Size, 211)
	value := seededKVBytes(heads*seqLen*headDim*bf16Size, 223)
	view := sessionStateLayerView{kvHeads: heads, headDim: headDim}
	layer := kv.LayerSnapshot{
		KeyDType:   nativeKVSnapshotDTypeBF16,
		KeyBytes:   key,
		KeyShape:   []int32{1, heads, seqLen, headDim},
		ValueDType: nativeKVSnapshotDTypeBF16,
		ValueBytes: value,
		ValueShape: []int32{1, heads, seqLen, headDim},
	}

	gotKey, gotValue, err := nativeKVLayerSnapshotPrefixSlabs(layer, view, prefixTokens)
	if err != nil {
		t.Fatalf("nativeKVLayerSnapshotPrefixSlabs: %v", err)
	}
	wantKey, err := nativeKVLayerSlabPrefix(key, seqLen, prefixTokens, heads, headDim)
	if err != nil {
		t.Fatalf("nativeKVLayerSlabPrefix(key): %v", err)
	}
	wantValue, err := nativeKVLayerSlabPrefix(value, seqLen, prefixTokens, heads, headDim)
	if err != nil {
		t.Fatalf("nativeKVLayerSlabPrefix(value): %v", err)
	}
	if !bytes.Equal(gotKey, wantKey) || !bytes.Equal(gotValue, wantValue) {
		t.Fatal("nativeKVLayerSnapshotPrefixSlabs returned bytes outside the requested token prefix")
	}
}

func TestNativeKVLayerSnapshotPrefixSlabs_Bad(t *testing.T) {
	view := sessionStateLayerView{kvHeads: 1, headDim: 2}
	layer := kv.LayerSnapshot{
		KeyDType:   nativeKVSnapshotDTypeBF16,
		KeyBytes:   seededKVBytes(8, 227),
		KeyShape:   []int32{1, 1, 2, 2},
		ValueDType: nativeKVSnapshotDTypeBF16,
		ValueBytes: seededKVBytes(8, 229),
		ValueShape: []int32{1, 1, 2, 2},
	}
	if _, _, err := nativeKVLayerSnapshotPrefixSlabs(layer, view, 3); err == nil {
		t.Fatal("prefix outside the raw layer window must fail")
	}
}

func TestArchSessionRestoreKVSnapshotBlockLayers_Good(t *testing.T) {
	const heads, tokens, headDim, start = 2, 2, 2, 1
	rowBytes := heads * headDim * bf16Size
	view := sessionStateLayerView{
		layer: 0, cacheIndex: 0, cacheMode: nativeStateCacheModeFixed,
		kvHeads: heads, headDim: headDim, rowBytes: rowBytes, cacheRows: 5,
		keyBytes: seededKVBytes(5*rowBytes, 233), valueBytes: seededKVBytes(5*rowBytes, 239),
	}
	beforeKey := append([]byte(nil), view.keyBytes...)
	beforeValue := append([]byte(nil), view.valueBytes...)
	keySlab := seededKVBytes(heads*tokens*headDim*bf16Size, 241)
	valueSlab := seededKVBytes(heads*tokens*headDim*bf16Size, 251)
	block := kv.Block{
		TokenStart: start,
		TokenCount: tokens,
		Snapshot: &kv.Snapshot{Layers: []kv.LayerSnapshot{{
			Layer: 0, CacheIndex: 0, CacheMode: nativeStateCacheModeFixed,
			KeyDType: nativeKVSnapshotDTypeBF16, KeyBytes: keySlab, KeyShape: []int32{1, heads, tokens, headDim},
			ValueDType: nativeKVSnapshotDTypeBF16, ValueBytes: valueSlab, ValueShape: []int32{1, heads, tokens, headDim},
		}}},
	}

	if err := (&ArchSession{}).restoreKVSnapshotBlockLayers(block, start+tokens, []sessionStateLayerView{view}); err != nil {
		t.Fatalf("restoreKVSnapshotBlockLayers: %v", err)
	}
	wantKey := append([]byte(nil), beforeKey...)
	wantValue := append([]byte(nil), beforeValue...)
	keyRows := make([]byte, len(keySlab))
	valueRows := make([]byte, len(valueSlab))
	nativeKVLayerSlabToTokenRows(keyRows, keySlab, tokens, heads, headDim)
	nativeKVLayerSlabToTokenRows(valueRows, valueSlab, tokens, heads, headDim)
	copy(wantKey[start*rowBytes:], keyRows)
	copy(wantValue[start*rowBytes:], valueRows)
	if !bytes.Equal(view.keyBytes, wantKey) || !bytes.Equal(view.valueBytes, wantValue) {
		t.Fatal("restoreKVSnapshotBlockLayers changed bytes outside the requested block")
	}
}

func TestArchSessionRestoreKVSnapshotBlockLayersPrefix_Good(t *testing.T) {
	const heads, blockTokens, prefixTokens, headDim, start = 2, 4, 2, 2, 1
	rowBytes := heads * headDim * bf16Size
	view := sessionStateLayerView{
		layer: 0, cacheIndex: 0, cacheMode: nativeStateCacheModeFixed,
		kvHeads: heads, headDim: headDim, rowBytes: rowBytes, cacheRows: 6,
		keyBytes: seededKVBytes(6*rowBytes, 257), valueBytes: seededKVBytes(6*rowBytes, 263),
	}
	beforeKey := append([]byte(nil), view.keyBytes...)
	beforeValue := append([]byte(nil), view.valueBytes...)
	keySlab := seededKVBytes(heads*blockTokens*headDim*bf16Size, 269)
	valueSlab := seededKVBytes(heads*blockTokens*headDim*bf16Size, 271)
	block := kv.Block{
		TokenStart: start,
		TokenCount: blockTokens,
		Snapshot: &kv.Snapshot{
			Tokens: []int32{1, 2, 3, 4}, SeqLen: blockTokens,
			Layers: []kv.LayerSnapshot{{
				Layer: 0, CacheIndex: 0, CacheMode: nativeStateCacheModeFixed,
				KeyDType: nativeKVSnapshotDTypeBF16, KeyBytes: keySlab, KeyShape: []int32{1, heads, blockTokens, headDim},
				ValueDType: nativeKVSnapshotDTypeBF16, ValueBytes: valueSlab, ValueShape: []int32{1, heads, blockTokens, headDim},
			}},
		},
	}

	if err := (&ArchSession{}).restoreKVSnapshotBlockLayersPrefix(block, prefixTokens, start+prefixTokens, []sessionStateLayerView{view}); err != nil {
		t.Fatalf("restoreKVSnapshotBlockLayersPrefix: %v", err)
	}
	wantKey := append([]byte(nil), beforeKey...)
	wantValue := append([]byte(nil), beforeValue...)
	prefixKey, err := nativeKVLayerSlabPrefix(keySlab, blockTokens, prefixTokens, heads, headDim)
	if err != nil {
		t.Fatalf("nativeKVLayerSlabPrefix(key): %v", err)
	}
	prefixValue, err := nativeKVLayerSlabPrefix(valueSlab, blockTokens, prefixTokens, heads, headDim)
	if err != nil {
		t.Fatalf("nativeKVLayerSlabPrefix(value): %v", err)
	}
	keyRows := make([]byte, len(prefixKey))
	valueRows := make([]byte, len(prefixValue))
	nativeKVLayerSlabToTokenRows(keyRows, prefixKey, prefixTokens, heads, headDim)
	nativeKVLayerSlabToTokenRows(valueRows, prefixValue, prefixTokens, heads, headDim)
	copy(wantKey[start*rowBytes:], keyRows)
	copy(wantValue[start*rowBytes:], valueRows)
	if !bytes.Equal(view.keyBytes, wantKey) || !bytes.Equal(view.valueBytes, wantValue) {
		t.Fatal("restoreKVSnapshotBlockLayersPrefix changed bytes outside the requested prefix")
	}
}

func TestNativeKVLayerBlockSnapshot_Good(t *testing.T) {
	key := seededKVBytes(24, 7)
	value := seededKVBytes(24, 11)
	got, err := nativeKVLayerBlockSnapshot(SessionStateLayerBlock{Layer: 1, CacheIndex: 2, CacheMode: "fixed", KVHeads: 2, HeadDim: 2, RowBytes: 8, KeyBytes: key, ValueBytes: value}, 3, true)
	if err != nil {
		t.Fatalf("nativeKVLayerBlockSnapshot: %v", err)
	}
	if got.KeyDType != "bfloat16" || len(got.KeyShape) != 4 || got.KeyShape[2] != 3 || len(got.KeyBytes) != len(key) {
		t.Fatalf("raw layer = %+v", got)
	}
}

func TestNativeKVLayerBlockSnapshot_Bad(t *testing.T) {
	_, err := nativeKVLayerBlockSnapshot(SessionStateLayerBlock{KVHeads: 1, HeadDim: 2, RowBytes: 4, KeyBytes: seededKVBytes(4, 1), ValueBytes: seededKVBytes(2, 2)}, 1, true)
	if err == nil {
		t.Fatal("mismatched payloads must fail")
	}
}

func TestNativeKVLayerBlockSnapshot_Ugly(t *testing.T) {
	got, err := nativeKVLayerBlockSnapshot(SessionStateLayerBlock{Layer: 4}, 1, false)
	if err != nil || got.Layer != 4 || len(got.Heads) != 0 {
		t.Fatalf("empty metadata layer = %+v, %v", got, err)
	}
}

func TestNativeKVLayerSnapshotDirectBF16Slabs_Good(t *testing.T) {
	view := sessionStateLayerView{kvHeads: 2, headDim: 2}
	raw := seededKVBytes(24, 43)
	layer := kv.LayerSnapshot{KeyDType: "BF16", KeyBytes: raw, KeyShape: []int32{1, 2, 3, 2}, ValueDType: "bfloat16", ValueBytes: append([]byte(nil), raw...), ValueShape: []int32{1, 2, 3, 2}}
	k, v, n, ok, err := nativeKVLayerSnapshotDirectBF16Slabs("test", layer, view)
	if err != nil || !ok || n != 3 || string(k) != string(raw) || string(v) != string(raw) {
		t.Fatalf("direct slabs = n:%d ok:%v err:%v", n, ok, err)
	}
}

func TestNativeKVLayerSnapshotDirectBF16Slabs_Bad(t *testing.T) {
	view := sessionStateLayerView{kvHeads: 1, headDim: 2}
	layer := kv.LayerSnapshot{KeyDType: "bf16", KeyBytes: seededKVBytes(4, 1), KeyShape: []int32{1, 1, 1, 2}, ValueDType: "bf16", ValueBytes: seededKVBytes(8, 2), ValueShape: []int32{1, 1, 2, 2}}
	if _, _, _, ok, err := nativeKVLayerSnapshotDirectBF16Slabs("test", layer, view); !ok || err == nil {
		t.Fatalf("window mismatch = ok:%v err:%v", ok, err)
	}
}

func TestNativeKVLayerSnapshotDirectBF16Slabs_Ugly(t *testing.T) {
	_, _, _, ok, err := nativeKVLayerSnapshotDirectBF16Slabs("test", kv.LayerSnapshot{KeyDType: "float32", KeyBytes: []byte{1}, ValueBytes: []byte{2}}, sessionStateLayerView{})
	if ok || err != nil {
		t.Fatalf("non-bf16 must decline, got ok:%v err:%v", ok, err)
	}
}

func TestRestoreNativeKVLayerSlabs_Good(t *testing.T) {
	view := sessionStateLayerView{kvHeads: 2, headDim: 2, rowBytes: 8, cacheRows: 4, keyBytes: make([]byte, 32), valueBytes: make([]byte, 32)}
	key := seededKVBytes(16, 31)
	value := seededKVBytes(16, 37)
	if err := restoreNativeKVLayerSlabs("test", view, 1, 2, 3, key, value); err != nil {
		t.Fatalf("restoreNativeKVLayerSlabs: %v", err)
	}
	wantK := make([]byte, 16)
	wantV := make([]byte, 16)
	nativeKVLayerSlabToTokenRows(wantK, key, 2, 2, 2)
	nativeKVLayerSlabToTokenRows(wantV, value, 2, 2, 2)
	if string(view.keyBytes[8:24]) != string(wantK) || string(view.valueBytes[8:24]) != string(wantV) {
		t.Fatal("restored rows differ from varied slabs")
	}
}

func TestRestoreNativeKVLayerSlabs_Bad(t *testing.T) {
	view := sessionStateLayerView{kvHeads: 1, headDim: 2, rowBytes: 4, cacheRows: 2, keyBytes: make([]byte, 8), valueBytes: make([]byte, 8)}
	if err := restoreNativeKVLayerSlabs("test", view, 0, 2, 2, []byte{1}, []byte{2}); err == nil {
		t.Fatal("short slabs must fail")
	}
}

func TestRestoreNativeKVLayerSlabs_Ugly(t *testing.T) {
	view := sessionStateLayerView{kvHeads: 1, headDim: 1, rowBytes: 2, cacheRows: 2, maxSize: 2, keyBytes: make([]byte, 4), valueBytes: make([]byte, 4)}
	key := seededKVBytes(4, 71)
	if err := restoreNativeKVLayerSlabs("test", view, 3, 2, 5, key, key); err != nil {
		t.Fatalf("wrapped restore: %v", err)
	}
	if string(view.keyBytes) != string(append(key[2:], key[:2]...)) {
		t.Fatalf("wrapped rows = %v, slab %v", view.keyBytes, key)
	}
}

func TestNativeKVResidentLayerRows_Good(t *testing.T) {
	view := sessionStateLayerView{rowBytes: 3, cacheRows: 4, keyBytes: seededKVBytes(12, 5), valueBytes: seededKVBytes(12, 9)}
	k, v, ok, err := nativeKVResidentLayerRows(view, 1, 2, 3)
	if err != nil || !ok || string(k) != string(view.keyBytes[3:9]) || string(v) != string(view.valueBytes[3:9]) {
		t.Fatalf("resident rows = %v %v %v %v", k, v, ok, err)
	}
}

func TestNativeKVResidentLayerRows_Bad(t *testing.T) {
	_, _, _, err := nativeKVResidentLayerRows(sessionStateLayerView{}, 0, 1, 1)
	if err == nil {
		t.Fatal("zero geometry must fail")
	}
}

func TestNativeKVResidentLayerRows_Ugly(t *testing.T) {
	view := sessionStateLayerView{rowBytes: 2, cacheRows: 4, maxSize: 4, keyBytes: make([]byte, 8), valueBytes: make([]byte, 8)}
	if _, _, ok, err := nativeKVResidentLayerRows(view, 3, 2, 6); err != nil || ok {
		t.Fatalf("wrapped range should decline, ok:%v err:%v", ok, err)
	}
}

func TestNativeKVRawToBF16_Good(t *testing.T) {
	values := []float32{-1.75, 0.125, 3.5}
	raw := make([]byte, len(values)*4)
	for i, v := range values {
		bits := math.Float32bits(v)
		raw[i*4] = byte(bits)
		raw[i*4+1] = byte(bits >> 8)
		raw[i*4+2] = byte(bits >> 16)
		raw[i*4+3] = byte(bits >> 24)
	}
	got := make([]byte, len(values)*2)
	if err := nativeKVRawToBF16(got, raw, "F32"); err != nil {
		t.Fatalf("nativeKVRawToBF16: %v", err)
	}
	for i, v := range values {
		bits := uint16(got[i*2]) | uint16(got[i*2+1])<<8
		if bits != f32ToBF16(v) {
			t.Fatalf("value %d was not converted", i)
		}
	}
}

func TestNativeKVRawToBF16_Bad(t *testing.T) {
	if err := nativeKVRawToBF16(make([]byte, 4), seededKVBytes(3, 1), "float32"); err == nil {
		t.Fatal("wrong raw size must fail")
	}
}

func TestNativeKVRawToBF16_Ugly(t *testing.T) {
	if err := nativeKVRawToBF16(make([]byte, 2), []byte{1, 2}, "int8"); err == nil {
		t.Fatal("unsupported dtype must fail")
	}
}

func TestNativeKVRawDType_Good(t *testing.T) {
	name, width, ok := nativeKVRawDType("FlOaT16")
	if !ok || name != "float16" || width != 2 {
		t.Fatalf("dtype = %q/%d/%v", name, width, ok)
	}
}

func TestNativeKVRawDType_Bad(t *testing.T) {
	if _, _, ok := nativeKVRawDType("float64"); ok {
		t.Fatal("float64 must be rejected")
	}
}

func TestNativeKVASCIIEqualFold_Ugly(t *testing.T) {
	if !nativeKVASCIIEqualFold("BF16", "bf16") || nativeKVASCIIEqualFold("BF-16", "bf16") || nativeKVASCIIEqualFold("bf1X", "bf16") {
		t.Fatal("ASCII folding boundary mismatch")
	}
}

func TestNativeKVTrimStateRestoreBlocks_Good(t *testing.T) {
	source := SessionStateBlockSource{BlockCount: 3, firstBlockIndex: 1, blockBoundaries: []int{0, 3, 7, 10}, totalBlockCount: 3}
	if err := nativeKVTrimStateRestoreBlocks(&source, 5); err != nil {
		t.Fatalf("nativeKVTrimStateRestoreBlocks: %v", err)
	}
	if source.BlockCount != 1 || len(source.blockBoundaries) != 3 || source.blockBoundaries[2] != 5 {
		t.Fatalf("trimmed source = %+v", source)
	}
}

func TestNativeKVTrimStateRestoreBlocks_Bad(t *testing.T) {
	if err := nativeKVTrimStateRestoreBlocks(nil, 1); err == nil {
		t.Fatal("nil source must fail")
	}
}

func TestNativeKVTrimStateRestoreBlocks_Ugly(t *testing.T) {
	source := SessionStateBlockSource{firstBlockIndex: 2, blockBoundaries: []int{0, 3}}
	if err := nativeKVTrimStateRestoreBlocks(&source, 3); err == nil {
		t.Fatal("block index beyond boundaries must fail")
	}
}

func TestNativeKVSnapshotHasTurboQuantPayload_Good(t *testing.T) {
	s := &kv.Snapshot{Layers: []kv.LayerSnapshot{{}, {TurboQuantPayloads: [][]byte{seededKVBytes(5, 99)}}}}
	if !nativeKVSnapshotHasTurboQuantPayload(s) {
		t.Fatal("payload was not detected")
	}
}

func TestNativeKVSnapshotHasTurboQuantPayload_Ugly(t *testing.T) {
	if nativeKVSnapshotHasTurboQuantPayload(nil) || nativeKVSnapshotHasTurboQuantPayload(&kv.Snapshot{}) {
		t.Fatal("empty snapshots must not report turboquant payloads")
	}
}

func TestNativeKVHeadSnapshotSeqLen_Good(t *testing.T) {
	if got, err := nativeKVHeadSnapshotSeqLen([]float32{1, 2, 3, 4, 5, 6}, nil, "", 3); err != nil || got != 2 {
		t.Fatalf("float head seq = %d, %v", got, err)
	}
	if got, err := nativeKVHeadSnapshotSeqLen(nil, seededKVBytes(16, 4), "f32", 2); err != nil || got != 2 {
		t.Fatalf("raw head seq = %d, %v", got, err)
	}
}

func TestNativeKVHeadSnapshotSeqLen_Bad(t *testing.T) {
	if _, err := nativeKVHeadSnapshotSeqLen([]float32{1, 2, 3}, nil, "", 2); err == nil {
		t.Fatal("partial float row must fail")
	}
}

func TestNativeKVHeadSnapshotSeqLen_Ugly(t *testing.T) {
	if _, err := nativeKVHeadSnapshotSeqLen(nil, nil, "", 0); err == nil {
		t.Fatal("zero head dim must fail")
	}
}

func TestNativeKVFillHeadBF16_Good(t *testing.T) {
	values := []float32{-2.5, 0.75, 9.125, -0.0625}
	dst := make([]byte, len(values)*2)
	if err := nativeKVFillHeadBF16(dst, values, nil, "", 2, 2); err != nil {
		t.Fatalf("nativeKVFillHeadBF16: %v", err)
	}
	bits := uint16(dst[4]) | uint16(dst[5])<<8
	if bits != f32ToBF16(values[2]) {
		t.Fatal("varied float fixture was not converted")
	}
}

func TestNativeKVFillHeadBF16_Bad(t *testing.T) {
	if err := nativeKVFillHeadBF16(make([]byte, 2), []float32{1, 2}, nil, "", 1, 2); err == nil {
		t.Fatal("short destination must fail")
	}
}

func TestNativeKVFillHeadBF16_Ugly(t *testing.T) {
	if err := nativeKVFillHeadBF16(make([]byte, 4), nil, seededKVBytes(3, 8), "f16", 1, 2); err == nil {
		t.Fatal("partial raw payload must fail")
	}
}

func TestNativeKVLayerSlabHeads_Good(t *testing.T) {
	key := seededKVBytes(24, 77)
	value := seededKVBytes(24, 79)
	heads := nativeKVLayerSlabHeads(key, value, 3, 2, 2)
	if len(heads) != 2 || len(heads[0].Key) != 6 || len(heads[1].Value) != 6 {
		t.Fatalf("heads geometry = %+v", heads)
	}
}

func TestNativeKVLayerSlabHeads_Ugly(t *testing.T) {
	if got := nativeKVLayerSlabHeads(nil, nil, 0, 2, 2); got != nil {
		t.Fatalf("zero-token heads = %+v", got)
	}
}

func TestNativeKVLayerCaptureWindow_Good(t *testing.T) {
	if start, count, err := nativeKVLayerCaptureWindow(sessionStateLayerView{rowBytes: 4, cacheRows: 4, maxSize: 4}, 7); err != nil || start != 3 || count != 4 {
		t.Fatalf("window = %d/%d, %v", start, count, err)
	}
}

func TestNativeKVLayerCaptureWindow_Bad(t *testing.T) {
	if _, _, err := nativeKVLayerCaptureWindow(sessionStateLayerView{cacheRows: 2}, 3); err == nil {
		t.Fatal("fixed cache overflow must fail")
	}
}

func TestNativeKVSnapshotLayer_Good(t *testing.T) {
	s := &kv.Snapshot{Layers: []kv.LayerSnapshot{{Layer: 3}, {Layer: 1}}}
	if got, ok := nativeKVSnapshotLayer(s, 1); !ok || got.Layer != 1 {
		t.Fatalf("layer lookup = %+v/%v", got, ok)
	}
}

func TestNativeKVSnapshotLayer_Ugly(t *testing.T) {
	if _, ok := nativeKVSnapshotLayer(nil, 0); ok {
		t.Fatal("nil snapshot must not have a layer")
	}
}

func TestNativeKVValidateLayerMetadata_Good(t *testing.T) {
	view := sessionStateLayerView{layer: 2, cacheIndex: 1, cacheMode: "fixed", maxSize: 0}
	layer := kv.LayerSnapshot{Layer: 2, CacheIndex: 1, CacheMode: "fixed"}
	if err := nativeKVValidateLayerMetadata("test", layer, view); err != nil {
		t.Fatalf("nativeKVValidateLayerMetadata: %v", err)
	}
}

func TestNativeKVValidateLayerMetadata_Bad(t *testing.T) {
	if err := nativeKVValidateLayerMetadata("test", kv.LayerSnapshot{CacheIndex: 9}, sessionStateLayerView{cacheIndex: 2}); err == nil {
		t.Fatal("cache-index mismatch must fail")
	}
}

func TestNativeKVLayerHasPayload_Ugly(t *testing.T) {
	if nativeKVLayerHasPayload(kv.LayerSnapshot{}) || !nativeKVLayerHasPayload(kv.LayerSnapshot{Heads: []kv.HeadSnapshot{{Key: []float32{1}}}}) {
		t.Fatal("payload detection mismatch")
	}
}

func TestArchSessionKVBlockCachedIDScratch_Good(t *testing.T) {
	s := &ArchSession{kvBlockCachedIDs: make([]int32, 2, 8)}
	got := s.kvBlockCachedIDScratch(5)
	if len(got) != 0 || cap(got) != 8 {
		t.Fatalf("scratch len/cap = %d/%d", len(got), cap(got))
	}
}

func TestArchSessionKVBlockCachedIDScratch_Ugly(t *testing.T) {
	var s *ArchSession
	if got := s.kvBlockCachedIDScratch(3); got != nil {
		t.Fatalf("nil session scratch = %v", got)
	}
}

func TestArchSessionTurboQuantKVDecodeScratch_Good(t *testing.T) {
	s := &ArchSession{}
	r, n := s.turboQuantKVDecodeScratch(7)
	if len(r) != 7 || len(n) != 7 {
		t.Fatalf("scratch lengths = %d/%d", len(r), len(n))
	}
	r[0] = 42
	r2, _ := s.turboQuantKVDecodeScratch(3)
	if len(r2) != 3 || r2[0] != 42 {
		t.Fatal("scratch capacity was not reused")
	}
}

func TestArchSessionTurboQuantKVDecodeScratch_Ugly(t *testing.T) {
	var s *ArchSession
	r, n := s.turboQuantKVDecodeScratch(4)
	if r != nil || n != nil {
		t.Fatal("nil session must return nil scratch")
	}
}
