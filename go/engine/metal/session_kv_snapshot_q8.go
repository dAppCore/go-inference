// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"encoding/binary"
	"math"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/kv"
	"github.com/tmc/apple/metal"
)

// session_kv_snapshot_q8.go — the BIT-EXACT q8 KV snapshot codec (#1846 state
// lane). A q8 store holds int8 codes + f32 group scales; the portable bf16
// capture/restore round trip dequantises to bf16 then re-quantises to int8, and
// that double-quantise is NOT an identity (bf16 rounds the dequantised value,
// then the re-quantise recomputes the group scale from the rounded maxima and
// re-rounds every code) — it perturbs every restored prefix row, so a woken q8
// session diverges from the stateless whole. This codec instead carries the
// store's RAW block verbatim under kv.KVNativeDTypeQ8, so a q8→q8 sleep/wake is
// byte-for-byte the live state. Non-q8 stores keep the bf16 path unchanged.
//
// Packed layout per side (K and V each): [int8 codes: tokenCount·kvd bytes]
// followed by [f32 group scales, little-endian: tokenCount·(kvd/kvQ8GroupSize)
// values], token-major — exactly the store's own row order. kv treats the block
// as opaque; the codes/scales split lives here (kvQ8GroupSize is the engine's).

// q8NativeGroupsPerRow is the number of f32 group scales one cache row carries.
func q8NativeGroupsPerRow(kvd int) int { return kvd / kvQ8GroupSize }

// q8NativePackedRowBytes is the packed size of one token row (codes + scales).
func q8NativePackedRowBytes(kvd int) int {
	return kvd + q8NativeGroupsPerRow(kvd)*4
}

// packQ8NativeLayer joins a window's int8 codes and its f32 scale bytes into one
// KeyBytes/ValueBytes payload. The inputs are copied, so the snapshot owns bytes
// independent of the live (still-decoding) store.
func packQ8NativeLayer(codes, scaleBytes []byte) []byte {
	out := make([]byte, 0, len(codes)+len(scaleBytes))
	out = append(out, codes...)
	out = append(out, scaleBytes...)
	return out
}

// splitQ8NativeLayer splits a packed q8-native payload back into its int8 codes
// and f32 scale bytes for the given window geometry.
func splitQ8NativeLayer(packed []byte, tokenCount, kvd int) (codes, scaleBytes []byte, err error) {
	if tokenCount <= 0 || kvd <= 0 || kvd%kvQ8GroupSize != 0 {
		return nil, nil, core.NewError("native.RestoreKV: invalid q8-native layer geometry")
	}
	codeLen := tokenCount * kvd
	scaleLen := tokenCount * q8NativeGroupsPerRow(kvd) * 4
	if len(packed) != codeLen+scaleLen {
		return nil, nil, core.NewError("native.RestoreKV: q8-native payload size mismatch")
	}
	return packed[:codeLen], packed[codeLen:], nil
}

// captureQ8LayerRaw reads layer li's RAW q8 block for the window [start,
// start+tokenCount) — the int8 codes and their f32 scales, copied out of the
// live store into portable KeyBytes/ValueBytes payloads.
func (r *archICBReplay) captureQ8LayerRaw(li, start, tokenCount int) (kPacked, vPacked []byte, err error) {
	kvd, rows, err := r.q8LayerGeometry(li)
	if err != nil {
		return nil, nil, err
	}
	if start < 0 || tokenCount <= 0 || start+tokenCount > rows {
		return nil, nil, core.NewError("native.CaptureKV: q8 window outside cache rows")
	}
	groupsPerRow := q8NativeGroupsPerRow(kvd)
	kCodes := q8CacheBytes(r.kCaches[li], rows*kvd)[start*kvd : (start+tokenCount)*kvd]
	vCodes := q8CacheBytes(r.vCaches[li], rows*kvd)[start*kvd : (start+tokenCount)*kvd]
	kScales := q8CacheBytes(r.kvQ8.kScales[li], rows*groupsPerRow*4)[start*groupsPerRow*4 : (start+tokenCount)*groupsPerRow*4]
	vScales := q8CacheBytes(r.kvQ8.vScales[li], rows*groupsPerRow*4)[start*groupsPerRow*4 : (start+tokenCount)*groupsPerRow*4]
	return packQ8NativeLayer(kCodes, kScales), packQ8NativeLayer(vCodes, vScales), nil
}

// restoreQ8LayerRaw writes a captured q8-native block back into layer li's live
// store verbatim — no quantise pass, so the restored rows are bit-identical to
// what was captured. The mirror is deliberately NOT flushed for these layers
// (that is the double-quantise the raw path exists to avoid); the caller skips
// flushQ8Mirrors when any layer took this path.
func (r *archICBReplay) restoreQ8LayerRaw(li, start, tokenCount int, kPacked, vPacked []byte) error {
	kvd, rows, err := r.q8LayerGeometry(li)
	if err != nil {
		return err
	}
	if start < 0 || tokenCount <= 0 || start+tokenCount > rows {
		return core.NewError("native.RestoreKV: q8 window outside cache rows")
	}
	kCodes, kScales, err := splitQ8NativeLayer(kPacked, tokenCount, kvd)
	if err != nil {
		return err
	}
	vCodes, vScales, err := splitQ8NativeLayer(vPacked, tokenCount, kvd)
	if err != nil {
		return err
	}
	groupsPerRow := q8NativeGroupsPerRow(kvd)
	copy(q8CacheBytes(r.kCaches[li], rows*kvd)[start*kvd:], kCodes)
	copy(q8CacheBytes(r.vCaches[li], rows*kvd)[start*kvd:], vCodes)
	copy(q8CacheBytes(r.kvQ8.kScales[li], rows*groupsPerRow*4)[start*groupsPerRow*4:], kScales)
	copy(q8CacheBytes(r.kvQ8.vScales[li], rows*groupsPerRow*4)[start*groupsPerRow*4:], vScales)
	return nil
}

// q8LayerGeometry returns layer li's (kvd, cacheRows) after validating it is a
// live q8 layer with allocated code + scale buffers.
func (r *archICBReplay) q8LayerGeometry(li int) (kvd, rows int, err error) {
	if r == nil || r.kvQ8 == nil || !r.kvQ8.on(li) {
		return 0, 0, core.NewError("native.q8LayerGeometry: not a q8 layer")
	}
	if li >= len(r.rowBytes) || li >= len(r.cacheRows) || r.rowBytes[li] <= 0 || r.cacheRows[li] <= 0 {
		return 0, 0, core.NewError("native.q8LayerGeometry: bad q8 layer geometry")
	}
	if r.kCaches[li] == nil || r.vCaches[li] == nil || r.kvQ8.kScales[li] == nil || r.kvQ8.vScales[li] == nil {
		return 0, 0, core.NewError("native.q8LayerGeometry: missing q8 layer buffers")
	}
	kvd = r.rowBytes[li] / bf16Size
	if kvd <= 0 || kvd%kvQ8GroupSize != 0 {
		return 0, 0, core.NewError("native.q8LayerGeometry: q8 kvDim not a whole number of groups")
	}
	return kvd, r.cacheRows[li], nil
}

// q8CacheBytes views a q8 store buffer's first n bytes.
func q8CacheBytes(buf metal.MTLBuffer, n int) []byte {
	return unsafe.Slice((*byte)(buf.Contents()), n)
}

// q8NativeToTokenRowsBF16 dequantises a packed q8-native block into bf16 TOKEN
// rows (the layout restoreStateBlockLayer consumes) — the path taken when a q8
// snapshot is restored into a NON-q8 session, which cannot land the int8 codes
// directly. The dequantise matches the store kernel (code·groupScale), so it is
// the same value a live q8 read produces; the target simply keeps it in bf16.
func q8NativeToTokenRowsBF16(packed []byte, tokenCount, kvd int) ([]byte, error) {
	codes, scaleBytes, err := splitQ8NativeLayer(packed, tokenCount, kvd)
	if err != nil {
		return nil, err
	}
	groupsPerRow := q8NativeGroupsPerRow(kvd)
	out := make([]byte, tokenCount*kvd*bf16Size)
	for row := range tokenCount {
		codeBase := row * kvd
		scaleBase := row * groupsPerRow
		for i := range kvd {
			c := int8(codes[codeBase+i])
			s := math.Float32frombits(binary.LittleEndian.Uint32(scaleBytes[(scaleBase+i/kvQ8GroupSize)*4:]))
			lo, hi := bf16BytesOfF32(float32(c) * s)
			o := (codeBase + i) * bf16Size
			out[o], out[o+1] = lo, hi
		}
	}
	return out, nil
}

// nativeKVLayerIsQ8Native reports whether a snapshot layer carries the raw q8
// block on both K and V.
func nativeKVLayerIsQ8Native(layer kv.LayerSnapshot) bool {
	return layer.KeyDType == kv.KVNativeDTypeQ8 && layer.ValueDType == kv.KVNativeDTypeQ8
}

// nativeKVQ8NativeWindow validates a q8-native layer's [1, heads, seq, headDim]
// shapes against the target view and returns the captured window length (seq)
// and the row width (kvd = heads·headDim).
func nativeKVQ8NativeWindow(layer kv.LayerSnapshot, view sessionStateLayerView) (tokenCount, kvd int, err error) {
	for _, shape := range [][]int32{layer.KeyShape, layer.ValueShape} {
		if len(shape) != 4 || shape[0] != 1 || int(shape[1]) != view.kvHeads || int(shape[3]) != view.headDim {
			return 0, 0, core.NewError("native.RestoreKV: q8-native layer shape mismatch")
		}
	}
	if layer.KeyShape[2] != layer.ValueShape[2] {
		return 0, 0, core.NewError("native.RestoreKV: q8-native key/value window mismatch")
	}
	tokenCount = int(layer.KeyShape[2])
	kvd = view.kvHeads * view.headDim
	if tokenCount <= 0 || kvd <= 0 {
		return 0, 0, core.NewError("native.RestoreKV: q8-native window invalid")
	}
	return tokenCount, kvd, nil
}
